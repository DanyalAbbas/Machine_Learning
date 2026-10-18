# %% [markdown]
# # Name That LLM — v2: Qwen3-8B embeddings + stylometry + cluster-aware assignment
#
# ## What changed from v1 (grouped CV 0.3771, public LB 0.360)
#
# **1. Frozen Qwen3-8B-Base features.** Mean-pooled hidden states, several layers, a
# regularised logistic head per layer. Early layers beat late ones here — author style is
# a surface phenomenon, and the deep layers specialise for next-token semantics.
#
# **2. Topic-centred heads.** Recover prompt groups by average-linkage clustering of
# content-word TF-IDF (ARI 0.955 against the true prompts). Subtracting the cluster mean
# removes topic and leaves style.
#
# **3. Cluster-aware assignment — the big one.** The train set is 150 prompts x 18
# models, so every prompt is a *permutation* of the 18 labels. Recovering those groups by
# clustering lets us replace 18 independent argmaxes with one joint assignment that uses
# each label exactly once. On v1's out-of-fold probabilities this alone is worth
# **+0.075 balanced accuracy** (0.3771 -> 0.4523) — more than the entire jump from
# swapping the lexical model for an 8B embedder. It composes with whatever the base model
# predicts, and it needs no labels, so it transfers to the hidden test set.
#
# The rule is deliberately narrow: apply it only to clusters of size 2-8 or exactly 18.
# On train, every size-18 cluster holds exactly one passage per model (123/123), and small
# clusters are fragments of one prompt. Sizes 9-17 are the residue of imperfect splits and
# measured *negative* (-0.016), so they are skipped.
#
# **4. Fitted blend weights.** The log-space blend weights are fitted on out-of-fold
# probabilities rather than hand-set.
#
# ## Reproduced measurements (all on train, all leak-free)
#
# | claim | source |
# |---|---|
# | clustering recovers prompts | `verify_clusters.py` |
# | size-18 clusters are exactly one-per-model, 123/123 | `verify_clusters.py` |
# | assignment gain +0.075 on v1 OOF | `test_gate_design.py` |
#
# **Kaggle settings: GPU T4 x2, Internet OFF.** Inputs: competition data +
# `qwen-lm/qwen-3/transformers/8b-base`.

# %%
import gc
import glob
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# `KAGGLE_IS_COMPETITION_RERUN` is set by Kaggle while scoring. CV is a ~30 minute
# diagnostic; the scoring pass only needs the final fit, so skip it there.
IS_RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
RUN_CV = not IS_RERUN
NO_QWEN = os.getenv("NTL_NO_QWEN") == "1"  # local smoke test escape hatch

SEED = 42
N_FOLDS = 5
INNER_FOLDS = 3

# Layers to mean-pool. Filled in from the interactive run that selected them by CV.
HEAD_LAYERS = (4, 6, 8, 22)
MAX_LEN, BATCH = 512, 8
EMB_C = 0.002

# Cluster assignment gate: fragment sizes plus exactly one full prompt.
ASSIGN_EXACT = {18}
ASSIGN_MAX_SMALL = 8
CLUSTER_THRESH = 0.9

# Fallback blend weights (used on reruns and if the fitted ones are unavailable).
BLEND_FALLBACK = {"svm": 0.4, "emb_plain": 0.3, "emb_centred": 0.3}

LABELS = np.array(sorted([
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]))
K = len(LABELS)

T0 = time.time()
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


DATA = next(
    (p for p in [Path("/kaggle/input/competitions/name-that-llm"),
                 Path("/kaggle/input/name-that-llm"),
                 Path("data"), Path("../../data")]
     if (p / "train.csv").is_file()),
    None,
)
if DATA is None:
    raise FileNotFoundError("Attach the 'name-that-llm' competition data.")

train = pd.read_csv(DATA / "train.csv", dtype=str)
test = pd.read_csv(DATA / "test.csv", dtype=str)
train["text"] = train.text.fillna("").astype(str)
test["text"] = test.text.fillna("").astype(str)
assert train.id.is_unique and test.id.is_unique and set(train.model) == set(LABELS)

texts = train.text.to_numpy(dtype=object)
test_texts = test.text.to_numpy(dtype=object)
test_ids = test.id.to_numpy(dtype=str)
y = train.model.map({label: i for i, label in enumerate(LABELS)}).to_numpy()
groups = train.prompt_id.to_numpy(dtype=str)
folds = list(GroupKFold(N_FOLDS).split(texts, y, groups))
log(f"train {train.shape} ({len(set(groups))} prompts) | test {test.shape} | rerun {IS_RERUN}")

# %% [markdown]
# ## 1. Topic clusters — a stand-in for the hidden prompt ids
#
# Test rows carry no `prompt_id`, but passages answering the same prompt are topically
# similar, so average-linkage clustering of content-word TF-IDF recovers the grouping
# well enough to exploit it. On train this reproduces the true prompts at ARI 0.955.

# %%
def topic_clusters(texts_, thresh=CLUSTER_THRESH):
    if len(texts_) < 2:
        return np.ones(len(texts_), dtype=int)
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts_)
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), method="average"),
                    thresh, criterion="distance")


train_cl = topic_clusters(texts)
test_cl = topic_clusters(test_texts)
sizes = np.bincount(train_cl)[train_cl]
hist = {int(k): int(v) for k, v in zip(*np.unique(np.bincount(test_cl)[1:], return_counts=True))}
log(f"train clusters {len(np.unique(train_cl))} | size-18 rows {int((sizes == 18).sum())}")
log(f"test clusters {len(np.unique(test_cl))} | size histogram {hist}")

# Train the centred heads on the same group geometry they will meet at prediction time.
# Use the cluster sizes actually observed in the test set, but keep only the most frequent
# few: every extra chunking is another full copy of the embedding matrix.
observed = np.bincount(test_cl)[1:]
sizes_seen = sorted((int(s), int((observed == s).sum())) for s in np.unique(observed) if 2 <= s <= K)
CENTRE_CHUNKS = tuple(sorted(s for s, _ in sorted(sizes_seen, key=lambda t: -t[1])[:4])) or (2, 3, 4, 6)
log(f"observed test cluster sizes (size, rows) {sizes_seen[:8]} -> centre chunks {CENTRE_CHUNKS}")

# %% [markdown]
# ## 2. Frozen Qwen3-8B-Base embeddings
#
# One forward pass per passage; every layer is mean-pooled (padding excluded) and stored
# as float16, which keeps the full 37-layer set at ~2 GB and lets the layer choice be a
# measured decision rather than a guess. fp16 weights are sharded across both T4s with
# `device_map="auto"`; any row that comes out non-finite is recomputed in bfloat16.

# %%
def find_qwen_dir():
    hits = []
    for cfg in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        try:
            c = json.load(open(cfg))
        except (OSError, ValueError):
            continue
        if c.get("model_type") != "qwen3" or "base" not in cfg.lower():
            continue
        if not list(Path(cfg).parent.glob("*.safetensors")):
            continue
        hits.append((Path(cfg).parent, c.get("hidden_size"), c.get("num_hidden_layers")))
    if not hits:
        raise FileNotFoundError(
            "No Qwen3 *Base* safetensors found under /kaggle/input. "
            "Attach qwen-lm/qwen-3/transformers/8b-base as an input."
        )
    # prefer the largest (8B over 4B over 1.7B) if several are attached
    hits.sort(key=lambda h: (h[1] or 0), reverse=True)
    log(f"qwen candidates: {[(str(h[0]), h[1], h[2]) for h in hits]}")
    return hits[0][0]


def embed(texts_, model_dir, save_layers):
    """Mean-pool `save_layers` of every passage, returned in the input row order."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    assert torch.cuda.is_available(), "Enable a GPU accelerator (T4 x2 for the 8B model)"
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lengths = np.array([len(i) for i in tok(list(texts_), add_special_tokens=False)["input_ids"]])
    order = np.argsort(-lengths)  # longest first, so any OOM surfaces immediately

    def run(rows, dtype):
        model = AutoModel.from_pretrained(str(model_dir), dtype=dtype, device_map="auto").eval()
        device = next(model.parameters()).device
        # NOTE: always scatter back by the *original* row index. Writing the batch results
        # in loop order instead leaves the array in length-sorted order, silently
        # misaligning every embedding with its label.
        out = np.zeros((len(texts_), len(save_layers), model.config.hidden_size), dtype=np.float16)
        bs, pos = BATCH, 0
        while pos < len(rows):
            idx = rows[pos:pos + bs]
            enc = tok([texts_[i] for i in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN, add_special_tokens=False).to(device)
            try:
                with torch.inference_mode():
                    hs = model(**enc, output_hidden_states=True, use_cache=False,
                               return_dict=True).hidden_states
                    m = enc["attention_mask"].unsqueeze(-1).to(hs[0].device)
                    for j, L in enumerate(save_layers):
                        h = hs[L].float()
                        out[idx, j] = ((h * m).sum(1) / m.sum(1)).half().cpu().numpy()
                        del h
                    del hs
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise RuntimeError("A single passage does not fit; lower MAX_LEN.")
                bs = max(1, bs // 2)
                log(f"OOM -> batch {bs}")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            pos += len(idx)
            if pos % (bs * 50) == 0 or pos >= len(rows):
                log(f"  embedded {pos}/{len(rows)}")

        # Alignment guard. Re-run the first few passages in a fresh single-batch forward
        # and require the result to match the rows they landed in. This is the one failure
        # mode that is invisible downstream: if results are written in loop order instead
        # of by original index, every embedding is paired with the wrong label and the
        # heads silently collapse to chance (1/K) without any error being raised.
        check = rows[:4]
        enc = tok([texts_[i] for i in check], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN, add_special_tokens=False).to(device)
        with torch.inference_mode():
            hs = model(**enc, output_hidden_states=True, use_cache=False,
                       return_dict=True).hidden_states
            m = enc["attention_mask"].unsqueeze(-1).to(hs[0].device)
            L = save_layers[min(4, len(save_layers) - 1)]
            again = ((hs[L].float() * m).sum(1) / m.sum(1)).half().cpu().numpy()
        if not np.allclose(out[check, min(4, len(save_layers) - 1)].astype(np.float32),
                           again.astype(np.float32), atol=5e-2):
            raise AssertionError(
                "embeddings are not aligned with the input rows: a fresh forward pass on "
                f"rows {check} disagrees with the stored values"
            )
        log(f"  alignment guard passed (rows {check})")

        del model
        gc.collect()
        torch.cuda.empty_cache()
        return out

    log(f"embedding {len(texts_)} passages from {model_dir.parent.name}")
    E = run(order, torch.float16)
    bad = np.where(~np.isfinite(E.astype(np.float32)).all(axis=(1, 2)))[0]
    if len(bad):
        log(f"{len(bad)} non-finite fp16 rows -> recomputing in bf16")
        E[bad] = run(bad, torch.bfloat16)
    assert np.isfinite(E.astype(np.float32)).all(), "non-finite embeddings remain"
    log(f"embeddings {E.shape}; max tokens {int(lengths.max())}")
    return E


if NO_QWEN:
    log("NTL_NO_QWEN=1 -> skipping the embedding branch (smoke test only)")
    E_tr = E_te = None
else:
    qwen_dir = find_qwen_dir()
    save_layers = tuple(range(int(json.load(open(qwen_dir / "config.json"))["num_hidden_layers"]) + 1))
    E = embed(list(texts) + list(test_texts), qwen_dir, save_layers)
    E_tr, E_te = E[:len(train)], E[len(train):]
    del E
    gc.collect()

# %% [markdown]
# ## 3. The three branches
#
# * `svm` — case-sensitive char 1-5 grams + word 1-2 grams + explicit stylometry, into a
#   sigmoid-calibrated `LinearSVC`. Calibration folds are grouped by prompt too, and the
#   whole feature pipeline is refit inside each one.
# * `emb_plain` — per-layer `StandardScaler` + logistic head, probabilities averaged.
# * `emb_centred` — same heads on cluster-centred embeddings, so the classifier sees
#   within-cluster contrast rather than absolute topic.

# %%
class StyleCounts(TransformerMixin, BaseEstimator):
    CHARS = list("—–;:,.!?\"'’“”*()#-%+/") + ["\n", "\t"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            n = max(len(t), 1)
            w = t.split()
            sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s]
            slen = [len(s) for s in sents]
            f = [np.log1p(n), np.log1p(len(w)), np.log1p(len(sents))]
            f += [t.count(c) / n * 1000 for c in self.CHARS]
            f += [t.count("\n\n"), t.count("  "), t.count("..."), t.count(" — ")]
            f += [
                sum(c.isupper() for c in t) / n * 100,
                sum(c.isdigit() for c in t) / n * 100,
                sum(ord(c) > 127 for c in t) / n * 100,
                float(np.mean([len(x) for x in w])) if w else 0.0,
                float(np.mean(slen)) if slen else 0.0,
                float(np.std(slen)) if len(slen) > 1 else 0.0,
                len(w) / max(len(sents), 1),
                len({x.lower().strip(".,;:!?()\"'") for x in w}) / max(len(w), 1),
                len({x.lower() for x in w if len(x) > 6}) / max(len(w), 1),
            ]
            rows.append(f)
        return np.asarray(rows, dtype=np.float64)


def make_svm():
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True,
                                 lowercase=False, min_df=2, max_features=300_000)),
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                                 min_df=2, token_pattern=r"\S+")),
        ("style", Pipeline([("x", StyleCounts()), ("s", StandardScaler())])),
    ], transformer_weights={"char": 1.0, "word": 1.0, "style": 0.3})
    return Pipeline([("feats", feats),
                     ("svm", LinearSVC(C=1.0, max_iter=5000, random_state=SEED))])


def svm_proba(texts_fit, y_fit, groups_fit, texts_pred, n_jobs=1):
    inner = list(GroupKFold(INNER_FOLDS).split(texts_fit, y_fit, groups_fit))
    cal = CalibratedClassifierCV(make_svm(), method="sigmoid", cv=inner,
                                 ensemble=False, n_jobs=n_jobs)
    cal.fit(texts_fit, y_fit)
    P = cal.predict_proba(texts_pred)
    return P[:, [list(cal.classes_).index(k) for k in range(K)]]


def _center(X, cl):
    """Subtract the per-cluster mean, so a head reads within-cluster contrast."""
    X = np.asarray(X, dtype=np.float32).copy()
    for c in np.unique(cl):
        i = cl == c
        if i.sum() > 1:
            X[i] -= X[i].mean(0)
    return X


def _random_chunks(groups_, size, rng):
    out = np.empty(len(groups_), dtype=object)
    for g in np.unique(groups_):
        members = rng.permutation(np.where(groups_ == g)[0])
        for c, start in enumerate(range(0, len(members), size)):
            out[members[start:start + size]] = f"{g}_{c}"
    return out


def emb_proba(E_fit, y_fit, E_pred):
    P = np.zeros((len(E_pred), K))
    for L in HEAD_LAYERS:
        head = Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(C=EMB_C, max_iter=5000, random_state=SEED))])
        head.fit(E_fit[:, L].astype(np.float32), y_fit)
        P += head.predict_proba(E_pred[:, L].astype(np.float32)) / len(HEAD_LAYERS)
    return P


def emb_centred_proba(E_fit, y_fit, groups_fit, E_pred, cl_pred, plain_pred):
    """Heads on cluster-centred embeddings.

    Training rows are centred within random chunks of each `CENTRE_CHUNKS` size, so the
    head sees the same group geometry it faces at prediction time. One layer is built at a
    time: stacking every layer over every chunking at once would not fit in memory.
    """
    rng = np.random.default_rng(SEED)
    chunkings = [_random_chunks(groups_fit, s, rng) for s in CENTRE_CHUNKS]
    y_stack = np.tile(y_fit, len(chunkings))
    P = np.zeros((len(E_pred), K))
    for L in HEAD_LAYERS:
        col = E_fit[:, L].astype(np.float32)
        X = np.concatenate([_center(col, ch) for ch in chunkings])
        head = Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(C=EMB_C, max_iter=3000, random_state=SEED))])
        head.fit(X, y_stack)
        P += head.predict_proba(_center(E_pred[:, L].astype(np.float32), cl_pred)) / len(HEAD_LAYERS)
        del X, col
    # a cluster of one has nothing to contrast against; keep the plain head there
    _, inv, cnt = np.unique(cl_pred, return_inverse=True, return_counts=True)
    singleton = cnt[inv] < 2
    P[singleton] = plain_pred[singleton]
    return P


def branches(texts_fit, y_fit, groups_fit, texts_pred, E_fit, cl_pred, E_pred, n_jobs=1):
    P_svm = svm_proba(texts_fit, y_fit, groups_fit, texts_pred, n_jobs=n_jobs)
    if NO_QWEN:
        n = len(texts_pred)
        return {"svm": P_svm, "emb_plain": np.full((n, K), 1 / K), "emb_centred": np.full((n, K), 1 / K)}
    P_plain = emb_proba(E_fit, y_fit, E_pred)
    P_centred = emb_centred_proba(E_fit, y_fit, groups_fit, E_pred, cl_pred, P_plain)
    return {"svm": P_svm, "emb_plain": P_plain, "emb_centred": P_centred}


def logblend(parts, w):
    return sum(w[k] * np.log(np.clip(parts[k], 1e-12, 1)) for k in parts)


# %% [markdown]
# ## 4. Cluster-aware assignment
#
# A prompt is a permutation of the 18 labels, so once the passages of one prompt are
# grouped, the labels should be *used up* together rather than chosen independently.
# `linear_sum_assignment` on the negated log-probabilities finds the maximum-probability
# bijection, which is exactly the balanced assignment.

# %%
def assign_clusters(logits, cl, report=True):
    sizes = np.bincount(cl)[cl]
    pred = logits.argmax(1)
    touched = np.zeros(len(pred), dtype=bool)
    for c in np.unique(cl):
        rows = np.where(cl == c)[0]
        m = len(rows)
        if m in ASSIGN_EXACT or 2 <= m <= ASSIGN_MAX_SMALL:
            _, cols = linear_sum_assignment(-logits[rows])
            pred[rows] = cols
            touched[rows] = True
    if report:
        log(f"  assignment fired on {int(touched.sum())}/{len(pred)} rows "
            f"({touched.mean():.1%}); sizes hit: "
            f"{sorted(set(int(s) for s in sizes[touched])) if touched.any() else []}")
    return pred


def fit_blend(oof, y_true, names, fallback):
    """Coarse simplex grid search on OOF balanced accuracy (3 free parameters).

    Ties break toward the fallback weights: adding a constant to every log-probability
    leaves the argmax unchanged, so a near-uniform branch can produce spurious winners.
    """
    best, best_w = -1.0, None
    for a in np.arange(0, 1.01, 0.1):
        for b in np.arange(0, 1.01 - a + 1e-9, 0.1):
            c = 1 - a - b
            w = dict(zip(names, (a, b, c)))
            s = balanced_accuracy_score(y_true, logblend(oof, w).argmax(1))
            closer = sum(abs(w[k] - fallback[k]) for k in names)
            if s > best + 1e-9 or (abs(s - best) <= 1e-9 and best_w is not None
                                   and closer < sum(abs(best_w[k] - fallback[k]) for k in names)):
                best, best_w = s, w
    return best_w, best


# %% [markdown]
# ## 5. Prompt-grouped 5-fold CV (interactive runs only)

# %%
BLEND = dict(BLEND_FALLBACK)
fitted_note = "fallback weights"
oof_parts = None

if RUN_CV:
    n_jobs = min(N_FOLDS, os.cpu_count() or 1)
    svm_parts = Parallel(n_jobs=n_jobs)(
        delayed(svm_proba)(texts[f], y[f], groups[f], texts[v]) for f, v in folds)

    oof_parts = {k: np.zeros((len(train), K)) for k in ["svm", "emb_plain", "emb_centred"]}
    fold_id = np.zeros(len(train), dtype=int)
    for i, ((f, v), ps) in enumerate(zip(folds, svm_parts)):
        oof_parts["svm"][v] = ps
        if not NO_QWEN:
            oof_parts["emb_plain"][v] = emb_proba(E_tr[f], y[f], E_tr[v])
            oof_parts["emb_centred"][v] = emb_centred_proba(
                E_tr[f], y[f], groups[f], E_tr[v], train_cl[v], oof_parts["emb_plain"][v])
        fold_id[v] = i
        accs = {k: balanced_accuracy_score(y[v], p[v].argmax(1)) for k, p in oof_parts.items()}
        log(f"fold {i}: " + " ".join(f"{k} {v_:.4f}" for k, v_ in accs.items()))

    log("individual branches:")
    for k, p in oof_parts.items():
        log(f"  {k:<12} {balanced_accuracy_score(y, p.argmax(1)):.4f}")

    names = list(oof_parts)
    w, s = fit_blend(oof_parts, y, names, BLEND_FALLBACK)
    BLEND, fitted_note = w, f"fitted on OOF ({s:.4f})"
    log(f"blend weights {({k: round(float(v), 2) for k, v in w.items()})} [{fitted_note}]")
    log(f"blend          {s:.4f}")

    logits = logblend(oof_parts, BLEND)
    log("cluster-aware assignment on OOF:")
    final = assign_clusters(logits, train_cl)
    acc_assign = balanced_accuracy_score(y, final)
    log(f"before {balanced_accuracy_score(y, logits.argmax(1)):.4f} -> after {acc_assign:.4f} "
        f"({acc_assign - balanced_accuracy_score(y, logits.argmax(1)):+.4f})")

    recall = pd.Series([(final[y == k] == k).mean() for k in range(K)], index=LABELS)
    log("weakest classes:\n" + recall.sort_values().head(8).round(3).to_string())
    fam = np.array([m.split("-")[0] for m in LABELS])[final]
    log(f"provider-family accuracy {(fam == np.array([m.split('-')[0] for m in train.model])).mean():.4f}")

    pd.DataFrame(confusion_matrix(y, final, labels=range(K)), index=LABELS, columns=LABELS).to_csv(
        OUT / "oof_confusion_matrix.csv")
    pd.DataFrame({"model": LABELS, "oof_recall": recall.loc[LABELS].values}).to_csv(
        OUT / "oof_per_class_recall.csv", index=False)
    np.savez_compressed(OUT / "oof.npz", labels=LABELS, y=y, folds=fold_id,
                        **{k: v for k, v in oof_parts.items()})
    (OUT / "cv_summary.json").write_text(json.dumps(
        {"blend": BLEND, "blend_source": fitted_note, "blend_oof": float(s),
         "with_assignment": float(acc_assign)}, indent=2))

# %% [markdown]
# ## 6. Final fit on all training rows -> `submission.parquet`

# %%
log(f"final fit with blend {({k: round(float(v), 3) for k, v in BLEND.items()})} [{fitted_note}]")
parts = branches(texts, y, groups, test_texts, E_tr, test_cl, E_te, n_jobs=-1)
for k, p in parts.items():
    assert p.shape == (len(test), K) and np.isfinite(p).all(), f"bad probabilities for {k}"
    log(f"  {k:<12} rows={len(p)} mean top prob={p.max(1).mean():.3f}")

logits = logblend(parts, BLEND)
pred = assign_clusters(logits, test_cl)
log(f"predicted class distribution: {len(np.unique(pred))}/{K} classes used")

submission = pd.DataFrame({"id": test_ids, "model": LABELS[pred]})
assert list(submission.columns) == ["id", "model"]
assert len(submission) == len(test) and submission.id.is_unique
assert submission.id.tolist() == test.id.tolist()
assert not submission.isna().any().any() and submission.model.isin(LABELS).all()

path = OUT / "submission.parquet"
submission.to_parquet(path, index=False)
assert pd.read_parquet(path).equals(submission)
np.savez_compressed(OUT / "test_proba.npz", labels=LABELS, ids=test_ids,
                    clusters=test_cl, **{k: v for k, v in parts.items()})

log(f"wrote {len(submission)} rows to {path}")
print(submission.model.value_counts().to_string())