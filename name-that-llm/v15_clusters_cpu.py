# %% [markdown]
# # Name That LLM — v1.5: v1's proven SVM branch + cluster-aware assignment
#
# Same model as v1 (grouped CV 0.3771, public LB 0.360). **CPU only, no GPU, no downloads.**
#
# The only change is the decision rule. Train is 150 prompts x 18 models, so every prompt
# is a *permutation* of the 18 labels — a structural constraint on the predictions that
# also holds for the hidden test set, because test passages group by prompt the same way.
#
# So instead of taking 18 independent argmaxes per prompt, solve one
# `linear_sum_assignment` over negated log-probabilities: the maximum-probability
# bijection, which uses each label exactly once.
#
# Measured on v1's out-of-fold probabilities, so the comparison is exact — identical
# predictions, only the rule for reading them changes:
#
# | decision rule | balanced accuracy |
# |---|---|
# | independent argmax (v1) | 0.3771 |
# | Hungarian on clusters of size exactly 18 | 0.4508 |
# | **Hungarian on size 2-8 and exactly 18** | **0.4523** |
# | Hungarian on every cluster of size 2-18 | 0.4508 |
# | soft Sinkhorn, strength 0.5 | 0.4134 |
# | control: shuffle labels inside clusters | 0.0670 |
#
# **+0.075, and it needs no labels, so it transfers to the hidden set.**
#
# The gate is deliberately narrow. A per-size-band breakdown at threshold 0.9:
#
# | cluster size | n | argmax | after assignment |
# |---|---|---|---|
# | 2-3 | 30 | 0.3519 | 0.4074 |
# | 4-8 | 73 | 0.2380 | 0.2815 |
# | 9-17 | 252 | 0.4394 | **0.4238** |
# | 18 | 2214 | 0.3749 | 0.4648 |
# | 19-40 | 99 | 0.4157 | **0.3917** |
#
# Size 18 is verified safe: all 123 size-18 train clusters hold exactly one passage per
# model. Sizes 9-17 are the residue of imperfect splits and measurably *hurt*, so they are
# skipped, as are singletons (nothing to constrain) and oversized merged clusters.

# %%
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
N_FOLDS = 5
SVC_C = 1.0
CHAR_NGRAM = (1, 5)
CHAR_MAX_FEATURES = 300_000
STYLE_WEIGHT = 0.3

CLUSTER_THRESH = 0.9
ASSIGN_EXACT = {18}      # a whole prompt: verified exactly one-per-model, 123/123 on train
ASSIGN_MAX_SMALL = 8     # a fragment of one prompt: labels must still be distinct

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
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# %% [markdown]
# ## 1. Data

# %%
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

log(f"train {train.shape} ({train.prompt_id.nunique()} prompts) | test {test.shape}")


# %% [markdown]
# ## 2. Features — unchanged from v1
#
# Case-sensitive char 1-5 grams + word 1-2 grams TF-IDF + explicit stylometry.
# Punctuation habits are the strongest cheap signal here: Anthropic/GLM/Kimi use em
# dashes, Gemini essentially never does, OpenAI uses curly quotes while Claude/GLM/Qwen
# use straight ones.

# %%
class StyleCounts(TransformerMixin, BaseEstimator):
    """Length, punctuation, casing and lexical-diversity statistics."""

    CHARS = list("—–;:,.!?\"'’“”*()#-%+/") + ["\n", "\t"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            n = max(len(t), 1)
            words = t.split()
            sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s]
            slen = [len(s) for s in sents]
            f = [np.log1p(n), np.log1p(len(words)), np.log1p(len(sents))]
            f += [t.count(c) / n * 1000 for c in self.CHARS]
            f += [t.count("\n\n"), t.count("  "), t.count("..."), t.count(" — ")]
            f += [
                sum(c.isupper() for c in t) / n * 100,
                sum(c.isdigit() for c in t) / n * 100,
                sum(ord(c) > 127 for c in t) / n * 100,
                float(np.mean([len(w) for w in words])) if words else 0.0,
                float(np.mean(slen)) if slen else 0.0,
                float(np.std(slen)) if len(slen) > 1 else 0.0,
                len(words) / max(len(sents), 1),
                len({w.lower().strip(".,;:!?()\"'") for w in words}) / max(len(words), 1),
                len({w.lower() for w in words if len(w) > 6}) / max(len(words), 1),
            ]
            rows.append(f)
        return np.asarray(rows, dtype=np.float64)


def make_model():
    features = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char", ngram_range=CHAR_NGRAM, sublinear_tf=True,
                                 lowercase=False, min_df=2, max_features=CHAR_MAX_FEATURES)),
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                                 min_df=2, token_pattern=r"\S+")),
        ("style", Pipeline([("extract", StyleCounts()), ("scale", StandardScaler())])),
    ], transformer_weights={"char": 1.0, "word": 1.0, "style": STYLE_WEIGHT})
    return Pipeline([
        ("features", features),
        ("clf", LinearSVC(C=SVC_C, max_iter=5000, random_state=SEED)),
    ])


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def aligned_proba(model, X):
    classes = list(model.classes_)
    return softmax(model.decision_function(X))[:, [classes.index(c) for c in LABELS]]


# %% [markdown]
# ## 3. Topic clusters
#
# Test rows carry no `prompt_id`, but passages answering the same prompt are topically
# similar, so average-linkage clustering of content-word TF-IDF recovers the grouping.
# On train this reproduces the true prompts at ARI 0.955.

# %%
def topic_clusters(texts, thresh=CLUSTER_THRESH):
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), method="average"),
                    thresh, criterion="distance")


train_cl = topic_clusters(train.text.to_numpy(dtype=object))
test_cl = topic_clusters(test.text.to_numpy(dtype=object))
sizes = np.bincount(train_cl)[train_cl]
hist = {int(k): int(v) for k, v in zip(*np.unique(np.bincount(test_cl)[1:], return_counts=True))}
log(f"train clusters {len(np.unique(train_cl))} | size-18 rows {int((sizes == 18).sum())}")
log(f"test clusters {len(np.unique(test_cl))} | size histogram {hist}")


# %% [markdown]
# ## 4. Cluster-aware assignment


# %%
def assign_clusters(logits, cl):
    pred = logits.argmax(1)
    sizes = np.bincount(cl)[cl]
    touched = np.zeros(len(pred), dtype=bool)
    for c in np.unique(cl):
        rows = np.where(cl == c)[0]
        m = len(rows)
        if m in ASSIGN_EXACT or 2 <= m <= ASSIGN_MAX_SMALL:
            _, cols = linear_sum_assignment(-logits[rows])
            pred[rows] = cols
            touched[rows] = True
    changed = int((pred != logits.argmax(1)).sum())
    log(f"  fired on {int(touched.sum())}/{len(pred)} rows ({touched.mean():.1%}); "
        f"sizes {sorted({int(s) for s in sizes[touched]}) if touched.any() else []}; "
        f"changed {changed} predictions")
    return pred


# %% [markdown]
# ## 5. Prompt-grouped 5-fold CV
#
# Unchanged from v1: all 18 answers to a prompt stay in one fold, which is what makes the
# local number comparable to the leaderboard.

# %%
X_all = train.text
y_all = train.model
groups = train.prompt_id
folds = list(GroupKFold(N_FOLDS).split(X_all, y_all, groups))

oof = np.zeros((len(train), K))
for i, (fit, val) in enumerate(folds):
    assert not (set(groups.iloc[fit]) & set(groups.iloc[val])), "prompt leakage between folds"
    model = make_model()
    model.fit(X_all.iloc[fit], y_all.iloc[fit])
    oof[val] = aligned_proba(model, X_all.iloc[val])
    log(f"fold {i + 1}/{N_FOLDS}: {len(fit)} fit / {len(val)} val -> "
        f"bal_acc {balanced_accuracy_score(y_all.iloc[val], LABELS[oof[val].argmax(1)]):.4f}")

logits = np.log(np.clip(oof, 1e-12, 1))
before = balanced_accuracy_score(y_all, LABELS[logits.argmax(1)])
log(f"argmax:            {before:.4f}")
log("with assignment:")
final = assign_clusters(logits, train_cl)
after = balanced_accuracy_score(y_all, LABELS[final])
log(f"assignment:        {after:.4f}  ({after - before:+.4f})")

recall = pd.Series([(LABELS[final][y_all.values == c] == c).mean() for c in LABELS], index=LABELS)
log("weakest classes:\n" + recall.sort_values().head(8).round(3).to_string())
fam = np.array([m.split("-")[0] for m in LABELS])[final]
log(f"provider-family accuracy: {(fam == np.array([m.split('-')[0] for m in y_all])).mean():.4f}")

pd.DataFrame(confusion_matrix(y_all, LABELS[final], labels=LABELS),
             index=LABELS, columns=LABELS).to_csv(OUT / "oof_confusion_matrix.csv")
pd.DataFrame({"model": LABELS, "oof_recall": recall.loc[LABELS].values}).to_csv(
    OUT / "oof_per_class_recall.csv", index=False)
np.save(OUT / "oof_proba.npy", oof)

# %% [markdown]
# ## 6. Fold ensemble -> `submission.parquet`

# %%
test_logits = np.zeros((len(test), K))
for i, (fit, _) in enumerate(folds):
    model = make_model()
    model.fit(X_all.iloc[fit], y_all.iloc[fit])
    test_logits += np.log(np.clip(aligned_proba(model, test.text), 1e-12, 1)) / N_FOLDS
    log(f"fold model {i + 1}/{N_FOLDS} scored on test")

pred = assign_clusters(test_logits, test_cl)
submission = pd.DataFrame({"id": test.id.to_numpy(), "model": LABELS[pred]})

assert list(submission.columns) == ["id", "model"]
assert len(submission) == len(test) and submission.id.is_unique
assert submission.id.tolist() == test.id.tolist()
assert not submission.isna().any().any() and submission.model.isin(LABELS).all()

path = OUT / "submission.parquet"
submission.to_parquet(path, index=False)
assert pd.read_parquet(path).equals(submission)
log(f"wrote {len(submission)} rows to {path}")
print(submission.model.value_counts().to_string())