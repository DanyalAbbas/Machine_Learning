# %% [markdown]
# # Name That LLM — Baseline (v1): stylometry + lexical TF-IDF, fold ensemble
#
# A fast, dependency-light, **CPU-only** proof of concept that produces a valid
# `submission.parquet` on the first try.
#
# | piece | choice | why |
# |---|---|---|
# | validation | 5-fold `GroupKFold` on `prompt_id` | test prompts are *unseen*, so a random split leaks topic and inflates the score |
# | features | case-sensitive char 1–5 grams + word 1–2 grams + hand-made stylometry | punctuation habits are the strongest per-family signal in this data |
# | model | `LinearSVC`, decision scores averaged over the 5 folds | linear + tiny data; fold averaging beats one refit |
# | runtime | a few minutes on Kaggle CPU, no downloads, no GPU | nothing that can break at scoring time |
#
# **Kaggle notebook settings: Internet OFF. GPU not required.**

# %%
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
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

# The 18 labels are fixed by the competition. Hard-coding them means a malformed or
# reordered `train.csv` can never silently change the submission schema.
LABELS = np.array(sorted([
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]))
K = len(LABELS)

T0 = time.time()
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# %% [markdown]
# ## 1. Data
#
# Train is 2,699 passages = 150 prompts × 18 models (one response per model per prompt),
# so the classes are balanced by construction. The hidden test set is ~4,000 passages
# written for prompts that **do not** appear in `train.csv`, and its class mix is *not*
# balanced — which is why the metric is balanced accuracy rather than plain accuracy.

# %%
DATA = next(
    (p for p in [Path("/kaggle/input/competitions/name-that-llm"),
                 Path("/kaggle/input/name-that-llm"),
                 Path("data"), Path("../../data")]
     if (p / "train.csv").is_file()),
    None,
)
if DATA is None:
    raise FileNotFoundError("Attach the 'name-that-llm' competition data to this notebook.")

train = pd.read_csv(DATA / "train.csv", dtype=str)
test = pd.read_csv(DATA / "test.csv", dtype=str)

for frame, cols in [(train, ["id", "prompt_id", "text", "model"]), (test, ["id", "text"])]:
    missing = set(cols) - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns {missing}")

train["text"] = train.text.fillna("").astype(str)
test["text"] = test.text.fillna("").astype(str)
assert train.id.is_unique and test.id.is_unique
assert set(train.model) == set(LABELS), "unexpected label set in train.csv"

log(f"data dir {DATA}")
log(f"train {train.shape} ({train.prompt_id.nunique()} prompts) | test {test.shape}")

# %% [markdown]
# ## 2. Features
#
# Character n-grams pick up punctuation fingerprints, which are unusually discriminative
# here. Measured means per 1,000 characters on the training set:
#
# | model | em dash `—` | curly `’` | `:` |
# |---|---|---|---|
# | `glm-5.3` | 2.36 | 0.00 | 1.17 |
# | `claude-opus-5` | 1.97 | 0.00 | 1.29 |
# | `gpt-5.5` | 0.28 | 1.08 | 1.30 |
# | `gemini-3.6-flash` | 0.00 | 0.30 | **0.25** |
# | `qwen3.8-max` | 0.08 | 0.77 | 0.75 |
#
# Provider families separate almost perfectly on these habits; telling models *within* a
# family is the hard part (see the per-class recall printed below).
#
# The explicit `StyleCounts` block measures those habits directly — normalised by length
# so 40-word and 250-word passages are comparable — instead of forcing the linear model to
# rediscover them from sparse n-grams.

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
    """Decision scores -> probabilities with columns ordered exactly like `LABELS`."""
    classes = list(model.classes_)
    return softmax(model.decision_function(X))[:, [classes.index(c) for c in LABELS]]


# %% [markdown]
# ## 3. Prompt-grouped 5-fold CV
#
# Grouping by `prompt_id` is the whole point: all 18 answers to a prompt stay in the same
# fold, which mimics the unseen-prompt test condition and stops the model from memorising
# topic vocabulary instead of author style.

# %%
X_all = train.text
y_all = train.model
groups = train.prompt_id
folds = list(GroupKFold(N_FOLDS).split(X_all, y_all, groups))

oof = np.zeros((len(train), K))
fold_scores = []
for i, (fit, val) in enumerate(folds):
    assert not (set(groups.iloc[fit]) & set(groups.iloc[val])), "prompt leakage between folds"
    model = make_model()
    model.fit(X_all.iloc[fit], y_all.iloc[fit])
    p = aligned_proba(model, X_all.iloc[val])
    oof[val] = p
    s = balanced_accuracy_score(y_all.iloc[val], LABELS[p.argmax(1)])
    fold_scores.append(s)
    log(f"fold {i + 1}/{N_FOLDS}: {len(fit)} train rows / {len(val)} val rows -> bal_acc {s:.4f}")

oof_pred = LABELS[oof.argmax(1)]
cv_score = balanced_accuracy_score(y_all, oof_pred)
log(f"OOF balanced accuracy: {cv_score:.4f} +/- {np.std(fold_scores):.4f}")

recall = pd.Series([(oof_pred[y_all.values == c] == c).mean() for c in LABELS], index=LABELS)
log("weakest classes:\n" + recall.sort_values().head(8).round(3).to_string())
family_pred = np.array([m.split("-")[0] for m in oof_pred])
family_true = np.array([m.split("-")[0] for m in y_all])
log(f"provider-family accuracy: {(family_pred == family_true).mean():.4f}")

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    confusion_matrix(y_all, oof_pred, labels=LABELS), index=LABELS, columns=LABELS,
).to_csv(OUT / "oof_confusion_matrix.csv")
pd.DataFrame({"model": LABELS, "oof_recall": recall.loc[LABELS].values}).to_csv(
    OUT / "oof_per_class_recall.csv", index=False)
np.save(OUT / "oof_proba.npy", oof)

# %% [markdown]
# ## 4. Fold ensemble -> `submission.parquet`
#
# For a linear model on 2.7k rows, averaging the five fold models beats refitting once on
# everything, so we keep the folds and average their decision scores on the test set.

# %%
test_proba = np.zeros((len(test), K))
for i, (fit, _) in enumerate(folds):
    model = make_model()
    model.fit(X_all.iloc[fit], y_all.iloc[fit])
    test_proba += aligned_proba(model, test.text) / N_FOLDS
    log(f"fold model {i + 1}/{N_FOLDS} fitted and scored on test")

submission = pd.DataFrame({"id": test.id.to_numpy(), "model": LABELS[test_proba.argmax(1)]})

# Submission contract: exactly these two columns, one row per test id, no nulls, valid labels.
assert list(submission.columns) == ["id", "model"]
assert len(submission) == len(test) and submission.id.is_unique
assert submission.id.tolist() == test.id.tolist()
assert not submission.isna().any().any()
assert submission.model.isin(LABELS).all()

path = OUT / "submission.parquet"
submission.to_parquet(path, index=False)
assert pd.read_parquet(path).equals(submission), "parquet round-trip mismatch"
np.save(OUT / "test_proba.npy", test_proba)

log(f"wrote {len(submission)} rows to {path}")
print(submission.head(10).to_string(index=False))
print(submission.model.value_counts().to_string())