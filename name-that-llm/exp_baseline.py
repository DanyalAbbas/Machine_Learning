"""Local experiment harness: prompt-grouped CV for Name That LLM baselines."""

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

LABELS = sorted([
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
])
DATA = Path(__file__).resolve().parents[1] / "data"
SEED = 42
N_FOLDS = 5


class StyleCounts(TransformerMixin, BaseEstimator):
    """Hand-crafted stylometry: punctuation habits, lengths, casing, lexical diversity."""

    CHARS = list("—–;:,.!?\"'’“”*()#-%+/") + ["\n", "\t"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            n = max(len(t), 1)
            w = t.split()
            sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s]
            f = [np.log1p(n), np.log1p(len(w)), np.log1p(len(sents))]
            f += [t.count(c) / n * 1000 for c in self.CHARS]
            f += [t.count("\n\n"), t.count("  "), t.count("..."), t.count(" — ")]
            f += [
                sum(c.isupper() for c in t) / n * 100,
                sum(c.isdigit() for c in t) / n * 100,
                sum(ord(c) > 127 for c in t) / n * 100,
                np.mean([len(x) for x in w]) if w else 0.0,
                np.mean([len(x) for x in sents]) if sents else 0.0,
                np.std([len(x) for x in sents]) if len(sents) > 1 else 0.0,
                len(w) / max(len(sents), 1),
                len(set(x.lower().strip(".,;:!?()\"'") for x in w)) / max(len(w), 1),
                len(set(x.lower() for x in w if len(x) > 6)) / max(len(w), 1),
                np.mean([len(x) for x in w if x[:1].isupper()]) if any(x[:1].isupper() for x in w) else 0.0,
            ]
            rows.append(f)
        return np.asarray(rows, dtype=np.float64)


def starter() -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=35000, sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(C=2.0, max_iter=1000)),
    ])


def lexical(C: float = 1.0, clf: str = "svc", char_ng=(1, 5), maxf=300_000, style_w=0.3) -> Pipeline:
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char", ngram_range=char_ng, sublinear_tf=True,
                                 lowercase=False, min_df=2, max_features=maxf)),
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                                 min_df=2, token_pattern=r"\S+")),
        ("style", Pipeline([("x", StyleCounts()), ("s", StandardScaler())])),
    ], transformer_weights={"char": 1.0, "word": 1.0, "style": style_w})
    model = LinearSVC(C=C, max_iter=5000, random_state=SEED) if clf == "svc" \
        else LogisticRegression(C=C, max_iter=3000, random_state=SEED)
    return Pipeline([("features", feats), ("clf", model)])


def load():
    tr = pd.read_csv(DATA / "train.csv", dtype=str)
    te = pd.read_csv(DATA / "test.csv", dtype=str)
    tr["text"] = tr.text.fillna("")
    te["text"] = te.text.fillna("")
    return tr, te


def oof_proba(model, X, y, folds):
    P = np.zeros((len(y), len(LABELS)))
    for f, v in folds:
        m = model()
        m.fit(X.iloc[f], y.iloc[f])
        p = m.predict_proba(X.iloc[v]) if hasattr(m, "predict_proba") else _softmax(m.decision_function(X.iloc[v]))
        P[v] = p[:, [list(m.classes_).index(l) for l in LABELS]]
    return P


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def run(name, model, X, y, groups, folds, return_oof=False):
    t0 = time.time()
    P = oof_proba(model, X, y, folds)
    pred = np.array(LABELS)[P.argmax(1)]
    acc = balanced_accuracy_score(y, pred)
    fam = np.array([m.split("-")[0] for m in LABELS])[P.argmax(1)]
    famacc = (fam == np.array([m.split("-")[0] for m in y])).mean()
    print(f"{name:44s} bal_acc={acc:.4f}  plain_acc={ (pred==y).mean():.4f}  family_acc={famacc:.4f}  ({time.time()-t0:.0f}s)",
          flush=True)
    return (P, acc) if return_oof else acc


def main():
    tr, te = load()
    X, y, groups = tr.text, tr.model, tr.prompt_id
    folds = list(GroupKFold(N_FOLDS).split(X, y, groups))
    print(f"train {tr.shape} | folds {[len(v) for _, v in folds]}\n")

    results = {}
    oofs = {}
    configs = [
        ("starter word12+char35 LR C2", starter),
        ("svc char15+word12+style C1", lambda: lexical(1.0, "svc")),
        ("svc char15+word12+style C0.3", lambda: lexical(0.3, "svc")),
        ("svc char15+word12+style C3", lambda: lexical(3.0, "svc")),
        ("svc char15+word12+style C10", lambda: lexical(10.0, "svc")),
        ("svc char13+word12+style C1", lambda: lexical(1.0, "svc", char_ng=(1, 3))),
        ("svc char15 only +style C1", lambda: svc_char_only(1.0)),
        ("svc word12 only +style C1", lambda: svc_word_only(1.0)),
        ("lr  char15+word12+style C4", lambda: lexical(4.0, "lr")),
        ("style only LR", style_only),
    ]
    for name, fn in configs:
        try:
            r = run(name, fn, X, y, groups, folds, return_oof=True)
            if isinstance(r, tuple):
                oofs[name], results[name] = r
        except Exception as e:  # noqa: BLE001
            print(f"{name:44s} FAILED: {type(e).__name__}: {e}", flush=True)

    print()
    best = max(results, key=results.get)
    print("best single:", best, round(results[best], 4))
    if len(oofs) > 1:
        names = list(oofs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                P = 0.5 * np.log(np.clip(oofs[a], 1e-12, 1)) + 0.5 * np.log(np.clip(oofs[b], 1e-12, 1))
                acc = balanced_accuracy_score(y, np.array(LABELS)[P.argmax(1)])
                if acc > max(results[a], results[b]):
                    print(f"  blend {a} + {b}: {acc:.4f}")


def svc_char_only(C):
    return Pipeline([
        ("features", FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True,
                                     lowercase=False, min_df=2, max_features=300_000)),
            ("style", Pipeline([("x", StyleCounts()), ("s", StandardScaler())])),
        ])),
        ("clf", LinearSVC(C=C, max_iter=5000, random_state=SEED)),
    ])


def svc_word_only(C):
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                                     min_df=2, token_pattern=r"\S+")),
            ("style", Pipeline([("x", StyleCounts()), ("s", StandardScaler())])),
        ])),
        ("clf", LinearSVC(C=C, max_iter=5000, random_state=SEED)),
    ])


def style_only():
    return Pipeline([
        ("features", Pipeline([("x", StyleCounts()), ("s", StandardScaler())])),
        ("clf", LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)),
    ])


if __name__ == "__main__":
    sys.exit(main())