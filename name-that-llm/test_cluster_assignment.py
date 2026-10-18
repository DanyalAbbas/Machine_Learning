"""Does a within-cluster balanced assignment beat independent argmax?

Verified fact (verify_clusters.py): at clustering threshold 0.9, all 123 train clusters
with exactly 18 members contain exactly one passage per model. So "each cluster is a
permutation of the 18 labels" is a valid prior for 18-member clusters.

Test it honestly: take the prompt-grouped OOF probabilities from v1 (no refitting, so no
leakage), cluster the passages, and compare independent argmax against several
assignment schemes. Only clusters of size 18 can support the hard prior, so the gain is
measured on that subset as well as globally.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from scipy.special import logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
LABELS = np.array(sorted([
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]))
K = len(LABELS)


def topic_clusters(texts, thresh=0.9):
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), method="average"), thresh, criterion="distance")


def hungarian(P, cl, size=18):
    """Hard prior: within clusters of exactly `size` members, use every label once."""
    out = P.argmax(1)
    L = np.log(np.clip(P, 1e-12, 1))
    for c in np.unique(cl):
        rows = np.where(cl == c)[0]
        if len(rows) == size:
            _, cols = linear_sum_assignment(-L[rows])
            out[rows] = cols
    return out


def sinkhorn(P, cl, strength, size=18, iters=30):
    """Soft prior: pull each size-`size` cluster toward uniform label usage."""
    L = np.log(np.clip(P, 1e-12, 1)).copy()
    for _ in range(iters):
        for c in np.unique(cl):
            rows = np.where(cl == c)[0]
            if len(rows) != size:
                continue
            col = logsumexp(L[rows], axis=0) / len(rows)
            L[rows] -= strength * (col - col.mean())
    return L


def bal_acc(pred, y, mask=None):
    return balanced_accuracy_score(y if mask is None else y[mask], pred if mask is None else pred[mask])


def main():
    tr = pd.read_csv(HERE.parent / "data" / "train.csv", dtype=str)
    P = np.load(HERE / "artifacts" / "v1_stylometry" / "oof_proba.npy")
    assert P.shape == (len(tr), K), P.shape
    y = np.array([list(LABELS).index(m) for m in tr.model])
    texts = tr.text.to_numpy(dtype=object)

    base = P.argmax(1)
    print(f"independent argmax (v1 OOF)   {bal_acc(base, y):.4f}\n")

    for th in [0.85, 0.88, 0.9, 0.92, 0.95]:
        cl = topic_clusters(texts, th)
        s = np.bincount(cl)[cl]
        m18 = s == 18
        hp = hungarian(P, cl)
        line = (f"thresh {th}: {len(np.unique(cl)):>3} clusters | size-18 rows {int(m18.sum()):>4} "
                f"| all {bal_acc(base, y):.4f} -> hungarian {bal_acc(hp, y):.4f} "
                f"({bal_acc(hp, y) - bal_acc(base, y):+.4f})")
        if m18.sum():
            line += f" | on those rows {bal_acc(base, y, m18):.4f} -> {bal_acc(hp, y, m18):.4f}"
        print(line)

    print("\nsoft version at thresh 0.9")
    cl = topic_clusters(texts, 0.9)
    for st in [0.1, 0.25, 0.5, 0.75, 1.0]:
        sp = sinkhorn(P, cl, st).argmax(1)
        print(f"  strength {st:<5} {bal_acc(sp, y):.4f} ({bal_acc(sp, y) - bal_acc(base, y):+.4f})")

    print("\ncontrol: shuffle labels within cluster (should destroy the prior, not help)")
    rng = np.random.default_rng(0)
    cl = topic_clusters(texts, 0.9)
    shuffled = base.copy()
    for c in np.unique(cl):
        rows = np.where(cl == c)[0]
        shuffled[rows] = rng.permutation(shuffled[rows])
    print(f"  {bal_acc(shuffled, y):.4f}")


if __name__ == "__main__":
    main()