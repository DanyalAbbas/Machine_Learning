"""Generalise the cluster prior to clusters of any size, and find a safe gate.

Key question the Hungarian test left open: the hidden test set may not have 18-member
clusters. Instead of requiring exactly 18, give every label a capacity of
ceil(m / K) inside a cluster of m members and solve the assignment. This is strictly
conservative: it never forbids a label the balanced-prompt structure allows, it only
forbids over-using one.

Run against several clustering thresholds and capacity rules, on the same leak-free v1
OOF probabilities.
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


def topic_clusters(texts, thresh):
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), method="average"), thresh, criterion="distance")


def capacity_assign(P, cl, min_size=1, mode="ceil", scale=1.0):
    """Assign labels inside each cluster with per-label capacity = f(m)."""
    L = np.log(np.clip(P, 1e-12, 1))
    out = P.argmax(1)
    for c in np.unique(cl):
        rows = np.where(cl == c)[0]
        m = len(rows)
        if m < max(2, min_size):
            continue
        if mode == "ceil":
            cap = int(np.ceil(m / K * scale))
        elif mode == "one":
            cap = 1
        else:
            raise ValueError(mode)
        cap = max(1, cap)
        # rows x (K * cap) cost matrix: column (k, j) == "label k used for the j-th time"
        cost = -np.repeat(L[rows], cap, axis=1)
        r, cidx = linear_sum_assignment(cost)
        out[rows[r]] = cidx // cap
    return out


def sinkhorn(P, cl, strength, min_size=2, iters=30):
    L = np.log(np.clip(P, 1e-12, 1)).copy()
    for _ in range(iters):
        for c in np.unique(cl):
            rows = np.where(cl == c)[0]
            if len(rows) < min_size:
                continue
            col = logsumexp(L[rows], axis=0) / len(rows)
            L[rows] -= strength * (col - col.mean())
    return L


def main():
    tr = pd.read_csv(HERE.parent / "data" / "train.csv", dtype=str)
    P = np.load(HERE / "artifacts" / "v1_stylometry" / "oof_proba.npy")
    y = np.array([list(LABELS).index(m) for m in tr.model])
    texts = tr.text.to_numpy(dtype=object)
    base = P.argmax(1)
    b = balanced_accuracy_score(y, base)
    print(f"independent argmax {b:.4f}\n")

    print(f"{'thresh':>6} {'cl':>4} {'>=18':>5} {'2-17':>5} {'1':>4} | "
          f"{'cap all>=2':>10} {'cap>=9':>8} {'cap>=4':>8} {'cap one':>8} {'sink.5':>8}")
    for th in [0.85, 0.88, 0.9, 0.92, 0.95]:
        cl = topic_clusters(texts, th)
        s = np.bincount(cl)[cl]
        row = [f"{th:>6}", f"{len(np.unique(cl)):>4}", f"{int((s == 18).sum()):>5}",
               f"{int(((s >= 2) & (s < 18)).sum()):>5}", f"{int((s == 1).sum()):>4}"]
        for kw in [dict(min_size=2), dict(min_size=9), dict(min_size=4), dict(min_size=2, mode="one")]:
            p = capacity_assign(P, cl, **kw)
            row.append(f"{balanced_accuracy_score(y, p):>10.4f}" if kw.get("mode") is None
                       else f"{balanced_accuracy_score(y, p):>8.4f}")
        sp = sinkhorn(P, cl, 0.5).argmax(1)
        row.append(f"{balanced_accuracy_score(y, sp):>8.4f}")
        print(" ".join(row))

    # Does capacity assignment ever *hurt* a specific cluster-size band?
    print("\nper-size-band effect at thresh 0.9 (capacity=ceil(m/18), all sizes>=2):")
    cl = topic_clusters(texts, 0.9)
    s = np.bincount(cl)[cl]
    new = capacity_assign(P, cl, min_size=2)
    for lo, hi in [(2, 3), (4, 8), (9, 17), (18, 18), (19, 40)]:
        m = (s >= lo) & (s <= hi)
        if m.sum() < 20:
            continue
        print(f"  size {lo:>2}-{hi:<2} n={int(m.sum()):>4}  {balanced_accuracy_score(y[m], base[m]):.4f} "
              f"-> {balanced_accuracy_score(y[m], new[m]):.4f} "
              f"({balanced_accuracy_score(y[m], new[m]) - balanced_accuracy_score(y[m], base[m]):+.4f})")

    print("\ncontrol (shuffle within cluster):", end=" ")
    rng = np.random.default_rng(0)
    sh = base.copy()
    for c in np.unique(cl):
        rows = np.where(cl == c)[0]
        sh[rows] = rng.permutation(sh[rows])
    print(f"{balanced_accuracy_score(y, sh):.4f}")


if __name__ == "__main__":
    main()