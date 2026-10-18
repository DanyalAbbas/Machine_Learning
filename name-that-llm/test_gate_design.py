"""Final gate design: hard Hungarian on clean clusters + soft Sinkhorn elsewhere.

Band analysis showed the gain is concentrated in size-18 clusters (+0.090) and small
clusters, while 9-17 and 19-40 bands are neutral-to-negative. That makes sense: at
threshold 0.9 the clustering merges some prompts and splits others, so only the
cleanly-sized clusters have a trustworthy balanced structure.

Validate the combined rule on the leak-free v1 OOF probabilities.
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


def hard_assign(L, cl, sizes_ok):
    """Hungarian inside clusters whose size is in `sizes_ok`."""
    out = L.argmax(1)
    for c in np.unique(cl):
        rows = np.where(cl == c)[0]
        if len(rows) in sizes_ok:
            _, cols = linear_sum_assignment(-L[rows])
            out[rows] = cols
    return out


def soft_assign(L, cl, strength, min_size=2, iters=30):
    S = L.copy()
    for _ in range(iters):
        for c in np.unique(cl):
            rows = np.where(cl == c)[0]
            if len(rows) < min_size:
                continue
            col = logsumexp(S[rows], axis=0) / len(rows)
            S[rows] -= strength * (col - col.mean())
    return S


def main():
    tr = pd.read_csv(HERE.parent / "data" / "train.csv", dtype=str)
    P = np.load(HERE / "artifacts" / "v1_stylometry" / "oof_proba.npy")
    y = np.array([list(LABELS).index(m) for m in tr.model])
    prompts = tr.prompt_id.to_numpy(dtype=str)
    texts = tr.text.to_numpy(dtype=object)
    L = np.log(np.clip(P, 1e-12, 1))
    base = balanced_accuracy_score(y, L.argmax(1))
    print(f"independent argmax {base:.4f}\n")

    # ANALYSIS ONLY (uses labels, never used at prediction time): is a cluster a clean
    # subset of exactly one prompt?
    for th in [0.85, 0.9, 0.92]:
        cl = topic_clusters(texts, th)
        sizes = np.bincount(cl)[cl]
        single = np.array([len(set(prompts[cl == c])) == 1 for c in cl])
        print(f"thresh {th}: rows in single-prompt clusters {int(single.sum()):>4}/{len(tr)} "
              f"({single.mean():.1%}); exact-18 rows {int((sizes == 18).sum()):>4}")
        print(f"{'  rule':<42}{'score':>8}{'delta':>9}")

        rules = {
            "none (argmax)": L.argmax(1),
            "hard on size==18": hard_assign(L, cl, {18}),
            "hard on size 2-8 or ==18": hard_assign(L, cl, {2, 3, 4, 5, 6, 7, 8, 18}),
            "hard on size 9-18": hard_assign(L, cl, set(range(9, 19))),
            "hard on size 2-18": hard_assign(L, cl, set(range(2, 19))),
            "soft 0.25 then hard 18": soft_assign(L, cl, 0.25).argmax(1),
        }
        for name, pred in rules.items():
            s = balanced_accuracy_score(y, pred)
            print(f"  {name:<40}{s:>8.4f}{s - base:>+9.4f}")

        for st in [0.25, 0.5]:
            p = soft_assign(L, cl, st).argmax(1)
            s = balanced_accuracy_score(y, p)
            print(f"  {'soft sinkhorn ' + str(st):<40}{s:>8.4f}{s - base:>+9.4f}")

        # hard first (size 18), then a soft pass over whatever clusters remain
        h = hard_assign(L, cl, {18})
        done = np.zeros(len(tr), dtype=bool)
        for c in np.unique(cl):
            r = np.where(cl == c)[0]
            if len(r) == 18:
                done[r] = True
        S = L.copy()
        S[done] = -1e9
        S[~done] = soft_assign(L[~done], cl[~done], 0.4)
        s = balanced_accuracy_score(y, S.argmax(1))
        print(f"  {'hard 18 + soft 0.4 on the rest':<40}{s:>8.4f}{s - base:>+9.4f}\n")


if __name__ == "__main__":
    main()