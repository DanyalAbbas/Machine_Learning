"""Regime-based validation harness for Name That LLM.

Two jobs:

1. ANSWER THE QUESTION that invalidated the cluster-assignment idea. Train has exactly
   18 passages per prompt; the hidden test set does not. So how well does test-time
   clustering actually recover prompt groups as passages-per-prompt (k) drops from 18 to 1?

2. Measure any candidate decision rule *per regime* rather than as a single number, since
   a rule that only helps at k=18 is worthless if the test set runs at k=2.

Train only. No test labels anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
LABELS = np.array(sorted([
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]))
K = len(LABELS)
REGIMES = (1, 2, 3, 5, 8, 18)
N_DRAWS = 5
CLUSTER_THRESH = 0.85


def topic_clusters(texts, thresh=CLUSTER_THRESH):
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), method="average"), thresh,
                    criterion="distance")


def pair_scores(cl, prompts):
    """Precision / recall of the clustering against the true prompt partition."""
    same_c = cl[:, None] == cl[None, :]
    same_p = prompts[:, None] == prompts[None, :]
    off = ~np.eye(len(cl), dtype=bool)
    tp = int((same_c & same_p & off).sum())
    prec = tp / max(int((same_c & off).sum()), 1)
    rec = tp / max(int((same_p & off).sum()), 1)
    return prec, rec, np.bincount(cl)[cl]


def assign_clusters(logits, cl, exact=(18,), max_small=8):
    """Balanced assignment inside clusters that look like a whole prompt or a fragment."""
    pred = logits.argmax(1)
    fired = np.zeros(len(pred), dtype=bool)
    for c in np.unique(cl):
        rows = np.where(cl == c)[0]
        m = int(len(rows))
        if m in exact or 2 <= m <= max_small:
            _, cols = linear_sum_assignment(-logits[rows])
            pred[rows] = cols
            fired[rows] = True
    return pred, fired


def subsample(prompts, k, rng):
    """Take k passages per prompt -> a pseudo-test set shaped like the hidden one."""
    idx = []
    for p in np.unique(prompts):
        members = np.where(prompts == p)[0]
        idx.append(rng.choice(members, min(k, len(members)), replace=False))
    return np.concatenate(idx)


def load():
    tr = pd.read_csv(DATA / "train.csv", dtype=str)
    y = tr.model.map({m: i for i, m in enumerate(LABELS)}).to_numpy()
    P = None
    oof = HERE / "artifacts" / "v1_stylometry" / "oof_proba.npy"
    if oof.exists():
        P = np.load(oof).astype(np.float64)
        P /= P.sum(1, keepdims=True)
    return tr, P, y


def q1(texts, prompts):
    print("=" * 94)
    print("Q1  does test-time clustering recover prompt groups as k (passages/prompt) drops?")
    print("=" * 94)
    print(f"{'k':>3} {'rows':>6} {'clust':>6} {'prec':>6} {'rec':>6} {'mean sz':>8} "
          f"{'sz=18':>6} {'sz 2-8':>7} {'sz=1':>6}")
    for k in REGIMES:
        P_, R_, S_, n18, nsm, n1, nc = [], [], [], [], [], [], []
        for d in range(N_DRAWS):
            rng = np.random.default_rng(1000 + d)
            idx = subsample(prompts, k, rng)
            cl = topic_clusters(texts[idx])
            p, r, s = pair_scores(cl, prompts[idx])
            P_.append(p); R_.append(r); S_.append(s)
            nc.append(len(np.unique(cl)))
            n18.append(int((s == 18).sum())); nsm.append(int(((s >= 2) & (s <= 8)).sum()))
            n1.append(int((s == 1).sum()))
        print(f"{k:>3} {len(idx):>6} {int(np.mean(nc)):>6} {np.mean(P_):>6.3f} {np.mean(R_):>6.3f} "
              f"{np.mean(S_):>8.2f} {int(np.mean(n18)):>6} {int(np.mean(nsm)):>7} {int(np.mean(n1)):>6}")
    print("\nprecision is the number that decides whether a 'this cluster is a permutation of")
    print("the 18 labels' rule is sound. size-18 clusters only exist at k=18.\n")


def q2(texts, prompts, y, P):
    print("=" * 94)
    print("Q2  does balanced assignment help, per regime? (v1 out-of-fold probabilities)")
    print("=" * 94)
    print(f"{'k':>3} {'argmax':>8} {'assigned':>9} {'delta':>9} {'fired':>7}")
    for k in REGIMES:
        base_s, asg_s, deltas, fire = [], [], [], []
        for d in range(N_DRAWS):
            rng = np.random.default_rng(1000 + d)
            idx = subsample(prompts, k, rng)
            L = np.log(np.clip(P[idx], 1e-12, 1))
            base = L.argmax(1)
            cl = topic_clusters(texts[idx])
            new, fired = assign_clusters(L, cl)
            b = float((base == y[idx]).mean())
            a = float((new == y[idx]).mean())
            base_s.append(b); asg_s.append(a); deltas.append(a - b); fire.append(fired.mean())
        print(f"{k:>3} {np.mean(base_s):>8.4f} {np.mean(asg_s):>9.4f} "
              f"{np.mean(deltas):>+9.4f} {np.mean(fire):>7.1%}")
    print("\nthe k=18 row is what plain CV measures. If the gain collapses as k falls, a test")
    print("set running at small k never sees the benefit - exactly what the LB reported.\n")


def main():
    tr, P, y = load()
    texts = tr.text.to_numpy(dtype=object)
    prompts = tr.prompt_id.to_numpy(dtype=str)
    print(f"train {tr.shape} ({len(set(prompts))} prompts) | OOF probs: "
          f"{'yes' if P is not None else 'MISSING'}\n")
    only = set(sys.argv[1:])
    if not only or "q1" in only:
        q1(texts, prompts)
    if P is not None and (not only or "q2" in only):
        q2(texts, prompts, y, P)
    elif P is None:
        print("! no OOF probabilities at artifacts/v1_stylometry/oof_proba.npy - Q2 skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())