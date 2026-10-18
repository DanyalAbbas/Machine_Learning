"""Verify the structural claims the top notebooks rely on, using train only (cheap, CPU).

1. Does average-linkage TF-IDF clustering at the published threshold recover prompt groups?
2. What does the within-cluster size distribution look like?
3. Is a within-cluster balanced assignment (one label per model per cluster) a valid prior?
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

DATA = Path(__file__).resolve().parents[1] / "data"


def topic_clusters(texts, thresh):
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), method="average"), thresh, criterion="distance")


def main():
    tr = pd.read_csv(f"{DATA}/train.csv", dtype=str)
    texts = tr.text.to_numpy(dtype=object)
    prompts = tr.prompt_id.to_numpy(dtype=str)
    models = tr.model.to_numpy(dtype=str)

    print(f"{'thresh':>7} {'n_clusters':>11} {'ARI vs prompt':>14} {'mean size':>10} {'min':>4} {'max':>4}")
    for th in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]:
        c = topic_clusters(texts, th)
        sizes = np.bincount(c)[1:]
        print(f"{th:>7} {len(sizes):>11} {adjusted_rand_score(prompts, c):>14.3f} "
              f"{sizes.mean():>10.2f} {sizes.min():>4} {sizes.max():>4}")

    # Does a cluster ever contain two passages from the same model?
    c = topic_clusters(texts, 0.9)
    dup = sum(len(set(models[c == g])) != (c == g).sum() for g in np.unique(c))
    print(f"\nclusters containing a repeated model at thresh 0.9: {dup}/{len(np.unique(c))}")

    # Baseline: how often does an 18-member cluster hold exactly one of each model?
    big = [g for g in np.unique(c) if (c == g).sum() == 18]
    exact = sum(len(set(models[c == g])) == 18 for g in big)
    print(f"18-member clusters: {len(big)}, of which exactly one-per-model: {exact}")

    # Cost of assuming one-per-model: upper bound on the gain from a balanced assignment
    print("\n--- oracle check: what does perfect within-cluster labelling buy? ---")
    # If we knew the set of models in each cluster but not which row, how much better
    # can we do than independent argmax? Simulated with the model's own probabilities
    # is out of scope here; instead measure label entropy within a cluster.
    ent = []
    for g in np.unique(c):
        lab = models[c == g]
        if len(lab) > 1:
            _, cnt = np.unique(lab, return_counts=True)
            p = cnt / cnt.sum()
            ent.append(-(p * np.log(p)).sum())
    print(f"mean within-cluster label entropy: {np.mean(ent):.3f} nats "
          f"(max possible {np.log(18):.3f})")


if __name__ == "__main__":
    main()