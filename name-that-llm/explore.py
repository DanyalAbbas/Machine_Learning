import re

import numpy as np
import pandas as pd

pd.set_option('display.width', 250)

CH = ["—", "–", ";", ":", ",", ".", '"', "'", "’", "“", "”", "!", "?", "*", "\n", "#", "(", ")", "-", "%", "+", "/"]


def rates(t):
    n = max(len(t), 1)
    d = {f"cnt_{repr(c)}": t.count(c) / n * 1000 for c in CH}
    w = t.split()
    d["dbl_nl"] = t.count("\n\n")
    d["avg_wlen"] = np.mean([len(x) for x in w]) if w else 0
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s]
    d["avg_slen"] = np.mean([len(s) for s in sents]) if sents else 0
    d["n_sent"] = len(sents)
    d["upper_frac"] = sum(c.isupper() for c in t) / n * 100
    d["digit_frac"] = sum(c.isdigit() for c in t) / n * 100
    d["nonascii_frac"] = sum(ord(c) > 127 for c in t) / n * 100
    toks = [x.lower().strip(".,;:!?()\"'") for x in w]
    d["ttr"] = len(set(toks)) / max(len(w), 1)
    return d


def main():
    tr = pd.read_csv("data/train.csv", dtype=str)
    tr["nchar"] = tr.text.str.len()
    tr["nword"] = tr.text.str.split().str.len()
    tr["nsent"] = tr.text.apply(lambda t: len([s for s in re.split(r"(?<=[.!?])\s+", t) if s]))

    print("=== word-count stats by model ===")
    print(tr.groupby("model")["nword"].agg(["mean", "std", "min", "max"]).sort_values("mean").round(1).to_string())
    print("\nnword overall:", tr.nword.describe().round(1).to_dict())
    print("\n=== is length correlated with model? (mean nword per prompt, std across prompts) ===")
    pm = tr.groupby(["prompt_id", "model"]).nword.mean().unstack()
    print(pm.std(axis=1).describe().round(2).to_dict(), " <- within-prompt across-model spread")

    F = tr.text.apply(rates).apply(pd.Series)
    R = pd.concat([tr[["model"]], F], axis=1)
    agg = R.groupby("model").mean().T
    print("\n=== mean rate per model (per 1000 chars unless noted) ===")
    print(agg.round(2).to_string())

    X = F.values.astype(np.float64)
    print("\nfeature matrix", X.shape)

    print("\n=== how separable are pure style features? single-feature best balanced acc ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = tr.model.values
    groups = tr.prompt_id.values
    X = R.drop(columns=["model"]).values
    oof = np.empty(len(y), dtype=object)
    for f, v in GroupKFold(5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[f], y[f])
        oof[v] = m.predict(X[v])
    print("style-only OOF balanced acc:", round(balanced_accuracy_score(y, oof), 4))


if __name__ == "__main__":
    main()