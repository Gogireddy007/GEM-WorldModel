#!/usr/bin/env python
"""This tests the final model on species it has never seen.

The model is trained on the original corpus only and predicts the Madin
extension species (bacterial and archaeal), with the final feature set:
genome traits, phylogeny, 16S and codon usage. Reported per target, for
bacteria and archaea separately, and by how close the nearest relative in the
training corpus is. Archaea are in the test set although the training corpus has
none, so their rows show what happens without archaeal training and the bacterial
rows are the fair test. Writes independent_test.csv for the report charts.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = [42, 1, 7, 13, 2024]
TARGETS = {"own temperature": "doubling_time_hours", "20 C": "doubling_time_hours_ref"}


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else None


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, _ = deploy_features.load_training_table(processed, k, with_cub=True)
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    train = (df["origin"] == "original").to_numpy()
    test = ~train
    seen = {lvl: set(df.loc[train, "gtdb_taxonomy"].map(lambda t, p=p: token(t, p)).dropna()) for lvl, p in (("genus", "g__"), ("family", "f__"), ("order", "o__"))}

    def tier(t):
        if token(t, "g__") in seen["genus"]:
            return "same genus"
        if token(t, "f__") in seen["family"]:
            return "same family, new genus"
        if token(t, "o__") in seen["order"]:
            return "same order, new family"
        return "new order"

    df["tier"] = df["gtdb_taxonomy"].map(tier)
    out = []
    for tname, col in TARGETS.items():
        y = np.log(df[col].to_numpy(float))
        pred = np.mean([GradientBoostingRegressor(random_state=s, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5).fit(X[train], y[train]).predict(X[test]) for s in SEEDS], axis=0)
        d = df[test].assign(truth=y[test], pred=pred, target=tname)
        out.append(d[["species", "target", "tier", "archaeon", "truth", "pred"]])
        print(f"\n{tname}: trained on {train.sum()} original rows, tested on {test.sum()} unseen species")
        print(f"{'':32s}{'n':>5s}{'R2 log':>9s}{'Spearman':>10s}{'typical error':>15s}")
        groups = [("all unseen species", np.ones(test.sum(), bool)), ("bacteria", d.archaeon.to_numpy() == 0), ("archaea", d.archaeon.to_numpy() == 1)] + \
                 [(f"relative: {t}", (d.tier == t).to_numpy()) for t in ("same genus", "same family, new genus", "same order, new family", "new order")]
        for name, m in groups:
            if m.sum() >= 5:
                a, b = d.truth.to_numpy()[m], d.pred.to_numpy()[m]
                print(f"{name:32s}{m.sum():5d}{r2_score(a, b):9.3f}{spearmanr(a, b).correlation:10.3f}{np.exp(np.median(np.abs(a - b))):14.2f}x")
    pd.concat(out).to_csv(processed / "independent_test.csv", index=False)


if __name__ == "__main__":
    main()
