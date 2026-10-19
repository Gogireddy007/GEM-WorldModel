#!/usr/bin/env python
"""This looks at the organisms the deployed model wrongly calls slow.

Held-out predictions of the deployed model (own-temperature doubling time, the features of
predict_gem_genomes_gbm.py: traits, phylogeny, 16S, gene features with lifestyle genes), with whole
species groups or whole orders held out, averaged over five seeds. Two kinds of error are listed:

  false slow calls       predicted slower than 24 hours, measured faster
  large overpredictions  predicted 10 hours or more and at least 5 times the measured value

For each, the median measured doubling time of its labeled relatives (same genus, else family, the
species itself left out) tells whether the prediction was borrowed from slow relatives, and the
precision of slow calls (share of predictions above 24 hours that are truly above 24 hours) is
reported. Writes false_slow_calls.csv.
"""

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.models.deployed import DeployedRegressor
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = [42, 1, 7, 13, 2024]
SLOW = 24.0


def rank(t, p):
    m = re.search(rf"{p}([^;]+)", str(t))
    return m.group(1) if m else None


def classify(pred, meas):
    """Boolean arrays: called slow, truly slow, false slow call, large overprediction."""
    pred, meas = np.asarray(pred, float), np.asarray(meas, float)
    called, slow = pred > SLOW, meas > SLOW
    return called, slow, called & ~slow, (pred >= 10) & (pred / meas >= 5)


def oof(X, y, groups, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, np.array([perm[v] for v in groups])):
        model = DeployedRegressor("own temperature", seed)
        pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
    return pred


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, names = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True)
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    y = np.log(df["doubling_time_hours"].to_numpy(float))
    df["genus"] = df["gtdb_taxonomy"].map(lambda t: rank(t, "g__"))
    df["family"] = df["gtdb_taxonomy"].map(lambda t: rank(t, "f__"))
    df["order"] = df["gtdb_taxonomy"].map(lambda t: rank(t, "o__"))
    relatives = []
    for _, r in df.iterrows():
        for level in ("genus", "family"):
            rel = df[(df[level] == r[level]) & (df["species_group"] != r["species_group"])] if r[level] else df.iloc[0:0]
            if len(rel):
                relatives.append((float(np.exp(np.log(rel["doubling_time_hours"]).median())), level, len(rel)))
                break
        else:
            relatives.append((np.nan, "none", 0))
    df["relatives_hours"], df["relatives_level"], df["relatives_n"] = zip(*relatives, strict=True)

    out = []
    for level in ("species", "order"):
        groups = df["species_group"].to_numpy() if level == "species" else df["order"].fillna("none").to_numpy()
        pred = np.exp(np.mean([oof(X, y, groups, s) for s in SEEDS], axis=0))
        meas = df["doubling_time_hours"].to_numpy(float)
        called, slow, false_slow, over = classify(pred, meas)
        print(f"\n===== {level.upper()} HELD OUT ({len(df)} rows, {int(slow.sum())} measured slower than a day)")
        print(f"predicted slower than a day: {int(called.sum())}, of which truly slow {int((called & slow).sum())} "
              f"(precision {100 * (called & slow).sum() / max(1, called.sum()):.0f}%); false slow calls {int(false_slow.sum())}; "
              f"large overpredictions {int(over.sum())}")
        sel = df[false_slow | over].assign(measured=meas[false_slow | over], predicted=pred[false_slow | over],
                                           kind=np.where(false_slow[false_slow | over], "false slow", "overprediction"))
        for r in sel.sort_values("predicted", ascending=False).itertuples():
            print(f"  {r.kind:15s} {r.species:36s} measured {r.measured:6.2f} h  predicted {r.predicted:6.1f} h  "
                  f"{r.order or '':22s} relatives {r.relatives_hours:6.1f} h ({r.relatives_level}, n {r.relatives_n})")
        if len(sel):
            slow_rel = (sel["relatives_hours"] >= 10).mean()
            print(f"share of these with relatives at 10 h or slower: {100 * slow_rel:.0f}%; "
                  f"orders: {sel['order'].value_counts().head(6).to_dict()}")
        out.append(sel.assign(held_out=level)[["held_out", "kind", "species", "order", "measured", "predicted",
                                               "relatives_hours", "relatives_level", "relatives_n", "growth_temp_c", "origin"]])
    pd.concat(out).to_csv(processed / "false_slow_calls.csv", index=False)


if __name__ == "__main__":
    main()
