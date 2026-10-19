#!/usr/bin/env python
"""This compares learners and simple ensembles with the deployed model, on the deployed features and
the doubling time at the organism's own temperature.

  deployed        gradient-boosted trees, median loss, 200 trees, depth 3 (unchanged since the start)
  hist boosting   sklearn's histogram gradient boosting (LightGBM-style), median loss
  random forest   500 trees on log doubling time, min 3 rows per leaf, a third of the inputs per split
  extra trees     500 extremely randomized trees, min 3 rows per leaf, half the inputs per split
  ensembles       the plain average of the log predictions of two or three of the above

Same rows and folds for all, whole species groups, families or orders held out, five seeds; for each
against the deployed model, the mean change, the seeds in which it did better and a paired Wilcoxon p.
Writes learners_evaluation.csv.
"""

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

SLOW = np.log(24.0)


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def split(groups, seed, n=5):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    return list(GroupKFold(n).split(groups, groups=np.array([perm[v] for v in groups])))


def learners(seed):
    return {
        "deployed": GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5),
        "hist boosting": HistGradientBoostingRegressor(random_state=seed, loss="quantile", quantile=0.5, max_iter=300,
                                                       learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=1.0),
        "random forest": RandomForestRegressor(random_state=seed, n_estimators=500, min_samples_leaf=3, max_features=0.33, n_jobs=8),
        "extra trees": ExtraTreesRegressor(random_state=seed, n_estimators=500, min_samples_leaf=3, max_features=0.5, n_jobs=8),
    }


ENSEMBLES = {
    "deployed + random forest": ["deployed", "random forest"],
    "deployed + extra trees": ["deployed", "extra trees"],
    "deployed + hist + random forest": ["deployed", "hist boosting", "random forest"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, _ = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True)
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    y = np.log(df["doubling_time_hours"].to_numpy(float))
    print(f"{len(y)} rows, {X.shape[1]} inputs")

    rows = []
    for level, prefix in (("species", None), ("family", "f__"), ("order", "o__")):
        groups = df["species_group"].to_numpy() if prefix is None else df["gtdb_taxonomy"].map(lambda t, p=prefix: token(t, p)).to_numpy()
        for seed in seeds:
            preds = {name: np.full(len(y), np.nan) for name in learners(seed)}
            for tr, te in split(groups, seed):
                for name, model in learners(seed).items():
                    preds[name][te] = model.fit(X[tr], y[tr]).predict(X[te])
            for name, members in ENSEMBLES.items():
                preds[name] = np.mean([preds[m] for m in members], axis=0)
            for name, p in preds.items():
                rows.append({"held_out": level, "seed": seed, "setup": name,
                             "typical error": float(np.exp(np.median(np.abs(y - p)))),
                             "Spearman": spearmanr(y, p).correlation, "AUC slow": roc_auc_score(y > SLOW, p)})
        print(f"{level} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(processed / "learners_evaluation.csv", index=False)
    metrics = ["typical error", "Spearman", "AUC slow"]
    for level in ("species", "family", "order"):
        d = out[out["held_out"] == level]
        piv = d.pivot_table(index="seed", columns="setup", values=metrics)
        print(f"\n===== {level.upper()} HELD OUT, {len(seeds)} seeds")
        print(d.groupby("setup", sort=False)[metrics].mean().round(3).to_string())
        for name in d["setup"].unique():
            if name == "deployed":
                continue
            parts = []
            for mtr in metrics:
                diff = piv[mtr][name] - piv[mtr]["deployed"]
                better = int((diff < 0).sum()) if mtr == "typical error" else int((diff > 0).sum())
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                parts.append(f"{mtr} {diff.mean():+.3f} ({better}/{len(seeds)}, p {p:.2f})")
            print(f"  {name:34s} " + "; ".join(parts))


if __name__ == "__main__":
    main()
