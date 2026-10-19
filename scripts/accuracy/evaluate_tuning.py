#!/usr/bin/env python
"""This tunes the two learners that did best in evaluate_learners.py (histogram gradient boosting
with median loss, random forest on log doubling time) with nested cross-validation, and compares
them and their average with the deployed model.

Nested: inside every outer training fold, a random sample of settings is scored by an inner
three-fold split grouped by GTDB family (most GEM genomes have no labeled relative in their family),
the best setting by inner typical error is refit on the whole outer training fold and predicts the
outer test fold. The outer test rows never take part in choosing. Outer folds hold out whole species
groups, families or orders; same rows and folds for every setup; for each against the deployed
model, the mean change, the seeds in which it did better and a paired Wilcoxon p. The settings
chosen in each outer fold are written too, to see whether the choice is stable.

Writes tuning_evaluation.csv and tuning_choices.csv.
"""

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, ParameterSampler

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

SLOW = np.log(24.0)
N_CANDIDATES = 10
HIST_GRID = {"learning_rate": [0.03, 0.05, 0.1], "max_iter": [200, 400, 800], "max_leaf_nodes": [7, 15, 31],
             "min_samples_leaf": [5, 10, 20], "l2_regularization": [0.0, 1.0, 5.0], "max_features": [0.5, 1.0]}
RF_GRID = {"min_samples_leaf": [1, 3, 5, 10], "max_features": [0.2, 0.33, 0.5], "max_samples": [0.7, None]}


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def split(groups, seed, n=5):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    return list(GroupKFold(n).split(groups, groups=np.array([perm[v] for v in groups])))


def make(kind, params, seed):
    if kind == "hist":
        return HistGradientBoostingRegressor(random_state=seed, loss="quantile", quantile=0.5, **params)
    return RandomForestRegressor(random_state=seed, n_estimators=300, n_jobs=8, **params)


def tune(kind, X, y, family, seed):
    """Best setting by typical error over an inner three-fold split grouped by family."""
    grid = HIST_GRID if kind == "hist" else RF_GRID
    candidates = list(ParameterSampler(grid, N_CANDIDATES, random_state=seed))
    inner = split(family, seed + 500, n=3)
    best, best_err = None, np.inf
    for params in candidates:
        pred = np.full(len(y), np.nan)
        for itr, ite in inner:
            pred[ite] = make(kind, params, seed).fit(X[itr], y[itr]).predict(X[ite])
        err = np.median(np.abs(y - pred))
        if err < best_err:
            best, best_err = params, err
    return best


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
    family = df["gtdb_taxonomy"].map(lambda t: token(t, "f__")).to_numpy()
    print(f"{len(y)} rows, {X.shape[1]} inputs, {N_CANDIDATES} settings per learner per outer fold", flush=True)

    rows, choices = [], []
    for level, prefix in (("species", None), ("family", "f__"), ("order", "o__")):
        groups = df["species_group"].to_numpy() if prefix is None else df["gtdb_taxonomy"].map(lambda t, p=prefix: token(t, p)).to_numpy()
        for seed in seeds:
            preds = {n: np.full(len(y), np.nan) for n in ("deployed", "hist tuned", "forest tuned")}
            for f, (tr, te) in enumerate(split(groups, seed)):
                deployed = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5)
                preds["deployed"][te] = deployed.fit(X[tr], y[tr]).predict(X[te])
                for kind, name in (("hist", "hist tuned"), ("forest", "forest tuned")):
                    params = tune(kind, X[tr], y[tr], family[tr], seed)
                    preds[name][te] = make(kind, params, seed).fit(X[tr], y[tr]).predict(X[te])
                    choices.append({"held_out": level, "seed": seed, "fold": f, "learner": kind, **{k_: str(v) for k_, v in params.items()}})
            preds["hist + forest tuned"] = (preds["hist tuned"] + preds["forest tuned"]) / 2
            for name, p in preds.items():
                rows.append({"held_out": level, "seed": seed, "setup": name,
                             "typical error": float(np.exp(np.median(np.abs(y - p)))),
                             "Spearman": spearmanr(y, p).correlation, "AUC slow": roc_auc_score(y > SLOW, p)})
            print(f"{level} seed {seed} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(processed / "tuning_evaluation.csv", index=False)
    pd.DataFrame(choices).to_csv(processed / "tuning_choices.csv", index=False)
    metrics = ["typical error", "Spearman", "AUC slow"]
    for level in ("species", "family", "order"):
        d = out[out["held_out"] == level]
        piv = d.pivot_table(index="seed", columns="setup", values=metrics)
        print(f"\n===== {level.upper()} HELD OUT, {len(seeds)} seeds")
        print(d.groupby("setup", sort=False)[metrics].mean().round(3).to_string())
        for name in ("hist tuned", "forest tuned", "hist + forest tuned"):
            parts = []
            for mtr in metrics:
                diff = piv[mtr][name] - piv[mtr]["deployed"]
                better = int((diff < 0).sum()) if mtr == "typical error" else int((diff > 0).sum())
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                parts.append(f"{mtr} {diff.mean():+.3f} ({better}/{len(seeds)}, p {p:.2f})")
            print(f"  {name:22s} " + "; ".join(parts))
    ch = pd.DataFrame(choices)
    for kind in ("hist", "forest"):
        c = ch[ch["learner"] == kind].drop(columns=["held_out", "seed", "fold", "learner"]).dropna(axis=1, how="all")
        print(f"\nmost common {kind} settings:", {col: c[col].value_counts().index[0] for col in c.columns})


if __name__ == "__main__":
    main()
