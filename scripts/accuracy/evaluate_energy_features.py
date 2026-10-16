#!/usr/bin/env python
"""This tests whether the energy-metabolism features (lifestyle_genes.energy_features: electron
acceptors, ATP synthase type, sodium energetics, fermentation) improve the model, on top of the
current deployed features.

Same rows for both setups (species groups with gene features), doubling time at the organism's own
temperature, median-loss boosted trees, whole species groups, families or orders held out, five
seeds. Reports typical error, Spearman, and the area under the curve for picking out organisms
slower than a day by ranking on the prediction and with a separate classifier, with the number of
seeds in which the energy features did better. Writes energy_features_evaluation.csv.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features, lifestyle_genes
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = [42, 1, 7, 13, 2024]
SLOW = np.log(24.0)


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def folds(groups, seed, n=5):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    return list(GroupKFold(n).split(groups, groups=np.array([perm[v] for v in groups])))


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, names = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True, with_energy=True)
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    energy = [names.index(f) for f in lifestyle_genes.ENERGY_FEATURES]
    setups = {"current": np.delete(X, energy, axis=1), "+ energy features": X}
    y = np.log(df["doubling_time_hours"].to_numpy(float))
    slow = y > SLOW
    print(f"{len(y)} rows, {slow.sum()} slower than a day; current {setups['current'].shape[1]} inputs, with energy {X.shape[1]}")
    rows = []
    for level, prefix in (("species", None), ("family", "f__"), ("order", "o__")):
        groups = df["species_group"].to_numpy() if prefix is None else df["gtdb_taxonomy"].map(lambda t, p=prefix: token(t, p)).to_numpy()
        for seed in SEEDS:
            split = folds(groups, seed)
            for name, Xs in setups.items():
                pred, prob = np.full(len(y), np.nan), np.full(len(y), np.nan)
                for tr, te in split:
                    reg = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5)
                    pred[te] = reg.fit(Xs[tr], y[tr]).predict(Xs[te])
                    clf = GradientBoostingClassifier(random_state=seed, n_estimators=200, max_depth=3)
                    prob[te] = clf.fit(Xs[tr], slow[tr]).predict_proba(Xs[te])[:, 1]
                rows.append({"held_out": level, "seed": seed, "setup": name,
                             "typical error": np.exp(np.median(np.abs(y - pred))), "Spearman": spearmanr(y, pred).correlation,
                             "AUC, ranking": roc_auc_score(slow, pred), "AUC, classifier": roc_auc_score(slow, prob)})
    out = pd.DataFrame(rows)
    out.to_csv(processed / "energy_features_evaluation.csv", index=False)
    metrics = ["typical error", "Spearman", "AUC, ranking", "AUC, classifier"]
    for level in ("species", "family", "order"):
        d = out[out.held_out == level]
        mean = d.groupby("setup")[metrics].mean()
        piv = d.pivot_table(index="seed", columns="setup", values=metrics)
        better = {m: int(((piv[m]["+ energy features"] < piv[m]["current"]) if m == "typical error" else (piv[m]["+ energy features"] > piv[m]["current"])).sum()) for m in metrics}
        print(f"\n===== {level.upper()} HELD OUT, mean of {len(SEEDS)} seeds")
        print(mean.round(3).to_string())
        print("seeds better with energy features: " + ", ".join(f"{m} {v}/{len(SEEDS)}" for m, v in better.items()))


if __name__ == "__main__":
    main()
