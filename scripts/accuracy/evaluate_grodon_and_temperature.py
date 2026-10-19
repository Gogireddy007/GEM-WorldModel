#!/usr/bin/env python
"""This tests two additions to the deployed model, on the doubling time at the organism's own
temperature:

  gRodon statistics      the released gRodon's codon usage bias of the highly expressed genes, its
                         consistency, codon pair bias, and gRodon's own doubling time estimate (log),
                         from labeled_grodon_stats.csv (scripts/r/grodon_stats.R)
  predicted temperature  a two-stage model: a growth-temperature model is fitted on the training
                         fold, training rows get out-of-fold predicted temperatures (inner five-fold
                         split by species group) so the doubling-time model learns from realistically
                         noisy values, and test rows get the fold model's prediction. No measured
                         temperature of a test row is used anywhere.

Four setups (deployed, + gRodon statistics, + predicted temperature, + both) on the same rows and
folds, whole species groups, families or orders held out, ten seeds; for each setup against the
deployed one, the mean change, the seeds in which it did better and a paired Wilcoxon p.
Writes grodon_and_temperature_evaluation.csv.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = list(range(10))
SLOW = np.log(24.0)
STATS = ["CUBHE", "ConsistencyHE", "CPB", "log_grodon_d"]


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def split(groups, seed, n=5):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    return list(GroupKFold(n).split(groups, groups=np.array([perm[v] for v in groups])))


def gbr(seed):
    return GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5)


def grodon_matrix(df, processed):
    stats = pd.read_csv(processed / "labeled_grodon_stats.csv").drop_duplicates("genome_name").set_index("genome_name")
    stats["log_grodon_d"] = np.log(stats["grodon_d"])
    names = df["accession"].map(lambda a: ("RS_" if str(a).startswith("GCF_") else "GB_") + str(a))
    cols = []
    for c in STATS:
        v = names.map(stats[c]).to_numpy(float)
        cols.append(np.where(np.isnan(v), np.nanmedian(v), v))
    return np.column_stack(cols)


def predicted_temperature(X, temp, groups_species, tr, te, seed):
    """Out-of-fold predicted temperature for the training rows and fold-model prediction for the test rows."""
    out = np.full(len(temp), np.nan)
    known = tr[~np.isnan(temp[tr])]
    out[te] = gbr(seed).fit(X[known], temp[known]).predict(X[te])
    for itr, ite in split(groups_species[tr], seed + 1000):
        inner_tr, inner_te = tr[itr], tr[ite]
        inner_known = inner_tr[~np.isnan(temp[inner_tr])]
        out[inner_te] = gbr(seed).fit(X[inner_known], temp[inner_known]).predict(X[inner_te])
    return out


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, names = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True)
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    # "deployed" here is the model before gRodon's statistics were added (they are now part of the loader)
    X = np.delete(X, [names.index(c) for c in deploy_features.GRODON_FEATURES], axis=1)
    G = grodon_matrix(df, processed)
    y = np.log(df["doubling_time_hours"].to_numpy(float))
    temp = df["growth_temp_c"].to_numpy(float)
    sp = df["species_group"].to_numpy()
    print(f"{len(y)} rows; gRodon statistics for {int((~np.isnan(G[:, 0])).sum())}; measured temperature for {int((~np.isnan(temp)).sum())}")

    rows = []
    for level, prefix in (("species", None), ("family", "f__"), ("order", "o__")):
        groups = sp if prefix is None else df["gtdb_taxonomy"].map(lambda t, p=prefix: token(t, p)).to_numpy()
        for seed in SEEDS:
            preds = {name: np.full(len(y), np.nan) for name in ("deployed", "+ gRodon statistics", "+ predicted temperature", "+ both")}
            for tr, te in split(groups, seed):
                t_hat = predicted_temperature(X, temp, sp, tr, te, seed)
                t_both = predicted_temperature(np.hstack([X, G]), temp, sp, tr, te, seed)
                for name, Xs in (("deployed", X), ("+ gRodon statistics", np.hstack([X, G])),
                                 ("+ predicted temperature", np.hstack([X, t_hat[:, None]])),
                                 ("+ both", np.hstack([X, G, t_both[:, None]]))):
                    preds[name][te] = gbr(seed).fit(Xs[tr], y[tr]).predict(Xs[te])
            for name, p in preds.items():
                rows.append({"held_out": level, "seed": seed, "setup": name,
                             "typical error": float(np.exp(np.median(np.abs(y - p)))),
                             "Spearman": spearmanr(y, p).correlation, "AUC slow": roc_auc_score(y > SLOW, p)})
        print(f"{level} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(processed / "grodon_and_temperature_evaluation.csv", index=False)
    metrics = ["typical error", "Spearman", "AUC slow"]
    for level in ("species", "family", "order"):
        d = out[out["held_out"] == level]
        piv = d.pivot_table(index="seed", columns="setup", values=metrics)
        print(f"\n===== {level.upper()} HELD OUT, {len(SEEDS)} seeds, {len(y)} rows")
        print(d.groupby("setup", sort=False)[metrics].mean().round(3).to_string())
        for name in ("+ gRodon statistics", "+ predicted temperature", "+ both"):
            parts = []
            for mtr in metrics:
                diff = piv[mtr][name] - piv[mtr]["deployed"]
                better = int((diff < 0).sum()) if mtr == "typical error" else int((diff > 0).sum())
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                parts.append(f"{mtr} {diff.mean():+.3f} (better {better}/{len(SEEDS)}, p {p:.3f})")
            print(f"  {name}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
