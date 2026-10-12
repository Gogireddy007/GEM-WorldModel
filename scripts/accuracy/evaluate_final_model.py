#!/usr/bin/env python
"""This measures the accuracy of the final deployment model for every feature
combination a GEM genome can end up with, at every level of lineage novelty,
and writes the table the prediction script uses for its error labels and ranges.

Whole species, genera, families or orders are held out, which correspond to a
new genome whose nearest labeled relative is in the same genus, the same family,
the same order, or in none of them. For each combination of available features
(phylogeny, 16S, gene features: codon usage and amino acid composition), each target and each level the table records the
log R2, Spearman, median fold error, and the 10th and 90th percentiles of the
held-out residuals in log space, so that point prediction plus those residual
quantiles gives a range that covered about 80% of held-out species.

Targets: the doubling time at the organism's own growth temperature (raw), the
doubling time corrected to 20 C, and the growth temperature.

Writes final_accuracy_by_tier.csv.
"""

import itertools
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

SEEDS = [42, 1, 7]
PREFIX = {"genus": "g__", "family": "f__", "order": "o__"}
TIER = {"species": "same genus", "genus": "same family, new genus", "family": "same order, new family", "order": "new order"}
TARGETS = {"own temperature": "doubling_time_hours", "20 C": "doubling_time_hours_ref", "growth temperature": "growth_temp_c"}


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def columns(names, lm, s16, cub):
    keep = [i for i, n in enumerate(names) if n in deploy_features.TRAITS or n == "archaeon"]
    keep += [i for i, n in enumerate(names) if (lm and n.startswith("lm_")) or (s16 and n.startswith("s16_")) or (cub and n in ("cub", "ivywrel"))]
    return sorted(keep)


def oof(X, y, groups, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug))))
    g = np.array([perm[v] for v in groups])
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5).fit(X[tr], y[tr]).predict(X[te])
    return pred


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, names = deploy_features.load_training_table(processed, k, with_cub=True)
    rows = []
    for lm, s16, cub in itertools.product([True, False], repeat=3):
        use = df["has_cub"].to_numpy() if cub else np.ones(len(df), bool)
        d, Xs = df[use].reset_index(drop=True), X[use][:, columns(names, lm, s16, cub)]
        label = deploy_features.combo_label(lm, s16, cub)
        logger.info(f"{label}: {len(d)} training rows")
        for tname, col in TARGETS.items():
            ok = d[col].notna().to_numpy()
            dd, XX = d[ok].reset_index(drop=True), Xs[ok]
            y = dd[col].to_numpy(float)
            y = y if tname == "growth temperature" else np.log(y)
            for level in ("species", "genus", "family", "order"):
                groups = dd["species"].to_numpy() if level == "species" else dd["gtdb_taxonomy"].map(lambda t, p=PREFIX.get(level): token(t, p)).to_numpy()
                preds = [oof(XX, y, groups, s) for s in SEEDS]
                resid = np.concatenate([y - p for p in preds])
                rows.append({
                    "model": label, "target": tname, "tier": TIER[level], "held_out": level, "n_rows": len(dd),
                    "r2": np.mean([r2_score(y, p) for p in preds]),
                    "spearman": np.mean([spearmanr(y, p).correlation for p in preds]),
                    "median_error": np.mean([np.exp(np.median(np.abs(y - p))) if tname != "growth temperature" else np.mean(np.abs(y - p)) for p in preds]),
                    "resid_q10": np.quantile(resid, 0.10), "resid_q90": np.quantile(resid, 0.90),
                })
    out = pd.DataFrame(rows)
    out.to_csv(processed / "final_accuracy_by_tier.csv", index=False)
    print(out[out.target == "own temperature"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
