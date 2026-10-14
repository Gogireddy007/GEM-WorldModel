#!/usr/bin/env python
"""This turns the predicted doubling time into a calibrated chance that the true
doubling time is longer than 24 hours, separately for each level of lineage
novelty.

The model squeezes very slow growth toward the middle, so a long prediction is
rare even for truly slow organisms, but a long prediction is usually right
(evaluate_slow_growth.py). Held-out predictions of the full model (whole
species, genera, families or orders held out, three seeds) are paired with
whether the measured doubling time is above 24 hours, and an isotonic
(monotone) curve maps prediction to probability. The curve is checked by
fitting on two seeds and scoring on the third.

Writes slow_probability_by_tier.csv (tier, predicted hours on a grid, probability)
and slow_probability_check.csv (per level: ranking AUC, cross-seed Brier score, and
the Brier score of always guessing the overall share). The curve is not used in
the predictions: its long end is driven by one clade of slow-growing relatives,
see research_log.md 2026-10-13.
"""

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = [42, 1, 7]
SLOW_H = 24.0
PREFIX = {"genus": "g__", "family": "f__", "order": "o__"}
TIER = {"species": "same genus", "genus": "same family, new genus", "family": "same order, new family", "order": "new order"}


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def oof(X, y, groups, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    g = np.array([perm[v] for v in groups])
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5).fit(X[tr], y[tr]).predict(X[te])
    return pred


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    df, X, _ = deploy_features.load_training_table(processed, load_config("features")["rrna16s"]["kmer_k"], with_cub=True, with_lifestyle=True)
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    y = np.log(df["doubling_time_hours"].to_numpy(float))
    slow = y > np.log(SLOW_H)
    grid_h = np.exp(np.linspace(np.log(0.1), np.log(200), 200))
    rows, checks = [], []
    print(f"{len(y)} rows, {slow.sum()} measured above {SLOW_H:.0f} h ({100 * slow.mean():.1f}%)\n")
    print(f"{'held out':9s}{'AUC':>7s}{'Brier':>8s}{'Brier, no model':>17s}{'P at 5 h':>10s}{'P at 10 h':>11s}{'P at 20 h':>11s}")
    for level, tier in TIER.items():
        groups = df["species_group"].to_numpy() if level == "species" else df["gtdb_taxonomy"].map(lambda t, p=PREFIX.get(level): token(t, p)).to_numpy()
        preds = [oof(X, y, groups, s) for s in SEEDS]
        briers, aucs = [], []
        for i in range(len(SEEDS)):
            fit_x = np.concatenate([preds[j] for j in range(len(SEEDS)) if j != i])
            fit_y = np.concatenate([slow for j in range(len(SEEDS)) if j != i])
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(fit_x, fit_y)
            prob = iso.predict(preds[i])
            briers.append(brier_score_loss(slow, prob))
            aucs.append(roc_auc_score(slow, preds[i]))
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.concatenate(preds), np.tile(slow, len(SEEDS)))
        probs = iso.predict(np.log(grid_h))
        rows += [{"tier": tier, "predicted_hours": h, "probability_slower_than_24h": p} for h, p in zip(grid_h, probs, strict=True)]
        at = {h: iso.predict(np.log([h]))[0] for h in (5, 10, 20)}
        checks.append({"held_out": level, "tier": tier, "auc": np.mean(aucs), "brier": np.mean(briers),
                       "brier_base_rate": brier_score_loss(slow, np.full(len(slow), slow.mean()))})
        print(f"{level:9s}{np.mean(aucs):7.3f}{np.mean(briers):8.3f}{brier_score_loss(slow, np.full(len(slow), slow.mean())):17.3f}"
              f"{at[5]:10.2f}{at[10]:11.2f}{at[20]:11.2f}")
    pd.DataFrame(rows).to_csv(processed / "slow_probability_by_tier.csv", index=False)
    pd.DataFrame(checks).to_csv(processed / "slow_probability_check.csv", index=False)
    print("\nwrote slow_probability_by_tier.csv")


if __name__ == "__main__":
    main()
