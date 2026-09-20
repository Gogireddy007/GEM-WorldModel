#!/usr/bin/env python
"""Phase 1 of the accuracy improvement plan, checking whether blending the new
best model (raw features, gradient-boosted trees) with gRodon and Phydon beats
any of the three alone.

Blend weights are chosen honestly, not fit on the same fold they are scored
on. For each of the five outer folds, the weights come from a small linear
fit over the other four folds' held-out predictions only, then those weights
are applied to the fold being scored. No test-fold label is ever used to
choose the weights that score that same fold.

Blending happens in log space, since doubling time is a skewed, multiplicative
quantity and everything else in this project already works in log space for
the same reason.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from gem_worldmodel.eval.benchmark import stratified_benchmark
from gem_worldmodel.training.baselines import GRodonBaseline, PhydonBaseline
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.raw_baseline import concat_branch_tensors
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def fit_blend_weights(preds_log: np.ndarray, target_log: np.ndarray) -> np.ndarray:
    """Non-negative least squares so no model gets a negative weight, then
    normalized to sum to one, an average, not an unconstrained regression
    that could overfit three correlated predictions on a handful of folds.
    """
    weights, _ = nnls(preds_log, target_log)
    if weights.sum() == 0:
        return np.full(preds_log.shape[1], 1.0 / preds_log.shape[1])
    return weights / weights.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-file", type=str, default="features_sample_expanded.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])
    split_hours = data_cfg["doubling_time_split_hours"]
    all_branches = [b["name"] for b in model_cfg["branches"]]

    df = pd.read_csv(processed_dir / args.features_file)
    target_log = np.log(df["doubling_time_hours_ref"].to_numpy())
    y_true = df["doubling_time_hours_ref"].to_numpy()
    stratify_labels = (y_true < split_hours).astype(int)
    n = len(df)

    tensors, _ = build_branch_tensors(df, model_cfg, branches=all_branches)
    raw_x = concat_branch_tensors(tensors).numpy()

    fold_id = np.full(n, -1)
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.zeros(n), stratify_labels))
    for fold, (_, test_idx) in enumerate(folds):
        fold_id[test_idx] = fold

    # Out-of-fold log predictions from each of the three models, same fold membership for all.
    gbm_log = np.full(n, np.nan)
    for fold, (train_idx, test_idx) in enumerate(folds):
        model = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3)
        model.fit(raw_x[train_idx], target_log[train_idx])
        gbm_log[test_idx] = model.predict(raw_x[test_idx])

    grodon_log = np.full(n, np.nan)
    phydon_log = np.full(n, np.nan)
    for fold in sorted(set(fold_id)):
        test_mask = fold_id == fold
        train_df, test_df = df[~test_mask], df[test_mask]
        grodon_log[test_mask] = np.log(GRodonBaseline().fit(train_df).predict(test_df))
        phydon_log[test_mask] = np.log(PhydonBaseline(seed=args.seed).fit(train_df).predict(test_df))

    # gRodon can return NaN for genomes with no computable CUB. Blend weight
    # fitting and the blended prediction both fall back to the other two
    # models for those rows rather than propagating NaN through everything.
    preds_log = np.stack([gbm_log, grodon_log, phydon_log], axis=1)
    blended_log = np.full(n, np.nan)

    for fold in sorted(set(fold_id)):
        test_mask = fold_id == fold
        other_mask = ~test_mask & ~np.isnan(preds_log).any(axis=1)
        weights = fit_blend_weights(preds_log[other_mask], target_log[other_mask])

        test_preds = preds_log[test_mask]
        for i, row in enumerate(test_preds):
            row_valid = ~np.isnan(row)
            row_weights = weights[row_valid]
            if row_weights.sum() == 0:
                row_weights = np.ones(row_valid.sum())
            row_weights = row_weights / row_weights.sum()
            blended_log[np.where(test_mask)[0][i]] = np.dot(row[row_valid], row_weights)
        logger.info(f"fold {fold}: blend weights (gbm, grodon, phydon) = {np.round(weights, 3)}")

    predictions = {
        "raw_features_gbm": np.exp(gbm_log),
        "grodon_reproduction": np.exp(grodon_log),
        "phydon_reproduction": np.exp(phydon_log),
        "blend": np.exp(blended_log),
    }
    table = stratified_benchmark(predictions, y_true, split_hours)
    logger.info(f"k={args.k}-fold cross-validation (seed={args.seed}), n={n}, blend of gbm + grodon + phydon")
    print(table.to_string(index=False))

    out_path = processed_dir / "benchmark_results_blend.csv"
    table.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
