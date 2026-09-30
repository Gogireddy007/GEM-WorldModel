#!/usr/bin/env python
"""This tests whether regime-balanced sample weighting fixes the fast-growth
compression problem: plain squared error, fit on a corpus whose median
doubling time (6.7h) sits well above the fast regime's own median (1.7h),
pulls fast-grower predictions toward the slow-heavy middle of the
distribution.

Reweights each training sample so the fast (<5h) and slow (>=5h) regimes
contribute equal total weight to the loss, regardless of how many species
are in each, then runs the identical k-fold CV benchmark as
benchmark_gbm_raw.py so the two are directly comparable.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from gem_worldmodel.eval.benchmark import stratified_benchmark
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.raw_baseline import concat_branch_tensors
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-file", type=str, default="features_sample_gapfilled.csv")
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

    tensors, _ = build_branch_tensors(df, model_cfg, branches=all_branches)
    raw_x = concat_branch_tensors(tensors).numpy()

    n = len(df)
    oof_plain = np.full(n, np.nan)
    oof_weighted = np.full(n, np.nan)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    for train_idx, test_idx in skf.split(np.zeros(n), stratify_labels):
        train_labels = stratify_labels[train_idx]
        n_fast, n_slow = (train_labels == 1).sum(), (train_labels == 0).sum()
        sample_weight = np.where(train_labels == 1, n_fast + n_slow, 0) / (2 * n_fast) + \
                         np.where(train_labels == 0, n_fast + n_slow, 0) / (2 * n_slow)

        plain = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3)
        plain.fit(raw_x[train_idx], target_log[train_idx])
        oof_plain[test_idx] = plain.predict(raw_x[test_idx])

        weighted = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3)
        weighted.fit(raw_x[train_idx], target_log[train_idx], sample_weight=sample_weight)
        oof_weighted[test_idx] = weighted.predict(raw_x[test_idx])

    predictions = {
        "raw_gbm_plain": np.exp(oof_plain),
        "raw_gbm_regime_weighted": np.exp(oof_weighted),
    }
    table = stratified_benchmark(predictions, y_true, split_hours)
    logger.info(f"k={args.k}-fold cross-validation (seed={args.seed}), n={len(df)}, plain vs. regime-weighted")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
