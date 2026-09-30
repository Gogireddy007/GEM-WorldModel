#!/usr/bin/env python
"""This tests quantile (median, alpha=0.5) loss against Huber loss (the
current default, see research_log.md 2026-09-30) and plain squared error,
as a third attempt at the fast-growth compression problem.

Quantile loss at the median is an L1-type loss, it should be far less
pulled by the slow regime's large-magnitude values than squared error is,
at the cost of not optimizing specifically for minimizing squared error
where that dynamic range is actually wide and informative.
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
    oof_squared = np.full(n, np.nan)
    oof_huber = np.full(n, np.nan)
    oof_quantile = np.full(n, np.nan)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    for train_idx, test_idx in skf.split(np.zeros(n), stratify_labels):
        squared = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3)
        squared.fit(raw_x[train_idx], target_log[train_idx])
        oof_squared[test_idx] = squared.predict(raw_x[test_idx])

        huber = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3, loss="huber", alpha=0.9)
        huber.fit(raw_x[train_idx], target_log[train_idx])
        oof_huber[test_idx] = huber.predict(raw_x[test_idx])

        quantile = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5)
        quantile.fit(raw_x[train_idx], target_log[train_idx])
        oof_quantile[test_idx] = quantile.predict(raw_x[test_idx])

    predictions = {
        "raw_gbm_squared_error": np.exp(oof_squared),
        "raw_gbm_huber": np.exp(oof_huber),
        "raw_gbm_quantile_median": np.exp(oof_quantile),
    }
    table = stratified_benchmark(predictions, y_true, split_hours)
    logger.info(f"k={args.k}-fold cross-validation (seed={args.seed}), n={len(df)}, squared error vs. Huber vs. quantile-median")
    print(table.to_string(index=False))

    out_path = processed_dir / "benchmark_results_gbm_quantile.csv"
    table.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
