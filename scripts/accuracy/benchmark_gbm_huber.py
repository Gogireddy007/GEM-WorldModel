#!/usr/bin/env python
"""This tests Huber loss in place of plain squared error for the raw-features
plus gradient-boosted-trees model, as a second attempt at the fast-growth
compression problem after regime-balanced sample weighting failed to fix it
(see research_log.md, 2026-09-30).

Huber loss is quadratic near zero error and linear beyond a threshold, so it
should be less sensitive to the narrow-range instability that made
regime-weighting backfire: the fast regime's true values sit in a narrow
absolute range, so a model pushed to fit them tightly under squared error
can overshoot on a few species and blow up R2, Huber caps how much a single
large residual can dominate the fit.
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
    parser.add_argument("--alpha", type=float, default=0.9, help="Huber loss quantile threshold")
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
    oof_huber = np.full(n, np.nan)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    for train_idx, test_idx in skf.split(np.zeros(n), stratify_labels):
        plain = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3)
        plain.fit(raw_x[train_idx], target_log[train_idx])
        oof_plain[test_idx] = plain.predict(raw_x[test_idx])

        huber = GradientBoostingRegressor(
            random_state=args.seed, n_estimators=200, max_depth=3, loss="huber", alpha=args.alpha
        )
        huber.fit(raw_x[train_idx], target_log[train_idx])
        oof_huber[test_idx] = huber.predict(raw_x[test_idx])

    predictions = {
        "raw_gbm_squared_error": np.exp(oof_plain),
        "raw_gbm_huber": np.exp(oof_huber),
    }
    table = stratified_benchmark(predictions, y_true, split_hours)
    logger.info(f"k={args.k}-fold cross-validation (seed={args.seed}), n={len(df)}, squared error vs. Huber loss")
    print(table.to_string(index=False))

    out_path = processed_dir / "benchmark_results_gbm_huber.csv"
    table.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
