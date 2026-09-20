#!/usr/bin/env python
"""This runs the full stratified benchmark for the raw-features-plus-gradient-boosted-trees
approach found in scripts/probe_stronger_head.py, the same way every
other benchmark in this project is run, so it can sit in the same table as
gRodon, Phydon, and the JEPA-based model.

Also reports gradient-boosted trees' own feature importances, a free,
built-in check against the necessity/sufficiency findings elsewhere in this
project: does the tree model agree that 16S carries the most signal?
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from gem_worldmodel.eval.benchmark import stratified_benchmark
from gem_worldmodel.training.baselines import GRodonBaseline, PhydonBaseline
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.raw_baseline import concat_branch_tensors
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def cross_validate_baseline(baseline_cls, df: pd.DataFrame, fold_id: np.ndarray) -> np.ndarray:
    oof_pred = np.full(len(df), np.nan)
    for fold in sorted(set(fold_id)):
        test_mask = fold_id == fold
        train_df, test_df = df[~test_mask], df[test_mask]
        model = baseline_cls().fit(train_df)
        oof_pred[test_mask] = model.predict(test_df)
    return oof_pred


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
    stratify_labels = (df["doubling_time_hours_ref"].to_numpy() < split_hours).astype(int)
    y_true = df["doubling_time_hours_ref"].to_numpy()

    tensors, _ = build_branch_tensors(df, model_cfg, branches=all_branches)
    raw_x = concat_branch_tensors(tensors).numpy()

    names = sorted(tensors.keys())
    col_branch = []
    for name in names:
        col_branch += [name] * tensors[name].shape[1]

    n = len(df)
    oof_log = np.full(n, np.nan)
    fold_id = np.full(n, -1)
    importances = np.zeros(raw_x.shape[1])

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n), stratify_labels)):
        model = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3)
        model.fit(raw_x[train_idx], target_log[train_idx])
        oof_log[test_idx] = model.predict(raw_x[test_idx])
        fold_id[test_idx] = fold
        importances += model.feature_importances_ / args.k

    our_pred = np.exp(oof_log)
    grodon_pred = cross_validate_baseline(GRodonBaseline, df, fold_id)
    phydon_pred = cross_validate_baseline(PhydonBaseline, df, fold_id)

    predictions = {
        "raw_features_gbm": our_pred,
        "grodon_reproduction": grodon_pred,
        "phydon_reproduction": phydon_pred,
    }
    table = stratified_benchmark(predictions, y_true, split_hours)
    logger.info(f"k={args.k}-fold cross-validation (seed={args.seed}), n={len(df)}, raw features + gradient-boosted trees")
    print(table.to_string(index=False))

    imp_by_branch = pd.Series(importances, index=col_branch).groupby(level=0).sum().sort_values(ascending=False)
    logger.info("feature importance by branch (gradient-boosted trees' own accounting):")
    for branch, imp in imp_by_branch.items():
        logger.info(f"  {branch}: {imp:.3f}")

    out_path = processed_dir / "benchmark_results_gbm_raw.csv"
    table.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
