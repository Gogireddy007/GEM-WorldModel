#!/usr/bin/env python
"""Does the pretrained JEPA context encoder actually add value over the raw
(standardized) branch features, with no encoder at all? Runs the raw-feature
baseline through the exact same k-fold cross-validation as
finetune_benchmark.py (same seed, same stratify_labels, same k, so fold
membership is identical), then prints all three side by side: raw baseline,
gRodon reproduction, Phydon reproduction, for a genuinely fair comparison,
not three separately-run numbers that happen to be in the same table.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gem_worldmodel.eval.benchmark import stratified_benchmark
from gem_worldmodel.training.baselines import GRodonBaseline, PhydonBaseline
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.raw_baseline import cross_validate_raw
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
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="override configs/train.yaml's seed for the CV fold splits (e.g. for a robustness check "
        "across seeds). Also changes the output CSV filename to include the seed.",
    )
    parser.add_argument(
        "--features-file", type=str, default="features_sample.csv",
        help="which processed feature table to benchmark on, e.g. features_sample_expanded.csv.",
    )
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    if args.seed is not None:
        train_cfg = {**train_cfg, "seed": args.seed}
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])

    df = pd.read_csv(processed_dir / args.features_file)
    tensors, _ = build_branch_tensors(df, model_cfg)
    target_log = torch.tensor(np.log(df["doubling_time_hours_ref"].to_numpy()), dtype=torch.float32)

    split_hours = data_cfg["doubling_time_split_hours"]
    stratify_labels = (df["doubling_time_hours_ref"].to_numpy() < split_hours).astype(int)

    raw_result = cross_validate_raw(tensors, target_log, stratify_labels, train_cfg, k=args.k)
    raw_pred = np.exp(raw_result["oof_pred_log"])
    fold_id = raw_result["fold_id"]

    grodon_pred = cross_validate_baseline(GRodonBaseline, df, fold_id)
    phydon_pred = cross_validate_baseline(PhydonBaseline, df, fold_id)

    predictions = {
        "raw_feature_baseline": raw_pred,
        "grodon_reproduction": grodon_pred,
        "phydon_reproduction": phydon_pred,
    }
    y_true = df["doubling_time_hours_ref"].to_numpy()

    table = stratified_benchmark(predictions, y_true, split_hours)
    logger.info(
        f"k={args.k}-fold cross-validation (seed={train_cfg['seed']}), n={len(df)} "
        "(raw features, no JEPA encoder)"
    )
    print(table.to_string(index=False))

    is_default = args.features_file == "features_sample.csv"
    tag = "" if is_default else f"_{Path(args.features_file).stem.removeprefix('features_sample_')}"
    out_name = (
        f"raw_baseline_results{tag}.csv" if args.seed is None else f"raw_baseline_results{tag}_seed{args.seed}.csv"
    )
    out_path = processed_dir / out_name
    table.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
