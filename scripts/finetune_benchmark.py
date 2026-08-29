#!/usr/bin/env python
"""Fine-tune the growth-rate head on the pretrained context encoder,
reproduce gRodon/Phydon baselines on the same folds, and run the stratified
(<5h / >=5h) benchmark.

Uses k-fold cross-validation rather than one fixed train/val/test split: at
175 species a single split leaves a ~27-sample test set, too small for the
benchmark numbers on it to mean much. With k-fold CV every species gets held
out exactly once, and the benchmark runs over all 175 out-of-fold
predictions instead of a 27-sample slice, real fine-tuning still happens (the
context encoder is unfrozen after the freeze schedule, same as before), it's
just repeated once per fold from a fresh copy of the pretrained checkpoint
so there's no leakage between folds.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from gem_worldmodel.eval.benchmark import stratified_benchmark
from gem_worldmodel.training.baselines import GRodonBaseline, PhydonBaseline
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.finetune import cross_validate
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def cross_validate_baseline(baseline_cls, df: pd.DataFrame, fold_id: np.ndarray) -> np.ndarray:
    """Fit/predict a baseline on the exact same folds the JEPA model used, so
    the comparison is apples-to-apples, not just "same overall dataset."
    """
    oof_pred = np.full(len(df), np.nan)
    for fold in sorted(set(fold_id)):
        test_mask = fold_id == fold
        train_df, test_df = df[~test_mask], df[test_mask]
        model = baseline_cls().fit(train_df)
        oof_pred[test_mask] = model.predict(test_df)
    return oof_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="number of cross-validation folds")
    parser.add_argument(
        "--checkpoint", type=str, default="jepa_pretrained.pt",
        help="which pretrain checkpoint to fine-tune from (e.g. jepa_pretrained_full.pt for the combined-corpus run)",
    )
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])
    ckpt_dir = resolve_path(train_cfg["pretrain"]["checkpoint_dir"])

    df = pd.read_csv(processed_dir / "features_sample.csv")
    tensors, _ = build_branch_tensors(df, model_cfg)
    target_log = torch.tensor(np.log(df["doubling_time_hours_ref"].to_numpy()), dtype=torch.float32)

    split_hours = data_cfg["doubling_time_split_hours"]
    stratify_labels = (df["doubling_time_hours_ref"].to_numpy() < split_hours).astype(int)

    cv_result = cross_validate(
        model_cfg, ckpt_dir / args.checkpoint, tensors, target_log, stratify_labels, train_cfg, k=args.k
    )
    our_pred = np.exp(cv_result["oof_pred_log"])
    fold_id = cv_result["fold_id"]

    grodon_pred = cross_validate_baseline(GRodonBaseline, df, fold_id)
    phydon_pred = cross_validate_baseline(PhydonBaseline, df, fold_id)

    predictions = {
        "gem_worldmodel": our_pred,
        "grodon_reproduction": grodon_pred,
        "phydon_reproduction": phydon_pred,
    }
    y_true = df["doubling_time_hours_ref"].to_numpy()

    table = stratified_benchmark(predictions, y_true, split_hours)
    logger.info(f"k={args.k}-fold cross-validation, n={len(df)} (every sample held out exactly once)")
    print(table.to_string(index=False))

    out_path = processed_dir / "benchmark_results.csv"
    table.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
