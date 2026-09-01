#!/usr/bin/env python
"""Necessity/sufficiency masking per branch, split by <5h vs >=5h
doubling time, cross-checked against the benchmark run. This is the actual
target result: the regime-specific growth-control answer.

Uses the same k-fold cross-validation as finetune_benchmark.py: every
species' necessity/sufficiency ablation is evaluated on a fold where it was
genuinely held out, not on data the model was fine-tuned on.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from gem_worldmodel.eval.necessity_sufficiency import necessity_sufficiency_report_cv
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.finetune import cross_validate
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


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
    fast_mask = df["doubling_time_hours_ref"].to_numpy() < split_hours
    slow_mask = ~fast_mask
    stratify_labels = fast_mask.astype(int)

    logger.info(f"regime sizes: fast(<{split_hours}h)={fast_mask.sum()} slow(>={split_hours}h)={slow_mask.sum()}")

    cv_result = cross_validate(
        model_cfg, ckpt_dir / args.checkpoint, tensors, target_log, stratify_labels, train_cfg, k=args.k
    )
    fold_models = cv_result["fold_models"]

    y_true_log = target_log.numpy()
    for regime_name, mask in [("all", np.ones_like(fast_mask)), ("fast", fast_mask), ("slow", slow_mask)]:
        report = necessity_sufficiency_report_cv(fold_models, tensors, y_true_log, mask)
        logger.info(f"=== regime={regime_name} (n={report['n']}) ===")
        logger.info(f"  full R2={report['full_metrics']['r2']:.3f}")
        for branch, res in report["necessity"].items():
            logger.info(f"  necessity[{branch}]: R2 drop when removed = {res['r2_drop']:.4f}")
        for branch, res in report["sufficiency"].items():
            logger.info(f"  sufficiency[{branch}]: R2 alone = {res['r2']:.4f}")


if __name__ == "__main__":
    main()
