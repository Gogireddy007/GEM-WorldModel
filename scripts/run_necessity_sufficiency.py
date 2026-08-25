#!/usr/bin/env python
"""Necessity/sufficiency masking per branch, split by <5h vs >=5h
doubling time, cross-checked against the benchmark run, this is the actual
target result: the regime-specific growth-control answer.
"""

import pickle

import numpy as np
import pandas as pd
import torch

from gem_worldmodel.eval.necessity_sufficiency import necessity_sufficiency_report
from gem_worldmodel.models.heads import GrowthRateHead
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.finetune import finetune
from gem_worldmodel.training.pretrain import load_checkpoint
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])
    ckpt_dir = resolve_path(train_cfg["pretrain"]["checkpoint_dir"])

    df = pd.read_csv(processed_dir / "features_sample.csv")
    with open(ckpt_dir / "standardizers.pkl", "rb") as f:
        standardizers = pickle.load(f)
    tensors, _ = build_branch_tensors(df, model_cfg, standardizers)

    jepa = load_checkpoint(model_cfg, ckpt_dir / "jepa_pretrained.pt")
    target_log = torch.tensor(np.log(df["doubling_time_hours_ref"].to_numpy()), dtype=torch.float32)
    ft_result = finetune(jepa, tensors, target_log, train_cfg)
    head: GrowthRateHead = ft_result["head"]
    jepa = ft_result["jepa"]

    split_hours = data_cfg["doubling_time_split_hours"]
    fast_mask = df["doubling_time_hours_ref"].to_numpy() < split_hours
    slow_mask = ~fast_mask

    logger.info(f"regime sizes: fast(<{split_hours}h)={fast_mask.sum()} slow(>={split_hours}h)={slow_mask.sum()}")
    if fast_mask.sum() < 5 or slow_mask.sum() < 5:
        logger.warning(
            "fewer than 5 samples in one regime at this sample size, necessity/sufficiency numbers below "
            "are illustrative only. Scale up scripts/build_features.py for a real result."
        )

    y_true_log = target_log.numpy()
    for regime_name, mask in [("all", np.ones_like(fast_mask, dtype=bool)), ("fast", fast_mask), ("slow", slow_mask)]:
        if mask.sum() < 2:
            continue
        report = necessity_sufficiency_report(jepa, head, tensors, y_true_log, mask)
        logger.info(f"=== regime={regime_name} (n={report['n']}) ===")
        logger.info(f"  full R2={report['full_metrics']['r2']:.3f}")
        for branch, res in report["necessity"].items():
            logger.info(f"  necessity[{branch}]: R2 drop when removed = {res['r2_drop']:.4f}")
        for branch, res in report["sufficiency"].items():
            logger.info(f"  sufficiency[{branch}]: R2 alone = {res['r2']:.4f}")


if __name__ == "__main__":
    main()
