#!/usr/bin/env python
"""Phase 0 check: is the growth-rate head itself leaving accuracy on the
table, separate from the encoder? This takes the same frozen, pretrained encoder
representation used everywhere else in this project and swaps only the
regression head, gradient-boosted trees instead of the small fine-tuned MLP,
using the identical k-fold split every other benchmark in this project uses,
so the comparison is fair.

Runs two versions: the full 3-branch representation, and the genomic_traits
branch alone, to check directly whether a stronger head recovers any of the
gap found between gRodon's plain linear regression on CUB (R2 -0.032) and
this project's own encoder+head using the whole genomic_traits branch
(R2 -0.467, see FINDINGS.md's necessity/sufficiency table).
"""

import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.pretrain import load_checkpoint
from gem_worldmodel.training.raw_baseline import concat_branch_tensors
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def cross_validated_gbm(latents: np.ndarray, target_log: np.ndarray, stratify_labels: np.ndarray, seed: int, k: int = 5) -> np.ndarray:
    oof = np.full(len(target_log), np.nan)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(target_log)), stratify_labels)):
        model = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3)
        model.fit(latents[train_idx], target_log[train_idx])
        oof[test_idx] = model.predict(latents[test_idx])
    return oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="jepa_pretrained_full_expanded.pt")
    parser.add_argument("--features-file", type=str, default="features_sample_expanded.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])
    ckpt_dir = resolve_path(train_cfg["pretrain"]["checkpoint_dir"])
    split_hours = data_cfg["doubling_time_split_hours"]
    all_branches = [b["name"] for b in model_cfg["branches"]]

    df = pd.read_csv(processed_dir / args.features_file)
    target_log = np.log(df["doubling_time_hours_ref"].to_numpy())
    stratify_labels = (df["doubling_time_hours_ref"].to_numpy() < split_hours).astype(int)
    y_true = df["doubling_time_hours_ref"].to_numpy()

    jepa = load_checkpoint(model_cfg, ckpt_dir / args.checkpoint)
    jepa.eval()

    for branches, label in [(all_branches, "full 3-branch"), (["genomic_traits"], "genomic_traits alone")]:
        tensors, _ = build_branch_tensors(df, model_cfg, branches=branches)
        with torch.no_grad():
            latents = jepa.joint_representation(tensors).numpy()

        oof_log = cross_validated_gbm(latents, target_log, stratify_labels, args.seed, args.k)
        pred_hours = np.exp(oof_log)
        r2 = r2_score(y_true, pred_hours)
        rho, _ = spearmanr(y_true, pred_hours)
        logger.info(f"[{label}, frozen encoder + gradient-boosted head] R2={r2:.3f} Spearman={rho:.3f}")

    # Control: does gradient-boosted trees alone, no encoder at all, do just
    # as well? If so, the win above is about the head, not the encoder.
    tensors, _ = build_branch_tensors(df, model_cfg, branches=all_branches)
    raw_x = concat_branch_tensors(tensors).numpy()
    oof_log_raw = cross_validated_gbm(raw_x, target_log, stratify_labels, args.seed, args.k)
    pred_hours_raw = np.exp(oof_log_raw)
    r2_raw = r2_score(y_true, pred_hours_raw)
    rho_raw, _ = spearmanr(y_true, pred_hours_raw)
    logger.info(f"[raw features, no encoder, gradient-boosted head] R2={r2_raw:.3f} Spearman={rho_raw:.3f}")


if __name__ == "__main__":
    main()
