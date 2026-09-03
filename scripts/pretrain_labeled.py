#!/usr/bin/env python
"""Self-supervised masked-branch pretraining on the feature table built by
build_features.py (real data). This runs on the labeled sample only. Pretraining
on the "full corpus including species without a labeled growth rate" needs
features built for the unlabeled GEM MAG corpus too
(data/processed/unlabeled_corpus.csv), see pretrain_full.py for that.
"""

import argparse
import pickle

import pandas as pd

from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.pretrain import pretrain, save_checkpoint
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=None,
        help="override configs/train.yaml's seed, e.g. for a robustness check across seeds. "
        "Also changes the output checkpoint filename to jepa_pretrained_seed{N}.pt so it "
        "doesn't overwrite the default-seed checkpoint.",
    )
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    if args.seed is not None:
        train_cfg = {**train_cfg, "seed": args.seed}
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])

    df = pd.read_csv(processed_dir / "features_sample.csv")
    logger.info(f"pretraining on {len(df)} real genomes (seed={train_cfg['seed']})")

    tensors, standardizers = build_branch_tensors(df, model_cfg)
    result = pretrain(tensors, model_cfg, train_cfg)

    ckpt_dir = resolve_path(train_cfg["pretrain"]["checkpoint_dir"])
    ckpt_name = "jepa_pretrained.pt" if args.seed is None else f"jepa_pretrained_seed{args.seed}.pt"
    save_checkpoint(result["model"], ckpt_dir / ckpt_name)
    if args.seed is None:
        with open(ckpt_dir / "standardizers.pkl", "wb") as f:
            pickle.dump(standardizers, f)

    logger.info(
        f"final loss={result['history']['loss'][-1]:.4f} "
        f"final target_std={result['history']['target_std'][-1]:.4f} "
        f"collapsed_at={result['collapsed_at']}"
    )


if __name__ == "__main__":
    main()
