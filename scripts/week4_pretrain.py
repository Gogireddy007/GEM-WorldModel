#!/usr/bin/env python
"""Week 4: self-supervised masked-branch pretraining on the Week 2 feature
table (real data). This runs on the labeled sample only, the plan's "full
corpus including species without a labeled growth rate" additionally
requires building features for the unlabeled GEM MAG corpus
(data/processed/unlabeled_corpus.csv), which needs a much larger genome-
download+feature-extraction pass than this session's smoke run covers; see
README "Known limitations" for how to extend this script to that corpus.
"""

import pickle

import pandas as pd

from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.pretrain import pretrain, save_checkpoint
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])

    df = pd.read_csv(processed_dir / "features_sample.csv")
    logger.info(f"pretraining on {len(df)} real genomes")

    tensors, standardizers = build_branch_tensors(df, model_cfg)
    result = pretrain(tensors, model_cfg, train_cfg)

    ckpt_dir = resolve_path(train_cfg["pretrain"]["checkpoint_dir"])
    save_checkpoint(result["model"], ckpt_dir / "jepa_pretrained.pt")
    with open(ckpt_dir / "standardizers.pkl", "wb") as f:
        pickle.dump(standardizers, f)

    logger.info(
        f"final loss={result['history']['loss'][-1]:.4f} "
        f"final target_std={result['history']['target_std'][-1]:.4f} "
        f"collapsed_at={result['collapsed_at']}"
    )


if __name__ == "__main__":
    main()
