#!/usr/bin/env python
"""Week 6: linear/nonlinear probing of the joint latent for oligotroph vs.
copiotroph status (heuristic proxy label, see eval/probing.py caveat),
followed by activation intervention on the most predictive dimension.
"""

import pickle

import numpy as np
import pandas as pd
import torch

from gem_worldmodel.eval.intervention import intervene_on_dimension
from gem_worldmodel.eval.probing import (
    heuristic_trophic_label,
    linear_probe,
    most_predictive_latent_dim,
    nonlinear_probe,
)
from gem_worldmodel.models.heads import GrowthRateHead
from gem_worldmodel.training.dataset import build_branch_tensors
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
    with torch.no_grad():
        latents = jepa.joint_representation(tensors).numpy()

    labels = heuristic_trophic_label(
        df["genome_size_bp"].to_numpy(), df["rrna_16s_count"].fillna(0).to_numpy()
    )
    logger.info(f"trophic label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    if len(set(labels)) < 2 or len(df) < 10:
        logger.warning(
            f"only {len(df)} samples / {len(set(labels))} label classes, probing results below are illustrative "
            "only, not statistically meaningful at this sample size. Scale up scripts/week2_build_features.py "
            "for a real result."
        )

    lin = linear_probe(latents, labels)
    nonlin = nonlinear_probe(latents, labels)
    logger.info(f"linear probe: accuracy={lin['accuracy']:.3f} auc={lin.get('auc', float('nan')):.3f}")
    logger.info(f"nonlinear probe: accuracy={nonlin['accuracy']:.3f} auc={nonlin.get('auc', float('nan')):.3f}")

    best_dim = most_predictive_latent_dim(latents, labels)
    logger.info(f"most predictive latent dimension: {best_dim}")

    head = GrowthRateHead(jepa.latent_dim, load_config("model")["growth_rate_head"]["hidden_dim"], 1)
    result = intervene_on_dimension(torch.tensor(latents, dtype=torch.float32), head, best_dim)
    logger.info(f"intervention on dim {best_dim}: shift_by_delta={result['shift_by_delta']}")
    logger.info(
        "reminder: oligotroph status is partly defined by absence of CUB signal, so treat this probing "
        "result as necessary but not sufficient (see eval/probing.py for why)"
    )


if __name__ == "__main__":
    main()
