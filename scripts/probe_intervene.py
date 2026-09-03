#!/usr/bin/env python
"""Linear/nonlinear probing of the joint latent for oligotroph vs.
copiotroph status, followed by activation intervention on the most
predictive dimension.

Uses the real, genome-independent trophic label built from Madin et al.
2020's isolation_source data (features/ecological_traits.py) by default,
falling back to the old genome-derived heuristic (features + rRNA-count
based, see eval/probing.py) only with --use-heuristic. The two are compared
directly below when both are available, since the real label existing at all
is new enough (added 2026-08-31) that checking they roughly agree is worth
doing before trusting either.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from gem_worldmodel.data.madin_traits import fetch_madin_traits
from gem_worldmodel.eval.intervention import intervene_on_dimension
from gem_worldmodel.eval.probing import (
    heuristic_trophic_label,
    linear_probe_cv,
    most_predictive_latent_dim,
    nonlinear_probe_cv,
)
from gem_worldmodel.features.ecological_traits import real_trophic_label
from gem_worldmodel.models.heads import GrowthRateHead
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.pretrain import load_checkpoint
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-heuristic", action="store_true", help="use the old genome-derived proxy instead"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="jepa_pretrained.pt",
        help="which pretrain checkpoint to probe (e.g. jepa_pretrained_full.pt for the combined-corpus run)",
    )
    parser.add_argument("--k", type=int, default=5, help="number of cross-validation folds for the probes")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="override configs/train.yaml's seed for the probe CV folds (e.g. for a robustness check across seeds)",
    )
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    if args.seed is not None:
        train_cfg = {**train_cfg, "seed": args.seed}
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])
    ckpt_dir = resolve_path(train_cfg["pretrain"]["checkpoint_dir"])

    df = pd.read_csv(processed_dir / "features_sample.csv")
    tensors, _ = build_branch_tensors(df, model_cfg)

    jepa = load_checkpoint(model_cfg, ckpt_dir / args.checkpoint)
    with torch.no_grad():
        latents = jepa.joint_representation(tensors).numpy()

    heuristic_labels = heuristic_trophic_label(
        df["genome_size_bp"].to_numpy(), df["rrna_16s_count"].fillna(0).to_numpy()
    )

    if args.use_heuristic:
        labels_series = pd.Series(heuristic_labels, index=df.index)
        label_source = "heuristic (genome-derived)"
    else:
        madin_traits = fetch_madin_traits(data_cfg)
        labels_series = real_trophic_label(df["species"], madin_traits)
        label_source = "real (isolation_source, genome-independent)"

        overlap = labels_series.notna()
        if overlap.any():
            agree = (labels_series[overlap] == heuristic_labels[overlap.to_numpy()]).mean()
            logger.info(
                f"real vs. heuristic label agreement on the {int(overlap.sum())} species where both exist: "
                f"{agree:.1%}"
            )

    labeled_mask = labels_series.notna().to_numpy()
    latents_labeled = latents[labeled_mask]
    labels = labels_series[labeled_mask].to_numpy().astype(int)
    logger.info(
        f"using {label_source} label: {len(labels)}/{len(df)} species labeled "
        f"({dict(zip(*np.unique(labels, return_counts=True)))})"
    )

    if len(set(labels)) < 2 or len(labels) < 10:
        logger.warning(
            f"only {len(labels)} labeled samples / {len(set(labels))} classes, probing results below are "
            "illustrative only, not statistically meaningful at this sample size."
        )

    probe_seed = args.seed if args.seed is not None else 0
    lin = linear_probe_cv(latents_labeled, labels, k=args.k, seed=probe_seed)
    nonlin = nonlinear_probe_cv(latents_labeled, labels, k=args.k, seed=probe_seed)
    logger.info(
        f"linear probe (k={lin['k']}-fold CV, n={lin['n']}): "
        f"accuracy={lin['accuracy']:.3f} auc={lin.get('auc', float('nan')):.3f}"
    )
    logger.info(
        f"nonlinear probe (k={nonlin['k']}-fold CV, n={nonlin['n']}): "
        f"accuracy={nonlin['accuracy']:.3f} auc={nonlin.get('auc', float('nan')):.3f}"
    )

    best_dim = most_predictive_latent_dim(latents_labeled, labels, k=args.k, seed=probe_seed)
    logger.info(f"most predictive latent dimension (cross-validated): {best_dim}")

    head = GrowthRateHead(jepa.latent_dim, load_config("model")["growth_rate_head"]["hidden_dim"], 1)
    result = intervene_on_dimension(torch.tensor(latents_labeled, dtype=torch.float32), head, best_dim)
    logger.info(f"intervention on dim {best_dim}: shift_by_delta={result['shift_by_delta']}")

    if args.use_heuristic:
        logger.info(
            "reminder: this run used the heuristic label, which is partly defined by absence of CUB signal, "
            "so treat it as necessary but not sufficient (see eval/probing.py). Drop --use-heuristic for the "
            "real, genome-independent label instead."
        )


if __name__ == "__main__":
    main()
