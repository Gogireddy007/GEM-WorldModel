#!/usr/bin/env python
"""Proof-of-concept end-to-end deployment: fine-tune one final growth-rate
head on the entire labeled corpus (no held-out fold, this is not a benchmark
run), then run it on a real sample of GEM MAG genomes that were never
labeled with a measured growth rate, to produce actual predicted doubling
times for genomes outside the training set.

This is deliberately NOT a new evaluation of model quality, the honest R2/
Spearman numbers for that live in FINDINGS.md and come from cross-validated
held-out predictions. This script answers a different, narrower question:
can the pipeline take a genome it has never seen a growth-rate label for and
produce a number for it end to end. It can, this is the artifact that proves
it, predictions should be read with the same caveats as everywhere else in
this project (R2 is frequently negative on held-out data, so treat the
output as a rough rank-ordering signal, not a precise doubling-time
estimate).

Preferentially samples GEM genomes that have a real barrnap-extracted 16S
sequence, so predictions use the full 3-branch model, the same input shape
the model was fine-tuned on, rather than a 2-branch fallback.
"""

import argparse
from ast import literal_eval

import numpy as np
import pandas as pd
import torch

from gem_worldmodel.features.rrna16s import build_16s_embeddings_from_profiles
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.finetune import finetune, split_indices
from gem_worldmodel.training.pretrain import load_checkpoint
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-genomes", type=int, default=1000,
        help="how many unlabeled genomes to predict on (ignored if --genome-ids-file is given, "
        "then every listed genome with full 3-branch data is used)",
    )
    parser.add_argument(
        "--genome-ids-file", type=str, default=None,
        help="CSV with a genome_id column restricting which genomes to predict on, e.g. a "
        "specific quality-filtered set, instead of a random sample of the full unlabeled corpus",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="jepa_pretrained_full_expanded.pt",
        help="which pretrain checkpoint to fine-tune from",
    )
    parser.add_argument(
        "--features-file", type=str, default="features_sample_expanded.csv",
        help="labeled corpus to fine-tune the deployable model on",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    train_cfg = {**train_cfg, "seed": args.seed}
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])
    ckpt_dir = resolve_path(train_cfg["pretrain"]["checkpoint_dir"])
    all_branches = [b["name"] for b in model_cfg["branches"]]

    labeled_df = pd.read_csv(processed_dir / args.features_file)
    labeled_tensors, standardizers = build_branch_tensors(labeled_df, model_cfg, branches=all_branches)
    target_log = torch.tensor(
        np.log(labeled_df["doubling_time_hours_ref"].to_numpy()), dtype=torch.float32
    )
    logger.info(f"fine-tuning the deployable model on all {len(labeled_df)} labeled species (no held-out fold)")

    n = len(labeled_df)
    val_frac = 0.1
    splits = split_indices(n, train_frac=1.0 - val_frac, val_frac=val_frac, seed=args.seed)
    splits["train"] = np.concatenate([splits["train"], splits["test"]])
    splits["test"] = np.array([], dtype=int)

    jepa = load_checkpoint(model_cfg, ckpt_dir / args.checkpoint)
    result = finetune(jepa, labeled_tensors, target_log, train_cfg, splits=splits)
    jepa, head = result["jepa"], result["head"]

    gem_base_path = processed_dir / "unlabeled_corpus_features.csv"
    gem_enriched_path = processed_dir / "unlabeled_corpus_features_enriched.csv"
    gem_16s_path = processed_dir / "unlabeled_corpus_features_16s.csv"
    gem_df = pd.read_csv(gem_base_path)
    if gem_enriched_path.exists():
        enriched = pd.read_csv(gem_enriched_path)
        gem_df = gem_df.merge(enriched, on="genome_id", how="left", suffixes=("", "_enriched"))
        if "gc_content_enriched" in gem_df.columns:
            gem_df["gc_content"] = gem_df["gc_content_enriched"]

    s16 = pd.read_csv(gem_16s_path)[["genome_id", "kmer_profile_16s"]]
    gem_df = gem_df.merge(s16, on="genome_id", how="left")
    gem_df = gem_df.dropna(subset=[f"gtdb_dist_{i}" for i in range(model_cfg["branches"][1]["input_dim"])])
    has_16s = gem_df["kmer_profile_16s"].notna()
    gem_3branch = gem_df[has_16s].copy()
    logger.info(f"{len(gem_3branch)} unlabeled GEM genomes have real 16S + traits + phylogeny (full 3-branch)")

    if args.genome_ids_file:
        wanted_ids = pd.read_csv(args.genome_ids_file)["genome_id"]
        sample_df = gem_3branch[gem_3branch["genome_id"].isin(set(wanted_ids))].reset_index(drop=True)
        logger.info(
            f"restricting to {len(wanted_ids)} genome IDs from {args.genome_ids_file}: "
            f"{len(sample_df)} of those have full 3-branch data available"
        )
    else:
        rng = np.random.default_rng(args.seed)
        n_sample = min(args.n_genomes, len(gem_3branch))
        sample_idx = rng.choice(len(gem_3branch), size=n_sample, replace=False)
        sample_df = gem_3branch.iloc[sample_idx].reset_index(drop=True)

    profiles = {row.genome_id: literal_eval(row.kmer_profile_16s) for row in sample_df.itertuples()}
    embeddings = build_16s_embeddings_from_profiles(profiles, load_config("features"))
    dim = model_cfg["branches"][2]["input_dim"]
    for i in range(dim):
        sample_df[f"rrna16s_{i}"] = sample_df["genome_id"].map(lambda gid, i=i: embeddings[gid][i])

    sample_tensors, _ = build_branch_tensors(sample_df, model_cfg, standardizers=standardizers, branches=all_branches)

    jepa.eval()
    head.eval()
    with torch.no_grad():
        latents = jepa.joint_representation(sample_tensors)
        pred_log = head(latents)
    pred_hours = np.exp(pred_log.numpy())

    out = pd.DataFrame(
        {
            "genome_id": sample_df["genome_id"],
            "predicted_doubling_time_hours": pred_hours,
            "genome_size_bp": sample_df["genome_size_bp"],
            "gc_content": sample_df["gc_content"],
        }
    )
    out = out.sort_values("predicted_doubling_time_hours").reset_index(drop=True)

    out_name = "unlabeled_predictions_poc.csv" if not args.genome_ids_file else "unlabeled_predictions_hq.csv"
    out_path = processed_dir / out_name
    out.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}: {len(out)} real GEM genomes with predicted doubling times")
    logger.info(
        f"predicted doubling time (hours): min={out['predicted_doubling_time_hours'].min():.2f} "
        f"median={out['predicted_doubling_time_hours'].median():.2f} "
        f"max={out['predicted_doubling_time_hours'].max():.2f}"
    )
    logger.info(
        "reminder: these are proof-of-concept end-to-end predictions, not validated accuracy claims, "
        "see FINDINGS.md for the honest cross-validated R2/Spearman numbers this model actually achieves"
    )


if __name__ == "__main__":
    main()
