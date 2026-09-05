#!/usr/bin/env python
"""Self-supervised masked-branch pretraining at real scale, across the
labeled corpus (gRodon/Madin, 3 real branches) and the GEM MAG corpus, split
into two sub-corpora by what each genome actually has real data for:
genomes with a real barrnap-extracted 16S sequence get all 3 branches,
everyone else gets genomic_traits + gtdb_distance only. Pretraining doesn't
need growth-rate labels, so species without one still contribute here.

Run scripts/pull_data.py, scripts/build_features.py,
scripts/gem_fast_features.py, and scripts/gem_slow_features.py first.
"""

import argparse
from ast import literal_eval
from pathlib import Path

import pandas as pd

from gem_worldmodel.features.rrna16s import build_16s_embeddings_from_profiles
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.pretrain import pretrain_multi_corpus, save_checkpoint
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="override configs/train.yaml's seed. Also changes the output checkpoint filename "
        "to jepa_pretrained_full_seed{N}.pt so it doesn't overwrite the default-seed checkpoint.",
    )
    parser.add_argument(
        "--features-file", type=str, default="features_sample.csv",
        help="which processed feature table to use for the labeled sub-corpus, e.g. "
        "features_sample_expanded.csv for the 304-species corpus. Using a non-default file "
        "tags the output checkpoint filename so it never overwrites the default corpus's checkpoint.",
    )
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    if args.seed is not None:
        train_cfg = {**train_cfg, "seed": args.seed}
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])
    all_branches = [b["name"] for b in model_cfg["branches"]]

    corpora = []

    labeled_path = processed_dir / args.features_file
    if labeled_path.exists():
        labeled_df = pd.read_csv(labeled_path)
        labeled_tensors, _ = build_branch_tensors(labeled_df, model_cfg, branches=all_branches)
        corpora.append({"name": "labeled_grodon", "tensors": labeled_tensors, "branches": all_branches})
        logger.info(f"labeled corpus: {len(labeled_df)} rows, branches={all_branches}")
    else:
        logger.warning(f"{labeled_path} not found, run scripts/build_features.py first")

    gem_base_path = processed_dir / "unlabeled_corpus_features.csv"
    gem_enriched_path = processed_dir / "unlabeled_corpus_features_enriched.csv"
    gem_16s_path = processed_dir / "unlabeled_corpus_features_16s.csv"
    if gem_base_path.exists():
        gem_df = pd.read_csv(gem_base_path)
        if gem_enriched_path.exists():
            enriched = pd.read_csv(gem_enriched_path)
            gem_df = gem_df.merge(enriched, on="genome_id", how="left", suffixes=("", "_enriched"))
            if "gc_content_enriched" in gem_df.columns:
                gem_df["gc_content"] = gem_df["gc_content_enriched"]
        if gem_16s_path.exists():
            s16 = pd.read_csv(gem_16s_path)[["genome_id", "kmer_profile_16s"]]
            gem_df = gem_df.merge(s16, on="genome_id", how="left")

        gem_df = gem_df.dropna(subset=[f"gtdb_dist_{i}" for i in range(model_cfg["branches"][1]["input_dim"])])
        two_branch = [b for b in all_branches if b != "rrna16s"]

        has_16s = gem_df["kmer_profile_16s"].notna() if "kmer_profile_16s" in gem_df.columns else pd.Series(False, index=gem_df.index)
        gem_3branch_df = gem_df[has_16s].copy()
        gem_2branch_df = gem_df[~has_16s].copy()

        if not gem_2branch_df.empty:
            gem_tensors, _ = build_branch_tensors(gem_2branch_df, model_cfg, branches=two_branch)
            corpora.append({"name": "gem_mags_2branch", "tensors": gem_tensors, "branches": two_branch})
            logger.info(f"GEM corpus (2-branch, no real 16S): {len(gem_2branch_df)} rows, branches={two_branch}")

        if not gem_3branch_df.empty:
            profiles = {
                row.genome_id: literal_eval(row.kmer_profile_16s) for row in gem_3branch_df.itertuples()
            }
            embeddings = build_16s_embeddings_from_profiles(profiles, load_config("features"))
            dim = model_cfg["branches"][2]["input_dim"]
            for i in range(dim):
                gem_3branch_df[f"rrna16s_{i}"] = gem_3branch_df["genome_id"].map(
                    lambda gid, i=i: embeddings[gid][i]
                )
            gem_tensors_3, _ = build_branch_tensors(gem_3branch_df, model_cfg, branches=all_branches)
            corpora.append({"name": "gem_mags_3branch", "tensors": gem_tensors_3, "branches": all_branches})
            logger.info(f"GEM corpus (3-branch, real 16S): {len(gem_3branch_df)} rows, branches={all_branches}")
    else:
        logger.warning(f"{gem_base_path} not found, run scripts/gem_fast_features.py first")

    if not corpora:
        raise RuntimeError("no corpora available, run the data pipeline scripts first")

    logger.info(f"pretraining (seed={train_cfg['seed']})")
    result = pretrain_multi_corpus(corpora, model_cfg, train_cfg, epochs=args.epochs)

    is_default = args.features_file == "features_sample.csv"
    tag = "" if is_default else f"_{Path(args.features_file).stem.removeprefix('features_sample_')}"
    ckpt_dir = resolve_path(train_cfg["pretrain"]["checkpoint_dir"])
    ckpt_name = (
        f"jepa_pretrained_full{tag}.pt" if args.seed is None else f"jepa_pretrained_full{tag}_seed{args.seed}.pt"
    )
    save_checkpoint(result["model"], ckpt_dir / ckpt_name)
    logger.info(
        f"final loss={result['history']['loss'][-1]:.4f} collapsed_at={result['collapsed_at']}"
    )


if __name__ == "__main__":
    main()
