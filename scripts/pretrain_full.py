#!/usr/bin/env python
"""Self-supervised masked-branch pretraining at real scale, across BOTH
the labeled corpus (gRodon/Madin, 3 real branches) and the GEM MAG corpus
(genomic_traits + gtdb_distance for the full 52,515; + real rrna16s for
whatever subset scripts/gem_slow_features.py --with-16s has completed by the
time this runs), in one model. Pretraining doesn't need growth-rate labels,
so species without one still contribute here.

Run scripts/pull_data.py, scripts/build_features.py,
scripts/gem_fast_features.py, and scripts/gem_slow_features.py first.
"""

import argparse

import pandas as pd

from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.pretrain import pretrain_multi_corpus, save_checkpoint
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])
    all_branches = [b["name"] for b in model_cfg["branches"]]

    corpora = []

    labeled_path = processed_dir / "features_sample.csv"
    if labeled_path.exists():
        labeled_df = pd.read_csv(labeled_path)
        labeled_tensors, _ = build_branch_tensors(labeled_df, model_cfg, branches=all_branches)
        corpora.append({"name": "labeled_grodon", "tensors": labeled_tensors, "branches": all_branches})
        logger.info(f"labeled corpus: {len(labeled_df)} rows, branches={all_branches}")
    else:
        logger.warning(f"{labeled_path} not found, run scripts/build_features.py first")

    gem_base_path = processed_dir / "unlabeled_corpus_features.csv"
    gem_enriched_path = processed_dir / "unlabeled_corpus_features_enriched.csv"
    if gem_base_path.exists():
        gem_df = pd.read_csv(gem_base_path)
        if gem_enriched_path.exists():
            enriched = pd.read_csv(gem_enriched_path)
            gem_df = gem_df.merge(enriched, on="genome_id", how="left", suffixes=("", "_enriched"))
            if "gc_content_enriched" in gem_df.columns:
                gem_df["gc_content"] = gem_df["gc_content_enriched"]

        gem_df = gem_df.dropna(subset=[f"gtdb_dist_{i}" for i in range(model_cfg["branches"][1]["input_dim"])])
        two_branch = [b for b in all_branches if b != "rrna16s"]
        gem_tensors, _ = build_branch_tensors(gem_df, model_cfg, branches=two_branch)
        corpora.append({"name": "gem_mags", "tensors": gem_tensors, "branches": two_branch})
        logger.info(f"GEM corpus: {len(gem_df)} rows (with GTDB placement), branches={two_branch}")

        if "kmer_profile_16s" in gem_df.columns:
            has_16s = gem_df["kmer_profile_16s"].notna().sum()
            logger.info(f"  ({has_16s} of these also have real 16S data, not yet used as a 3rd branch subset)")
    else:
        logger.warning(f"{gem_base_path} not found, run scripts/gem_fast_features.py first")

    if not corpora:
        raise RuntimeError("no corpora available, run the data pipeline scripts first")

    result = pretrain_multi_corpus(corpora, model_cfg, train_cfg, epochs=args.epochs)

    ckpt_dir = resolve_path(train_cfg["pretrain"]["checkpoint_dir"])
    save_checkpoint(result["model"], ckpt_dir / "jepa_pretrained_full.pt")
    logger.info(
        f"final loss={result['history']['loss'][-1]:.4f} collapsed_at={result['collapsed_at']}"
    )


if __name__ == "__main__":
    main()
