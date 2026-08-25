#!/usr/bin/env python
"""Build the feature table for a stratified sample of the labeled
corpus (real genomes/sequences from NCBI + GTDB), bounded sample size for a
tractable smoke run; increase `--n-per-class` to scale up.
"""

import argparse

import pandas as pd

from gem_worldmodel.features import build
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def stratified_sample(labeled: pd.DataFrame, split_hours: float, n_per_class: int, seed: int = 0) -> pd.DataFrame:
    usable = labeled[labeled["in_gtdb_tree"]].dropna(subset=["doubling_time_hours"])
    fast = usable[usable["doubling_time_hours"] < split_hours]
    slow = usable[usable["doubling_time_hours"] >= split_hours]
    fast_sample = fast.drop_duplicates(subset=["species"]).sample(
        n=min(n_per_class, fast["species"].nunique()), random_state=seed
    )
    slow_sample = slow.drop_duplicates(subset=["species"]).sample(
        n=min(n_per_class, slow["species"].nunique()), random_state=seed
    )
    return pd.concat([fast_sample, slow_sample]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=10)
    args = parser.parse_args()

    data_cfg = load_config("data")
    feat_cfg = load_config("features")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])

    labeled = pd.read_csv(processed_dir / "labeled_corpus.csv")
    sample = stratified_sample(labeled, data_cfg["doubling_time_split_hours"], args.n_per_class)
    logger.info(f"stratified sample: {len(sample)} accessions ({sample['species'].nunique()} species)")

    features = build.build_feature_table(sample, data_cfg, feat_cfg)
    features = build.add_gtdb_distance_embeddings(features, data_cfg, feat_cfg)
    features = build.add_16s_embeddings(features, feat_cfg)

    out_path = processed_dir / "features_sample.csv"
    features.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path} ({len(features)} rows, {features.shape[1]} columns)")


if __name__ == "__main__":
    main()
