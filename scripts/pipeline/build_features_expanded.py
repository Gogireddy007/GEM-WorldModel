#!/usr/bin/env python
"""Expand the labeled corpus past the 175 exact-GTDB-tree-tip species by
adding species that have real gRodon/Madin growth-rate data and real GTDB
taxonomy, but aren't themselves a tip in the reference tree.

Those species get every other real feature (genome traits, CUB, 16S) exactly
like the original 175, downloaded fresh from NCBI. The one thing they can't
get is an exact phylogenetic placement, since they're not a tree tip, so
their gtdb_distance branch is approximated as the centroid of the embeddings
of tree-tip species sharing the same GTDB genus
(features/phylogeny.py:genus_centroid_embeddings). Species whose genus has no
tree-tip representative at all are skipped, not guessed at with a worse
approximation.

Writes features_sample_expanded.csv, a superset of features_sample.csv, never
overwrites the original 175-species file, so every existing benchmark/seed-
robustness result stays reproducible against features_sample.csv exactly as
before. A `placement_type` column marks which rows are "exact_tip" (the
original 175) and which are "genus_centroid_approx" (the new ones), so any
downstream analysis can filter to exact-only if it wants to.
"""

import argparse

import numpy as np
import pandas as pd

from gem_worldmodel.features import build, phylogeny
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def select_approximable_species(labeled: pd.DataFrame, features_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One accession per non-tip species whose genus matches an already-
    embedded tip species, plus the genus each one was matched on.
    """
    tip_rows = labeled[labeled["in_gtdb_tree"]].drop_duplicates(subset=["accession"])
    tip_taxonomy = dict(zip(tip_rows["accession"], tip_rows["gtdb_taxonomy"]))

    dim = sum(1 for c in features_sample.columns if c.startswith("gtdb_dist_"))
    gtdb_cols = [f"gtdb_dist_{i}" for i in range(dim)]
    tip_embeddings = {
        row["accession"]: row[gtdb_cols].to_numpy(dtype=float)
        for _, row in features_sample.iterrows()
        if row[gtdb_cols].notna().all()
    }

    non_tip = labeled[labeled["has_gtdb_taxonomy"] & ~labeled["in_gtdb_tree"]].dropna(subset=["doubling_time_hours"])
    non_tip = non_tip.drop_duplicates(subset=["species"]).reset_index(drop=True)

    embeddings, matched_genus = phylogeny.genus_centroid_embeddings(non_tip, tip_embeddings, tip_taxonomy)
    approximable = non_tip[non_tip["accession"].isin(embeddings)].copy()
    logger.info(
        f"{len(non_tip)} non-tip labeled species with real growth-rate data, "
        f"{len(approximable)} of those genus-matchable to an existing embedded tip"
    )
    approximable["_gtdb_embedding"] = approximable["accession"].map(embeddings)
    approximable["matched_genus"] = approximable["accession"].map(matched_genus)
    return approximable, matched_genus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process at most this many new species (for testing)")
    args = parser.parse_args()

    data_cfg = load_config("data")
    feat_cfg = load_config("features")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])

    labeled = pd.read_csv(processed_dir / "labeled_corpus.csv")
    features_sample = pd.read_csv(processed_dir / "features_sample.csv")

    approximable, _ = select_approximable_species(labeled, features_sample)
    if args.limit:
        approximable = approximable.iloc[: args.limit].reset_index(drop=True)
    logger.info(f"building features for {len(approximable)} new species")

    new_features = build.build_feature_table(approximable, data_cfg, feat_cfg)

    dim = sum(1 for c in features_sample.columns if c.startswith("gtdb_dist_"))
    gtdb_cols = [f"gtdb_dist_{i}" for i in range(dim)]
    emb_lookup = dict(zip(approximable["accession"], approximable["_gtdb_embedding"]))
    emb_rows = [emb_lookup.get(acc, np.full(dim, np.nan)) for acc in new_features["accession"]]
    emb_df = pd.DataFrame(emb_rows, columns=gtdb_cols, index=new_features.index)
    new_features = pd.concat([new_features, emb_df], axis=1)

    new_features = build.add_16s_embeddings(new_features, feat_cfg)

    new_features["placement_type"] = "genus_centroid_approx"
    matched_genus_map = dict(zip(approximable["accession"], approximable["matched_genus"]))
    new_features["matched_genus"] = new_features["accession"].map(matched_genus_map)

    features_sample = features_sample.copy()
    features_sample["placement_type"] = "exact_tip"
    features_sample["matched_genus"] = None

    combined = pd.concat([features_sample, new_features], ignore_index=True)
    out_path = processed_dir / "features_sample_expanded.csv"
    combined.to_csv(out_path, index=False)
    logger.info(
        f"wrote {out_path}: {len(combined)} rows total "
        f"({(combined['placement_type'] == 'exact_tip').sum()} exact_tip + "
        f"{(combined['placement_type'] == 'genus_centroid_approx').sum()} genus_centroid_approx)"
    )


if __name__ == "__main__":
    main()
