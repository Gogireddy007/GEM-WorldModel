#!/usr/bin/env python
"""This adds the remaining real, labeled species that still aren't in the corpus
after the genus-level expansion, the ones whose genus has no tree tip at
all, using a family-level fallback for the phylogeny branch instead of just
skipping them.

Genome traits, CUB, and 16S don't need any tree placement to compute, only
the gtdb_distance branch does, so the genus requirement in the earlier
expansion script was stricter than the data actually demanded. This script
uses features/phylogeny.py:taxonomic_centroid_embeddings, which tries genus
first and falls back to family only when genus truly has no match anywhere
in the tree. Family-level is a real, but weaker, approximation than genus,
grouping more distantly related organisms together, and every added row is
tagged with exactly which rank it was actually matched on so this is never
mixed in as if it were the same quality of evidence as an exact tip or a
genus match.

Every centroid is still built only from real, exact tree-tip embeddings,
never from another approximation, so uncertainty doesn't compound across
rounds of expansion.

Writes features_sample_full.csv, a superset of features_sample_expanded.csv,
which itself stays untouched, so every existing result stays reproducible
against the file it was actually computed on.
"""

import argparse

import numpy as np
import pandas as pd

from gem_worldmodel.features import build, phylogeny
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def select_addable_species(labeled: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    tip_rows = labeled[labeled["in_gtdb_tree"]].drop_duplicates(subset=["accession"])
    tip_taxonomy = dict(zip(tip_rows["accession"], tip_rows["gtdb_taxonomy"]))

    dim = sum(1 for c in current.columns if c.startswith("gtdb_dist_"))
    gtdb_cols = [f"gtdb_dist_{i}" for i in range(dim)]
    exact_tip_rows = current[current["placement_type"] == "exact_tip"]
    tip_embeddings = {
        row["accession"]: row[gtdb_cols].to_numpy(dtype=float)
        for _, row in exact_tip_rows.iterrows()
        if row[gtdb_cols].notna().all()
    }

    already_have = set(current["species"])
    non_tip = labeled[labeled["has_gtdb_taxonomy"] & ~labeled["in_gtdb_tree"]].dropna(subset=["doubling_time_hours"])
    non_tip = non_tip.drop_duplicates(subset=["species"]).reset_index(drop=True)
    non_tip = non_tip[~non_tip["species"].isin(already_have)].reset_index(drop=True)

    embeddings, matched_rank = phylogeny.taxonomic_centroid_embeddings(
        non_tip, tip_embeddings, tip_taxonomy, ranks=("g__", "f__")
    )
    addable = non_tip[non_tip["accession"].isin(embeddings)].copy()
    n_genus = sum(1 for r in matched_rank if r.startswith("g__"))
    n_family = sum(1 for r in matched_rank if r.startswith("f__"))
    logger.info(
        f"{len(non_tip)} labeled species still missing, {len(addable)} addable "
        f"({n_genus} via genus match, {n_family} via family fallback)"
    )
    addable["_gtdb_embedding"] = addable["accession"].map(embeddings)
    addable["matched_rank"] = addable["accession"].map(matched_rank)
    return addable, matched_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process at most this many new species (for testing)")
    args = parser.parse_args()

    data_cfg = load_config("data")
    feat_cfg = load_config("features")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])

    labeled = pd.read_csv(processed_dir / "labeled_corpus.csv")
    current = pd.read_csv(processed_dir / "features_sample_expanded.csv")

    addable, _ = select_addable_species(labeled, current)
    if args.limit:
        addable = addable.iloc[: args.limit].reset_index(drop=True)
    logger.info(f"building features for {len(addable)} new species")

    new_features = build.build_feature_table(addable, data_cfg, feat_cfg)

    dim = sum(1 for c in current.columns if c.startswith("gtdb_dist_"))
    gtdb_cols = [f"gtdb_dist_{i}" for i in range(dim)]
    emb_lookup = dict(zip(addable["accession"], addable["_gtdb_embedding"]))
    emb_rows = [emb_lookup.get(acc, np.full(dim, np.nan)) for acc in new_features["accession"]]
    emb_df = pd.DataFrame(emb_rows, columns=gtdb_cols, index=new_features.index)
    new_features = pd.concat([new_features, emb_df], axis=1)

    new_features = build.add_16s_embeddings(new_features, feat_cfg)

    rank_lookup = dict(zip(addable["accession"], addable["matched_rank"]))
    new_features["matched_rank"] = new_features["accession"].map(rank_lookup)
    new_features["placement_type"] = new_features["matched_rank"].apply(
        lambda r: "genus_centroid_approx" if isinstance(r, str) and r.startswith("g__") else "family_centroid_approx"
    )
    new_features["matched_genus"] = new_features["matched_rank"].apply(
        lambda r: r if isinstance(r, str) and r.startswith("g__") else None
    )

    current = current.copy()
    if "matched_rank" not in current.columns:
        current["matched_rank"] = current["matched_genus"]

    combined = pd.concat([current, new_features], ignore_index=True)
    out_path = processed_dir / "features_sample_full.csv"
    combined.to_csv(out_path, index=False)
    counts = combined["placement_type"].value_counts()
    logger.info(f"wrote {out_path}: {len(combined)} rows total")
    for placement_type, count in counts.items():
        logger.info(f"  {placement_type}: {count}")


if __name__ == "__main__":
    main()
