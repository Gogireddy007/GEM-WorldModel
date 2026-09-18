#!/usr/bin/env python
"""Add a small, deliberately targeted set of real species to the corpus,
chosen for one reason only: they land inside the two training-data gaps
found and quantified in FINDINGS.md (the thermophile-heavy and Vibrio-heavy
regions), and their growth rates are real, published measurements from
Madin et al. 2020's full dataset, not a subset this project had already
pulled.

Every one of these species turned out to be an exact GTDB tree tip once
matched to GTDB's own pinned representative genome accession (not always
the same accession NCBI's search returns as "current best"), so this is
real phylogenetic placement, not an approximation. Their gtdb_distance
embedding is recomputed jointly with the existing 175 exact-tip species,
the same way the original corpus was built, since adding tips to the set
changes the embedding space for everyone in it.

Reads --species-file, a small CSV of species, accession, doubling_time_hours,
built by hand from a targeted search, not derived automatically. Writes
features_sample_gapfilled.csv, a superset of features_sample_full.csv, which
stays untouched.
"""

import argparse

import numpy as np
import pandas as pd

from gem_worldmodel.data import gtdb
from gem_worldmodel.features import build, phylogeny
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species-file", type=str, required=True)
    parser.add_argument("--base-file", type=str, default="features_sample_full.csv")
    args = parser.parse_args()

    data_cfg = load_config("data")
    feat_cfg = load_config("features")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])

    base = pd.read_csv(processed_dir / args.base_file)
    new_species = pd.read_csv(args.species_file)
    new_species = new_species.rename(columns={"doubling_time_hours": "doubling_time_hours_input"})
    logger.info(f"adding {len(new_species)} targeted gap-filling species to {len(base)} existing rows")

    new_species["doubling_time_hours"] = new_species["doubling_time_hours_input"]
    new_features = build.build_feature_table(new_species, data_cfg, feat_cfg)
    new_features["doubling_time_hours_ref"] = new_features["doubling_time_hours"]

    exact_tip_accessions = set(base[base["placement_type"] == "exact_tip"]["accession"]) | set(new_features["accession"])
    tree = gtdb.load_tree(data_cfg)
    pruned = gtdb.prune_to_accessions(tree, exact_tip_accessions)
    embeddings = phylogeny.build_gtdb_distance_embeddings(pruned, feat_cfg)

    dim = feat_cfg["phylogeny"]["embedding_dim"]
    gtdb_cols = [f"gtdb_dist_{i}" for i in range(dim)]

    def apply_embeddings(df: pd.DataFrame) -> pd.DataFrame:
        emb_rows = [embeddings.get(acc, np.full(dim, np.nan)) for acc in df["accession"]]
        emb_df = pd.DataFrame(emb_rows, columns=gtdb_cols, index=df.index)
        return pd.concat([df.drop(columns=[c for c in gtdb_cols if c in df.columns]), emb_df], axis=1)

    new_features = apply_embeddings(new_features)
    new_features = build.add_16s_embeddings(new_features, feat_cfg)
    new_features["placement_type"] = "exact_tip"
    new_features["matched_genus"] = None
    if "matched_rank" in base.columns:
        new_features["matched_rank"] = None

    base_exact = base[base["placement_type"] == "exact_tip"].copy()
    base_exact = apply_embeddings(base_exact)
    base_rest = base[base["placement_type"] != "exact_tip"]

    combined = pd.concat([base_exact, base_rest, new_features], ignore_index=True)
    out_path = processed_dir / "features_sample_gapfilled.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}: {len(combined)} rows total ({len(new_features)} newly added gap-filling species)")
    logger.info(f"new species added: {new_features['species'].tolist()}")


if __name__ == "__main__":
    main()
