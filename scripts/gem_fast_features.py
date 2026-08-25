#!/usr/bin/env python
"""Fast pass over ALL 52,515 GEM MAGs: genomic traits directly from GEM's own
metadata (genome_length, real rRNA/tRNA counts from their pipeline, no
barrnap needed, they already ran it) + phylogenetic landmark embedding from
GEM's own tree (matched via otu_id, since GEM's JGI genome_ids don't match
NCBI/GTDB accessions, see data/gem_tree.py). No genome downloads required,
so this covers the full corpus in minutes, not hours.

GC content and real 16S sequences aren't in the metadata and require
downloading each genome, that's the slow pass (gem_slow_features.py),
which fills in those two columns incrementally on top of this file.
"""

import numpy as np
import pandas as pd

from gem_worldmodel.data.gem_tree import load_gem_tree
from gem_worldmodel.features.phylogeny import landmark_distance_embedding
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    data_cfg = load_config("data")
    feat_cfg = load_config("features")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])

    meta = pd.read_csv(resolve_path(data_cfg["paths"]["raw_dir"]) / "gem_genome_metadata.tsv", sep="\t")
    logger.info(f"loaded metadata for {len(meta)} GEM genomes")

    df = pd.DataFrame(
        {
            "genome_id": meta["genome_id"],
            "otu_id": meta["otu_id"],
            "genome_size_bp": meta["genome_length"].astype(float),
            "rrna_16s_count": meta["num_16s"].astype(float),
            "rrna_23s_count": meta["num_23s"].astype(float),
            "rrna_5s_count": meta["num_5s"].astype(float),
            "trna_count": meta["num_trna"].astype(float),
            "regulatory_gene_count": np.nan,  # needs annotated CDS, not available for MAGs
            "gc_content": np.nan,  # filled by gem_slow_features.py
            "completeness": meta["completeness"],
            "contamination": meta["contamination"],
            "quality_score": meta["quality_score"],
            "ecosystem_category": meta["ecosystem_category"],
        }
    )

    logger.info("loading GEM's own phylogenetic tree (43,979 OTUs)...")
    tree = load_gem_tree(data_cfg)
    dim = feat_cfg["phylogeny"]["embedding_dim"]
    logger.info(f"computing landmark distance embedding (dim={dim}) for all OTU tips...")
    otu_embeddings = landmark_distance_embedding(tree, n_landmarks=dim, seed=0)
    logger.info(f"embedded {len(otu_embeddings)} OTU tips")

    n_otus_with_embedding = df["otu_id"].isin(otu_embeddings.keys()).sum()
    logger.info(f"{n_otus_with_embedding}/{len(df)} genomes have an OTU that matched a tree tip")

    emb_cols = [f"gtdb_dist_{i}" for i in range(dim)]
    emb_matrix = np.full((len(df), dim), np.nan)
    for i, otu in enumerate(df["otu_id"]):
        if otu in otu_embeddings:
            emb_matrix[i] = otu_embeddings[otu]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index)
    df = pd.concat([df, emb_df], axis=1)

    out_path = processed_dir / "unlabeled_corpus_features.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path} ({len(df)} rows, {df.shape[1]} columns)")
    logger.info(
        "gc_content is NaN for all rows, run scripts/gem_slow_features.py to fill it in "
        "(and real 16S sequences) via streaming genome download."
    )


if __name__ == "__main__":
    main()
