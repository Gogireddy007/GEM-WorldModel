"""Turn a labeled-corpus row set into the full feature
table (genomic traits, CUB, GTDB-distance embedding, 16S baseline embedding,
Arrhenius-corrected target).

`build_feature_table` downloads real genome assemblies per accession (NCBI)
and real 16S sequences (NCBI), it's slow per-genome, so callers typically
pass a bounded sample (see scripts/build_features.py) rather than the
full labeled corpus in one call.
"""

import numpy as np
import pandas as pd

from gem_worldmodel.data import gtdb, ncbi_genomes
from gem_worldmodel.features import cub, genome_traits, phylogeny, rrna16s, temperature
from gem_worldmodel.utils.config import load_config
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def build_feature_table(labeled_rows: pd.DataFrame, data_cfg: dict | None = None, feat_cfg: dict | None = None) -> pd.DataFrame:
    data_cfg = data_cfg or load_config("data")
    feat_cfg = feat_cfg or load_config("features")

    records = []
    for _, row in labeled_rows.iterrows():
        accession = row["accession"]
        logger.info(f"processing {accession} ({row.get('species', '?')})")

        files = ncbi_genomes.fetch_assembly_files(accession, data_cfg)
        record = {"accession": accession, "species": row.get("species")}

        if files["genomic_fna"] is not None:
            try:
                size, gc = genome_traits.genome_size_and_gc(files["genomic_fna"])
                record["genome_size_bp"] = size
                record["gc_content"] = gc
                record.update(genome_traits.count_rrna_trna_genes(files["genomic_fna"]))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"genome_traits failed for {accession}: {exc}")

        if files["cds_fna"] is not None:
            try:
                seqs, headers = cub.load_cds_fasta(files["cds_fna"])
                record["cub"] = cub.compute_cub(seqs, highly_expressed_headers=headers)
                record["regulatory_gene_count"] = genome_traits.regulatory_gene_count(headers)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"CUB computation failed for {accession}: {exc}")

        records.append(record)

    features = pd.DataFrame(records)
    merged = labeled_rows.merge(features, on="accession", suffixes=("", "_feat"))
    merged = temperature.add_reference_temperature_target(merged, feat_cfg)
    return merged


def add_gtdb_distance_embeddings(feature_table: pd.DataFrame, data_cfg: dict | None = None, feat_cfg: dict | None = None) -> pd.DataFrame:
    data_cfg = data_cfg or load_config("data")
    feat_cfg = feat_cfg or load_config("features")

    tree = gtdb.load_tree(data_cfg)
    accessions = set(feature_table["accession"])
    pruned = gtdb.prune_to_accessions(tree, accessions)
    embeddings = phylogeny.build_gtdb_distance_embeddings(pruned, feat_cfg)

    dim = feat_cfg["phylogeny"]["embedding_dim"]
    cols = [f"gtdb_dist_{i}" for i in range(dim)]
    emb_rows = []
    for acc in feature_table["accession"]:
        vec = embeddings.get(acc, np.full(dim, np.nan))
        emb_rows.append(vec)
    emb_df = pd.DataFrame(emb_rows, columns=cols, index=feature_table.index)
    return pd.concat([feature_table, emb_df], axis=1)


def add_16s_embeddings(feature_table: pd.DataFrame, feat_cfg: dict | None = None) -> pd.DataFrame:
    feat_cfg = feat_cfg or load_config("features")
    species_list = feature_table["species"].dropna().unique().tolist()

    sequences = {}
    for sp in species_list:
        seq = rrna16s.fetch_16s_sequence(sp)
        if seq:
            sequences[sp] = seq
        else:
            logger.warning(f"no 16S sequence found for {sp}")

    if len(sequences) < 3:
        logger.warning("fewer than 3 species with 16S sequences; skipping 16S embedding")
        return feature_table

    embeddings = rrna16s.build_16s_embeddings(sequences, feat_cfg)
    dim = feat_cfg["rrna16s"]["embedding_dim"]
    cols = [f"rrna16s_{i}" for i in range(dim)]
    emb_rows = [embeddings.get(sp, np.full(dim, np.nan)) for sp in feature_table["species"]]
    emb_df = pd.DataFrame(emb_rows, columns=cols, index=feature_table.index)
    return pd.concat([feature_table, emb_df], axis=1)
