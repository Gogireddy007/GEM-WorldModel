"""Pull the GTDB reference tree + taxonomy for phylogenetic placement.

GTDB serves per-release static files at data.gtdb.ecogenomic.org/releases/latest/.
We fetch the bac120 (bacterial) tree and taxonomy table; the full metadata
table (~289MB compressed) is only fetched if explicitly requested since
taxonomy + tree are sufficient for the GTDB-distance embedding branch.
"""

import gzip
import shutil
from pathlib import Path

import dendropy
import pandas as pd
import requests

from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def _download(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"cached: {dest}")
        return dest
    logger.info(f"downloading {url}")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
    return dest


def _gunzip(src: Path, dest: Path) -> Path:
    if dest.exists():
        return dest
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dest


def fetch_bac_tree(cfg: dict | None = None) -> Path:
    """Download (and decompress) the GTDB bac120 reference tree, return the local path."""
    cfg = cfg or load_config("data")
    g = cfg["gtdb"]
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    gz_path = _download(f"{g['base_url']}/{g['bac_tree_file']}", raw_dir / g["bac_tree_file"])
    tree_path = raw_dir / g["bac_tree_file"].removesuffix(".gz")
    return _gunzip(gz_path, tree_path)


def fetch_bac_taxonomy(cfg: dict | None = None) -> pd.DataFrame:
    """Download the GTDB bac120 taxonomy table (accession -> full lineage string)."""
    cfg = cfg or load_config("data")
    g = cfg["gtdb"]
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    gz_path = _download(
        f"{g['base_url']}/{g['bac_taxonomy_file']}", raw_dir / g["bac_taxonomy_file"]
    )
    tsv_path = raw_dir / g["bac_taxonomy_file"].removesuffix(".gz")
    _gunzip(gz_path, tsv_path)
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["accession", "gtdb_taxonomy"])
    # GTDB tip labels/accessions are prefixed RS_/GB_; keep both raw and bare forms for joins.
    df["accession_bare"] = df["accession"].str.replace(r"^(RS_|GB_)", "", regex=True)
    return df


def fetch_full_metadata(cfg: dict | None = None) -> pd.DataFrame:
    """Download the full GTDB metadata table (~289MB compressed). Opt-in only."""
    cfg = cfg or load_config("data")
    g = cfg["gtdb"]
    if not g.get("fetch_full_metadata", False):
        raise RuntimeError(
            "fetch_full_metadata is disabled in configs/data.yaml (gtdb.fetch_full_metadata: false). "
            "This file is ~289MB compressed; enable explicitly if you need it."
        )
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    gz_path = _download(
        f"{g['base_url']}/{g['bac_metadata_file']}", raw_dir / g["bac_metadata_file"]
    )
    return pd.read_csv(gz_path, sep="\t", compression="gzip", low_memory=False)


def load_tree(cfg: dict | None = None) -> dendropy.Tree:
    """Load the GTDB bac120 tree into a dendropy Tree object."""
    tree_path = fetch_bac_tree(cfg)
    logger.info(f"parsing tree from {tree_path}")
    return dendropy.Tree.get(path=str(tree_path), schema="newick", preserve_underscores=True)


def get_tree_tip_accessions(cfg: dict | None = None) -> set[str]:
    """Bare accessions (RS_/GB_ prefix stripped) that are ACTUAL tips in the
    GTDB reference tree.

    This is a materially smaller set than the taxonomy table
    (`fetch_bac_taxonomy`): the taxonomy table lists every genome GTDB has
    classified (~879k in R220), while the tree itself only has representative
    genomes as tips (~190k). A genome having GTDB taxonomy does NOT mean it
    has a tree placement, only tree tip membership gives you a real
    patristic-distance embedding via `phylogeny.build_gtdb_distance_embeddings`.
    """
    tree = load_tree(cfg)
    accessions = set()
    for leaf in tree.leaf_node_iter():
        if leaf.taxon is None:
            continue
        label = leaf.taxon.label
        bare = label.split("_", 1)[1] if label.startswith(("RS_", "GB_")) else label
        accessions.add(bare)
    return accessions


def prune_to_accessions(tree: dendropy.Tree, accessions_bare: set[str]) -> dendropy.Tree:
    """Return a copy of `tree` pruned to only the taxa matching the given bare accessions.

    GTDB tip labels look like 'RS_GCF_000006945.2' or 'GB_GCA_...'; matching is
    done against the accession with the RS_/GB_ prefix stripped.
    """
    tree = tree.clone(depth=1)
    keep = []
    for leaf in tree.leaf_node_iter():
        label = leaf.taxon.label if leaf.taxon else None
        if label is None:
            continue
        bare = label.split("_", 1)[1] if label.startswith(("RS_", "GB_")) else label
        if bare in accessions_bare:
            keep.append(leaf.taxon)
    if not keep:
        raise ValueError("none of the requested accessions were found as tree tips")
    tree.retain_taxa(keep)
    return tree
