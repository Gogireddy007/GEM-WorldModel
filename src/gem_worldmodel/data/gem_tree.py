"""GEM's own phylogenetic tree (43,979 species-level OTUs, 30 universal
marker genes, FastTree, see portal.nersc.gov/GEM/README.md's --tree/
section), used as the phylogeny branch source for GEM MAGs.

This is deliberately NOT the official GTDB release tree used for the labeled
gRodon/Madin corpus (data/gtdb.py): GEM's genome_id values are JGI IMG
identifiers (e.g. '3300001683_29'), not NCBI GCA/GCF accessions, so they
never match tips in the GTDB release tree. GEM ships its own real tree whose
tip labels are 'OTU-<n>', matching the `otu_id` column in
genome_metadata.tsv directly, that's the correct, real phylogeny source
for this corpus, not a workaround.
"""

import shutil
from pathlib import Path

import dendropy
import requests

from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

TREE_URL = "https://portal.nersc.gov/GEM/tree/multi_marker.rooted.tree"


def fetch_gem_tree(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config("data")
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    dest = raw_dir / "gem_multi_marker.rooted.tree"
    if dest.exists():
        logger.info(f"cached: {dest}")
        return dest
    logger.info(f"downloading {TREE_URL}")
    with requests.get(TREE_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
    return dest


def load_gem_tree(cfg: dict | None = None) -> dendropy.Tree:
    path = fetch_gem_tree(cfg)
    logger.info(f"parsing GEM tree from {path}")
    return dendropy.Tree.get(path=str(path), schema="newick", preserve_underscores=True)
