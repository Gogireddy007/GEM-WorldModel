"""Pull the DOE NERSC GEM (Genomes from Earth's Microbiomes) MAG catalogue.

portal.nersc.gov/GEM serves 52,515 metagenome-assembled genomes (MAGs) with a
~12MB genome_metadata.tsv and per-genome .fna.gz files under genomes/fna/.
Most of these MAGs have no measured growth rate, so they're used
as the large unlabeled corpus for self-supervised pretraining, which doesn't
need labels in the first place.

The lab's Dropbox folder of GEM genomes is treated as an alternative source
for the same underlying genomes; `fetch_dropbox_folder_zip` downloads it as a
single zip (Dropbox supports this for shared folder links via `dl=1`), but is
not invoked automatically since its size is unbounded and unknown ahead of
time, call it explicitly when you actually want that bulk archive.
"""

import gzip
import shutil
from pathlib import Path

import pandas as pd
import requests

from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def _download(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    logger.info(f"downloading {url}")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
    return dest


def fetch_genome_metadata(cfg: dict | None = None) -> pd.DataFrame:
    """Download the GEM catalogue's genome_metadata.tsv (quality, taxonomy, OTU, env metadata)."""
    cfg = cfg or load_config("data")
    g = cfg["gem_portal"]
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    dest = raw_dir / "gem_genome_metadata.tsv"
    _download(f"{g['base_url']}/{g['metadata_path']}", dest)
    return pd.read_csv(dest, sep="\t", low_memory=False)


def fetch_genome_fna(genome_id: str, cfg: dict | None = None, decompress: bool = False) -> Path:
    """Download a single MAG's nucleotide FASTA by its GEM genome_id (e.g. '2004178001_1')."""
    cfg = cfg or load_config("data")
    g = cfg["gem_portal"]
    raw_dir = resolve_path(cfg["paths"]["raw_dir"]) / "gem_genomes"
    url = f"{g['base_url']}/{g['genome_fna_path_template'].format(genome_id=genome_id)}"
    gz_dest = raw_dir / f"{genome_id}.fna.gz"
    _download(url, gz_dest)
    if not decompress:
        return gz_dest
    fna_dest = raw_dir / f"{genome_id}.fna"
    if not fna_dest.exists():
        with gzip.open(gz_dest, "rb") as f_in, open(fna_dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return fna_dest


def select_unlabeled_pretrain_genomes(cfg: dict | None = None) -> pd.DataFrame:
    """Select a subset of GEM MAGs to use as the unlabeled pretraining corpus.

    Filters to genomes that pass the catalogue's own quality bar (already
    required > 50 CheckM quality score per the portal README) and caps the
    count at `gem_portal.max_unlabeled_genomes` for tractable local runs.
    """
    cfg = cfg or load_config("data")
    g = cfg["gem_portal"]
    meta = fetch_genome_metadata(cfg)

    quality_cols = [c for c in meta.columns if "quality" in c.lower() or "completeness" in c.lower()]
    if quality_cols:
        logger.info(f"quality columns available for filtering: {quality_cols}")

    cap = g.get("max_unlabeled_genomes")
    if cap is not None and len(meta) > cap:
        meta = meta.sample(n=cap, random_state=0).reset_index(drop=True)
    logger.info(f"selected {len(meta)} GEM MAGs for the unlabeled pretraining corpus")
    return meta


def fetch_dropbox_folder_zip(cfg: dict | None = None, timeout: int = 600) -> Path:
    """Download the lab's shared Dropbox folder of GEM genomes as a single zip.

    Not called automatically by the data pipeline, the folder's total size
    is unknown ahead of time. Call this explicitly when you want that bulk
    archive as an alternative/supplement to the NERSC portal fetch above.
    """
    cfg = cfg or load_config("data")
    g = cfg["gem_portal"]
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    dest = raw_dir / "gem_dropbox_folder.zip"
    logger.info("fetching Dropbox shared folder as zip (this can be large; streaming to disk)")
    _download(g["dropbox_folder_url"], dest, timeout=timeout)
    return dest
