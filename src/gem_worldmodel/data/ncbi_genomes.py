"""Fetch NCBI RefSeq/GenBank genome assemblies (full genomic FASTA + annotated
CDS FASTA) by accession, for feature extraction.

Uses NCBI Entrez esearch/esummary (db=assembly) to resolve an accession to its
FTP directory, then downloads the standard `_genomic.fna.gz` (used for
genome size/GC/rRNA-tRNA/gene-calling fallback) and `_cds_from_genomic.fna.gz`
(annotated CDS, used for CUB) files from that directory.
"""

import gzip
import shutil
import time
from pathlib import Path

import requests

from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def resolve_ftp_dir(accession: str, retries: int = 3, sleep_s: float = 0.4) -> str | None:
    """Resolve an assembly accession (e.g. GCF_000005845.2) to its NCBI FTP/HTTPS directory URL."""
    for attempt in range(retries):
        try:
            r = requests.get(
                ESEARCH, params={"db": "assembly", "term": accession, "retmode": "json"}, timeout=20
            )
            r.raise_for_status()
            ids = r.json()["esearchresult"]["idlist"]
            if not ids:
                return None
            r2 = requests.get(
                ESUMMARY, params={"db": "assembly", "id": ids[0], "retmode": "json"}, timeout=20
            )
            r2.raise_for_status()
            result = r2.json()["result"][ids[0]]
            ftp_path = result.get("ftppath_refseq") or result.get("ftppath_genbank")
            time.sleep(sleep_s)
            if not ftp_path:
                return None
            return ftp_path.replace("ftp://", "https://")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"NCBI lookup failed for {accession} (attempt {attempt + 1}): {exc}")
            time.sleep(sleep_s)
    return None


def _download_gz(url: str, dest_gz: Path, decompress_to: Path) -> Path | None:
    if decompress_to.exists():
        return decompress_to
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code != 200:
            return None
        dest_gz.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_gz, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        with gzip.open(dest_gz, "rb") as f_in, open(decompress_to, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return decompress_to
    except (requests.RequestException, OSError, gzip.BadGzipFile) as exc:
        logger.warning(f"download/decompress failed for {url}: {exc}")
        return None


def fetch_assembly_files(accession: str, cfg: dict | None = None) -> dict[str, Path | None]:
    """Download the genomic and CDS FASTA for one accession. Returns paths (or
    None per-file if that file couldn't be resolved/downloaded)."""
    cfg = cfg or load_config("data")
    raw_dir = resolve_path(cfg["paths"]["raw_dir"]) / "ncbi_genomes" / accession

    ftp_dir = resolve_ftp_dir(accession)
    if ftp_dir is None:
        logger.warning(f"could not resolve FTP directory for {accession}")
        return {"genomic_fna": None, "cds_fna": None}

    basename = ftp_dir.rstrip("/").split("/")[-1]
    genomic = _download_gz(
        f"{ftp_dir}/{basename}_genomic.fna.gz",
        raw_dir / f"{basename}_genomic.fna.gz",
        raw_dir / f"{basename}_genomic.fna",
    )
    cds = _download_gz(
        f"{ftp_dir}/{basename}_cds_from_genomic.fna.gz",
        raw_dir / f"{basename}_cds_from_genomic.fna.gz",
        raw_dir / f"{basename}_cds_from_genomic.fna",
    )
    return {"genomic_fna": genomic, "cds_fna": cds}
