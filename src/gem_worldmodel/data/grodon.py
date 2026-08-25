"""Pull the gRodon/Madin labeled growth-rate corpus + lab-added species.

gRodon2's published training data (Madin et al. 2020, "A synthesis of
bacterial and archaeal genomes...") ships as R .rda files, not CSV. We fetch
the raw .rda files from GitHub and parse them with pyreadr (no R runtime
needed).
"""

from pathlib import Path

import pandas as pd
import requests

from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def _download(url: str, dest: Path, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"cached: {dest}")
        return dest
    logger.info(f"downloading {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def fetch_growth_rates(cfg: dict | None = None) -> pd.DataFrame:
    """Fetch the Madin et al. growth-rate table (species, d, OptTemp, GrowthTemp, Extremophile)."""
    import pyreadr

    cfg = cfg or load_config("data")
    g = cfg["grodon"]
    base = g["raw_base"].format(repo=g["repo"], branch=g["branch"])
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])

    rda_path = _download(f"{base}/{g['growth_rates_file']}", raw_dir / g["growth_rates_file"])
    result = pyreadr.read_r(str(rda_path))
    df = next(iter(result.values())).copy()
    df = df.rename(
        columns={
            "d": "doubling_time_hours",
            "OptTemp": "opt_temp_c",
            "GrowthTemp": "growth_temp_c",
            "Extremophile": "extremophile",
        }
    )
    return df


def fetch_accession_species_map(cfg: dict | None = None) -> pd.DataFrame:
    """Fetch the accession -> species name mapping used to join genomes to growth rates."""
    import pyreadr

    cfg = cfg or load_config("data")
    g = cfg["grodon"]
    base = g["raw_base"].format(repo=g["repo"], branch=g["branch"])
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])

    rda_path = _download(f"{base}/{g['accession_map_file']}", raw_dir / g["accession_map_file"])
    result = pyreadr.read_r(str(rda_path))
    df = next(iter(result.values())).copy()
    df.columns = ["accession", "species"]
    return df


def load_lab_added_species(cfg: dict | None = None) -> pd.DataFrame:
    """Load lab-added species ("Iyanu's data") if the CSV has been placed locally.

    Returns an empty frame with the expected schema if the file doesn't exist yet, this is an optional supplement, not a hard dependency of Week 1.
    """
    cfg = cfg or load_config("data")
    path = resolve_path(cfg["lab_added_species"]["local_csv"])
    columns = ["accession", "species", "doubling_time_hours", "opt_temp_c", "growth_temp_c"]
    if not path.exists():
        logger.warning(f"lab-added species file not found at {path}; continuing without it")
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"lab-added species CSV at {path} is missing columns: {missing}")
    return df


def build_labeled_corpus(cfg: dict | None = None) -> pd.DataFrame:
    """Join accession<->species<->growth-rate into one labeled-corpus table, plus lab-added rows."""
    cfg = cfg or load_config("data")
    growth = fetch_growth_rates(cfg)
    acc_map = fetch_accession_species_map(cfg)

    merged = acc_map.merge(growth, on="species", how="inner")
    merged["source"] = "grodon_madin"

    lab_added = load_lab_added_species(cfg)
    if not lab_added.empty:
        lab_added = lab_added.copy()
        lab_added["source"] = "lab_added"
        merged = pd.concat([merged, lab_added], ignore_index=True)

    merged = merged.drop_duplicates(subset=["accession"]).reset_index(drop=True)
    logger.info(f"labeled corpus: {len(merged)} accession-level rows across {merged['species'].nunique()} species")
    return merged
