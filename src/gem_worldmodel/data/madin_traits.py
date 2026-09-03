"""Pull the real, independently-curated bacterial/archaeal phenotype trait
database from Madin et al. 2020 ("A synthesis of bacterial and archaeal
phenotypic trait data", Scientific Data), via its companion GitHub repo
(bacteria-archaea-traits/bacteria-archaea-traits).

This is a different, much richer dataset than the growth-rate-only table
already pulled in data/grodon.py: 14,893 species with 79 real trait columns
(metabolism, isolation_source, carbon_substrates, gram_stain, motility,
sporulation, cell_shape, etc.), assembled from published physiological
records, not computed from the genome sequence at all. That independence is
exactly what's needed for a real oligotroph/copiotroph label, see
features/ecological_traits.py, since a label derived from genome composition
(like the old heuristic in eval/probing.py) risks confirming CUB-correlated
structure circularly rather than discovering it.
"""

import pandas as pd
import requests

from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

TRAITS_URL = (
    "https://raw.githubusercontent.com/bacteria-archaea-traits/"
    "bacteria-archaea-traits/master/output/condensed_species_NCBI.csv"
)


def fetch_madin_traits(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config("data")
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    dest = raw_dir / "condensed_species_NCBI.csv"
    if not dest.exists():
        logger.info(f"downloading {TRAITS_URL}")
        resp = requests.get(TRAITS_URL, timeout=60)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    else:
        logger.info(f"cached: {dest}")
    return pd.read_csv(dest, low_memory=False)
