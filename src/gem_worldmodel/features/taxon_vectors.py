"""Look up a phylogeny feature vector for any genome from its GTDB taxonomy string.

The vectors come from scripts/accuracy/build_gtdb_landmark_features.py: the
mean distance-to-landmark vector of the GTDB tree tips in a species, genus,
family or order. Lookup tries each rank from most to least specific and
returns the first match along with the rank that matched, so a species-level
placement and a family-level approximation are never treated as the same
quality of evidence.
"""

import re

import numpy as np
import pandas as pd

RANKS = (("species", "s__"), ("genus", "g__"), ("family", "f__"), ("order", "o__"))
VEC_COLS = [f"lm_{i}" for i in range(16)]


def load_taxon_table(path) -> dict[tuple[str, str], np.ndarray]:
    table = pd.read_csv(path)
    return {
        (row["rank"], row["taxon"]): row[VEC_COLS].to_numpy(dtype=float)
        for _, row in table.iterrows()
    }


def _token(taxonomy: str, prefix: str):
    m = re.search(rf"{prefix}[^;]*", str(taxonomy))
    return m.group(0) if m and m.group(0) != prefix else None


def lookup(taxonomy: str, table: dict, max_rank: str = "family"):
    """Return (vector, rank_used), or (None, None) if nothing matches down to max_rank."""
    allowed = [r for r in RANKS if RANKS.index(r) <= [x[0] for x in RANKS].index(max_rank)]
    for rank, prefix in allowed:
        token = _token(taxonomy, prefix)
        if token is not None and (rank, token) in table:
            return table[(rank, token)], rank
    return None, None


def is_archaeon(taxonomy) -> float:
    """1.0 for an archaeal GTDB taxonomy string, else 0.0.

    Bacteria and archaea have separate landmark vectors from separate trees, so
    the same column means different things for the two. The model needs this
    flag to tell them apart.
    """
    return 1.0 if str(taxonomy).startswith("d__Archaea") else 0.0
