"""Turns a Week-2 feature table into per-branch torch tensors: column-median
imputation (genomic traits can have NaN rRNA/tRNA counts when barrnap/
tRNAscan-SE aren't installed, see features/genome_traits.py) followed by
z-score standardization, fit on the given table.
"""

import warnings

import numpy as np
import pandas as pd
import torch

from gem_worldmodel.utils.config import load_config

GENOMIC_TRAIT_COLUMNS = [
    "genome_size_bp", "gc_content", "rrna_16s_count", "rrna_23s_count",
    "rrna_5s_count", "trna_count", "regulatory_gene_count",
]


class BranchStandardizer:
    """Fits column median/std on one table, applies to any table with the same columns."""

    def __init__(self, columns: list[str]):
        self.columns = columns
        self.median_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "BranchStandardizer":
        values = df[self.columns].to_numpy(dtype=float)
        # An all-NaN column (e.g. rRNA/tRNA counts when barrnap/tRNAscan-SE
        # aren't installed, see features/genome_traits.py) is an expected,
        # explicitly-handled case here, not a bug, suppress numpy's warning
        # for it specifically and fall back to 0 for that column's median.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            self.median_ = np.nanmedian(values, axis=0)
        self.median_ = np.nan_to_num(self.median_, nan=0.0)
        filled = np.where(np.isnan(values), self.median_, values)
        self.std_ = filled.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        values = df[self.columns].to_numpy(dtype=float)
        filled = np.where(np.isnan(values), self.median_, values)
        return (filled - self.median_) / self.std_


def build_branch_tensors(
    df: pd.DataFrame,
    model_cfg: dict | None = None,
    standardizers: dict[str, BranchStandardizer] | None = None,
    branches: list[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, BranchStandardizer]]:
    """Build per-branch tensors. `branches` restricts to a subset of model_cfg's
    configured branches, e.g. a GEM-derived table without real 16S sequences
    only has genomic_traits/gtdb_distance columns, so trying to build a
    rrna16s tensor for it would fail (no rrna16s_* columns exist there).
    """
    model_cfg = model_cfg or load_config("model")
    fit_new = standardizers is None
    standardizers = standardizers or {}

    branch_columns = {
        "genomic_traits": GENOMIC_TRAIT_COLUMNS,
        "gtdb_distance": [c for c in df.columns if c.startswith("gtdb_dist_")],
        "rrna16s": [c for c in df.columns if c.startswith("rrna16s_")],
    }

    wanted = set(branches) if branches is not None else None
    tensors = {}
    for branch in model_cfg["branches"]:
        name = branch["name"]
        if wanted is not None and name not in wanted:
            continue
        cols = branch_columns[name]
        if fit_new:
            standardizers[name] = BranchStandardizer(cols).fit(df)
        arr = standardizers[name].transform(df)
        tensors[name] = torch.tensor(arr, dtype=torch.float32)
    return tensors, standardizers
