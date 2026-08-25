"""gRodon and Phydon baseline reproductions, refit on the same train/test
split as our model, for a fair comparison.

These are re-implementations of each method's *feature set and model class*,
refit on our split, not the original papers' fixed published coefficients,
which would (a) not be available to us outside their R packages and (b) not
be a fair comparison anyway since their coefficients were fit on a different,
overlapping sample. This is the standard way to benchmark against a prior
method when you don't have its exact trained artifact.

  - gRodon: log(doubling time) ~ CUB (their core single-feature predictor).
  - Phydon: gradient-boosted trees over the fuller genomic-trait feature set
    (CUB + genome size + GC + rRNA/tRNA counts + regulatory gene count),
    matching Phydon's use of multiple genomic features beyond CUB alone.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from gem_worldmodel.training.dataset import GENOMIC_TRAIT_COLUMNS


class GRodonBaseline:
    """gRodon genuinely cannot predict without a computable CUB, rows missing
    it (e.g. failed NCBI CDS download) are dropped at fit time and predicted
    as NaN, matching the real method's actual limitation rather than
    fabricating a value for it.
    """

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, df, target_col: str = "doubling_time_hours_ref"):
        valid = df["cub"].notna()
        x = df.loc[valid, ["cub"]].to_numpy(dtype=float)
        y = np.log(df.loc[valid, target_col].to_numpy(dtype=float))
        self.model.fit(x, y)
        return self

    def predict(self, df) -> np.ndarray:
        preds = np.full(len(df), np.nan)
        valid = df["cub"].notna().to_numpy()
        if valid.any():
            x = df.loc[valid, ["cub"]].to_numpy(dtype=float)
            preds[valid] = np.exp(self.model.predict(x))
        return preds


class PhydonBaseline:
    def __init__(self, seed: int = 0):
        self.model = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3)
        self.feature_cols = ["cub"] + GENOMIC_TRAIT_COLUMNS

    def fit(self, df, target_col: str = "doubling_time_hours_ref"):
        self.fill_values_ = df[self.feature_cols].median().fillna(0.0)
        x = df[self.feature_cols].fillna(self.fill_values_).to_numpy(dtype=float)
        y = np.log(df[target_col].to_numpy(dtype=float))
        self.model.fit(x, y)
        return self

    def predict(self, df) -> np.ndarray:
        x = df[self.feature_cols].fillna(self.fill_values_).to_numpy(dtype=float)
        return np.exp(self.model.predict(x))
