"""Benchmark our model vs. gRodon & Phydon, stratified above/below the
5h doubling-time split. Metrics: RMSE, R^2, Spearman correlation.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from gem_worldmodel.utils.config import load_config
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearmanr(y_true, y_pred).correlation),
        "n": int(len(y_true)),
    }


def stratified_benchmark(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    split_hours: float | None = None,
) -> pd.DataFrame:
    """predictions: {model_name: predicted_doubling_time_array}. Returns a tidy
    table of metrics per (model, regime), regime in {all, fast (<split), slow (>=split)}.
    """
    split_hours = split_hours or load_config("train")["benchmark"]["doubling_time_split_hours"]
    fast_mask = y_true < split_hours
    slow_mask = ~fast_mask

    rows = []
    for model_name, y_pred in predictions.items():
        # A model like gRodon genuinely can't predict for every row (e.g. no
        # computable CUB), its own NaN predictions are excluded from its
        # metrics rather than crashing sklearn or silently corrupting them.
        valid_pred = ~np.isnan(y_pred)
        n_dropped = (~valid_pred).sum()
        if n_dropped:
            logger.info(f"{model_name}: excluding {n_dropped} rows with no prediction from its own metrics")

        for regime, mask in [("all", np.ones_like(y_true, dtype=bool)), ("fast_<5h", fast_mask), ("slow_>=5h", slow_mask)]:
            combined_mask = mask & valid_pred
            if combined_mask.sum() < 2:
                logger.warning(f"skipping {model_name}/{regime}: fewer than 2 valid predictions")
                continue
            metrics = compute_metrics(y_true[combined_mask], y_pred[combined_mask])
            rows.append({"model": model_name, "regime": regime, **metrics})
    return pd.DataFrame(rows)
