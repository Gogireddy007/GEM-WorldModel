"""Week 1 validation report: species count, growth-rate coverage, doubling-time
distribution above/below the 5h split, and temperature metadata coverage.
"""

import pandas as pd

from gem_worldmodel.utils.config import load_config
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def validate_labeled_corpus(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config("data")
    split = cfg["doubling_time_split_hours"]

    usable = df[df["in_gtdb_tree"]]
    n_fast = int((usable["doubling_time_hours"] < split).sum())
    n_slow = int((usable["doubling_time_hours"] >= split).sum())
    n_temp = int(usable["growth_temp_c"].notna().sum()) if "growth_temp_c" in usable else 0

    report = {
        "n_total_labeled_rows": len(df),
        "n_usable_rows": len(usable),
        "n_usable_species": usable["species"].nunique() if "species" in usable else None,
        "n_fast_below_split": n_fast,
        "n_slow_above_split": n_slow,
        "n_with_temperature_metadata": n_temp,
        "split_hours": split,
    }

    logger.info("=== Week 1 validation report ===")
    for k, v in report.items():
        logger.info(f"  {k}: {v}")

    if report["n_usable_rows"] < 50:
        logger.warning(
            "usable sample size is small (<50), check GTDB accession matching "
            "before proceeding to Week 2 feature engineering"
        )
    return report
