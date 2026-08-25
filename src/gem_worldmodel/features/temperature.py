"""Arrhenius temperature correction: raw growth rate -> reference-temperature target.

Confirmed by the plan: this produces the *target*, never a model input.

mu_ref = mu_raw * exp( -(Ea/R) * (1/T_ref - 1/T_obs) )

where T is in Kelvin. Doubling time d [hours] and growth rate mu [1/hour]
are related by mu = ln(2) / d, so we convert, correct in rate-space, and
convert back to a corrected doubling time.
"""

import numpy as np
import pandas as pd

from gem_worldmodel.utils.config import load_config

LN2 = np.log(2.0)


def celsius_to_kelvin(temp_c: float) -> float:
    return temp_c + 273.15


def arrhenius_correct_rate(
    mu_raw: float | np.ndarray,
    t_obs_c: float | np.ndarray,
    cfg: dict | None = None,
) -> float | np.ndarray:
    """Correct a raw growth rate (1/hour) observed at t_obs_c to the reference temperature."""
    cfg = cfg or load_config("features")
    t = cfg["temperature"]
    ea = t["activation_energy_j_per_mol"]
    r = t["gas_constant_j_per_mol_k"]
    t_ref_k = t["reference_temp_k"]

    t_obs_k = celsius_to_kelvin(np.asarray(t_obs_c, dtype=float))
    exponent = -(ea / r) * (1.0 / t_ref_k - 1.0 / t_obs_k)
    return np.asarray(mu_raw, dtype=float) * np.exp(exponent)


def correct_doubling_time(
    doubling_time_hours: float | np.ndarray,
    t_obs_c: float | np.ndarray,
    cfg: dict | None = None,
) -> float | np.ndarray:
    """Correct a raw doubling time to the reference temperature, returned in hours."""
    mu_raw = LN2 / np.asarray(doubling_time_hours, dtype=float)
    mu_ref = arrhenius_correct_rate(mu_raw, t_obs_c, cfg)
    return LN2 / mu_ref


def add_reference_temperature_target(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add a `doubling_time_hours_ref` column: the Arrhenius-corrected target.

    Rows lacking growth-temperature metadata keep the raw doubling time
    unchanged (no correction applied) and are flagged via `temp_corrected`.
    """
    cfg = cfg or load_config("features")
    df = df.copy()
    has_temp = df["growth_temp_c"].notna() if "growth_temp_c" in df else pd.Series(False, index=df.index)

    df["doubling_time_hours_ref"] = df["doubling_time_hours"]
    if has_temp.any():
        corrected = correct_doubling_time(
            df.loc[has_temp, "doubling_time_hours"].to_numpy(),
            df.loc[has_temp, "growth_temp_c"].to_numpy(),
            cfg,
        )
        df.loc[has_temp, "doubling_time_hours_ref"] = corrected
    df["temp_corrected"] = has_temp
    return df
