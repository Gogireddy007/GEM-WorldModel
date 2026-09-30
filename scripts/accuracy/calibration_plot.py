#!/usr/bin/env python
"""This makes a calibration plot, predicted doubling time against known
doubling time, for the labeled corpus. Predictions are out-of-fold from the
same k-fold cross-validation used everywhere else in this project (raw
features plus gradient-boosted trees, the current best model), so every
point is a prediction for a species the model never saw during that fold's
training, not a fitted value.

Writes a per-species CSV (species, accession, known and predicted doubling
time, which fold held it out) alongside the PNG, so the plot's own numbers
can be checked directly rather than taken on faith.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.raw_baseline import concat_branch_tensors
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-file", type=str, default="features_sample_gapfilled.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out-prefix", type=str, default="calibration")
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])
    split_hours = data_cfg["doubling_time_split_hours"]
    all_branches = [b["name"] for b in model_cfg["branches"]]

    df = pd.read_csv(processed_dir / args.features_file)
    target_log = np.log(df["doubling_time_hours_ref"].to_numpy())
    y_true = df["doubling_time_hours_ref"].to_numpy()
    stratify_labels = (y_true < split_hours).astype(int)

    tensors, _ = build_branch_tensors(df, model_cfg, branches=all_branches)
    raw_x = concat_branch_tensors(tensors).numpy()

    n = len(df)
    oof_log = np.full(n, np.nan)
    fold_id = np.full(n, -1)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n), stratify_labels)):
        model = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3)
        model.fit(raw_x[train_idx], target_log[train_idx])
        oof_log[test_idx] = model.predict(raw_x[test_idx])
        fold_id[test_idx] = fold

    y_pred = np.exp(oof_log)

    out = pd.DataFrame({
        "species": df["species"],
        "accession": df["accession"],
        "known_doubling_time_hours": y_true,
        "predicted_doubling_time_hours": y_pred,
        "fold": fold_id,
        "regime": np.where(y_true < split_hours, "fast", "slow"),
    })
    csv_path = processed_dir / f"{args.out_prefix}_data.csv"
    out.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}: {len(out)} out-of-fold predictions")

    plt.rcParams.update({
        "font.family": "Helvetica",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fast = out["regime"] == "fast"
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(out.loc[~fast, "known_doubling_time_hours"], out.loc[~fast, "predicted_doubling_time_hours"],
               s=22, alpha=0.6, color="#3465a4", edgecolors="none", label=f"slow (>= {split_hours}h), n={(~fast).sum()}")
    ax.scatter(out.loc[fast, "known_doubling_time_hours"], out.loc[fast, "predicted_doubling_time_hours"],
               s=22, alpha=0.6, color="#e08214", edgecolors="none", label=f"fast (< {split_hours}h), n={fast.sum()}")

    lo = min(out["known_doubling_time_hours"].min(), out["predicted_doubling_time_hours"].min()) * 0.8
    hi = max(out["known_doubling_time_hours"].max(), out["predicted_doubling_time_hours"].max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color="#1a1a2e", linestyle="--", linewidth=1.2, label="perfect calibration")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Known doubling time (hours, log scale)")
    ax.set_ylabel("Predicted doubling time (hours, log scale)")
    ax.set_title(f"Predicted vs. known growth rate\n(out-of-fold, {args.k}-fold CV, n={len(out)}, seed={args.seed})",
                 fontsize=12, pad=12)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    png_path = processed_dir / f"{args.out_prefix}_plot.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    logger.info(f"wrote {png_path}")


if __name__ == "__main__":
    main()
