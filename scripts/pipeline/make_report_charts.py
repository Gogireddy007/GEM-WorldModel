#!/usr/bin/env python
"""This makes the charts for the HQ genome report from the prediction table and
the measured accuracy files, so every figure traces back to data on disk.

Reads hq_predictions_gbm.csv, final_accuracy_by_tier.csv, mag_validation.csv and
the training table, and writes PNGs into the report folder's charts directory.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.models.deployed import DeployedRegressor
from gem_worldmodel.utils.config import load_config, resolve_path

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "figure.facecolor": "white", "axes.facecolor": "white",
})
INK, BLUE, TEAL, AMBER, GREY = "#1a1a2e", "#3465a4", "#16a394", "#e08214", "#8a8a8a"
TIERS = ["same genus", "same family, new genus", "same order, new family", "new order"]
FULL = "traits+phylogeny+16S+gene features"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "Test on HQ genomes/charts")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)
    print("wrote", OUT / name)


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    pred = pd.read_csv(processed / "hq_predictions_gbm.csv")
    acc = pd.read_csv(processed / "final_accuracy_by_tier.csv")
    lab, X, names = deploy_features.load_training_table(processed, k, with_cub=True)
    d = "doubling_time_hours_at_growth_temp"
    share = pred["nearest_labeled_relative"].value_counts(normalize=True) * 100

    # 1. accuracy by novelty
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    width = 0.38
    for ax, (col, label, fmt) in zip(axes, [("median_error", "Typical error (fold)", "{:.1f}x"), ("spearman", "Ranking accuracy (Spearman)", "{:.2f}")]):
        for j, (target, color, text) in enumerate([("own temperature", BLUE, "at own growth temperature"), ("20 C", TEAL, "scaled to 20 C")]):
            sub = acc[(acc.model == FULL) & (acc.target == target)].set_index("tier").loc[TIERS]
            bars = ax.bar(np.arange(4) + (j - 0.5) * width, sub[col], width, color=color, label=text)
            for b, v in zip(bars, sub[col]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(range(4))
        labels = [t.replace(", ", ",\n") + "\n(" + f"{share.get(t, 0):.0f}" + "% of genomes)" for t in TIERS]
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(label)
        if col == "median_error":
            ax.set_ylim(1, None)
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle("Accuracy when a whole lineage is held out, by how close the nearest labeled relative is", fontsize=12, color=INK)
    save(fig, "01_accuracy_by_novelty.png")

    # 2. distribution of predictions
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.linspace(np.log10(0.2), np.log10(1500), 60)
    ax.hist(np.log10(lab["doubling_time_hours"]), bins=bins, density=True, color=GREY, alpha=0.55, label=f"measured, {len(lab)} labeled species")
    ax.hist(np.log10(pred[d]), bins=bins, density=True, color=BLUE, alpha=0.65, label=f"predicted, {len(pred):,} GEM genomes")
    ticks = [0.3, 1, 3, 10, 30, 100, 300, 1000]
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("Doubling time at the organism's own growth temperature (hours, log scale)")
    ax.set_ylabel("Share of genomes (density)")
    ax.axvline(np.log10(pred[d].median()), color=AMBER, linestyle="--", linewidth=1.4)
    ax.text(np.log10(pred[d].median()), ax.get_ylim()[1] * 0.93, f"  predicted median {pred[d].median():.1f} h", color=AMBER, fontsize=10)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    save(fig, "02_distribution.png")

    # 3. predicted growth temperature
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    arch = pred["gtdb_taxonomy"].str.startswith("d__Archaea")
    bins = np.arange(0, 105, 3)
    ax.hist(pred.loc[~arch, "predicted_growth_temp_c"], bins=bins, color=BLUE, alpha=0.75, label=f"bacteria ({(~arch).sum():,})")
    ax.hist(pred.loc[arch, "predicted_growth_temp_c"], bins=bins, color=AMBER, alpha=0.8, label=f"archaea ({arch.sum()})")
    ax.axvline(45, color=INK, linestyle="--", linewidth=1)
    ax.text(46, ax.get_ylim()[1] * 0.9, f"45 C: {100 * pred.hot_organism.mean():.1f}% of genomes above", fontsize=10)
    ax.set_xlabel("Predicted growth temperature (C)")
    ax.set_ylabel("Genomes")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    save(fig, "03_growth_temperature.png")

    # 4. by habitat
    g = pred.groupby("ecosystem_category")[d].agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), n="size")
    g = g[g.n >= 100].sort_values("median")
    fig, ax = plt.subplots(figsize=(7.6, 0.45 * len(g) + 1.6))
    ax.barh(g.index, g["median"], color=TEAL, xerr=[g["median"] - g.q25, g.q75 - g["median"]], error_kw={"ecolor": GREY, "capsize": 3})
    for i, (m, n) in enumerate(zip(g["median"], g.n)):
        ax.text(g.q75.iloc[i] + 0.5, i, f"{m:.1f} h  (n={n:,})", va="center", fontsize=9)
    ax.set_xlabel("Median predicted doubling time (hours); bars show the middle half of genomes")
    ax.set_title("Predicted growth by habitat (the model was never told the habitat)", fontsize=12, color=INK)
    save(fig, "04_by_habitat.png")

    # 5. calibration on labeled species, species held out
    use = lab["has_cub"].to_numpy()
    lab_rows, Xl = lab[use].reset_index(drop=True), X[use]
    y = np.log(lab_rows["doubling_time_hours"].to_numpy())
    oof = np.mean([_oof(Xl, y, lab_rows["species_group"].to_numpy(), s) for s in (42, 1, 7)], axis=0)
    pd.DataFrame({"species": lab_rows["species"], "archaeon": lab_rows["archaeon"].astype(int), "measured_hours": np.exp(y), "predicted_hours": np.exp(oof)}).to_csv(
        OUT.parent / "calibration_data_labeled_species.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    isarch = lab_rows["archaeon"].to_numpy() == 1
    ax.scatter(np.exp(y[~isarch]), np.exp(oof[~isarch]), s=16, alpha=0.55, color=BLUE, edgecolors="none", label=f"bacteria ({(~isarch).sum()})")
    ax.scatter(np.exp(y[isarch]), np.exp(oof[isarch]), s=16, alpha=0.7, color=AMBER, edgecolors="none", label=f"archaea ({isarch.sum()})")
    lo, hi = 0.15, 3000
    ax.plot([lo, hi], [lo, hi], color=INK, linestyle="--", linewidth=1.1, label="perfect")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Measured doubling time (hours)"); ax.set_ylabel("Predicted doubling time (hours)")
    err = np.exp(np.median(np.abs(y - oof)))
    ax.set_title(f"Held-out predictions for {len(lab_rows)} labeled genomes from {lab_rows['species'].nunique()} species\ntypical error {err:.1f}x, Spearman {spearmanr(y, oof).correlation:.2f}", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    save(fig, "05_calibration.png")

    # 6. real MAGs
    mag = pd.read_csv(processed / "mag_validation.csv")
    mag = mag[(mag.target == "own temperature") & (mag.variant == "without 16S")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    for ax, (col, title) in zip(axes, [("isolate", "predicted from the isolate genome"), ("mag", "predicted from GEM MAGs of the species")]):
        ax.scatter(np.exp(mag.truth), np.exp(mag[col]), s=26, color=BLUE if col == "isolate" else TEAL, alpha=0.8, edgecolors="none")
        lo, hi = 0.2, 500
        ax.plot([lo, hi], [lo, hi], color=INK, linestyle="--", linewidth=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        e = np.exp(np.median(np.abs(mag.truth - mag[col])))
        ax.set_title(f"{title}\ntypical error {e:.1f}x, Spearman {spearmanr(mag.truth, mag[col]).correlation:.2f}", fontsize=11, color=INK)
        ax.set_xlabel("Measured doubling time (hours)")
    axes[0].set_ylabel("Predicted doubling time (hours)")
    fig.suptitle(f"{len(mag)} species with a measured growth rate and {int(mag.n_mags.sum())} GEM genomes, each species held out of training", fontsize=11, color=INK)
    save(fig, "06_real_mags.png")


def _oof(X, y, groups, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug))))
    g = np.array([perm[v] for v in groups])
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = DeployedRegressor("own temperature", seed).fit(X[tr], y[tr]).predict(X[te])
    return pred


if __name__ == "__main__":
    main()
