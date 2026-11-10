#!/usr/bin/env python
"""Draws every data figure of the paper from the result files in data/processed.

Run from the repository root: .venv/bin/python paper/scripts/figures.py [--only fig3 ...]
Writes paper/figures/<name>.pdf. A figure whose result file is missing is skipped with a message,
never drawn from stale or typed-in numbers.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
OUT = ROOT / "paper" / "figures"

# categorical slots 1 to 3 of the reference palette (validated all-pairs for colour vision deficiency);
# every series also differs by marker, and slot 3 is always labelled, because it is below 3:1 contrast
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"
TIERS = ["species", "genus", "family", "order"]
TIER_LABEL = ["same genus", "new genus", "new family", "new order"]
SINGLE, DOUBLE = 3.35, 7.0  # inches, Bioinformatics column widths

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
    "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.4, "lines.markersize": 4,
})


def panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="right", va="bottom")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf")


def need(*files):
    missing = [f for f in files if not (P / f).exists()]
    if missing:
        print(f"skipped: missing {', '.join(missing)}")
    return not missing


def fig_accuracy():
    """Figure 2: accuracy by lineage novelty, nested, for the three targets."""
    if not need("paper_headline_accuracy.csv"):
        return
    a = pd.read_csv(P / "paper_headline_accuracy.csv")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.3), constrained_layout=True)
    x = np.arange(len(TIERS))
    specs = [("own temperature", "typical error", BLUE, "o", "own temperature"), ("20 C", "typical error", ORANGE, "s", "at 20 °C")]
    for ax, metric, ylabel in ((axes[0], "typical error", "typical fold error (×)"), (axes[1], "Spearman", "Spearman correlation")):
        for target, _, color, marker, label in specs:
            d = a[(a["target"] == target) & (a["scheme"] == "nested")]
            g = d.groupby("held_out")[metric].agg(["mean", "min", "max"]).reindex(TIERS)
            ax.plot(x, g["mean"], color=color, marker=marker, label=f"doubling time, {label}")
            ax.vlines(x, g["min"], g["max"], color=color, linewidth=0.9)
        f = a[(a["target"] == "own temperature") & (a["scheme"] == "fixed, single")].groupby("held_out")[metric].mean().reindex(TIERS)
        ax.plot(x, f, color=GREY, linestyle="--", marker="o", markerfacecolor="white", linewidth=1.0,
                label="own temperature, settings tuned on all rows, trained once per genome")
        if (P / "pressure_tests.csv").exists():
            pt = pd.read_csv(P / "pressure_tests.csv")
            b = pt[(pt["test"] == "baseline") & (pt["method"] == "relative median")].groupby("held_out")[metric].mean().reindex(TIERS)
            ax.plot(x, b, color=INK2, linestyle=":", marker="x", linewidth=1.0,
                    label="own temperature, median of the closest labeled relatives")
        ax.set_xticks(x, TIER_LABEL)
        ax.set_ylabel(ylabel)
    # one legend above the panels, so that it covers no data
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, handlelength=2.2)
    axes[1].set_ylim(top=1.0)
    d = a[(a["target"] == "growth temperature") & (a["scheme"] == "nested")]
    g = d.groupby("held_out")["mean abs error"].agg(["mean", "min", "max"]).reindex(TIERS)
    axes[2].plot(x, g["mean"], color=AQUA, marker="D", label="growth temperature")
    axes[2].vlines(x, g["min"], g["max"], color=AQUA, linewidth=0.9)
    for xi, v in zip(x, g["mean"], strict=True):
        axes[2].annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6, color=INK2)
    axes[2].set_xticks(x, TIER_LABEL)
    axes[2].set_ylabel("growth temperature, mean error (°C)")
    for ax, letter in zip(axes, "abc", strict=True):
        panel(ax, letter)
        ax.set_xlabel("closest labeled relative in training")
    axes[2].set_ylim(0, 1.2 * g["max"].max())
    save(fig, "fig2_accuracy")


def fig_released():
    """Figure 3: the released gRodon and Phydon on the same genomes and folds."""
    # our model built inside each split, as in the text (compare_released_tools.py --nested)
    if not need("released_tools_comparison_nested.csv"):
        return
    r = pd.read_csv(P / "released_tools_comparison_nested.csv")
    r = r[r["subset"] == "all genomes"]
    rows = [("without the measured temperature", "this model", "this model", BLUE, "o"),
            ("without the measured temperature", "gRodon (released), no temperature", "gRodon", ORANGE, "s"),
            ("given the measured temperature", "this model + measured temperature", "this model", BLUE, "o"),
            ("given the measured temperature", "gRodon (released)", "gRodon", ORANGE, "s"),
            ("given the measured temperature", "Phydon (released)", "Phydon", AQUA, "D")]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.1), sharey=True, constrained_layout=True)
    ypos = [5.0, 4.0, 2.0, 1.0, 0.0]
    for ax, metric, xlabel in zip(axes, ("typical error", "Spearman", "AUC slow"),
                                  ("typical fold error (×), lower is better", "Spearman correlation", "AUC, slower than a day"), strict=True):
        for y0, (_, method, _, color, marker) in zip(ypos, rows, strict=True):
            v = r.loc[r["method"] == method, metric]
            ax.scatter(v, np.full(len(v), y0), s=10, color=color, alpha=0.45, linewidths=0, marker=marker)
            ax.scatter([v.mean()], [y0], s=34, color=color, marker=marker, edgecolors="white", linewidths=0.8, zorder=3)
            ax.annotate(f"{v.mean():.2f}", (v.mean(), y0), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6, color=INK2)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25)
    axes[0].set_yticks(ypos, [name for _, _, name, _, _ in rows])
    axes[0].set_ylim(-0.6, 6.1)
    for ax in axes:
        ax.axhline(3.1, color=GRID, linewidth=0.8)
    for y0, text in ((5.55, "without the measured growth temperature"), (2.55, "given the measured growth temperature")):
        axes[0].text(0.02, y0, text, transform=axes[0].get_yaxis_transform(), fontsize=6.5, color=INK2, style="italic", va="bottom")
    for ax, letter in zip(axes, "abc", strict=True):
        panel(ax, letter)
    save(fig, "fig3_released_tools")


def fig_mags_gem():
    """Figure 4: real MAGs against isolates of the same species, and the GEM predictions by habitat."""
    if not need("mag_validation.csv", "hq_predictions_gbm.csv"):
        return
    m = pd.read_csv(P / "mag_validation.csv")
    m = m[(m["target"] == "own temperature") & (m["variant"] == "without 16S")]
    h = pd.read_csv(P / "hq_predictions_gbm.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(DOUBLE, 2.5), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.25, 0.6]})
    hours = np.exp
    lim = [hours(min(m[["truth", "isolate", "mag"]].min())) / 1.5, hours(max(m[["truth", "isolate", "mag"]].max())) * 1.5]
    a.plot(lim, lim, color=GREY, linewidth=0.8, linestyle="--")
    for xi, yi, yj in zip(hours(m["truth"]), hours(m["isolate"]), hours(m["mag"]), strict=True):
        a.plot([xi, xi], [yi, yj], color=GRID, linewidth=0.7, zorder=1)
    a.scatter(hours(m["truth"]), hours(m["isolate"]), s=13, color=GREY, marker="o", facecolors="white", linewidths=0.8, label="isolate genome", zorder=2)
    a.scatter(hours(m["truth"]), hours(m["mag"]), s=13, color=BLUE, marker="o", linewidths=0, label="metagenome-assembled genomes (median)", zorder=3)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(lim)
    a.set_ylim(lim)
    ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30, 100) if lim[0] <= t <= lim[1]]
    a.set_xticks(ticks, [f"{t:g}" for t in ticks])
    a.set_yticks(ticks, [f"{t:g}" for t in ticks])
    a.minorticks_off()
    a.set_xlabel("measured doubling time (h)")
    a.set_ylabel("predicted doubling time (h)")
    a.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    panel(a, "a")

    top = h["ecosystem_category"].value_counts()
    top = top[top >= 100].index.tolist()
    tiers = ["same genus", "same family, new genus", "same order, new family", "new order"]
    colors = {"same genus": "#1f5aa3", "same family, new genus": BLUE, "same order, new family": "#7fb0ea", "new order": "#c4dcf6"}
    data = [np.log10(h.loc[h["ecosystem_category"] == e, "doubling_time_hours_at_growth_temp"]) for e in top]
    bp = b.boxplot(data, vert=False, widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True, medianprops={"color": INK, "linewidth": 1.0},
                   whiskerprops={"color": INK2, "linewidth": 0.7}, capprops={"color": INK2, "linewidth": 0.7},
                   boxprops={"facecolor": "#e9f1fb", "edgecolor": INK2, "linewidth": 0.7})
    del bp
    b.set_yticks(np.arange(1, len(top) + 1), [f"{e} ({(h['ecosystem_category'] == e).sum():,})" for e in top])
    b.invert_yaxis()
    ticks = [0.5, 1, 2, 5, 10, 24]
    b.set_xticks(np.log10(ticks), [f"{t:g}" for t in ticks])
    b.axvline(np.log10(24), color=GREY, linewidth=0.8, linestyle="--")
    b.set_xlabel("predicted doubling time at the growth temperature (h)")
    b.grid(axis="y", visible=False)
    share = h["nearest_labeled_relative"].value_counts(normalize=True).reindex(tiers) * 100
    c.barh(np.arange(4), share.to_numpy(), color=[colors[t] for t in tiers], height=0.62)
    c.set_yticks(np.arange(4), ["same genus", "new genus", "new family", "new order"])
    c.invert_yaxis()
    c.set_xlabel("% of GEM genomes")
    c.grid(axis="y", visible=False)
    for i, v in enumerate(share):
        c.text(v + 1, i, f"{v:.0f}", va="center", fontsize=6, color=INK2)
    c.set_xlim(0, share.max() * 1.3)
    panel(c, "c")
    panel(b, "b")
    save(fig, "fig4_mags_gem")


def fig_lifestyle():
    """Figure 5: gene content and doubling time across species and with relatedness controlled."""
    if not need("lifestyle_phylogenetic_test.csv"):
        return
    t = pd.read_csv(P / "lifestyle_phylogenetic_test.csv")
    t = t[t["analysis"] == "all labeled species"].copy()
    t = t.sort_values("fold_pgls").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(SINGLE, 4.6), constrained_layout=True)
    y = np.arange(len(t))
    ax.axvline(1.0, color=GREY, linewidth=0.8)
    ax.hlines(y, t["ci_low_pgls"], t["ci_high_pgls"], color=BLUE, linewidth=1.0)
    ax.scatter(t["fold_across_species"], y, s=14, marker="o", facecolors="white", edgecolors=ORANGE, linewidths=0.9,
               label="across species, no control", zorder=3)
    ax.scatter(t["fold_pgls"], y, s=14, marker="s", color=BLUE, label="PGLS, relatedness controlled (95% CI)", zorder=4)
    labels = []
    for r in t.itertuples():
        name = r.label.split(" (")[0].replace("Na+", "Na$^+$")
        if r.input in ("transporters_per_1000", "regulators_per_1000", "oxidase_count", "hydrogenase_count",
                       "electron_acceptor_classes", "sodium_pumps"):
            name += " (per SD)"
        if r.input == "log2_genome_size":
            name = "genome size (per doubling)"
        labels.append(name + (" *" if r.p_pgls_bh < 0.05 else ""))
    ax.set_yticks(y, labels, fontsize=6)
    ax.set_xscale("log")
    ticks = [0.25, 0.5, 1, 2, 4, 8]
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    ax.set_xlim(0.2, 14)
    ax.set_ylim(-0.8, len(t) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("fold change in doubling time (above 1: slower)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1, borderaxespad=0.2)
    save(fig, "fig5_lifestyle")


FIGURES = {"fig2": fig_accuracy, "fig3": fig_released, "fig4": fig_mags_gem, "fig5": fig_lifestyle}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES))
    args = ap.parse_args()
    for name in args.only or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main()
