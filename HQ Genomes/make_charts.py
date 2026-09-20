"""Generate the chart set for the HQ-genome prediction report. Standalone,
reads only from data/processed and data/raw, writes PNGs into charts/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

INK = "#1a1a2e"
BLUE = "#3465a4"
TEAL = "#16a394"
AMBER = "#e08214"
GREY = "#8a8a8a"

pred = pd.read_csv("data/processed/unlabeled_predictions_hq.csv")
meta = pd.read_csv("data/raw/gem_genome_metadata.tsv", sep="\t", low_memory=False)
merged = pred.merge(meta[["genome_id", "ecosystem_category"]], on="genome_id", how="left")

# 1. Coverage: how much of the target set got a real prediction, and how
coverage = pd.Series(
    {
        "Full 3-branch\n(real 16S)": (pred["branches_used"] == "genomic_traits+gtdb_distance+rrna16s").sum(),
        "2-branch fallback\n(no 16S available)": (pred["branches_used"] == "genomic_traits+gtdb_distance").sum(),
        "Not covered\n(missing phylogeny)": 9143 - len(pred),
    }
)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
colors = [BLUE, TEAL, "#d8d8d8"]
bars = ax.bar(coverage.index, coverage.values, color=colors, width=0.6)
for b, v in zip(bars, coverage.values):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 60, f"{v:,}", ha="center", fontsize=12, color=INK, fontweight="bold")
ax.set_ylabel("Genomes")
ax.set_title("Coverage of the 9,143 high-quality GEM genomes", fontsize=13, color=INK, pad=14)
ax.set_ylim(0, 9143 * 1.12)
fig.tight_layout()
fig.savefig("HQ Genomes/charts/01_coverage.png", dpi=180)
plt.close(fig)

# 2. Distribution of predicted doubling times
fig, ax = plt.subplots(figsize=(7, 4.2))
vals = np.log10(pred["predicted_doubling_time_hours"])
ax.hist(vals, bins=60, color=BLUE, edgecolor="white", linewidth=0.3)
ax.set_xlabel("Predicted doubling time (hours, log scale)")
ax.set_ylabel("Number of genomes")
ticks = [0.1, 0.3, 1, 3, 10, 30]
ax.set_xticks(np.log10(ticks))
ax.set_xticklabels([str(t) for t in ticks])
ax.set_title("Predicted growth rate across 9,134 real GEM genomes", fontsize=13, color=INK, pad=14)
median = pred["predicted_doubling_time_hours"].median()
ax.axvline(np.log10(median), color=AMBER, linestyle="--", linewidth=1.5)
ax.text(np.log10(median), ax.get_ylim()[1] * 0.92, f"  median = {median:.2f}h", color=AMBER, fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig("HQ Genomes/charts/02_distribution.png", dpi=180)
plt.close(fig)

# 3. Predicted growth rate by ecosystem, the real sanity check
top_eco = merged["ecosystem_category"].value_counts().head(9).index
eco_medians = merged[merged["ecosystem_category"].isin(top_eco)].groupby("ecosystem_category")[
    "predicted_doubling_time_hours"
].median().sort_values()
fig, ax = plt.subplots(figsize=(7.5, 4.6))
bar_colors = [TEAL if v < eco_medians.median() else AMBER for v in eco_medians.values]
ax.barh(eco_medians.index, eco_medians.values, color=bar_colors)
for i, (name, v) in enumerate(eco_medians.items()):
    ax.text(v + 0.01, i, f"{v:.2f}h", va="center", fontsize=10, color=INK)
ax.set_xlabel("Median predicted doubling time (hours)")
ax.set_title("Predicted growth rate by habitat\n(model was never told the ecosystem)", fontsize=13, color=INK, pad=14)
fig.tight_layout()
fig.savefig("HQ Genomes/charts/03_by_ecosystem.png", dpi=180)
plt.close(fig)

# 4. GC content vs predicted doubling time
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(
    pred["gc_content"], pred["predicted_doubling_time_hours"], s=4, alpha=0.25, color=BLUE, linewidths=0
)
ax.set_yscale("log")
ax.set_xlabel("GC content")
ax.set_ylabel("Predicted doubling time (hours, log scale)")
ax.set_title("GC content vs. predicted growth rate", fontsize=13, color=INK, pad=14)
fig.tight_layout()
fig.savefig("HQ Genomes/charts/04_gc_vs_growth.png", dpi=180)
plt.close(fig)

print("wrote 4 charts to HQ Genomes/charts/")
print(coverage.to_string())
