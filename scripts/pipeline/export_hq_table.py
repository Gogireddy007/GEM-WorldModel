#!/usr/bin/env python
"""This turns the GEM prediction output (hq_predictions_gbm.csv) into the table released with the paper,
release/gem_growth_predictions.csv, and its column description, release/COLUMNS.md.

Each of the three predictions gets the typical error measured for its model and novelty level (final_accuracy_by_tier.csv)
and an 80% range. For genomes predicted with the full model the range comes from the nested held-out residuals for the
genome's novelty level and domain (range_calibration.csv, calibrate_ranges.py), which holds for archaea as well as bacteria;
for genomes that lack an input block it comes from that model's held-out residuals at the novelty level. The probability of
a doubling time above 24 h is the cross-validated logistic model of calibrate_ranges.py. Rows are sorted by genome
identifier. Run after `make predict` and calibrate_ranges.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gem_worldmodel.utils.config import load_config, resolve_path

OUT = Path("release")

# column, description; the order is the order in the file
COLUMNS = [
    ("genome_id", "GEM genome identifier (Nayfach et al. 2021)"),
    ("gtdb_taxonomy", "GTDB taxonomy from the GEM metadata (an older GTDB release than the training data, so the novelty level is a worst case)"),
    ("nearest_labeled_relative", "closest rank shared with a labeled training species: same genus, same family, new genus, "
                                 "same order, new family, or new order"),
    ("doubling_time_h", "predicted maximal doubling time at the predicted growth temperature, hours"),
    ("doubling_time_low_h", "lower end of the 80% range, hours"),
    ("doubling_time_high_h", "upper end of the 80% range, hours"),
    ("doubling_time_fold_error", "typical fold error at this novelty level (median of measured/predicted or its inverse)"),
    ("doubling_time_20C_h", "predicted doubling time standardized to 20 C, hours"),
    ("doubling_time_20C_low_h", "lower end of the 80% range, hours"),
    ("doubling_time_20C_high_h", "upper end of the 80% range, hours"),
    ("doubling_time_20C_fold_error", "typical fold error at this novelty level"),
    ("growth_temp_c", "predicted optimal growth temperature, C"),
    ("growth_temp_low_c", "lower end of the 80% range, C"),
    ("growth_temp_high_c", "upper end of the 80% range, C"),
    ("growth_temp_error_c", "mean absolute error at this novelty level, C"),
    ("slower_than_a_day", "predicted doubling time above 24 hours"),
    ("prob_slower_than_a_day", "calibrated probability that the doubling time is above 24 hours (useful for ranking; the model "
                               "rarely predicts above a day for a new lineage)"),
    ("model_used", "inputs available for this genome, and so the model used"),
    ("domain", "bacteria or archaea"),
    ("phylogeny_rank_used", "GTDB rank at which the genome was placed for the phylogeny input; empty when the genome could not be placed"),
    ("completeness", "CheckM completeness, % (GEM metadata)"),
    ("contamination", "CheckM contamination, % (GEM metadata)"),
    ("ecosystem_category", "GEM metadata"),
    ("ecosystem_type", "GEM metadata"),
    ("habitat", "GEM metadata"),
]

# prediction-output column for each target; the doubling times are on a log scale in the accuracy table
FULL = "traits+phylogeny+16S+gene features"
LEVEL = {"same genus": "species", "same family, new genus": "genus", "same order, new family": "family", "new order": "order"}
TARGETS = {"own temperature": ("doubling_time_h", "doubling_time_hours_at_growth_temp"),
           "20 C": ("doubling_time_20C_h", "doubling_time_hours_at_20C"),
           "growth temperature": ("growth_temp_c", "predicted_growth_temp_c")}


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    p = pd.read_csv(processed / "hq_predictions_gbm.csv")
    acc = pd.read_csv(processed / "final_accuracy_by_tier.csv").set_index(["model", "target", "tier"])
    meta = pd.read_csv(resolve_path("data/raw/gem_genome_metadata.tsv"), sep="\t", low_memory=False)[
        ["genome_id", "completeness", "contamination", "ecosystem_type", "habitat"]]
    out = p.merge(meta, on="genome_id", how="left", validate="one_to_one")
    out["domain"] = np.where(out["gtdb_taxonomy"].astype(str).str.startswith("d__Archaea"), "archaea", "bacteria")
    cal_path = processed / "range_calibration.csv"
    # before the paper experiments have run (make predict), the ranges are by novelty level only and there is no probability
    cal = pd.read_csv(cal_path) if cal_path.exists() else None
    ranges = cal[cal["kind"] == "range"].set_index(["target", "held_out", "domain"]) if cal is not None else None
    full = (out["model_used"] == FULL).to_numpy() & (cal is not None)
    level = out["nearest_labeled_relative"].map(LEVEL)

    for target, (name, col) in TARGETS.items():
        q = acc.loc[list(zip(out["model_used"], [target] * len(out), out["nearest_labeled_relative"]))]
        lo, hi, err = q["resid_q10"].to_numpy(), q["resid_q90"].to_numpy(), q["median_error"].to_numpy()
        if ranges is not None:
            r = ranges.loc[list(zip([target] * len(out), level, out["domain"]))]
            lo = np.where(full, r["q10"].to_numpy(), lo)
            hi = np.where(full, r["q90"].to_numpy(), hi)
        value = out[col].to_numpy(float)
        if target == "growth temperature":
            out[name], out["growth_temp_low_c"], out["growth_temp_high_c"] = value.round(1), (value + lo).round(1), (value + hi).round(1)
            out["growth_temp_error_c"] = err.round(1)
        else:
            out[name] = value.round(3)
            out[name.replace("_h", "_low_h")] = np.exp(np.log(value) + lo).round(3)
            out[name.replace("_h", "_high_h")] = np.exp(np.log(value) + hi).round(3)
            out[name.replace("_h", "_fold_error")] = err.round(2)
    out["slower_than_a_day"] = out["doubling_time_hours_at_growth_temp"] > 24
    if cal is not None:
        lg = cal[cal["kind"] == "slow logistic"].set_index("held_out")
        c = lg.loc[level]
        z = c["intercept"].to_numpy() + c["coef_pred"].to_numpy() * np.log(out["doubling_time_hours_at_growth_temp"].to_numpy()) \
            + c["coef_archaeon"].to_numpy() * (out["domain"] == "archaea").to_numpy()
        out["prob_slower_than_a_day"] = (1 / (1 + np.exp(-z))).round(3)
    else:
        out["prob_slower_than_a_day"] = np.nan

    OUT.mkdir(exist_ok=True)
    cols = [c for c, _ in COLUMNS]
    out[cols].sort_values("genome_id").to_csv(OUT / "gem_growth_predictions.csv", index=False)
    lines = ["# Columns of gem_growth_predictions.csv", "",
             f"One row per high-quality GEM genome with a prediction ({len(out):,} genomes). Ranges and errors were measured with "
             "whole species, genera, families or orders held out, for the genome's model and novelty level; for the full model the "
             "ranges also depend on the domain.", "",
             "| Column | Meaning |", "|---|---|"] + [f"| `{c}` | {d} |" for c, d in COLUMNS]
    (OUT / "COLUMNS.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'gem_growth_predictions.csv'}: {len(out):,} genomes, {int(out['slower_than_a_day'].sum())} slower than a day")


if __name__ == "__main__":
    main()
