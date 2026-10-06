#!/usr/bin/env python
"""This turns the GEM prediction output into the table that goes in the HQ
genome report folder, with the metadata columns a reader would want."""

import re

import pandas as pd

from gem_worldmodel.utils.config import load_config, resolve_path


def level(taxonomy, prefix):
    m = re.search(rf"{prefix}[^;]*", str(taxonomy))
    return m.group(0)[3:] if m and m.group(0) != prefix else ""


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    p = pd.read_csv(processed / "hq_predictions_gbm.csv")
    meta = pd.read_csv("data/raw/gem_genome_metadata.tsv", sep="\t", low_memory=False)[
        ["genome_id", "mimag_quality", "completeness", "contamination", "ecosystem_type", "habitat"]
    ]
    for name, prefix in (("gtdb_phylum", "p__"), ("gtdb_family", "f__"), ("gtdb_genus", "g__")):
        p[name] = p["gtdb_taxonomy"].map(lambda t, x=prefix: level(t, x))
    out = p.merge(meta, on="genome_id", how="left").rename(columns={"predicted_doubling_time_hours": "doubling_time_hours"})
    out["doubling_time_hours"] = out["doubling_time_hours"].round(3)
    cols = ["genome_id", "doubling_time_hours", "typical_fold_error", "nearest_labeled_relative", "model_used",
            "phylogeny_rank_used", "genome_size_bp", "gc_content", "trna_count", "mimag_quality", "completeness",
            "contamination", "gtdb_phylum", "gtdb_family", "gtdb_genus", "ecosystem_category", "ecosystem_type", "habitat"]
    path = "Test on HQ genomes/hq_genome_predictions_v2.csv"
    out[cols].sort_values("genome_id").to_csv(path, index=False)
    print(f"wrote {path}: {len(out)} genomes")


if __name__ == "__main__":
    main()
