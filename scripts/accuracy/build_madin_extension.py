#!/usr/bin/env python
"""This builds extra labeled species from Madin et al.'s table, for species
that were left out of the corpus because they had no match on the pruned
corpus tree.

The landmark phylogeny only needs a GTDB taxonomy string (bacterial or
archaeal), so a species is usable if its genome size, GC content and tRNA count are in Madin's table (they
agree closely with the values computed from genomes for the species in both,
correlations 0.99, 0.997 and 0.90) and it can be placed in the GTDB taxonomy by
species name, else by genus, else by family. Doubling times are corrected to
the reference temperature the same way as for the original corpus.

Madin's values are species-level, so these rows are not strain-matched to any
specific genome, the same caveat the original corpus carries.

Writes madin_extension.csv.
"""

import re

import numpy as np
import pandas as pd
import pyreadr

from gem_worldmodel.data import gtdb
from gem_worldmodel.features import temperature
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def token(taxonomy, prefix):
    m = re.search(rf"{prefix}[^;]*", str(taxonomy))
    return m.group(0) if m and m.group(0) != prefix else None


def main():
    data_cfg = load_config("data")
    feat_cfg = load_config("features")
    processed = resolve_path(data_cfg["paths"]["processed_dir"])

    growth = pyreadr.read_r("data/raw/GrowthRates_Madin.rda")["d"]
    growth["species"] = growth["species"].str.replace(r"^\[(\w+)\]", r"\1", regex=True)
    traits = pd.read_csv("data/raw/condensed_species_NCBI.csv", low_memory=False)[
        ["species", "genome_size", "gc_content", "tRNA_genes", "genus", "family"]
    ]
    corpus = pd.read_csv(processed / "features_sample_expanded.csv")

    df = growth.merge(traits, on="species", how="left")
    df = df[~df["species"].isin(corpus["species"])]
    df = df.dropna(subset=["genome_size", "gc_content", "tRNA_genes"]).drop_duplicates("species")
    logger.info(f"{len(df)} species not in the corpus with size, GC and tRNA count")

    taxonomy = pd.concat([gtdb.fetch_bac_taxonomy(data_cfg), gtdb.fetch_arc_taxonomy(data_cfg)], ignore_index=True)
    tax = taxonomy.drop_duplicates("gtdb_taxonomy")["gtdb_taxonomy"]
    by_species, by_genus, by_family = {}, {}, {}
    for t in tax:
        s, g, f = token(t, "s__"), token(t, "g__"), token(t, "f__")
        if s:
            by_species.setdefault(s[3:], t)
        if g:
            by_genus.setdefault(re.sub(r"_[A-Z]+$", "", g[3:]), {}).setdefault(g, []).append(t)
        if f:
            by_family.setdefault(re.sub(r"_[A-Z]+$", "", f[3:]), {}).setdefault(f, []).append(t)

    def place(row):
        if row["species"] in by_species:
            return by_species[row["species"]], "species"
        genus_groups = by_genus.get(str(row["genus"]))
        if genus_groups:
            best = max(genus_groups.values(), key=len)[0]
            return re.sub(r"s__[^;]*", "s__", best), "genus"
        family_groups = by_family.get(str(row["family"]))
        if family_groups:
            best = max(family_groups.values(), key=len)[0]
            return re.sub(r"s__[^;]*", "s__", re.sub(r"g__[^;]*", "g__", best)), "family"
        return None, None

    placed = df.apply(place, axis=1, result_type="expand")
    df["gtdb_taxonomy"], df["match_rank"] = placed[0], placed[1]
    logger.info(f"placement: {df['match_rank'].value_counts(dropna=False).to_dict()}")
    df = df.dropna(subset=["gtdb_taxonomy"]).copy()

    df = df.rename(columns={"d": "doubling_time_hours", "GrowthTemp": "growth_temp_c", "genome_size": "genome_size_bp",
                            "tRNA_genes": "trna_count"})
    df["gc_content"] = df["gc_content"] / 100.0
    df = temperature.add_reference_temperature_target(df, feat_cfg)
    cols = ["species", "doubling_time_hours", "growth_temp_c", "doubling_time_hours_ref", "temp_corrected",
            "genome_size_bp", "gc_content", "trna_count", "gtdb_taxonomy", "match_rank"]
    out = df[cols].reset_index(drop=True)
    out_path = processed / "madin_extension.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}: {len(out)} species, {int(out['temp_corrected'].sum())} temperature-corrected")
    print(out["match_rank"].value_counts().to_string())
    print(out["doubling_time_hours_ref"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
