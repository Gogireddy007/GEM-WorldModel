#!/usr/bin/env python
"""This evaluates the syntrophy rule in lifestyle_genes.syntrophy_candidate, which was fixed from the
literature before being run, against the known syntrophs in the labeled set.

Reports how many known syntrophs it finds, how many other genomes it calls, and which orders those
false calls come from. The rule was not adjusted after this evaluation (2026-10-15): with six known
syntrophs any rule tuned to them would only memorize them. Result: it is not used as a feature.
"""

import pandas as pd

from gem_worldmodel.features import deploy_features, lifestyle_genes
from gem_worldmodel.utils.config import load_config, resolve_path

# obligate or facultative syntrophs in the labeled set (Pelotomaculum thermopropionicum has no genome);
# the last two oxidize acetate syntrophically, which the rule does not cover
KNOWN = ["Syntrophus aciditrophicus", "Syntrophobacter fumaroxidans", "Syntrophothermus lipocalidus",
         "Thermosyntropha lipolytica", "Thermacetogenium phaeum", "Clostridium ultunense"]


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    df, _, _ = deploy_features.load_training_table(processed, load_config("features")["rrna16s"]["kmer_k"], with_cub=True)
    counts = pd.read_csv(processed / "labeled_lifestyle_genes.csv").set_index("accession")
    t = df.drop_duplicates("species").join(counts, on="accession").dropna(subset=["n_genes"]).reset_index(drop=True)
    t["candidate"] = lifestyle_genes.syntrophy_candidate(t)
    known = t["species"].isin(KNOWN)
    found = t.loc[known & (t["candidate"] == 1), "species"].tolist()
    false = t[~known & (t["candidate"] == 1)]
    print(f"known syntrophs with a genome: {int(known.sum())}; found {len(found)}: {found}")
    print(f"missed: {sorted(set(t.loc[known, 'species']) - set(found))}")
    print(f"candidates: {int((t['candidate'] == 1).sum())} of {len(t)}; not known syntrophs: {len(false)}; "
          f"share of candidates that are known syntrophs {len(found) / max(1, int((t['candidate'] == 1).sum())):.0%}")
    print("false calls by order:", false["gtdb_taxonomy"].str.extract(r"o__([^;]+)")[0].value_counts().head(8).to_dict())


if __name__ == "__main__":
    main()
