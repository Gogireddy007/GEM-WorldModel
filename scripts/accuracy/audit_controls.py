#!/usr/bin/env python
"""Two checks of the label audit, for the paper.

Positive control: the study-level check (screen_label_sources.check) is run on the training table with
the exclusions switched off, and the excluded values found before that check existed (all but the last,
which the check itself found) are counted as caught when they sit in a flagged study, are at least
three-fold from another source for the same species, or match no source record and are at least
three-fold from the closest one.

Effect of one study: the phylogenetic test of test_lifestyle_phylogenetic.py for methane or ammonia
monooxygenase, with the current exclusions, with the four Kevbrina et al. values put back, and with no
exclusions at all.

Writes audit_controls.csv.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import screen_label_sources as sources
import test_lifestyle_phylogenetic as phylo
from screen_labels import source_records

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

EXCLUSIONS = deploy_features.EXCLUSIONS
FOUND_BY_STUDY_CHECK = {"Streptomyces coelicolor"}
KEVBRINA = "Kevbrina"


def with_exclusions(table):
    """Point the training table loader at an exclusion list (None: no exclusions)."""
    if table is None:
        deploy_features.EXCLUSIONS = Path("/nonexistent/label_exclusions.csv")
        return
    tmp = Path(tempfile.mkdtemp()) / "label_exclusions.csv"
    table.to_csv(tmp, index=False)
    deploy_features.EXCLUSIONS = tmp


def positive_control(processed, k, excluded):
    with_exclusions(None)
    df, _, _ = deploy_features.load_training_table(processed, k, with_cub=True)
    lab, comp, studies, conflict, near = sources.check(df, source_records())
    flagged_studies = set(studies.loc[studies["flag"], "study"])
    rows = []
    for sp in excluded["species"]:
        if sp in FOUND_BY_STUDY_CHECK:
            continue
        how = []
        if sp in set(comp.loc[comp["study"].isin(flagged_studies), "species"]):
            how.append("flagged study")
        if sp in set(conflict["species"]):
            how.append("other source")
        if sp in set(near["species"]):
            how.append("no matching record")
        label = lab.loc[sp, "label"] if sp in lab.index else np.nan
        closest = near.set_index("species")["evidence_hours"].get(sp, np.nan)
        rows.append({"check": "positive control", "species": sp, "caught": bool(how), "how": "; ".join(how),
                     "label_hours": label, "closest_record_hours": closest})
    out = pd.DataFrame(rows)
    print(f"positive control: {int(out['caught'].sum())} of {len(out)} earlier exclusions caught by the study-level check")
    print(out.to_string(index=False))
    return out


def monooxygenase_effect(processed, k, excluded):
    rows = []
    settings = (("current exclusions", excluded),
                ("Kevbrina values put back", excluded[~excluded["source"].str.contains(KEVBRINA)]),
                ("no exclusions", None))
    for label, table in settings:
        with_exclusions(table)
        g = phylo.group_table(processed, k, True)
        g, C = phylo.place_on_tree(g)
        r = phylo.run(g, C, label).set_index("input").loc["ammonia_monooxygenase"]
        rows.append({"check": "monooxygenase PGLS", "species": label, "n_groups": int(r["n_groups"]), "n_with": int(r["n_with"]),
                     "fold_pgls": r["fold_pgls"], "ci_low_pgls": r["ci_low_pgls"], "ci_high_pgls": r["ci_high_pgls"],
                     "p_pgls_bh": r["p_pgls_bh"], "fold_across_species": r["fold_across_species"]})
        print(f"{label:28s} groups {int(r['n_groups'])}, with the gene {int(r['n_with'])}: PGLS {r['fold_pgls']:.2f}x "
              f"({r['ci_low_pgls']:.2f} to {r['ci_high_pgls']:.2f}), adjusted p {r['p_pgls_bh']:.3g}; across species {r['fold_across_species']:.2f}x")
    return pd.DataFrame(rows)


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    excluded = pd.read_csv(EXCLUSIONS)
    try:
        out = pd.concat([positive_control(processed, k, excluded), monooxygenase_effect(processed, k, excluded)], ignore_index=True)
    finally:
        deploy_features.EXCLUSIONS = EXCLUSIONS
    out.to_csv(processed / "audit_controls.csv", index=False)
    print("wrote audit_controls.csv")


if __name__ == "__main__":
    main()
