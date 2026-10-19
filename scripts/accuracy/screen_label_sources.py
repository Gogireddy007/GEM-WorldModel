#!/usr/bin/env python
"""This checks the labeled doubling times study by study, to catch errors that a species-level
screen misses: one source study that made a whole genus look slow (or fast) leaves no outlier among
relatives, because the relatives come from the same study.

Each training label is traced to the source record it came from in Madin et al.'s per-source files
(screen_labels.source_records), by matching the value. Every traced label is then compared with
evidence that does not come from the same study:

  other sources   the same species measured in a different study (median)
  relatives       other labeled species in the same genus (else family) whose labels come from a
                  different study (median, at least two)

and the log ratio label / evidence is recorded (other sources first, relatives otherwise). A study
is flagged when at least two of its labels can be compared, their median ratio is at least FOLD in
either direction and at least 80% of them point the same way. Two more lists: species whose label
is at least FOLD from another source's value for the same species, and labels that match no source
record but differ at least FOLD from the closest one (a unit or conversion step, like the methanotroph
values that came out about 24 times smaller than their source).

Growth temperature is not adjusted, so a study of cold-adapted organisms can be flagged for a real
reason; every flag is checked by hand before anything goes into data/curated/label_exclusions.csv.
Writes label_sources_studies.csv and label_sources_flags.csv.
"""

import re

import numpy as np
import pandas as pd
from screen_labels import source_records

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

FOLD = 3.0


def rank(t, p):
    m = re.search(rf"{p}([^;]+)", str(t))
    return m.group(1) if m else None


def flag_studies(comp, fold=FOLD):
    """Per-study summary of label / evidence log ratios, flagging studies off by at least fold, 80% one way."""
    studies = []
    for study, g in comp.dropna(subset=["log_ratio"]).groupby("study"):
        if len(g) < 2:
            continue
        med = g["log_ratio"].median()
        same = max((g["log_ratio"] > 0).mean(), (g["log_ratio"] < 0).mean())
        studies.append({"study": study, "n": len(g), "median_fold": float(np.exp(med)), "same_direction": same,
                        "flag": bool(abs(med) >= np.log(fold) and same >= 0.8), "species": "; ".join(g["species"])})
    out = pd.DataFrame(studies, columns=["study", "n", "median_fold", "same_direction", "flag", "species"])
    return out.sort_values("median_fold", key=lambda s: -np.abs(np.log(s)))


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, _, _ = deploy_features.load_training_table(processed, k, with_cub=True)
    lab = df.groupby("species").agg(label=("doubling_time_hours", "first"), tax=("gtdb_taxonomy", "first"),
                                    temp=("growth_temp_c", "first"), origin=("origin", "first"))
    lab["genus"] = lab["tax"].map(lambda t: rank(t, "g__"))
    lab["family"] = lab["tax"].map(lambda t: rank(t, "f__"))

    recs = source_records()
    recs["study"] = recs["reference"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str[:90]
    recs = recs[recs["doubling_h"] > 0]
    joined = recs.merge(lab[["label"]], left_on="species", right_index=True)
    joined["match"] = np.isclose(joined["doubling_h"], joined["label"], rtol=0.02)
    traced = joined[joined["match"]].drop_duplicates("species").set_index("species")["study"]
    lab["study"] = traced
    print(f"{len(lab)} labeled species; {lab['study'].notna().sum()} traced to a source record")

    log_label = np.log(lab["label"])
    rows = []
    for sp, r in lab[lab["study"].notna()].iterrows():
        others = recs[(recs["species"] == sp) & (recs["study"] != r["study"])]["doubling_h"]
        evidence, kind = (np.log(others).median(), "other sources") if len(others) else (np.nan, None)
        if kind is None:
            for level in ("genus", "family"):
                rel = lab[(lab[level] == r[level]) & (lab.index != sp) & (lab["study"] != r["study"])]
                if r[level] and len(rel) >= 2:
                    evidence, kind = np.log(rel["label"]).median(), f"{level} relatives (n {len(rel)})"
                    break
        rows.append({"species": sp, "study": r["study"], "label": r["label"], "temp": r["temp"],
                     "evidence_hours": np.exp(evidence) if kind else np.nan, "evidence": kind,
                     "log_ratio": log_label[sp] - evidence if kind else np.nan})
    comp = pd.DataFrame(rows)

    studies = flag_studies(comp)
    studies.to_csv(processed / "label_sources_studies.csv", index=False)
    print(f"{len(studies)} studies with at least two comparable labels; {int(studies['flag'].sum())} flagged "
          f"(median at least {FOLD:.0f}x off, 80% in one direction)")
    for s in studies[studies["flag"]].itertuples():
        print(f"\n  FLAG {s.median_fold:.2f}x over {s.n} labels: {s.study}")
        for c in comp[comp["study"] == s.study].itertuples():
            print(f"     {c.species:38s} label {c.label:8.3g} h at {c.temp} C, evidence {c.evidence_hours:8.3g} h ({c.evidence})")

    flags = [comp.assign(kind="study").merge(studies.loc[studies["flag"], ["study"]], on="study")]
    conflict = comp[(comp["evidence"] == "other sources") & (comp["log_ratio"].abs() >= np.log(FOLD))]
    print(f"\nspecies at least {FOLD:.0f}x from another source for the same species: {len(conflict)}")
    for c in conflict.sort_values("log_ratio", key=np.abs, ascending=False).itertuples():
        print(f"  {c.species:38s} label {c.label:8.3g} h, other sources {c.evidence_hours:8.3g} h   label from: {c.study[:70]}")
    flags.append(conflict.assign(kind="other source"))

    untraced = lab[lab["study"].isna()]
    near = []
    for sp, r in untraced.iterrows():
        v = recs.loc[recs["species"] == sp, ["doubling_h", "study"]]
        if len(v):
            i = (np.log(v["doubling_h"]) - np.log(r["label"])).abs().idxmin()
            ratio = r["label"] / v.loc[i, "doubling_h"]
            if abs(np.log(ratio)) >= np.log(FOLD):
                near.append({"species": sp, "label": r["label"], "evidence_hours": v.loc[i, "doubling_h"], "study": v.loc[i, "study"],
                             "log_ratio": np.log(ratio), "kind": "untraced"})
    near = pd.DataFrame(near)
    print(f"\nlabels that match no source record and are at least {FOLD:.0f}x from the closest one: {len(near)}")
    for c in near.itertuples() if len(near) else []:
        print(f"  {c.species:38s} label {c.label:8.3g} h, closest record {c.evidence_hours:8.3g} h   {c.study[:70]}")
    flags.append(near)
    pd.concat(flags, ignore_index=True).to_csv(processed / "label_sources_flags.csv", index=False)


if __name__ == "__main__":
    main()
