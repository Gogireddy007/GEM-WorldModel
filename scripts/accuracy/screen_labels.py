#!/usr/bin/env python
"""This screens the labeled doubling times for values that are probably not a
maximal growth rate (a units slip, growth under poor conditions, a mismatched
temperature), and lists the source records behind each suspect so it can be
checked by hand.

A species is flagged when its measured doubling time is at least FOLD times away
from both its held-out prediction (calibration_data_labeled_species.csv, species
held out) and the median of the other labeled species in its genus (at least two
of them). Each flagged species is then looked up in the per-source files of
Madin et al.'s compilation (output/prepared_data in their repository, downloaded
once to data/raw/madin_prepared), which give every record with its temperature
and reference.

Nothing is excluded automatically: a value goes into
data/curated/label_exclusions.csv only after the source has been checked.
Writes label_screen.csv.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

FOLD = 5.0
SOURCES = ["corkrey", "silva", "methanogen", "bergeys", "mediadb", "masonmm", "kremer", "prochlorococcus", "edwards", "nielsensl"]
URL = "https://raw.githubusercontent.com/bacteria-archaea-traits/bacteria-archaea-traits/master/output/prepared_data/{}.csv"
RAW = Path("data/raw/madin_prepared")


def source_records():
    RAW.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in SOURCES:
        path = RAW / f"{name}.csv"
        if not path.exists():
            path.write_bytes(requests.get(URL.format(name), timeout=120).content)
        d = pd.read_csv(path, low_memory=False, encoding="latin-1")
        name_col = "species" if "species" in d and "org_name" not in d else "org_name"
        d = d.rename(columns={name_col: "name"})
        for col in ("growth_tmp", "reference"):
            if col not in d:
                d[col] = np.nan
        d = d.dropna(subset=["doubling_h"])
        frames.append(d[["name", "doubling_h", "growth_tmp", "reference"]].assign(source=name))
    out = pd.concat(frames, ignore_index=True)
    out["species"] = out["name"].map(lambda s: " ".join(str(s).replace("[", "").replace("]", "").split()[:2]))
    return out


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, _, _ = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True)
    df = df.drop_duplicates("species")
    cal = pd.read_csv("Test on HQ genomes/calibration_data_labeled_species.csv").groupby("species")[["measured_hours", "predicted_hours"]].mean()
    df = df.join(cal, on="species")
    df["genus"] = df["gtdb_taxonomy"].map(lambda t: (re.search(r"g__([^;]+)", str(t)) or [None, None])[1])
    logy = np.log(df["doubling_time_hours"].astype(float))
    genus_median, genus_n = [], []
    for i, (g, _) in enumerate(zip(df["genus"], logy, strict=True)):
        others = logy[(df["genus"] == g).to_numpy() & (np.arange(len(df)) != i)] if g else pd.Series(dtype=float)
        genus_median.append(others.median() if len(others) >= 2 else np.nan)
        genus_n.append(len(others))
    df["fold_vs_prediction"] = np.exp(np.abs(logy - np.log(df["predicted_hours"])))
    df["fold_vs_genus"] = np.exp(np.abs(logy - np.array(genus_median)))
    df["genus_relatives"] = genus_n
    flag = (df["fold_vs_prediction"] >= FOLD) & (df["fold_vs_genus"] >= FOLD)
    sus = df[flag].sort_values("fold_vs_genus", ascending=False)
    print(f"{len(df)} labeled species; {int(flag.sum())} are at least {FOLD:.0f}x from both their held-out prediction and their genus")

    recs = source_records()
    rows = []
    for r in sus.itertuples():
        hits = recs[recs["species"] == r.species]
        genus_vals = df.loc[(df["genus"] == r.genus) & (df["species"] != r.species), "doubling_time_hours"]
        print(f"\n{r.species}: {r.doubling_time_hours:.3g} h at {r.growth_temp_c} C ({r.origin}); predicted {r.predicted_hours:.3g} h; "
              f"genus {r.genus} relatives median {genus_vals.median():.3g} h (n {len(genus_vals)})")
        if hits.empty:
            print("   no record in the per-source files (value from the original corpus or another table)")
        for h in hits.itertuples():
            print(f"   {h.source:10s} {h.doubling_h:>10.4g} h at {h.growth_tmp} C   {str(h.reference)[:110]}")
            rows.append({"species": r.species, "label_hours": r.doubling_time_hours, "predicted_hours": r.predicted_hours,
                         "genus_median_hours": genus_vals.median(), "source": h.source, "source_hours": h.doubling_h,
                         "source_temp_c": h.growth_tmp, "reference": h.reference})
        if hits.empty:
            rows.append({"species": r.species, "label_hours": r.doubling_time_hours, "predicted_hours": r.predicted_hours,
                         "genus_median_hours": genus_vals.median()})
    pd.DataFrame(rows).to_csv(processed / "label_screen.csv", index=False)
    print(f"\nwrote label_screen.csv ({len(sus)} species)")


if __name__ == "__main__":
    main()
