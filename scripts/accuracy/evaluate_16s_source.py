#!/usr/bin/env python
"""Does it matter where a 16S sequence comes from?

Most labeled species get their 16S gene from NCBI by species name (fetch_labeled_16s.py), while a GEM
genome's 16S is cut from the genome itself with barrnap. The model is trained on the first kind and
applied to the second. Here the 16S gene is cut with barrnap from the genome of every labeled row, in
the same way as for GEM genomes (extract_16s_from_genomes.first_16s), and three things are measured:

  sequences   k-mer cosine distance between the NCBI copy and the genome copy of the same species,
              against the distance from the NCBI copy to the closest other labeled species
  inputs      how far the 16 16S inputs of a row move when they are computed from the genome copy,
              in units of their spread across rows
  accuracy    the deployed model trained and tested on NCBI copies, trained on NCBI copies and tested on
              genome copies (how GEM genomes were predicted before 2026-10-27), and trained and tested on
              genome copies (how the training table is built since); same rows, folds and seeds

--extract runs barrnap (resumable, cached in labeled_16s_from_genomes.csv); the default mode evaluates.
Writes 16s_source_evaluation.csv.
"""

import argparse
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from extract_16s_from_genomes import first_16s
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features, rrna16s
from gem_worldmodel.models.deployed import DeployedRegressor
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = [0, 1, 2]
PREFIX = {"family": "f__", "order": "o__"}


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def extract(processed, df, workers):
    path = processed / "labeled_16s_from_genomes.csv"
    done = pd.read_csv(path).fillna("") if path.exists() else pd.DataFrame(columns=["accession", "sequence"])
    have = set(done["accession"])
    jobs = sorted({(a, a, "arc" if t.startswith("d__Archaea") else "bac")
                   for a, t in zip(df["accession"], df["gtdb_taxonomy"], strict=True) if isinstance(a, str) and a not in have})
    print(f"{len(jobs)} genomes to search, {len(have)} cached", flush=True)
    rows = done.to_dict("records")
    with Pool(workers) as pool:
        for i, (acc, seq) in enumerate(pool.imap_unordered(first_16s, jobs), 1):
            rows.append({"accession": acc, "sequence": seq})
            if i % 25 == 0 or i == len(jobs):
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"{i}/{len(jobs)}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, names = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True)
    if args.extract:
        extract(processed, df, args.workers)
        return
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]

    genome = pd.read_csv(processed / "labeled_16s_from_genomes.csv").fillna("").set_index("accession")["sequence"]
    ncbi = pd.read_csv(processed / "labeled_16s_sequences.csv").fillna("").set_index("species")["sequence"]
    landmarks = [rrna16s.kmer_profile(ncbi[n], k) for n in np.load(processed / "labeled_16s_landmark_names.npy")]
    s16 = [i for i, n in enumerate(names) if n.startswith("s16_")]

    gseq = df["accession"].map(genome).fillna("")
    nseq = df["species"].map(ncbi).fillna("")
    both = ((gseq != "") & (nseq != "")).to_numpy()
    differs = both & (gseq != nseq).to_numpy()
    print(f"{len(df)} rows; genome copy found for {(gseq != '').sum()}, NCBI copy for {(nseq != '').sum()}, "
          f"both for {both.sum()}, and the two differ for {differs.sum()}")

    # sequences: own-species distance against the closest other species
    prof_n = {s: rrna16s.kmer_profile(q, k) for s, q in ncbi.items() if q}
    species = list(prof_n)
    rows = []
    own, nearest = [], []
    for i in np.where(differs)[0]:
        pg = rrna16s.kmer_profile(gseq[i], k)
        own.append(rrna16s.kmer_cosine_distance(pg, prof_n[df["species"][i]]))
        nearest.append(min(rrna16s.kmer_cosine_distance(prof_n[df["species"][i]], prof_n[s]) for s in species if s != df["species"][i]))
    own, nearest = np.array(own), np.array(nearest)
    print(f"\nsequences ({len(own)} rows): NCBI against genome copy, median distance {np.median(own):.4f} "
          f"(90th percentile {np.quantile(own, 0.9):.4f}); NCBI copy against the closest other species {np.median(nearest):.4f}; "
          f"own copy farther than the closest other species for {(own > nearest).sum()}")
    rows.append({"measure": "sequence distance, own copies (median)", "value": float(np.median(own)), "n": len(own)})
    rows.append({"measure": "sequence distance, own copies (90th percentile)", "value": float(np.quantile(own, 0.9)), "n": len(own)})
    rows.append({"measure": "sequence distance, closest other species (median)", "value": float(np.median(nearest)), "n": len(own)})
    rows.append({"measure": "rows whose own copies differ more than the closest other species", "value": int((own > nearest).sum()), "n": len(own)})

    # inputs: X has the genome copy where there is one (load_training_table); Xn has the NCBI copy instead
    Xn, Xg = X.copy(), X.copy()
    for i in np.where(nseq != "")[0]:
        pn = rrna16s.kmer_profile(nseq[i], k)
        Xn[i, s16] = [rrna16s.kmer_cosine_distance(pn, lp) for lp in landmarks]
    shift = np.abs(Xg[differs][:, s16] - Xn[differs][:, s16]) / Xn[:, s16].std(axis=0)
    print(f"inputs: median shift {np.median(shift):.3f} SD, 90th percentile {np.quantile(shift, 0.9):.3f} SD")
    rows.append({"measure": "16S input shift in SD (median)", "value": float(np.median(shift)), "n": int(differs.sum())})
    rows.append({"measure": "16S input shift in SD (90th percentile)", "value": float(np.quantile(shift, 0.9)), "n": int(differs.sum())})

    # accuracy
    y = np.log(df["doubling_time_hours"].to_numpy(float))
    schemes = ("NCBI copy", "trained on NCBI, genome copy", "genome copy")
    for level in ("species", "family", "order"):
        groups = df["species_group"].to_numpy() if level == "species" else df["gtdb_taxonomy"].map(lambda t, p=PREFIX[level]: token(t, p)).to_numpy()
        errs = {name: [] for name in schemes}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            ug = np.unique(groups)
            perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
            pred = {name: np.full(len(y), np.nan) for name in schemes}
            for tr, te in GroupKFold(5).split(X, groups=np.array([perm[v] for v in groups])):
                m = DeployedRegressor("own temperature", seed).fit(Xn[tr], y[tr])
                pred["NCBI copy"][te], pred["trained on NCBI, genome copy"][te] = m.predict(Xn[te]), m.predict(Xg[te])
                pred["genome copy"][te] = DeployedRegressor("own temperature", seed).fit(Xg[tr], y[tr]).predict(Xg[te])
            for name in schemes:
                errs[name].append(np.exp(np.median(np.abs(y[differs] - pred[name][differs]))))
        print(f"{level:8s} held out, {differs.sum()} rows: " + ", ".join(f"{name} {np.mean(e):.3f}x" for name, e in errs.items()))
        for name, e in errs.items():
            rows.append({"measure": f"typical error, {level} held out, {name}", "value": float(np.mean(e)), "n": int(differs.sum())})
        for name in schemes[1:]:
            d = np.array(errs[name]) - np.array(errs["NCBI copy"])
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            rows.append({"measure": f"seeds worse than NCBI copy, {level} held out, {name}", "value": int((d > 0).sum()), "n": len(SEEDS), "p": p})
    pd.DataFrame(rows).to_csv(processed / "16s_source_evaluation.csv", index=False)
    print("wrote 16s_source_evaluation.csv")


if __name__ == "__main__":
    main()
