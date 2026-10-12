#!/usr/bin/env python
"""This checks the deployment pipeline on real MAGs.

GEM genomes whose GTDB species matches a labeled species with a measured growth
rate are predicted with that species held out of training (whole species held
out in a 5-fold split), and compared with the prediction for the same species
made from its isolate genome. Both use the full feature set: genome traits,
phylogeny, 16S and gene features (codon usage, IVYWREL). MAGs are the genomes with completeness of at
least 70 and contamination of at most 10 that have all four inputs.

Needs gem_cub.csv for mag_validation_ids.csv (compute_cub_gem.py).
Writes mag_validation.csv, one row per species, for the report charts.
"""

import re
import warnings
from ast import literal_eval

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features, rrna16s
from gem_worldmodel.features import taxon_vectors as tv
from gem_worldmodel.utils.config import load_config, resolve_path

warnings.filterwarnings("ignore", message="Mean of empty slice")
SEEDS = [42, 1, 7, 13, 2024]
TARGETS = {"own temperature": "doubling_time_hours", "20 C": "doubling_time_hours_ref"}


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else None


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    table = tv.load_taxon_table(processed / "gtdb_taxon_vectors.csv")
    lab, X, names = deploy_features.load_training_table(processed, k, with_cub=True)
    lab = lab.assign(sp=lab["gtdb_taxonomy"].map(lambda t: token(t, "s__")))
    seqs = pd.read_csv(processed / "labeled_16s_sequences.csv").fillna("")
    profiles = {s: rrna16s.kmer_profile(q, k) for s, q in zip(seqs.species, seqs.sequence) if q}
    landmarks = [profiles[n] for n in np.load(processed / "labeled_16s_landmark_names.npy")]

    ids = pd.read_csv(processed / "mag_validation_ids.csv")
    meta = pd.read_csv("data/raw/gem_genome_metadata.tsv", sep="\t", low_memory=False)[["genome_id", "ecosystem", "completeness", "mimag_quality"]]
    base = pd.read_csv(processed / "unlabeled_corpus_features.csv")[["genome_id", "genome_size_bp", "trna_count"]]
    gc = pd.read_csv(processed / "unlabeled_corpus_features_enriched.csv")[["genome_id", "gc_content"]]
    prof = pd.read_csv(processed / "unlabeled_corpus_features_16s.csv")[["genome_id", "kmer_profile_16s"]]
    cub = pd.read_csv(processed / "gem_cub.csv").drop_duplicates("genome_id")[["genome_id", "cub", "ivywrel"]]
    mag = (ids.merge(meta, on="genome_id").merge(base, on="genome_id").merge(gc, on="genome_id")
           .merge(prof, on="genome_id", how="left").merge(cub, on="genome_id"))
    mag["sp"] = mag["ecosystem"].map(lambda t: token(t, "s__"))
    mag = mag.dropna(subset=["genome_size_bp", "trna_count", "gc_content", "cub", "ivywrel", "sp"])
    mag["has_16s"] = mag["kmer_profile_16s"].notna()
    mag = mag[mag["sp"].isin(set(lab["sp"].dropna()))].reset_index(drop=True)

    lm = []
    for t in mag["ecosystem"]:
        v, _ = tv.lookup(t, table, "order")
        lm.append(v if v is not None else np.full(16, np.nan))
    lm = np.vstack(lm)
    s16 = np.vstack([
        [rrna16s.kmer_cosine_distance(literal_eval(p), lp) for lp in landmarks] if isinstance(p, str) else np.full(16, np.nan)
        for p in mag["kmer_profile_16s"]
    ])
    flag = mag["ecosystem"].map(tv.is_archaeon).to_numpy()[:, None]
    Xm = np.hstack([mag[deploy_features.TRAITS].to_numpy(float), lm, s16, flag, mag[["cub", "ivywrel"]].to_numpy(float)])
    # training matrix columns are traits, lm, s16, archaeon, cub, ivywrel, the same order as Xm
    ok = ~np.isnan(np.delete(Xm, range(19, 35), axis=1)).any(axis=1)
    mag, Xm = mag[ok].reset_index(drop=True), Xm[ok]
    print(f"{len(mag)} MAGs from {mag['sp'].nunique()} species with a measured growth rate ({(mag.mimag_quality == 'HQ').sum()} high quality, {mag.has_16s.sum()} with a 16S gene)")

    use = lab["has_cub"].to_numpy()
    lab, X = lab[use].reset_index(drop=True), X[use]
    species = lab["sp"].fillna(lab["species"]).to_numpy()
    rows = []
    for variant, drop in (("all four inputs", []), ("without 16S", list(range(19, 35)))):
        Xv = np.delete(X, drop, axis=1)
        Xmv = np.delete(Xm, drop, axis=1)
        mag_ok = mag["has_16s"].to_numpy() if not drop else np.ones(len(mag), bool)
        for tname, col in TARGETS.items():
            y = np.log(lab[col].to_numpy(float))
            per_seed = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                ug = np.unique(species)
                perm = dict(zip(ug, rng.permutation(len(ug))))
                g = np.array([perm[s_] for s_ in species])
                iso = np.full(len(lab), np.nan)
                mag_pred = np.full(len(mag), np.nan)
                for tr, te in GroupKFold(5).split(Xv, y, g):
                    m = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5).fit(Xv[tr], y[tr])
                    iso[te] = m.predict(Xv[te])
                    sel = mag["sp"].isin(set(species[te])).to_numpy() & mag_ok
                    if sel.any():
                        mag_pred[sel] = m.predict(Xmv[sel])
                per_seed.append((iso, mag_pred))
            iso_mean = np.mean([p[0] for p in per_seed], axis=0)
            mag_mean = np.nanmean([p[1] for p in per_seed], axis=0)
            sp_table = (pd.DataFrame({"sp": species, "truth": y, "isolate": iso_mean}).groupby("sp").mean()
                        .join(pd.DataFrame({"sp": mag["sp"], "mag": mag_mean}).groupby("sp").agg(mag=("mag", "median"), n_mags=("mag", "count"))).dropna())
            t, a_, b_ = sp_table["truth"].to_numpy(), sp_table["isolate"].to_numpy(), sp_table["mag"].to_numpy()
            print(f"\n{variant}, {tname}: {len(sp_table)} species, {int(sp_table.n_mags.sum())} MAGs (whole species held out, mean of {len(SEEDS)} seeds)")
            print(f"  typical fold error : isolate {np.exp(np.median(np.abs(t - a_))):.2f}x   MAG {np.exp(np.median(np.abs(t - b_))):.2f}x")
            print(f"  Spearman           : isolate {spearmanr(t, a_).correlation:.3f}   MAG {spearmanr(t, b_).correlation:.3f}")
            print(f"  R2 (log)           : isolate {r2_score(t, a_):.3f}   MAG {r2_score(t, b_):.3f}")
            print(f"  MAG vs isolate prediction for the same species: Spearman {spearmanr(a_, b_).correlation:.3f}, median ratio {np.exp(np.median(b_ - a_)):.2f}")
            rows.append(sp_table.reset_index().assign(target=tname, variant=variant))
    pd.concat(rows).to_csv(processed / "mag_validation.csv", index=False)


if __name__ == "__main__":
    main()
