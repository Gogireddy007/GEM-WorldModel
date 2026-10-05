#!/usr/bin/env python
"""This predicts doubling times for GEM genomes with the gradient-boosted-trees
model, using features that are defined the same way for the labeled species
and for the GEM genomes.

The earlier prediction script fed GEM genomes phylogeny and 16S coordinates
from different coordinate systems than the labeled species had (landmark
distances for GEM, MDS coordinates for the labeled set, and a 16S MDS that
depended on the batch being predicted). This one uses:

  genome traits       genome size, GC content, tRNA count (the traits that
                      exist for both isolate genomes and MAGs)
  phylogeny           mean distance-to-landmark vector of the GTDB tree tips in
                      the genome's species, genus, family or order, from the
                      GTDB taxonomy string in the GEM metadata
  16S                 cosine distance of the 16S k-mer profile to 16 fixed
                      labeled-species reference sequences

A genome gets whichever model its available features support. Each row also
records how close its nearest labeled relative is (by name) and the typical
error measured for that level of novelty when whole lineages were held out.
GEM names come from an older GTDB release than the labeled set, so a renamed
taxon looks more novel than it is, which makes the novelty level a worst case.
"""

import argparse
import re
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from gem_worldmodel.features import rrna16s, taxon_vectors as tv
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

TRAITS = ["genome_size_bp", "gc_content", "trna_count"]
# typical fold error measured with whole lineages held out (see research_log.md, 2026-10-04)
TYPICAL_ERROR = {"same genus": 2.1, "same family, new genus": 2.2, "same order, new family": 2.5, "new order": 2.5}
RANKS = ("species", "genus", "family", "order")


def token(taxonomy, prefix):
    m = re.search(rf"{prefix}[^;]*", str(taxonomy))
    return m.group(0) if m and m.group(0) != prefix else None


def coarsen(taxonomy, rank):
    """Blank out ranks finer than `rank` so lookup lands on that rank."""
    t = str(taxonomy)
    if rank in ("genus", "family", "order"):
        t = re.sub(r"s__[^;]*", "s__", t)
    if rank in ("family", "order"):
        t = re.sub(r"g__[^;]*", "g__", t)
    if rank == "order":
        t = re.sub(r"f__[^;]*", "f__", t)
    return t


def profile_distances(profile, landmark_profiles):
    return np.array([rrna16s.kmer_cosine_distance(profile, lp) for lp in landmark_profiles])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome-ids-file", default="hq_genome_ids_all.csv")
    parser.add_argument("--out", default="hq_predictions_gbm.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_cfg = load_config("data")
    k = load_config("features")["rrna16s"]["kmer_k"]
    processed = resolve_path(data_cfg["paths"]["processed_dir"])
    table = tv.load_taxon_table(processed / "gtdb_taxon_vectors.csv")

    lab = pd.read_csv(processed / "features_sample_expanded.csv")
    seqs = pd.read_csv(processed / "labeled_16s_sequences.csv").fillna("")
    lab_profiles = {s: rrna16s.kmer_profile(q, k) for s, q in zip(seqs.species, seqs.sequence) if q}
    landmark_names = list(np.load(processed / "labeled_16s_landmark_names.npy"))
    landmark_profiles = [lab_profiles[n] for n in landmark_names]

    # ---- labeled training matrices, one block per phylogeny precision level
    S_lab = np.vstack([
        profile_distances(lab_profiles[s], landmark_profiles) if s in lab_profiles else np.full(16, np.nan)
        for s in lab.species
    ])
    S_med = np.nanmedian(S_lab, axis=0)
    S_lab = np.where(np.isnan(S_lab), S_med, S_lab)
    T_lab = lab[TRAITS].to_numpy(float)
    y = np.log(lab["doubling_time_hours_ref"].to_numpy())

    def lm_block(rank):
        return np.vstack([tv.lookup(coarsen(t, rank), table, rank)[0] for t in lab.gtdb_taxonomy])

    lm_blocks = [lm_block(r) for r in RANKS]

    def fit(use_lm, use_s):
        blocks, targets = [], []
        for lm in (lm_blocks if use_lm else [None]):
            cols = [T_lab] + ([lm] if use_lm else []) + ([S_lab] if use_s else [])
            blocks.append(np.hstack(cols))
            targets.append(y)
        model = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3)
        return model.fit(np.vstack(blocks), np.concatenate(targets))

    models = {(lm, s): fit(lm, s) for lm in (True, False) for s in (True, False)}
    logger.info(f"trained {len(models)} models on {len(lab)} labeled genomes")

    # ---- GEM genomes
    ids = pd.read_csv(processed / args.genome_ids_file)[["genome_id"]]
    meta = pd.read_csv("data/raw/gem_genome_metadata.tsv", sep="\t", low_memory=False)
    base = pd.read_csv(processed / "unlabeled_corpus_features.csv")[["genome_id", "genome_size_bp", "trna_count"]]
    gc = pd.read_csv(processed / "unlabeled_corpus_features_enriched.csv")[["genome_id", "gc_content"]]
    prof = pd.read_csv(processed / "unlabeled_corpus_features_16s.csv")[["genome_id", "kmer_profile_16s"]]
    gem = (ids.merge(base, on="genome_id").merge(gc, on="genome_id")
           .merge(meta[["genome_id", "ecosystem", "ecosystem_category"]], on="genome_id")
           .merge(prof, on="genome_id", how="left"))
    gem = gem.rename(columns={"ecosystem": "gtdb_taxonomy"})
    gem = gem.dropna(subset=TRAITS).reset_index(drop=True)
    logger.info(f"{len(gem)} GEM genomes with size, GC and tRNA count")

    lm_vec, lm_rank = [], []
    for t in gem.gtdb_taxonomy:
        v, r = tv.lookup(t, table, "order")
        lm_vec.append(v if v is not None else np.full(16, np.nan))
        lm_rank.append(r)
    LM = np.vstack(lm_vec)
    has_lm = ~np.isnan(LM).any(axis=1)

    S = np.full((len(gem), 16), np.nan)
    for i, p in enumerate(gem.kmer_profile_16s):
        if isinstance(p, str):
            S[i] = profile_distances(literal_eval(p), landmark_profiles)
    has_s = ~np.isnan(S).any(axis=1)

    T = gem[TRAITS].to_numpy(float)
    pred = np.full(len(gem), np.nan)
    used = np.empty(len(gem), dtype=object)
    for (use_lm, use_s), model in models.items():
        mask = (has_lm == use_lm) & (has_s == use_s)
        if not mask.any():
            continue
        cols = [T[mask]] + ([LM[mask]] if use_lm else []) + ([S[mask]] if use_s else [])
        pred[mask] = np.exp(model.predict(np.hstack(cols)))
        used[mask] = "+".join(["traits"] + (["phylogeny"] if use_lm else []) + (["16S"] if use_s else []))

    # ---- how close is the nearest labeled relative (by name)
    lab_tokens = {r: set(lab.gtdb_taxonomy.map(lambda t, p=p: token(t, p)).dropna())
                  for r, p in (("genus", "g__"), ("family", "f__"), ("order", "o__"))}
    levels = []
    for t in gem.gtdb_taxonomy:
        if token(t, "g__") in lab_tokens["genus"]:
            levels.append("same genus")
        elif token(t, "f__") in lab_tokens["family"]:
            levels.append("same family, new genus")
        elif token(t, "o__") in lab_tokens["order"]:
            levels.append("same order, new family")
        else:
            levels.append("new order")

    out = pd.DataFrame({
        "genome_id": gem.genome_id,
        "predicted_doubling_time_hours": pred,
        "typical_fold_error": [TYPICAL_ERROR[l] for l in levels],
        "nearest_labeled_relative": levels,
        "model_used": used,
        "phylogeny_rank_used": lm_rank,
        "genome_size_bp": gem.genome_size_bp,
        "gc_content": gem.gc_content,
        "trna_count": gem.trna_count,
        "gtdb_taxonomy": gem.gtdb_taxonomy,
        "ecosystem_category": gem.ecosystem_category,
    })
    out_path = processed / args.out
    out.sort_values("predicted_doubling_time_hours").to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}: {len(out)} predictions")

    ref = lab["doubling_time_hours_ref"]
    d = out["predicted_doubling_time_hours"]
    print(f"labeled corpus : median {ref.median():.2f} h, 5th pct {ref.quantile(.05):.2f}, 95th pct {ref.quantile(.95):.1f}")
    print(f"GEM predictions: median {d.median():.2f} h, 5th pct {d.quantile(.05):.2f}, 95th pct {d.quantile(.95):.1f}, min {d.min():.2f}, max {d.max():.0f}")
    print(f"share slower than 24 h: {100 * (d > 24).mean():.1f}% (labeled {100 * (ref > 24).mean():.1f}%) | faster than 10 min: {100 * (d < 0.167).mean():.2f}%")
    print(out.model_used.value_counts().to_string())
    print(out.nearest_labeled_relative.value_counts(normalize=True).mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
