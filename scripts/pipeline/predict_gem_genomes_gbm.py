#!/usr/bin/env python
"""This predicts growth for GEM genomes with the gradient-boosted-trees model,
using features that are defined the same way for the labeled species and for
the GEM genomes.

Inputs for every genome:

  genome traits       genome size, GC content, tRNA count, plus a flag for
                      archaea (bacteria and archaea have their own landmarks on
                      GTDB's two trees)
  phylogeny           mean distance-to-landmark vector of the GTDB tree tips in
                      the genome's species, genus, family or order, from the
                      GTDB taxonomy string in the GEM metadata
  16S                 cosine distance of the 16S k-mer profile to 16 fixed
                      labeled-species reference sequences
  gene features       codon usage bias (MILC bias of the ribosomal proteins
                      against the genome background) and the IVYWREL amino acid
                      fraction, a signature of growth temperature, from
                      compute_cub_gem.py (the same annotation-free pipeline that
                      was run on the labeled genomes)

Three things are predicted: the doubling time at the organism's own growth
temperature, the growth temperature, and the doubling time scaled to 20 C (the
original target). A genome gets whichever model its available features support.
Models use median (quantile) loss, which held up better than squared error at
every level of lineage novelty.

Ranges and error labels come from final_accuracy_by_tier.csv, which holds the
accuracy measured with whole species, genera, families or orders held out
(evaluate_final_model.py). GEM names come from an older GTDB release than the
labeled set, so a renamed taxon looks more novel than it is and the novelty
level is a worst case.

Before predicting, the inputs are compared with the training inputs and the
run stops if any phylogeny or 16S column has moved by more than 3 training
standard deviations, which is what happened when the first predictions were
made with coordinates from a different system.
"""

import argparse
import itertools
import re
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from gem_worldmodel.features import deploy_features, rrna16s
from gem_worldmodel.features import taxon_vectors as tv
from gem_worldmodel.features.deploy_features import coarsen, combo_label, drift_report
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

TRAITS = deploy_features.TRAITS
RANKS = ("species", "genus", "family", "order")
TARGET_COLUMNS = {"own temperature": "doubling_time_hours", "20 C": "doubling_time_hours_ref", "growth temperature": "growth_temp_c"}
HOT_C = 45.0


def token(taxonomy, prefix):
    m = re.search(rf"{prefix}[^;]*", str(taxonomy))
    return m.group(0) if m and m.group(0) != prefix else None


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
    accuracy = pd.read_csv(processed / "final_accuracy_by_tier.csv")

    # ---- training data
    lab, X_lab, names = deploy_features.load_training_table(processed, k, with_cub=True)
    idx = {n: i for i, n in enumerate(names)}
    T_lab = X_lab[:, [idx[n] for n in TRAITS]]
    S_lab = X_lab[:, [i for n, i in idx.items() if n.startswith("s16_")]]
    D_lab = X_lab[:, [idx["archaeon"]]]
    C_lab = X_lab[:, [idx["cub"], idx["ivywrel"]]]
    lm_blocks = [
        np.vstack([tv.lookup(coarsen(t, r), table, "order")[0] for t in lab.gtdb_taxonomy]) for r in RANKS
    ]
    has_cub = lab["has_cub"].to_numpy()

    def fit(lm, s16, cub, target):
        col = TARGET_COLUMNS[target]
        rows = (has_cub if cub else np.ones(len(lab), bool)) & lab[col].notna().to_numpy()
        y = lab.loc[rows, col].to_numpy(float)
        y = y if target == "growth temperature" else np.log(y)
        blocks, targets = [], []
        for block in (lm_blocks if lm else [None]):
            parts = [T_lab[rows]] + ([block[rows]] if lm else []) + ([S_lab[rows]] if s16 else []) + ([C_lab[rows]] if cub else []) + [D_lab[rows]]
            blocks.append(np.hstack(parts))
            targets.append(y)
        model = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5)
        return model.fit(np.vstack(blocks), np.concatenate(targets))

    combos = list(itertools.product([True, False], repeat=3))
    models = {(c, t): fit(*c, t) for c in combos for t in TARGET_COLUMNS}
    logger.info(f"trained {len(models)} models on {len(lab)} labeled rows")

    # ---- profiles for the 16S reference sequences
    seqs = pd.read_csv(processed / "labeled_16s_sequences.csv").fillna("")
    lab_profiles = {s: rrna16s.kmer_profile(q, k) for s, q in zip(seqs.species, seqs.sequence) if q}
    landmark_profiles = [lab_profiles[n] for n in np.load(processed / "labeled_16s_landmark_names.npy")]

    # ---- GEM genomes
    ids = pd.read_csv(processed / args.genome_ids_file)[["genome_id"]]
    meta = pd.read_csv("data/raw/gem_genome_metadata.tsv", sep="\t", low_memory=False)
    base = pd.read_csv(processed / "unlabeled_corpus_features.csv")[["genome_id", "genome_size_bp", "trna_count"]]
    gc = pd.read_csv(processed / "unlabeled_corpus_features_enriched.csv")[["genome_id", "gc_content"]]
    prof = pd.read_csv(processed / "unlabeled_corpus_features_16s.csv")[["genome_id", "kmer_profile_16s"]]
    cub = pd.read_csv(processed / "gem_cub.csv").drop_duplicates("genome_id")[["genome_id", "cub", "ivywrel", "n_ribosomal"]]
    gem = (ids.merge(base, on="genome_id").merge(gc, on="genome_id")
           .merge(meta[["genome_id", "ecosystem", "ecosystem_category"]], on="genome_id")
           .merge(prof, on="genome_id", how="left").merge(cub, on="genome_id", how="left"))
    gem = gem.rename(columns={"ecosystem": "gtdb_taxonomy"}).dropna(subset=TRAITS).reset_index(drop=True)
    logger.info(f"{len(gem)} GEM genomes with size, GC and tRNA count")

    lm_vec, lm_rank = [], []
    for t in gem.gtdb_taxonomy:
        v, r = tv.lookup(t, table, "order")
        lm_vec.append(v if v is not None else np.full(16, np.nan))
        lm_rank.append(r)
    LM = np.vstack(lm_vec)
    S = np.full((len(gem), 16), np.nan)
    for i, p in enumerate(gem.kmer_profile_16s):
        if isinstance(p, str):
            S[i] = [rrna16s.kmer_cosine_distance(literal_eval(p), lp) for lp in landmark_profiles]
    has_lm, has_s = ~np.isnan(LM).any(axis=1), ~np.isnan(S).any(axis=1)
    C = gem[["cub", "ivywrel"]].to_numpy(float)
    has_c = ~np.isnan(C).any(axis=1)
    T = gem[TRAITS].to_numpy(float)
    D = gem["gtdb_taxonomy"].map(tv.is_archaeon).to_numpy()[:, None]

    bacterial = lab["archaeon"].to_numpy() == 0
    for name, train_mask, new_mask in (("bacteria", bacterial, D[:, 0] == 0), ("archaea", ~bacterial, D[:, 0] == 1)):
        if train_mask.sum() > 10 and new_mask.sum() > 10:
            logger.info(f"checking {name}")
            drift_report(
                {"traits": T_lab[train_mask], "phylogeny": lm_blocks[0][train_mask], "16S": S_lab[train_mask], "gene features": C_lab[has_cub & train_mask]},
                {"traits": T[new_mask], "phylogeny": LM[has_lm & new_mask], "16S": S[has_s & new_mask], "gene features": C[has_c & new_mask]},
            )

    # ---- relative novelty by name
    lab_tokens = {r: set(lab.gtdb_taxonomy.map(lambda t, p=p: token(t, p)).dropna()) for r, p in (("genus", "g__"), ("family", "f__"), ("order", "o__"))}
    tiers = []
    for t in gem.gtdb_taxonomy:
        if token(t, "g__") in lab_tokens["genus"]:
            tiers.append("same genus")
        elif token(t, "f__") in lab_tokens["family"]:
            tiers.append("same family, new genus")
        elif token(t, "o__") in lab_tokens["order"]:
            tiers.append("same order, new family")
        else:
            tiers.append("new order")
    gem["tier"] = tiers

    # ---- predictions
    n = len(gem)
    out = {t: np.full(n, np.nan) for t in TARGET_COLUMNS}
    label = np.empty(n, dtype=object)
    for (lm, s16, c) in combos:
        mask = (has_lm == lm) & (has_s == s16) & (has_c == c)
        if not mask.any():
            continue
        parts = [T[mask]] + ([LM[mask]] if lm else []) + ([S[mask]] if s16 else []) + ([C[mask]] if c else []) + [D[mask]]
        Xm = np.hstack(parts)
        for target in TARGET_COLUMNS:
            out[target][mask] = models[((lm, s16, c), target)].predict(Xm)
        label[mask] = combo_label(lm, s16, c)
    gem["model_used"] = label
    gem["phylogeny_rank_used"] = lm_rank

    acc = accuracy.set_index(["model", "target", "tier"])
    err, lo, hi = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i, (m, tier) in enumerate(zip(gem.model_used, gem.tier)):
        row = acc.loc[(m, "own temperature", tier)]
        err[i] = row["median_error"]
        lo[i], hi[i] = out["own temperature"][i] + row["resid_q10"], out["own temperature"][i] + row["resid_q90"]

    result = pd.DataFrame({
        "genome_id": gem.genome_id,
        "doubling_time_hours_at_growth_temp": np.exp(out["own temperature"]),
        "range_low_hours": np.exp(lo),
        "range_high_hours": np.exp(hi),
        "typical_fold_error": err,
        "predicted_growth_temp_c": out["growth temperature"],
        "doubling_time_hours_at_20C": np.exp(out["20 C"]),
        "hot_organism": out["growth temperature"] >= HOT_C,
        "nearest_labeled_relative": gem.tier,
        "model_used": gem.model_used,
        "phylogeny_rank_used": gem.phylogeny_rank_used,
        "codon_usage_bias": gem.cub,
        "ivywrel_fraction": gem.ivywrel,
        "ribosomal_proteins_found": gem.n_ribosomal,
        "genome_size_bp": gem.genome_size_bp,
        "gc_content": gem.gc_content,
        "trna_count": gem.trna_count,
        "gtdb_taxonomy": gem.gtdb_taxonomy,
        "ecosystem_category": gem.ecosystem_category,
    })
    out_path = processed / args.out
    result.sort_values("doubling_time_hours_at_growth_temp").to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}: {len(result)} predictions")

    ref = lab["doubling_time_hours"]
    d = result["doubling_time_hours_at_growth_temp"]
    print(f"labeled corpus, raw doubling time: median {ref.median():.2f} h, 5th pct {ref.quantile(.05):.2f}, 95th pct {ref.quantile(.95):.1f}")
    print(f"GEM predictions                  : median {d.median():.2f} h, 5th pct {d.quantile(.05):.2f}, 95th pct {d.quantile(.95):.1f}, min {d.min():.2f}, max {d.max():.0f}")
    print(f"share slower than 24 h: {100 * (d > 24).mean():.1f}% (labeled {100 * (ref > 24).mean():.1f}%) | faster than 10 min: {100 * (d < 0.167).mean():.2f}%")
    print(f"predicted growth temperature: median {result.predicted_growth_temp_c.median():.0f} C, share at or above {HOT_C:.0f} C: {100 * result.hot_organism.mean():.1f}%")
    print(result.model_used.value_counts().to_string())
    print(result.nearest_labeled_relative.value_counts(normalize=True).mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
