#!/usr/bin/env python
"""This evaluates the extra Madin species (build_madin_extension.py) two ways,
with the median-loss deployment model and the deployment features.

  1. As an independent test set: train on the original corpus only and predict
     the extension species, which the model has never seen. Reported overall,
     by how precisely the species could be placed in GTDB, and by how close its
     nearest relative in the training corpus is.
  2. As extra training data: add them to the training folds only, hold whole
     species, genera, families or orders of the original corpus out, and score
     on the original species. Extension species that share a held-out lineage
     with the test rows are dropped from that fold, so they cannot supply
     relatives that the held-out level is meant to remove.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import rrna16s, taxon_vectors as tv
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

SEEDS = [42, 1, 7, 13, 2024, 123, 99]
TRAITS = ["genome_size_bp", "gc_content", "trna_count"]
PARAMS = {"loss": "quantile", "alpha": 0.5}
PREFIX = {"genus": "g__", "family": "f__", "order": "o__"}


def token(taxonomy, prefix):
    m = re.search(rf"{prefix}[^;]*", str(taxonomy))
    return m.group(0) if m and m.group(0) != prefix else None


def features(df, table, profiles, landmarks, medians):
    traits = np.where(np.isnan(df[TRAITS].to_numpy(float)), medians["t"], df[TRAITS].to_numpy(float))
    lm = np.vstack([
        (tv.lookup(t, table, "family")[0] if tv.lookup(t, table, "family")[0] is not None else np.full(16, np.nan))
        for t in df["gtdb_taxonomy"]
    ])
    s16 = np.vstack([
        np.array([rrna16s.kmer_cosine_distance(profiles[s], lp) for lp in landmarks]) if s in profiles else medians["s"]
        for s in df["species"]
    ])
    return np.hstack([traits, lm, s16])


def metrics(truth, pred):
    return {
        "R2 log": r2_score(truth, pred),
        "Spearman": spearmanr(truth, pred).correlation,
        "fold error": np.exp(np.median(np.abs(truth - pred))),
    }


def main():
    data_cfg = load_config("data")
    processed = resolve_path(data_cfg["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    table = tv.load_taxon_table(processed / "gtdb_taxon_vectors.csv")
    base = pd.read_csv(processed / "features_sample_expanded.csv")
    ext = pd.read_csv(processed / "madin_extension.csv")
    seqs = pd.read_csv(processed / "labeled_16s_sequences.csv").fillna("")
    profiles = {s: rrna16s.kmer_profile(q, k) for s, q in zip(seqs.species, seqs.sequence) if q}
    landmarks = [profiles[n] for n in np.load(processed / "labeled_16s_landmark_names.npy")]
    s_base = np.vstack([
        np.array([rrna16s.kmer_cosine_distance(profiles[s], lp) for lp in landmarks]) if s in profiles else np.full(16, np.nan)
        for s in base["species"]
    ])
    medians = {"t": np.nanmedian(base[TRAITS].to_numpy(float), axis=0), "s": np.nanmedian(s_base, axis=0)}

    Xb = features(base, table, profiles, landmarks, medians)
    Xe = features(ext, table, profiles, landmarks, medians)
    ok_e = ~np.isnan(Xe).any(axis=1)
    ext, Xe = ext[ok_e].reset_index(drop=True), Xe[ok_e]
    have16 = ext["species"].isin(profiles).sum()
    logger.info(f"{len(ext)} extension species usable ({have16} with a 16S sequence), {len(base)} original rows")
    yb, ye = np.log(base["doubling_time_hours_ref"].to_numpy()), np.log(ext["doubling_time_hours_ref"].to_numpy())

    # ---- 1. independent test set
    base_tax = base["gtdb_taxonomy"]
    seen = {lvl: set(base_tax.map(lambda t, p=p: token(t, p)).dropna()) for lvl, p in PREFIX.items()}
    level = []
    for t in ext["gtdb_taxonomy"]:
        if token(t, "g__") in seen["genus"]:
            level.append("same genus in training")
        elif token(t, "f__") in seen["family"]:
            level.append("same family, new genus")
        elif token(t, "o__") in seen["order"]:
            level.append("same order, new family")
        else:
            level.append("new order")
    ext["relative"] = level
    preds = np.mean(
        [GradientBoostingRegressor(random_state=s, n_estimators=200, max_depth=3, **PARAMS).fit(Xb, yb).predict(Xe) for s in SEEDS],
        axis=0,
    )
    print("\n1. INDEPENDENT TEST: trained on the original corpus only, predicting species it never saw")
    print(f"{'':32s}{'n':>5s}{'R2 log':>9s}{'Spearman':>10s}{'fold error':>12s}")
    m = metrics(ye, preds)
    print(f"{'all extension species':32s}{len(ext):5d}{m['R2 log']:9.3f}{m['Spearman']:10.3f}{m['fold error']:11.2f}x")
    for col, vals in (("match_rank", ["species", "genus", "family"]), ("relative", ["same genus in training", "same family, new genus", "same order, new family", "new order"])):
        for v in vals:
            mask = (ext[col] == v).to_numpy()
            if mask.sum() >= 5:
                m = metrics(ye[mask], preds[mask])
                print(f"{col + ': ' + v:32s}{mask.sum():5d}{m['R2 log']:9.3f}{m['Spearman']:10.3f}{m['fold error']:11.2f}x")

    # ---- 2. as extra training data
    species_level = (ext["match_rank"] == "species").to_numpy()
    configs = {"none": np.zeros(len(ext), bool), "+ all extension species": np.ones(len(ext), bool), "+ species-level placements only": species_level}
    print("\n2. EXTRA TRAINING DATA, scored on the original species (mean of 7 seeds; in brackets: seeds better than none)")
    print(f"{'held out':9s}{'extra species':34s}{'R2 log':>14s}{'Spearman':>14s}{'fold error':>16s}")
    for lvl in ("species", "genus", "family", "order"):
        groups_all = base["species"].to_numpy() if lvl == "species" else base_tax.map(lambda t, p=PREFIX.get(lvl): token(t, p)).fillna("none").to_numpy()
        ext_tok = ext["gtdb_taxonomy"].map(lambda t, p=PREFIX.get(lvl): token(t, p)) if lvl != "species" else ext["species"]
        results = {n: [] for n in configs}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            ug = np.unique(groups_all)
            perm = dict(zip(ug, rng.permutation(len(ug))))
            g = np.array([perm[v] for v in groups_all])
            for name, use in configs.items():
                pred = np.full(len(base), np.nan)
                for tr, te in GroupKFold(5).split(Xb, yb, g):
                    test_tokens = set(groups_all[te])
                    keep = use & ~ext_tok.isin(test_tokens).to_numpy()
                    Xtr = np.vstack([Xb[tr], Xe[keep]])
                    ytr = np.concatenate([yb[tr], ye[keep]])
                    pred[te] = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, **PARAMS).fit(Xtr, ytr).predict(Xb[te])
                results[name].append(metrics(yb, pred))
        base_runs = results["none"]
        for name, runs in results.items():
            mean = pd.DataFrame(runs).mean()
            cells = []
            for c in ("R2 log", "Spearman", "fold error"):
                better = sum((r[c] < b[c]) if c == "fold error" else (r[c] > b[c]) for r, b in zip(runs, base_runs))
                cells.append(f"{mean[c]:.3f}" + ("" if name == "none" else f" ({better}/7)"))
            print(f"{lvl:9s}{name:34s}{cells[0]:>14s}{cells[1]:>14s}{cells[2]:>16s}")
        print()


if __name__ == "__main__":
    main()
