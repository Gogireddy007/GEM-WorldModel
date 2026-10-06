#!/usr/bin/env python
"""This settles three model choices on the deployment feature set (genome
traits, GTDB landmark phylogeny, 16S landmark distances), always holding whole
species out of training and over the same seven seeds:

  A. which loss: squared error, Huber, or quantile (median)
  B. how to treat species that appear as several genomes with the same label:
     as they are, weighted so each species counts once, or one genome each
  C. whether the 54 extra labeled species (the 15 targeted gap species and the
     39 family-approximated ones) help, measured on the original species only,
     since the extra rows are only ever added to the training folds

Needs labeled_16s_sequences.csv for the extra species (fetch_labeled_16s.py
with both features files) and gtdb_taxon_vectors.csv.
"""

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.data import gtdb
from gem_worldmodel.features import rrna16s, taxon_vectors as tv
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

SEEDS = [42, 1, 7, 13, 2024, 123, 99]
ROUND1 = {
    "Acidothermus cellulolyticus", "Marinithermus hydrothermalis", "Bacillus amyloliquefaciens",
    "Bacillus subtilis", "Acidimicrobium ferrooxidans", "Demequina aestuarii", "Bacillus licheniformis",
    "Desulfobacter curvatus", "Desulfitobacterium dehalogenans", "Shewanella oneidensis",
}
ROUND2 = {
    "Calditerricola satsumensis", "Guyparkeria hydrothermalis", "Tepidiphilus margaritifer",
    "Thermus igniterrae", "Shewanella profunda",
}
TRAITS = ["genome_size_bp", "gc_content", "trna_count"]
LOSSES = {
    "squared error": {},
    "Huber": {"loss": "huber", "alpha": 0.9},
    "quantile (median)": {"loss": "quantile", "alpha": 0.5},
}


def load_data(processed, data_cfg, k):
    base = pd.read_csv(processed / "features_sample_expanded.csv")
    base["origin"] = "base"
    full = pd.read_csv(processed / "features_sample_gapfilled_round2_consistent.csv", low_memory=False)
    extra = full[~full["species"].isin(base["species"])].copy()
    extra["origin"] = np.where(extra["species"].isin(ROUND1), "round1",
                               np.where(extra["species"].isin(ROUND2), "round2", "family"))
    tax = gtdb.fetch_bac_taxonomy(data_cfg)
    lookup = dict(zip(tax["accession_bare"], tax["gtdb_taxonomy"]))
    missing = extra["gtdb_taxonomy"].isna()
    extra.loc[missing, "gtdb_taxonomy"] = extra.loc[missing, "accession"].map(lookup)
    df = pd.concat([base, extra], ignore_index=True)

    table = tv.load_taxon_table(processed / "gtdb_taxon_vectors.csv")
    lm = []
    for t in df["gtdb_taxonomy"]:
        v, _ = tv.lookup(t, table, "family") if isinstance(t, str) else (None, None)
        lm.append(v if v is not None else np.full(16, np.nan))
    lm = np.vstack(lm)

    seqs = pd.read_csv(processed / "labeled_16s_sequences.csv").fillna("")
    profiles = {s: rrna16s.kmer_profile(q, k) for s, q in zip(seqs.species, seqs.sequence) if q}
    landmarks = [profiles[n] for n in np.load(processed / "labeled_16s_landmark_names.npy")]
    s16 = np.vstack([
        np.array([rrna16s.kmer_cosine_distance(profiles[s], lp) for lp in landmarks]) if s in profiles else np.full(16, np.nan)
        for s in df["species"]
    ])
    traits = df[TRAITS].to_numpy(float)
    for block in (traits, s16):
        med = np.nanmedian(block[: len(base)], axis=0)
        block[:] = np.where(np.isnan(block), med, block)
    keep = ~np.isnan(lm).any(axis=1)
    logger.info(f"{keep.sum()} of {len(df)} rows have a phylogeny vector ({(~keep).sum()} dropped)")
    X = np.hstack([traits, lm, s16])
    return df[keep].reset_index(drop=True), X[keep]


def metrics(truth_log, pred_log, species):
    y, p = np.exp(truth_log), np.exp(pred_log)
    out = {
        "R2 log": r2_score(truth_log, pred_log),
        "Spearman": spearmanr(truth_log, pred_log).correlation,
        "fold error": np.exp(np.median(np.abs(truth_log - pred_log))),
        "R2 hours": r2_score(y, p),
    }
    fast = y < 5
    out["fast Spearman"] = spearmanr(truth_log[fast], pred_log[fast]).correlation
    per = pd.DataFrame({"s": species, "t": truth_log, "p": pred_log}).groupby("s").mean()
    out["species R2 log"] = r2_score(per.t, per.p)
    out["species Spearman"] = spearmanr(per.t, per.p).correlation
    return out


def run(df, X, seed, params, mode="as is", extras=(), regions=False):
    base_idx = np.where(df["origin"] == "base")[0]
    extra_idx = np.where(df["origin"].isin(extras))[0]
    y = np.log(df["doubling_time_hours_ref"].to_numpy())
    species = df["species"].to_numpy()
    rng = np.random.default_rng(seed)
    ug = np.unique(species[base_idx])
    perm = dict(zip(ug, rng.permutation(len(ug))))
    groups = np.array([perm[s] for s in species[base_idx]])
    pred = np.full(len(df), np.nan)
    for tr, te in GroupKFold(5).split(base_idx, groups=groups):
        train = np.concatenate([base_idx[tr], extra_idx])
        weight = np.ones(len(train))
        if mode == "weighted":
            counts = pd.Series(species[train]).map(pd.Series(species[train]).value_counts()).to_numpy()
            weight = 1.0 / counts
        elif mode == "one per species":
            keep = pd.Series(species[train]).drop_duplicates().index.to_numpy()
            train, weight = train[keep], weight[keep]
        model = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, **params)
        model.fit(X[train], y[train], sample_weight=weight)
        pred[base_idx[te]] = model.predict(X[base_idx[te]])
    res = metrics(y[base_idx], pred[base_idx], species[base_idx])
    if regions:
        gc, size = df["gc_content"].to_numpy()[base_idx], df["genome_size_bp"].to_numpy()[base_idx]
        in_a = (gc >= 0.64) & (size <= 2.5e6)
        in_b = (gc >= 0.44) & (gc < 0.50) & (size >= 4.0e6) & (size <= 5.7e6)
        reg = in_a | in_b
        res["gap-region fold error"] = np.exp(np.median(np.abs(y[base_idx][reg] - pred[base_idx][reg])))
        res["gap-region Spearman"] = spearmanr(y[base_idx][reg], pred[base_idx][reg]).correlation
        res["_n_region"] = int(reg.sum())
    return res


def table(results: dict, baseline: str, cols):
    base = results[baseline]
    print(f"{'':32s}" + "".join(f"{c:>18s}" for c in cols))
    for name, runs in results.items():
        mean = pd.DataFrame(runs).mean()
        cells = []
        for c in cols:
            better = sum(
                (r[c] < b[c]) if c == "fold error" or c == "gap-region fold error" else (r[c] > b[c])
                for r, b in zip(runs, base)
            )
            tag = "" if name == baseline else f" ({better}/7)"
            cells.append(f"{mean[c]:.3f}{tag}")
        print(f"{name:32s}" + "".join(f"{x:>18s}" for x in cells))


def main():
    argparse.ArgumentParser().parse_args()
    data_cfg = load_config("data")
    processed = resolve_path(data_cfg["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X = load_data(processed, data_cfg, k)
    n_base = (df["origin"] == "base").sum()
    logger.info(f"{n_base} original rows, {df.loc[df.origin == 'base', 'species'].nunique()} species")

    cols = ["R2 log", "Spearman", "fold error", "fast Spearman", "species R2 log", "species Spearman", "R2 hours"]

    print("\nA. LOSS FUNCTION (species held out, mean of 7 seeds; in brackets: seeds better than squared error)")
    res = {n: [run(df, X, s, p) for s in SEEDS] for n, p in LOSSES.items()}
    table(res, "squared error", cols)

    print("\nB. REPEATED SPECIES, squared error (in brackets: seeds better than 'as is')")
    res = {m: [run(df, X, s, {}, mode=m) for s in SEEDS] for m in ("as is", "weighted", "one per species")}
    table(res, "as is", cols)

    print("\nC. EXTRA TRAINING SPECIES, squared error, scored on the original species only")
    configs = {
        "none (original only)": (),
        "+ 10 round-1 gap species": ("round1",),
        "+ 5 round-2 gap species": ("round2",),
        "+ both rounds (15)": ("round1", "round2"),
        "+ 39 family-approximated": ("family",),
        "+ all 54": ("round1", "round2", "family"),
    }
    res = {n: [run(df, X, s, {}, extras=e, regions=True) for s in SEEDS] for n, e in configs.items()}
    table(res, "none (original only)", cols + ["gap-region fold error", "gap-region Spearman"])
    print(f"(rows of the original corpus inside the two gap regions: {res['none (original only)'][0]['_n_region']})")


if __name__ == "__main__":
    main()
