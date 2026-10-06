#!/usr/bin/env python
"""This evaluates adding archaea to the labeled set, with the median-loss
deployment model, the deployment features and a domain flag (bacteria and
archaea have separate landmark vectors from separate GTDB trees).

  A. Accuracy on archaeal species with whole species, genera, families or
     orders held out, with and without archaeal species in training. Without
     them the model has only ever seen bacteria.
  B. Whether adding archaea changes accuracy on bacterial species (species
     held out), which it should not.

Needs madin_extension.csv built with archaea and 16S sequences for them
(fetch_labeled_16s.py), and gtdb_taxon_vectors.csv with both domains.
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


def main():
    data_cfg = load_config("data")
    processed = resolve_path(data_cfg["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    table = tv.load_taxon_table(processed / "gtdb_taxon_vectors.csv")
    keep = ["species", "gtdb_taxonomy", "doubling_time_hours_ref"] + TRAITS
    base = pd.read_csv(processed / "features_sample_expanded.csv")[keep]
    ext = pd.read_csv(processed / "madin_extension.csv")[keep]
    df = pd.concat([base, ext], ignore_index=True)
    df["archaeon"] = df["gtdb_taxonomy"].map(tv.is_archaeon)

    seqs = pd.read_csv(processed / "labeled_16s_sequences.csv").fillna("")
    profiles = {s: rrna16s.kmer_profile(q, k) for s, q in zip(seqs.species, seqs.sequence) if q}
    landmarks = [profiles[n] for n in np.load(processed / "labeled_16s_landmark_names.npy")]
    s16 = np.vstack([
        np.array([rrna16s.kmer_cosine_distance(profiles[s], lp) for lp in landmarks]) if s in profiles else np.full(16, np.nan)
        for s in df["species"]
    ])
    lm = np.vstack([
        (tv.lookup(t, table, "family")[0] if tv.lookup(t, table, "family")[0] is not None else np.full(16, np.nan))
        for t in df["gtdb_taxonomy"]
    ])
    traits = df[TRAITS].to_numpy(float)
    bac = df["archaeon"].to_numpy() == 0
    for block in (traits, s16):
        med = np.nanmedian(block[bac], axis=0)
        block[:] = np.where(np.isnan(block), med, block)
    ok = ~np.isnan(lm).any(axis=1)
    X = np.hstack([traits, lm, s16, df[["archaeon"]].to_numpy()])
    df, X = df[ok].reset_index(drop=True), X[ok]
    y = np.log(df["doubling_time_hours_ref"].to_numpy())
    arch = df["archaeon"].to_numpy() == 1
    logger.info(f"{(~arch).sum()} bacterial rows, {arch.sum()} archaeal rows ({df.loc[arch, 'species'].isin(profiles).sum()} with 16S)")

    def fit_predict(train, test, seed):
        m = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, **PARAMS)
        return m.fit(X[train], y[train]).predict(X[test])

    def score(truth, pred):
        return {"R2 log": r2_score(truth, pred), "Spearman": spearmanr(truth, pred).correlation, "fold error": np.exp(np.median(np.abs(truth - pred)))}

    a_idx, b_idx = np.where(arch)[0], np.where(~arch)[0]
    print("\nA. ARCHAEAL SPECIES AS THE TEST SET (n=%d), whole lineage held out, mean of 7 seeds" % len(a_idx))
    print(f"{'held out':9s}{'training data':34s}{'R2 log':>10s}{'Spearman':>11s}{'fold error':>12s}")
    for lvl in ("species", "genus", "family", "order"):
        groups = df["species"].to_numpy() if lvl == "species" else df["gtdb_taxonomy"].map(lambda t, p=PREFIX.get(lvl): token(t, p)).fillna("none").to_numpy()
        res = {"bacteria only (no archaea)": [], "bacteria + other archaea": []}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            ug = np.unique(groups[a_idx])
            perm = dict(zip(ug, rng.permutation(len(ug))))
            g = np.array([perm[v] for v in groups[a_idx]])
            preds = {n: np.full(len(a_idx), np.nan) for n in res}
            for tr, te in GroupKFold(5).split(a_idx, groups=g):
                test_tokens = set(groups[a_idx[te]])
                bac_train = b_idx[~pd.Series(groups[b_idx]).isin(test_tokens).to_numpy()] if lvl != "species" else b_idx
                preds["bacteria only (no archaea)"][te] = fit_predict(bac_train, a_idx[te], seed)
                preds["bacteria + other archaea"][te] = fit_predict(np.concatenate([bac_train, a_idx[tr]]), a_idx[te], seed)
            for n in res:
                res[n].append(score(y[a_idx], preds[n]))
        base_runs = res["bacteria only (no archaea)"]
        for n, runs in res.items():
            mean = pd.DataFrame(runs).mean()
            cells = []
            for c in ("R2 log", "Spearman", "fold error"):
                better = sum((r[c] < b[c]) if c == "fold error" else (r[c] > b[c]) for r, b in zip(runs, base_runs))
                cells.append(f"{mean[c]:.3f}" + ("" if n.startswith("bacteria only") else f" ({better}/7)"))
            print(f"{lvl:9s}{n:34s}{cells[0]:>10s}{cells[1]:>11s}{cells[2]:>12s}")
        print()

    print("B. BACTERIAL SPECIES AS THE TEST SET (n=%d), species held out, with and without archaea in training" % len(b_idx))
    res = {"without archaea": [], "with archaea": []}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sp = df["species"].to_numpy()[b_idx]
        ug = np.unique(sp)
        perm = dict(zip(ug, rng.permutation(len(ug))))
        g = np.array([perm[v] for v in sp])
        pw, pa = np.full(len(b_idx), np.nan), np.full(len(b_idx), np.nan)
        for tr, te in GroupKFold(5).split(b_idx, groups=g):
            pw[te] = fit_predict(b_idx[tr], b_idx[te], seed)
            pa[te] = fit_predict(np.concatenate([b_idx[tr], a_idx]), b_idx[te], seed)
        res["without archaea"].append(score(y[b_idx], pw))
        res["with archaea"].append(score(y[b_idx], pa))
    for n, runs in res.items():
        mean = pd.DataFrame(runs).mean()
        print(f"  {n:18s} R2 log {mean['R2 log']:.3f}  Spearman {mean['Spearman']:.3f}  fold error {mean['fold error']:.3f}x")


if __name__ == "__main__":
    main()
