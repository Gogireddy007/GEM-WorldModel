#!/usr/bin/env python
"""This rebuilds the gap-filled corpus with every centroid-approximated row
re-derived in the same phylogeny coordinate system as the exact tips, then
benchmarks each corpus variant against Phydon and gRodon on the same seeds.

The earlier gap-filling runs recomputed the exact-tip embeddings but left the
approximated rows in the old coordinate system, so every corpus built that way
(353 and 358 species) had a corrupted phylogeny branch. This repeats the
comparisons with that fixed, and includes the original 304-species file as the
historical reference.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from gem_worldmodel.data import gtdb
from gem_worldmodel.eval.benchmark import compute_metrics
from gem_worldmodel.features import phylogeny
from gem_worldmodel.training.baselines import GRodonBaseline, PhydonBaseline
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.raw_baseline import concat_branch_tensors
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

ROUND1 = {
    "Acidothermus cellulolyticus", "Marinithermus hydrothermalis", "Bacillus amyloliquefaciens",
    "Bacillus subtilis", "Acidimicrobium ferrooxidans", "Demequina aestuarii", "Bacillus licheniformis",
    "Desulfobacter curvatus", "Desulfitobacterium dehalogenans", "Shewanella oneidensis",
}
ROUND2 = {
    "Calditerricola satsumensis", "Guyparkeria hydrothermalis", "Tepidiphilus margaritifer",
    "Thermus igniterrae", "Shewanella profunda",
}
SEEDS = [42, 1, 7, 13, 2024, 123, 99]


def repair(df: pd.DataFrame, data_cfg: dict) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("gtdb_dist_")]
    exact = df[df["placement_type"] == "exact_tip"]
    tip_embeddings = {r["accession"]: r[cols].to_numpy(dtype=float) for _, r in exact.iterrows()}
    taxonomy = gtdb.fetch_bac_taxonomy(data_cfg)
    lookup = dict(zip(taxonomy["accession_bare"], taxonomy["gtdb_taxonomy"]))
    tip_taxonomy = {a: lookup[a] for a in tip_embeddings if a in lookup}
    rest = df[df["placement_type"] != "exact_tip"]
    fixed = phylogeny.recompute_centroid_rows(rest, tip_embeddings, tip_taxonomy, cols)
    out = df.copy()
    out.loc[fixed.index, cols] = fixed[cols]
    return out


def cross_validate_baseline(baseline_cls, df, fold_id):
    oof = np.full(len(df), np.nan)
    for fold in sorted(set(fold_id)):
        mask = fold_id == fold
        oof[mask] = baseline_cls().fit(df[~mask]).predict(df[mask])
    return oof


def evaluate(df: pd.DataFrame, model_cfg: dict, split_hours: float) -> pd.DataFrame:
    branches = [b["name"] for b in model_cfg["branches"]]
    y = df["doubling_time_hours_ref"].to_numpy()
    target_log = np.log(y)
    strat = (y < split_hours).astype(int)
    tensors, _ = build_branch_tensors(df, model_cfg, branches=branches)
    x = concat_branch_tensors(tensors).numpy()
    rows = []
    for seed in SEEDS:
        fold_id = np.full(len(df), -1)
        oof = {"gbm_squared": np.full(len(df), np.nan), "gbm_huber": np.full(len(df), np.nan)}
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(x, strat)):
            fold_id[te] = fold
            for name, kw in (("gbm_squared", {}), ("gbm_huber", {"loss": "huber", "alpha": 0.9})):
                m = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, **kw)
                m.fit(x[tr], target_log[tr])
                oof[name][te] = m.predict(x[te])
        preds = {k: np.exp(v) for k, v in oof.items()}
        preds["phydon"] = cross_validate_baseline(PhydonBaseline, df, fold_id)
        preds["grodon"] = cross_validate_baseline(GRodonBaseline, df, fold_id)
        for name, p in preds.items():
            ok = ~np.isnan(p)
            allm = compute_metrics(y[ok], p[ok])
            fm = ok & (y < split_hours)
            fastm = compute_metrics(y[fm], p[fm])
            rows.append({"seed": seed, "model": name, "r2": allm["r2"], "spearman": allm["spearman"], "fast_r2": fastm["r2"]})
    return pd.DataFrame(rows)


def main():
    argparse.ArgumentParser().parse_args()
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    processed = resolve_path(data_cfg["paths"]["processed_dir"])
    split_hours = data_cfg["doubling_time_split_hours"]

    orig304 = pd.read_csv(processed / "features_sample_expanded.csv")
    full = repair(pd.read_csv(processed / "features_sample_gapfilled_round2.csv"), data_cfg)
    full.to_csv(processed / "features_sample_gapfilled_round2_consistent.csv", index=False)

    base_mask = (full["placement_type"] != "family_centroid_approx") & ~full["species"].isin(ROUND1 | ROUND2)
    fam_mask = full["placement_type"] == "family_centroid_approx"
    variants = {
        "304 original file (historical reference)": orig304,
        "304 control (same space as the rest)": full[base_mask],
        "304 + 10 round-1 gap species": full[base_mask | full["species"].isin(ROUND1)],
        "304 + 5 round-2 gap species": full[base_mask | full["species"].isin(ROUND2)],
        "304 + both rounds (15)": full[base_mask | full["species"].isin(ROUND1 | ROUND2)],
        "304 + 39 family-approximated": full[base_mask | fam_mask],
        "everything (358)": full,
    }
    results = {}
    for name, d in variants.items():
        logger.info(f"evaluating: {name} (n={len(d)})")
        r = evaluate(d.reset_index(drop=True), model_cfg, split_hours)
        r["variant"] = name
        r["n"] = len(d)
        results[name] = r
    all_r = pd.concat(results.values())
    all_r.to_csv(processed / "corpus_variant_results.csv", index=False)

    print(f"{'variant':44s} {'n':>4s} | {'model':11s} {'R2':>7s} {'Spearman':>9s} {'fast R2':>9s} | R2 wins vs Phydon")
    for name, r in results.items():
        piv = r.pivot(index="seed", columns="model", values="r2")
        for model in ("gbm_squared", "gbm_huber", "phydon", "grodon"):
            sub = r[r["model"] == model]
            wins = f"{int((piv[model] > piv['phydon']).sum())}/7" if model.startswith("gbm") else ""
            print(f"{name:44s} {len(variants[name]):4d} | {model:11s} {sub.r2.mean():7.3f} {sub.spearman.mean():9.3f} {sub.fast_r2.mean():9.1f} | {wins}")
        print()


if __name__ == "__main__":
    main()
