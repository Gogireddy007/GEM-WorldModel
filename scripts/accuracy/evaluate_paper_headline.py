#!/usr/bin/env python
"""The accuracy reported in the paper: the full deployment procedure repeated inside every
held-out split, so that no number is optimistic.

The deployed model (predict_gem_genomes_gbm.py) is built in two steps: its settings are chosen on
all labeled rows (tune_deployed_model.py), and it is then trained on every labeled genome four
times, with the phylogeny vector looked up at the genome's species, genus, family and order, so
that it learns what a coarse placement looks like. evaluate_final_model.py, which gives the error
labels of the GEM predictions, does neither inside its splits: it uses the settings chosen on all
rows and trains each genome once. Here three schemes are compared on the same rows and folds:

  fixed, single     settings from configs/deployed_model.yaml, each genome once (evaluate_final_model.py)
  fixed, coarsened  the same settings, each training genome four times (the deployed training scheme)
  nested            settings chosen by the deployed search on the training rows of each split only,
                    then trained four times; the test rows never take part. This is the headline.

Test genomes always get the phylogeny vector of their own species, as a GEM genome does. Only the
full model (traits, phylogeny, 16S and gene features) is evaluated, on the rows that have gene
features. Whole species groups, genera, families or orders are held out (GroupKFold(5), groups
shuffled per seed).

Writes paper_headline_accuracy.csv (one row per scheme, target, level and seed) and
paper_headline_predictions.csv (held-out predictions of every scheme), and with --summary prints
the means.
"""

import argparse
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from gem_worldmodel.features import deploy_features
from gem_worldmodel.features import taxon_vectors as tv
from gem_worldmodel.features.deploy_features import coarsen
from gem_worldmodel.models.deployed import DeployedRegressor
from gem_worldmodel.utils.config import load_config, resolve_path

import tune_deployed_model as tuning

RANKS = ("species", "genus", "family", "order")
PREFIX = {"genus": "g__", "family": "f__", "order": "o__"}
TARGETS = {"own temperature": "doubling_time_hours", "20 C": "doubling_time_hours_ref", "growth temperature": "growth_temp_c"}
SCHEMES = ("fixed, single", "fixed, coarsened", "nested")
SLOW = np.log(24.0)

DATA = {}


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def load():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, names = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True)
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    table = tv.load_taxon_table(processed / "gtdb_taxon_vectors.csv")
    lm_cols = [i for i, n in enumerate(names) if n.startswith("lm_")]
    copies = []
    for r in RANKS:
        Xr = X.copy()
        Xr[:, lm_cols] = np.vstack([tv.lookup(coarsen(t, r), table, "order")[0] for t in df["gtdb_taxonomy"]])
        copies.append(Xr)
    return processed, df, X, copies


def init(state):
    # one thread per worker: histogram boosting otherwise starts one OpenMP thread per core in every worker
    threadpool_limits(1)
    DATA.update(state)


def run_fold(job):
    """Held-out predictions of every scheme for one split; returns (job key, test rows, {scheme: predictions}, settings)."""
    target, level, seed, fold, tr, te = job
    X, copies, y, family = DATA["X"], DATA["copies"], DATA["y"][target], DATA["family"]
    ok = ~np.isnan(y)
    tr, te = tr[ok[tr]], te[ok[te]]
    coarse = np.vstack([c[tr] for c in copies])
    coarse_y = np.tile(y[tr], len(copies))
    out = {}
    out["fixed, single"] = DeployedRegressor(target, seed, n_jobs=1).fit(X[tr], y[tr]).predict(X[te])
    out["fixed, coarsened"] = DeployedRegressor(target, seed, n_jobs=1).fit(coarse, coarse_y).predict(X[te])
    settings, _ = tuning.choose_settings(X[tr], y[tr], family[tr], n_jobs=1)
    out["nested"] = DeployedRegressor(target, seed, settings=settings, n_jobs=1).fit(coarse, coarse_y).predict(X[te])
    return (target, level, seed, fold), te, out, settings


def metrics(target, y, p):
    if target == "growth temperature":
        return {"mean abs error": float(np.mean(np.abs(y - p))), "Spearman": spearmanr(y, p).correlation}
    resid = y - p
    row = {"typical error": float(np.exp(np.median(np.abs(resid)))), "Spearman": spearmanr(y, p).correlation,
           "resid_q10": float(np.quantile(resid, 0.10)), "resid_q90": float(np.quantile(resid, 0.90))}
    if target == "own temperature":
        row["AUC slow"] = roc_auc_score(y > SLOW, p)
    return row


def summary(processed):
    out = pd.read_csv(processed / "paper_headline_accuracy.csv")
    for target in TARGETS:
        d = out[out["target"] == target]
        cols = [c for c in ("typical error", "mean abs error", "Spearman", "AUC slow") if c in d and d[c].notna().any()]
        print(f"\n===== {target}")
        print(d.groupby(["held_out", "scheme"], sort=False)[cols].mean().round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    processed, df, X, copies = load()
    if args.summary:
        summary(processed)
        return
    y = {}
    for target, col in TARGETS.items():
        v = df[col].to_numpy(float)
        y[target] = v if target == "growth temperature" else np.log(v)
    family = df["gtdb_taxonomy"].map(lambda t: token(t, "f__")).to_numpy()
    jobs = []
    for level in ("species", "genus", "family", "order"):
        groups = df["species_group"].to_numpy() if level == "species" else df["gtdb_taxonomy"].map(lambda t, p=PREFIX[level]: token(t, p)).to_numpy()
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            ug = np.unique(groups)
            perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
            g = np.array([perm[v] for v in groups])
            for fold, (tr, te) in enumerate(GroupKFold(5).split(X, groups=g)):
                for target in TARGETS:
                    jobs.append((target, level, seed, fold, tr, te))
    print(f"{len(df)} rows, {len(jobs)} splits, {args.workers} workers", flush=True)
    preds = {}
    chosen = []
    with Pool(args.workers, initializer=init, initargs=({"X": X, "copies": copies, "y": y, "family": family},)) as pool:
        for i, (key, te, out, settings) in enumerate(pool.imap_unordered(run_fold, jobs), 1):
            target, level, seed, fold = key
            for scheme, p in out.items():
                arr = preds.setdefault((target, level, seed, scheme), np.full(len(df), np.nan))
                arr[te] = p
            chosen.append({"target": target, "held_out": level, "seed": seed, "fold": fold,
                           **{f"{kind} {k}": str(v) for kind, params in settings.items() for k, v in params.items()}})
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} splits done", flush=True)
    rows, pred_rows = [], []
    for (target, level, seed, scheme), p in preds.items():
        ok = ~np.isnan(p)
        rows.append({"scheme": scheme, "target": target, "held_out": level, "seed": seed, "n_rows": int(ok.sum()),
                     **metrics(target, y[target][ok], p[ok])})
        pred_rows.append(pd.DataFrame({"species": df["species"][ok], "species_group": df["species_group"][ok],
                                       "gtdb_taxonomy": df["gtdb_taxonomy"][ok], "target": target, "held_out": level,
                                       "seed": seed, "scheme": scheme, "measured": y[target][ok], "predicted": p[ok]}))
    order = {s: i for i, s in enumerate(SCHEMES)}
    out = pd.DataFrame(rows).sort_values(["target", "held_out", "scheme", "seed"], key=lambda c: c.map(order) if c.name == "scheme" else c)
    out.to_csv(processed / "paper_headline_accuracy.csv", index=False)
    pd.concat(pred_rows).to_csv(processed / "paper_headline_predictions.csv", index=False)
    pd.DataFrame(chosen).to_csv(processed / "paper_headline_settings.csv", index=False)
    summary(processed)


if __name__ == "__main__":
    main()
