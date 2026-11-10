#!/usr/bin/env python
"""Ablations of the final model for the paper: each part removed (or a rejected part added) against the
deployed model itself, on the same rows, folds and seeds.

The earlier ablation scripts (evaluate_grodon_and_temperature.py, evaluate_energy_features.py,
evaluate_added_species.py) each compared against the model of their day, so their baselines differ.
Here every variant is the deployed model (DeployedRegressor, settings from configs/deployed_model.yaml)
with one change:

  - gRodon statistics      without the four statistics of the released gRodon
  - lifestyle genes        without the 14 lifestyle inputs
  - all gene features      without codon usage, IVYWREL, lifestyle genes and gRodon statistics
  - 16S                    without the 16 16S inputs
  - phylogeny              without the 16 GTDB landmark inputs
  - added species          trained without the Madin genome extension and the literature strains
                           (test rows unchanged, so every variant is scored on the same genomes)
  + energy features        with the energy inputs of lifestyle_genes.energy_features
  + predicted temperature  with a predicted growth temperature (out of fold inside the training rows)

Settings were chosen for the full model, which can favour it slightly over a variant; the learner
itself is compared nested in evaluate_tuning.py. Each genome is trained once (not once per placement
rank), since the comparison is relative. Whole species groups, families or orders are held out,
GroupKFold(5), groups shuffled per seed.

Writes ablation_evaluation.csv (per variant, level and seed) and prints the mean change against the
final model with the seeds in which the variant was better and a paired Wilcoxon p.
"""

import argparse
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from gem_worldmodel.features import deploy_features, lifestyle_genes
from gem_worldmodel.models.deployed import DeployedRegressor
from gem_worldmodel.utils.config import load_config, resolve_path

SLOW = np.log(24.0)
PREFIX = {"family": "f__", "order": "o__"}
ADDED = ("Madin genome extension", "literature")
VARIANTS = ["final model", "- gRodon statistics", "- lifestyle genes", "- all gene features", "- 16S", "- phylogeny",
            "- added species", "+ energy features", "+ predicted temperature"]
METRICS = ["typical error", "Spearman", "AUC slow"]
DATA = {}


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def split(groups, seed, n=5):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    return list(GroupKFold(n).split(groups, groups=np.array([perm[v] for v in groups])))


def predicted_temperature(X, temp, species, tr, te, seed):
    """Predicted growth temperature: out of fold for the training rows, from the training-fold model for the test rows."""
    out = np.full(len(temp), np.nan)
    known = tr[~np.isnan(temp[tr])]
    out[te] = DeployedRegressor("growth temperature", seed, n_jobs=1).fit(X[known], temp[known]).predict(X[te])
    for itr, ite in split(species[tr], seed + 1000):
        inner_known = tr[itr][~np.isnan(temp[tr[itr]])]
        out[tr[ite]] = DeployedRegressor("growth temperature", seed, n_jobs=1).fit(X[inner_known], temp[inner_known]).predict(X[tr[ite]])
    return out


def init(state):
    threadpool_limits(1)  # one thread per worker
    DATA.update(state)


def run_fold(job):
    level, seed, tr, te = job
    X, E, y, temp, species, cols, added = (DATA[k] for k in ("X", "E", "y", "temp", "species", "cols", "added"))
    out = {}
    for name in VARIANTS:
        Xv, rows = X, tr
        if name in cols:
            Xv = np.delete(X, cols[name], axis=1)
        elif name == "- added species":
            rows = tr[~added[tr]]
        elif name == "+ energy features":
            Xv = np.hstack([X, E])
        elif name == "+ predicted temperature":
            Xv = np.hstack([X, predicted_temperature(X, temp, species, tr, te, seed)[:, None]])
        out[name] = DeployedRegressor("own temperature", seed, n_jobs=1).fit(Xv[rows], y[rows]).predict(Xv[te])
    return level, seed, te, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, names = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True, with_energy=True)
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    energy = [names.index(c) for c in lifestyle_genes.ENERGY_FEATURES]
    E, X = X[:, energy], np.delete(X, energy, axis=1)
    names = [n for i, n in enumerate(names) if i not in set(energy)]
    idx = {n: i for i, n in enumerate(names)}
    cols = {"- gRodon statistics": [idx[c] for c in deploy_features.GRODON_FEATURES],
            "- lifestyle genes": [idx[c] for c in lifestyle_genes.FEATURES],
            "- all gene features": [idx[c] for c in deploy_features.GENE_FEATURES],
            "- 16S": [i for n, i in idx.items() if n.startswith("s16_")],
            "- phylogeny": [i for n, i in idx.items() if n.startswith("lm_")]}
    _, X_dep, _ = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True)
    assert np.allclose(X_dep[use], X), "the energy inputs must be the only addition to the deployed inputs"
    y = np.log(df["doubling_time_hours"].to_numpy(float))
    species = df["species_group"].to_numpy()
    state = {"X": X, "E": E, "y": y, "temp": df["growth_temp_c"].to_numpy(float), "species": species, "cols": cols,
             "added": df["origin"].isin(ADDED).to_numpy()}
    jobs = []
    for level in ("species", "family", "order"):
        groups = species if level == "species" else df["gtdb_taxonomy"].map(lambda t, p=PREFIX[level]: token(t, p)).to_numpy()
        for seed in range(args.seeds):
            jobs += [(level, seed, tr, te) for tr, te in split(groups, seed)]
    print(f"{len(y)} rows, {X.shape[1]} inputs, {len(jobs)} splits", flush=True)
    preds = {}
    with Pool(args.workers, initializer=init, initargs=(state,)) as pool:
        for i, (level, seed, te, out) in enumerate(pool.imap_unordered(run_fold, jobs), 1):
            for name, p in out.items():
                preds.setdefault((level, seed, name), np.full(len(y), np.nan))[te] = p
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} splits done", flush=True)
    rows = [{"held_out": level, "seed": seed, "setup": name, "typical error": float(np.exp(np.median(np.abs(y - p)))),
             "Spearman": spearmanr(y, p).correlation, "AUC slow": roc_auc_score(y > SLOW, p)}
            for (level, seed, name), p in preds.items()]
    out = pd.DataFrame(rows).sort_values(["held_out", "setup", "seed"])
    out.to_csv(processed / "ablation_evaluation.csv", index=False)
    for level in ("species", "family", "order"):
        d = out[out["held_out"] == level]
        piv = d.pivot_table(index="seed", columns="setup", values=METRICS)
        print(f"\n===== {level.upper()} HELD OUT, {args.seeds} seeds, {len(y)} rows")
        print(d.groupby("setup")[METRICS].mean().reindex(VARIANTS).round(3).to_string())
        for name in VARIANTS[1:]:
            parts = []
            for m in METRICS:
                diff = piv[m][name] - piv[m]["final model"]
                better = int((diff < 0).sum()) if m == "typical error" else int((diff > 0).sum())
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                parts.append(f"{m} {diff.mean():+.3f} (variant better {better}/{args.seeds}, p {p:.3f})")
            print(f"  {name:24s} " + "; ".join(parts))


if __name__ == "__main__":
    main()
