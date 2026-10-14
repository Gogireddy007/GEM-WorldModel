#!/usr/bin/env python
"""This compares the final model with reimplementations of the published tools
on the final data, on identical folds, with whole species, genera, families or
orders held out.

  ours          gradient-boosted trees, median loss, genome traits, GTDB landmark
                phylogeny, 16S landmark distances, codon usage bias, IVYWREL and
                the archaea flag
  gRodon-style  linear regression of log doubling time on codon usage bias
  Phydon-style  Phydon's design: codon-based estimate blended with the growth
                rates of the nearest training relatives, weighted by how far the
                nearest relative is (training/baselines.py:PhydonStyleBaseline)
  old stand-in  gradient-boosted trees on codon usage bias, size, GC and tRNA,
                the "Phydon" used in earlier benchmarks here, kept for continuity

All are refit on our training folds, so this compares methods, not the
published fits. Both targets are scored: the doubling time at the organism's
own temperature and the doubling time scaled to 20 C (which is what Phydon's
database reports). Writes published_comparison.csv.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.training.baselines import PhydonStyleBaseline
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = [42, 1, 7, 13, 2024]
PREFIX = {"genus": "g__", "family": "f__", "order": "o__"}
TARGETS = {"own temperature": "doubling_time_hours", "20 C": "doubling_time_hours_ref"}
MODELS = ["ours", "Phydon-style", "gRodon-style", "old stand-in"]


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def score(y, p):
    return {"r2_log": r2_score(y, p), "spearman": spearmanr(y, p).correlation, "fold_error": np.exp(np.median(np.abs(y - p)))}


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, names = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True)
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    idx = {n: i for i, n in enumerate(names)}
    cub = df["cub"].to_numpy(float)
    lm = X[:, [i for n, i in idx.items() if n.startswith("lm_")]]
    domain = df["archaeon"].to_numpy()
    species = df["species_group"].to_numpy()
    stand_in = X[:, [idx["cub"], idx["genome_size_bp"], idx["gc_content"], idx["trna_count"]]]
    bacteria = domain == 0
    print(f"{len(df)} rows ({bacteria.sum()} bacterial, {(~bacteria).sum()} archaeal), {df.species.nunique()} species")

    rows = []
    for tname, col in TARGETS.items():
        y = np.log(df[col].to_numpy(float))
        for level in ("species", "genus", "family", "order"):
            groups = species if level == "species" else df["gtdb_taxonomy"].map(lambda t, p=PREFIX.get(level): token(t, p)).to_numpy()
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                ug = np.unique(groups)
                perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
                g = np.array([perm[v] for v in groups])
                pred = {m: np.full(len(y), np.nan) for m in MODELS}
                for tr, te in GroupKFold(5).split(X, y, g):
                    pred["ours"][te] = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5).fit(X[tr], y[tr]).predict(X[te])
                    pred["gRodon-style"][te] = LinearRegression().fit(cub[tr, None], y[tr]).predict(cub[te, None])
                    ph = PhydonStyleBaseline(k=3).fit(cub[tr], lm[tr], domain[tr], species[tr], y[tr])
                    pred["Phydon-style"][te] = ph.predict_log(cub[te], lm[te], domain[te], species[te])
                    pred["old stand-in"][te] = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3).fit(stand_in[tr], y[tr]).predict(stand_in[te])
                for m in MODELS:
                    for subset, mask in (("all", np.ones(len(y), bool)), ("bacteria", bacteria)):
                        rows.append({"target": tname, "held_out": level, "seed": seed, "model": m, "subset": subset, **score(y[mask], pred[m][mask])})
    res = pd.DataFrame(rows)
    res.to_csv(processed / "published_comparison.csv", index=False)

    for subset in ("all", "bacteria"):
        print(f"\n===== {subset.upper()} (mean of {len(SEEDS)} seeds; in brackets, seeds where ours is better on that metric)")
        for tname in TARGETS:
            print(f"\n{tname}")
            print(f"{'held out':9s}{'model':15s}{'R2 log':>14s}{'Spearman':>14s}{'typical error':>16s}")
            for level in ("species", "genus", "family", "order"):
                sub = res[(res.subset == subset) & (res.target == tname) & (res.held_out == level)]
                ours = sub[sub.model == "ours"].set_index("seed")
                for m in MODELS:
                    s = sub[sub.model == m].set_index("seed")
                    if m == "ours":
                        cells = [f"{s.r2_log.mean():.3f}", f"{s.spearman.mean():.3f}", f"{s.fold_error.mean():.2f}x"]
                    else:
                        cells = [f"{s.r2_log.mean():.3f} ({int((ours.r2_log > s.r2_log).sum())}/{len(SEEDS)})",
                                 f"{s.spearman.mean():.3f} ({int((ours.spearman > s.spearman).sum())}/{len(SEEDS)})",
                                 f"{s.fold_error.mean():.2f}x ({int((ours.fold_error < s.fold_error).sum())}/{len(SEEDS)})"]
                    print(f"{level:9s}{m:15s}{cells[0]:>14s}{cells[1]:>14s}{cells[2]:>16s}")


if __name__ == "__main__":
    main()
