#!/usr/bin/env python
"""This tests whether adding codon usage bias (annotation-free pipeline, the same
one that is run on GEM MAGs) improves the deployment model.

Same rows with and without the column, median loss, seven seeds, whole species,
genera, families or orders held out. Three targets: the doubling time corrected
to 20 C (the original target), the raw doubling time at the organism's own
growth temperature, and the growth temperature. Also reports ranking among fast
growers (corrected doubling time under 5 hours) and for bacteria and archaea
separately.

Needs labeled_cub_pipeline.csv and extension_cub.csv.
"""

import re

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = [42, 1, 7, 13, 2024, 123, 99]
PREFIX = {"genus": "g__", "family": "f__", "order": "o__"}


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def oof(X, y, groups, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug))))
    g = np.array([perm[v] for v in groups])
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5).fit(X[tr], y[tr]).predict(X[te])
    return pred


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, _ = deploy_features.load_training_table(processed, k, with_cub=True)
    use = df["has_cub"].to_numpy() & df["growth_temp_c"].notna().to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    arch = df["archaeon"].to_numpy() == 1
    print(f"{len(df)} rows with codon usage and a growth temperature ({arch.sum()} archaeal, {df.species.nunique()} species)")
    print(f"Spearman of codon usage with log corrected doubling time: all {spearmanr(df.cub, np.log(df.doubling_time_hours_ref)).correlation:.3f}, "
          f"bacteria {spearmanr(df.cub[~arch], np.log(df.doubling_time_hours_ref[~arch])).correlation:.3f}, archaea {spearmanr(df.cub[arch], np.log(df.doubling_time_hours_ref[arch])).correlation:.3f}\n")
    targets = {
        "corrected to 20 C": np.log(df["doubling_time_hours_ref"].to_numpy()),
        "raw, own temperature": np.log(df["doubling_time_hours"].to_numpy()),
        "growth temperature": df["growth_temp_c"].to_numpy(float),
    }
    fast = df["doubling_time_hours_ref"].to_numpy() < 5
    X_no = X[:, :-2]  # drops codon usage and IVYWREL
    for level in ("species", "genus", "family", "order"):
        groups = df["species"].to_numpy() if level == "species" else df["gtdb_taxonomy"].map(lambda t, p=PREFIX.get(level): token(t, p)).to_numpy()
        print(f"--- {level.upper()} HELD OUT (mean of 7 seeds; in brackets: seeds better with codon usage)")
        print(f"{'target':24s}{'features':14s}{'R2':>8s}{'Spearman':>10s}{'error':>14s}{'fast Spearman':>16s}")
        for name, y in targets.items():
            res = {}
            for label, Xm in (("without", X_no), ("with", X)):
                runs = []
                for s in SEEDS:
                    p = oof(Xm, y, groups, s)
                    err = np.exp(np.median(np.abs(y - p))) if name != "growth temperature" else np.mean(np.abs(y - p))
                    runs.append((r2_score(y, p), spearmanr(y, p).correlation, err, spearmanr(y[fast], p[fast]).correlation))
                res[label] = np.array(runs)
            for label in ("without", "with"):
                r = res[label]
                b = res["without"]
                def mark(i, lower=False):
                    return "" if label == "without" else f" ({int(((r[:, i] < b[:, i]) if lower else (r[:, i] > b[:, i])).sum())}/7)"

                unit = " C" if name == "growth temperature" else "x"
                print(f"{name:24s}{label:14s}{r[:, 0].mean():8.3f}{mark(0):>8s}{r[:, 1].mean():8.3f}{mark(1):>8s}{r[:, 2].mean():8.2f}{unit}{mark(2, True):>5s}{r[:, 3].mean():11.3f}{mark(3):>6s}")
        print()


if __name__ == "__main__":
    main()
