#!/usr/bin/env python
"""This tests whether the IVYWREL amino acid fraction (the share of I, V, Y, W, R, E
and L among all residues of the predicted proteins) improves growth temperature
prediction, with whole species, genera, families or orders held out.

IVYWREL rises with growth temperature across prokaryotes because those residues
stabilise proteins when it is hot. It is computed in the same gene-prediction
pass as codon usage bias, so it costs nothing extra on GEM genomes, and unlike
phylogeny it does not need a close labeled relative. Also reports how often
organisms that grow below 45 C are called hot and the reverse, since a model
that leans on genome size and GC can mistake small high-GC mesophiles for
thermophiles.
"""

import re

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

PREFIX = {"genus": "g__", "family": "f__", "order": "o__"}


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m else "none"


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    df, X, names = deploy_features.load_training_table(processed, load_config("features")["rrna16s"]["kmer_k"], with_cub=True)
    use = (df.has_cub & df.growth_temp_c.notna()).to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    without = [i for i, n in enumerate(names) if n != "ivywrel"]
    y = df.growth_temp_c.to_numpy(float)
    meso = y < 45
    print(f"{len(df)} rows with a growth temperature, {meso.sum()} below 45 C; "
          f"Spearman of IVYWREL with growth temperature {spearmanr(df.ivywrel, y).correlation:.3f}")
    for level in ("species", "genus", "family", "order"):
        groups = df.species_group.to_numpy() if level == "species" else df.gtdb_taxonomy.map(lambda t, p=PREFIX.get(level): token(t, p)).to_numpy()
        for label, cols in (("without IVYWREL", without), ("with IVYWREL", list(range(X.shape[1])))):
            res = []
            for seed in (42, 1, 7):
                rng = np.random.default_rng(seed)
                ug = np.unique(groups)
                perm = dict(zip(ug, rng.permutation(len(ug))))
                g = np.array([perm[v] for v in groups])
                p = np.full(len(y), np.nan)
                for tr, te in GroupKFold(5).split(X, y, g):
                    p[te] = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5).fit(X[tr][:, cols], y[tr]).predict(X[te][:, cols])
                res.append((r2_score(y, p), np.mean(np.abs(y - p)), 100 * (p[meso] >= 45).mean(), 100 * (p[~meso] < 45).mean(), roc_auc_score(~meso, p)))
            r = np.mean(res, axis=0)
            print(f"{level:8s} held out, {label:16s}: R2 {r[0]:.3f}, error {r[1]:.1f} C, below-45 C organisms called hot {r[2]:.1f}%, "
                  f"hot organisms called cooler {r[3]:.1f}%, AUC {r[4]:.3f}")


if __name__ == "__main__":
    main()
