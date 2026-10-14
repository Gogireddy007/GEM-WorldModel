#!/usr/bin/env python
"""This tests whether the new labeled species and the lifestyle gene-content
features improve accuracy and the detection of slow growers.

Four setups, all with the final features (traits, phylogeny, 16S, codon usage,
IVYWREL) and the doubling time at the organism's own temperature:

  before              original corpus plus the Madin extension
  + new species       plus the Madin genome extension and the literature rows
  + gene content      before, plus the lifestyle gene-content features
  + both              new species and gene-content features

Whole species groups (deploy_features.species_group) or whole orders are held
out, 5 folds, 3 seeds. Scores are given on every row of the setup and on the
"before" rows only, which is the same test set for all four setups, so the
effect of more training data is not confused with a change of test set.

Slow means a measured doubling time above 24 hours. "AUC, ranking" ranks rows by
the predicted doubling time, "AUC, classifier" uses a separate boosted
classifier for slow vs not. The target set for this work was an AUC above
about 0.75 with orders held out, without more false slow calls.

Writes new_data_and_genes.csv.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = [42, 1, 7]
SLOW = np.log(24.0)
OLD = {"original", "bacterial extension", "archaeal extension"}
SETUPS = {
    "before": {"genome_rows": False, "with_lifestyle": False},
    "+ new species": {"genome_rows": True, "with_lifestyle": False},
    "+ gene content": {"genome_rows": False, "with_lifestyle": True},
    "+ both": {"genome_rows": True, "with_lifestyle": True},
}


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def folds(groups, seed, n=5):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    return list(GroupKFold(n).split(groups, groups=np.array([perm[v] for v in groups])))


def scores(y, pred, prob):
    slow, called = y > SLOW, pred > SLOW
    return {
        "typical error": np.exp(np.median(np.abs(y - pred))),
        "Spearman": spearmanr(y, pred).correlation,
        "AUC, ranking": roc_auc_score(slow, pred),
        "AUC, classifier": roc_auc_score(slow, prob),
        "slow found %": 100 * called[slow].mean(),
        "fast called slow %": 100 * called[~slow].mean(),
        "n": len(y),
        "n slow": int(slow.sum()),
    }


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    rows = []
    for name, opts in SETUPS.items():
        df, X, names = deploy_features.load_training_table(processed, k, with_cub=True, **opts)
        use = df["has_cub"].to_numpy()
        df, X = df[use].reset_index(drop=True), X[use]
        y = np.log(df["doubling_time_hours"].to_numpy(float))
        old = df["origin"].isin(OLD).to_numpy()
        for level in ("species", "order"):
            groups = df["species_group"].to_numpy() if level == "species" else df["gtdb_taxonomy"].map(lambda t: token(t, "o__")).to_numpy()
            for seed in SEEDS:
                pred, prob = np.full(len(y), np.nan), np.full(len(y), np.nan)
                for tr, te in folds(groups, seed):
                    reg = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5)
                    pred[te] = reg.fit(X[tr], y[tr]).predict(X[te])
                    clf = GradientBoostingClassifier(random_state=seed, n_estimators=200, max_depth=3)
                    prob[te] = clf.fit(X[tr], y[tr] > SLOW).predict_proba(X[te])[:, 1]
                for subset, m in (("all rows", np.ones(len(y), bool)), ("before rows", old)):
                    rows.append({"setup": name, "held_out": level, "seed": seed, "test rows": subset, "features": len(names),
                                 **scores(y[m], pred[m], prob[m])})
        print(f"{name}: {len(df)} rows, {len(names)} features, {(y > SLOW).sum()} slow", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(processed / "new_data_and_genes.csv", index=False)
    cols = ["typical error", "Spearman", "AUC, ranking", "AUC, classifier", "slow found %", "fast called slow %", "n", "n slow"]
    for subset in ("before rows", "all rows"):
        for level in ("species", "order"):
            t = out[(out["test rows"] == subset) & (out["held_out"] == level)].groupby("setup", sort=False)[cols].mean()
            print(f"\n===== {level.upper()} HELD OUT, scored on {subset} (mean of {len(SEEDS)} seeds)")
            print(t.round(3).to_string())


if __name__ == "__main__":
    main()
