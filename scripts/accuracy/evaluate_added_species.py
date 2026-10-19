#!/usr/bin/env python
"""This decides whether the added labeled species help: the 162 Madin species whose traits were
computed from their genomes, and the eight hand-checked literature strains.

Four setups, all with the deployed features (traits, phylogeny, 16S, gene features with lifestyle
genes) and the doubling time at the organism's own temperature:

  A  original corpus and Madin extension only
  B  A plus the Madin genome extension
  C  A plus the literature strains
  D  A plus both (the deployed model)

Every setup is scored on the same rows, those of setup A, so a difference comes from the training
data and not from the test set. Whole species groups, families or orders are held out, ten seeds.
For each setup against A the mean change and a paired Wilcoxon signed-rank test over seeds are
reported for typical error, Spearman and the area under the curve for picking out organisms slower
than a day by ranking on the prediction. Writes added_species_evaluation.csv.
"""

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

SETUPS = {
    "A original": False,
    "B + Madin genome extension": ["Madin genome extension"],
    "C + literature": ["literature"],
    "D + both (deployed)": True,
}
SLOW = np.log(24.0)


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]

    data = {}
    for name, rows in SETUPS.items():
        df, X, _ = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True, genome_rows=rows)
        use = df["has_cub"].to_numpy()
        data[name] = (df[use].reset_index(drop=True), X[use])
    base_df = data["A original"][0]
    test_species = set(base_df["species"])
    print("rows per setup:", {n: len(d[0]) for n, d in data.items()}, f"; scored on the {len(test_species)} species of A")

    rows = []
    for level, prefix in (("species", None), ("family", "f__"), ("order", "o__")):
        for seed in seeds:
            for name, (df, X) in data.items():
                y = np.log(df["doubling_time_hours"].to_numpy(float))
                groups = df["species_group"].to_numpy() if prefix is None else df["gtdb_taxonomy"].map(lambda t, p=prefix: token(t, p)).to_numpy()
                rng = np.random.default_rng(seed)
                ug = np.unique(groups)
                perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
                pred = np.full(len(y), np.nan)
                for tr, te in GroupKFold(5).split(X, y, np.array([perm[v] for v in groups])):
                    model = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, loss="quantile", alpha=0.5)
                    pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
                m = df["species"].isin(test_species).to_numpy()
                yt, pt = y[m], pred[m]
                rows.append({"held_out": level, "seed": seed, "setup": name, "n_test": int(m.sum()),
                             "typical error": float(np.exp(np.median(np.abs(yt - pt)))),
                             "Spearman": spearmanr(yt, pt).correlation, "AUC slow": roc_auc_score(yt > SLOW, pt)})
        print(f"{level} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(processed / "added_species_evaluation.csv", index=False)

    metrics = ["typical error", "Spearman", "AUC slow"]
    for level in ("species", "family", "order"):
        d = out[out["held_out"] == level]
        piv = d.pivot_table(index="seed", columns="setup", values=metrics)
        print(f"\n===== {level.upper()} HELD OUT, {len(seeds)} seeds, scored on the same {d['n_test'].iloc[0]} rows")
        print(d.groupby("setup", sort=False)[metrics].mean().round(3).to_string())
        for name in list(SETUPS)[1:]:
            parts = []
            for mtr in metrics:
                diff = piv[mtr][name] - piv[mtr]["A original"]
                better = int((diff < 0).sum()) if mtr == "typical error" else int((diff > 0).sum())
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                parts.append(f"{mtr} {diff.mean():+.3f} (better in {better}/{len(seeds)}, p {p:.3f})")
            print(f"  {name} vs A: " + "; ".join(parts))


if __name__ == "__main__":
    main()
