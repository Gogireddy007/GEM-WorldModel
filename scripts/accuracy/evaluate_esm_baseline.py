#!/usr/bin/env python
"""A learned-representation baseline: protein language-model embeddings (embed_proteins_esm.py) against the genome
features of the paper, on the headline's folds (evaluate_paper_headline.py: same seeds, same held-out groups).

Every variant uses the deployed settings and each genome once (the scheme of the ablations), predicts the doubling time at
the growth temperature, and sees only the training rows of each split; scalers and PCA are fitted inside the split.

  features              the paper's inputs, deployed model
  ESM-2, linear probe   ridge regression on the standardized genome embedding, penalty chosen by inner cross-validation
  ESM-2, trees          the deployed model on the first 64 principal components of the embedding
  features + ESM-2      the paper's inputs plus the first 32 principal components of the embedding

Writes esm_baseline.csv (one row per variant, level and seed) and esm_baseline_genomes.csv (held-out predictions), and
prints the means with paired tests against the paper's features.
"""

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import evaluate_paper_headline as headline
from gem_worldmodel.models.deployed import DeployedRegressor
from pressure_tests import folds

SEEDS = (0, 1, 2)
SLOW = np.log(24.0)
VARIANTS = ("features", "ESM-2, linear probe", "ESM-2, trees", "features + ESM-2")
STATE = {}


def init(state):
    threadpool_limits(1)
    STATE.update(state)


def run(job):
    variant, level, seed, k, tr, te = job
    X, E, y = STATE["X"], STATE["E"], STATE["y"]
    if variant == "features":
        p = DeployedRegressor("own temperature", seed, n_jobs=1).fit(X[tr], y[tr]).predict(X[te])
    elif variant == "ESM-2, linear probe":
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
        p = m.fit(E[tr], y[tr]).predict(E[te])
    elif variant == "ESM-2, trees":
        pca = make_pipeline(StandardScaler(), PCA(64, random_state=seed)).fit(E[tr])
        p = DeployedRegressor("own temperature", seed, n_jobs=1).fit(pca.transform(E[tr]), y[tr]).predict(pca.transform(E[te]))
    else:
        pca = make_pipeline(StandardScaler(), PCA(32, random_state=seed)).fit(E[tr])
        Z = np.hstack([X, pca.transform(E)])
        p = DeployedRegressor("own temperature", seed, n_jobs=1).fit(Z[tr], y[tr]).predict(Z[te])
    return (variant, level, seed), te, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    processed, df, X, _ = headline.load()
    # one file per embedding worker (GPU and CPU); a genome embedded by both is kept once
    emb = pd.concat([pd.read_csv(f) for f in sorted(processed.glob("esm_genome_embeddings*.csv"))])
    emb = emb.drop_duplicates("accession").set_index("accession")
    keep = df["accession"].isin(emb.index).to_numpy()
    print(f"{keep.sum()} of {len(df)} genomes have an embedding", flush=True)
    df, X = df[keep].reset_index(drop=True), X[keep]
    E = emb.loc[df["accession"]].to_numpy(float)
    y = np.log(df["doubling_time_hours"].to_numpy(float))
    jobs = []
    for level in ("species", "genus", "family", "order"):
        groups = df["species_group"].to_numpy() if level == "species" else df["gtdb_taxonomy"].map(
            lambda t, p=headline.PREFIX[level]: headline.token(t, p)).to_numpy()
        for seed in SEEDS:
            for k, (tr, te) in enumerate(folds(groups, seed)):
                jobs += [(v, level, seed, k, tr, te) for v in VARIANTS]
    preds = {}
    with Pool(args.workers, initializer=init, initargs=({"X": X, "E": E, "y": y},)) as pool:
        for i, (key, te, p) in enumerate(pool.imap_unordered(run, jobs), 1):
            preds.setdefault(key, np.full(len(y), np.nan))[te] = p
            if i % 40 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    rows, gen = [], []
    for (variant, level, seed), p in preds.items():
        rows.append({"variant": variant, "held_out": level, "seed": seed, "n": len(y),
                     "typical error": float(np.exp(np.median(np.abs(y - p)))), "Spearman": spearmanr(y, p).correlation,
                     "AUC slow": roc_auc_score(y > SLOW, p)})
        gen.append(pd.DataFrame({"accession": df["accession"], "variant": variant, "held_out": level, "seed": seed,
                                 "measured": y, "predicted": p}))
    out = pd.DataFrame(rows)
    out.to_csv(processed / "esm_baseline.csv", index=False)
    g = pd.concat(gen)
    g.to_csv(processed / "esm_baseline_genomes.csv", index=False)
    print(out.groupby(["held_out", "variant"], sort=False)[["typical error", "Spearman", "AUC slow"]].mean().round(3).to_string())
    g["ae"] = (g["measured"] - g["predicted"]).abs()
    for level in ("species", "order"):
        a = g[g["held_out"] == level].pivot_table(index="accession", columns="variant", values="ae")
        for v in VARIANTS[1:]:
            d = a[v] - a["features"]
            print(f"{level}: {v} vs features: smaller error for {int((d < 0).sum())}/{len(d)} genomes, Wilcoxon p {wilcoxon(d).pvalue:.2g}")


if __name__ == "__main__":
    main()
