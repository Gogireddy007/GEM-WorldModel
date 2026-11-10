#!/usr/bin/env python
"""Second round of pressure tests: leakage, label robustness and the number of splits.

All variants use the deployed settings and each genome once (the scheme of the ablations), on the official splits of
gem_worldmodel.benchmark, and predict the log doubling time at the growth temperature.

  gRodon estimate   gRodon's own doubling-time estimate is an input, and gRodon's regression was fitted on growth rates
                    that include our test species, so it can carry label information into a held-out split. The model is
                    run without that one input, and with it replaced by a gRodon-style regression refitted inside each
                    split on the training rows only (the codon statistics themselves are computed from the genome alone).
  flagged studies   the study-level check flags studies whose labels sit far from independent evidence; the flagged
                    labels we kept (each with a biological explanation) are removed from training and testing, and the
                    model is compared with the full model on the same remaining genomes.
  duplicate rows    one genome appears twice under two spellings of the same species; it is kept once.
  ten splits        the model against the relatives' median over ten seeds at every level, instead of three.

Paired comparisons are by seed (Wilcoxon) and by genome (absolute log errors averaged over seeds, Wilcoxon).
Writes pressure_tests_round2.csv.
"""

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits

from gem_worldmodel import benchmark as gb
from gem_worldmodel.models.deployed import DeployedRegressor
from gem_worldmodel.utils.config import load_config, resolve_path

STATE = {}


def init(state):
    threadpool_limits(1)
    STATE.update(state)


def fit_predict(job):
    variant, level, seed, tr, te, cols = job
    X, y = STATE["X"][:, cols], STATE["y"]
    if variant == "in-split gRodon refit":
        # replace the released gRodon estimate by a gRodon-style regression (log doubling time on CUBHE, ConsistencyHE and
        # CPB) fitted on the training rows of this split only, so no held-out label can reach the input
        X = X.copy()
        s, d = STATE["stat_cols"], STATE["d_col"]
        A = np.column_stack([np.ones(len(X)), X[:, s]])
        coef, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
        X[:, d] = A @ coef
    return (variant, level, seed), te, DeployedRegressor("own temperature", seed, n_jobs=1).fit(X[tr], y[tr]).predict(X[te])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    data = gb.load()
    t, X = data.table, data.X
    keep = ~t["accession"].duplicated().to_numpy()            # one row per genome
    t, X = t[keep].reset_index(drop=True), X[keep]
    y = t["y"].to_numpy()
    all_cols = np.arange(X.shape[1])
    no_grodon_d = np.array([i for i, n in enumerate(data.names) if n != "log_grodon_d"])
    assert len(no_grodon_d) == len(all_cols) - 1

    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    studies = pd.read_csv(processed / "label_sources_studies.csv")
    flagged = set("; ".join(studies.loc[studies["flag"], "species"]).split("; "))
    clean = ~t["species"].isin(flagged).to_numpy()

    seeds = range(args.seeds)
    jobs = []
    for level in gb.LEVELS:
        for seed in seeds:
            for tr, te in gb.splits(t, level, seed):
                jobs.append(("full", level, seed, tr, te, all_cols))
                if level in ("species", "order"):
                    jobs.append(("without gRodon estimate", level, seed, tr, te, no_grodon_d))
                    jobs.append(("in-split gRodon refit", level, seed, tr, te, all_cols))
            if level in ("species", "order"):
                tc = t[clean].reset_index(drop=True)
                idx = np.flatnonzero(clean)
                for tr, te in gb.splits(tc, level, seed):
                    jobs.append(("without flagged studies", level, seed, idx[tr], idx[te], all_cols))
    preds = {}
    stat_cols = [data.names.index(n) for n in ("CUBHE", "ConsistencyHE", "CPB")]
    state = {"X": X, "y": y, "stat_cols": stat_cols, "d_col": data.names.index("log_grodon_d")}
    with Pool(args.workers, initializer=init, initargs=(state,)) as pool:
        for i, (key, te, p) in enumerate(pool.imap_unordered(fit_predict, jobs), 1):
            preds.setdefault(key, np.full(len(y), np.nan))[te] = p
            if i % 100 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    base = {}
    for level in gb.LEVELS:
        for seed in seeds:
            b = np.full(len(y), np.nan)
            for tr, te in gb.splits(t, level, seed):
                b[te] = gb.relatives_median(t.iloc[tr], t.iloc[te])
            base[(level, seed)] = b

    rows = []

    def compare(test, level, a_name, a, b_name, b, mask):
        """a and b: dict seed -> predictions; scored on rows in mask."""
        ea = np.array([gb.score(y[mask], a[s][mask])["typical error"] for s in seeds])
        eb = np.array([gb.score(y[mask], b[s][mask])["typical error"] for s in seeds])
        ra = np.array([gb.score(y[mask], a[s][mask])["Spearman"] for s in seeds])
        rb = np.array([gb.score(y[mask], b[s][mask])["Spearman"] for s in seeds])
        ga = np.mean([np.abs(y[mask] - a[s][mask]) for s in seeds], axis=0)
        gbb = np.mean([np.abs(y[mask] - b[s][mask]) for s in seeds], axis=0)
        rows.append({"test": test, "held_out": level, "a": a_name, "b": b_name, "n_genomes": int(mask.sum()), "n_seeds": len(seeds),
                     "error_a": ea.mean(), "error_b": eb.mean(), "rho_a": ra.mean(), "rho_b": rb.mean(),
                     "seeds_a_better": int((ea < eb).sum()), "p_seeds": wilcoxon(ea - eb).pvalue if np.any(ea != eb) else 1.0,
                     "genomes_a_better": int((ga < gbb).sum()), "p_genomes": wilcoxon(ga - gbb).pvalue})

    everyone = np.ones(len(y), bool)
    for level in gb.LEVELS:
        full = {s: preds[("full", level, s)] for s in seeds}
        compare("ten splits", level, "model", full, "relatives' median", {s: base[(level, s)] for s in seeds}, everyone)
        if level in ("species", "order"):
            compare("gRodon estimate", level, "without gRodon estimate", {s: preds[("without gRodon estimate", level, s)] for s in seeds},
                    "full", full, everyone)
            compare("gRodon estimate", level, "in-split gRodon refit", {s: preds[("in-split gRodon refit", level, s)] for s in seeds},
                    "full", full, everyone)
            compare("flagged studies", level, "without flagged studies", {s: preds[("without flagged studies", level, s)] for s in seeds},
                    "full", full, clean)
    out = pd.DataFrame(rows)
    out.to_csv(processed / "pressure_tests_round2.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string())
    print(f"genomes: {len(y)} (one duplicate row removed); flagged-study labels removed in that test: {int((~clean).sum())}")


if __name__ == "__main__":
    main()
