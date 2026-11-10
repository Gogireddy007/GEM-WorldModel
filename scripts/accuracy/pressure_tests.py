#!/usr/bin/env python
"""Pressure tests of the paper's headline accuracy and of the comparison with the released tools.

  baselines      on the folds of the headline (evaluate_paper_headline.py, same seeds and groups): the median of the training
                 labels, and the median of the labeled relatives at the closest rank that still has training members
                 (genus, family, order, class, phylum, domain), which is what a reader could do with a taxonomy and the
                 training table alone
  bootstrap      95% intervals, resampling genera (families or orders when those are held out), for the nested model's
                 typical error and Spearman correlation and for its error difference
                 to the relative baseline (2,000 draws, each statistic averaged over seeds)
  shuffled       negative control: the deployed model trained on labels shuffled across genomes, with species and orders
                 held out; a model that learned no artefact of the folds should be no better than the median
  paired         per-genome comparison with the released gRodon and Phydon (compare_released_tools.py --nested writes
                 released_tools_genomes_nested.csv): Wilcoxon signed-rank test on the absolute log errors averaged over seeds,
                 and a bootstrap interval for the difference in typical error
  MAGs           bootstrap intervals over species for the typical error from metagenome-assembled genomes, from the isolate
                 genomes, and their difference (mag_validation.csv, model without 16S)

Doubling time at the organism's own temperature only. Writes pressure_tests.csv.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import evaluate_paper_headline as headline
from gem_worldmodel.models.deployed import DeployedRegressor

SEEDS = (0, 1, 2)
DRAWS = 2000
RANKS = ("g__", "f__", "o__", "c__", "p__", "d__")
SLOW = np.log(24.0)


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else None


def folds(groups, seed):
    """The headline's folds: groups shuffled by the seed, then GroupKFold(5)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    g = np.array([perm[v] for v in groups])
    return list(GroupKFold(5).split(np.zeros(len(g)), groups=g))


def scores(y, p):
    out = {"typical error": float(np.exp(np.median(np.abs(y - p))))}
    out["Spearman"] = spearmanr(y, p).correlation if np.ptp(p) > 0 else np.nan
    out["AUC slow"] = roc_auc_score(y > SLOW, p) if np.ptp(p) > 0 else 0.5
    return out


def relative_median(tax, y, tr, te):
    """Median label of the training genomes that share the closest rank with each test genome."""
    tok = {r: np.array([token(t, r) for t in tax], dtype=object) for r in RANKS}
    out = np.full(len(te), np.median(y[tr]))
    for i, j in enumerate(te):
        for r in RANKS:
            if tok[r][j] is None:
                continue
            same = tr[tok[r][tr] == tok[r][j]]
            if len(same):
                out[i] = np.median(y[same])
                break
    return out


def boot(groups, fn, rng):
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(DRAWS):
        rows = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        vals.append(fn(rows))
    return np.percentile(vals, [2.5, 97.5])


def main():
    processed, df, X, _ = headline.load()
    y = np.log(df["doubling_time_hours"].to_numpy(float))
    tax = df["gtdb_taxonomy"].to_numpy()
    hp = pd.read_csv(processed / "paper_headline_predictions.csv")
    hp = hp[(hp["target"] == "own temperature") & (hp["scheme"] == "nested")]
    rows = []
    rng = np.random.default_rng(0)

    for level in ("species", "genus", "family", "order"):
        groups = df["species_group"].to_numpy() if level == "species" else df["gtdb_taxonomy"].map(
            lambda t, p=headline.PREFIX[level]: headline.token(t, p)).to_numpy()
        model, base_rel, base_med = [], [], []
        for seed in SEEDS:
            h = hp[(hp["held_out"] == level) & (hp["seed"] == seed)]
            assert len(h) == len(df) and np.allclose(h["measured"].to_numpy(), y), "headline rows are not in table order"
            model.append(h["predicted"].to_numpy())
            rel, med = np.full(len(y), np.nan), np.full(len(y), np.nan)
            for tr, te in folds(groups, seed):
                rel[te] = relative_median(tax, y, tr, te)
                med[te] = np.median(y[tr])
            base_rel.append(rel)
            base_med.append(med)
            for name, p in (("nested model", model[-1]), ("relative median", rel), ("training median", med)):
                rows.append({"test": "baseline", "held_out": level, "method": name, "seed": seed, **scores(y, p)})
        # resample whole genera (or the held-out taxon, if coarser): species of one genus are not independent
        rank = {"species": "g__", "genus": "g__", "family": "f__", "order": "o__"}[level]
        sg = np.array([token(t, rank) or g for t, g in zip(tax, df["species_group"])], dtype=object)
        err = lambda p, r: np.exp(np.median(np.abs(y[r] - p[r])))  # noqa: E731
        # each statistic is the mean over seeds, as in the headline, recomputed on every resample of species groups
        for stat, fn in (("typical error", lambda r: np.mean([err(m, r) for m in model])),
                         ("Spearman", lambda r: np.mean([spearmanr(y[r], m[r]).correlation for m in model])),
                         ("error difference to relative median", lambda r: np.mean([err(m, r) - err(b, r) for m, b in zip(model, base_rel)]))):
            full = fn(np.arange(len(y)))
            lo, hi = boot(sg, fn, rng)
            rows.append({"test": "bootstrap", "held_out": level, "method": "nested model", "statistic": stat,
                         "value": full, "low": lo, "high": hi})
        print(f"{level}: done", flush=True)

    # negative control: shuffled labels, deployed settings, each genome once
    for level in ("species", "order"):
        groups = df["species_group"].to_numpy() if level == "species" else df["gtdb_taxonomy"].map(
            lambda t: headline.token(t, "o__")).to_numpy()
        for seed in SEEDS:
            ys = np.random.default_rng(100 + seed).permutation(y)
            p = np.full(len(y), np.nan)
            for tr, te in folds(groups, seed):
                p[te] = DeployedRegressor("own temperature", seed).fit(X[tr], ys[tr]).predict(X[te])
            rows.append({"test": "shuffled labels", "held_out": level, "method": "deployed model", "seed": seed, **scores(ys, p)})
            med = np.full(len(y), np.nan)
            for tr, te in folds(groups, seed):
                med[te] = np.median(ys[tr])
            rows.append({"test": "shuffled labels", "held_out": level, "method": "training median", "seed": seed, **scores(ys, med)})
        print(f"shuffled labels, {level} held out: done", flush=True)

    # paired comparison with the released tools
    path = processed / "released_tools_genomes_nested.csv"
    if path.exists():
        g = pd.read_csv(path)
        g = g[g["gRodonpred"].notna() & (g["gRodonpred"] > 0)]
        cols = {"this model": "this model", "this model + measured temperature": "this model + measured temperature",
                "gRodon (released)": "gRodonpred", "gRodon (released), no temperature": "gRodon without temperature",
                "Phydon (released)": "combopred"}
        ae = pd.DataFrame({k: np.abs(np.log(g[c]) - np.log(g["measured"])) for k, c in cols.items()})
        ae["genome_name"] = g["genome_name"].to_numpy()
        per_all = ae.groupby("genome_name").mean()
        for ours in ("this model", "this model + measured temperature"):
            for other in ("gRodon (released)", "Phydon (released)", "gRodon (released), no temperature"):
                # genomes that both methods predicted (Phydon places fewer than gRodon)
                per = per_all[[ours, other]].dropna()
                d = per[ours] - per[other]
                diff = lambda r, d=d, o=ours, t=other: np.exp(np.median(per[o].to_numpy()[r])) - np.exp(np.median(per[t].to_numpy()[r]))  # noqa: E731
                lo, hi = np.percentile([diff(rng.integers(0, len(per), len(per))) for _ in range(DRAWS)], [2.5, 97.5])
                rows.append({"test": "paired", "held_out": "species", "method": f"{ours} vs {other}", "statistic": "typical error difference",
                             "value": diff(np.arange(len(per))), "low": lo, "high": hi, "n": len(per),
                             "better": int((d < 0).sum()), "p": wilcoxon(d).pvalue})
        print(f"paired: {len(per)} genomes", flush=True)
    else:
        print("paired: skipped, run compare_released_tools.py --nested first")

    # metagenome-assembled genomes against isolates, by species
    mg = pd.read_csv(processed / "mag_validation.csv")
    mg = mg[(mg["target"] == "own temperature") & (mg["variant"] == "without 16S")].reset_index(drop=True)
    t, iso, mag = (mg[c].to_numpy() for c in ("truth", "isolate", "mag"))
    e = lambda p, r: np.exp(np.median(np.abs(t[r] - p[r])))  # noqa: E731
    for stat, fn in (("MAG typical error", lambda r: e(mag, r)), ("isolate typical error", lambda r: e(iso, r)),
                     ("MAG minus isolate typical error", lambda r: e(mag, r) - e(iso, r))):
        lo, hi = np.percentile([fn(rng.integers(0, len(mg), len(mg))) for _ in range(DRAWS)], [2.5, 97.5])
        rows.append({"test": "MAGs", "held_out": "species", "method": "species held out", "statistic": stat,
                     "value": fn(np.arange(len(mg))), "low": lo, "high": hi, "n": len(mg)})

    out = pd.DataFrame(rows)
    out.to_csv(processed / "pressure_tests.csv", index=False)
    s = out[out["test"].isin(["baseline", "shuffled labels"])].groupby(["test", "held_out", "method"], sort=False)[
        ["typical error", "Spearman", "AUC slow"]].mean().round(3)
    print(s.to_string())
    print(out[out["test"].isin(["bootstrap", "paired", "MAGs"])].drop(columns=["typical error", "Spearman", "AUC slow", "seed"], errors="ignore").round(4).to_string())


if __name__ == "__main__":
    main()
