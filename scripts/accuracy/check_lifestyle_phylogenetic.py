#!/usr/bin/env python
"""Robustness checks for test_lifestyle_phylogenetic.py.

  1. leave one lineage out: refit after dropping each order that has a carrier
     (ammonia or methane oxidation), and each phylum with at least 10 groups
     (the other inputs that came out significant)
  2. ammonia oxidizers and methanotrophs as separate inputs, by genus name
  3. slow growers specifically: PGLS among groups slower than 5 hours, and a
     logistic regression for slower than 24 hours with phylogenetic eigenvectors
     (the leading eigenvectors of the tree covariance) as covariates
  4. the groups without a growth temperature: own-temperature doubling time with
     no temperature covariate on every group, and the 20 C value on groups with
     a temperature
  5. tree placement: every exact tip must carry the group's GTDB species and every
     stand-in tip the group's genus, plus a printed sample
  6. influence: drop one species group at a time (lambda held at its estimate)
     and report the range of the effect

Prints everything and saves the same text to lifestyle_phylogenetic_checks.txt in
the processed data folder.
"""


import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from test_lifestyle_phylogenetic import (
    BINARY,
    CONTINUOUS,
    LABEL,
    effect,
    group_table,
    pgls,
    place_on_tree,
    species_of,
    token,
)

from gem_worldmodel.data import gtdb
from gem_worldmodel.utils.config import load_config, resolve_path

ENERGY = ["nitrate_respiration", "denitrification", "dmso_tmao_reduction", "fumarate_reductase", "extracellular_electron_transfer",
          "wood_ljungdahl", "f_type_atpase", "va_type_atpase", "na_nqr", "rnf", "na_decarboxylase", "fermentation_only",
          "electron_acceptor_classes", "sodium_pumps"]
K_EIGEN = 20
KEY = ["ammonia_monooxygenase", "catalase", "regulators_per_1000", "hydrogenase_count", "nitrogen_fixation", "transporters_per_1000"]
AMMONIA = {"Nitrosomonas", "Nitrosospira", "Nitrosococcus", "Nitrosopumilus", "Nitrososphaera", "Nitrosotalea", "Nitrosocosmicus"}
METHANE = {"Methylococcus", "Methylomonas", "Methylosinus", "Methylocystis", "Methylacidiphilum", "Methylobacter", "Methylocaldum"}


def design(g, cols, with_temp=True):
    parts = [np.ones(len(g))]
    if with_temp:
        parts.append(g["growth_temp_c"].to_numpy() - 37.0)
    parts.append(g["archaeon"].to_numpy(float))
    for c in cols:
        x = g[c].to_numpy(float)
        parts.append((x - x.mean()) / x.std() if c in CONTINUOUS else x)
    return np.column_stack(parts)


def fit(g, C, col, y_col="log_y", with_temp=True, lam=None):
    X = design(g, [col], with_temp)
    y = g[y_col].to_numpy()
    if g[col].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    beta, cov, dfr, lam_used = pgls(y, X, C, lam)
    fold, lo, hi, p = effect(beta[-1], np.sqrt(cov[-1, -1]), dfr)
    return fold, lo, hi, p


def logit_slow(g, col, top, slow24):
    """Odds ratio, interval and p for slower than a day, with phylogenetic eigenvectors as covariates.

    NaN when the input is too rare among the slow groups for the model to be estimated (the interval
    then spans many orders of magnitude, a sign of separation).
    """
    Xl = np.column_stack([design(g, [col]), top])
    m = sm.Logit(slow24, Xl).fit(disp=0, method="bfgs", maxiter=2000)
    b, se = m.params[3], m.bse[3]
    if not np.isfinite(se) or se > 5:
        return (np.nan,) * 4
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[3]


def eigen_axes(C, k):
    evals, evecs = np.linalg.eigh(C)
    return evecs[:, np.argsort(evals)[::-1][:k]]


def influence(g, C, col, lam):
    base = fit(g, C, col, lam=lam)
    idx = np.where(g[col].to_numpy() > 0)[0] if col in BINARY and g[col].sum() < 60 else np.arange(len(g))
    folds, ps = [], []
    for i in idx:
        keep = np.ones(len(g), bool)
        keep[i] = False
        f = fit(g[keep], C[np.ix_(keep, keep)], col, lam=lam)
        folds.append(f[0])
        ps.append(f[3])
    i_max = idx[int(np.argmax(ps))]
    return base, len(idx), min(folds), max(folds), max(ps), g.index[i_max]


def line(name, res):
    fold, lo, hi, p = res
    return f"  {name:58s} {fold:6.2f}x ({lo:.2f} to {hi:.2f}) p {p:.3g}" if np.isfinite(fold) else f"  {name:58s} not estimable"


class Tee:
    """Send print output to the terminal and to a file."""

    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    sys.stdout = Tee(processed / "lifestyle_phylogenetic_checks.txt")
    k = load_config("features")["rrna16s"]["kmer_k"]
    g_all = group_table(processed, k, True)
    g, C = place_on_tree(g_all)
    y = g["log_y"].to_numpy()
    lam_hat = pgls(y, design(g, []), C)[3]
    print(f"{len(g)} groups on the tree, lambda {lam_hat:.3f}")
    order = g["gtdb_taxonomy"].map(lambda t: token(t, "o__"))
    phylum = g["gtdb_taxonomy"].map(lambda t: token(t, "p__"))
    genus = g["gtdb_taxonomy"].map(lambda t: token(t, "g__")).str.replace(r"^g__", "", regex=True).str.replace(r"_[A-Z]+$", "", regex=True)

    print("\n=== 1. leave one lineage out")
    print(line("ammonia or methane oxidation, all groups", fit(g, C, "ammonia_monooxygenase")))
    for o in sorted(order[g["ammonia_monooxygenase"] == 1].unique()):
        keep = (order != o).to_numpy()
        n_out = int(((order == o) & (g["ammonia_monooxygenase"] == 1)).sum())
        print(line(f"without {o} ({n_out} carriers, {int((order == o).sum())} groups)", fit(g[keep], C[np.ix_(keep, keep)], "ammonia_monooxygenase")))
    big = phylum.value_counts()
    big = big[big >= 10].index
    for col in ["catalase", "regulators_per_1000", "hydrogenase_count"]:
        res = []
        for p in big:
            keep = (phylum != p).to_numpy()
            res.append((p, fit(g[keep], C[np.ix_(keep, keep)], col)))
        folds = [r[1][0] for r in res]
        worst = max(res, key=lambda r: r[1][3])
        print(f"  {LABEL[col]:58s} leave one phylum out ({len(big)} phyla): fold {min(folds):.2f}x to {max(folds):.2f}x; "
              f"largest p {worst[1][3]:.3g} (without {worst[0]})")

    print("\n=== 2. ammonia oxidizers and methanotrophs separately")
    g2 = g.assign(ammonia_oxidizer=(genus.isin(AMMONIA) & (g["ammonia_monooxygenase"] == 1)).astype(float),
                  methanotroph=(genus.isin(METHANE) & (g["ammonia_monooxygenase"] == 1)).astype(float))
    other = (g2["ammonia_monooxygenase"] == 1) & (g2["ammonia_oxidizer"] == 0) & (g2["methanotroph"] == 0)
    print(f"  ammonia oxidizers: {sorted(g2.index[g2.ammonia_oxidizer == 1])}")
    print(f"  methanotrophs: {sorted(g2.index[g2.methanotroph == 1])}")
    print(f"  carriers in neither list: {sorted(g2.index[other])}")
    X = design(g2, ["ammonia_oxidizer", "methanotroph"])
    beta, cov, dfr, lam = pgls(y, X, C)
    for i, name in ((-2, "ammonia oxidizers (both in one model)"), (-1, "methanotrophs (both in one model)")):
        print(line(name, effect(beta[i], np.sqrt(cov[i, i]), dfr)))
    lit = set(pd.read_csv(processed / "literature_extension.csv")["species"])
    keep = ~g2.index.isin(lit)
    print(line("ammonia oxidizers, without the literature strains", fit(g2[keep], C[np.ix_(keep, keep)], "ammonia_oxidizer")))

    print("\n=== 3. slow growers specifically")
    slow5 = np.exp(y) > 5
    gs, Cs = g[slow5], C[np.ix_(slow5, slow5)]
    print(f"  (a) PGLS among the {slow5.sum()} groups slower than 5 hours, lambda {pgls(gs['log_y'].to_numpy(), design(gs, []), Cs)[3]:.2f}")
    for col in KEY + ["phototrophy", "sulfate_reduction", "log2_genome_size"]:
        print(line(LABEL[col], fit(gs, Cs, col)))
    # 20 eigenvectors: with 10, the slow Anaerolineales clade is not absorbed and inputs it happens to
    # carry (an E. coli-type fumarate reductase) look strongly linked to slow growth; 10 and 30 are shown
    # as a sensitivity check
    top = eigen_axes(C, K_EIGEN)
    slow24 = (np.exp(y) > 24).astype(float)
    print(f"  (b) slower than 24 hours ({int(slow24.sum())} of {len(g)}), logistic regression with {K_EIGEN} phylogenetic "
          "eigenvectors; odds ratio (in brackets: 10 and 30 eigenvectors)")
    for col in KEY + ["phototrophy", "sulfate_reduction", "log2_genome_size"]:
        orr, lo, hi, p = logit_slow(g, col, top, slow24)
        alt = [logit_slow(g, col, eigen_axes(C, kk), slow24) for kk in (10, 30)]
        alt_s = ", ".join(f"{a[0]:.2f} p {a[3]:.2g}" if np.isfinite(a[0]) else "n/a" for a in alt)
        main_s = f"{orr:6.2f} ({lo:.2f} to {hi:.2f}) p {p:.3g}" if np.isfinite(orr) else "  not estimable"
        print(f"  {LABEL[col]:58s} {main_s}   [{alt_s}]")

    print("\n=== 4. groups without a growth temperature")
    g_all = group_table(processed, k, True, require_temp=False)
    ga, Ca = place_on_tree(g_all)
    print(f"  own temperature, no temperature covariate, all {len(ga)} groups:")
    for col in KEY:
        print(line(LABEL[col], fit(ga, Ca, col, with_temp=False)))
    print(f"  20 C value, groups with a growth temperature ({len(g)}), no temperature covariate:")
    for col in KEY:
        print(line(LABEL[col], fit(g, C, col, y_col="log_y_ref", with_temp=False)))

    print("\n=== 5. tree placement")
    data_cfg = load_config("data")
    tax = pd.concat([gtdb.fetch_bac_taxonomy(data_cfg), gtdb.fetch_arc_taxonomy(data_cfg)]).set_index("accession")["gtdb_taxonomy"]
    tip_tax = g["tip"].map(tax)
    sp_ok = (tip_tax.map(species_of) == g["gtdb_taxonomy"].map(species_of))[g["exact_tip"]]
    gen_ok = (tip_tax.map(lambda t: token(t, "g__")) == g["gtdb_taxonomy"].map(lambda t: token(t, "g__")))[~g["exact_tip"]]
    print(f"  exact tips with the group's species: {int(sp_ok.sum())} of {len(sp_ok)}; stand-in tips in the group's genus: {int(gen_ok.sum())} of {len(gen_ok)}")
    sample = g.sample(8, random_state=0)
    for name, tip in zip(sample.index, sample["tip"], strict=True):
        print(f"  {name:38s} {tip:22s} {species_of(tax.get(tip, ''))}")

    print("\n=== 6. influence of single species groups (lambda held at its estimate)")
    for col in KEY:
        base, n, lo, hi, pmax, who = influence(g, C, col, lam_hat)
        print(f"  {LABEL[col]:40s} {base[0]:.2f}x p {base[3]:.3g}; dropping one of {n}: {lo:.2f}x to {hi:.2f}x, "
              f"largest p {pmax:.3g} (without {who})")

    print("\n=== 7. energy-metabolism features: slow growers specifically, then robustness for any that come out significant")
    print("  p adj is Benjamini-Hochberg across the energy features within each test")
    res_a = {col: fit(gs, Cs, col) for col in ENERGY}
    res_b = {}
    for col in ENERGY:
        try:
            res_b[col] = logit_slow(g, col, top, slow24)
        except Exception:  # noqa: BLE001
            res_b[col] = (np.nan,) * 4
    pa = pd.Series({c: r[3] for c, r in res_a.items()}).dropna()
    pb = pd.Series({c: r[3] for c, r in res_b.items()}).dropna()
    adj_a = pd.Series(multipletests(pa, method="fdr_bh")[1], index=pa.index)
    adj_b = pd.Series(multipletests(pb, method="fdr_bh")[1], index=pb.index)
    print(f"  {'':46s}{'among groups slower than 5 h (PGLS)':>40s}   {'slower than 24 h (odds ratio)':>36s}")
    for col in ENERGY:
        a, b = res_a[col], res_b[col]
        sa = f"{a[0]:5.2f}x ({a[1]:.2f}-{a[2]:.2f}) p adj {adj_a.get(col, np.nan):.3f}" if np.isfinite(a[0]) else "not estimable"
        sb = f"{b[0]:5.2f} ({b[1]:.2f}-{b[2]:.2f}) p adj {adj_b.get(col, np.nan):.3f}" if np.isfinite(b[0]) else "not estimable"
        print(f"  {LABEL[col]:46s}{sa:>40s}   {sb:>36s}")
    flagged = sorted(set(adj_a[adj_a < 0.05].index) | set(adj_b[adj_b < 0.05].index))
    print(f"  significant after correction in either test: {flagged if flagged else 'none'}")
    for col in flagged:
        # robustness of the slower-than-a-day result itself (the test that flagged it)
        print(f"  {LABEL[col]}, slower than 24 h:")
        for kk in (5, 10, 20, 30):
            r = logit_slow(g, col, eigen_axes(C, kk), slow24)
            print(f"    with {kk:2d} phylogenetic eigenvectors: odds ratio {r[0]:.2f} p {r[3]:.3g}" if np.isfinite(r[0]) else f"    with {kk:2d} eigenvectors: not estimable")
        slow_carriers = order[(g[col] > 0).to_numpy() & (slow24 == 1)]
        print(f"    slow carriers by order: {slow_carriers.value_counts().to_dict()}")
        for o in slow_carriers.unique():
            keep = (order != o).to_numpy()
            r = logit_slow(g[keep], col, eigen_axes(C[np.ix_(keep, keep)], K_EIGEN), slow24[keep])
            print(f"    without {o}: odds ratio {r[0]:.2f} p {r[3]:.3g}" if np.isfinite(r[0]) else f"    without {o}: not estimable")


if __name__ == "__main__":
    main()
