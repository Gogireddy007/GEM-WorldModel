#!/usr/bin/env python
"""This tests whether the energy-metabolism and lifestyle genes still go with
slow growth once relatedness is accounted for (the low-energy explanation of
slow growth).

Close relatives share both their genes and their growth rates, so a raw
comparison across species counts one slow clade many times over. Three views:

  across species   ordinary regression, no control for relatedness (for comparison)
  PGLS             phylogenetic generalized least squares on the GTDB trees,
                   with Pagel's lambda estimated by maximum likelihood
                   (0 means no phylogenetic signal in the residuals, 1 means
                   Brownian motion on the tree)
  within families  in families that have both carriers and non-carriers of a
                   gene, is the median carrier slower? Counted over families,
                   with a Wilcoxon signed-rank test

One row per species group (deploy_features.species_group). The response is the
log doubling time at the organism's own growth temperature, with growth
temperature and an archaea flag as covariates, so rows need a growth
temperature. Each group is placed on the GTDB tree at its species representative;
a group placed only to genus uses another species of its genus as a stand-in
tip. Bacteria and archaea are on separate trees, so the covariance matrix has
one block for each, each scaled to a mean root-to-tip depth of 1.

Effects are reported as the fold change in doubling time for having the gene
(or for one standard deviation of a count or density, or for doubling genome
size), with 95% intervals and Benjamini-Hochberg adjusted p values across the
tested inputs. Everything is run again without the eight literature strains,
which were chosen as slow nitrifiers. Growth rates listed in
data/curated/label_exclusions.csv are left out (by the training table loader);
with the four methanotroph values among them in, those species alone produce an
apparent fivefold slowdown for methane oxidizers.

Writes lifestyle_phylogenetic_test.csv.
"""

import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from statsmodels.stats.multitest import multipletests

from gem_worldmodel.data import gtdb
from gem_worldmodel.features import deploy_features, lifestyle_genes
from gem_worldmodel.utils.config import load_config, resolve_path

BINARY = ["aerobic_respiration", "complex_i", "methanogenesis", "sulfate_reduction", "nitrogen_fixation", "rubisco",
          "ammonia_monooxygenase", "phototrophy", "motility", "catalase"]
CONTINUOUS = ["transporters_per_1000", "regulators_per_1000", "oxidase_count", "hydrogenase_count"]
# energy metabolism (lifestyle_genes.energy_features), added 2026-10-15
BINARY += ["nitrate_respiration", "denitrification", "dmso_tmao_reduction", "fumarate_reductase", "extracellular_electron_transfer",
           "wood_ljungdahl", "f_type_atpase", "va_type_atpase", "na_nqr", "rnf", "na_decarboxylase", "fermentation_only"]
CONTINUOUS += ["electron_acceptor_classes", "sodium_pumps"]
LABEL = {
    "aerobic_respiration": "aerobic respiration (terminal oxidase)", "complex_i": "complex I",
    "methanogenesis": "methanogenesis", "sulfate_reduction": "sulfate reduction", "nitrogen_fixation": "nitrogen fixation",
    "rubisco": "RuBisCO (carbon fixation)", "ammonia_monooxygenase": "ammonia or methane oxidation",
    "phototrophy": "phototrophy", "motility": "flagellar motility", "catalase": "catalase",
    "transporters_per_1000": "transporter density (per SD)", "regulators_per_1000": "regulator density (per SD)",
    "oxidase_count": "number of terminal oxidases (per SD)", "hydrogenase_count": "number of hydrogenases (per SD)",
    "log2_genome_size": "genome size (per doubling)",
    "nitrate_respiration": "nitrate respiration", "denitrification": "denitrification (NosZ)",
    "dmso_tmao_reduction": "DMSO/TMAO respiration", "fumarate_reductase": "fumarate respiration (E. coli type)",
    "extracellular_electron_transfer": "metal reduction outside the cell", "wood_ljungdahl": "Wood-Ljungdahl pathway",
    "f_type_atpase": "F-type ATP synthase", "va_type_atpase": "V/A-type ATP synthase", "na_nqr": "Na+-NQR",
    "rnf": "Rnf complex", "na_decarboxylase": "Na+-pumping decarboxylase", "fermentation_only": "fermentation only",
    "electron_acceptor_classes": "number of electron acceptor classes (per SD)", "sodium_pumps": "number of sodium pumps (per SD)",
}


def species_of(t):
    m = re.search(r"s__([^;]+)", str(t))
    return m.group(1) if m else None


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else None


def group_table(processed, k, include_literature, require_temp=True):
    df, X, names = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True, with_energy=True)
    keep = df["has_lifestyle"].to_numpy()
    df, X = df[keep].reset_index(drop=True), X[keep]
    cols = lifestyle_genes.FEATURES + lifestyle_genes.ENERGY_FEATURES
    feats = pd.DataFrame(X[:, [names.index(f) for f in cols]], columns=cols)
    df = pd.concat([df, feats], axis=1)
    if not include_literature:
        df = df[df["origin"] != "literature"]
    df["log_y"] = np.log(df["doubling_time_hours"].astype(float))
    df["log2_genome_size"] = np.log2(df["genome_size_bp"].astype(float))
    df["log_y_ref"] = np.log(df["doubling_time_hours_ref"].astype(float))
    agg = {"log_y": "mean", "log_y_ref": "mean", "growth_temp_c": "mean", "archaeon": "first", "gtdb_taxonomy": "first", "log2_genome_size": "mean"}
    agg.update({f: "mean" for f in BINARY + CONTINUOUS})
    g = df.groupby("species_group").agg(agg)
    for f in BINARY:
        g[f] = (g[f] >= 0.5).astype(float)
    return g.dropna(subset=["growth_temp_c"]) if require_temp else g


def leaf_covariance(tree, labels):
    """Shared root-to-MRCA path length for the given leaf labels (Brownian covariance)."""
    keep = set(labels)
    pruned = tree.extract_tree_with_taxa_labels(labels=keep)
    leaves = [n for n in pruned.leaf_node_iter()]
    pos = {n.taxon.label: i for i, n in enumerate(leaves)}
    cov = np.zeros((len(leaves), len(leaves)))
    below = {}
    for node in pruned.postorder_node_iter():
        if node.is_leaf():
            below[node] = [pos[node.taxon.label]]
        else:
            below[node] = [i for c in node.child_node_iter() for i in below[c]]
        length = node.edge.length or 0.0
        if length:
            idx = below[node]
            cov[np.ix_(idx, idx)] += length
    order = [pos[label] for label in labels]
    return cov[np.ix_(order, order)]


def place_on_tree(g):
    """A GTDB tree tip for every group, and the block-diagonal covariance matrix."""
    data_cfg = load_config("data")
    blocks, tips_all = [], []
    keep_rows, exact = [], set()
    for domain, tree_fn, tax_fn in (("bacteria", gtdb.load_tree, gtdb.fetch_bac_taxonomy), ("archaea", gtdb.load_arc_tree, gtdb.fetch_arc_taxonomy)):
        tree = tree_fn(data_cfg)
        tip_labels = {n.taxon.label for n in tree.leaf_node_iter() if n.taxon is not None}
        tax = tax_fn(data_cfg)
        tax = tax[tax["accession"].isin(tip_labels)]
        rep_of_species = dict(zip(tax["gtdb_taxonomy"].map(species_of), tax["accession"], strict=True))
        tips_of_genus = tax.groupby(tax["gtdb_taxonomy"].map(lambda t: token(t, "g__")))["accession"].apply(list).to_dict()
        rows = g[(g["archaeon"] == 1) == (domain == "archaea")]
        used, tips, idx = set(), [], []
        for name, t in zip(rows.index, rows["gtdb_taxonomy"], strict=True):
            tip = rep_of_species.get(species_of(t))
            if tip is not None and tip not in used:
                exact.add(name)
            if tip is None or tip in used:
                tip = next((c for c in tips_of_genus.get(token(t, "g__"), []) if c not in used), None)
            if tip is None:
                continue
            used.add(tip)
            tips.append(tip)
            idx.append(name)
        cov = leaf_covariance(tree, tips)
        blocks.append(cov / np.mean(np.diag(cov)))
        tips_all += tips
        keep_rows += idx
    n = sum(len(b) for b in blocks)
    C = np.zeros((n, n))
    start = 0
    for b in blocks:
        C[start:start + len(b), start:start + len(b)] = b
        start += len(b)
    g = g.loc[keep_rows].copy()
    g["exact_tip"] = g.index.isin(exact)
    g["tip"] = tips_all
    return g, C


def gls(y, X, V):
    """GLS estimate, covariance of the estimate and log likelihood (ML) for a fixed correlation matrix V."""
    L = np.linalg.cholesky(V)
    Xs, ys = np.linalg.solve(L, X), np.linalg.solve(L, y)
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    resid = ys - Xs @ beta
    n = len(y)
    sigma2 = resid @ resid / n
    loglik = -0.5 * (n * np.log(2 * np.pi * sigma2) + n + 2 * np.sum(np.log(np.diag(L))))
    df_resid = n - X.shape[1]
    cov_beta = np.linalg.inv(Xs.T @ Xs) * (resid @ resid / df_resid)
    return beta, cov_beta, loglik, df_resid


def pgls(y, X, C, lam=None):
    diag = np.diag(np.diag(C))

    def negll(lam):
        return -gls(y, X, lam * C + (1 - lam) * diag)[2]

    if lam is None:
        lam = minimize_scalar(negll, bounds=(0.0, 1.0), method="bounded").x
    beta, cov, _, dfr = gls(y, X, lam * C + (1 - lam) * diag)
    return beta, cov, dfr, lam


def effect(beta, se, dfr, scale=1.0):
    t = beta / se
    q = stats.t.ppf(0.975, dfr)
    return np.exp(beta * scale), np.exp((beta - q * se) * scale), np.exp((beta + q * se) * scale), 2 * stats.t.sf(abs(t), dfr)


def within_families(g, feature):
    """Median log doubling time of carriers minus non-carriers, per family that has both."""
    fam = g["gtdb_taxonomy"].map(lambda t: token(t, "f__"))
    diffs = []
    for _, sub in g.groupby(fam):
        a, b = sub.loc[sub[feature] == 1, "log_y"], sub.loc[sub[feature] == 0, "log_y"]
        if len(a) and len(b):
            diffs.append(a.median() - b.median())
    diffs = np.array(diffs)
    if len(diffs) < 3:
        return len(diffs), int((diffs > 0).sum()), float(np.exp(np.median(diffs))) if len(diffs) else np.nan, np.nan
    p = stats.wilcoxon(diffs).pvalue if np.any(diffs != 0) else 1.0
    return len(diffs), int((diffs > 0).sum()), float(np.exp(np.median(diffs))), p


def run(g, C, label, lam=None):
    base = np.column_stack([np.ones(len(g)), g["growth_temp_c"].to_numpy() - 37.0, g["archaeon"].to_numpy(float)])
    y = g["log_y"].to_numpy()
    rows = []
    for f in BINARY + CONTINUOUS + ["log2_genome_size"]:
        x = g[f].to_numpy(float)
        scale = 1.0
        if f in CONTINUOUS:
            scale = x.std()
            x = (x - x.mean()) / scale
            scale = 1.0
        X = np.column_stack([base, x])
        ols = gls(y, X, np.eye(len(y)))
        b_o, se_o = ols[0][-1], np.sqrt(ols[1][-1, -1])
        beta_p, cov_p, dfr_p, lam_used = pgls(y, X, C, lam)
        b_p, se_p = beta_p[-1], np.sqrt(cov_p[-1, -1])
        fo = effect(b_o, se_o, ols[3], scale)
        fp = effect(b_p, se_p, dfr_p, scale)
        n_fam, n_slower, fam_fold, fam_p = within_families(g, f) if f in BINARY else (np.nan,) * 4
        rows.append({
            "analysis": label, "input": f, "label": LABEL[f], "n_groups": len(y),
            "n_with": int((g[f] > 0).sum()) if f in BINARY else np.nan,
            "fold_across_species": fo[0], "p_across_species": fo[3],
            "fold_pgls": fp[0], "ci_low_pgls": fp[1], "ci_high_pgls": fp[2], "p_pgls": fp[3], "lambda": lam_used,
            "families_with_both": n_fam, "families_carriers_slower": n_slower, "median_fold_within_family": fam_fold,
            "p_within_family": fam_p,
        })
    out = pd.DataFrame(rows)
    out["p_pgls_bh"] = multipletests(out["p_pgls"], method="fdr_bh")[1]
    ok = out["p_within_family"].notna()
    out.loc[ok, "p_within_family_bh"] = multipletests(out.loc[ok, "p_within_family"], method="fdr_bh")[1]

    # all inputs together in one PGLS model
    cols = BINARY + CONTINUOUS + ["log2_genome_size"]
    Z = g[cols].to_numpy(float)
    sd = np.where(np.isin(cols, CONTINUOUS), Z.std(axis=0), 1.0)
    Z = np.where(np.isin(cols, CONTINUOUS), (Z - Z.mean(axis=0)) / sd, Z)
    b, cov, dfr, lam_all = pgls(y, np.column_stack([base, Z]), C, lam)
    se = np.sqrt(np.diag(cov))[base.shape[1]:]
    joint = [effect(bi, si, dfr) for bi, si in zip(b[base.shape[1]:], se, strict=True)]
    out["fold_pgls_joint"] = [j[0] for j in joint]
    out["p_pgls_joint"] = [j[3] for j in joint]
    out["lambda_joint"] = lam_all
    return out


def show(out):
    print(f"\n===== {out['analysis'].iloc[0]}: {out['n_groups'].iloc[0]} species groups, "
          f"lambda {out['lambda'].median():.2f} (single input), {out['lambda_joint'].iloc[0]:.2f} (all inputs)")
    print(f"{'input':40s}{'n':>5s}{'across':>9s}{'PGLS fold (95% CI)':>24s}{'p adj':>9s}{'joint':>8s}{'families slower':>17s}{'p adj':>8s}")
    for r in out.itertuples():
        n = "" if np.isnan(r.n_with) else f"{int(r.n_with)}"
        fam = "" if np.isnan(r.families_with_both) else f"{int(r.families_carriers_slower)}/{int(r.families_with_both)}"
        fam_p = "" if pd.isna(getattr(r, "p_within_family_bh", np.nan)) else f"{r.p_within_family_bh:.3f}"
        print(f"{r.label:40s}{n:>5s}{r.fold_across_species:8.2f}x{r.fold_pgls:8.2f}x ({r.ci_low_pgls:.2f} to {r.ci_high_pgls:.2f})"
              f"{r.p_pgls_bh:9.3f}{r.fold_pgls_joint:7.2f}x{fam:>17s}{fam_p:>8s}")


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    results = []
    for include, label in ((True, "all labeled species"), (False, "without the literature strains")):
        g = group_table(processed, k, include)
        g, C = place_on_tree(g)
        runs = [run(g, C, label)]
        if include:
            runs.append(run(g, C, "lambda fixed at 1 (Brownian motion)", lam=1.0))
            ex = g["exact_tip"].to_numpy()
            runs.append(run(g[ex], C[np.ix_(ex, ex)], "only species placed at their own GTDB species"))
            amo = g[g["ammonia_monooxygenase"] == 1]
            print("\nspecies with ammonia or methane monooxygenase:")
            print(amo.assign(doubling_h=np.exp(amo["log_y"]))[["doubling_h", "growth_temp_c"]].round(1).to_string())
        for out in runs:
            show(out)
        results += runs
    pd.concat(results).to_csv(processed / "lifestyle_phylogenetic_test.csv", index=False)
    print("\nfold: doubling time with the gene against without (or per SD, or per genome size doubling); above 1 means slower."
          "\nacross: no control for relatedness. joint: all inputs in one PGLS model. p adj: Benjamini-Hochberg.")


if __name__ == "__main__":
    main()
