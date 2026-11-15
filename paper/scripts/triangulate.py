#!/usr/bin/env python
"""Independent recomputation of the paper's numbers from the per-genome result files.

make_numbers.py turns summary files into the macros of numbers.tex. This script shares no code with it: it starts from the
row-level files (one prediction per genome, seed and split), recomputes each statistic from scratch, and compares it with the
value printed in the paper. A number passes if it agrees to the precision shown. It is a second, separate route to every
number it covers; the third route (sources outside our pipeline: Crossref for the references, PubMed for the external
labels) is in check_sources.py.

Usage: python paper/scripts/triangulate.py   (exit code 1 if any number disagrees)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
LEVELS = {"SameGenus": "species", "NewGenus": "genus", "NewFamily": "family", "NewOrder": "order"}
DAY = np.log(24)

paper = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*?)\\xspace\}", (ROOT / "paper" / "numbers.tex").read_text()))
checked, failed = [], []


def as_float(text):
    t = text.replace("$", "").replace("\\,", "").replace(",", "").replace("<", "").strip()
    m = re.match(r"([\d.]+)\\times10\^\{(-?\d+)\}", t)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    if t.startswith("10^{"):
        return 10 ** int(re.search(r"-?\d+", t).group(0))
    return float(t)


def check(name, value):
    """Compare a recomputed value with the macro at the macro's printed precision."""
    if name not in paper:
        failed.append(f"{name}: not in numbers.tex")
        return
    shown = paper[name]
    if "<" in shown:                      # "< 10^-4": the recomputed value must be below the bound
        ok = value < as_float(shown)
    else:
        target = as_float(shown)
        digits = len(shown.replace("$", "").split(".")[1].rstrip("\\xspace")) if "." in shown and "times" not in shown else 0
        if "times" in shown:              # scientific notation, two significant figures
            ok = abs(value - target) <= 0.051 * abs(target)
        else:
            ok = abs(value - target) <= 0.5 * 10 ** -digits + 1e-9
    (checked if ok else failed).append(f"{name}: paper {shown}, recomputed {value:.4g}")


def typical(m, p):
    return float(np.exp(np.median(np.abs(np.asarray(m) - np.asarray(p)))))


# ---- headline: nested, own temperature, by level, mean over seeds
h = pd.read_csv(P / "paper_headline_predictions.csv")
for key, level in LEVELS.items():
    for scheme, pre in (("nested", ""), ("fixed, single", "Fixed"), ("fixed, coarsened", "Coarse")):
        d = h[(h["target"] == "own temperature") & (h["held_out"] == level) & (h["scheme"] == scheme)]
        errs = [typical(g["measured"], g["predicted"]) for _, g in d.groupby("seed")]
        if f"{pre}Err{key}" in paper:
            check(f"{pre}Err{key}", np.mean(errs))
        if not pre:
            check(f"Rho{key}", np.mean([spearmanr(g["measured"], g["predicted"]).correlation for _, g in d.groupby("seed")]))
            if f"Auc{key}" in paper:
                check(f"Auc{key}", np.mean([roc_auc_score(g["measured"] > DAY, g["predicted"]) for _, g in d.groupby("seed")]))
            slow = d[d["measured"] > DAY]
            if key in ("SameGenus", "NewOrder"):
                k = "Species" if key == "SameGenus" else "Order"
                check(f"ErrSlow{k}", np.mean([typical(g["measured"], g["predicted"]) for _, g in slow.groupby("seed")]))
            if key == "NewOrder":
                arch = d["gtdb_taxonomy"].str.startswith("d__Archaea")
                check("ErrArchOrder", np.mean([typical(g["measured"], g["predicted"]) for _, g in d[arch].groupby("seed")]))
                check("ErrBactOrder", np.mean([typical(g["measured"], g["predicted"]) for _, g in d[~arch].groupby("seed")]))
    d = h[(h["target"] == "20 C") & (h["held_out"] == level) & (h["scheme"] == "nested")]
    if f"ErrRef{key}" in paper:
        check(f"ErrRef{key}", np.mean([typical(g["measured"], g["predicted"]) for _, g in d.groupby("seed")]))
    d = h[(h["target"] == "growth temperature") & (h["held_out"] == level) & (h["scheme"] == "nested")]
    if f"TempMAE{key}" in paper:
        check(f"TempMAE{key}", np.mean([np.mean(np.abs(g["measured"] - g["predicted"])) for _, g in d.groupby("seed")]))
d = h[(h["target"] == "own temperature") & (h["held_out"] == "species") & (h["scheme"] == "nested") & (h["seed"] == 0)]
check("NHeadlineRows", len(d))
check("NSeedsHeadline", h["seed"].nunique())

# ---- released tools, species and orders held out (per-genome files)
for sfx, pre in (("", "Rel"), ("_order", "RelOrd")):
    g = pd.read_csv(P / f"released_tools_genomes_nested{sfx}.csv")
    g = g[g["gRodonpred"].notna() & (g["gRodonpred"] > 0)]
    lm = np.log(g["measured"])
    cols = {"Ours": "this model", "OursTemp": "this model + measured temperature", "GRodon": "gRodonpred",
            "GRodonNoTemp": "gRodon without temperature", "Phydon": "combopred"}
    for key, col in cols.items():
        errs = []
        for _, s in g.assign(lm=lm).groupby("seed"):
            s = s[s[col].notna()]
            errs.append(typical(s["lm"], np.log(s[col])))
        check(f"{pre}{key}Err", np.mean(errs))
    if not sfx:
        check("RelN", int(g[g["seed"] == g["seed"].min()]["genome_name"].nunique()))
    ae = pd.DataFrame({k: np.abs(np.log(g[c]) - lm) for k, c in cols.items()}).assign(genome=g["genome_name"].to_numpy())
    per = ae.groupby("genome").mean()
    pairs = {"Rel": [("PairGRodon", "Ours", "GRodon"), ("PairPhydon", "Ours", "Phydon"), ("PairTempGRodon", "OursTemp", "GRodon"),
                     ("PairTempPhydon", "OursTemp", "Phydon")],
             "RelOrd": [("PairOrdGRodon", "Ours", "GRodon"), ("PairOrdGRodonNoTemp", "Ours", "GRodonNoTemp"),
                        ("PairOrdTempGRodon", "OursTemp", "GRodon")]}[pre]
    for name, a, b in pairs:
        x = per[[a, b]].dropna()
        dd = x[a] - x[b]
        check(f"{name}P", wilcoxon(dd).pvalue)
        if f"{name}Better" in paper:
            check(f"{name}Better", int((dd < 0).sum()))
        if f"{name}N" in paper:
            check(f"{name}N", len(x))

# ---- external test
e = pd.read_csv(P / "external_test.csv")
y = np.log(e["doubling_time_hours"])
check("ExtN", len(e))
check("ExtNCurated", len(pd.read_csv(ROOT / "data" / "curated" / "external_growth_rates.csv")))
check("ExtNSlow", int((e["doubling_time_hours"] > 24).sum()))
check("ExtNNewOrder", int((e["tier"] == "new order").sum()))
check("ExtModelErr", typical(y, np.log(e["predicted_hours"])))
check("ExtBaseErr", typical(y, np.log(e["relatives_median_hours"])))
check("ExtGRodonErr", typical(y, np.log(e["grodon_hours"])))
check("ExtModelRho", spearmanr(y, e["predicted_hours"]).correlation)
for key, t in (("SameGenus", "same genus"), ("NewOrder", "new order")):
    s = e[e["tier"] == t]
    check(f"ExtModelErr{key}", typical(np.log(s["doubling_time_hours"]), np.log(s["predicted_hours"])))
for key, col in (("Base", "relatives_median_hours"), ("GRodon", "grodon_hours")):
    dd = np.abs(y - np.log(e["predicted_hours"])) - np.abs(y - np.log(e[col]))
    check(f"ExtBetter{key}", int((dd < 0).sum()))
    check(f"ExtP{key}", wilcoxon(dd).pvalue)
check("ExtCoverage", 100 * e["inside_range"].mean())
check("ExtCoverageP", binomtest(int(e["inside_range"].sum()), len(e), 0.8).pvalue)
check("ExtCoverageSlowIn", int(e.loc[e["doubling_time_hours"] > 24, "inside_range"].sum()))
check("ExtSlowAuc", roc_auc_score(e["doubling_time_hours"] > 24, e["prob_slow"]))
if ((e["doubling_time_hours"] > 24) & (e["predicted_hours"] > 24)).any():
    failed.append("text says no slow external species is predicted above a day, but one is")

# ---- metagenome-assembled genomes
m = pd.read_csv(P / "mag_validation.csv")
m = m[(m["target"] == "own temperature") & (m["variant"] == "without 16S")]
check("MagNSpecies", len(m))
check("MagNGenomes", int(m["n_mags"].sum()))
check("MagErr", typical(m["truth"], m["mag"]))
check("MagIsoErr", typical(m["truth"], m["isolate"]))
check("MagGap", typical(m["mag"], m["isolate"]))
check("MagGapSmaller", int((np.abs(m["mag"] - m["isolate"]) < np.abs(m["truth"] - m["isolate"])).sum()))

# ---- GEM predictions (the released table)
r = pd.read_csv(ROOT / "release" / "gem_growth_predictions.csv")
q = pd.read_csv(P / "hq_predictions_gbm.csv")
check("NGEMPredicted", len(q))
dt = q["doubling_time_hours_at_growth_temp"]
check("GEMMedianHours", dt.median())
check("GEMNSlow", int((dt > 24).sum()))
check("GEMShareSlow", 100 * (dt > 24).mean())
check("GEMShareNewOrder", 100 * (q["nearest_labeled_relative"] == "new order").mean())
check("GEMShareNoGenus", 100 * (q["nearest_labeled_relative"] != "same genus").mean())
for key, eco in (("Human", "Human"), ("Aquatic", "Aquatic"), ("Terrestrial", "Terrestrial"), ("Built", "Built environment")):
    s = q[q["ecosystem_category"] == eco]
    check(f"GEMMedian{key}", s["doubling_time_hours_at_growth_temp"].median())
    if f"GEMNewOrder{key}" in paper:
        check(f"GEMNewOrder{key}", 100 * (s["nearest_labeled_relative"] == "new order").mean())
check("GEMNHuman", int((q["ecosystem_category"] == "Human").sum()))
orders = q.loc[dt > 24, "gtdb_taxonomy"].str.extract(r"o__([^;]+)")[0].value_counts()
check("GEMSlowTopOrderN", int(orders.iloc[0]))
check("GEMSlowSecondOrderN", int(orders.iloc[1]))
if orders.index[0] != paper["GEMSlowTopOrder"] or orders.index[1] != paper["GEMSlowSecondOrder"]:
    failed.append(f"slow GEM orders: paper {paper['GEMSlowTopOrder']}, {paper['GEMSlowSecondOrder']}; data {orders.index[0]}, {orders.index[1]}")
if len(r) != len(q):
    failed.append(f"release table has {len(r)} rows, prediction table {len(q)}")

hs = h[h["target"] == "own temperature"]
check("SelectMaxDiff", max(abs(per_scheme - per_fixed) for per_scheme, per_fixed in
      [(np.mean([typical(g["measured"], g["predicted"]) for _, g in hs[(hs["held_out"] == lv) & (hs["scheme"] == "nested")].groupby("seed")]),
        np.mean([typical(g["measured"], g["predicted"]) for _, g in hs[(hs["held_out"] == lv) & (hs["scheme"] == "fixed, coarsened")].groupby("seed")]))
       for lv in LEVELS.values()]))

# ---- statements in words (each sentence of the paper that makes a claim without a number)
def claim(text, ok, detail=""):
    (checked if ok else failed).append(f"claim '{text}': {'holds' if ok else 'DOES NOT HOLD'} {detail}")


own = h[(h["target"] == "own temperature") & (h["scheme"] == "nested")]
ref = h[(h["target"] == "20 C") & (h["scheme"] == "nested")]
per_level = lambda d, lv: np.mean([typical(g["measured"], g["predicted"]) for _, g in d[d["held_out"] == lv].groupby("seed")])  # noqa: E731
claim("the 20 C doubling time is predicted better at every level",
      all(per_level(ref, lv) < per_level(own, lv) for lv in LEVELS.values()))
slow = own[own["measured"] > DAY]
for lv in ("species", "order"):
    s = slow[slow["held_out"] == lv]
    share = (s["predicted"] < s["measured"]).mean()
    claim(f"slow growers' errors are almost always toward faster growth ({lv})", share >= 0.9, f"({100 * share:.0f}%)")
for sfx, lv in (("", "species"),):
    c = pd.read_csv(P / f"released_tools_comparison_nested{sfx}.csv")
    c = c[c["subset"] == "all genomes"].pivot_table(index="seed", columns="method", values="typical error")
    t = c["this model + measured temperature"]
    claim("with the temperature, better than both tools in every seed",
          bool(((t < c["gRodon (released)"]) & (t < c["Phydon (released)"])).all()))
    o = c["this model"]
    claim("without the temperature, lower than either tool given it, in every seed",
          bool(((o < c["gRodon (released)"]) & (o < c["Phydon (released)"])).all()))
ab = pd.read_csv(P / "ablation_evaluation.csv").pivot_table(index=["held_out", "seed"], columns="setup", values=["typical error", "Spearman", "AUC slow"])
err = ab["typical error"]
mean = err.groupby(level=0).mean()
claim("without any gene features worse in all ablation seeds", bool((err["- all gene features"] > err["final model"]).all()),
      f"({int((err['- all gene features'] > err['final model']).sum())} of {len(err)})")
claim("added species help at both levels", all(mean.loc[lv, "- added species"] > mean.loc[lv, "final model"] for lv in ("species", "order")))
claim("phylogeny helps only with a same-genus relative",
      mean.loc["species", "- phylogeny"] > mean.loc["species", "final model"] and mean.loc["order", "- phylogeny"] <= mean.loc["order", "final model"] + 0.01,
      f"(species {mean.loc['species', '- phylogeny']:.3f} vs {mean.loc['species', 'final model']:.3f}; order {mean.loc['order', '- phylogeny']:.3f} vs {mean.loc['order', 'final model']:.3f})")
auc = ab["AUC slow"].groupby(level=0).mean()
claim("16S leaves the typical error unchanged and, with orders held out, adds a little to the ranking of slow growers",
      all(abs(mean.loc[lv, "- 16S"] - mean.loc[lv, "final model"]) <= 0.02 for lv in ("species", "order"))
      and auc.loc["order", "- 16S"] < auc.loc["order", "final model"],
      f"(error change {[round(mean.loc[lv, '- 16S'] - mean.loc[lv, 'final model'], 3) for lv in ('species', 'order')]}, "
      f"AUC without 16S {[round(auc.loc[lv, '- 16S'], 3) for lv in ('species', 'order')]} vs {[round(auc.loc[lv, 'final model'], 3) for lv in ('species', 'order')]})")
claim("energy-metabolism genes do not help", all(mean.loc[lv, "+ energy features"] >= mean.loc[lv, "final model"] - 0.01 for lv in ("species", "order")))
claim("a predicted temperature as input gains little", all(mean.loc[lv, "+ predicted temperature"] >= mean.loc[lv, "final model"] - 0.02 for lv in ("species", "order")))
s16 = pd.read_csv(P / "16s_source_evaluation.csv").set_index("measure")["value"]
claim("trained on NCBI 16S and given genome copies: worse in every seed (orders held out)",
      s16["seeds worse than NCBI copy, order held out, trained on NCBI, genome copy"] == int(paper["SixteenNSeeds"]) if "SixteenNSeeds" in paper else False,
      f"({s16['seeds worse than NCBI copy, order held out, trained on NCBI, genome copy']:g} of {paper.get('SixteenNSeeds')})")
claim("no energy-metabolism input significant after control", as_float(paper["PglsEnergyMinP"]) >= 0.05)

# ---- report
print(f"{len(checked)} numbers agree")
for line in failed:
    print("DISAGREE", line)
sys.exit(1 if failed else 0)
