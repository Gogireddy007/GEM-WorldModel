#!/usr/bin/env python
"""Writes paper/numbers.tex: every number quoted in the paper as a LaTeX macro, read from the
result files in data/processed (and data/curated). No result is typed into the manuscript by hand.

Run from the repository root after the analyses: .venv/bin/python paper/scripts/make_numbers.py
Each macro is listed with its source in paper/numbers_sources.csv, which the claim audit uses.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
OUT = ROOT / "paper" / "numbers.tex"
SOURCES = ROOT / "paper" / "numbers_sources.csv"
FULL = "traits+phylogeny+16S+gene features"
TIERS = {"SameGenus": "same genus", "NewGenus": "same family, new genus", "NewFamily": "same order, new family", "NewOrder": "new order"}

macros, sources = {}, []
words = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def put(name, value, source, fmt=None):
    text = fmt.format(value) if fmt else str(value)
    macros[name] = text
    sources.append({"macro": name, "value": text, "source": source})


def x(v):
    return f"{v:.2f}$\\times$"


def main():
    # ---- data
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from gem_worldmodel.features import deploy_features
    df, X, names = deploy_features.load_training_table(P, 6, with_cub=True, with_lifestyle=True)
    put("NGenomes", len(df), "load_training_table rows", "{:,}")
    put("NSpecies", df["species"].nunique(), "load_training_table species", "{:,}")
    put("NSlow", int((df["doubling_time_hours"] > 24).sum()), "labels above 24 h")
    put("NArchaea", int((df["archaeon"] == 1).sum()), "archaeal rows")
    put("NInputs", len(names), "deployed feature count")
    put("NLiterature", int((df["origin"] == "literature").sum()), "literature rows")
    put("NGenomeExtension", int((df["origin"] == "Madin genome extension").sum()), "Madin genome extension rows")
    put("NOriginal", int((df["origin"] == "original").sum()), "original corpus rows")
    put("NMadinExtension", int(df["origin"].isin(["bacterial extension", "archaeal extension"]).sum()), "Madin extension rows (traits from the table)")
    ex = pd.read_csv(ROOT / "data" / "curated" / "label_exclusions.csv")
    put("NExcluded", len(ex), "data/curated/label_exclusions.csv")
    put("NExcludedKevbrina", int(ex["source"].str.contains("Kevbrina").sum()), "label_exclusions.csv, Kevbrina rows")
    put("NExcludedWord", words[len(ex)], "label_exclusions.csv (word)")
    put("NExcludedKevbrinaWord", words[int(ex["source"].str.contains("Kevbrina").sum())], "label_exclusions.csv (word)")
    put("NExcludedSourcesWord", words[ex["source"].str.split(" ").str[0].nunique()], "label_exclusions.csv sources (word)")
    put("NLiteratureWord", words[int((df["origin"] == "literature").sum())], "literature rows (word)")

    # ---- accuracy by novelty (full model), nested: settings chosen inside each split, deployed training scheme
    a = pd.read_csv(P / "paper_headline_accuracy.csv")
    LEVEL = {"SameGenus": "species", "NewGenus": "genus", "NewFamily": "family", "NewOrder": "order"}
    for key, level in LEVEL.items():
        def mean(target, metric, scheme="nested", level=level):
            return a[(a["target"] == target) & (a["scheme"] == scheme) & (a["held_out"] == level)][metric].mean()
        put(f"Err{key}", mean("own temperature", "typical error"), "paper_headline_accuracy.csv nested own temperature", "{:.2f}")
        put(f"Rho{key}", mean("own temperature", "Spearman"), "paper_headline_accuracy.csv nested own temperature", "{:.2f}")
        put(f"Auc{key}", mean("own temperature", "AUC slow"), "paper_headline_accuracy.csv nested own temperature", "{:.2f}")
        put(f"ErrRef{key}", mean("20 C", "typical error"), "paper_headline_accuracy.csv nested 20 C", "{:.2f}")
        put(f"RhoRef{key}", mean("20 C", "Spearman"), "paper_headline_accuracy.csv nested 20 C", "{:.2f}")
        put(f"TempMAE{key}", mean("growth temperature", "mean abs error"), "paper_headline_accuracy.csv nested growth temperature", "{:.1f}")
        put(f"RhoTemp{key}", mean("growth temperature", "Spearman"), "paper_headline_accuracy.csv nested growth temperature", "{:.2f}")
        put(f"FixedErr{key}", mean("own temperature", "typical error", "fixed, single"), "paper_headline_accuracy.csv fixed, single", "{:.2f}")
        put(f"CoarseErr{key}", mean("own temperature", "typical error", "fixed, coarsened"), "paper_headline_accuracy.csv fixed, coarsened", "{:.2f}")
        put(f"FixedErrRef{key}", mean("20 C", "typical error", "fixed, single"), "paper_headline_accuracy.csv fixed, single, 20 C", "{:.2f}")
    put("NSeedsHeadline", a["seed"].nunique(), "paper_headline_accuracy.csv seeds")
    put("NHeadlineRows", int(a.loc[a["target"] == "own temperature", "n_rows"].max()), "paper_headline_accuracy.csv rows", "{:,}")

    # slow calls and range coverage from the nested held-out predictions (mean over seeds)
    hp = pd.read_csv(P / "paper_headline_predictions.csv")
    hp = hp[(hp["target"] == "own temperature") & (hp["scheme"] == "nested")]
    acc = pd.read_csv(P / "final_accuracy_by_tier.csv").set_index(["model", "target", "held_out"])
    for key, level in (("Species", "species"), ("Order", "order")):
        d = hp[hp["held_out"] == level].groupby("species_group")[["measured", "predicted"]].mean()
        called, slow = d["predicted"] > np.log(24), d["measured"] > np.log(24)
        put(f"SlowCalled{key}", int(called.sum()), f"paper_headline_predictions.csv nested, {level} held out")
        put(f"SlowCalledTrue{key}", int((called & slow).sum()), f"paper_headline_predictions.csv nested, {level} held out")
        put(f"SlowTotal{key}", int(slow.sum()), f"paper_headline_predictions.csv nested, {level} held out")
    for key, level in LEVEL.items():
        d = hp[hp["held_out"] == level]
        q = acc.loc[(FULL, "own temperature", level)]
        resid = d["measured"] - d["predicted"]
        put(f"Cover{key}", 100 * ((resid >= q["resid_q10"]) & (resid <= q["resid_q90"])).mean(),
            "deployed 80% ranges (final_accuracy_by_tier.csv) applied to nested held-out predictions", "{:.0f}")
    # the same check for the two other targets, reported as the lowest and highest coverage over the four levels
    hp_all = pd.read_csv(P / "paper_headline_predictions.csv")
    hp_all = hp_all[hp_all["scheme"] == "nested"]
    for key, target in (("Ref", "20 C"), ("Temp", "growth temperature")):
        cover = []
        for level in LEVEL.values():
            d = hp_all[(hp_all["target"] == target) & (hp_all["held_out"] == level)]
            q = acc.loc[(FULL, target, level)]
            resid = d["measured"] - d["predicted"]
            cover.append(100 * ((resid >= q["resid_q10"]) & (resid <= q["resid_q90"])).mean())
        put(f"Cover{key}Min", min(cover), f"deployed 80% ranges for {target} on nested held-out predictions, lowest level", "{:.0f}")
        put(f"Cover{key}Max", max(cover), f"deployed 80% ranges for {target} on nested held-out predictions, highest level", "{:.0f}")

    # ---- GEM predictions
    h = pd.read_csv(P / "hq_predictions_gbm.csv")
    put("NGEMPredicted", len(h), "hq_predictions_gbm.csv", "{:,}")
    share = h["nearest_labeled_relative"].value_counts(normalize=True) * 100
    for key, tier in TIERS.items():
        put(f"GEMShare{key}", share.get(tier, 0), "hq_predictions_gbm.csv tier shares", "{:.0f}")
    put("GEMShareNoGenus", 100 - share.get(TIERS["SameGenus"], 0), "hq_predictions_gbm.csv, no labeled relative in the same genus", "{:.0f}")
    put("GEMMedianHours", h["doubling_time_hours_at_growth_temp"].median(), "hq_predictions_gbm.csv", "{:.1f}")
    put("GEMShareSlow", 100 * (h["doubling_time_hours_at_growth_temp"] > 24).mean(), "hq_predictions_gbm.csv", "{:.1f}")
    put("GEMNSlow", int((h["doubling_time_hours_at_growth_temp"] > 24).sum()), "hq_predictions_gbm.csv")
    slow_orders = h.loc[h["doubling_time_hours_at_growth_temp"] > 24, "gtdb_taxonomy"].str.extract(r"o__([^;]+)")[0].value_counts()
    put("GEMSlowTopOrder", slow_orders.index[0], "hq_predictions_gbm.csv, order with most genomes slower than a day")
    put("GEMSlowTopOrderN", int(slow_orders.iloc[0]), "hq_predictions_gbm.csv")
    for key, eco in (("Human", "Human"), ("Aquatic", "Aquatic"), ("Built", "Built environment"), ("Terrestrial", "Terrestrial"),
                     ("Wastewater", "Wastewater")):
        put(f"GEMMedian{key}", h.loc[h["ecosystem_category"] == eco, "doubling_time_hours_at_growth_temp"].median(), f"hq_predictions_gbm.csv {eco}", "{:.1f}")
        put(f"GEMN{key}", int((h["ecosystem_category"] == eco).sum()), f"hq_predictions_gbm.csv {eco}", "{:,}")
        put(f"GEMNewOrder{key}", 100 * (h.loc[h["ecosystem_category"] == eco, "nearest_labeled_relative"] == "new order").mean(),
            f"hq_predictions_gbm.csv {eco}, share without a labeled relative in the order", "{:.0f}")
    assert slow_orders.index[1] == "Nitrososphaerales", "the text calls the second order ammonia-oxidizing archaea"
    put("GEMSlowSecondOrder", slow_orders.index[1], "hq_predictions_gbm.csv")
    put("GEMSlowSecondOrderN", int(slow_orders.iloc[1]), "hq_predictions_gbm.csv")

    # ---- released tools (species held out, same genomes); our model built nested inside each split
    r = pd.read_csv(P / "released_tools_comparison_nested.csv")
    r = r[r["subset"] == "all genomes"]
    piv = r.pivot_table(index="seed", columns="method", values="typical error")
    r = r.groupby("method")[["n", "typical error", "Spearman", "AUC slow"]].mean()
    for key, name in (("Ours", "this model"), ("OursTemp", "this model + measured temperature"), ("GRodon", "gRodon (released)"),
                      ("GRodonNoTemp", "gRodon (released), no temperature"), ("Phydon", "Phydon (released)")):
        put(f"Rel{key}Err", r.loc[name, "typical error"], "released_tools_comparison_nested.csv", "{:.2f}")
        put(f"Rel{key}Rho", r.loc[name, "Spearman"], "released_tools_comparison_nested.csv", "{:.2f}")
        put(f"Rel{key}Auc", r.loc[name, "AUC slow"], "released_tools_comparison_nested.csv", "{:.2f}")
    put("RelN", int(r.loc["this model", "n"]), "released_tools_comparison_nested.csv")
    put("RelNPhydon", int(r.loc["Phydon (released)", "n"]), "released_tools_comparison_nested.csv")
    put("RelNSeeds", len(piv), "released_tools_comparison_nested.csv seeds")
    put("RelOursTempBeatsPhydon", int((piv["this model + measured temperature"] < piv["Phydon (released)"]).sum()), "seeds, typical error")
    put("RelOursTempBeatsGRodon", int((piv["this model + measured temperature"] < piv["gRodon (released)"]).sum()), "seeds, typical error")
    put("RelOursBeatsPhydon", int((piv["this model"] < piv["Phydon (released)"]).sum()), "seeds, typical error")
    put("RelOursBeatsGRodon", int((piv["this model"] < piv["gRodon (released)"]).sum()), "seeds, typical error")

    # ---- label audit checks
    ac = pd.read_csv(P / "audit_controls.csv")
    pc = ac[ac["check"] == "positive control"]
    put("PosCtrlCaughtWord", words[int(pc["caught"].sum())], "audit_controls.csv positive control (word)")
    put("PosCtrlNWord", words[len(pc)], "audit_controls.csv positive control (word)")
    mo = ac[ac["check"] == "monooxygenase PGLS"].set_index("species")
    for key, row in (("With", "Kevbrina values put back"), ("Without", "current exclusions")):
        put(f"KevFold{key}", mo.loc[row, "fold_pgls"], "audit_controls.csv", "{:.1f}")
        put(f"KevLow{key}", mo.loc[row, "ci_low_pgls"], "audit_controls.csv", "{:.1f}")
        put(f"KevHigh{key}", mo.loc[row, "ci_high_pgls"], "audit_controls.csv", "{:.1f}")
        put(f"KevP{key}", mo.loc[row, "p_pgls_bh"], "audit_controls.csv", "{:.2f}")
        put(f"KevN{key}", int(mo.loc[row, "n_with"]), "audit_controls.csv")
    kev = pc[pc["species"].isin(ex.loc[ex["source"].str.contains("Kevbrina"), "species"])]
    put("KevLabelMin", kev["label_hours"].min(), "audit_controls.csv Kevbrina labels", "{:.0f}")
    put("KevLabelMax", kev["label_hours"].max(), "audit_controls.csv Kevbrina labels", "{:.0f}")
    put("KevRecordMin", kev["closest_record_hours"].min(), "audit_controls.csv Kevbrina source records", "{:,.0f}")
    put("KevRecordMax", kev["closest_record_hours"].max(), "audit_controls.csv Kevbrina source records", "{:,.0f}")
    put("KevRatio", (kev["closest_record_hours"] / kev["label_hours"]).median(), "audit_controls.csv record / label, median", "{:.0f}")
    st = pd.read_csv(P / "label_sources_studies.csv")
    put("NStudiesCompared", len(st), "label_sources_studies.csv")
    put("NStudiesFlagged", int(st["flag"].sum()), "label_sources_studies.csv")

    # ---- 16S source
    s16 = pd.read_csv(P / "16s_source_evaluation.csv").set_index("measure")
    put("SixteenOwnDist", s16.loc["sequence distance, own copies (median)", "value"], "16s_source_evaluation.csv", "{:.3f}")
    put("SixteenNearDist", s16.loc["sequence distance, closest other species (median)", "value"], "16s_source_evaluation.csv", "{:.3f}")
    put("SixteenN", int(s16.loc["sequence distance, own copies (median)", "n"]), "16s_source_evaluation.csv")
    put("SixteenFarther", int(s16.loc["rows whose own copies differ more than the closest other species", "value"]), "16s_source_evaluation.csv")
    put("SixteenShift", s16.loc["16S input shift in SD (median)", "value"], "16s_source_evaluation.csv", "{:.2f}")
    own16 = pd.read_csv(P / "labeled_16s_from_genomes.csv").fillna("")
    put("SixteenNGenomeCopy", int(df["accession"].isin(set(own16.loc[own16["sequence"] != "", "accession"])).sum()), "labeled_16s_from_genomes.csv, training rows with a genome copy")
    for key, level in (("Species", "species"), ("Order", "order")):
        put(f"SixteenErrNcbi{key}", s16.loc[f"typical error, {level} held out, NCBI copy", "value"], "16s_source_evaluation.csv", "{:.2f}")
        put(f"SixteenErrGenome{key}", s16.loc[f"typical error, {level} held out, genome copy", "value"], "16s_source_evaluation.csv", "{:.2f}")
        put(f"SixteenErrMixed{key}", s16.loc[f"typical error, {level} held out, trained on NCBI, genome copy", "value"], "16s_source_evaluation.csv", "{:.2f}")
        put(f"SixteenMixedWorse{key}", int(s16.loc[f"seeds worse than NCBI copy, {level} held out, trained on NCBI, genome copy", "value"]), "16s_source_evaluation.csv")
    put("SixteenNSeeds", int(s16.loc["seeds worse than NCBI copy, species held out, genome copy", "n"]), "16s_source_evaluation.csv")

    # ---- ablations against the final model (same rows, folds and seeds)
    ab = pd.read_csv(P / "ablation_evaluation.csv")
    put("NSeedsAblation", ab["seed"].nunique(), "ablation_evaluation.csv seeds")
    keys = {"GStat": "- gRodon statistics", "Life": "- lifestyle genes", "Genes": "- all gene features", "Sixteen": "- 16S",
            "Phylo": "- phylogeny", "Added": "- added species", "Energy": "+ energy features", "PredTemp": "+ predicted temperature"}
    for level, lk in (("species", "Species"), ("order", "Order")):
        d = ab[ab["held_out"] == level]
        piv = d.pivot_table(index="seed", columns="setup", values=["typical error", "Spearman", "AUC slow"])
        base = piv.xs("final model", axis=1, level=1)
        put(f"AblFinalErr{lk}", base["typical error"].mean(), f"ablation_evaluation.csv {level}", "{:.2f}")
        put(f"AblFinalRho{lk}", base["Spearman"].mean(), f"ablation_evaluation.csv {level}", "{:.2f}")
        put(f"AblFinalAuc{lk}", base["AUC slow"].mean(), f"ablation_evaluation.csv {level}", "{:.2f}")
        for key, name in keys.items():
            v = piv.xs(name, axis=1, level=1)
            put(f"Abl{key}Err{lk}", v["typical error"].mean(), f"ablation_evaluation.csv {level} {name}", "{:.2f}")
            put(f"Abl{key}Rho{lk}", v["Spearman"].mean(), f"ablation_evaluation.csv {level} {name}", "{:.2f}")
            put(f"Abl{key}Auc{lk}", v["AUC slow"].mean(), f"ablation_evaluation.csv {level} {name}", "{:.2f}")
            put(f"Abl{key}Worse{lk}", int((v["typical error"] > base["typical error"]).sum()), f"ablation_evaluation.csv {level} {name}, seeds")

    def ablation(file, base, new, key):
        d = pd.read_csv(P / file)
        d = d[d["held_out"] == "order"].groupby("setup")[["typical error", "Spearman", "AUC slow"]].mean()
        for metric, short in (("typical error", "Err"), ("Spearman", "Rho"), ("AUC slow", "Auc")):
            put(f"{key}{short}Before", d.loc[base, metric], f"{file} order {base}", "{:.2f}")
            put(f"{key}{short}After", d.loc[new, metric], f"{file} order {new}", "{:.2f}")
    ablation("tuning_evaluation.csv", "deployed", "hist + forest tuned", "Tune")
    tune = pd.read_csv(P / "tuning_evaluation.csv")
    tune = tune[tune["held_out"] == "order"].pivot_table(index="seed", columns="setup", values="typical error")
    put("TuneBetter", int((tune["hist + forest tuned"] < tune["deployed"]).sum()), "tuning_evaluation.csv order, seeds")
    put("TuneNSeeds", len(tune), "tuning_evaluation.csv order, seeds")

    # ---- real MAGs (without 16S, own temperature)
    m = pd.read_csv(P / "mag_validation.csv")
    m = m[(m["target"] == "own temperature") & (m["variant"] == "without 16S")]
    put("MagErr", np.exp(np.median(np.abs(m["truth"] - m["mag"]))), "mag_validation.csv", "{:.2f}")
    put("MagIsoErr", np.exp(np.median(np.abs(m["truth"] - m["isolate"]))), "mag_validation.csv", "{:.2f}")
    put("MagNSpecies", len(m), "mag_validation.csv")
    put("MagNGenomes", int(m["n_mags"].sum()), "mag_validation.csv", "{:,}")
    gap, iso_err = (m["mag"] - m["isolate"]).abs(), (m["truth"] - m["isolate"]).abs()
    put("MagGap", np.exp(gap.median()), "mag_validation.csv median |MAG - isolate prediction|", "{:.2f}")
    put("MagGapMax", np.exp(gap.max()), "mag_validation.csv largest |MAG - isolate prediction|", "{:.1f}")
    put("MagGapSmaller", int((gap < iso_err).sum()), "mag_validation.csv species where the gap is below the isolate error")

    # ---- independent test (bacteria never seen in training)
    i = pd.read_csv(P / "independent_test.csv")
    ib = i[(i["archaeon"] == 0) & (i["target"] == "own temperature")]
    put("IndepErr", np.exp(np.median(np.abs(ib["truth"] - ib["pred"]))), "independent_test.csv", "{:.2f}")
    put("IndepN", len(ib), "independent_test.csv")

    # ---- pressure tests (pressure_tests.py): baselines, bootstrap intervals, shuffled labels, paired tests, MAG intervals
    pt = pd.read_csv(P / "pressure_tests.csv")

    def signed2(v):
        # two decimals with a typeset minus sign
        return f"{v:.2f}".replace("-", "$-$")

    base = pt[pt["test"] == "baseline"].groupby(["held_out", "method"])[["typical error", "Spearman", "AUC slow"]].mean()
    for key, level in LEVEL.items():
        put(f"BaseRelErr{key}", base.loc[(level, "relative median"), "typical error"], "pressure_tests.csv baseline", "{:.2f}")
        put(f"BaseRelRho{key}", base.loc[(level, "relative median"), "Spearman"], "pressure_tests.csv baseline", "{:.2f}")
        put(f"BaseRelAuc{key}", base.loc[(level, "relative median"), "AUC slow"], "pressure_tests.csv baseline", "{:.2f}")
        put(f"BaseMedErr{key}", base.loc[(level, "training median"), "typical error"], "pressure_tests.csv baseline", "{:.2f}")
        b = pt[(pt["test"] == "bootstrap") & (pt["held_out"] == level)].set_index("statistic")
        for stat, name in (("typical error", "Err"), ("Spearman", "Rho"), ("error difference to relative median", "Diff")):
            put(f"CI{name}Lo{key}", signed2(b.loc[stat, "low"]), "pressure_tests.csv bootstrap 2.5%")
            put(f"CI{name}Hi{key}", signed2(b.loc[stat, "high"]), "pressure_tests.csv bootstrap 97.5%")
    sh = pt[pt["test"] == "shuffled labels"].groupby(["held_out", "method"])[["typical error", "Spearman"]].mean()
    for key, level in (("Species", "species"), ("Order", "order")):
        put(f"ShuffleErr{key}", sh.loc[(level, "deployed model"), "typical error"], "pressure_tests.csv shuffled labels", "{:.2f}")
        put(f"ShuffleRho{key}", signed2(sh.loc[(level, "deployed model"), "Spearman"]), "pressure_tests.csv shuffled labels")
        put(f"ShuffleMedErr{key}", sh.loc[(level, "training median"), "typical error"], "pressure_tests.csv shuffled labels", "{:.2f}")
    pr = pt[pt["test"] == "paired"].set_index("method")
    if len(pr):
        for key, other in (("GRodon", "gRodon (released)"), ("Phydon", "Phydon (released)")):
            r = pr.loc[f"this model vs {other}"]
            put(f"Pair{key}N", int(r["n"]), "pressure_tests.csv paired")
            put(f"Pair{key}Better", int(r["better"]), "pressure_tests.csv paired, genomes where ours has the smaller error")
            put(f"Pair{key}P", r["p"], "pressure_tests.csv paired Wilcoxon", "{:.1e}" if r["p"] < 0.001 else "{:.2f}")
            put(f"Pair{key}DiffLo", signed2(r["low"]), "pressure_tests.csv paired bootstrap 2.5%")
            put(f"Pair{key}DiffHi", signed2(r["high"]), "pressure_tests.csv paired bootstrap 97.5%")
            t = pr.loc[f"this model + measured temperature vs {other}"]
            put(f"PairTemp{key}N", int(t["n"]), "pressure_tests.csv paired, with temperature")
            put(f"PairTemp{key}Better", int(t["better"]), "pressure_tests.csv paired, with temperature")
            mant, expo = f"{t['p']:.1e}".split("e")
            put(f"PairTemp{key}P", f"${mant}\\times10^{{{int(expo)}}}$" if t["p"] < 0.001 else f"{t['p']:.3f}",
                "pressure_tests.csv paired Wilcoxon, with temperature")
    mg = pt[pt["test"] == "MAGs"].set_index("statistic")
    put("CIMagErrLo", mg.loc["MAG typical error", "low"], "pressure_tests.csv MAGs", "{:.2f}")
    put("CIMagErrHi", mg.loc["MAG typical error", "high"], "pressure_tests.csv MAGs", "{:.2f}")
    put("CIMagDiffLo", signed2(mg.loc["MAG minus isolate typical error", "low"]), "pressure_tests.csv MAGs")
    put("CIMagDiffHi", signed2(mg.loc["MAG minus isolate typical error", "high"]), "pressure_tests.csv MAGs")

    # ---- ranges by novelty and domain, and the slow-growth probability (calibrate_ranges.py; cross-fitted over genera)
    rv = pd.read_csv(P / "range_evaluation.csv")
    cv = rv[(rv["test"] == "coverage") & (rv["target"] == "own temperature")]
    for key, level in (("Species", "species"), ("Order", "order")):
        for scheme, sk in (("novelty", "Nov"), ("novelty and domain", "Dom")):
            for group, gk in (("all", "All"), ("archaea", "Arch"), ("slow in truth", "Slow"), ("fast in truth", "Fast")):
                v = cv[(cv["held_out"] == level) & (cv["ranges"] == scheme) & (cv["group"] == group)]["value"].iloc[0]
                put(f"Cov{sk}{gk}{key}", 100 * v, f"range_evaluation.csv {level} {scheme} {group}", "{:.0f}")
    cvd = cv[cv["ranges"] == "novelty and domain"]
    put("CovDomArchMin", 100 * cvd[cvd["group"] == "archaea"]["value"].min(), "range_evaluation.csv, archaea, lowest level", "{:.0f}")
    put("CovDomSlowMin", 100 * cvd[cvd["group"] == "slow in truth"]["value"].min(), "range_evaluation.csv, slow, lowest level", "{:.0f}")
    put("CovDomSlowMax", 100 * cvd[cvd["group"] == "slow in truth"]["value"].max(), "range_evaluation.csv, slow, highest level", "{:.0f}")
    sp = rv[rv["test"] == "slow probability"].pivot_table(index="group", columns="held_out", values="value")
    for key, level in (("Species", "species"), ("Order", "order")):
        put(f"SlowProbAuc{key}", sp.loc["AUC", level], "range_evaluation.csv slow probability", "{:.2f}")
        put(f"SlowProbTop{key}", 100 * sp.loc["top fifth, share slow", level], "range_evaluation.csv slow probability", "{:.0f}")
        put(f"SlowProbBrier{key}", sp.loc["Brier", level], "range_evaluation.csv slow probability", "{:.3f}")
        put(f"SlowProbBrierBase{key}", sp.loc["Brier of base rate", level], "range_evaluation.csv slow probability", "{:.3f}")
    put("SlowProbECEMax", sp.loc["calibration error"].max(), "range_evaluation.csv slow probability, largest over levels", "{:.3f}")
    put("SlowBaseRate", 100 * sp.loc["base rate", "species"], "range_evaluation.csv slow probability", "{:.0f}")
    # slow growers and archaea: typical error of the nested predictions, mean over seeds
    hpn = pd.read_csv(P / "paper_headline_predictions.csv")
    hpn = hpn[(hpn["target"] == "own temperature") & (hpn["scheme"] == "nested")]
    hpn = hpn.assign(arch=hpn["gtdb_taxonomy"].str.startswith("d__Archaea"), slow=hpn["measured"] > np.log(24))
    te = lambda x: np.exp(np.median(np.abs(x["measured"] - x["predicted"])))  # noqa: E731
    for key, level in (("Species", "species"), ("Order", "order")):
        d = hpn[hpn["held_out"] == level]
        put(f"ErrSlow{key}", d[d["slow"]].groupby("seed").apply(te).mean(), "paper_headline_predictions.csv nested, slow", "{:.1f}")
        put(f"ErrArch{key}", d[d["arch"]].groupby("seed").apply(te).mean(), "paper_headline_predictions.csv nested, archaea", "{:.2f}")
        put(f"ErrBact{key}", d[~d["arch"]].groupby("seed").apply(te).mean(), "paper_headline_predictions.csv nested, bacteria", "{:.2f}")
    put("GEMNArchaea", int(h["gtdb_taxonomy"].astype(str).str.startswith("d__Archaea").sum()), "hq_predictions_gbm.csv archaea", "{:,}")

    # ---- learned-representation baseline (embed_proteins_esm.py, evaluate_esm_baseline.py)
    if (P / "esm_baseline.csv").exists():
        eb = pd.read_csv(P / "esm_baseline.csv")
        m = eb.groupby(["variant", "held_out"])[["typical error", "Spearman", "AUC slow"]].mean()
        put("EsmN", int(eb["n"].iloc[0]), "esm_baseline.csv genomes with an embedding")
        for key, level in LEVEL.items():
            for variant, vk in (("features", "Feat"), ("ESM-2, linear probe", "Probe"), ("ESM-2, trees", "Trees"), ("features + ESM-2", "Comb")):
                put(f"Esm{vk}Err{key}", m.loc[(variant, level), "typical error"], "esm_baseline.csv", "{:.2f}")
                put(f"Esm{vk}Rho{key}", m.loc[(variant, level), "Spearman"], "esm_baseline.csv", "{:.2f}")
        g = pd.read_csv(P / "esm_baseline_genomes.csv")
        g["ae"] = (g["measured"] - g["predicted"]).abs()
        for key, level in (("SameGenus", "species"), ("NewOrder", "order")):
            w = g[g["held_out"] == level].pivot_table(index="accession", columns="variant", values="ae")
            for variant, vk in (("ESM-2, linear probe", "Probe"), ("features + ESM-2", "Comb")):
                d = w[variant] - w["features"]
                pv = wilcoxon(d).pvalue
                mant, expo = f"{pv:.1e}".split("e")
                put(f"Esm{vk}P{key}", f"${mant}\\times10^{{{int(expo)}}}$" if pv < 0.001 else f"{pv:.2f}", "esm_baseline_genomes.csv paired Wilcoxon")
                put(f"Esm{vk}Better{key}", int((d < 0).sum()), "esm_baseline_genomes.csv genomes with the smaller error")
            put(f"EsmPairN{key}", len(w), "esm_baseline_genomes.csv genomes")

    # ---- second-round pressure tests (pressure_tests_round2.py): ten splits, gRodon-estimate leakage, flagged studies
    r2 = pd.read_csv(P / "pressure_tests_round2.csv")

    def r2row(test, level, a):
        return r2[(r2["test"] == test) & (r2["held_out"] == level) & (r2["a"] == a)].iloc[0]

    def ptex(pv):
        return "$<$\\,10$^{-4}$" if pv < 1e-4 else (f"{pv:.3f}" if pv < 0.01 else f"{pv:.2f}")

    t0 = r2row("ten splits", "species", "model")
    put("TenNSeeds", int(t0["n_seeds"]), "pressure_tests_round2.csv")
    put("TenModelErrSpecies", t0["error_a"], "pressure_tests_round2.csv", "{:.2f}")
    put("TenBaseErrSpecies", t0["error_b"], "pressure_tests_round2.csv", "{:.2f}")
    put("TenModelRhoSpecies", t0["rho_a"], "pressure_tests_round2.csv", "{:.2f}")
    put("TenBaseRhoSpecies", t0["rho_b"], "pressure_tests_round2.csv", "{:.2f}")
    put("TenModelWinsSpecies", int(t0["seeds_a_better"]), "pressure_tests_round2.csv")
    put("TenPGenomeSpecies", ptex(t0["p_genomes"]), "pressure_tests_round2.csv")
    deep = r2[(r2["test"] == "ten splits") & (r2["held_out"] != "species")]
    put("TenModelWinsDeepMin", int(deep["seeds_a_better"].min()), "pressure_tests_round2.csv genus to order")
    put("TenPGenomeDeepMax", ptex(deep["p_genomes"].max()), "pressure_tests_round2.csv genus to order")
    for key, level in (("Species", "species"), ("Order", "order")):
        rf = r2row("gRodon estimate", level, "in-split gRodon refit")
        put(f"RefitErr{key}", rf["error_a"], "pressure_tests_round2.csv", "{:.2f}")
        put(f"RefitFullErr{key}", rf["error_b"], "pressure_tests_round2.csv", "{:.2f}")
        fl = r2row("flagged studies", level, "without flagged studies")
        put(f"FlagErr{key}", fl["error_a"], "pressure_tests_round2.csv", "{:.2f}")
        put(f"FlagFullErr{key}", fl["error_b"], "pressure_tests_round2.csv", "{:.2f}")
    refit = r2[r2["a"] == "in-split gRodon refit"]
    put("RefitMaxDiff", (refit["error_a"] - refit["error_b"]).max(), "pressure_tests_round2.csv largest increase", "{:.2f}")
    put("FlagNLabels", int(r2row("ten splits", "species", "model")["n_genomes"] - r2row("flagged studies", "species", "without flagged studies")["n_genomes"]),
        "pressure_tests_round2.csv labels from flagged studies")

    # ---- the one genome listed twice (two spellings of one species, one species group): effect on the headline scores
    from scipy.stats import spearmanr
    hh = pd.read_csv(P / "paper_headline_predictions.csv")
    hh = hh[(hh["target"] == "own temperature") & (hh["scheme"] == "nested")]
    dup_change = 0.0
    for _, d in hh.groupby(["held_out", "seed"]):
        dd = d[d["species"] != "Clostridium thermoalcaliphilum"]
        e = lambda x: np.exp(np.median(np.abs(x["measured"] - x["predicted"])))  # noqa: E731
        dup_change = max(dup_change, abs(e(d) - e(dd)), abs(spearmanr(d["measured"], d["predicted"]).correlation
                                                         - spearmanr(dd["measured"], dd["predicted"]).correlation))
    put("DupMaxChange", dup_change, "paper_headline_predictions.csv, duplicate row removed", "{:.3f}")

    # ---- released tools with whole orders held out (Phydon's training table cut by order); released_tools_order_test.py
    ro = pd.read_csv(P / "released_tools_comparison_nested_order.csv")
    ro = ro[ro["subset"] == "all genomes"].groupby("method")[["n", "typical error", "Spearman", "AUC slow"]].mean()
    for key, name in (("Ours", "this model"), ("OursTemp", "this model + measured temperature"), ("GRodon", "gRodon (released)"),
                      ("GRodonNoTemp", "gRodon (released), no temperature"), ("Phydon", "Phydon (released)")):
        put(f"RelOrd{key}Err", ro.loc[name, "typical error"], "released_tools_comparison_nested_order.csv", "{:.2f}")
        put(f"RelOrd{key}Rho", ro.loc[name, "Spearman"], "released_tools_comparison_nested_order.csv", "{:.2f}")
        put(f"RelOrd{key}Auc", ro.loc[name, "AUC slow"], "released_tools_comparison_nested_order.csv", "{:.2f}")
    po = pd.read_csv(P / "released_tools_order_paired.csv").set_index("method")
    for key, m in (("GRodon", "this model vs gRodon (released)"), ("Phydon", "this model vs Phydon (released)"),
                   ("GRodonNoTemp", "this model vs gRodon (released), no temperature"),
                   ("TempGRodon", "this model + measured temperature vs gRodon (released)")):
        put(f"PairOrd{key}N", int(po.loc[m, "n"]), "released_tools_order_paired.csv")
        put(f"PairOrd{key}Better", int(po.loc[m, "better"]), "released_tools_order_paired.csv genomes where ours is closer")
        put(f"PairOrd{key}P", ptex(po.loc[m, "p"]), "released_tools_order_paired.csv Wilcoxon")

    # ---- external test (evaluate_external.py): species descriptions outside Madin and our labels
    from scipy.stats import binomtest
    cur = pd.read_csv(ROOT / "data/curated/external_growth_rates.csv")
    et = pd.read_csv(P / "external_test.csv")
    es = pd.read_csv(P / "external_summary.csv")
    put("ExtNCurated", len(cur), "data/curated/external_growth_rates.csv")
    put("NSeedsExternal", len(re.search(r"SEEDS = \(([^)]*)\)", (ROOT / "scripts/accuracy/evaluate_external.py").read_text()).group(1).split(",")),
        "evaluate_external.py SEEDS")
    pro = pd.read_csv(P / "phydon_rep_orders.csv")
    put("PhydonOrderMapped", 100 * pro["order"].notna().mean(), "phydon_rep_orders.csv, Phydon species with a GTDB order", "{:.1f}")
    put("ExtNSources", cur["source"].nunique(), "data/curated/external_growth_rates.csv distinct sources")
    put("ExtN", len(et), "external_test.csv")
    put("ExtNSlow", int((et["doubling_time_hours"] > 24).sum()), "external_test.csv slower than a day")
    put("ExtNArchaea", int((et["domain"] == "archaea").sum()), "external_test.csv")
    for key, t in TIERS.items():
        put(f"ExtN{key}", int((et["tier"] == t).sum()), f"external_test.csv {t}")

    def es_row(group, method):
        return es[(es["group"] == group) & (es["method"] == method)].iloc[0]

    for key, method in (("Model", "model"), ("Base", "relatives' median"), ("GRodon", "gRodon, no temperature")):
        r = es_row("all", method)
        put(f"Ext{key}Err", r["typical error"], "external_summary.csv", "{:.2f}")
        put(f"Ext{key}Lo", r["low"], "external_summary.csv genus bootstrap", "{:.2f}")
        put(f"Ext{key}Hi", r["high"], "external_summary.csv genus bootstrap", "{:.2f}")
        put(f"Ext{key}Rho", r["Spearman"], "external_summary.csv", "{:.2f}")
    for key, t in TIERS.items():
        put(f"ExtModelErr{key}", es_row(t, "model")["typical error"], f"external_summary.csv {t}", "{:.2f}")
        put(f"ExtBaseErr{key}", es_row(t, "relatives' median")["typical error"], f"external_summary.csv {t}", "{:.2f}")
    for key, other in (("Base", "relatives' median"), ("GRodon", "gRodon, no temperature")):
        r = es_row("paired", f"model vs {other}")
        put(f"ExtBetter{key}", int(r["better"]), "external_summary.csv paired, genomes where the model is closer")
        put(f"ExtP{key}", ptex(r["p"]), "external_summary.csv paired Wilcoxon")
    cov = es_row("ranges", "coverage")["value"]
    put("ExtCoverage", 100 * cov, "external_summary.csv 80% ranges", "{:.0f}")
    put("ExtCoverageP", ptex(binomtest(int(round(cov * len(et))), len(et), 0.8).pvalue), "binomial test against 80%")
    put("ExtCoverageSlowIn", int(et.loc[et["doubling_time_hours"] > 24, "inside_range"].sum()), "external_test.csv slow genomes inside the range")
    put("ExtSlowAuc", es_row("slow", "AUC of the slow-growth probability")["value"], "external_summary.csv", "{:.2f}")
    slow_pred = int(((et["doubling_time_hours"] > 24) & (et["predicted_hours"] > 24)).sum())
    assert slow_pred == 0, "the text says none of the slow external species is predicted slower than a day"
    put("ExtBias", np.exp(np.median(np.log(et["doubling_time_hours"] / et["predicted_hours"]))), "external_test.csv median measured / predicted", "{:.2f}")

    # ---- low-energy test
    lt = pd.read_csv(P / "lifestyle_phylogenetic_test.csv")
    lt = lt[lt["analysis"] == "all labeled species"].set_index("input")
    put("PglsLambda", lt["lambda"].median(), "lifestyle_phylogenetic_test.csv", "{:.2f}")
    put("PglsNGroups", int(lt["n_groups"].iloc[0]), "lifestyle_phylogenetic_test.csv")
    put("PglsRegFold", lt.loc["regulators_per_1000", "fold_pgls"], "lifestyle_phylogenetic_test.csv", "{:.2f}")
    put("PglsRegP", lt.loc["regulators_per_1000", "p_pgls_bh"], "lifestyle_phylogenetic_test.csv", "{:.3f}")
    energy = ["nitrate_respiration", "denitrification", "dmso_tmao_reduction", "fumarate_reductase", "extracellular_electron_transfer",
              "wood_ljungdahl", "f_type_atpase", "va_type_atpase", "na_nqr", "rnf", "na_decarboxylase", "fermentation_only",
              "electron_acceptor_classes", "sodium_pumps", "methanogenesis", "sulfate_reduction", "nitrogen_fixation",
              "ammonia_monooxygenase", "phototrophy", "rubisco"]
    present = [e for e in energy if e in lt.index]
    put("PglsEnergyMinP", lt.loc[present, "p_pgls_bh"].min(), "lifestyle_phylogenetic_test.csv energy inputs", "{:.2f}")
    checks = (P / "lifestyle_phylogenetic_checks.txt").read_text()
    mt = re.search(r"\(b\) slower than 24 hours.*?transporter density \(per SD\)\s+([\d.]+) \(([\d.]+) to ([\d.]+)\) p ([\deE.+-]+)", checks, re.S)
    put("SlowTransporterOR", float(mt.group(1)), "lifestyle_phylogenetic_checks.txt", "{:.2f}")
    mant, expo = f"{float(mt.group(4)):.1e}".split("e")
    put("SlowTransporterP", f"${mant}\\times10^{{{int(expo)}}}$", "lifestyle_phylogenetic_checks.txt")

    # ---- method constants, read from the code and configs so the text cannot drift from them
    import yaml

    def code(path, pattern, cast=float):
        text = (ROOT / path).read_text()
        return cast(re.search(pattern, text).group(1))

    feats = yaml.safe_load((ROOT / "configs" / "features.yaml").read_text())
    put("KmerK", feats["rrna16s"]["kmer_k"], "configs/features.yaml")
    put("EaKJ", feats["temperature"]["activation_energy_j_per_mol"] / 1000, "configs/features.yaml", "{:.0f}")
    put("TRefC", feats["temperature"]["reference_temp_k"] - 273.15, "configs/features.yaml", "{:.0f}")
    put("NLandmarks", code("scripts/accuracy/build_gtdb_landmark_features.py", r"N_LANDMARKS = (\d+)", int), "build_gtdb_landmark_features.py")
    put("MinContig", code("src/gem_worldmodel/features/cub_pipeline.py", r"len\(r\.seq\) >= (\d+)", int), "cub_pipeline.py")
    put("MinTrainKb", code("src/gem_worldmodel/features/cub_pipeline.py", r"MIN_TRAINING_BP = ([\d_]+)", lambda v: int(v.replace("_", "")) // 1000), "cub_pipeline.py")
    put("MinRibosomal", code("src/gem_worldmodel/features/cub_pipeline.py", r"len\(rp_seqs\) >= (\d+)", int), "cub_pipeline.py")
    put("NRibosomalFamilies", sum(1 for ln in (ROOT / "data/raw/ribosomal_pfam.hmm").read_text().splitlines() if ln.startswith("NAME ")), "data/raw/ribosomal_pfam.hmm")
    from gem_worldmodel.features import lifestyle_genes
    import inspect
    used = set(re.findall(r"PF\d{5}", inspect.getsource(lifestyle_genes.lifestyle_features))) | set(lifestyle_genes.TRANSPORTERS) | set(lifestyle_genes.REGULATORS)
    put("NLifestyleFamilies", len(used), "lifestyle_genes.lifestyle_features source")
    put("NLifestyleFeatures", len(lifestyle_genes.FEATURES), "lifestyle_genes.FEATURES")
    used_e = set(re.findall(r"PF\d{5}", inspect.getsource(lifestyle_genes.energy_features)))
    put("NEnergyFamilies", len(used_e - used), "energy_features source, families not in lifestyle_features")
    put("NEnergyFeatures", len(lifestyle_genes.ENERGY_FEATURES), "lifestyle_genes.ENERGY_FEATURES")
    put("NGRodonFeatures", len(deploy_features.GRODON_FEATURES), "deploy_features.GRODON_FEATURES")
    put("NGRodonFeaturesWord", words[len(deploy_features.GRODON_FEATURES)], "deploy_features.GRODON_FEATURES (word)")
    put("DriftSD", code("src/gem_worldmodel/features/deploy_features.py", r"limit_sd: float = ([\d.]+)"), "deploy_features.drift_report", "{:.0f}")
    put("NSeedsTable", len(re.search(r"SEEDS = \[([^\]]*)\]", (ROOT / "scripts/accuracy/evaluate_final_model.py").read_text()).group(1).split(",")), "evaluate_final_model.py SEEDS")
    put("NSeedsMag", len(re.search(r"SEEDS = \[([^\]]*)\]", (ROOT / "scripts/accuracy/validate_on_gem_mags.py").read_text()).group(1).split(",")), "validate_on_gem_mags.py SEEDS")
    put("GTDBRelease", "232", "data/raw/*_taxonomy.tsv.gz byte-identical to GTDB R232 (checked 2026-10-27)")
    put("NGrowthTable", len(__import__("pyreadr").read_r(str(ROOT / "data/raw/GrowthRates_Madin.rda"))["d"]), "gRodon2 GrowthRates_Madin.rda")
    put("NGEMTotal", sum(1 for _ in open(ROOT / "data/raw/gem_genome_metadata.tsv")) - 1, "gem_genome_metadata.tsv", "{:,}")
    put("NGEMHQ", len(pd.read_csv(P / "hq_genome_ids_all.csv")), "hq_genome_ids_all.csv", "{:,}")

    put("ScreenFoldWord", words[int(code("scripts/accuracy/screen_labels.py", r"FOLD = ([\d.]+)"))], "screen_labels.py FOLD")
    put("StudyFoldWord", words[int(code("scripts/accuracy/screen_label_sources.py", r"FOLD = ([\d.]+)"))], "screen_label_sources.py FOLD")
    put("StudyFold", int(code("scripts/accuracy/screen_label_sources.py", r"FOLD = ([\d.]+)")), "screen_label_sources.py FOLD")
    put("NBootDraws", code("scripts/accuracy/pressure_tests.py", r"DRAWS = (\d+)", int), "pressure_tests.py DRAWS", "{:,}")
    put("TraceTolPct", 100 * code("scripts/accuracy/screen_label_sources.py", r"rtol=([\d.]+)"), "screen_label_sources.py rtol", "{:.0f}")
    put("KEigen", code("scripts/accuracy/check_lifestyle_phylogenetic.py", r"K_EIGEN = (\d+)", int), "check_lifestyle_phylogenetic.py K_EIGEN")
    put("NForestTrees", code("src/gem_worldmodel/models/deployed.py", r"n_estimators=(\d+)", int), "models/deployed.py")
    put("EsmNRibosomal", code("scripts/accuracy/embed_proteins_esm.py", r"N_RIBOSOMAL, N_OTHER, MAX_LEN = (\d+)", int), "embed_proteins_esm.py")
    put("EsmNOther", code("scripts/accuracy/embed_proteins_esm.py", r"N_RIBOSOMAL, N_OTHER, MAX_LEN = \d+, (\d+)", int), "embed_proteins_esm.py")
    put("EsmPCTrees", code("scripts/accuracy/evaluate_esm_baseline.py", r"PCA\((\d+), random_state=seed\)\)\.fit\(E\[tr\]\)\n        p = Deployed", int), "evaluate_esm_baseline.py")
    put("EsmPCComb", code("scripts/accuracy/evaluate_esm_baseline.py", r"PCA\((\d+), random_state=seed\)\)\.fit\(E\[tr\]\)\n        Z =", int), "evaluate_esm_baseline.py")
    put("NTuneCandidates", code("scripts/accuracy/tune_deployed_model.py", r"N_CANDIDATES = (\d+)", int), "tune_deployed_model.py")
    put("NTuneShuffles", len(re.search(r"SPLITS = \[(.*?)\]", (ROOT / "scripts/accuracy/tune_deployed_model.py").read_text()).group(1).split(",")),
        "tune_deployed_model.py SPLITS")
    put("NTuneFolds", code("scripts/accuracy/tune_deployed_model.py", r"GroupKFold\((\d+)\)", int), "tune_deployed_model.py")
    put("NFolds", code("scripts/accuracy/evaluate_paper_headline.py", r"GroupKFold\((\d+)\)", int), "evaluate_paper_headline.py")

    # syntrophy rule, evaluated live (same as scripts/accuracy/evaluate_syntrophy_marker.py)
    syn = re.search(r"KNOWN = \[(.*?)\]", (ROOT / "scripts/accuracy/evaluate_syntrophy_marker.py").read_text(), re.S).group(1)
    known = re.findall(r'"([^"]+)"', syn)
    counts = pd.read_csv(P / "labeled_lifestyle_genes.csv").set_index("accession")
    t = df.drop_duplicates("species").join(counts, on="accession").dropna(subset=["n_genes"]).reset_index(drop=True)
    cand = lifestyle_genes.syntrophy_candidate(t) == 1
    is_known = t["species"].isin(known).to_numpy()
    put("SynKnownWord", words[int(is_known.sum())], "syntrophs with a genome (word)")
    put("SynFoundWord", words[int((cand & is_known).sum())], "syntrophs found (word)")
    put("SynFalse", int((cand & ~is_known).sum()), "syntrophy false calls")

    # ---- write
    lines = ["% Generated by paper/scripts/make_numbers.py. Do not edit by hand."]
    for name, text in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{text}\\xspace}}")
    OUT.write_text("\n".join(lines) + "\n")
    pd.DataFrame(sources).to_csv(SOURCES, index=False)
    print(f"wrote {OUT.name}: {len(macros)} numbers")


if __name__ == "__main__":
    main()
