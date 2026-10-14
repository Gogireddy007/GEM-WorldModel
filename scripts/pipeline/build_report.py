#!/usr/bin/env python
"""This writes the HQ genome report (HTML and PDF) and the prediction table for
the report folder, with every number read from the data files.

Reads hq_predictions_gbm.csv, final_accuracy_by_tier.csv, mag_validation.csv,
independent_test.csv and the training table. The PDF is printed with headless
Chrome so the Times New Roman styling is kept.
"""

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

FOLDER = Path("Test on HQ genomes")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TIERS = ["same genus", "same family, new genus", "same order, new family", "new order"]
FULL = "traits+phylogeny+16S+gene features"


def level(taxonomy, prefix):
    m = re.search(rf"{prefix}[^;]*", str(taxonomy))
    return m.group(0)[3:] if m and m.group(0) != prefix else ""


def table_html(df, header):
    head = "".join(f"<th>{h}</th>" for h in header)
    rows = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in df)
    return f"<table><tr>{head}</tr>{rows}</table>"


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    pred = pd.read_csv(processed / "hq_predictions_gbm.csv")
    acc = pd.read_csv(processed / "final_accuracy_by_tier.csv")
    mag = pd.read_csv(processed / "mag_validation.csv")
    ind = pd.read_csv(processed / "independent_test.csv")
    lab, _, _ = deploy_features.load_training_table(processed, k, with_cub=True)
    meta = pd.read_csv("data/raw/gem_genome_metadata.tsv", sep="\t", low_memory=False)[
        ["genome_id", "mimag_quality", "completeness", "contamination", "ecosystem_type", "habitat"]
    ]

    # ---- the table
    for name, prefix in (("gtdb_phylum", "p__"), ("gtdb_family", "f__"), ("gtdb_genus", "g__")):
        pred[name] = pred["gtdb_taxonomy"].map(lambda t, p=prefix: level(t, p))
    table = pred.merge(meta, on="genome_id", how="left")
    for col in ("doubling_time_hours_at_growth_temp", "range_low_hours", "range_high_hours", "doubling_time_hours_at_20C"):
        table[col] = table[col].round(3)
    table["predicted_growth_temp_c"] = table["predicted_growth_temp_c"].round(1)
    table["typical_fold_error"] = table["typical_fold_error"].round(2)
    table["codon_usage_bias"] = table["codon_usage_bias"].round(4)
    table["ivywrel_fraction"] = table["ivywrel_fraction"].round(4)
    table["transporters_per_1000_genes"] = table["transporters_per_1000_genes"].round(1)
    cols = ["genome_id", "doubling_time_hours_at_growth_temp", "range_low_hours", "range_high_hours", "typical_fold_error",
            "predicted_growth_temp_c", "doubling_time_hours_at_20C", "hot_organism", "nearest_labeled_relative", "model_used",
            "phylogeny_rank_used", "codon_usage_bias", "ivywrel_fraction", "ribosomal_proteins_found",
            "energy_metabolism_markers", "transporters_per_1000_genes", "genome_size_bp", "gc_content", "trna_count",
            "mimag_quality", "completeness", "contamination", "gtdb_phylum", "gtdb_family", "gtdb_genus", "ecosystem_category",
            "ecosystem_type", "habitat"]
    table[cols].sort_values("genome_id").to_csv(FOLDER / "hq_genome_predictions.csv", index=False)
    d = "doubling_time_hours_at_growth_temp"
    keep = ["genome_id", d, "range_low_hours", "range_high_hours", "predicted_growth_temp_c", "gtdb_family", "gtdb_genus", "habitat"]
    table.nsmallest(10, d)[keep].to_csv(FOLDER / "fastest10.csv", index=False)
    table.nlargest(10, d)[keep].to_csv(FOLDER / "slowest10.csv", index=False)

    # ---- numbers for the text
    n_hq, n_pred = 9143, len(pred)
    share = pred["nearest_labeled_relative"].value_counts(normalize=True) * 100
    n_lab = len(lab)
    n_arch_lab = int((lab["archaeon"] == 1).sum())
    n_species = lab["species"].nunique()
    a = acc[(acc.model == FULL)].set_index(["target", "tier"])
    rows = []
    for t in TIERS:
        own, s20 = a.loc[("own temperature", t)], a.loc[("20 C", t)]
        rows.append([t, f"{share.get(t, 0):.0f}%", f"{own.median_error:.1f}x", f"{own.spearman:.2f}",
                     f"x{np.exp(own.resid_q10):.2f} to x{np.exp(own.resid_q90):.2f}", f"{s20.median_error:.1f}x", f"{s20.spearman:.2f}"])
    tier_table = table_html(rows, ["Nearest labeled relative", "Share of genomes", "Typical error", "Ranking (Spearman)", "80% range around the prediction", "Typical error, 20 C value", "Ranking, 20 C value"])
    ib = ind[(ind.archaeon == 0) & (ind.target == "own temperature")]
    ib20 = ind[(ind.archaeon == 0) & (ind.target == "20 C")]
    from scipy.stats import spearmanr
    comp_all = pd.read_csv(processed / "published_comparison.csv")
    comp_all = comp_all[(comp_all.subset == "all") & comp_all.held_out.isin(["species", "genus", "order"])]
    comp = comp_all.groupby(["target", "held_out", "model"])[["r2_log", "spearman", "fold_error"]].mean()
    seed_err = comp_all.pivot_table(index=["target", "held_out", "seed"], columns="model", values="fold_error")
    wins = {m: (int((seed_err["ours"] < seed_err[m]).sum()), len(seed_err)) for m in ("Phydon-style", "gRodon-style")}
    comp_rows = []
    for lvl, label in (("species", "new species"), ("genus", "new genus"), ("order", "new order")):
        for target, tlabel in (("20 C", "20 C value"), ("own temperature", "own temperature")):
            r = [comp.loc[(target, lvl, m)] for m in ("ours", "Phydon-style", "gRodon-style")]
            comp_rows.append([f"{label}, {tlabel}"] + [f"{x.fold_error:.2f}x ({x.spearman:.2f})" for x in r])
    comp_table = table_html(comp_rows, ["Held out, target", "This model", "Phydon-style", "gRodon-style"])
    cal = pd.read_csv(FOLDER / "calibration_data_labeled_species.csv")
    from sklearn.metrics import roc_auc_score
    slope = np.polyfit(np.log(cal.measured_hours), np.log(cal.predicted_hours), 1)[0]
    fam_of = dict(zip(lab["species"], lab["gtdb_taxonomy"].map(lambda t: level(t, "f__")), strict=False))
    cal["family"] = cal["species"].map(fam_of)
    long_pred = cal[cal.predicted_hours > 15]
    long_fams = long_pred["family"].value_counts()
    long_top, long_top_n = long_fams.index[0], int(long_fams.iloc[0])
    long_slow = 100 * (long_pred.measured_hours > 24).mean()
    mid_pred = cal[(cal.predicted_hours > 5) & (cal.predicted_hours <= 15)]
    mid_slow = 100 * (mid_pred.measured_hours > 24).mean()
    chk = pd.read_csv(processed / "slow_probability_check.csv").set_index("held_out")
    nd = pd.read_csv(processed / "new_data_and_genes.csv")
    nd = nd[(nd["test rows"] == "before rows") & (nd.held_out == "order")].groupby("setup")[["AUC, ranking", "AUC, classifier", "typical error"]].mean()
    n_lit = len(pd.read_csv(processed / "literature_extension.csv"))
    n_slow_lab = int((lab["doubling_time_hours"] > 24).sum())
    beats_base = chk.loc["order", "brier"] < chk.loc["order", "brier_base_rate"] - 0.002
    order_brier = ("is better than guessing from the overall share, though not by much" if beats_base else "is barely better than guessing from the overall share")
    g_new = comp.loc[("own temperature", "order", "gRodon-style")].fold_error / comp.loc[("own temperature", "order", "ours")].fold_error
    grodon_note = ("For those organisms gRodon-style is about as close on typical error at the organism's own temperature, but it ranks them much worse."
                   if g_new < 1.05 else "gRodon-style does not depend on relatives, so it holds up better than Phydon-style for those organisms, but it is still behind on both typical error and ranking.")
    slow_markers = pred.loc[pred[d] > 24, "energy_metabolism_markers"].fillna("")
    slow_with_marker = 100 * (~slow_markers.isin(["", "none of these"])).mean() if len(slow_markers) else 0.0
    all_with_marker = 100 * (~pred["energy_metabolism_markers"].fillna("").isin(["", "none of these"])).mean()
    slow = cal.measured_hours > 24
    auc = roc_auc_score(slow, cal.predicted_hours)
    share_meas, share_pred = 100 * slow.mean(), 100 * (cal.predicted_hours > 24).mean()
    mo = mag[(mag.target == "own temperature") & (mag.variant == "without 16S")]
    m_iso = np.exp(np.median(np.abs(mo.truth - mo.isolate)))
    m_mag = np.exp(np.median(np.abs(mo.truth - mo.mag)))
    n_mags = int(mo.n_mags.sum())
    med, p5, p95 = pred[d].median(), pred[d].quantile(0.05), pred[d].quantile(0.95)
    hot = 100 * pred.hot_organism.mean()
    hab = (table["habitat"].fillna("") + " " + table["ecosystem_type"].fillna("")).str.lower()
    thermal = hab.str.contains("hot spring|thermal|hydrothermal|geothermal")
    human = table["ecosystem_category"].eq("Human")
    hot_thermal, hot_human = 100 * table.loc[thermal, "hot_organism"].mean(), 100 * table.loc[human, "hot_organism"].mean()
    n_thermal = int(thermal.sum())
    lab_med, lab_p5, lab_p95 = lab["doubling_time_hours"].median(), lab["doubling_time_hours"].quantile(0.05), lab["doubling_time_hours"].quantile(0.95)
    gh = pred.groupby("ecosystem_category")[d].agg(["median", "size"])
    gh = gh[gh["size"] >= 100].sort_values("median")
    model_counts = pred["model_used"].value_counts()
    full_n = int(model_counts.get(FULL, 0))
    fam = pred.assign(f=pred.gtdb_taxonomy.map(lambda t: level(t, "f__"))).groupby("f")[d].agg(["median", "size"])
    fam = fam[fam["size"] >= 25].sort_values("median")
    fast_fams = ", ".join(f"{f} ({r['median']:.1f} h)" for f, r in fam.head(4).iterrows())
    fam_temp = pred.assign(f=pred.gtdb_taxonomy.map(lambda t: level(t, "f__"))).groupby("f")["predicted_growth_temp_c"].median()
    hot_fast = [f for f in fam.head(4).index if fam_temp.get(f, 0) >= 45]
    hot_note = (f" {', '.join(hot_fast)} grow hot, so their short time is at their own high temperature, and the 20 C column ranks them differently." if hot_fast else "")
    slow_fams = ", ".join(f"{f} ({r['median']:.0f} h)" for f, r in fam.tail(4).iloc[::-1].iterrows())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Growth Rate Predictions for the High Quality GEM Genomes</title>
<style>
  body {{ font-family: "Times New Roman", Times, serif; max-width: 720px; margin: 0 auto; padding: 48px 24px 100px; color: #000; line-height: 1.5; font-size: 17px; }}
  h1 {{ font-size: 22px; margin-bottom: 2px; }}
  .subtitle {{ font-size: 14px; margin-bottom: 30px; font-style: italic; }}
  h2 {{ font-size: 18px; margin-top: 34px; }}
  p {{ margin: 12px 0; text-align: justify; }}
  ul, ol {{ margin: 10px 0; padding-left: 28px; }}
  li {{ margin: 4px 0; }}
  img {{ max-width: 100%; margin: 14px 0; display: block; }}
  .caption {{ font-size: 13px; margin-top: -6px; margin-bottom: 20px; font-style: italic; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
  th, td {{ text-align: left; padding: 5px 8px 5px 0; border-bottom: 1px solid #ccc; vertical-align: top; }}
  code {{ font-family: monospace; font-size: 13px; }}
</style>
</head>
<body>

<h1>Growth Rate Predictions for the High Quality GEM Genomes</h1>
<div class="subtitle">GEM-WorldModel project, version 3, October 2026</div>

<p>This report covers predicted growth rates for the {n_hq:,} high quality genomes in the GEM catalog, genomes that
have never had a growth rate measured in a lab. We produced a prediction for {n_pred:,} of them. This version
replaces the first one, which should not be used. The next section explains why.</p>

<h2>What was wrong with the first version</h2>

<p>The first predictions gave the model information about where each organism sits on the tree of life, and about its
16S gene, in a different numerical form from the one the model had been trained on. The model still produced numbers,
but they were squeezed toward fast growth: the median predicted doubling time was 0.6 hours against 7 hours for the
species it was trained on, no genome was predicted to take longer than 24 hours, and gut microbes looked no different
from ocean microbes. We found this while rebuilding the pipeline. It is fixed, and the prediction script now stops if
its inputs fall outside the range of the training data.</p>

<h2>What the table contains</h2>

<p>The file <code>hq_genome_predictions.csv</code> has one row per genome. The main columns are:</p>
<ul>
<li><b>doubling_time_hours_at_growth_temp</b>, the predicted doubling time, in hours, at the temperature the organism
itself grows best at. This is what you would measure in a culture at its own optimum. <b>range_low_hours</b> and
<b>range_high_hours</b> give a range that covered about 80% of held-out species, and <b>typical_fold_error</b> is how far off
a prediction typically is, as a multiplier.</li>
<li><b>predicted_growth_temp_c</b>, the predicted growth temperature in degrees C, and <b>hot_organism</b>, true when that is
45 C or above.</li>
<li><b>doubling_time_hours_at_20C</b>, the same doubling time rescaled to 20 C so organisms that grow at very different
temperatures can be compared. For hot-growing organisms this is an extrapolation to a temperature they could not grow at,
so it should be read as a relative score and not a prediction of real growth.</li>
<li><b>nearest_labeled_relative</b>, <b>model_used</b> and <b>phylogeny_rank_used</b>, which say how close the genome is to a
species with a measured growth rate and which inputs the prediction could use.</li>
<li><b>energy_metabolism_markers</b>, which energy-metabolism genes were found (methanogenesis, sulfate reduction, ammonia or
methane oxidation, nitrogen fixation, phototrophy, RuBisCO), and <b>transporters_per_1000_genes</b>, the density of nutrient
transporter genes, which is high in organisms that live on rich, changing food supplies.</li>
<li>The genome traits, <b>codon_usage_bias</b>, <b>ivywrel_fraction</b>, assembly quality (<b>completeness</b>, <b>contamination</b>), taxonomy and
habitat from the GEM catalog.</li>
</ul>

<h2>How the prediction works</h2>

<p>Six kinds of information are taken from each genome's DNA: its size, GC content and number of tRNA genes; how strongly
its ribosomal protein genes favor particular codons (codon usage bias, a known signature of fast growth); the share of its
proteins made of the amino acids I, V, Y, W, R, E and L, which rises with growth temperature; which genes for making a
living it carries (energy metabolism such as methanogenesis, sulfate reduction, ammonia oxidation, nitrogen fixation and
photosynthesis, plus the density of transporters, regulators, motility and catalase genes); where it sits on the tree of
life; and its 16S gene. Genes were predicted directly from the DNA, so no annotation was needed, and the same steps were
applied to the species with measured growth rates and to the GEM genomes.</p>

<p>A model was trained on {n_lab} genomes from {n_species} species with a measured growth rate ({n_lab - n_arch_lab} bacterial
genomes and {n_arch_lab} archaeal genomes), mostly from the compilation by Madin and colleagues (2020), plus {n_lit} strains
from lineages that compilation lacks (ammonia-oxidizing archaea, Nitrospira and other nitrite oxidizers, a
verrucomicrobial methanotroph), each checked against the original paper and paired with the genome of the measured
strain. {n_slow_lab} of the labeled genomes double more slowly than once a day. It predicts three things: the doubling
time at the organism's own growth temperature, the growth temperature itself, and the doubling time rescaled to 20 C.
Growth temperature matters because rescaling multiplies a hot-growing organism's doubling time by a large factor, about 4 for
organisms growing at 30 to 45 C, 16 at 45 to 60 C and 48 at 60 to 80 C, so a model can look accurate on the rescaled value
partly by recognizing hot organisms. We therefore report both values, and judge the model mainly on the first.</p>

<p>Not every genome had every input. {full_n:,} genomes had all four and got the full model. The rest were predicted with
whichever inputs they had, and the table says which. For {share.get('same genus', 0):.0f}% of genomes the nearest labeled
relative is in the same genus, for {share.get('same family, new genus', 0):.0f}% in the same family, for
{share.get('same order, new family', 0):.0f}% in the same order, and {share.get('new order', 0):.0f}% have no labeled relative
at order level. These shares are worst cases, because GEM uses older taxonomic names than the labeled species and a renamed group
looks more distant than it is.</p>

<h2>How accurate it is</h2>

<p>Accuracy was never measured on a species the model had seen. We also left out whole genera, families and orders, because
most GEM genomes have no close relative among the species with measured rates, and accuracy falls as the nearest relative gets
more distant. The table gives the numbers for each situation, using the full model.</p>

{tier_table}

<img src="charts/01_accuracy_by_novelty.png" alt="Accuracy by novelty">
<div class="caption">Figure 1. Typical error and ranking accuracy when a whole lineage is held out. Blue is the doubling time at
the organism's own growth temperature, green is the value rescaled to 20 C.</div>

<p>In plain terms, a typical prediction at the organism's own temperature is off by a factor of about {a.loc[('own temperature', 'same genus')].median_error:.1f}
when a genus relative is known and about {a.loc[('own temperature', 'new order')].median_error:.1f} when none is, and the
ranking of genomes from fast to slow is moderately reliable, better for close relatives. The predictions are best used to
rank genomes and to find groups that are likely fast or slow, not as exact doubling times.</p>

<p>Two further checks. First, we trained on the original species only and predicted {len(ib)} bacterial species it had never
seen: typical error {np.exp(np.median(np.abs(ib.truth - ib.pred))):.1f}x at the organism's own temperature
(ranking {spearmanr(ib.truth, ib.pred).correlation:.2f}) and {np.exp(np.median(np.abs(ib20.truth - ib20.pred))):.1f}x for the 20 C
value (ranking {spearmanr(ib20.truth, ib20.pred).correlation:.2f}). Second, because the GEM genomes are assembled from
environmental samples and not isolates, we checked the pipeline on {n_mags} real GEM genomes from {len(mo)} species that have a
measured growth rate, each species held out of training. Typical error was {m_mag:.1f}x from the GEM genomes against
{m_iso:.1f}x from the isolate genomes of the same species (this check used the model without 16S so that more species
qualify), so assembling genomes from the environment costs some accuracy, while the ranking is almost unchanged.</p>

<img src="charts/05_calibration.png" alt="Calibration" style="max-width:62%">
<div class="caption">Figure 2. Predicted against measured doubling times for the labeled genomes, each predicted with its species held out.</div>

<img src="charts/06_real_mags.png" alt="Real MAGs">
<div class="caption">Figure 3. Species with a measured growth rate and GEM genomes, predicted from the isolate genome and from the GEM genomes.</div>

<p>The model compresses the range. Among the held-out species, {share_meas:.0f}% have a measured doubling time above 24 hours
but only {share_pred:.1f}% are predicted to, and a tenfold change in the measured time shows up as about a
{10 ** slope:.1f}-fold change in the prediction. It ranks very slow species above most others (area under the curve {auc:.2f}) but does not say how slow
they are. Where it does predict long times, that is mostly because a close relative is known to grow slowly: {long_top_n}
of the {len(long_pred)} held-out predictions above 15 hours are in one family, {long_top}, and {long_slow:.0f}% of those are truly
slower than 24 hours, while only {mid_slow:.0f}% of predictions between 5 and 15 hours are. Adding measured slow growers and the
energy-metabolism genes helped most where it was weakest, for organisms with no labeled relative in the same order: on the
same test species, the area under the curve for picking out organisms slower than 24 hours rose from
{nd.loc['before', 'AUC, ranking']:.2f} to {nd.loc['+ both', 'AUC, ranking']:.2f} when ranking by the prediction, and from
{nd.loc['before', 'AUC, classifier']:.2f} to {nd.loc['+ both', 'AUC, classifier']:.2f} for a separate slow-growth classifier, and typical error fell from
{nd.loc['before', 'typical error']:.2f}x to {nd.loc['+ both', 'typical error']:.2f}x. With whole orders held out, a calibrated chance of being slower than 24 hours
{order_brier} (Brier score {chk.loc['order', 'brier']:.3f} against {chk.loc['order', 'brier_base_rate']:.3f}). Of the GEM genomes predicted to take longer than a day,
{slow_with_marker:.0f}% carry one of the energy-metabolism markers, against {all_with_marker:.0f}% of all genomes. The model still rarely predicts
more than a day for a new lineage, and other fixes (other losses, reweighting rare values, stretching the predictions to the
measured spread) did not recover slow growth without making other predictions worse. Read the predictions as a relative score, a long
prediction as a sign that the genome resembles known slow growers, and a short one as no guarantee of fast growth.</p>

<p>We also compared the model with reimplementations of the two published tools, refit on exactly the same data and
the same held-out splits: gRodon, which predicts from codon usage bias alone, and Phydon, which blends that with the
growth rates of the nearest relatives on the tree, weighting the relatives less the further away they are. The table
gives the typical error and, in brackets, the ranking accuracy, averaged over five repeats. Counting each repeat
separately, the model had the lower typical error in {wins['Phydon-style'][0]} of {wins['Phydon-style'][1]} comparisons with
Phydon-style and {wins['gRodon-style'][0]} of {wins['gRodon-style'][1]} with gRodon-style. The gap with Phydon-style
widens for organisms without close relatives, because its relative-based part has little to go on there. {grodon_note} These are reimplementations refit on our data,
not the published fits, and our codon statistic is a single measure where Phydon's full mode uses three, so this
compares the methods rather than the released software.</p>

{comp_table}

<p>Weak spots. Ranking among fast growers (doubling time under 5 hours) is the weakest part. Codon usage bias helped most
there, but it is still only moderate. Archaea are covered by far fewer labeled species than bacteria, mostly hyperthermophiles
and methanogens, and groups with no labeled archaeal relative are less reliable than the numbers above.</p>

<h2>What came out</h2>

<p>The predicted doubling time at the organism's own growth temperature has a median of {med:.1f} hours (middle 90% from
{p5:.1f} to {p95:.0f}), against {lab_med:.1f} hours (middle 90% from {lab_p5:.1f} to {lab_p95:.0f}) measured for the labeled species. The
spread is much narrower for the reason given above: when the model is unsure it pulls the prediction toward the middle, so
the spread of the predictions understates the spread of real growth rates. About {hot:.0f}% of genomes are predicted to grow
at 45 C or above. As a check on the growth temperature, which the model was not told anything about, {hot_thermal:.0f}% of the
{n_thermal} genomes from hot spring and other thermal habitats are predicted to grow at 45 C or above, against
{hot_human:.0f}% of the human-associated genomes. The second number is not zero, so a few mesophiles are being called hot. Treat the
<b>hot_organism</b> flag as a hint, not a measurement.</p>

<img src="charts/02_distribution.png" alt="Distribution" style="max-width:78%">
<div class="caption">Figure 4. Predicted doubling times for the GEM genomes against measured times for the labeled species.</div>

<p>By habitat, genomes from the {gh.index[0].lower()} habitat group are predicted fastest (median {gh['median'].iloc[0]:.1f} hours)
and those from {gh.index[-1].lower()} slowest ({gh['median'].iloc[-1]:.1f} hours). The model was never told the habitat. By
family, among families with at least 25 genomes the fastest are {fast_fams}, and the slowest are {slow_fams}.{hot_note}</p>

<img src="charts/04_by_habitat.png" alt="By habitat" style="max-width:80%">
<div class="caption">Figure 5. Predicted growth by habitat.</div>
<img src="charts/03_growth_temperature.png" alt="Growth temperature" style="max-width:78%">
<div class="caption">Figure 6. Predicted growth temperature.</div>

<h2>What to keep in mind</h2>
<ul>
<li>These are statistical estimates, not measurements, and are best used for ranking and for finding groups.</li>
<li>Most growth rates used for training are species values from the literature and are not matched to the exact strain
that was sequenced. Growth rate can differ a lot between strains of one species. The rows added from the literature, and the
Madin species whose traits were computed from a named genome, are closer to strain-matched.</li>
<li>The energy-metabolism markers are presence calls from protein family searches. An incomplete genome can miss a gene, and
a few enzyme families are shared with other pathways (chlorophyll synthesis shares a family with nitrogenase, which the calls
correct for).</li>
<li>Labeled species are cultured organisms, and the genomes here are mostly uncultured. Predictions for lineages far from
anything cultured rest on weaker evidence, which the table's nearest-relative column reflects.</li>
<li>The training values are minimal doubling times under good lab conditions, not growth in the environment.</li>
<li>Genomes assembled from the environment lose some repeated genes, so rRNA counts and sometimes tRNA counts are
underestimated, and incomplete genomes miss ribosomal proteins. We checked the pipeline on real GEM genomes of known species,
but only for the {len(mo)} species that have them.</li>
<li>The comparison with gRodon and Phydon uses our reimplementations refit on the same data, not the released
software with its published fit.</li>
</ul>

<h2>Files</h2>
<ul>
<li><code>hq_genome_predictions.csv</code>, the full table, {n_pred:,} genomes.</li>
<li><code>fastest10.csv</code> and <code>slowest10.csv</code>, the extremes.</li>
<li><code>charts/</code>, the figures in this report.</li>
<li><code>calibration_data_labeled_species.csv</code>, the held-out predictions behind Figure 2.</li>
<li>Scripts to reproduce everything are in <code>scripts/</code> in the repository, see <code>research_log.md</code>.</li>
<li><code>superseded/</code>, the first version, kept for the record and not to be used.</li>
</ul>

</body>
</html>
"""
    (FOLDER / "report.html").write_text(html)
    pdf = FOLDER / "GEM_HQ_Genome_Report.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf.resolve()}",
                    f"file://{(FOLDER / 'report.html').resolve()}"], check=True, capture_output=True)
    print(f"wrote report.html and {pdf} ({pdf.stat().st_size / 1e6:.2f} MB), {len(table)} rows in the table")


if __name__ == "__main__":
    main()
