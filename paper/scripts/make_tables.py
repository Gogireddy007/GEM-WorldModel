#!/usr/bin/env python
"""Writes the LaTeX tables of the main text and the result tables of the supplement from the result files.

  tables/table1_data.tex         training data by source after the audit
  tables/table2_ablations.tex    ablations against the final model (ablation_evaluation.csv) and the nested learner comparison
  tables/table3_released.tex     released gRodon and Phydon against this model, with the paired per-genome test (pressure_tests.csv)
  supplement/supp_accuracy.tex   accuracy by target, novelty and evaluation scheme (paper_headline_accuracy.csv)
  supplement/supp_released.tex   the released-tool comparison, both subsets
  supplement/supp_ablations.tex  every ablation at species, family and order level, absolute values
  supplement/supp_independent.tex species outside gRodon's growth table, predicted by a model trained on that table only
  supplement/supp_mags.tex       metagenome-assembled genomes against isolate genomes, by species
  supplement/supp_pgls.tex       every input of the low-energy test, with the three checks

Run from the repository root: .venv/bin/python paper/scripts/make_tables.py
A table whose result file is missing is skipped with a message.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
from gem_worldmodel.features import deploy_features  # noqa: E402

TIERS = [("species", "same genus"), ("genus", "new genus"), ("family", "new family"), ("order", "new order")]


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path.relative_to(ROOT / 'paper')}")


def need(*files):
    missing = [f for f in files if not (P / f).exists()]
    if missing:
        print(f"skipped: missing {', '.join(missing)}")
    return not missing


def table_data():
    df, _, _ = deploy_features.load_training_table(P, 6, with_cub=True, with_lifestyle=True)
    groups = [("gRodon's growth table", ["original"], "from the genome"),
              ("Madin et al., with traits", ["bacterial extension", "archaeal extension"], "from the compilation"),
              ("Madin et al., genome only", ["Madin genome extension"], "from the genome"),
              ("Literature strains", ["literature"], "from the genome")]
    rows = []
    for name, origins, traits in groups:
        d = df[df["origin"].isin(origins)]
        rows.append(f"{name}{'$^{a}$' if traits == 'from the compilation' else ''} & {len(d):,} & {d['species'].nunique():,} & "
                    f"{int((d['archaeon'] == 1).sum())} & {int((d['doubling_time_hours'] > 24).sum())} & {int(d['has_cub'].sum()):,} \\\\")
    rows.append("\\midrule")
    rows.append(f"Total & {len(df):,} & {df['species'].nunique():,} & {int((df['archaeon'] == 1).sum())} & "
                f"{int((df['doubling_time_hours'] > 24).sum())} & {int(df['has_cub'].sum()):,} \\\\")
    ex = pd.read_csv(deploy_features.EXCLUSIONS)
    text = ("\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{Training data after the label audit. Rows are genomes with a measured doubling time; slow means a doubling time "
            f"above 24 h; gene features are available where the genes could be predicted and annotated. The audit removed {len(ex)} values "
            "(Supplementary Table~\\ref{S-tab:exclusions}). Genome size, GC content and tRNA count are computed from the genome, except "
            "in the rows marked $^{a}$, which take them from the compilation.}\\label{tab:data}\n"
            "\\begin{tabular}{lrrrrr}\n\\toprule\n"
            "Source & Genomes & Species & Archaea & Slow & Gene features \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(ROOT / "paper" / "tables" / "table1_data.tex", text)


def change(piv, metric, name, base):
    diff = piv[metric][name] - piv[metric][base]
    better = int((diff < 0).sum()) if metric == "typical error" else int((diff > 0).sum())
    p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
    return diff.mean(), better, len(diff), p


def signed(x):
    # three decimals so that small but consistent changes are not printed as zero; LaTeX minus sign
    return "0.000" if abs(x) < 0.0005 else f"{x:+.3f}".replace("-", "$-$")


def table_ablations():
    if not need("ablation_evaluation.csv", "tuning_evaluation.csv"):
        return
    a = pd.read_csv(P / "ablation_evaluation.csv")
    labels = {"- gRodon statistics": "without gRodon statistics", "- lifestyle genes": "without lifestyle genes",
              "- all gene features": "without any gene features", "- 16S": "without 16S", "- phylogeny": "without GTDB phylogeny",
              "- added species": "without the added species", "+ energy features": "with energy-metabolism genes",
              "+ predicted temperature": "with predicted growth temperature"}
    metrics = ("typical error", "Spearman", "AUC slow")
    piv = {lv: a[a["held_out"] == lv].pivot_table(index="seed", columns="setup", values=list(metrics)) for lv in ("species", "order")}

    def cells(p, name, base):
        out = []
        for metric in metrics:
            mean, better, n, pv = change(p, metric, name, base)
            out.append(f"{signed(mean)}{'$^{*}$' if pv < 0.05 else ''} ({better}/{n})")
        return out

    base = [f"{a[(a['held_out'] == lv) & (a['setup'] == 'final model')][m].mean():.3f}" for lv in ("species", "order") for m in metrics]
    lines = ["final model & " + " & ".join(base) + " \\\\", "\\midrule"]
    for name, label in labels.items():
        lines.append(f"{label} & " + " & ".join(cells(piv["species"], name, "final model") + cells(piv["order"], name, "final model")) + " \\\\")
    t = pd.read_csv(P / "tuning_evaluation.csv")
    tp = t[t["held_out"] == "order"].pivot_table(index="seed", columns="setup", values=list(metrics))
    lines.append("earlier learner, nested$^{\\dagger}$ & & & & " + " & ".join(cells(tp, "deployed", "hist + forest tuned")) + " \\\\")
    seeds = a["seed"].nunique()
    tune_seeds = t["seed"].nunique()
    text = ("\\begin{table*}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n"
            "\\caption{What each part contributes. Each row changes one thing in the final model and gives the mean change over "
            f"{seeds} seeds, with the number of seeds in which the variant did better in brackets; $^{{*}}$ paired Wilcoxon $p<0.05$. "
            "Typical error is the fold error of the median genome (lower is better); AUC ranks genomes slower than a day. All variants are scored "
            "on the same genomes and folds, with the deployed settings and each genome entered once. The seeds reshuffle the same genomes, so $p$ "
            "measures how consistent a change is across splits, not independent replication. $^{\\dagger}$The earlier learner (gradient boosting, "
            f"200 trees of depth 3) against the tuned average, both with settings chosen inside each split ({tune_seeds} seeds).}}"
            "\\label{tab:ablations}\n"
            "\\begin{tabular}{lcccccc}\n\\toprule\n"
            "& \\multicolumn{3}{c}{Species held out} & \\multicolumn{3}{c}{Orders held out} \\\\\n"
            "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n"
            "& Typical error ($\\times$) & Spearman & AUC, slow & Typical error ($\\times$) & Spearman & AUC, slow \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    write(ROOT / "paper" / "tables" / "table2_ablations.tex", text)


def table_released():
    if not need("released_tools_comparison_nested.csv", "pressure_tests.csv"):
        return
    r = pd.read_csv(P / "released_tools_comparison_nested.csv")
    d = r[r["subset"] == "all genomes"].groupby("method")[["n", "typical error", "Spearman", "AUC slow"]].mean()
    pt = pd.read_csv(P / "pressure_tests.csv")
    pr = pt[pt["test"] == "paired"].set_index("method")
    rows = [("this model", "no", None), ("this model + measured temperature", "yes", None),
            ("gRodon (released), no temperature", "no", "gRodon (released), no temperature"),
            ("gRodon (released)", "yes", "gRodon (released)"), ("Phydon (released)", "yes", "Phydon (released)")]
    names = {"this model": "This model", "this model + measured temperature": "This model", "gRodon (released), no temperature": "gRodon",
             "gRodon (released)": "gRodon", "Phydon (released)": "Phydon"}
    def wins(ours, other):
        q = pr.loc[f"{ours} vs {other}"]
        p = q["p"]
        ptxt = "$<$0.001" if p < 0.001 else f"{p:.2f}"
        return f"{int(q['better'])}/{int(q['n'])} ({ptxt})"

    lines = []
    for m, temp, other in rows:
        v = d.loc[m]
        if other is None:
            cells = "-- & --"
        else:
            cells = wins("this model", other) + " & " + wins("this model + measured temperature", other)
        lines.append(f"{names[m]} & {temp} & {v['typical error']:.2f} & {v['Spearman']:.2f} & {v['AUC slow']:.2f} & {cells} \\\\")
    n = int(d.loc["this model", "n"])
    text = ("\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
            f"\\caption{{The released gRodon and Phydon against this model on the same {n} genomes, genes and species-held-out folds "
            f"(mean of {r['seed'].nunique()} seeds; this model built inside each split as in Algorithm~\\ref{{alg:nested}}). T: the measured "
            "growth temperature is given as an input. Error: typical fold error; $\\rho$: Spearman correlation; AUC: for a doubling time "
            "above a day. Wins: genomes, of those both methods predict, for which this model (without or with "
            "the temperature) has the smaller absolute log error, averaged over seeds, with the paired Wilcoxon $p$. Phydon placed "
            f"{int(d.loc['Phydon (released)', 'n'])} genomes.}}\\label{{tab:released}}\n"
            "\\begin{tabular}{@{}lccccc@{\\hspace{4pt}}c@{}}\n\\toprule\n"
            "& & & & & \\multicolumn{2}{c}{Wins of this model} \\\\\n\\cmidrule(lr){6-7}\n"
            "Method & T & Error & $\\rho$ & AUC & without T & with T \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(ROOT / "paper" / "tables" / "table3_released.tex", text)


def supp_pressure():
    if not need("pressure_tests.csv"):
        return
    pt = pd.read_csv(P / "pressure_tests.csv")
    base = pt[pt["test"] == "baseline"].groupby(["held_out", "method"])[["typical error", "Spearman", "AUC slow"]].mean()
    boot = pt[pt["test"] == "bootstrap"].set_index(["held_out", "statistic"])
    lines = []
    for level, name in TIERS:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{name[0].upper() + name[1:]}}}}} \\\\")
        for method, label in (("nested model", "this model, nested"), ("relative median", "median of closest relatives"),
                              ("training median", "median training label")):
            v = base.loc[(level, method)]
            ci = ""
            if method == "nested model":
                e, r = boot.loc[(level, "typical error")], boot.loc[(level, "Spearman")]
                ci = f" & {e['low']:.2f}--{e['high']:.2f}, {r['low']:.2f}--{r['high']:.2f}"
            elif method == "relative median":
                dd = boot.loc[(level, "error difference to relative median")]
                ci = f" & {dd['low']:+.2f} to {dd['high']:+.2f}".replace("-", "$-$")
            else:
                ci = " & "
            rho = "--" if pd.isna(v["Spearman"]) else f"{v['Spearman']:.2f}".replace("-", "$-$")
            lines.append(f"\\quad {label} & {v['typical error']:.2f} & {rho} & {v['AUC slow']:.2f}{ci} \\\\")
    sh = pt[pt["test"] == "shuffled labels"].groupby(["held_out", "method"])[["typical error", "Spearman", "AUC slow"]].mean()
    lines.append("\\multicolumn{5}{l}{\\textit{Labels shuffled across genomes}} \\\\")
    for level in ("species", "order"):
        for method, label in (("deployed model", "this model"), ("training median", "median training label")):
            v = sh.loc[(level, method)]
            rho = "--" if pd.isna(v["Spearman"]) else f"{v['Spearman']:.2f}".replace("-", "$-$")
            lines.append(f"\\quad {label}, {level if level == 'order' else 'species'} held out & {v['typical error']:.2f} & {rho} & {v['AUC slow']:.2f} & \\\\")
    text = ("\\begin{table}[ht]\\centering\\small\n"
            f"\\caption{{Pressure tests of the headline (doubling time at the growth temperature, mean of {pt['seed'].nunique()} seeds, the folds "
            "of the headline). Intervals: 95\\% bootstrap over species groups; for this model, of the typical error and Spearman "
            "correlation; for the median of the closest relatives, of the model's error minus the baseline's (negative: the model is better). "
            "Shuffled labels: the deployed model trained and scored on labels permuted across genomes.}\\label{tab:pressure}\n"
            "\\begin{tabular}{lcccc}\n\\toprule\n& Error ($\\times$) & Spearman & AUC, slow & 95\\% interval \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(ROOT / "paper" / "supplement" / "supp_pressure.tex", text)


def supp_ranges():
    if not need("range_evaluation.csv"):
        return
    rv = pd.read_csv(P / "range_evaluation.csv")
    cv = rv[(rv["test"] == "coverage") & (rv["target"] == "own temperature")]
    groups = [("all", "All"), ("bacteria", "Bacteria"), ("archaea", "Archaea"), ("fast in truth", "Fast in truth"), ("slow in truth", "Slow in truth")]
    lines = []
    for level, name in TIERS:
        for scheme, label in (("novelty", "novelty"), ("novelty and domain", "novelty, domain")):
            d = cv[(cv["held_out"] == level) & (cv["ranges"] == scheme)].set_index("group")["value"]
            lines.append(f"{name if scheme == 'novelty' else ''} & {label} & " + " & ".join(f"{100 * d[g]:.0f}" for g, _ in groups) + " \\\\")
    sp = rv[rv["test"] == "slow probability"].pivot_table(index="group", columns="held_out", values="value")
    stats = [("AUC", "AUC", "{:.2f}"), ("Brier", "Brier score", "{:.3f}"), ("Brier of base rate", "Brier score of the base rate", "{:.3f}"),
             ("calibration error", "calibration error", "{:.3f}"), ("top fifth, share slow", "slow among the top fifth", "{:.2f}"),
             ("base rate", "slow overall", "{:.2f}")]
    plines = [f"{label} & " + " & ".join(fmt.format(sp.loc[k, lv]) for lv, _ in TIERS) + " \\\\" for k, label, fmt in stats]
    text = ("\\begin{table}[ht]\\centering\\small\n"
            "\\caption{Coverage (\\%) of the 80\\% ranges for the doubling time at the growth temperature, cross-fitted over genera, "
            "by group of genomes, for ranges set by novelty level alone and by novelty level and domain (released). Groups by the true "
            "doubling time cannot be targeted by any range built from the inputs.}\\label{tab:ranges}\n"
            "\\begin{tabular}{llccccc}\n\\toprule\nHeld out & Ranges by & " + " & ".join(g for _, g in groups) + " \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
            "\\begin{table}[ht]\\centering\\small\n"
            "\\caption{The probability of a doubling time above 24 h (logistic model on the predicted log doubling time and the domain), "
            "cross-fitted over genera, by the closest labeled relative. Calibration error: mean absolute difference between predicted "
            "probability and observed share over fifths of the predicted probability.}\\label{tab:slowprob}\n"
            "\\begin{tabular}{lcccc}\n\\toprule\n& " + " & ".join(n for _, n in TIERS) + " \\\\\n\\midrule\n"
            + "\n".join(plines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(ROOT / "paper" / "supplement" / "supp_ranges.tex", text)


def supp_esm():
    if not need("esm_baseline.csv"):
        return
    e = pd.read_csv(P / "esm_baseline.csv")
    m = e.groupby(["variant", "held_out"])[["typical error", "Spearman", "AUC slow"]].mean()
    names = {"features": "genome features", "ESM-2, linear probe": "ESM-2, ridge probe", "ESM-2, trees": "ESM-2, trees on 64 PCs",
             "features + ESM-2": "genome features + 32 ESM-2 PCs"}
    lines = []
    for v, label in names.items():
        lines.append(f"{label} & " + " & ".join(f"{m.loc[(v, lv), 'typical error']:.2f} & {m.loc[(v, lv), 'Spearman']:.2f}" for lv, _ in TIERS) + " \\\\")
    header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{t}}}" for _, t in TIERS)
    rules = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4))
    text = ("\\begin{table}[ht]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
            f"\\caption{{A learned-representation baseline: ESM-2 (35M) protein embeddings of {int(e['n'].iloc[0])} genomes (mean of all ribosomal "
            "proteins and of 100 other proteins per genome) against the genome features, on the folds of the headline, deployed settings, "
            f"each genome once, mean of {e['seed'].nunique()} seeds. Error: typical fold error of the doubling time at the growth temperature; "
            "$\\rho$: Spearman correlation.}\\label{tab:esm}\n"
            f"\\begin{{tabular}}{{l{'cc' * 4}}}\n\\toprule\n& {header} \\\\\n{rules}\n& "
            + " & ".join(["Error & $\\rho$"] * 4) + " \\\\\n\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(ROOT / "paper" / "supplement" / "supp_esm.tex", text)


def supp_robustness():
    if not need("pressure_tests_round2.csv"):
        return
    r = pd.read_csv(P / "pressure_tests_round2.csv")
    names = {"ten splits": "Model against relatives' median", "gRodon estimate": "", "flagged studies": "Without the flagged-study labels"}
    lines = []
    for row in r.itertuples():
        test = {"ten splits": "model vs relatives' median", "flagged studies": "without flagged-study labels vs full"}.get(row.test)
        if row.test == "gRodon estimate":
            test = ("without gRodon's estimate" if row.a == "without gRodon estimate" else "gRodon refitted in each split") + " vs full"
        pg = "$<$0.001" if row.p_genomes < 0.001 else f"{row.p_genomes:.3f}"
        lines.append(f"{test} & {row.held_out} & {row.n_genomes} & {row.error_a:.3f} & {row.error_b:.3f} & {row.rho_a:.2f} & {row.rho_b:.2f} & "
                     f"{row.seeds_a_better}/{row.n_seeds} & {row.genomes_a_better} & {pg} \\\\")
    text = ("\\begin{table}[ht]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
            f"\\caption{{Robustness checks, deployed settings and each genome once, {int(r['n_seeds'].iloc[0])} seeds on the official splits. "
            "Errors are typical fold errors of the first and second method (a, b); seeds and genomes: how many favour a; $p$: paired "
            "Wilcoxon test over genomes (absolute log errors averaged over seeds). The refit replaces gRodon's released estimate, whose "
            "coefficients were fitted on data that include our species, with a regression of the log doubling time on gRodon's codon "
            "statistics fitted on the training rows of each split. The flagged-study test removes the labels of the studies flagged by the "
            "study-level check but kept, and scores both models on the remaining genomes.}\\label{tab:robust}\n"
            "\\begin{tabular}{llcccccccc}\n\\toprule\nComparison (a vs b) & Held out & Genomes & Error a & Error b & $\\rho$ a & $\\rho$ b & "
            "Seeds & Genomes & $p$ \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(ROOT / "paper" / "supplement" / "supp_robustness.tex", text)


def supp_external():
    if not need("external_test.csv"):
        return
    e = pd.read_csv(P / "external_test.csv").sort_values(["tier", "species"], key=lambda s: s.map(
        {"same genus": 0, "same family, new genus": 1, "same order, new family": 2, "new order": 3}) if s.name == "tier" else s)
    cur = pd.read_csv(ROOT / "data" / "curated" / "external_growth_rates.csv").set_index("accession")
    lines = []
    for r in e.itertuples():
        src = str(cur.loc[r.accession, "source"])
        src = src.replace("PMID:", "PMID ").replace("_", "\\_")
        temp = cur.loc[r.accession, "growth_temp_c"]
        temp = "" if pd.isna(temp) else f"{temp:g}"
        lines.append(f"\\textit{{{r.species}}} & {r.tier} & {temp} & {r.doubling_time_hours:.3g} & {r.predicted_hours:.3g} & "
                     f"{r.relatives_median_hours:.3g} & {'yes' if r.inside_range else 'no'} & {src} \\\\")
    text = ("{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{llcccccl}\n"
            f"\\caption{{External test: {len(e)} species whose doubling time comes from the species description and that are in neither the "
            "Madin compilation (by NCBI taxonomy identifier) nor our labels, with no GTDB species in training. Measured, predicted and "
            "relatives' median doubling times in hours at the growth temperature; $T$: the growth temperature stated with the value "
            "(\\textdegree C, blank where none was stated); range: whether the measured value lies inside the released 80\\% range.}"
            "\\label{tab:external}\\\\\n\\toprule\nSpecies & Closest labeled relative & $T$ & Measured & Predicted & Relatives & Range & Source \\\\\n"
            "\\midrule\n\\endfirsthead\n\\toprule\nSpecies & Closest labeled relative & $T$ & Measured & Predicted & Relatives & Range & Source \\\\\n"
            "\\midrule\n\\endhead\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}}\n")
    write(ROOT / "paper" / "supplement" / "supp_external.tex", text)


def supp_accuracy():
    if not need("paper_headline_accuracy.csv"):
        return
    a = pd.read_csv(P / "paper_headline_accuracy.csv")
    lines = []
    for target, metric, unit in (("own temperature", "typical error", "$\\times$"), ("20 C", "typical error", "$\\times$"),
                                 ("growth temperature", "mean abs error", " \\textdegree C")):
        name = {"own temperature": "Doubling time at the growth temperature", "20 C": "Doubling time at 20 \\textdegree C",
                "growth temperature": "Growth temperature"}[target]
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{name}}}}} \\\\")
        for scheme in ("nested", "fixed, coarsened", "fixed, single"):
            d = a[(a["target"] == target) & (a["scheme"] == scheme)].groupby("held_out")[[metric, "Spearman"]].mean()
            cells = [f"{d.loc[lv, metric]:.2f} & {d.loc[lv, 'Spearman']:.2f}" for lv, _ in TIERS]
            lines.append(f"\\quad {scheme} & " + " & ".join(cells) + " \\\\")
    header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{t}}}" for _, t in TIERS)
    rules = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4))
    text = ("\\begin{table}[ht]\\centering\\small\n"
            f"\\caption{{Accuracy by closest labeled relative, mean of {a['seed'].nunique()} seeds. Error is the typical fold error for doubling "
            "times and the mean absolute error in \\textdegree C for growth temperature; $\\rho$ is the Spearman correlation. "
            "Nested: settings chosen inside each split, each training genome entered once per placement rank, as deployed (the "
            "estimates quoted in the main text). Fixed, coarsened: the deployed settings, chosen on all rows, with the deployed training "
            "scheme. Fixed, single: the deployed settings and each genome entered once.}\\label{tab:accuracy}\n"
            f"\\begin{{tabular}}{{l{'cc' * 4}}}\n\\toprule\n& {header} \\\\\n{rules}\n"
            + "& " + " & ".join(["Error & $\\rho$"] * 4) + " \\\\\n\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(ROOT / "paper" / "supplement" / "supp_accuracy.tex", text)


def supp_released():
    files = [f for f in ("released_tools_comparison_nested.csv", "released_tools_comparison.csv") if (P / f).exists()]
    if not files:
        print("skipped: missing released_tools_comparison.csv")
        return
    r = pd.read_csv(P / files[0])
    order = ["this model", "this model + measured temperature", "gRodon (released), no temperature", "gRodon (released)", "Phydon (released)"]
    names = {"this model": "This model", "this model + measured temperature": "This model, given the temperature",
             "gRodon (released), no temperature": "gRodon", "gRodon (released)": "gRodon, given the temperature",
             "Phydon (released)": "Phydon, given the temperature"}
    lines = []
    for subset in ("all genomes", "Phydon found the species on its tree"):
        d = r[r["subset"] == subset].groupby("method")[["n", "typical error", "Spearman", "AUC slow"]].mean()
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{subset[0].upper() + subset[1:]}}}}} \\\\")
        for m in order:
            if m in d.index:
                v = d.loc[m]
                lines.append(f"\\quad {names[m]} & {int(v['n'])} & {v['typical error']:.2f} & {v['Spearman']:.2f} & {v['AUC slow']:.2f} \\\\")
    nested = "with our model built inside each split as deployed" if "nested" in files[0] else "with the deployed settings"
    text = ("\\begin{table}[ht]\\centering\\small\n"
            f"\\caption{{Comparison with the released gRodon and Phydon on the same genomes and folds (species held out, mean of "
            f"{r['seed'].nunique()} seeds), {nested}.}}\\label{{tab:released}}\n"
            "\\begin{tabular}{lcccc}\n\\toprule\n& Genomes & Typical error ($\\times$) & Spearman & AUC, slow \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(ROOT / "paper" / "supplement" / "supp_released.tex", text)


ABLATIONS = {"final model": "final model", "- gRodon statistics": "without gRodon statistics", "- lifestyle genes": "without lifestyle genes",
             "- all gene features": "without any gene features", "- 16S": "without 16S", "- phylogeny": "without GTDB phylogeny",
             "- added species": "without the added species", "+ energy features": "with energy-metabolism genes",
             "+ predicted temperature": "with predicted growth temperature"}


def supp_ablations():
    if not need("ablation_evaluation.csv"):
        return
    a = pd.read_csv(P / "ablation_evaluation.csv")
    levels = [lv for lv in ("species", "genus", "family", "order") if lv in set(a["held_out"])]
    lines = []
    for level in levels:
        d = a[a["held_out"] == level].groupby("setup")[["typical error", "Spearman", "AUC slow"]]
        mean, sd = d.mean(), d.std()
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{level[0].upper() + level[1:]} held out}}}} \\\\")
        for name, label in ABLATIONS.items():
            cells = [f"{mean.loc[name, m]:.3f} ({sd.loc[name, m]:.3f})" for m in ("typical error", "Spearman", "AUC slow")]
            lines.append(f"\\quad {label} & " + " & ".join(cells) + " \\\\")
        if level != levels[-1]:
            lines.append("\\addlinespace")
    text = ("\\begin{table}[ht]\\centering\\small\n"
            f"\\caption{{Every ablation at each held-out level, mean (standard deviation) over {a['seed'].nunique()} seeds, with the deployed "
            "settings and each genome entered once. The ablation table of the main text gives the paired changes at the species and order "
            "levels.}\\label{tab:ablationsfull}\n"
            "\\begin{tabular}{lccc}\n\\toprule\n& Typical error ($\\times$) & Spearman & AUC, slow \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(ROOT / "paper" / "supplement" / "supp_ablations.tex", text)


def supp_independent():
    if not need("independent_test.csv"):
        return
    i = pd.read_csv(P / "independent_test.csv")
    tiers = ["same genus", "same family, new genus", "same order, new family", "new order"]

    def cells(d):
        out = [f"{len(d[d['target'] == 'own temperature'])}"]
        for target in ("own temperature", "20 C"):
            t = d[d["target"] == target]
            err = np.exp(np.median(np.abs(t["truth"] - t["pred"])))
            rho = t[["truth", "pred"]].corr(method="spearman").iloc[0, 1] if len(t) > 2 else np.nan
            out.append(f"{err:.2f} & " + ("--" if np.isnan(rho) else f"{rho:.2f}"))
        return " & ".join(out)

    bac = i[i["archaeon"] == 0]
    lines = ["\\multicolumn{6}{l}{\\textit{Bacteria, by closest relative in gRodon's growth table}} \\\\"]
    lines += [f"\\quad {t} & {cells(bac[bac['tier'] == t])} \\\\" for t in tiers if (bac["tier"] == t).any()]
    lines.append(f"\\quad all bacteria & {cells(bac)} \\\\")
    arc = i[i["archaeon"] == 1]
    lines.append(f"Archaea (none in training) & {cells(arc)} \\\\")
    n_train = len(deploy_features.load_training_table(P, 6, with_cub=True)[0].query("origin == 'original' and has_cub"))
    text = ("\\begin{table}[ht]\\centering\\small\n"
            f"\\caption{{Species that are not in gRodon's growth table, predicted by the final model trained on that table alone ({n_train} genomes, "
            "no archaea), averaged over five seeds. Error is the typical fold error; $\\rho$ the Spearman correlation. The training set is less "
            "than half of the full one, so these errors are higher than those of the deployed model; the test shows how the model carries to "
            "species from other sources.}\\label{tab:independent}\n"
            "\\begin{tabular}{lccccc}\n\\toprule\n& & \\multicolumn{2}{c}{At the growth temperature} & \\multicolumn{2}{c}{At 20 \\textdegree C} \\\\\n"
            "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}\n& Species & Error & $\\rho$ & Error & $\\rho$ \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(ROOT / "paper" / "supplement" / "supp_independent.tex", text)


def supp_mags():
    if not need("mag_validation.csv"):
        return
    m = pd.read_csv(P / "mag_validation.csv")
    m = m[(m["target"] == "own temperature") & (m["variant"] == "without 16S")].sort_values("sp")
    rows = [f"\\textit{{{tex_text(r.sp.removeprefix('s__'))}}} & {int(r.n_mags)} & {np.exp(r.truth):.2f} & {np.exp(r.isolate):.2f} & {np.exp(r.mag):.2f} \\\\"
            for r in m.itertuples()]
    head = "Species & MAGs & Measured (h) & Isolate (h) & MAGs (h) \\\\"
    text = ("\\begin{longtable}{p{6.2cm}cccc}\n"
            f"\\caption{{Doubling time at the growth temperature predicted from metagenome-assembled genomes (MAGs, median over the species' "
            f"MAGs) and from the isolate genome, with the species held out of training, for the {len(m)} species with a measured value and at "
            f"least one GEM genome; {int(m['n_mags'].sum())} MAGs in all. Both predictions use the model without 16S, which every MAG can be "
            "given.}\\label{tab:mags}\\\\\n\\toprule\n" + head + "\n\\midrule\n\\endfirsthead\n"
            "\\caption[]{(continued)}\\\\\n\\toprule\n" + head + "\n\\midrule\n\\endhead\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n")
    write(ROOT / "paper" / "supplement" / "supp_mags.tex", text)


def supp_pgls():
    if not need("lifestyle_phylogenetic_test.csv"):
        return
    lt = pd.read_csv(P / "lifestyle_phylogenetic_test.csv")
    main_ = lt[lt["analysis"] == "all labeled species"].set_index("input")
    checks = {a: lt[lt["analysis"] == a].set_index("input")["p_pgls_bh"] for a in
              ("lambda fixed at 1 (Brownian motion)", "only species placed at their own GTDB species", "without the literature strains")}

    def p(x):
        return "--" if pd.isna(x) else ("$<$0.001" if x < 0.001 else f"{x:.3f}")

    rows = []
    for inp, r in main_.iterrows():
        within = "--" if pd.isna(r["families_with_both"]) or r["families_with_both"] == 0 else f"{int(r['families_carriers_slower'])}/{int(r['families_with_both'])}"
        rows.append(f"{tex_text(r['label'])} & {int(r['n_with']) if r['n_with'] == r['n_with'] else '--'} & {r['fold_across_species']:.2f} & "
                    f"{r['fold_pgls']:.2f} ({r['ci_low_pgls']:.2f}--{r['ci_high_pgls']:.2f}) & {p(r['p_pgls_bh'])} & {within} & "
                    + " & ".join(p(c.get(inp)) for c in checks.values()) + " \\\\")
    head = ("Input & With & Across & PGLS (95\\% CI) & $p_{\\mathrm{adj}}$ & Families & $\\lambda=1$ & Own sp. & No lit. \\\\")
    text = ("{\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{p{4.6cm}rrlrrrrr}\n"
            f"\\caption{{The low-energy test, every input, {int(main_['n_groups'].iloc[0])} species groups. Effects are fold changes in doubling time for "
            "carrying the gene, per standard deviation of a count or density, or per doubling of genome size, with growth temperature and "
            "domain as covariates. With: groups carrying the gene (presence inputs). Across: without control for relatedness. PGLS: "
            "phylogenetic generalized least squares with Pagel's $\\lambda$ estimated; $p_{\\mathrm{adj}}$ is Benjamini-Hochberg adjusted "
            "over the inputs. Families: of the families with both carriers and non-carriers, those in which the median carrier is slower. "
            "The last three columns give the adjusted PGLS $p$ with $\\lambda$ fixed at 1, with only the groups placed at their own GTDB "
            "species, and without the eight literature strains.}\\label{tab:pgls}\\\\\n\\toprule\n" + head + "\n\\midrule\n\\endfirsthead\n"
            "\\caption[]{(continued)}\\\\\n\\toprule\n" + head + "\n\\midrule\n\\endhead\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}}\n")
    write(ROOT / "paper" / "supplement" / "supp_pgls.tex", text)


def tex_text(text):
    return str(text).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("Na+", "Na$^+$")


def main():
    table_data()
    table_ablations()
    table_released()
    supp_pressure()
    supp_ranges()
    supp_esm()
    supp_robustness()
    supp_external()
    supp_accuracy()
    supp_released()
    supp_ablations()
    supp_independent()
    supp_mags()
    supp_pgls()


if __name__ == "__main__":
    main()
