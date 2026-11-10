#!/usr/bin/env python
"""Tables of the NeurIPS Datasets and Benchmarks version, from the same result files as the ISMB paper.

  tables/data.tex       the benchmark's labeled genomes by source
  tables/benchmark.tex  every baseline at the four held-out levels: typical error and Spearman (doubling time at the growth
                        temperature), mean over seeds, from pressure_tests.csv, paper_headline_accuracy.csv and esm_baseline.csv
  tables/released.tex   the released tools on equal genes and folds, with the paired per-genome test

Run from the repository root: .venv/bin/python paper_neurips/make_neurips_tables.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = ROOT / "paper_neurips" / "tables"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
import make_tables  # noqa: E402

LEVELS = [("species", "Same genus"), ("genus", "New genus"), ("family", "New family"), ("order", "New order")]


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text)
    print(f"wrote tables/{name}")


def data_table():
    text = (ROOT / "paper" / "tables" / "table1_data.tex").read_text()
    text = text.replace("(Supplementary Table~\\ref{S-tab:exclusions})", "(released with their evidence)")
    write("data.tex", text)


def benchmark_table():
    pt = pd.read_csv(P / "pressure_tests.csv")
    base = pt[pt["test"] == "baseline"].groupby(["method", "held_out"])[["typical error", "Spearman"]].mean()
    head = pd.read_csv(P / "paper_headline_accuracy.csv")
    head = head[head["target"] == "own temperature"].groupby(["scheme", "held_out"])[["typical error", "Spearman"]].mean()
    rows = [("Training median", base, "training median", "taxonomy-free"),
            ("Relatives' median", base, "relative median", "taxonomy only")]
    lines = []

    def cells(table, key):
        out = []
        for level, _ in LEVELS:
            v = table.loc[(key, level)]
            rho = "--" if pd.isna(v["Spearman"]) else f"{v['Spearman']:.2f}".replace("-", "$-$")
            out.append(f"{v['typical error']:.2f} & {rho}")
        return " & ".join(out)

    for name, table, key, _ in rows:
        lines.append(f"{name} & {cells(table, key)} \\\\")
    lines.append("\\midrule")
    esm_path = P / "esm_baseline.csv"
    if esm_path.exists():
        esm = pd.read_csv(esm_path).groupby(["variant", "held_out"])[["typical error", "Spearman"]].mean()
        for variant, name in (("ESM-2, linear probe", "ESM-2, ridge probe"), ("ESM-2, trees", "ESM-2, trees (64 PCs)")):
            lines.append(f"{name} & {cells(esm, variant)} \\\\")
        lines.append("\\midrule")
        lines.append(f"Features + trees, fixed$^{{a}}$ & {cells(esm, 'features')} \\\\")
        lines.append(f"Features + ESM-2, fixed$^{{a}}$ & {cells(esm, 'features + ESM-2')} \\\\")
    lines.append(f"Features + trees, nested (ours) & {cells(head, 'nested')} \\\\")
    header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in LEVELS)
    rules = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4))
    text = ("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
            "\\caption{GrowthBench results: typical fold error $E$ (lower is better) and Spearman $\\rho$ of the doubling time at the "
            "growth temperature, by the closest labeled relative of the test genomes, mean of 3 seeds on the same folds. No method uses "
            "the measured temperature. $^{a}$Deployed settings, each genome entered once, on the genomes with an embedding; the nested "
            "row chooses settings inside each split and enters each genome at four tree placements.}\\label{tab:benchmark}\n"
            "\\begin{tabular}{l" + "cc" * 4 + "}\n\\toprule\n& " + header + " \\\\\n" + rules + "\n& "
            + " & ".join(["$E$ & $\\rho$"] * 4) + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write("benchmark.tex", text)


def appendix_tables():
    # copies of ISMB tables sized for the single-column page
    abl = (ROOT / "paper" / "tables" / "table2_ablations.tex").read_text()
    abl = abl.replace("\\begin{table*}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}",
                      "\\begin{table}[ht]\\centering\\scriptsize\\setlength{\\tabcolsep}{1.5pt}").replace("\\end{table*}", "\\end{table}")
    write("ablations.tex", abl)
    exc = (ROOT / "paper" / "supplement" / "supp_exclusions.tex").read_text()
    exc = exc.replace("{p{3cm}p{1.2cm}p{6cm}p{4.4cm}}", "{p{2.4cm}p{1.0cm}p{5.2cm}p{3.5cm}}")
    write("exclusions.tex", "{\\small\n" + exc + "}\n")


def released_table():
    make_tables.table_released()
    text = (ROOT / "paper" / "tables" / "table3_released.tex").read_text()
    write("released.tex", text)


if __name__ == "__main__":
    data_table()
    benchmark_table()
    released_table()
    appendix_tables()
