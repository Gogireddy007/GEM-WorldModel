#!/usr/bin/env python
"""Writes LaTeX tables for the supplement straight from the code and the environment:

  supp_pfam.tex      every Pfam family the model uses for lifestyle and energy inputs, with its role
  supp_software.tex  versions of the software used, read from the lock files
  supp_exclusions.tex the excluded labels with the evidence (data/curated/label_exclusions.csv)
  supp_literature.tex the doubling times added from the literature (data/curated/literature_growth_rates.csv)
  supp_columns.tex    the columns of the released GEM prediction table (scripts/pipeline/export_hq_table.py)

Run from the repository root: .venv/bin/python paper/scripts/make_supp_tables.py
"""

import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "supplement"
sys.path.insert(0, str(ROOT / "src"))
from gem_worldmodel.features import lifestyle_genes as lg  # noqa: E402


def tex(text):
    return (str(text).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("$", "\\$"))


def pfam_table():
    lifestyle = set(re.findall(r"PF\d{5}", inspect.getsource(lg.lifestyle_features))) | set(lg.TRANSPORTERS) | set(lg.REGULATORS)
    energy = set(re.findall(r"PF\d{5}", inspect.getsource(lg.energy_features)))
    names = {**lg.FAMILIES, **lg.EVALUE_FAMILIES}
    rows = []
    for acc in sorted(lifestyle | energy):
        use = []
        if acc in lifestyle:
            use.append("model input")
        if acc in energy:
            use.append("energy test")
        cutoff = f"E $<$ {lg.EVALUE:g}" if acc in lg.EVALUE_FAMILIES else "gathering"
        rows.append(f"{acc} & {tex(names.get(acc, ''))} & {cutoff} & {', '.join(use)} \\\\")
    body = "\n".join(rows)
    return ("\\begin{longtable}{p{1.6cm}p{8.2cm}p{1.8cm}p{3cm}}\n"
            "\\caption{Pfam families behind the lifestyle inputs of the model and the energy inputs of the low-energy test, with the "
            "search cutoff. Accessions are written without version.}\\label{tab:pfam}\\\\\n\\toprule\n"
            "Pfam & What it marks & Cutoff & Used in \\\\\n\\midrule\n\\endfirsthead\n"
            "\\caption[]{(continued)}\\\\\n\\toprule\nPfam & What it marks & Cutoff & Used in \\\\\n\\midrule\n\\endhead\n" + body + "\n\\bottomrule\n\\end{longtable}\n")


def software_table():
    """Versions from the lock files (requirements-lock.txt, scripts/r/R_packages.lock.tsv, tools-lock.tsv), so the table
    states what the results were computed with whatever machine builds the PDF."""
    pins = dict(line.strip().lower().split("==") for line in (ROOT / "requirements-lock.txt").read_text().splitlines()
                if "==" in line and not line.startswith("#"))
    py = {p: pins[p] for p in ("scikit-learn", "numpy", "pandas", "scipy", "statsmodels", "pyrodigal", "dendropy", "biopython")}
    tools = pd.read_csv(ROOT / "tools-lock.tsv", sep="\t").set_index("tool")["version"].astype(str).to_dict()
    lock = pd.read_csv(ROOT / "scripts" / "r" / "R_packages.lock.tsv", sep="\t").set_index("package")["version"].astype(str)
    r = {p: lock[p].replace("-", ".") for p in ("gRodon", "Phydon", "picante", "castor", "ape", "Biostrings", "coRdon", "R")}
    rows = [f"Python & {tools.pop('Python')} \\\\"] + [f"{tex(k)} & {v} \\\\" for k, v in py.items()]
    rows += [f"{tex(k)} & {v} \\\\" for k, v in tools.items()] + [f"{tex(k)} & {v} \\\\" for k, v in r.items()]
    return ("\\begin{table}[ht]\\centering\\small\n\\caption{Software versions used for the results in this paper.}\\label{tab:software}\n"
            "\\begin{tabular}{ll}\n\\toprule\nSoftware & Version \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def exclusions_table():
    ex = pd.read_csv(ROOT / "data" / "curated" / "label_exclusions.csv")
    rows = [f"\\textit{{{tex(r.species)}}} & {r.label_hours:g} & {tex(r.reason)} & {tex(r.source)} \\\\" for r in ex.itertuples()]
    return ("\\begin{longtable}{p{3cm}p{1.2cm}p{6cm}p{4.4cm}}\n"
            "\\caption{Doubling times removed from the training data because they are not maximal growth rates, with the "
            "evidence.}\\label{tab:exclusions}\\\\\n\\toprule\nSpecies & Label (h) & Evidence & Source \\\\\n\\midrule\n\\endfirsthead\n"
            "\\caption[]{(continued)}\\\\\n\\toprule\nSpecies & Label (h) & Evidence & Source \\\\\n\\midrule\n\\endhead\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n")


def literature_table():
    lit = pd.read_csv(ROOT / "data" / "curated" / "literature_growth_rates.csv")
    rows = [f"\\textit{{{tex(r.species)}}} {tex(r.strain)} & {r.doubling_time_hours:g} & {r.growth_temp_c:g} & {tex(r.how_measured)} & "
            f"{tex(r.source)} \\\\" for r in lit.itertuples()]
    return ("\\begin{table}[ht]\\centering\\small\n"
            "\\caption{Doubling times added from the primary literature for nitrifying and methanotrophic strains with a sequenced "
            "genome, with the temperature and conditions of the measurement.}\\label{tab:literature}\n"
            "\\begin{tabular}{p{3.6cm}rrp{4.8cm}p{4.2cm}}\n\\toprule\nStrain & Doubling (h) & \\textdegree C & How measured & Source \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def columns_table():
    spec = importlib.util.spec_from_file_location("export_hq_table", ROOT / "scripts" / "pipeline" / "export_hq_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [f"\\texttt{{{tex(c)}}} & {tex(d)} \\\\" for c, d in module.COLUMNS]
    return ("\\begin{longtable}{p{5.9cm}p{9.9cm}}\n"
            "\\caption{Columns of the released prediction table for the GEM genomes, \\texttt{gem\\_growth\\_predictions.csv}.}"
            "\\label{tab:columns}\\\\\n\\toprule\nColumn & Meaning \\\\\n\\midrule\n\\endfirsthead\n"
            "\\caption[]{(continued)}\\\\\n\\toprule\nColumn & Meaning \\\\\n\\midrule\n\\endhead\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n")


def settings_table():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "deployed_model.yaml").read_text())
    rows = []
    for target, members in cfg.items():
        for kind, params in members.items():
            learner = "histogram boosting" if kind == "hist" else "random forest"
            text = ", ".join(f"{tex(k)} = {v}" for k, v in params.items())
            rows.append(f"{tex(target)} & {learner} & {text} \\\\")
    return ("\\begin{table}[ht]\\centering\\small\n\\caption{Settings of the two members of the deployed model for each target, chosen by "
            "\\texttt{tune\\_deployed\\_model.py}. Settings not listed are scikit-learn defaults; the random forest has \\NForestTrees trees.}"
            "\\label{tab:settings}\n\\begin{tabular}{llp{9.5cm}}\n\\toprule\nTarget & Member & Settings \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "supp_pfam.tex").write_text(pfam_table())
    (OUT / "supp_software.tex").write_text(software_table())
    (OUT / "supp_exclusions.tex").write_text(exclusions_table())
    (OUT / "supp_settings.tex").write_text(settings_table())
    (OUT / "supp_literature.tex").write_text(literature_table())
    (OUT / "supp_columns.tex").write_text(columns_table())
    print("wrote supp_pfam.tex, supp_software.tex, supp_exclusions.tex, supp_settings.tex, supp_literature.tex, supp_columns.tex")


if __name__ == "__main__":
    main()
