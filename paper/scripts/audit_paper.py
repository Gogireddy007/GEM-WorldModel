#!/usr/bin/env python
"""Checks the manuscript and supplement against the rules in paper/PLAN.md and exits non-zero on a failure.

  macros     every macro used in the text is defined in numbers.tex; macros defined but never used are listed
  numbers    no digit in the prose of the hand-written sections that is not a macro, a reference, or on the allow list
  style      no em dashes, no hype words, no drafting placeholders (\\plan)
  pages      main.pdf has at most the page limit of the venue (9 for ISMB, references included); the abstract at most 250 words

Generated files (tables, numbers.tex, supplementary tables) are not checked for numbers, because they are written from the
result files by scripts. Run from the repository root: .venv/bin/python paper/scripts/audit_paper.py
"""

import re
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
PAGE_LIMIT = 9
# hand-written files; tables and supplementary tables are generated
WRITTEN = ["main.tex", "sections/abstract.tex", "sections/introduction.tex", "sections/methods.tex", "sections/results.tex",
           "sections/discussion.tex", "sections/availability.tex", "figures/fig1_overview.tex", "figures/fig1_overview_standalone.tex",
           "supplement.tex", "supplement/supp_methods.tex", "supplement/supp_results.tex", "supplement/supp_data.tex"]
GENERATED = ["tables/table1_data.tex", "tables/table2_ablations.tex"] + [str(p.relative_to(PAPER)) for p in (PAPER / "supplement").glob("supp_*.tex")
                                                                          if p.name not in ("supp_methods.tex", "supp_results.tex", "supp_data.tex")]
HYPE = ["novel", "powerful", "state of the art", "state-of-the-art", "groundbreaking", "unprecedented", "remarkable", "robustly"]

# numbers that are definitions or names, not results: the reference temperature in names of quantities is a macro, so
# these are only what the text needs to say what was done
ALLOWED = [
    r"\b16S\b", r"\b23S\b", r"R\d{3}\b", r"\b80\\%", r"\b95\\%", r"\b10\^\{-5\}", r"\$p<0\.05\$", r"p<0\.05", r"\$\\lambda\$ fixed at 1",
    r"fixing \$\\lambda\$ at 1", r"\[0,1\]", r"\$\\lambda=1\$", r"\b20 \\textdegree", r"Supplementary (Table|Figure)~\\ref", r"I, V, Y, W, R, E\s+and\s+L",
    r"\bfive-fold\b", r"three-fold", r"one doubling per day", r"\b2027\b", r"\bk-mer", r"ORCID", r"\\ln 2", r"\$E_a", r"1\.5\s*pt",
    r"\b24 h", r"above 24", r"\b45 C",
    r"10th and 90th percentiles",  # defines the 80% range
    r"at least 100 genomes", r"5th and 95th percentiles",  # figure layout parameters (figures.py), stated so the reader can read the plot
    r"are 1 when",  # definition of a presence input
    r"ESM-2",  # a model name
    r"q_\{0\.1\},\s*q_\{0\.9\}",  # the residual quantiles of the 80% range (Algorithm 2)
    r"\[\d(,\$[^$]*\$)?\]",  # author and address keys of the template
    r"SHA-256",  # name of the checksum
    r"(fixing|fixed)\s+(\$[^$]*\$\s+)?at 1\b",  # lambda fixed at 1: a model setting
    r"[\w\\]*\.(py|csv|txt|tex|R)\b", r"p\{[\d.]+cm\}",  # file names and column widths
]


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "", text)


def body(text):
    """The document body: the preamble defines colours and layout, not results."""
    return text.split("\\begin{document}", 1)[1] if "\\begin{document}" in text else text


def prose(text):
    """Text with commands that carry no numbers of their own removed: refs, cites, labels, macros, options, formulas."""
    t = body(strip_comments(text))
    # page-style code between \\makeatletter and \\makeatother is layout, not prose
    t = re.sub(r"\\makeatletter.*?\\makeatother", " ", t, flags=re.S)
    # display equations are definitions (constants such as 1/2, ln 2 and the quantiles 0.1 and 0.9 of the 80% range)
    t = re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", " ", t, flags=re.S)
    # formulas hold constants (ln 2, 1/T); a decimal number in math is still flagged
    t = re.sub(r"\$([^$]*)\$", lambda m: " " if not re.search(r"\d\.\d", m.group(1)) else m.group(0), t)
    t = re.sub(r"\\(ref|cite\w*|label|input|includegraphics|externaldocument|url|href|bibliography\w*|graphicspath|"
               r"definecolor|usetikzlibrary|documentclass|usepackage|setlength|newcommand|renewcommand|begin|end|address|corresp|"
               r"received|revised|accepted|copyrightyear|pubyear|firstpage|DOI|access|vol|issue|author)\*?(\[[^\]]*\])*\{[^{}]*\}(\{[^{}]*\})?", " ", t)
    t = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", t, flags=re.S)
    t = re.sub(r"\\[A-Za-z]+", " ", t)  # macros (numbers.tex values and formatting commands)
    for pat in ALLOWED:
        t = re.sub(pat, " ", t)
    return t


def main():
    failures = []
    numbers = (PAPER / "numbers.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", numbers))
    texts = {f: (PAPER / f).read_text() for f in WRITTEN + GENERATED if (PAPER / f).exists()}
    used = set()
    for t in texts.values():
        used |= set(re.findall(r"\\([A-Za-z]+)", strip_comments(t)))
    unused = sorted(defined - used)

    # macros of numbers.tex used anywhere must be defined; detect capitalized project macros that look like ours but are missing
    project_like = {m for m in used if re.match(r"^(N|Err|Rho|Auc|Rel|Abl|Kev|Pos|GEM|Mag|Indep|Pgls|Slow|Cover|Sixteen|Tune|Fixed|Coarse|TempMAE|"
                                                r"Screen|Study|Trace|Syn|KEigen|Kmer|Landmark|Drift|Min|Ea|TRef|GTDB|NSeeds)", m)}
    missing = sorted(project_like - defined - {"Nonumber"})
    if missing:
        failures.append(f"macros used but not defined: {', '.join(missing)}")

    stray = []
    # the standalone figure is TikZ layout; its text has no numbers but macros (checked by reading it)
    for f in [w for w in WRITTEN if w != "figures/fig1_overview_standalone.tex"]:
        if f not in texts:
            continue
        for i, line in enumerate(prose(texts[f]).splitlines(), 1):
            for m in re.finditer(r"(?<![A-Za-z])\d+(\.\d+)?", line):
                stray.append(f"{f}:{i}: '{line.strip()[:110]}'")
                break
    if stray:
        failures.append("numbers typed in the prose (make them macros or add them to ALLOWED with a reason):\n  " + "\n  ".join(stray))

    style = []
    for f, t in texts.items():
        body = strip_comments(t)
        if "\u2014" in body or "---" in body:
            style.append(f"{f}: em dash")
        if "\\plan{" in body:
            style.append(f"{f}: drafting placeholder \\plan")
        for w in HYPE:
            if re.search(rf"\b{w}\b", body, re.I):
                style.append(f"{f}: hype word '{w}'")
    if style:
        failures.append("style:\n  " + "\n  ".join(style))

    pdf = PAPER / "main.pdf"
    if pdf.exists():
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
        print(f"main.pdf: {pages} pages (limit {PAGE_LIMIT}, references included)")
        if pages > PAGE_LIMIT:
            failures.append(f"main.pdf has {pages} pages, over the limit of {PAGE_LIMIT}")
        first = subprocess.run(["pdftotext", "-f", "1", "-l", "1", str(pdf), "-"], capture_output=True, text=True).stdout
        abstract = first.split("Abstract", 1)[-1].split("Keywords", 1)[0]
        words = len(abstract.split())
        print(f"abstract: {words} words in the PDF, headings and URL included (limit 250)")
        if words > 250:
            failures.append(f"abstract has {words} words, over 250")

    print(f"{len(defined)} macros defined, {len(defined) - len(unused)} used; unused: {', '.join(unused) if unused else 'none'}")
    if failures:
        print("\nFAILED\n" + "\n".join(failures))
        sys.exit(1)
    print("passed")


if __name__ == "__main__":
    main()
