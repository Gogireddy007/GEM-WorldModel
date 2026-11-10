#!/usr/bin/env python
"""Makes Crossref BibTeX safe for LaTeX: HTML italics to \\textit, bare underscores and ampersands escaped outside
URL and DOI fields, and protects capitalised species and genus names in titles from case changes."""
import re
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "refs.bib"
out = []
for line in path.read_text().splitlines():
    field = re.match(r"\s*(\w+)\s*=", line)
    name = field.group(1).lower() if field else ""
    if name not in ("url", "doi", "howpublished"):
        line = re.sub(r"<i>(.*?)</i>", r"\\textit{\1}", line)
        line = re.sub(r"</?(sub|sup|b|scp)>", "", line)
        line = re.sub(r"(?<!\\)_", r"\\_", line)
        line = re.sub(r"(?<!\\)&(?!amp;)", r"\\&", line)
        line = line.replace("&amp;", r"\&")
    out.append(line)
path.write_text("\n".join(out) + "\n")
print("cleaned", path.name)
