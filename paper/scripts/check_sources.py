#!/usr/bin/env python
"""Checks against sources outside the pipeline (the third route of the triangulation; see triangulate.py).

1. References: every bibliography entry with a DOI is looked up on Crossref; its title must match ours and its year must
   be within one year (online and print years differ). Entries without a DOI are listed for a check by hand.
2. External labels: for every row of data/curated/external_growth_rates.csv the source (PubMed abstract, or PMC full text)
   is fetched from NCBI E-utilities and the number quoted in `how_measured` must appear in it.

Needs network access. Writes paper/source_checks.csv. Usage: python paper/scripts/check_sources.py
"""

import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UA = {"User-Agent": "gem-worldmodel-source-check/1.0"}


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            time.sleep(2 + 3 * attempt)
    return None


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def bib_entries(path):
    text = path.read_text()
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}?\s*(?=@|\Z)", text, re.S):
        key, body = m.group(1).strip(), m.group(2)
        field = lambda f: (re.search(rf"\b{f}\s*=\s*\{{(.*?)\}}\s*,", body, re.S | re.I) or [None, None])[1]  # noqa: E731
        year = re.search(r"\byear\s*=\s*\{?(\d{4})", body, re.I)
        yield key, field("title"), field("doi"), int(year.group(1)) if year else None


def check_refs(rows):
    for key, title, doi, year in bib_entries(ROOT / "paper" / "refs.bib"):
        if not doi:
            rows.append({"kind": "reference", "item": key, "status": "no DOI, check by hand", "detail": title})
            continue
        raw = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
        if raw is None:
            rows.append({"kind": "reference", "item": key, "status": "FAIL: DOI not found", "detail": doi})
            continue
        msg = json.loads(raw)["message"]
        theirs = " ".join(msg.get("title", [""]))
        sim = difflib.SequenceMatcher(None, norm(title or ""), norm(theirs)).ratio()
        years = [p["date-parts"][0][0] for k in ("published-print", "published-online", "issued") if (p := msg.get(k)) and p.get("date-parts")]
        year_ok = year is None or any(abs(y - year) <= 1 for y in years)
        ok = sim >= 0.85 and year_ok
        rows.append({"kind": "reference", "item": key, "status": "ok" if ok else "FAIL",
                     "detail": f"title match {sim:.2f}, years {sorted(set(years))} vs {year}; Crossref: {theirs[:90]}"})
        time.sleep(0.3)


def check_external(rows):
    ext = pd.read_csv(ROOT / "data" / "curated" / "external_growth_rates.csv")
    for r in ext.itertuples():
        src = str(r.source)
        m_pmid, m_pmc = re.search(r"PMID\D*(\d+)", src), re.search(r"PMC\D*(\d+)", src)
        if m_pmc:
            text = get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={m_pmc.group(1)}&rettype=xml")
        elif m_pmid:
            text = get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={m_pmid.group(1)}&rettype=abstract&retmode=text")
        else:
            text = None
        time.sleep(0.4)
        if not text:
            rows.append({"kind": "external label", "item": r.species, "status": "FAIL: source not fetched", "detail": src})
            continue
        flat = re.sub(r"<[^>]+>", " ", text)
        flat = re.sub(r"\s+", " ", flat)
        # the number that sets the doubling time: the first number in how_measured that is not the temperature
        temp = None if pd.isna(r.growth_temp_c) else f"{r.growth_temp_c:g}"
        nums = [n for n in re.findall(r"\d+(?:\.\d+)?", str(r.how_measured)) if n != temp]
        found = [n for n in nums if re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", flat)]
        genus = r.species.split()[0]
        named = genus.lower() in flat.lower()
        # the label itself, as the source would state it: hours, minutes or days, at the precision a paper uses
        h = float(r.doubling_time_hours)
        forms = set()
        for v in (h, h * 60, h / 24):
            for d in (0, 1, 2):
                s = f"{v:.{d}f}"
                if abs(float(s) - v) <= 0.051 * v:
                    forms.add(s.rstrip("0").rstrip(".") if "." in s else s)
        label_found = sorted(f for f in forms if float(f) > 0 and re.search(rf"(?<![\d.]){re.escape(f)}(?![\d])", flat))
        ok = bool(label_found) and named
        found = found + [f"label as {x}" for x in label_found]
        rows.append({"kind": "external label", "item": r.species, "status": "ok" if ok else "CHECK",
                     "detail": f"{src}; quoted '{r.how_measured}'; numbers found in source: {found}; genus named: {named}"})


def main():
    rows = []
    check_refs(rows)
    check_external(rows)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "paper" / "source_checks.csv", index=False)
    for kind, d in out.groupby("kind"):
        print(f"{kind}: {(d['status'] == 'ok').sum()} of {len(d)} ok")
        for row in d[d["status"] != "ok"].itertuples():
            print(f"  {row.status}: {row.item}: {row.detail}")
    sys.exit(1 if out["status"].str.startswith("FAIL").any() else 0)


if __name__ == "__main__":
    main()
