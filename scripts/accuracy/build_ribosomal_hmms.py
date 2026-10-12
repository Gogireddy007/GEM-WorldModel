#!/usr/bin/env python
"""This builds the ribosomal protein HMM file used to find the highly expressed
genes for codon usage bias.

It takes Pfam-A (downloaded if missing) and keeps the prokaryotic ribosomal
protein families by name, plus Ribosomal_L27A, which is where Pfam files
bacterial L15. Every kept family has curated gathering thresholds, which the
search uses as its cutoff. Eukaryote-only and mitochondrial families are left
out. Writes ribosomal_pfam.hmm.
"""

import gzip
import re
import urllib.request
from pathlib import Path

PFAM_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
RAW = Path("data/raw")
PATTERN = re.compile(r"^(Ribosomal_[LS][0-9]+(p|P)?(_[NCnc])?|ribosomal_L24|Ribosom_S12_S23|Ribosomal_L27A)$")


def main():
    pfam = RAW / "Pfam-A.hmm.gz"
    if not pfam.exists():
        print(f"downloading {PFAM_URL} (about 420 MB)")
        urllib.request.urlretrieve(PFAM_URL, pfam)
    kept = []
    with gzip.open(pfam, "rt") as handle:
        block = []
        for line in handle:
            block.append(line)
            if line.startswith("//"):
                name = next((ln.split()[1] for ln in block if ln.startswith("NAME ")), None)
                has_ga = any(ln.startswith("GA ") for ln in block)
                if name and PATTERN.match(name) and has_ga:
                    kept.append((name, "".join(block)))
                block = []
    (RAW / "ribosomal_pfam.hmm").write_text("".join(text for _, text in kept))
    print(f"kept {len(kept)} ribosomal protein families")


if __name__ == "__main__":
    main()
