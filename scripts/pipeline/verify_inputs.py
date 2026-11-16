#!/usr/bin/env python
"""Checks that the downloaded inputs are byte for byte the ones the paper used.

data/input_checksums.tsv lists every file the pipeline downloads (gRodon's growth-rate tables, GTDB release 232, Madin's
trait tables, the GEM catalog metadata and tree, Pfam 38.2 and the NCBI genome assemblies of the labeled species) with
its size and SHA-256. Run it after the downloads (make verify-inputs). A file that differs is reported; a file that is
missing is reported too (some are only downloaded by later steps). Exit code 1 if any present file differs.

  python scripts/pipeline/verify_inputs.py            check
  python scripts/pipeline/verify_inputs.py --write    rewrite the list from the files on disk (maintainers only)
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
LIST = ROOT / "data" / "input_checksums.tsv"

TOP = ["GrowthRates_Madin.rda", "Accession2Species_Madin.rda", "bac120.tree.gz", "bac120_taxonomy.tsv.gz", "ar53.tree.gz",
       "ar53_taxonomy.tsv.gz", "condensed_species_NCBI.csv", "gem_genome_metadata.tsv", "gem_multi_marker.rooted.tree",
       "Pfam-A.hmm.gz"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inputs_on_disk():
    files = [RAW / f for f in TOP]
    files += sorted((RAW / "madin_prepared").glob("*.csv"))
    files += sorted((RAW / "ncbi_genomes").glob("*/*.fna.gz"))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.write:
        rows = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in inputs_on_disk()]
        pd.DataFrame(rows).to_csv(LIST, sep="\t", index=False)
        print(f"wrote {LIST.relative_to(ROOT)}: {len(rows)} files")
        return
    want = pd.read_csv(LIST, sep="\t")
    ok, missing, differ = 0, [], []
    for r in want.itertuples():
        p = ROOT / r.path
        if not p.exists():
            missing.append(r.path)
        elif p.stat().st_size != r.bytes or sha256(p) != r.sha256:
            differ.append(r.path)
        else:
            ok += 1
    print(f"{ok} of {len(want)} inputs identical to the paper's, {len(missing)} missing, {len(differ)} different")
    for p in differ:
        print(f"  DIFFERENT: {p}")
    if missing:
        print("  missing (first 10): " + ", ".join(missing[:10]))
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
