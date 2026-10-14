#!/usr/bin/env python
"""This adds 16S sequences for the genome-based labeled rows (the Madin genome
extension and the literature rows) to labeled_16s_sequences.csv.

The sequence is cut from the genome of the measured strain with barrnap, the
same way it is cut from GEM MAGs, using the archaeal model for archaea. The
first copy of at least 1,200 bp is kept. Species that already have a sequence are left as
they are, so the 16 reference sequences and the original rows do not change.
"""

import subprocess
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from gem_worldmodel.features import cub_pipeline
from gem_worldmodel.utils.config import load_config, resolve_path

GENOMES = Path("data/raw/ncbi_genomes")


def first_16s(job):
    species, accession, kingdom = job
    path = cub_pipeline.genome_fasta(GENOMES / accession)
    if path is None or path.suffix == ".gz":
        return species, ""
    try:
        out = subprocess.run(["barrnap", "--quiet", "--kingdom", kingdom, "--outseq", "/dev/stdout", str(path)],
                             capture_output=True, text=True, timeout=600).stdout
    except (subprocess.SubprocessError, OSError):
        return species, ""
    seqs, current = [], None
    for line in out.splitlines():
        if line.startswith(">"):
            current = [] if line.startswith(">16S") else None
            if current is not None:
                seqs.append(current)
        elif current is not None:
            current.append(line.strip())
    full = [q for q in ("".join(c) for c in seqs) if len(q) >= 1200]
    return species, full[0] if full else ""


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    path = processed / "labeled_16s_sequences.csv"
    have = pd.read_csv(path).fillna("")
    known = set(have.loc[have["sequence"] != "", "species"])
    jobs = []
    for name in ("madin_genome_extension.csv", "literature_extension.csv"):
        df = pd.read_csv(processed / name)
        jobs += [(s, a, "arc" if t.startswith("d__Archaea") else "bac")
                 for s, a, t in zip(df["species"], df["accession"], df["gtdb_taxonomy"], strict=True) if s not in known]
    with Pool(8) as pool:
        found = dict(pool.map(first_16s, jobs))
    new = pd.DataFrame({"species": list(found), "sequence": list(found.values())})
    out = pd.concat([have[~have["species"].isin(found)], new], ignore_index=True)
    out.to_csv(path, index=False)
    lengths = new.loc[new["sequence"] != "", "sequence"].str.len()
    print(f"{len(jobs)} species looked up, {(new['sequence'] != '').sum()} with a full 16S copy "
          f"(length median {lengths.median():.0f}, min {lengths.min()}); file now has {len(out)} species")


if __name__ == "__main__":
    main()
