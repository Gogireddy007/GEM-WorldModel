#!/usr/bin/env python
"""This computes codon usage bias for the Madin extension species from a genome.

Madin's table has no genomes, so each species is matched by name to an NCBI
accession through the accession table Madin published, its genomic FASTA is
downloaded (only the genome, not the annotation), and the annotation-free CUB
pipeline is run, the same one used for the original corpus and for GEM MAGs.
Writes extension_cub.csv.
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

from gem_worldmodel.data import ncbi_genomes
from gem_worldmodel.features import cub_pipeline
from gem_worldmodel.utils.config import load_config, resolve_path

HMM = "data/raw/ribosomal_pfam.hmm"


def work(args):
    species, accession = args
    empty = {"species": species, "accession": accession, "cub": np.nan, "ivywrel": np.nan, "n_genes": np.nan, "n_ribosomal": np.nan}
    try:
        folder = Path("data/raw/ncbi_genomes") / accession
        cached = cub_pipeline.genome_fasta(folder)
        if cached is not None:
            gz = cached
        else:
            gz = ncbi_genomes.fetch_genomic_fasta(accession, folder)
            if gz is None:
                return empty
        r = cub_pipeline.genome_cub(gz, HMM)
        return {**empty, **r}
    except Exception as exc:  # noqa: BLE001
        print(f"{species} {accession}: {str(exc)[:80]}")
        return empty


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    acc = pyreadr.read_r("data/raw/Accession2Species_Madin.rda")["spp_acc"]
    acc.columns = ["accession", "name"]
    acc["species"] = acc["name"].map(lambda s: " ".join(str(s).replace("[", "").replace("]", "").split()[:2]))
    first = acc.drop_duplicates("species").set_index("species")["accession"]
    ext = pd.read_csv(processed / "madin_extension.csv")
    jobs = [(s, first[s]) for s in ext["species"] if s in first.index]
    rows = []
    with Pool(4) as pool:
        for i, row in enumerate(pool.imap(work, jobs, chunksize=1)):
            rows.append(row)
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(jobs)} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(processed / "extension_cub.csv", index=False)
    print(f"{len(out)} species attempted, {out.cub.notna().sum()} with a value, ribosomal proteins median {out.n_ribosomal.median():.0f}")


if __name__ == "__main__":
    main()
