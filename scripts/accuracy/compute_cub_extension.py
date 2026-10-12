#!/usr/bin/env python
"""This computes codon usage bias for the Madin extension species from a genome.

Madin's table has no genomes, so each species is matched by name to an NCBI
accession through the accession table Madin published, its genomic FASTA is
downloaded (only the genome, not the annotation), and the annotation-free CUB
pipeline is run, the same one used for the original corpus and for GEM MAGs.
Writes extension_cub.csv.
"""

import glob
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
import requests

from gem_worldmodel.data import ncbi_genomes
from gem_worldmodel.features import cub_pipeline
from gem_worldmodel.utils.config import load_config, resolve_path

HMM = "data/raw/ribosomal_pfam.hmm"


def ftp_dir(accession):
    """Genome folder on NCBI's FTP, from the directory listing (E-utilities rate-limit parallel lookups)."""
    prefix, digits = accession.split("_")[0], accession.split("_")[1].split(".")[0]
    parent = f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{prefix}/{digits[0:3]}/{digits[3:6]}/{digits[6:9]}/"
    for attempt in range(4):
        try:
            html = requests.get(parent, timeout=60).text
            m = re.search(rf'href="({accession}[^"/]*)/?"', html)
            return parent + m.group(1) if m else None
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return None


def work(args):
    species, accession = args
    empty = {"species": species, "accession": accession, "cub": np.nan, "ivywrel": np.nan, "n_genes": np.nan, "n_ribosomal": np.nan}
    try:
        folder = Path("data/raw/ncbi_genomes") / accession
        cached = sorted(glob.glob(str(folder / "*_genomic.fna.gz")))
        if cached:
            gz = Path(cached[0])
        else:
            ftp = ftp_dir(accession)
            if ftp is None:
                return empty
            base = ftp.rstrip("/").split("/")[-1]
            gz = folder / f"{base}_genomic.fna.gz"
            ncbi_genomes._download_gz(f"{ftp}/{base}_genomic.fna.gz", gz, folder / f"{base}_genomic.fna")
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
