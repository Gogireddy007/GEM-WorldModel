#!/usr/bin/env python
"""This counts the lifestyle Pfam families (lifestyle_genes.FAMILIES) in every
labeled genome: the original corpus, the Madin extension, the Madin genome
extension and the literature rows.

The HMM files are cut from Pfam-A by accession the first time (families without
a gathering threshold are reported and left out; EVALUE_FAMILIES go in a second
file searched with an E-value cutoff). Genomes are the cached NCBI
downloads used for the codon usage pipeline. Writes labeled_lifestyle_genes.csv
with one row per accession.
"""

import gzip
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from gem_worldmodel.features import cub_pipeline, lifestyle_genes
from gem_worldmodel.utils.config import load_config, resolve_path

RAW = Path("data/raw")
HMM = RAW / "lifestyle_pfam.hmm"
EVALUE_HMM = RAW / "lifestyle_pfam_evalue.hmm"


def build_hmm():
    kept, no_ga, by_evalue = [], [], []
    with gzip.open(RAW / "Pfam-A.hmm.gz", "rt") as handle:
        block = []
        for line in handle:
            block.append(line)
            if line.startswith("//"):
                acc = next((ln.split()[1].split(".")[0] for ln in block if ln.startswith("ACC ")), None)
                if acc in lifestyle_genes.FAMILIES:
                    (kept if any(ln.startswith("GA ") for ln in block) else no_ga).append((acc, "".join(block)))
                elif acc in lifestyle_genes.EVALUE_FAMILIES:
                    by_evalue.append((acc, "".join(block)))
                block = []
    HMM.write_text("".join(text for _, text in kept))
    EVALUE_HMM.write_text("".join(text for _, text in by_evalue))
    missing = set(lifestyle_genes.FAMILIES) - {a for a, _ in kept}
    print(f"kept {len(kept)} of {len(lifestyle_genes.FAMILIES)} families; without a gathering threshold or not in Pfam: {sorted(missing)}")


def work(accession):
    empty = {"accession": accession, **{acc: np.nan for acc in [*lifestyle_genes.FAMILIES, *lifestyle_genes.EVALUE_FAMILIES]}, "n_genes": np.nan}
    path = cub_pipeline.genome_fasta(RAW / "ncbi_genomes" / accession)
    if path is None:
        return empty
    try:
        return {"accession": accession, **lifestyle_genes.genome_lifestyle(path, HMM, EVALUE_HMM)}
    except Exception as exc:  # noqa: BLE001
        print(f"{accession}: {str(exc)[:100]}", flush=True)
        return empty


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    if not (HMM.exists() and EVALUE_HMM.exists()):
        build_hmm()
    accessions = set(pd.read_csv(processed / "features_sample_expanded.csv")["accession"].dropna())
    accessions |= set(pd.read_csv(processed / "extension_cub.csv")["accession"].dropna())
    for name in ("madin_genome_extension.csv", "literature_extension.csv"):
        if (processed / name).exists():
            accessions |= set(pd.read_csv(processed / name)["accession"].dropna())
    jobs = sorted(accessions)
    rows = []
    with Pool(8) as pool:
        for i, row in enumerate(pool.imap_unordered(work, jobs)):
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(jobs)} genomes done", flush=True)
    out = pd.DataFrame(rows).sort_values("accession")
    out.to_csv(processed / "labeled_lifestyle_genes.csv", index=False)
    print(f"wrote labeled_lifestyle_genes.csv: {len(out)} genomes, {out['n_genes'].notna().sum()} with counts")
    print("share of genomes with at least one hit, by family:")
    names = {**lifestyle_genes.FAMILIES, **lifestyle_genes.EVALUE_FAMILIES}
    print((out[list(names)] > 0).mean().round(2).rename(names).to_string())


if __name__ == "__main__":
    main()
