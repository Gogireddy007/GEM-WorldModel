#!/usr/bin/env python
"""Inputs for the external test set (data/curated/external_growth_rates.csv): doubling times from species descriptions
whose species are in neither the Madin compilation (by NCBI taxonomy ID) nor our labels, each with a sequenced genome.

Every step is the one used for the labeled genomes, applied to these accessions only; rows for them are appended to the
shared per-genome tables and nothing already computed is touched:

  traits, GTDB placement, codon usage   build_madin_genome_extension.build -> external_extension.csv
  lifestyle genes                       compute_lifestyle_labeled.work     -> labeled_lifestyle_genes.csv
  16S cut from the genome               extract_16s_from_genomes.first_16s -> labeled_16s_from_genomes.csv
  released gRodon statistics            prepare_released_tools_input.work and scripts/r/grodon_stats.R -> labeled_grodon_stats.csv

The external genomes are never part of the training table: deploy_features.load_training_table adds them only when
"external" is asked for by name (EXTERNAL_ROWS). evaluate_external.py scores them.
"""

import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_madin_genome_extension as gx  # noqa: E402
import compute_lifestyle_labeled as life  # noqa: E402
import prepare_released_tools_input as rel  # noqa: E402
from extract_16s_from_genomes import first_16s  # noqa: E402

from gem_worldmodel.utils.config import load_config, resolve_path  # noqa: E402


def main():
    data_cfg, feat_cfg = load_config("data"), load_config("features")
    processed = resolve_path(data_cfg["paths"]["processed_dir"])
    ext = pd.read_csv(resolve_path("data/curated/external_growth_rates.csv"))
    ext = ext.rename(columns={"doubling_time_hours": "d", "growth_temp_c": "GrowthTemp"})
    gx.build(ext, data_cfg, feat_cfg, processed / "external_extension.csv", 8, extra=["how_measured", "source"])
    built = pd.read_csv(processed / "external_extension.csv")
    accessions = sorted(built["accession"])

    path = processed / "labeled_lifestyle_genes.csv"
    have = set(pd.read_csv(path, usecols=["accession"])["accession"])
    todo = [a for a in accessions if a not in have]
    if todo:
        with Pool(8) as pool:
            rows = pool.map(life.work, todo)
        pd.DataFrame(rows).to_csv(path, mode="a", header=False, index=False,
                                  columns=pd.read_csv(path, nrows=0).columns)
    print(f"lifestyle genes: {len(todo)} genomes added")

    path = processed / "labeled_16s_from_genomes.csv"
    have = set(pd.read_csv(path, usecols=["accession"])["accession"])
    jobs = [(s, a, "arc" if t.startswith("d__Archaea") else "bac")
            for s, a, t in zip(built["species"], built["accession"], built["gtdb_taxonomy"], strict=True) if a not in have]
    if jobs:
        with Pool(8) as pool:
            seqs = pool.map(first_16s, jobs)
        pd.DataFrame({"accession": [j[1] for j in jobs], "sequence": [q for _, q in seqs]}).to_csv(path, mode="a", header=False, index=False)
    print(f"16S from genomes: {len(jobs)} genomes added, {sum(bool(q) for _, q in seqs) if jobs else 0} with a full-length copy")

    path = processed / "labeled_grodon_stats.csv"
    have = set(pd.read_csv(path, usecols=["genome_name"])["genome_name"])
    todo = [a for a in accessions if rel.gtdb_name(a) not in have]
    if todo:
        with Pool(8) as pool:
            genes = pool.map(rel.work, todo)
        inp = processed / "external_grodon_input.csv"
        pd.DataFrame({"genome_name": [rel.gtdb_name(a) for a, _, _ in genes], "gene_location": [loc for _, loc, _ in genes]}).to_csv(inp, index=False)
        out = processed / "external_grodon_stats.csv"
        subprocess.run(["Rscript", "scripts/r/grodon_stats.R", str(inp), str(out), "8"], check=True, cwd=resolve_path("."))
        new = pd.read_csv(out)
        new[pd.read_csv(path, nrows=0).columns].to_csv(path, mode="a", header=False, index=False)
        print(f"gRodon statistics: {len(new)} genomes added")


if __name__ == "__main__":
    main()
