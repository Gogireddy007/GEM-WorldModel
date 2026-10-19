#!/usr/bin/env python
"""This writes gene files for the released gRodon and Phydon R packages, for every labeled genome.

Genes come from the same annotation-free pipeline as our own codon usage feature (pyrodigal on the
genome, ribosomal proteins found with the Pfam HMMs), so all three methods see the same genes. Each
genome gets the layout Phydon expects (Prokka-style): <acc>/<acc>.ffn with the nucleotide genes, the
ribosomal proteins named "... ribosomal protein" (how gRodon picks its highly expressed set), and
<acc>/<acc>_CDS_names.txt listing every gene. The genome name passed to Phydon is the accession as
GTDB writes it (RS_GCF_..., GB_GCA_...), so Phydon can find its species on its own tree.

Writes data/raw/released_tools/genes/ and data/processed/released_tools_input.csv.
"""

from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from gem_worldmodel.features import cub_pipeline, deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

HMM = "data/raw/ribosomal_pfam.hmm"
OUT = Path("data/raw/released_tools/genes")


def gtdb_name(accession):
    return ("RS_" if accession.startswith("GCF_") else "GB_") + accession


def work(accession):
    name = gtdb_name(accession)
    folder = OUT / name
    ffn = folder / f"{name}.ffn"
    if ffn.exists():
        return accession, str(ffn.resolve()), None
    path = cub_pipeline.genome_fasta(Path("data/raw/ncbi_genomes") / accession)
    if path is None:
        return accession, None, "no genome"
    try:
        nucleotides, proteins = cub_pipeline.predict_genes(cub_pipeline.read_contigs(path))
        ribosomal = cub_pipeline.ribosomal_protein_indices(proteins, HMM)
        folder.mkdir(parents=True, exist_ok=True)
        ids = [f"{name}_{i:05d}" for i in range(len(nucleotides))]
        with open(ffn, "w") as handle:
            for i, (gid, seq) in enumerate(zip(ids, nucleotides, strict=True)):
                label = "ribosomal protein" if i in ribosomal else "predicted protein"
                handle.write(f">{gid} {label}\n{seq}\n")
        (folder / f"{name}_CDS_names.txt").write_text("\n".join(ids) + "\n")
        return accession, str(ffn.resolve()), None
    except Exception as exc:  # noqa: BLE001
        return accession, None, str(exc)[:80]


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    df, _, _ = deploy_features.load_training_table(processed, load_config("features")["rrna16s"]["kmer_k"], with_cub=True)
    df = df[df["accession"].notna()]
    accessions = sorted(set(df["accession"]))
    with Pool(8) as pool:
        res = pool.map(work, accessions)
    out = pd.DataFrame(res, columns=["accession", "gene_location", "error"])
    out["genome_name"] = out["accession"].map(gtdb_name)
    meta = df.drop_duplicates("accession").set_index("accession")[["species", "species_group", "growth_temp_c", "doubling_time_hours", "gtdb_taxonomy", "archaeon"]]
    out = out.join(meta, on="accession")
    out.to_csv(processed / "released_tools_input.csv", index=False)
    print(f"{len(out)} genomes, {out['gene_location'].notna().sum()} written, errors: {out['error'].dropna().value_counts().to_dict()}")


if __name__ == "__main__":
    main()
