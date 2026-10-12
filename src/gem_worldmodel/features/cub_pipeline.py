"""Codon usage bias from a raw genome sequence, with no annotation needed.

MAGs come with no gene annotation, so genes are predicted with pyrodigal and
the ribosomal proteins (the highly expressed set the bias is measured on) are
found with HMMER against Pfam's ribosomal protein families at their curated
gathering thresholds. The bias itself is the MILC statistic already used in
`cub.py`, so the number means the same thing for isolate genomes and MAGs as
long as both go through this function.
"""

import gzip
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyrodigal
from Bio import SeqIO

from gem_worldmodel.features import cub

MIN_TRAINING_BP = 100_000


def read_contigs(path) -> list[bytes]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [str(r.seq).upper().encode() for r in SeqIO.parse(handle, "fasta") if len(r.seq) >= 200]


def predict_genes(contigs: list[bytes]) -> tuple[list[str], list[str]]:
    """Gene nucleotide and protein sequences for every contig.

    Genomes of at least 100 kb are used to train the gene finder on themselves,
    smaller ones use the pre-trained metagenomic models.
    """
    total = sum(len(c) for c in contigs)
    if total >= MIN_TRAINING_BP:
        finder = pyrodigal.GeneFinder(meta=False)
        finder.train(*contigs)
    else:
        finder = pyrodigal.GeneFinder(meta=True)
    nucleotides, proteins = [], []
    for contig in contigs:
        for gene in finder.find_genes(contig):
            nucleotides.append(gene.sequence())
            proteins.append(gene.translate().rstrip("*"))
    return nucleotides, proteins


def ribosomal_protein_indices(proteins: list[str], hmm_path, workdir=None) -> set[int]:
    """Indices of proteins that hit a ribosomal protein HMM above its gathering threshold."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        faa, tbl = Path(tmp) / "p.faa", Path(tmp) / "hits.tbl"
        with open(faa, "w") as handle:
            for i, p in enumerate(proteins):
                if p:
                    handle.write(f">{i}\n{p}\n")
        subprocess.run(
            ["hmmsearch", "--cut_ga", "--noali", "--cpu", "1", "--tblout", str(tbl), "-o", "/dev/null", str(hmm_path), str(faa)],
            check=True,
        )
        hits = set()
        for line in open(tbl):
            if not line.startswith("#"):
                hits.add(int(line.split()[0]))
    return hits


IVYWREL = set("IVYWREL")


def ivywrel_fraction(proteins: list[str]) -> float:
    """Fraction of all amino acids that are I, V, Y, W, R, E or L.

    Across prokaryotes this proteome-wide fraction tracks optimal growth
    temperature closely, because those residues stabilise proteins at high
    temperature (Zeldovich et al. 2007).
    """
    total = sum(len(p) for p in proteins)
    return sum(1 for p in proteins for aa in p if aa in IVYWREL) / total if total else float("nan")


def genome_cub(genome_path, hmm_path, workdir=None) -> dict:
    """CUB, IVYWREL fraction, gene count and ribosomal protein count for one genome FASTA (plain or gzipped)."""
    nucleotides, proteins = predict_genes(read_contigs(genome_path))
    rp = ribosomal_protein_indices(proteins, hmm_path, workdir)
    rp_seqs = [nucleotides[i] for i in sorted(rp)]
    value = cub.compute_cub(nucleotides, highly_expressed_seqs=rp_seqs) if len(rp_seqs) >= 10 else np.nan
    return {"cub": value, "ivywrel": ivywrel_fraction(proteins), "n_genes": len(nucleotides), "n_ribosomal": len(rp_seqs)}
