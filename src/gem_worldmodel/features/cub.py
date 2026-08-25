"""Codon Usage Bias (CUB) of highly-expressed genes, following gRodon's approach
(Weissman et al. 2021) of using the MILC statistic (Supek & Vlahovicek 2005)
between a highly-expressed gene subset (ribosomal proteins, used as a proxy
for high translational demand) and the whole-genome codon usage background.

This is a from-scratch reimplementation of the MILC formula, not a call into
gRodon's R package, so expect close-but-not-bitwise-identical values, it's
documented as an approximation, not a guaranteed exact reproduction.

Gene sequences are supplied as in-frame nucleotide CDS. When only a genomic
assembly is available (no annotation), `predict_genes` calls pyrodigal to get
real ORF calls rather than fabricating gene boundaries.
"""

import re
from collections import defaultdict

import numpy as np
import pyrodigal
from Bio.Data.CodonTable import standard_dna_table

from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

# Synonymous codon families for the standard genetic code, excluding stop
# codons and the two single-codon amino acids (Met=ATG, Trp=TGG), which
# contribute no usable bias signal.
_CODON_TO_AA = standard_dna_table.forward_table
_AA_FAMILIES: dict[str, list[str]] = defaultdict(list)
for _codon, _aa in _CODON_TO_AA.items():
    _AA_FAMILIES[_aa].append(_codon)
_AA_FAMILIES = {aa: codons for aa, codons in _AA_FAMILIES.items() if len(codons) > 1}

_HIGHLY_EXPRESSED_REGEX = re.compile(r"ribosomal protein", re.IGNORECASE)


def load_cds_fasta(path) -> tuple[list[str], list[str]]:
    """Parse an NCBI `*_cds_from_genomic.fna` file into (sequences, headers)."""
    from Bio import SeqIO

    seqs, headers = [], []
    for record in SeqIO.parse(str(path), "fasta"):
        seqs.append(str(record.seq))
        headers.append(record.description)
    return seqs, headers


def predict_genes(genome_seq: str, meta: bool = True) -> list[str]:
    """Predict CDS nucleotide sequences from a raw genome/contig using pyrodigal."""
    orf_finder = pyrodigal.GeneFinder(meta=meta)
    genes = orf_finder.find_genes(genome_seq.encode())
    return [gene.sequence() for gene in genes]


def codon_counts(cds_seq: str) -> dict[str, int]:
    seq = cds_seq.upper().replace("U", "T")
    counts: dict[str, int] = defaultdict(int)
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if codon in _CODON_TO_AA:
            counts[codon] += 1
    return counts


def aggregate_codon_counts(cds_seqs: list[str]) -> dict[str, int]:
    total: dict[str, int] = defaultdict(int)
    for seq in cds_seqs:
        for codon, n in codon_counts(seq).items():
            total[codon] += n
    return total


def milc_distance(gene_counts: dict[str, int], reference_counts: dict[str, int]) -> float:
    """MILC (Supek & Vlahovicek 2005): length/composition-corrected codon-usage
    distance of one gene's codon usage from a reference distribution.
    """
    total_gene = sum(gene_counts.values())
    if total_gene == 0:
        return np.nan

    m_sum = 0.0
    correction = 0.0
    for aa, codons in _AA_FAMILIES.items():
        n_aa = sum(gene_counts.get(c, 0) for c in codons)
        if n_aa == 0:
            continue
        ref_aa_total = sum(reference_counts.get(c, 0) for c in codons)
        if ref_aa_total == 0:
            continue
        for c in codons:
            n_c = gene_counts.get(c, 0)
            if n_c == 0:
                continue
            e_c = (reference_counts.get(c, 0) / ref_aa_total) * n_aa
            if e_c <= 0:
                continue
            m_sum += 2.0 * n_c * np.log(n_c / e_c)
        correction += (len(codons) - 1) / 2.0

    milc = (m_sum - correction) / total_gene
    return max(milc, 0.0)


def compute_cub(
    all_cds_seqs: list[str],
    highly_expressed_headers: list[str] | None = None,
    highly_expressed_seqs: list[str] | None = None,
) -> float:
    """CUB of highly-expressed genes vs. whole-genome background codon usage.

    Either pass `highly_expressed_seqs` directly, or pass `highly_expressed_headers`
    (parallel to `all_cds_seqs`) and highly-expressed genes are selected via
    ribosomal-protein header matching (requires annotated headers).
    """
    reference_counts = aggregate_codon_counts(all_cds_seqs)

    if highly_expressed_seqs is None:
        if highly_expressed_headers is None or len(highly_expressed_headers) != len(all_cds_seqs):
            logger.warning(
                "no highly-expressed gene set available (need annotated headers or explicit "
                "sequences); returning NaN rather than guessing"
            )
            return np.nan
        highly_expressed_seqs = [
            seq
            for seq, header in zip(all_cds_seqs, highly_expressed_headers)
            if _HIGHLY_EXPRESSED_REGEX.search(header)
        ]

    if len(highly_expressed_seqs) < 10:
        logger.warning(
            f"only {len(highly_expressed_seqs)} highly-expressed genes found; "
            "CUB estimate will be noisy"
        )
    if not highly_expressed_seqs:
        return np.nan

    distances = [
        milc_distance(codon_counts(seq), reference_counts) for seq in highly_expressed_seqs
    ]
    distances = [d for d in distances if not np.isnan(d)]
    return float(np.median(distances)) if distances else np.nan
