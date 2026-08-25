"""Genomic trait extraction: genome size, GC content (computed directly from
assembly FASTA), plus rRNA/tRNA gene counts and regulatory gene count.

genome_size_bp and gc_content are computed directly and are exact. The
rRNA/tRNA counts and regulatory-gene count require dedicated annotators
(barrnap, tRNAscan-SE, a regulator HMM/COG scan) that are not available in
this environment, `count_rrna_trna_genes` calls out to `barrnap`/
`tRNAscan-SE` on PATH if present, and returns NaN with an explicit warning
(never a fabricated number) if they aren't. Install those tools and this
function starts producing real counts with no code changes.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
from Bio import SeqIO

from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def genome_size_and_gc(fasta_path: str | Path) -> tuple[int, float]:
    """Exact genome size (bp) and GC content (fraction) from an assembly FASTA."""
    total_len = 0
    gc_count = 0
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq = str(record.seq).upper()
        total_len += len(seq)
        gc_count += seq.count("G") + seq.count("C")
    if total_len == 0:
        raise ValueError(f"no sequences found in {fasta_path}")
    return total_len, gc_count / total_len


def count_rrna_trna_genes(fasta_path: str | Path) -> dict[str, float]:
    """rRNA (16S/23S/5S) and tRNA gene counts via barrnap/tRNAscan-SE if installed.

    Returns NaN for any count whose tool isn't found on PATH, explicit
    missingness, not a guess.
    """
    counts = {
        "rrna_16s_count": np.nan,
        "rrna_23s_count": np.nan,
        "rrna_5s_count": np.nan,
        "trna_count": np.nan,
    }

    if shutil.which("barrnap"):
        try:
            out = subprocess.run(
                ["barrnap", "--quiet", str(fasta_path)],
                capture_output=True, text=True, check=True, timeout=300,
            ).stdout
            rrna_counts = {"rrna_16s_count": 0, "rrna_23s_count": 0, "rrna_5s_count": 0}
            for line in out.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.split("\t")
                attrs = fields[8] if len(fields) > 8 else ""
                if "16S" in attrs:
                    rrna_counts["rrna_16s_count"] += 1
                elif "23S" in attrs:
                    rrna_counts["rrna_23s_count"] += 1
                elif "5S" in attrs:
                    rrna_counts["rrna_5s_count"] += 1
            counts.update(rrna_counts)
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning(f"barrnap failed on {fasta_path}: {exc}")
    else:
        logger.warning("barrnap not found on PATH; rRNA counts will be NaN")

    for exe in ("tRNAscan-SE", "trnascan"):
        if shutil.which(exe):
            try:
                out = subprocess.run(
                    [exe, "-B", "-q", "-o", "-", str(fasta_path)],
                    capture_output=True, text=True, check=True, timeout=300,
                ).stdout
                counts["trna_count"] = max(0, len(out.splitlines()) - 3)
            except (subprocess.SubprocessError, OSError) as exc:
                logger.warning(f"{exe} failed on {fasta_path}: {exc}")
            break
    else:
        logger.warning("tRNAscan-SE not found on PATH; trna_count will be NaN")

    return counts


def regulatory_gene_count(cds_headers: list[str]) -> float:
    """Approximate regulatory gene count via product-annotation keyword matching.

    Requires headers carrying `[protein=...]`/product annotations (e.g. NCBI
    RefSeq/GenBank *_cds_from_genomic.fna). Returns NaN if no headers carry
    recognizable product annotations rather than guessing.
    """
    keywords = (
        "transcriptional regulator", "sigma factor", "two-component",
        "response regulator", "sensor histidine kinase", "repressor", "activator",
    )
    annotated = [h for h in cds_headers if "[protein=" in h or "product=" in h]
    if not annotated:
        return np.nan
    hits = sum(1 for h in annotated if any(kw in h.lower() for kw in keywords))
    return float(hits)


def extract_traits(fasta_path: str | Path, cds_headers: list[str] | None = None) -> dict[str, float]:
    """Extract the full genomic-traits feature vector for one genome assembly."""
    size_bp, gc = genome_size_and_gc(fasta_path)
    traits = {"genome_size_bp": float(size_bp), "gc_content": gc}
    traits.update(count_rrna_trna_genes(fasta_path))
    traits["regulatory_gene_count"] = (
        regulatory_gene_count(cds_headers) if cds_headers is not None else np.nan
    )
    return traits
