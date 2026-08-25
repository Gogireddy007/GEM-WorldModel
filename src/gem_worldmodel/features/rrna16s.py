"""16S rRNA baseline branch, kept as a separate comparison point against the
GTDB-distance embedding, not the primary phylogeny signal (per the post-pivot plan).

We fetch a representative 16S rRNA sequence per species from NCBI's nucleotide
database (Entrez), rather than attempting de-novo extraction from raw genome
assemblies (that would require a dedicated rRNA predictor like barrnap, which
isn't available in this environment). This is the standard shortcut used when
the goal is a phylogenetic-distance proxy rather than exact annotation.

Distance between two 16S sequences is computed as an alignment-free k-mer
profile distance (fast, no MSA dependency), then reduced to a fixed-length
embedding via the same classical MDS routine used for the GTDB branch.
"""

import itertools
import time

import numpy as np
from Bio import Entrez, SeqIO

from gem_worldmodel.features.phylogeny import classical_mds_embedding
from gem_worldmodel.utils.config import load_config
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

Entrez.email = "ygogireddy@gmail.com"
Entrez.tool = "gem-worldmodel"


def fetch_16s_sequence(species: str, retries: int = 2, sleep_s: float = 0.4) -> str | None:
    """Fetch one representative 16S rRNA sequence for `species` from NCBI nucleotide."""
    term = f'"{species}"[Organism] AND 16S ribosomal RNA[All Fields] AND 1000:2000[SLEN]'
    for attempt in range(retries + 1):
        try:
            with Entrez.esearch(db="nucleotide", term=term, retmax=1) as handle:
                record = Entrez.read(handle)
            ids = record.get("IdList", [])
            if not ids:
                return None
            with Entrez.efetch(db="nucleotide", id=ids[0], rettype="fasta", retmode="text") as handle:
                seq_record = SeqIO.read(handle, "fasta")
            time.sleep(sleep_s)  # NCBI rate-limit courtesy
            return str(seq_record.seq).upper()
        except Exception as exc:  # noqa: BLE001 - network/parse errors, retry then give up
            logger.warning(f"16S fetch failed for {species!r} (attempt {attempt + 1}): {exc}")
            time.sleep(sleep_s)
    return None


def kmer_profile(seq: str, k: int) -> dict[str, int]:
    profile: dict[str, int] = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if "N" in kmer:
            continue
        profile[kmer] = profile.get(kmer, 0) + 1
    return profile


def kmer_cosine_distance(profile_a: dict[str, int], profile_b: dict[str, int]) -> float:
    keys = set(profile_a) | set(profile_b)
    a = np.array([profile_a.get(k, 0) for k in keys], dtype=float)
    b = np.array([profile_b.get(k, 0) for k in keys], dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    cosine_sim = float(np.dot(a, b) / denom)
    return 1.0 - cosine_sim


def build_16s_distance_matrix(sequences: dict[str, str], k: int) -> tuple[np.ndarray, list[str]]:
    """sequences: {species_or_accession: 16S sequence}. Returns (matrix, labels)."""
    labels = list(sequences.keys())
    profiles = {label: kmer_profile(sequences[label], k) for label in labels}
    n = len(labels)
    mat = np.zeros((n, n), dtype=float)
    for i, j in itertools.combinations(range(n), 2):
        d = kmer_cosine_distance(profiles[labels[i]], profiles[labels[j]])
        mat[i, j] = d
        mat[j, i] = d
    return mat, labels


def build_16s_embeddings(sequences: dict[str, str], cfg: dict | None = None) -> dict[str, np.ndarray]:
    cfg = cfg or load_config("features")
    r = cfg["rrna16s"]
    matrix, labels = build_16s_distance_matrix(sequences, k=r["kmer_k"])
    embedding = classical_mds_embedding(matrix, r["embedding_dim"])
    return {label: embedding[i] for i, label in enumerate(labels)}
