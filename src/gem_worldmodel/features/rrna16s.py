"""16S rRNA baseline branch, kept as a separate comparison point against the
GTDB-distance embedding, not the primary phylogeny signal (per the post-pivot plan).

For the labeled corpus, we fetch a representative 16S rRNA sequence per
species from NCBI's nucleotide database (Entrez) rather than extracting it
from the genome assembly directly, since that's one sequence per species and
NCBI already has curated 16S records for named organisms. For the GEM MAG
corpus, scripts/gem_slow_features.py extracts real 16S sequences directly
from each assembly via barrnap instead, since MAGs don't have NCBI records to
fetch, and stores the resulting k-mer profile inline
(build_16s_embeddings_from_profiles below consumes that directly).

Distance between two 16S sequences is computed as an alignment-free k-mer
profile distance (fast, no MSA dependency), then reduced to a fixed-length
embedding via the same classical MDS routine used for the GTDB branch.
"""

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
    profiles = {label: kmer_profile(seq, k) for label, seq in sequences.items()}
    return build_16s_distance_matrix_from_profiles(profiles)


def build_16s_embeddings(sequences: dict[str, str], cfg: dict | None = None) -> dict[str, np.ndarray]:
    cfg = cfg or load_config("features")
    r = cfg["rrna16s"]
    matrix, labels = build_16s_distance_matrix(sequences, k=r["kmer_k"])
    embedding = classical_mds_embedding(matrix, r["embedding_dim"])
    return {label: embedding[i] for i, label in enumerate(labels)}


def build_16s_distance_matrix_from_profiles(
    profiles: dict[str, dict[str, int]],
) -> tuple[np.ndarray, list[str]]:
    """Same math as pairwise kmer_cosine_distance, but vectorized: builds one
    dense (n_genomes x vocab_size) count matrix and computes all pairwise
    cosine distances as a single matrix multiply, instead of looping over
    pairs in Python. The naive per-pair loop is fine at labeled-corpus scale
    (~175 genomes, ~15k pairs) but chokes at GEM MAG scale (~1,900 genomes
    with real 16S, ~1.8M pairs, measured to hang for minutes). A missing
    k-mer contributes 0 to a profile's count vector either way, so building
    against the full observed vocabulary instead of each pair's own union
    gives mathematically identical distances, not an approximation.
    """
    labels = list(profiles.keys())
    n = len(labels)
    vocab = sorted({kmer for profile in profiles.values() for kmer in profile})
    vocab_index = {kmer: i for i, kmer in enumerate(vocab)}

    counts = np.zeros((n, len(vocab)), dtype=float)
    for row, label in enumerate(labels):
        for kmer, count in profiles[label].items():
            counts[row, vocab_index[kmer]] = count

    norms = np.linalg.norm(counts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # an all-zero profile stays all-zero after normalizing
    normalized = counts / norms
    cosine_sim = normalized @ normalized.T
    mat = 1.0 - cosine_sim
    np.fill_diagonal(mat, 0.0)
    return mat, labels


def build_16s_embeddings_from_profiles(
    profiles: dict[str, dict[str, int]], cfg: dict | None = None
) -> dict[str, np.ndarray]:
    cfg = cfg or load_config("features")
    r = cfg["rrna16s"]
    matrix, labels = build_16s_distance_matrix_from_profiles(profiles)
    embedding = classical_mds_embedding(matrix, r["embedding_dim"])
    return {label: embedding[i] for i, label in enumerate(labels)}
