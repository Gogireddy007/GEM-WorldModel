import dendropy
import numpy as np
import pytest

from gem_worldmodel.features import cub, genome_traits, phylogeny, rrna16s, temperature


def test_genome_size_and_gc(tmp_path):
    fasta = tmp_path / "toy.fna"
    fasta.write_text(">contig1\nGGCCGGCC\n>contig2\nAATTAATT\n")
    size, gc = genome_traits.genome_size_and_gc(fasta)
    assert size == 16
    assert gc == pytest.approx(0.5, abs=1e-9)


def test_arrhenius_correction_identity_at_reference_temp():
    cfg = {
        "temperature": {
            "reference_temp_k": 293.15,
            "activation_energy_j_per_mol": 65000.0,
            "gas_constant_j_per_mol_k": 8.314,
        }
    }
    t_ref_c = 293.15 - 273.15
    corrected = temperature.correct_doubling_time(10.0, t_ref_c, cfg)
    assert corrected == pytest.approx(10.0, rel=1e-6)


def test_arrhenius_correction_faster_growth_at_higher_temp():
    cfg = {
        "temperature": {
            "reference_temp_k": 293.15,
            "activation_energy_j_per_mol": 65000.0,
            "gas_constant_j_per_mol_k": 8.314,
        }
    }
    # Growth observed at a warmer temperature should be corrected DOWN toward
    # a slower reference-temperature rate -> longer doubling time.
    corrected = temperature.correct_doubling_time(5.0, 37.0, cfg)
    assert corrected > 5.0


def test_milc_distance_zero_for_identical_distributions():
    counts = {"GCT": 10, "GCC": 10, "GCA": 10, "GCG": 10}  # Ala family, uniform
    distance = cub.milc_distance(counts, counts)
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_milc_distance_positive_for_skewed_gene():
    reference = {"GCT": 25, "GCC": 25, "GCA": 25, "GCG": 25}
    skewed_gene = {"GCT": 40, "GCC": 2, "GCA": 2, "GCG": 2}
    distance = cub.milc_distance(skewed_gene, reference)
    assert distance > 0.0


def test_landmark_distance_embedding_matches_known_tree_distances():
    # (A:1,(B:1,C:1):1);  -> A-B patristic distance = 1+1+1=3, B-C = 1+1=2
    newick = "(A:1,(B:1,C:1):1);"
    tree = dendropy.Tree.get(data=newick, schema="newick")
    emb = phylogeny.landmark_distance_embedding(tree, n_landmarks=3, seed=0)
    assert set(emb.keys()) == {"A", "B", "C"}
    for vec in emb.values():
        assert vec.shape == (3,)
        assert (vec >= 0).all()
    # every tip is one of the 3 landmarks here, so each embedding contains a 0
    # (distance to itself) among its 3 entries
    for vec in emb.values():
        assert np.isclose(vec.min(), 0.0)


def test_classical_mds_embedding_shape():
    n = 6
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(n, 3))
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    emb = phylogeny.classical_mds_embedding(dist, n_components=4, seed=0)
    assert emb.shape == (n, 4)


def test_16s_embeddings_from_profiles_match_from_sequences():
    # Building from precomputed profiles should give the same distances (and
    # therefore the same embedding, up to MDS's own randomness) as building
    # from the raw sequences directly, since it's the same underlying math.
    sequences = {
        "a": "ACGTACGTACGTACGTAAAA",
        "b": "ACGTACGTACGTACGTTTTT",
        "c": "GGGGCCCCGGGGCCCCGGGG",
        "d": "GGGGCCCCGGGGCCCCAAAA",
    }
    k = 4
    profiles = {label: rrna16s.kmer_profile(seq, k) for label, seq in sequences.items()}

    mat_from_seq, labels_seq = rrna16s.build_16s_distance_matrix(sequences, k)
    mat_from_profiles, labels_profiles = rrna16s.build_16s_distance_matrix_from_profiles(profiles)

    assert labels_seq == labels_profiles
    assert np.allclose(mat_from_seq, mat_from_profiles)


def test_16s_embeddings_from_profiles_shape():
    profiles = {
        "a": {"AAAA": 3, "CCCC": 1},
        "b": {"AAAA": 1, "CCCC": 3},
        "c": {"GGGG": 4},
    }
    cfg = {"rrna16s": {"kmer_k": 4, "embedding_dim": 5, "method": "classical_mds"}}
    embeddings = rrna16s.build_16s_embeddings_from_profiles(profiles, cfg)
    assert set(embeddings.keys()) == {"a", "b", "c"}
    for vec in embeddings.values():
        assert vec.shape == (5,)
