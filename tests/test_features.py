import dendropy
import numpy as np
import pandas as pd
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


def test_genus_centroid_embeddings_averages_same_genus_tips():
    tip_embeddings = {
        "tipA1": np.array([0.0, 0.0]),
        "tipA2": np.array([2.0, 0.0]),
        "tipB1": np.array([10.0, 10.0]),
    }
    tip_taxonomy = {
        "tipA1": "d__Bacteria;p__X;c__X;o__X;f__X;g__Foo;s__Foo alpha",
        "tipA2": "d__Bacteria;p__X;c__X;o__X;f__X;g__Foo;s__Foo beta",
        "tipB1": "d__Bacteria;p__X;c__X;o__X;f__X;g__Bar;s__Bar alpha",
    }
    non_tip_rows = pd.DataFrame(
        [
            {"accession": "new1", "gtdb_taxonomy": "d__Bacteria;p__X;c__X;o__X;f__X;g__Foo;s__Foo gamma"},
            {"accession": "new2", "gtdb_taxonomy": "d__Bacteria;p__X;c__X;o__X;f__X;g__Baz;s__Baz alpha"},
        ]
    )
    embeddings, matched_genus = phylogeny.genus_centroid_embeddings(non_tip_rows, tip_embeddings, tip_taxonomy)
    # new1 is genus Foo, matches tipA1+tipA2 -> centroid (1.0, 0.0)
    assert np.allclose(embeddings["new1"], [1.0, 0.0])
    assert matched_genus["new1"] == "g__Foo"
    # new2 is genus Baz, no tip has that genus -> no entry, not silently dropped-to-zero
    assert "new2" not in embeddings
    assert "new2" not in matched_genus.index


def test_taxonomic_centroid_embeddings_falls_back_to_family_when_genus_missing():
    tip_embeddings = {
        "tipA1": np.array([0.0, 0.0]),
        "tipA2": np.array([2.0, 0.0]),
        "tipB1": np.array([10.0, 10.0]),
    }
    tip_taxonomy = {
        "tipA1": "d__Bacteria;p__X;c__X;o__X;f__Fooidae;g__Foo;s__Foo alpha",
        "tipA2": "d__Bacteria;p__X;c__X;o__X;f__Fooidae;g__Foo;s__Foo beta",
        "tipB1": "d__Bacteria;p__X;c__X;o__X;f__Baridae;g__Bar;s__Bar alpha",
    }
    non_tip_rows = pd.DataFrame(
        [
            # exact genus match still wins over the family fallback
            {"accession": "new1", "gtdb_taxonomy": "d__Bacteria;p__X;c__X;o__X;f__Fooidae;g__Foo;s__Foo gamma"},
            # no genus Baz anywhere, but its family Fooidae matches tipA1+tipA2
            {"accession": "new2", "gtdb_taxonomy": "d__Bacteria;p__X;c__X;o__X;f__Fooidae;g__Baz;s__Baz alpha"},
            # neither genus nor family has any embedded tip -> no entry at all
            {"accession": "new3", "gtdb_taxonomy": "d__Bacteria;p__X;c__X;o__X;f__Nowhere;g__Nada;s__Nada alpha"},
        ]
    )
    embeddings, matched_rank = phylogeny.taxonomic_centroid_embeddings(non_tip_rows, tip_embeddings, tip_taxonomy)
    assert np.allclose(embeddings["new1"], [1.0, 0.0])
    assert matched_rank["new1"] == "g__Foo"
    assert np.allclose(embeddings["new2"], [1.0, 0.0])
    assert matched_rank["new2"] == "f__Fooidae"
    assert "new3" not in embeddings


def test_recompute_centroid_rows_follows_new_tip_embeddings():
    tip_taxonomy = {
        "tipA1": "d__B;p__X;c__X;o__X;f__Fooidae;g__Foo;s__Foo alpha",
        "tipA2": "d__B;p__X;c__X;o__X;f__Fooidae;g__Foo;s__Foo beta",
        "tipB1": "d__B;p__X;c__X;o__X;f__Baridae;g__Bar;s__Bar alpha",
    }
    rows = pd.DataFrame(
        {"matched_rank": ["g__Foo", "f__Baridae"], "e0": [99.0, 99.0], "e1": [99.0, 99.0]},
        index=["r1", "r2"],
    )
    # the tips moved to a new coordinate system, the stale values must not survive
    new_tips = {"tipA1": np.array([4.0, 4.0]), "tipA2": np.array([6.0, 4.0]), "tipB1": np.array([-3.0, 7.0])}
    out = phylogeny.recompute_centroid_rows(rows, new_tips, tip_taxonomy, ["e0", "e1"])
    assert np.allclose(out.loc["r1", ["e0", "e1"]].astype(float), [5.0, 4.0])
    assert np.allclose(out.loc["r2", ["e0", "e1"]].astype(float), [-3.0, 7.0])


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


def test_taxon_vector_lookup_prefers_most_specific_rank():
    from gem_worldmodel.features import taxon_vectors as tv

    table = {
        ("species", "s__Foo alpha"): np.full(16, 1.0),
        ("genus", "g__Foo"): np.full(16, 2.0),
        ("family", "f__Fooidae"): np.full(16, 3.0),
    }
    v, rank = tv.lookup("d__B;f__Fooidae;g__Foo;s__Foo alpha", table)
    assert rank == "species" and v[0] == 1.0
    v, rank = tv.lookup("d__B;f__Fooidae;g__Foo;s__Foo beta", table)
    assert rank == "genus" and v[0] == 2.0
    v, rank = tv.lookup("d__B;f__Fooidae;g__Bar;s__Bar x", table)
    assert rank == "family" and v[0] == 3.0
    v, rank = tv.lookup("d__B;f__Nowhere;g__Nada;s__Nada x", table)
    assert v is None and rank is None


def test_is_archaeon_flag():
    from gem_worldmodel.features import taxon_vectors as tv

    assert tv.is_archaeon("d__Archaea;p__Halobacteriota;c__Methanosarcinia") == 1.0
    assert tv.is_archaeon("d__Bacteria;p__Bacillota;c__Bacilli") == 0.0
    assert tv.is_archaeon(float("nan")) == 0.0


def test_coarsen_blanks_finer_ranks():
    from gem_worldmodel.features.deploy_features import coarsen

    t = "d__B;p__X;c__X;o__Ord;f__Fam;g__Gen;s__Gen sp"
    assert "g__Gen" in coarsen(t, "genus") and "s__Gen sp" not in coarsen(t, "genus")
    assert "g__Gen" not in coarsen(t, "family") and "f__Fam" in coarsen(t, "family")
    assert "f__Fam" not in coarsen(t, "order") and "o__Ord" in coarsen(t, "order")


def test_combo_label_matches_accuracy_table_names():
    from gem_worldmodel.features.deploy_features import combo_label

    assert combo_label(True, True, True) == "traits+phylogeny+16S+gene features"
    assert combo_label(False, False, False) == "traits"
    assert combo_label(True, False, True) == "traits+phylogeny+gene features"


def test_drift_report_stops_inputs_from_a_different_coordinate_system():
    import numpy as np
    import pytest

    from gem_worldmodel.features.deploy_features import drift_report

    rng = np.random.default_rng(0)
    train = {"phylogeny": rng.normal(0.0, 0.2, size=(200, 16)), "16S": rng.normal(0.5, 0.1, size=(200, 16))}
    similar = {"phylogeny": rng.normal(0.02, 0.2, size=(300, 16)), "16S": rng.normal(0.5, 0.1, size=(300, 16))}
    assert drift_report(train, similar)["phylogeny"] < 1.0
    # the original failure: landmark distances centered near 2 where training coordinates were centered at 0
    shifted = {"phylogeny": rng.normal(2.0, 0.4, size=(300, 16)), "16S": rng.normal(0.5, 0.1, size=(300, 16))}
    with pytest.raises(RuntimeError):
        drift_report(train, shifted)


def test_phydon_style_uses_relatives_when_close_and_codon_usage_when_far():
    import numpy as np

    from gem_worldmodel.training.baselines import PhydonStyleBaseline

    rng = np.random.default_rng(0)
    # two well-separated clades; growth follows the clade, codon usage is only loosely related
    phylo = np.vstack([rng.normal(0, 0.05, size=(30, 2)), rng.normal(5, 0.05, size=(30, 2))])
    y = np.concatenate([np.full(30, 0.0), np.full(30, 3.0)]) + rng.normal(0, 0.05, 60)
    cub = y * 0.1 + rng.normal(0, 0.3, 60)
    species = np.array([f"s{i}" for i in range(60)])
    model = PhydonStyleBaseline(k=3).fit(cub, phylo, np.zeros(60), species, y)
    # a new genome inside clade 2: the prediction should follow its relatives (about 3)
    pred = model.predict_log(np.array([0.0]), np.array([[5.0, 5.0]]), np.zeros(1), np.array(["new"]))
    assert abs(pred[0] - 3.0) < 0.5


def test_genome_fasta_skips_cds_and_rna_files(tmp_path):
    from gem_worldmodel.features.cub_pipeline import genome_fasta

    for name in ("GCF_1_ASM1_cds_from_genomic.fna", "GCF_1_ASM1_rna_from_genomic.fna", "GCF_1_ASM1_genomic.fna"):
        (tmp_path / name).write_text(">x\nACGT\n")
    assert genome_fasta(tmp_path).name == "GCF_1_ASM1_genomic.fna"


def test_species_group_joins_synonyms_and_split_names():
    import pandas as pd

    from gem_worldmodel.features.deploy_features import species_group

    df = pd.DataFrame({
        "species": ["Microcystis aeruginosa", "Microcystis viridis", "Yersinia pestis", "Yersinia pestis", "Yersinia pseudotuberculosis", "Lonely one"],
        "gtdb_taxonomy": ["g__Microcystis;s__Microcystis aeruginosa", "g__Microcystis;s__Microcystis aeruginosa",
                          "g__Yersinia;s__Yersinia pestis", "g__Yersinia;s__Yersinia pseudotuberculosis",
                          "g__Yersinia;s__Yersinia pseudotuberculosis", "g__Lonely;s__"],
    })
    g = species_group(df).tolist()
    assert g[0] == g[1]
    assert g[2] == g[3] == g[4]
    assert g[5] == "Lonely one"
    assert len(set(g)) == 3


def test_lifestyle_features_rules():
    import pandas as pd

    from gem_worldmodel.features import lifestyle_genes as lg

    base = {acc: 0 for acc in [*lg.FAMILIES, *lg.EVALUE_FAMILIES]}
    cyano_no_nif = {**base, "PF00124": 4, "PF00223": 2, "PF00148": 2, "PF00142": 1, "n_genes": 2000}
    cyano_nif = {**base, "PF00124": 4, "PF00223": 2, "PF00148": 6, "PF00142": 2, "n_genes": 2000}
    purple_no_nif = {**base, "PF00124": 2, "PF00148": 4, "PF00142": 4, "n_genes": 2000}
    purple_nif = {**base, "PF00124": 2, "PF00148": 8, "PF00142": 4, "n_genes": 2000}
    aoa = {**base, "PF04896": 1, "PF00005": 20, "PF00072": 10, "n_genes": 2000}
    rows = [cyano_no_nif, cyano_nif, purple_no_nif, purple_nif, aoa]
    f = pd.DataFrame(lg.lifestyle_features(pd.DataFrame(rows)), columns=lg.FEATURES)
    assert f["nitrogen_fixation"].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert f["phototrophy"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert f["ammonia_monooxygenase"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert f.loc[4, "transporters_per_1000"] == 10.0
    assert f.loc[4, "regulators_per_1000"] == 5.0


def test_clean_species_keeps_candidatus_epithet():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/accuracy/build_madin_genome_extension.py"
    spec = importlib.util.spec_from_file_location("bmge", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.clean_species("Candidatus Pelagibacter ubique HTCC1062") == "Candidatus Pelagibacter ubique"
    assert mod.clean_species("Candidatus Pelagibacter sp.") == "Candidatus Pelagibacter sp."
    assert mod.clean_species("[Clostridium] thermoalcaliphilum DSM 1") == "Clostridium thermoalcaliphilum"


def _phylo_test_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/accuracy/test_lifestyle_phylogenetic.py"
    spec = importlib.util.spec_from_file_location("tlp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_leaf_covariance_is_shared_path_length():
    import dendropy
    import numpy as np

    mod = _phylo_test_module()
    tree = dendropy.Tree.get(data="((A:1,B:2):3,C:4);", schema="newick")
    cov = mod.leaf_covariance(tree, ["A", "B", "C"])
    assert np.allclose(cov, [[4, 3, 0], [3, 5, 0], [0, 0, 4]])


def test_gls_with_identity_matches_ordinary_least_squares():
    import numpy as np

    mod = _phylo_test_module()
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(50), rng.normal(size=50)])
    y = X @ np.array([1.0, 2.0]) + rng.normal(scale=0.1, size=50)
    beta = mod.gls(y, X, np.eye(50))[0]
    assert np.allclose(beta, np.linalg.lstsq(X, y, rcond=None)[0])
    # with a star tree (no shared branches) lambda has nothing to act on and PGLS equals OLS
    beta_p = mod.pgls(y, X, np.eye(50))[0]
    assert np.allclose(beta_p, beta)


def test_label_exclusions_file_is_well_formed():
    from pathlib import Path

    import pandas as pd

    path = Path(__file__).resolve().parents[1] / "data/curated/label_exclusions.csv"
    ex = pd.read_csv(path)
    assert {"species", "label_hours", "reason", "source"} <= set(ex.columns)
    assert ex["species"].is_unique
    assert ex["reason"].str.len().gt(10).all() and ex["source"].str.len().gt(5).all()


def test_energy_feature_rules():
    import numpy as np
    import pandas as pd

    from gem_worldmodel.features import lifestyle_genes as lg

    base = {acc: 0 for acc in [*lg.FAMILIES, *lg.EVALUE_FAMILIES]}
    fermenter = {**base, "PF02901": 1, "PF00213": 1, "n_genes": 2000}
    respirer = {**base, "PF00115": 1, "PF14710": 1, "PF02901": 1, "n_genes": 4000}
    nqr_like_only = {**base, "PF05896": 1, "n_genes": 3000}
    acetogen = {**base, "PF03598": 1, "PF13375": 1, "n_genes": 3000}
    delta_only = {**base, "PF03599": 2, "n_genes": 3000}
    missing = {**base, "n_genes": np.nan}
    f = pd.DataFrame(lg.energy_features(pd.DataFrame([fermenter, respirer, nqr_like_only, acetogen, delta_only, missing])),
                     columns=lg.ENERGY_FEATURES)
    assert f["fermentation_only"].tolist()[:5] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert f.loc[1, "electron_acceptor_classes"] == 2.0 and f.loc[1, "nitrate_respiration"] == 1.0
    assert f.loc[2, "na_nqr"] == 0.0  # the NqrA barrel alone is not enough
    assert f.loc[3, "wood_ljungdahl"] == 1.0 and f.loc[4, "wood_ljungdahl"] == 0.0
    assert f.loc[5].isna().all()


def _script_module(name):
    import importlib.util
    import sys
    from pathlib import Path

    folder = Path(__file__).resolve().parents[1] / "scripts/accuracy"
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_label_source_study_flags():
    import numpy as np
    import pandas as pd

    mod = _script_module("screen_label_sources")
    comp = pd.DataFrame({
        "species": ["a", "b", "c", "d", "e", "f", "g"],
        "study": ["slow study"] * 3 + ["mixed study"] * 2 + ["fine study"] * 2,
        "log_ratio": [np.log(20), np.log(30), np.log(25), np.log(10), np.log(0.1), np.log(1.2), np.log(0.9)],
    })
    s = mod.flag_studies(comp, fold=3.0).set_index("study")
    assert s.loc["slow study", "flag"]  # consistently far off
    assert not s.loc["mixed study", "flag"]  # far off but in both directions
    assert not s.loc["fine study", "flag"]
    single = pd.DataFrame({"species": ["x"], "study": ["one label"], "log_ratio": [np.log(50)]})
    assert mod.flag_studies(single).empty  # a study needs at least two comparable labels


def test_false_slow_call_classification():
    import numpy as np

    mod = _script_module("false_slow_calls")
    pred = np.array([30.0, 30.0, 12.0, 12.0, 2.0])
    meas = np.array([40.0, 5.0, 2.0, 8.0, 1.0])
    called, slow, false_slow, over = mod.classify(pred, meas)
    assert false_slow.tolist() == [False, True, False, False, False]
    assert over.tolist() == [False, True, True, False, False]


def test_grodon_matrix_and_gene_feature_order():
    import numpy as np
    import pandas as pd

    from gem_worldmodel.features import deploy_features as d
    from gem_worldmodel.features import lifestyle_genes as lg

    stats = pd.DataFrame({"CUBHE": [0.6, np.nan], "ConsistencyHE": [0.5, 0.4], "CPB": [-0.3, -0.2], "grodon_d": [2.0, 4.0]})
    g = d.grodon_matrix(stats)
    assert g.shape == (2, len(d.GRODON_FEATURES))
    assert np.isclose(g[0, 3], np.log(2.0)) and np.isnan(g[1, 0])
    counts = {acc: 0 for acc in [*lg.FAMILIES, *lg.EVALUE_FAMILIES]}
    table = pd.DataFrame([{**counts, "n_genes": 1000, "cub": 0.3, "ivywrel": 0.4, "CUBHE": 0.6, "ConsistencyHE": 0.5, "CPB": -0.3, "grodon_d": 2.0}])
    m = d.gene_feature_matrix(table)
    assert m.shape == (1, len(d.GENE_FEATURES))
    assert d.GENE_FEATURES[-len(d.GRODON_FEATURES):] == d.GRODON_FEATURES
    assert np.isclose(m[0, -1], np.log(2.0))


def test_deployed_regressor_averages_its_two_members():
    import numpy as np

    from gem_worldmodel.models.deployed import DeployedRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=200)
    settings = {"hist": {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 5,
                         "l2_regularization": 0.0, "max_features": 1.0},
                "forest": {"min_samples_leaf": 3, "max_features": 0.5, "max_samples": None}}
    model = DeployedRegressor(seed=0, settings=settings).fit(X, y)
    pred = model.predict(X)
    members = [m.predict(X) for m in model.members_]
    assert np.allclose(pred, (members[0] + members[1]) / 2)
    assert np.corrcoef(pred, y)[0, 1] > 0.95
