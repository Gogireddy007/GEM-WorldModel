"""Training table and feature matrix for the deployment model.

One place that defines the features, so the evaluation scripts and the
prediction script cannot drift apart: genome size, GC, tRNA count, the GTDB
landmark phylogeny vector, 16 distances to fixed 16S reference sequences, and a
flag for archaea.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from gem_worldmodel.features import lifestyle_genes, rrna16s
from gem_worldmodel.features import taxon_vectors as tv
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

TRAITS = ["genome_size_bp", "gc_content", "trna_count"]
KEEP = ["species", "gtdb_taxonomy", "doubling_time_hours", "doubling_time_hours_ref", "growth_temp_c"] + TRAITS
# everything computed from the predicted proteins; a genome has all or none
# the released gRodon's codon statistics (scripts/r/grodon_stats.R): codon usage bias of the highly
# expressed genes, its consistency, codon pair bias, and gRodon's own doubling time (log); they beat our
# single MILC value with every level of novelty held out (evaluate_grodon_and_temperature.py)
GRODON_FEATURES = ["CUBHE", "ConsistencyHE", "CPB", "log_grodon_d"]
GENE_FEATURES = ["cub", "ivywrel"] + lifestyle_genes.FEATURES + GRODON_FEATURES
EXCLUSIONS = Path(__file__).resolve().parents[3] / "data/curated/label_exclusions.csv"
GENOME_ROWS = {"madin_genome_extension.csv": "Madin genome extension", "literature_extension.csv": "literature"}


def species_group(df):
    """Group label joining rows that share a species name or a GTDB species.

    Held-out splits group on this, so two names for one GTDB species (a
    renamed species, a synonym) and one name placed in two GTDB species (the
    same measured value attached to different strains) always land on the same
    side of a split. Rows are linked through either key and the connected
    groups are labeled by their first species name.
    """
    gtdb = df["gtdb_taxonomy"].astype(str).str.extract(r"s__([^;]+)")[0]
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for name, g in zip(df["species"], gtdb, strict=True):
        if isinstance(g, str):
            parent[find(("name", name))] = find(("gtdb", g))
    roots = [find(("name", n)) for n in df["species"]]
    label = {}
    for root, name in zip(roots, df["species"], strict=True):
        label.setdefault(root, name)
    return pd.Series([label[r] for r in roots], index=df.index)


def load_training_table(processed, kmer_k: int, with_cub: bool = False, genome_rows=True,
                        with_lifestyle: bool = False, with_energy: bool = False):
    """Return (table, X, names). Rows are the original corpus plus the Madin extension.

    With `genome_rows=True` (the default) the rows whose traits were all
    computed from the genome of the measured strain are added too: the Madin
    species with a genome accession but no traits in Madin's table, and the
    hand-checked literature rows (data/curated/literature_growth_rates.csv).
    With `with_lifestyle=True` the gene features also include the lifestyle
    genes and the released gRodon's statistics (GRODON_FEATURES; median-filled
    where missing, `has_grodon` marks real values).
    `genome_rows` can also be a collection of origin names from GENOME_ROWS
    ("Madin genome extension", "literature") to add only those.
    The table has a `species_group` column (see species_group) for splits.
    Species listed in data/curated/label_exclusions.csv are dropped.

    Rows whose taxonomy cannot be placed in GTDB are dropped. Missing traits or
    16S values are filled with the median of the bacterial rows. With
    `with_cub=True` two final columns hold the codon usage bias and the
    IVYWREL amino acid fraction from the annotation-free gene pipeline
    (labeled_cub_pipeline.csv for the original corpus, extension_cub.csv for
    the extension), median-filled where missing, and the table gets a
    `has_cub` column marking rows where both are real.
    """
    base = pd.read_csv(processed / "features_sample_expanded.csv")
    ext = pd.read_csv(processed / "madin_extension.csv")
    base["origin"] = "original"
    ext["origin"] = np.where(ext["gtdb_taxonomy"].str.startswith("d__Archaea"), "archaeal extension", "bacterial extension")
    ext_acc = pd.read_csv(processed / "extension_cub.csv").set_index("species")["accession"]
    ext["accession"] = ext["species"].map(ext_acc)
    parts = [base[KEEP + ["origin", "accession"]], ext[KEEP + ["origin", "accession"]]]
    if genome_rows:
        wanted = set(GENOME_ROWS.values()) if genome_rows is True else set(genome_rows)
        for name, origin in GENOME_ROWS.items():
            if origin in wanted and (processed / name).exists():
                extra = pd.read_csv(processed / name)
                extra["origin"] = origin
                parts.append(extra[KEEP + ["origin", "accession", "cub", "ivywrel"]].rename(columns={"cub": "own_cub", "ivywrel": "own_ivywrel"}))
    df = pd.concat(parts, ignore_index=True)
    if EXCLUSIONS.exists():
        # growth rates that are not maximal growth at the stated temperature, see data/curated/README.md
        df = df[~df["species"].isin(set(pd.read_csv(EXCLUSIONS)["species"]))].reset_index(drop=True)
    df["species_group"] = species_group(df)
    df["archaeon"] = df["gtdb_taxonomy"].map(tv.is_archaeon)

    table = tv.load_taxon_table(processed / "gtdb_taxon_vectors.csv")
    lm = np.vstack([
        (tv.lookup(t, table, "family")[0] if tv.lookup(t, table, "family")[0] is not None else np.full(16, np.nan))
        for t in df["gtdb_taxonomy"]
    ])

    seqs = pd.read_csv(processed / "labeled_16s_sequences.csv").fillna("")
    profiles = {s: rrna16s.kmer_profile(q, kmer_k) for s, q in zip(seqs.species, seqs.sequence) if q}
    landmarks = [profiles[n] for n in np.load(processed / "labeled_16s_landmark_names.npy")]
    s16 = np.vstack([
        np.array([rrna16s.kmer_cosine_distance(profiles[s], lp) for lp in landmarks]) if s in profiles else np.full(16, np.nan)
        for s in df["species"]
    ])

    traits = df[TRAITS].to_numpy(float)
    bacterial = (df["archaeon"] == 0).to_numpy()
    for block in (traits, s16):
        block[:] = np.where(np.isnan(block), np.nanmedian(block[bacterial], axis=0), block)

    keep = ~np.isnan(lm).any(axis=1)
    X = np.hstack([traits, lm, s16, df[["archaeon"]].to_numpy(float)])
    names = TRAITS + [f"lm_{i}" for i in range(16)] + [f"s16_{i}" for i in range(16)] + ["archaeon"]
    if with_cub:
        orig = pd.read_csv(processed / "labeled_cub_pipeline.csv").set_index("accession")
        extn = pd.read_csv(processed / "extension_cub.csv").set_index("species")
        for out_col, orig_col, ext_col in (("cub", "cub_pipeline", "cub"), ("ivywrel", "ivywrel", "ivywrel")):
            vals = np.where(df["origin"] == "original", df["accession"].map(orig[orig_col]), df["species"].map(extn[ext_col])).astype(float)
            if f"own_{out_col}" in df:
                vals = np.where(df["origin"].isin(GENOME_ROWS.values()), df[f"own_{out_col}"], vals).astype(float)
            df[out_col] = vals
        df["has_cub"] = df["cub"].notna() & df["ivywrel"].notna()
        for col in ("cub", "ivywrel"):
            vals = df[col].to_numpy(float)
            X = np.hstack([X, np.where(np.isnan(vals), np.nanmedian(vals), vals)[:, None]])
            names = names + [col]
    if with_lifestyle:
        counts = pd.read_csv(processed / "labeled_lifestyle_genes.csv").drop_duplicates("accession").set_index("accession")
        table = counts.reindex(df["accession"]).reset_index(drop=True)
        feats = lifestyle_genes.lifestyle_features(table)
        df["has_lifestyle"] = table["n_genes"].notna().to_numpy()
        if with_cub:
            df["has_cub"] = df["has_cub"] & df["has_lifestyle"]
        med = np.nanmedian(feats[df["has_lifestyle"].to_numpy()], axis=0)
        X = np.hstack([X, np.where(np.isnan(feats), med, feats)])
        names = names + lifestyle_genes.FEATURES
        stats = pd.read_csv(processed / "labeled_grodon_stats.csv").drop_duplicates("genome_name").set_index("genome_name")
        keys = df["accession"].map(lambda a: ("RS_" if str(a).startswith("GCF_") else "GB_") + str(a))
        g = grodon_matrix(stats.reindex(keys).reset_index(drop=True))
        df["has_grodon"] = ~np.isnan(g).any(axis=1)
        X = np.hstack([X, np.where(np.isnan(g), np.nanmedian(g, axis=0), g)])
        names = names + GRODON_FEATURES
        if with_energy:
            energy = lifestyle_genes.energy_features(table)
            med = np.nanmedian(energy[df["has_lifestyle"].to_numpy()], axis=0)
            X = np.hstack([X, np.where(np.isnan(energy), med, energy)])
            names = names + lifestyle_genes.ENERGY_FEATURES
    return df[keep].reset_index(drop=True), X[keep], names


def grodon_matrix(stats) -> np.ndarray:
    """GRODON_FEATURES from a table with gRodon's CUBHE, ConsistencyHE, CPB and grodon_d (NaN where missing)."""
    return np.column_stack([stats["CUBHE"].to_numpy(float), stats["ConsistencyHE"].to_numpy(float),
                            stats["CPB"].to_numpy(float), np.log(stats["grodon_d"].to_numpy(float))])


def gene_feature_matrix(table) -> np.ndarray:
    """GENE_FEATURES for new genomes, from a table with cub, ivywrel, the lifestyle family counts and gRodon's statistics."""
    return np.hstack([table[["cub", "ivywrel"]].to_numpy(float), lifestyle_genes.lifestyle_features(table), grodon_matrix(table)])


def coarsen(taxonomy, rank):
    """Blank out ranks finer than `rank` so a lookup lands on that rank."""
    t = str(taxonomy)
    if rank in ("genus", "family", "order"):
        t = re.sub(r"s__[^;]*", "s__", t)
    if rank in ("family", "order"):
        t = re.sub(r"g__[^;]*", "g__", t)
    if rank == "order":
        t = re.sub(r"f__[^;]*", "f__", t)
    return t


def combo_label(lm, s16, cub):
    """Name of a feature combination, the same string the accuracy table uses."""
    return "+".join(["traits"] + (["phylogeny"] if lm else []) + (["16S"] if s16 else []) + (["gene features"] if cub else []))


def drift_report(train_blocks: dict, new_blocks: dict, limit_sd: float = 3.0) -> dict:
    """Largest mean shift of each input block, in training standard deviations.

    Raises if a phylogeny or 16S block is out of the training range, since those
    are the blocks that depend on a coordinate system, which is how the first
    GEM predictions went wrong.
    """
    worst = {}
    for block in train_blocks:
        a, b = np.asarray(train_blocks[block], float), np.asarray(new_blocks[block], float)
        sd = np.nanstd(a, axis=0)
        sd[sd == 0] = 1.0
        worst[block] = float(np.nanmax(np.abs(np.nanmean(b, axis=0) - np.nanmean(a, axis=0)) / sd))
        logger.info(f"input drift, {block}: largest column mean shift {worst[block]:.2f} training SD")
    bad = [k for k in ("phylogeny", "16S") if worst.get(k, 0) > limit_sd]
    if bad:
        raise RuntimeError(f"inputs out of the training range ({bad}, limit {limit_sd} SD): {worst}")
    return worst
