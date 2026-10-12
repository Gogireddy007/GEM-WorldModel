"""Training table and feature matrix for the deployment model.

One place that defines the features, so the evaluation scripts and the
prediction script cannot drift apart: genome size, GC, tRNA count, the GTDB
landmark phylogeny vector, 16 distances to fixed 16S reference sequences, and a
flag for archaea.
"""

import re

import numpy as np
import pandas as pd

from gem_worldmodel.features import rrna16s
from gem_worldmodel.features import taxon_vectors as tv
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

TRAITS = ["genome_size_bp", "gc_content", "trna_count"]
KEEP = ["species", "gtdb_taxonomy", "doubling_time_hours", "doubling_time_hours_ref", "growth_temp_c"] + TRAITS


def load_training_table(processed, kmer_k: int, with_cub: bool = False):
    """Return (table, X, names). Rows are the original corpus plus the Madin extension.

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
    df = pd.concat([base[KEEP + ["origin", "accession"]], ext[KEEP + ["origin"]]], ignore_index=True)
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
            df[out_col] = vals
        df["has_cub"] = df["cub"].notna() & df["ivywrel"].notna()
        for col in ("cub", "ivywrel"):
            vals = df[col].to_numpy(float)
            X = np.hstack([X, np.where(np.isnan(vals), np.nanmedian(vals), vals)[:, None]])
            names = names + [col]
    return df[keep].reset_index(drop=True), X[keep], names


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
