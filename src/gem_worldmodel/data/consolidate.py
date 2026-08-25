"""Week 1 deliverable: cross-reference genome + growth-rate + GTDB placement.

Produces two tables under data/processed/:
  - labeled_corpus.csv:   accessions with a growth-rate label AND a GTDB tree
                           placement (used for supervised fine-tuning/benchmark)
  - unlabeled_corpus.csv: GEM MAGs used as extra pretraining-only data (Week 4)

This determines the final *usable* sample size, not every gRodon/Madin
accession has a GTDB placement, and that gap is exactly what this step is
for surfacing.
"""

import pandas as pd

from gem_worldmodel.data import gem_portal, grodon, gtdb
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def build_labeled_corpus(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config("data")

    growth_corpus = grodon.build_labeled_corpus(cfg)
    taxonomy = gtdb.fetch_bac_taxonomy(cfg)

    merged = growth_corpus.merge(
        taxonomy[["accession_bare", "gtdb_taxonomy"]],
        left_on="accession",
        right_on="accession_bare",
        how="left",
    )
    merged["has_gtdb_taxonomy"] = merged["gtdb_taxonomy"].notna()
    merged = merged.drop(columns=["accession_bare"])

    # has_gtdb_taxonomy != in_gtdb_tree: the taxonomy table lists every genome
    # GTDB has classified (~879k), but the reference tree itself only has
    # representative genomes as tips (~190k), most classified genomes are
    # NOT tree tips and can't get a real patristic-distance embedding. Check
    # against actual tree tips, not just taxonomy-table membership.
    tree_tips = gtdb.get_tree_tip_accessions(cfg)
    merged["in_gtdb_tree"] = merged["accession"].isin(tree_tips)

    n_total = len(merged)
    n_taxonomy = int(merged["has_gtdb_taxonomy"].sum())
    n_placed = int(merged["in_gtdb_tree"].sum())
    logger.info(
        f"labeled corpus: {n_total} growth-rate accessions, "
        f"{n_taxonomy} ({n_taxonomy / n_total:.1%}) have GTDB taxonomy, "
        f"{n_placed} ({n_placed / n_total:.1%}) are ACTUAL tree tips (usable for phylogeny embedding)"
    )

    out_path = resolve_path(cfg["paths"]["processed_dir"]) / "labeled_corpus.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}")
    return merged


def build_unlabeled_corpus(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config("data")
    mags = gem_portal.select_unlabeled_pretrain_genomes(cfg)

    out_path = resolve_path(cfg["paths"]["processed_dir"]) / "unlabeled_corpus.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mags.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path} ({len(mags)} rows)")
    return mags


def run(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config("data")
    labeled = build_labeled_corpus(cfg)
    unlabeled = build_unlabeled_corpus(cfg)
    return {"labeled": labeled, "unlabeled": unlabeled}
