#!/usr/bin/env python
"""Assembles the archive to deposit on Zenodo with the paper, in dist/zenodo/. Nothing is uploaded.

  code.zip           the repository as it stands (tracked and new files, .gitignore respected)
  data.zip           unzipped inside the repository, fills data/processed and data/raw with the derived tables the models
                     are trained and evaluated on (so every result can be rerun without downloading genomes), the result
                     files behind every number, table and figure of the paper, and the few raw files the paper scripts read
  release/           the GEM prediction table and its column description
  paper/             main.pdf and supplement.pdf
  MANIFEST.tsv       size and SHA-256 of every file

A listed file that is missing stops the script. Run from the repository root after `make paper`:
.venv/bin/python paper/scripts/make_archive.py
"""

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dist" / "zenodo"
PROCESSED = ROOT / "data" / "processed"

# read by deploy_features.load_training_table and predict_gem_genomes_gbm.py
INPUTS = ["features_sample_expanded.csv", "madin_extension.csv", "madin_genome_extension.csv", "literature_extension.csv",
          "extension_cub.csv", "labeled_cub_pipeline.csv", "labeled_lifestyle_genes.csv", "labeled_grodon_stats.csv",
          "labeled_16s_sequences.csv", "labeled_16s_from_genomes.csv", "labeled_16s_landmark_names.npy", "gtdb_taxon_vectors.csv",
          "hq_genome_ids_all.csv", "mag_validation_ids.csv", "gem_cub.csv", "gem_grodon_stats.csv", "gem_lifestyle_genes.csv",
          "unlabeled_corpus_features.csv", "unlabeled_corpus_features_enriched.csv", "unlabeled_corpus_features_16s.csv"]
RAW_INPUTS = ["gem_genome_metadata.tsv", "ribosomal_pfam.hmm", "lifestyle_pfam.hmm", "lifestyle_pfam_evalue.hmm", "GrowthRates_Madin.rda"]

# read by paper/scripts (numbers, tables, figures)
RESULTS = ["paper_headline_accuracy.csv", "paper_headline_predictions.csv", "paper_headline_settings.csv", "final_accuracy_by_tier.csv",
           "released_tools_comparison.csv", "released_tools_comparison_nested.csv", "ablation_evaluation.csv", "tuning_evaluation.csv",
           "16s_source_evaluation.csv", "audit_controls.csv", "label_sources_studies.csv", "mag_validation.csv", "independent_test.csv",
           "lifestyle_phylogenetic_test.csv", "lifestyle_phylogenetic_checks.txt", "syntrophy_marker.csv", "hq_predictions_gbm.csv",
           "pressure_tests.csv", "pressure_tests_round2.csv", "released_tools_genomes_nested.csv", "range_calibration.csv", "range_evaluation.csv",
           "esm_genome_embeddings.csv", "esm_genome_embeddings_cpu.csv", "esm_baseline.csv", "esm_baseline_genomes.csv", "released_tools_predictions.csv", "released_tools_folds.csv",
           "released_tools_comparison_nested_order.csv", "released_tools_genomes_nested_order.csv", "released_tools_order_paired.csv", "phydon_rep_orders.csv",
           "external_test.csv", "external_summary.csv", "external_extension.csv"]


def copy(src, dst):
    if not src.exists():
        sys.exit(f"missing: {src.relative_to(ROOT)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.splitlines()
    with zipfile.ZipFile(OUT / "code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if (ROOT / f).is_file() and not f.startswith("dist/"):
                z.write(ROOT / f, f"GEM-WorldModel/{f}")
    data = [(PROCESSED / n, f"data/processed/{n}") for n in INPUTS + RESULTS] + [(ROOT / "data" / "raw" / n, f"data/raw/{n}") for n in RAW_INPUTS]
    with zipfile.ZipFile(OUT / "data.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in data:
            if not src.exists():
                sys.exit(f"missing: {src.relative_to(ROOT)}")
            z.write(src, arc)
    for name in ("gem_growth_predictions.csv", "COLUMNS.md"):
        copy(ROOT / "release" / name, OUT / "release" / name)
    for name in ("main.pdf", "supplement.pdf"):
        copy(ROOT / "paper" / name, OUT / "paper" / name)

    rows = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        rows.append(f"{path.relative_to(OUT)}\t{path.stat().st_size}\t{hashlib.sha256(path.read_bytes()).hexdigest()}")
    (OUT / "MANIFEST.tsv").write_text("file\tbytes\tsha256\n" + "\n".join(rows) + "\n")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} files, {len(files)} in code.zip, {len(data)} in data.zip, {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
