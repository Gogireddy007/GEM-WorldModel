#!/usr/bin/env python
"""This pulls the downloaded inputs that every later step reads, at the versions the paper used (configs/data.yaml):
gRodon's growth-rate tables (gRodon2 at a fixed commit), GTDB release 232 trees and taxonomy for bacteria and archaea,
Madin and colleagues' trait tables (fixed commit), the GEM catalog metadata and tree, and Pfam 38.2. It then
cross-references the growth rates with GTDB and prints the validation report (species count, growth-rate coverage,
doubling-time distribution above/below the 5h split). Genome assemblies are downloaded later, by the steps that use them.
`make verify-inputs` checks every downloaded file against data/input_checksums.tsv.
"""

import subprocess
import sys
from pathlib import Path

from gem_worldmodel.data import consolidate, gem_portal, gem_tree, gtdb, madin_traits, validate
from gem_worldmodel.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def main():
    cfg = load_config("data")
    gtdb.fetch_bac_tree(cfg)
    gtdb.fetch_arc_tree(cfg)
    gtdb.fetch_arc_taxonomy(cfg)
    gem_portal.fetch_genome_metadata(cfg)
    gem_tree.fetch_gem_tree(cfg)
    madin_traits.fetch_madin_traits(cfg)
    sys.path.insert(0, str(ROOT / "scripts" / "accuracy"))
    import screen_labels  # noqa: E402

    screen_labels.source_records()  # Madin's per-source tables
    subprocess.run([sys.executable, str(ROOT / "scripts" / "accuracy" / "build_ribosomal_hmms.py")], check=True, cwd=ROOT)  # Pfam 38.2
    result = consolidate.run(cfg)
    validate.validate_labeled_corpus(result["labeled"], cfg)


if __name__ == "__main__":
    main()
