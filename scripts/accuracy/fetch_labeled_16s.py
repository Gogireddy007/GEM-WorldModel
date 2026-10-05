#!/usr/bin/env python
"""This fetches and caches one 16S sequence per labeled species.

The original pipeline fetched these from NCBI by species name and only kept
the resulting embedding, so the sequences themselves were never saved. They
are needed to build a 16S embedding that is defined the same way for the
labeled species and for any new genome (distance to a fixed set of reference
sequences), instead of an MDS that depends on whichever batch it was run on.
"""

import pandas as pd

from gem_worldmodel.features import rrna16s
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    data_cfg = load_config("data")
    processed = resolve_path(data_cfg["paths"]["processed_dir"])
    out_path = processed / "labeled_16s_sequences.csv"
    labeled = pd.read_csv(processed / "features_sample_expanded.csv")
    species = labeled["species"].dropna().unique().tolist()

    done = {}
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = dict(zip(prev["species"], prev["sequence"]))
        logger.info(f"resuming, {len(done)} species already cached")

    for i, sp in enumerate(species):
        if sp in done:
            continue
        done[sp] = rrna16s.fetch_16s_sequence(sp) or ""
        if (i + 1) % 25 == 0:
            pd.DataFrame({"species": list(done), "sequence": list(done.values())}).to_csv(out_path, index=False)
            logger.info(f"{i + 1}/{len(species)} done, {sum(1 for v in done.values() if v)} with a sequence")

    pd.DataFrame({"species": list(done), "sequence": list(done.values())}).to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}: {sum(1 for v in done.values() if v)} of {len(species)} species have a 16S sequence")


if __name__ == "__main__":
    main()
