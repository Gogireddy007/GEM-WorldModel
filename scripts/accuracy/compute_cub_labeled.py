#!/usr/bin/env python
"""This computes codon usage bias for the labeled genomes with the annotation-free
pipeline (pyrodigal genes, Pfam ribosomal protein HMMs, MILC), the same method
that is applied to GEM MAGs, and compares it with the annotation-based value
already in the corpus.

Genomes come from the cached NCBI downloads. Writes labeled_cub_pipeline.csv.
"""

from multiprocessing import Pool

import numpy as np
import pandas as pd

from gem_worldmodel.features import cub_pipeline
from gem_worldmodel.utils.config import load_config, resolve_path

HMM = "data/raw/ribosomal_pfam.hmm"


def work(acc):
    path = cub_pipeline.genome_fasta(f"data/raw/ncbi_genomes/{acc}")
    if path is None:
        return {"accession": acc, "cub_pipeline": np.nan, "ivywrel": np.nan, "n_genes": np.nan, "n_ribosomal": np.nan}
    try:
        r = cub_pipeline.genome_cub(path, HMM)
    except Exception as exc:  # noqa: BLE001
        print(f"{acc}: {exc}")
        return {"accession": acc, "cub_pipeline": np.nan, "ivywrel": np.nan, "n_genes": np.nan, "n_ribosomal": np.nan}
    return {"accession": acc, "cub_pipeline": r["cub"], "ivywrel": r["ivywrel"], "n_genes": r["n_genes"], "n_ribosomal": r["n_ribosomal"]}


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    lab = pd.read_csv(processed / "features_sample_expanded.csv")
    accs = lab["accession"].dropna().unique().tolist()
    with Pool(8) as pool:
        rows = pool.map(work, accs)
    out = pd.DataFrame(rows)
    out.to_csv(processed / "labeled_cub_pipeline.csv", index=False)
    m = lab[["accession", "species", "cub"]].merge(out, on="accession")
    ok = m.dropna(subset=["cub", "cub_pipeline"])
    print(f"{len(ok)} of {len(accs)} genomes with both values; ribosomal proteins found: median {out.n_ribosomal.median():.0f}, "
          f"min {out.n_ribosomal.min():.0f}, share with fewer than 10: {100 * (out.n_ribosomal < 10).mean():.1f}%")
    print(f"pipeline CUB vs annotation-based CUB: Pearson {ok.cub.corr(ok.cub_pipeline):.3f}, Spearman {ok.cub.corr(ok.cub_pipeline, method='spearman'):.3f}, "
          f"median absolute difference {np.median(np.abs(ok.cub - ok.cub_pipeline)):.3f} (CUB spans {ok.cub.min():.2f} to {ok.cub.max():.2f})")
    print(f"slope of pipeline on annotation-based: {np.polyfit(ok.cub, ok.cub_pipeline, 1)[0]:.3f}")


if __name__ == "__main__":
    main()
