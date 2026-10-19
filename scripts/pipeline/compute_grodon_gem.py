#!/usr/bin/env python
"""This computes the released gRodon's codon statistics for GEM MAGs.

Genomes are processed in batches: each genome is downloaded from the GEM portal, genes are
predicted and ribosomal proteins found with the same pipeline as for the labeled genomes
(prepare_released_tools_input.py), and a gene file is written; then scripts/r/grodon_stats.R runs
gRodon on the batch and the gene files are deleted, so at most one batch is on disk. Output is
appended to gem_grodon_stats.csv and genomes already in it are skipped, so an interrupted run
resumes.
"""

import argparse
import gzip
import os
import shutil
import subprocess
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import requests
from compute_cub_gem import download

from gem_worldmodel.features import cub_pipeline
from gem_worldmodel.utils.config import load_config, resolve_path

HMM = "data/raw/ribosomal_pfam.hmm"


def write_genes(job):
    genome_id, folder = job
    content = download(requests.Session(), genome_id)
    if content is None:
        return genome_id, None
    fd, path = tempfile.mkstemp(suffix=".fna.gz")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        gzip.open(path, "rb").read(1)
        nucleotides, proteins = cub_pipeline.predict_genes(cub_pipeline.read_contigs(path))
        ribosomal = cub_pipeline.ribosomal_protein_indices(proteins, HMM)
        ffn = Path(folder) / f"{genome_id}.ffn"
        with open(ffn, "w") as out:
            for i, seq in enumerate(nucleotides):
                out.write(f">{genome_id}_{i:05d} {'ribosomal protein' if i in ribosomal else 'predicted protein'}\n{seq}\n")
        return genome_id, str(ffn)
    except Exception:  # noqa: BLE001
        return genome_id, None
    finally:
        os.unlink(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-file", nargs="+", default=["hq_genome_ids_all.csv"])
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    ids = pd.concat([pd.read_csv(processed / f)[["genome_id"]] for f in args.ids_file]).drop_duplicates()["genome_id"].tolist()
    out_path = processed / "gem_grodon_stats.csv"
    done = set(pd.read_csv(out_path)["genome_name"]) if out_path.exists() else set()
    todo = [g for g in ids if g not in done][: args.limit]
    print(f"{len(ids)} genomes requested, {len(done)} done, {len(todo)} to run", flush=True)
    start = time.time()
    for b in range(0, len(todo), args.batch):
        batch = todo[b:b + args.batch]
        folder = Path(tempfile.mkdtemp(prefix="grodon_batch_"))
        try:
            with Pool(args.workers) as pool:
                written = pool.map(write_genes, [(g, str(folder)) for g in batch])
            ok = pd.DataFrame([(g, p) for g, p in written if p], columns=["genome_name", "gene_location"])
            failed = [g for g, p in written if not p]
            stats_path = folder / "stats.csv"
            if len(ok):
                ok.to_csv(folder / "input.csv", index=False)
                subprocess.run(["Rscript", "scripts/r/grodon_stats.R", str(folder / "input.csv"), str(stats_path), str(args.workers)],
                               check=True, capture_output=True)
                stats = pd.read_csv(stats_path)
            else:
                stats = pd.DataFrame(columns=["genome_name"])
            if failed:
                stats = pd.concat([stats, pd.DataFrame({"genome_name": failed})], ignore_index=True)
            stats.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        n = b + len(batch)
        rate = n / (time.time() - start)
        print(f"{n}/{len(todo)} done ({len(failed)} failed downloads in this batch), {rate:.2f} genomes/s, "
              f"about {(len(todo) - n) / max(rate, 1e-9) / 60:.0f} min left", flush=True)
    res = pd.read_csv(out_path)
    print(f"finished: {len(res)} rows, {res['CUBHE'].notna().sum()} with statistics")


if __name__ == "__main__":
    main()
