#!/usr/bin/env python
"""This computes gene-based features for GEM MAGs: codon usage bias and the IVYWREL
amino acid fraction (a signature of growth temperature).

Each genome is downloaded from the GEM portal, run through the annotation-free
pipeline (pyrodigal genes, Pfam ribosomal protein HMMs, MILC) and deleted again,
so only the numbers are kept. The output is appended to gem_cub.csv as it goes,
and genomes already in the file are skipped, so an interrupted run resumes.

The portal throttles heavy parallel downloads, so downloads retry with backoff.
"""

import argparse
import gzip
import os
import random
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests

from gem_worldmodel.features import cub_pipeline
from gem_worldmodel.utils.config import load_config, resolve_path

HMM = "data/raw/ribosomal_pfam.hmm"
URL = "https://portal.nersc.gov/GEM/genomes/fna/{genome_id}.fna.gz"


def download(session, genome_id, retries=6):
    for attempt in range(retries):
        try:
            resp = session.get(URL.format(genome_id=genome_id), timeout=60)
            resp.raise_for_status()
            return resp.content
        except Exception:  # noqa: BLE001
            time.sleep(min(2.0 * (2**attempt), 30.0) + random.uniform(0, 1.0))
    return None


def work(genome_id):
    session = requests.Session()
    content = download(session, genome_id)
    if content is None:
        return {"genome_id": genome_id, "cub": np.nan, "ivywrel": np.nan, "n_genes": np.nan, "n_ribosomal": np.nan, "error": "download failed"}
    fd, path = tempfile.mkstemp(suffix=".fna.gz")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        gzip.open(path, "rb").read(1)
        result = cub_pipeline.genome_cub(path, HMM)
        return {"genome_id": genome_id, **result, "error": ""}
    except Exception as exc:  # noqa: BLE001
        return {"genome_id": genome_id, "cub": np.nan, "ivywrel": np.nan, "n_genes": np.nan, "n_ribosomal": np.nan, "error": str(exc)[:80]}
    finally:
        os.unlink(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-file", nargs="+", default=["hq_genome_ids_all.csv"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="gem_cub.csv")
    args = parser.parse_args()

    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    ids = pd.concat([pd.read_csv(processed / f)[["genome_id"]] for f in args.ids_file]).drop_duplicates()["genome_id"].tolist()
    out_path = processed / args.out
    done = set(pd.read_csv(out_path)["genome_id"]) if out_path.exists() else set()
    todo = [g for g in ids if g not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(ids)} genomes requested, {len(done)} already done, {len(todo)} to run with {args.workers} workers", flush=True)

    start, buffer, n = time.time(), [], 0
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(work, todo, chunksize=2):
            buffer.append(row)
            n += 1
            if len(buffer) >= 100 or n == len(todo):
                pd.DataFrame(buffer).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
                buffer = []
                rate = n / (time.time() - start)
                print(f"{n}/{len(todo)} done, {rate:.2f} genomes/s, about {(len(todo) - n) / max(rate, 1e-9) / 60:.0f} min left", flush=True)
    res = pd.read_csv(out_path)
    print(f"finished: {len(res)} rows, {res.cub.notna().sum()} with a value, {(res.error.fillna('') != '').sum()} errors")


if __name__ == "__main__":
    main()
