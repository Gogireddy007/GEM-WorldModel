#!/usr/bin/env python
"""This counts the lifestyle Pfam families (lifestyle_genes) in GEM MAGs, the
same way as for the labeled genomes (compute_lifestyle_labeled.py).

Each genome is downloaded from the GEM portal, genes are predicted, the two
HMM files are searched and the genome is deleted again, so only the counts are
kept. Output is appended to gem_lifestyle_genes.csv as it goes and genomes
already in the file are skipped, so an interrupted run resumes.
"""

import argparse
import gzip
import os
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from compute_cub_gem import download

from gem_worldmodel.features import lifestyle_genes
from gem_worldmodel.utils.config import load_config, resolve_path

HMM = Path("data/raw/lifestyle_pfam.hmm")
EVALUE_HMM = Path("data/raw/lifestyle_pfam_evalue.hmm")
COLUMNS = [*lifestyle_genes.FAMILIES, *lifestyle_genes.EVALUE_FAMILIES]


def work(genome_id):
    empty = {"genome_id": genome_id, **{c: np.nan for c in COLUMNS}, "n_genes": np.nan}
    content = download(requests.Session(), genome_id)
    if content is None:
        return {**empty, "error": "download failed"}
    fd, path = tempfile.mkstemp(suffix=".fna.gz")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        gzip.open(path, "rb").read(1)
        return {"genome_id": genome_id, **lifestyle_genes.genome_lifestyle(path, HMM, EVALUE_HMM), "error": ""}
    except Exception as exc:  # noqa: BLE001
        return {**empty, "error": str(exc)[:80]}
    finally:
        os.unlink(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-file", nargs="+", default=["hq_genome_ids_all.csv"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="gem_lifestyle_genes.csv")
    args = parser.parse_args()

    if not (HMM.exists() and EVALUE_HMM.exists()):
        raise SystemExit("run scripts/accuracy/compute_lifestyle_labeled.py first to build the HMM files")
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
    print(f"finished: {len(res)} rows, {res.n_genes.notna().sum()} with counts, {(res.error.fillna('') != '').sum()} errors")


if __name__ == "__main__":
    main()
