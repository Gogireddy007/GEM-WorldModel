#!/usr/bin/env python
"""ESM-2 genome embeddings of the GEM genomes (the HQ genomes and the MAG validation set), made exactly as for the labeled
genomes (scripts/accuracy/embed_proteins_esm.py): genes predicted with cub_pipeline, every ribosomal protein (at most
N_RIBOSOMAL) and N_OTHER other proteins drawn with a seed derived from the genome identifier, each protein the mean of its
last-layer ESM-2 (35M) residue embeddings, and a genome the mean over its ribosomal and over its other proteins.

Stage 1 downloads each genome from the GEM portal (as compute_cub_gem.py does; nothing is kept on disk) and writes the
protein sample to gem_esm_protein_sample.jsonl. Stage 2 embeds it; several workers can run at once on different devices,
each writing its own file gem_esm_embeddings*.csv (--out), the second from the end of the list (--reverse). Both stages
skip work already done, so the script can be stopped and restarted.
"""

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import requests
import torch

from gem_worldmodel.features import cub_pipeline
from gem_worldmodel.utils.config import load_config, resolve_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accuracy"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_cub_gem import download  # noqa: E402
from embed_proteins_esm import HMM, MAX_LEN, N_OTHER, N_RIBOSOMAL, embed  # noqa: E402


def sample(genome_id):
    content = download(requests.Session(), genome_id)
    if content is None:
        return {"genome_id": genome_id, "error": "download failed"}
    fd, path = tempfile.mkstemp(suffix=".fna.gz")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        gzip.open(path, "rb").read(1)
        _, proteins = cub_pipeline.predict_genes(cub_pipeline.read_contigs(path))
        ribo = sorted(cub_pipeline.ribosomal_protein_indices(proteins, resolve_path(HMM)))
    except Exception as e:  # noqa: BLE001
        return {"genome_id": genome_id, "error": str(e)[:200]}
    finally:
        os.unlink(path)
    rng = random.Random(int(hashlib.sha256(genome_id.encode()).hexdigest()[:8], 16))
    ribo_set = set(ribo)
    others = [i for i in range(len(proteins)) if i not in ribo_set and len(proteins[i]) >= 30]
    pick_r = rng.sample(ribo, min(N_RIBOSOMAL, len(ribo)))
    pick_o = rng.sample(others, min(N_OTHER, len(others)))
    return {"genome_id": genome_id, "ribosomal": [proteins[i][:MAX_LEN] for i in pick_r], "other": [proteins[i][:MAX_LEN] for i in pick_o]}


def embedded(processed):
    have = set()
    for f in processed.glob("gem_esm_embeddings*.csv"):
        have |= set(pd.read_csv(f, usecols=["genome_id"])["genome_id"])
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", nargs="+", default=["hq_genome_ids_all.csv", "mag_validation_ids.csv"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--out", default="gem_esm_embeddings.csv")
    args = ap.parse_args()
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    ids = pd.concat([pd.read_csv(processed / f)[["genome_id"]] for f in args.ids_file]).drop_duplicates()["genome_id"].tolist()
    cache = processed / "gem_esm_protein_sample.jsonl"

    if not args.reverse:
        # failed downloads are tried again
        done = {r["genome_id"] for r in map(json.loads, open(cache)) if "error" not in r} if cache.exists() else set()
        todo = [g for g in ids if g not in done]
        print(f"{len(ids)} genomes, {len(todo)} to sample", flush=True)
        with Pool(args.workers) as pool, open(cache, "a") as handle:
            for k, rec in enumerate(pool.imap_unordered(sample, todo, chunksize=2), 1):
                handle.write(json.dumps(rec) + "\n")
                handle.flush()
                if k % 200 == 0:
                    print(f"sampled {k}/{len(todo)}", flush=True)
    if args.sample_only:
        return

    import esm
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval().to(device)
    out_path = processed / args.out
    recs = [json.loads(line) for line in open(cache)]
    recs = list({r["genome_id"]: r for r in recs if "error" not in r and r["ribosomal"] and r["other"]}.values())
    if args.reverse:
        recs = recs[::-1]
    rows, k = [], 0
    have = embedded(processed)
    for rec in recs:
        if rec["genome_id"] in have:
            continue
        e_r = embed(model, alphabet, device, rec["ribosomal"]).mean(axis=0)
        e_o = embed(model, alphabet, device, rec["other"]).mean(axis=0)
        rows.append({"genome_id": rec["genome_id"], **{f"esm_r{i}": v for i, v in enumerate(e_r)},
                     **{f"esm_o{i}": v for i, v in enumerate(e_o)}})
        k += 1
        if len(rows) == 20:
            pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            rows = []
            have = embedded(processed)  # the other worker's progress
            print(f"embedded {k}", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    print(f"done: {len(embedded(processed))} GEM genomes embedded in all files", flush=True)


if __name__ == "__main__":
    main()
