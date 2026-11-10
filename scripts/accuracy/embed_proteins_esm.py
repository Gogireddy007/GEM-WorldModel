#!/usr/bin/env python
"""Protein language-model embeddings of the labeled genomes, for a learned-representation baseline.

For every labeled genome with gene features, genes are predicted as everywhere else (cub_pipeline.predict_genes), the
ribosomal proteins are found with the same HMMs, and a fixed sample is embedded with ESM-2 (35M parameters, Lin et al.
2023): every ribosomal protein (at most 60) and 100 other proteins drawn with a seed derived from the accession. Each protein
is the mean of its last-layer residue embeddings (truncated to 1,000 residues); a genome is the mean over its ribosomal
proteins and the mean over the other proteins, concatenated (2 x 480 values).

Stage 1 (CPU, parallel) writes the protein sample to esm_protein_sample.jsonl; stage 2 (Apple GPU if present) writes
esm_genome_embeddings.csv. Both stages skip work already done, so the script can be rerun after an interruption.
"""

import argparse
import hashlib
import json
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from gem_worldmodel.features import cub_pipeline, deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

N_RIBOSOMAL, N_OTHER, MAX_LEN = 60, 100, 1000
HMM = "data/raw/ribosomal_pfam.hmm"


def sample(acc):
    path = cub_pipeline.genome_fasta(resolve_path(f"data/raw/ncbi_genomes/{acc}"))
    if path is None:
        return {"accession": acc, "error": "no genome file"}
    try:
        _, proteins = cub_pipeline.predict_genes(cub_pipeline.read_contigs(path))
        ribo = sorted(cub_pipeline.ribosomal_protein_indices(proteins, resolve_path(HMM)))
    except Exception as e:  # noqa: BLE001
        return {"accession": acc, "error": str(e)[:200]}
    rng = random.Random(int(hashlib.sha256(acc.encode()).hexdigest()[:8], 16))
    others = [i for i in range(len(proteins)) if i not in set(ribo) and len(proteins[i]) >= 30]
    pick_r = rng.sample(ribo, min(N_RIBOSOMAL, len(ribo)))
    pick_o = rng.sample(others, min(N_OTHER, len(others)))
    return {"accession": acc, "ribosomal": [proteins[i][:MAX_LEN] for i in pick_r], "other": [proteins[i][:MAX_LEN] for i in pick_o]}


def embed(model, alphabet, device, seqs, batch_tokens=4000):
    """Mean residue embedding of each sequence, batched by length."""
    conv = alphabet.get_batch_converter()
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = np.zeros((len(seqs), model.embed_dim), dtype=np.float32)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * (len(seqs[order[j]]) + 2) <= batch_tokens:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        _, _, tok = conv([(str(k), seqs[k]) for k in idx])
        with torch.no_grad():
            rep = model(tok.to(device), repr_layers=[model.num_layers])["representations"][model.num_layers].float().cpu().numpy()
        for row, k in enumerate(idx):
            out[k] = rep[row, 1:len(seqs[k]) + 1].mean(axis=0)
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None, help="mps or cpu (default: mps if present)")
    ap.add_argument("--reverse", action="store_true", help="embed from the end of the list (a second worker)")
    ap.add_argument("--out", default="esm_genome_embeddings.csv")
    args = ap.parse_args()
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    df, _, _ = deploy_features.load_training_table(processed, load_config("features")["rrna16s"]["kmer_k"], with_cub=True)
    accs = df.loc[df["has_cub"], "accession"].dropna().unique().tolist()

    cache = processed / "esm_protein_sample.jsonl"
    done = {json.loads(line)["accession"] for line in open(cache)} if cache.exists() else set()
    todo = [a for a in accs if a not in done]
    print(f"{len(accs)} genomes, {len(todo)} to sample", flush=True)
    with Pool(args.workers) as pool, open(cache, "a") as handle:
        for k, rec in enumerate(pool.imap_unordered(sample, todo), 1):
            handle.write(json.dumps(rec) + "\n")
            if k % 50 == 0:
                print(f"sampled {k}/{len(todo)}", flush=True)

    import esm
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval().to(device)
    out_path = processed / args.out
    have = set()
    for f in processed.glob("esm_genome_embeddings*.csv"):
        have |= set(pd.read_csv(f, usecols=["accession"])["accession"])
    recs = [json.loads(line) for line in open(cache)]
    if args.reverse:
        recs = recs[::-1]
    rows = []
    for k, rec in enumerate(r for r in recs if "error" not in r and r["accession"] not in have and r["ribosomal"] and r["other"]):
        # another worker may have embedded it since this one started
        done_now = set()
        for f in processed.glob("esm_genome_embeddings*.csv"):
            done_now |= set(pd.read_csv(f, usecols=["accession"])["accession"])
        if rec["accession"] in done_now or rec["accession"] in {r_["accession"] for r_ in rows}:
            if args.reverse:
                break
            continue
        e_r = embed(model, alphabet, device, rec["ribosomal"]).mean(axis=0)
        e_o = embed(model, alphabet, device, rec["other"]).mean(axis=0)
        rows.append({"accession": rec["accession"], **{f"esm_r{i}": v for i, v in enumerate(e_r)},
                     **{f"esm_o{i}": v for i, v in enumerate(e_o)}})
        if len(rows) == 10:
            pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            rows = []
            print(f"embedded {k + 1}", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    errors = [r for r in recs if "error" in r]
    total = set()
    for f in processed.glob("esm_genome_embeddings*.csv"):
        total |= set(pd.read_csv(f, usecols=["accession"])["accession"])
    print(f"done: {len(total)} genomes embedded in all files; {len(errors)} without a genome or genes")


if __name__ == "__main__":
    main()
