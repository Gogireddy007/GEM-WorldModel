#!/usr/bin/env python
"""This builds a phylogeny feature that is defined the same way for every
genome, so a model trained on the labeled species can be applied to new ones.

Each GTDB tree tip gets a vector of its distances to a fixed set of landmark
tips (picked once, spread across the tree by farthest-point sampling). The
vector for a tip does not depend on which other genomes are in the batch,
unlike MDS coordinates. A genome that is not itself a tip gets the average
vector of the tips in its species, genus, or family, in that order, using
whichever rank first has a match, and the rank used is recorded.

Writes gtdb_taxon_vectors.csv, one row per taxon (species, genus, family,
order) with the mean landmark vector and how many tips it was built from.
"""

import re

import numpy as np
import pandas as pd

from gem_worldmodel.data import gtdb
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)

N_LANDMARKS = 16


def tree_arrays(tree):
    nodes = list(tree.preorder_node_iter())
    index = {id(n): i for i, n in enumerate(nodes)}
    parent = np.full(len(nodes), -1, dtype=np.int64)
    length = np.zeros(len(nodes))
    children = [[] for _ in nodes]
    for i, n in enumerate(nodes):
        length[i] = n.edge.length or 0.0
        if n.parent_node is not None:
            p = index[id(n.parent_node)]
            parent[i] = p
            children[p].append(i)
    leaves = [i for i, n in enumerate(nodes) if n.is_leaf()]
    labels = {i: nodes[i].taxon.label for i in leaves if nodes[i].taxon is not None}
    return parent, length, children, leaves, labels


def distances_from(start, parent, length, children):
    dist = np.full(len(parent), np.nan)
    dist[start] = 0.0
    stack = [start]
    while stack:
        u = stack.pop()
        du = dist[u]
        for c in children[u]:
            if np.isnan(dist[c]):
                dist[c] = du + length[c]
                stack.append(c)
        p = parent[u]
        if p >= 0 and np.isnan(dist[p]):
            dist[p] = du + length[u]
            stack.append(p)
    return dist


def rank_token(taxonomy: str, prefix: str):
    m = re.search(rf"{prefix}[^;]*", str(taxonomy))
    return m.group(0) if m and m.group(0) != prefix else None


def main():
    data_cfg = load_config("data")
    processed = resolve_path(data_cfg["paths"]["processed_dir"])
    tree = gtdb.load_tree(data_cfg)
    parent, length, children, leaves, labels = tree_arrays(tree)
    leaves = [i for i in leaves if i in labels]
    logger.info(f"{len(leaves)} tips, picking {N_LANDMARKS} landmarks by farthest-point sampling")

    rng = np.random.default_rng(0)
    current = int(rng.choice(leaves))
    leaves_arr = np.array(leaves)
    min_dist = np.full(len(leaves), np.inf)
    columns = []
    for k in range(N_LANDMARKS):
        d = distances_from(current, parent, length, children)[leaves_arr]
        columns.append(d)
        min_dist = np.minimum(min_dist, d)
        current = int(leaves_arr[int(np.argmax(min_dist))])
        logger.info(f"landmark {k + 1}/{N_LANDMARKS} done")
    matrix = np.stack(columns, axis=1)
    tip_labels = [labels[i] for i in leaves]

    taxonomy = gtdb.fetch_bac_taxonomy(data_cfg)
    tax_of = dict(zip(taxonomy["accession"], taxonomy["gtdb_taxonomy"]))
    vec_cols = [f"lm_{i}" for i in range(N_LANDMARKS)]
    tips = pd.DataFrame(matrix, columns=vec_cols)
    tips["tip"] = tip_labels
    tips["taxonomy"] = tips["tip"].map(tax_of)
    tips = tips.dropna(subset=["taxonomy"])
    logger.info(f"{len(tips)} tips have a GTDB taxonomy string")

    rows = []
    for rank, prefix in (("species", "s__"), ("genus", "g__"), ("family", "f__"), ("order", "o__")):
        tips[rank] = tips["taxonomy"].map(lambda t, p=prefix: rank_token(t, p))
        grouped = tips.dropna(subset=[rank]).groupby(rank)
        mean = grouped[vec_cols].mean()
        mean["n_tips"] = grouped.size()
        mean["rank"] = rank
        mean = mean.reset_index().rename(columns={rank: "taxon"})
        rows.append(mean)
        logger.info(f"{rank}: {len(mean)} taxa")
    out = pd.concat(rows, ignore_index=True)
    out_path = processed / "gtdb_taxon_vectors.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}: {len(out)} taxa")


if __name__ == "__main__":
    main()
