"""GTDB-distance phylogeny branch: rooted pairwise patristic distances -> a
fixed-length embedding via classical MDS (PCoA).

This is the primary phylogeny signal (per the post-pivot plan); the 16S branch
in rrna16s.py is kept as a separate baseline for comparison, not the primary
source.
"""

import dendropy
import numpy as np
from sklearn.manifold import MDS

from gem_worldmodel.utils.config import load_config


def patristic_distance_matrix(
    tree: dendropy.Tree,
) -> tuple[np.ndarray, list[str]]:
    """Compute the rooted pairwise patristic (branch-length) distance matrix.

    Returns (matrix, labels) where labels[i] is the tip's taxon label for row/col i.
    """
    pdm = tree.phylogenetic_distance_matrix()
    taxa = list(tree.taxon_namespace)
    taxa = [t for t in taxa if t in {leaf.taxon for leaf in tree.leaf_node_iter()}]
    labels = [t.label for t in taxa]
    n = len(taxa)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = pdm.patristic_distance(taxa[i], taxa[j])
            mat[i, j] = d
            mat[j, i] = d
    return mat, labels


def classical_mds_embedding(
    distance_matrix: np.ndarray,
    n_components: int,
    seed: int = 0,
) -> np.ndarray:
    """Reduce a precomputed distance matrix to a fixed-length embedding via classical MDS.

    Always returns exactly `n_components` columns, zero-padded on the right
    when the sample is too small to support that many MDS dimensions.
    """
    n = distance_matrix.shape[0]
    effective_components = min(n_components, max(1, n - 1))
    mds = MDS(
        n_components=effective_components,
        metric="precomputed",
        random_state=seed,
        normalized_stress="auto",
        n_init=4,
        init="random",
    )
    embedding = mds.fit_transform(distance_matrix)
    if embedding.shape[1] < n_components:
        pad = np.zeros((n, n_components - embedding.shape[1]))
        embedding = np.hstack([embedding, pad])
    return embedding


def landmark_distance_embedding(
    tree: dendropy.Tree,
    n_landmarks: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Scalable alternative to classical_mds_embedding for large trees (e.g.
    GEM's 43,979-tip OTU tree, where a full NxN patristic distance matrix is
    ~15GB and classical MDS is infeasible on a laptop).

    This is "landmark MDS" (a.k.a. Nystrom-style approximation): pick
    `n_landmarks` reference tips at random, compute each tip's patristic
    distance to just those landmarks, and use the resulting distance vector
    directly as that tip's embedding. It preserves the same phylogenetic-
    distance signal classical MDS targets, at a fraction of the cost, and is
    a well-established technique for embedding graphs/trees too large for
    exact MDS.

    Deliberately avoids dendropy's `Tree.phylogenetic_distance_matrix()`,
    which materializes the full NxN distance matrix internally (~15GB for a
    44k-tip tree), exactly the blowup this function exists to avoid. Instead
    it does a single-source BFS over the tree's node graph (parent+child
    edges, summing branch lengths) from each landmark, which is O(N) per
    landmark and reaches every tip directly from the tree structure.
    """
    from collections import deque

    leaves = list(tree.leaf_node_iter())
    rng = np.random.default_rng(seed)
    landmark_idx = rng.choice(len(leaves), size=min(n_landmarks, len(leaves)), replace=False)
    landmarks = [leaves[i] for i in landmark_idx]

    def bfs_distances_from(start_node) -> dict:
        dist = {start_node: 0.0}
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            d = dist[node]
            neighbors = list(node.child_nodes())
            if node.parent_node is not None:
                neighbors.append(node.parent_node)
            for nb in neighbors:
                if nb in dist:
                    continue
                edge_len = nb.edge.length if nb.parent_node is node else node.edge.length
                dist[nb] = d + (edge_len or 0.0)
                queue.append(nb)
        return dist

    landmark_distances = [bfs_distances_from(lm) for lm in landmarks]

    embeddings = {}
    for leaf in leaves:
        if leaf.taxon is None:
            continue
        vec = np.array([dist_map.get(leaf, np.nan) for dist_map in landmark_distances], dtype=float)
        embeddings[leaf.taxon.label] = vec
    return embeddings


def build_gtdb_distance_embeddings(
    tree: dendropy.Tree,
    cfg: dict | None = None,
) -> dict[str, np.ndarray]:
    """Compute the GTDB-distance embedding for every tip in `tree`.

    Returns {accession_bare: embedding_vector}.
    """
    cfg = cfg or load_config("features")
    p = cfg["phylogeny"]
    matrix, labels = patristic_distance_matrix(tree)
    embedding = classical_mds_embedding(matrix, p["embedding_dim"])

    def bare(label: str) -> str:
        return label.split("_", 1)[1] if label.startswith(("RS_", "GB_")) else label

    return {bare(label): embedding[i] for i, label in enumerate(labels)}
