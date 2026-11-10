"""Embedding quality on a common footing.

A library's self-reported KL is measured against its own P and its own estimate
of Z, so it cannot be compared across libraries. Here every embedding of a given
(size, rep, threads) is scored against ONE perplexity-30 P, built from the shared
k-NN graph, with Z estimated the same way for all:

    KL = sum p log p + sum p log(1 + d^2) + log Z,   Z = sum_{i != j} (1 + d_ij^2)^-1

plus kNN-preservation@10: the share of each point's 10 high-dimensional neighbors
that are still among its 10 nearest neighbors in the embedding.
"""
from os import path

import numpy as np
import scipy.sparse as sp

import config
import data


def tsne_neighbors(ind, dist):
    """(n, k) neighbor ids / distances without the point itself.

    pynndescent usually returns the point in column 0, but not for exact duplicates,
    so the self entry is located per row (and the last column dropped if absent).
    """
    n, k1 = ind.shape
    is_self = ind == np.arange(n)[:, None]
    drop = np.where(is_self.any(axis=1), is_self.argmax(axis=1), k1 - 1)
    keep = np.ones_like(ind, dtype=bool)
    keep[np.arange(n), drop] = False
    return ind[keep].reshape(n, k1 - 1), dist[keep].reshape(n, k1 - 1)


def common_P(dataset, n, rep, threads):
    """Symmetric P (sums to 1) from the shared graph: CSR (indptr, indices, data)."""
    cache = path.join(data.CACHE_DIR, f"{dataset}_P_{n}_r{rep}_t{threads}.npz")
    if path.exists(cache):
        z = np.load(cache)
        return z["indptr"], z["indices"], z["data"]
    import openTSNE
    from openTSNE.nearest_neighbors import PrecomputedNeighbors

    nbr, nbd = tsne_neighbors(*data.load_graph(dataset, n, rep, threads, config.K_TSNE))
    aff = openTSNE.affinity.PerplexityBasedNN(
        perplexity=config.PERPLEXITY, n_jobs=8, knn_index=PrecomputedNeighbors(nbr, nbd))
    P = sp.csr_matrix(aff.P)
    P.sort_indices()
    P.data = P.data / P.data.sum()
    np.savez(cache, indptr=P.indptr.astype(np.int64), indices=P.indices.astype(np.int32),
             data=P.data.astype(np.float64))
    return P.indptr, P.indices, P.data


def score(dataset, n, rep, threads, embedding, k=10, n_eval=5000, n_anchors=1000):
    """dict(kl, knn_preservation, span); NaNs if the embedding is not finite."""
    from sklearn.neighbors import KDTree

    Y = np.asarray(embedding, dtype=np.float64)
    if Y.shape != (n, config.DIM) or not np.isfinite(Y).all():
        return dict(kl=float("nan"), knn_preservation=float("nan"), span=float("nan"))

    rng = np.random.default_rng(0)
    queries = rng.choice(n, size=min(n_eval, n), replace=False)
    hd, _ = tsne_neighbors(*data.load_graph(dataset, n, rep, threads, config.K_TSNE))
    ld = KDTree(Y).query(Y[queries], k=k + 1, return_distance=False)[:, 1:]
    preserved = np.mean([len(set(hd[i, :k]) & set(ld[r])) for r, i in enumerate(queries)]) / k

    indptr, indices, p = common_P(dataset, n, rep, threads)
    rows = np.repeat(np.arange(n), np.diff(indptr))
    d2 = np.sum((Y[rows] - Y[indices]) ** 2, axis=1)
    anchors = rng.choice(n, size=min(n_anchors, n), replace=False)
    row_sums = np.empty(len(anchors))
    for a, i in enumerate(anchors):
        w = 1.0 / (1.0 + np.sum((Y - Y[i]) ** 2, axis=1))
        w[i] = 0.0
        row_sums[a] = w.sum()
    Z = n * row_sums.mean()
    kl = float(np.sum(p * np.log(np.maximum(p, 1e-300))) + np.sum(p * np.log1p(d2)) + np.log(Z))
    return dict(kl=kl, knn_preservation=float(preserved), span=float(np.ptp(Y, axis=0).max()))
