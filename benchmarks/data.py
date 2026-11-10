"""Datasets and the shared k-NN graphs, cached under benchmarks/_cache/."""
import gzip
import os
import pickle
import time
from os import path

import numpy as np

HERE = path.dirname(path.abspath(__file__))
REPO_ROOT = path.dirname(HERE)
CACHE_DIR = os.environ.get("BENCH_CACHE", path.join(HERE, "_cache"))
DIM = int(os.environ.get("BENCH_DIM", "2"))          # embedding dimension of the whole sweep
RESULTS_DIR = os.environ.get("BENCH_RESULTS", path.join(HERE, "results" if DIM == 2 else f"results_{DIM}d"))
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_full(dataset):
    if dataset == "zheng":                      # 1.3M mouse brain cells, 50 PCA dimensions
        # Zheng et al. 2017 (Nat. Commun. 8:14049), the 10x Genomics 1.3M mouse brain cells,
        # in the preprocessed form openTSNE's large-data example uses (a dict with "pca_50",
        # 1,306,127 x 50, and "CellType1"): download
        # http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz (either the .gz or
        # the unpacked .pkl) into data/.
        for name, opener in (("10x_mouse_zheng.pkl", open), ("10x_mouse_zheng.pkl.gz", gzip.open)):
            p = path.join(REPO_ROOT, "data", name)
            if path.exists(p):
                with opener(p, "rb") as f:
                    return np.asarray(pickle.load(f)["pca_50"])
        raise FileNotFoundError("data/10x_mouse_zheng.pkl[.gz] not found")
    if dataset == "glove":                      # 1,183,514 GloVe word vectors, 100 dimensions
        # http://ann-benchmarks.com/glove-100-angular.hdf5 (485 MB). The benchmark's metric
        # is angular, so rows are scaled to unit length: Euclidean distance on the unit
        # sphere is monotone in the angle, and both libraries only speak Euclidean.
        import h5py
        p = path.join(REPO_ROOT, "data", "glove-100-angular.hdf5")
        if not path.exists(p):
            raise FileNotFoundError(f"{p} not found (curl -L -o {p} http://ann-benchmarks.com/glove-100-angular.hdf5)")
        with h5py.File(p, "r") as f:
            x = np.asarray(f["train"], dtype=np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    if dataset == "mnist":                      # 60K digits, reduced to 50 PCA dimensions
        from sklearn.decomposition import PCA
        raw = np.loadtxt(path.join(REPO_ROOT, "data", "mnist_n60000.csv"), delimiter=",", dtype=np.float32)
        return PCA(n_components=50, random_state=0).fit_transform(raw / 255.0)
    raise ValueError(f"unknown dataset {dataset!r}")


def load_x(dataset, n, rep):
    """The same n points for every method and every process at a given (n, rep).

    Different reps draw different subsets, so the spread across reps includes
    data-sampling variance, not just timer noise.
    """
    cache = path.join(CACHE_DIR, f"{dataset}_x_{n}_r{rep}.npy")
    if path.exists(cache):
        return np.load(cache)
    full = _load_full(dataset)
    if n > full.shape[0]:
        raise ValueError(f"{dataset} has only {full.shape[0]} points")
    seed = (int(n) * 1000003 + int(rep)) & 0xFFFFFFFF
    idx = np.random.RandomState(seed).choice(full.shape[0], n, replace=False)
    x = np.ascontiguousarray(full[idx], dtype=np.float32)
    np.save(cache, x)
    return x


def graph_path(dataset, n, rep, threads, k):
    return path.join(CACHE_DIR, f"{dataset}_knn{k}_{n}_r{rep}_t{threads}.npz")


def build_graph(dataset, n, rep, threads, k):
    """pynndescent k-NN graph, built with `threads` threads. Returns (seconds, jit_warmup_seconds).

    One library and one code path for every consumer: tsne-h2pack, openTSNE and
    FIt-SNE use the k=90 graph, UMAP the k=15 one (its default n_neighbors).
    """
    from pynndescent import NNDescent

    x = load_x(dataset, n, rep)
    # numba compiles pynndescent on first use in a process (about 7 s, independent of n).
    # That is a per-process constant, not part of how the search scales, so it is paid
    # here on a toy input and reported separately.
    t0 = time.perf_counter()
    NNDescent(np.random.default_rng(0).normal(size=(2000, x.shape[1])).astype(np.float32),
              n_neighbors=k + 1, n_jobs=threads, random_state=0).neighbor_graph
    warmup = time.perf_counter() - t0

    t0 = time.perf_counter()
    index = NNDescent(x, n_neighbors=k + 1, metric="euclidean", n_jobs=threads,
                      random_state=rep, verbose=False)
    ind, dist = index.neighbor_graph            # (n, k + 1), column 0 is the point itself
    seconds = time.perf_counter() - t0
    np.savez(graph_path(dataset, n, rep, threads, k),
             ind=ind.astype(np.int32), dist=dist.astype(np.float32))
    return seconds, warmup


def load_graph(dataset, n, rep, threads, k):
    """(ind, dist), both (n, k + 1) with the point itself in column 0."""
    z = np.load(graph_path(dataset, n, rep, threads, k))
    return z["ind"], z["dist"]
