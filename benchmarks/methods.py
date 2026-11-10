"""One adapter per library, all with the same shape.

    run(x, dataset, n, rep, threads, workdir) -> dict(
        embedding,        # (n, config.DIM)
        t_setup,          # seconds from "here is the k-NN graph" to the first iteration
        t_optimize,       # seconds in the optimization loop
        detail,           # library-specific extras (phase split, settings)
    )

The k-NN graph is NOT built here: it is built once per (size, rep, threads) by
data.build_graph and shared, and its time is a record of its own. Every t-SNE
library runs its defaults under the common protocol in config.py; there are no
"fast" variants -- one configuration per library.
"""
import os
import re
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix

import config
import data
import quality


# --------------------------------------------------------------------------- tsne-h2pack
def run_tsne_h2pack(x, dataset, n, rep, threads, workdir):
    from tsne_h2pack import TSNE

    nbr, nbd = quality.tsne_neighbors(*data.load_graph(dataset, n, rep, threads, config.K_TSNE))
    k = nbr.shape[1]
    graph = csr_matrix((nbd.ravel(), nbr.ravel(), np.arange(0, (n + 1) * k, k)), shape=(n, n))

    profile = os.path.join(workdir, "h2pack_profile.csv")
    tsne = TSNE(n_components=config.DIM, perplexity=config.PERPLEXITY, early_exaggeration=config.EARLY_EXAGGERATION,
                max_iter=config.N_ITER, metric="precomputed", n_jobs=threads, random_state=rep,
                init="random",   # protocol: random init for every library (no PCA with a precomputed graph)
                enable_profiling=True, profiling_output_file=profile)   # everything else: package defaults
    t0 = time.perf_counter()
    embedding = tsne.fit_transform(graph)
    wall = time.perf_counter() - t0

    totals = {}
    for line in open(profile):
        if line.startswith("# Total"):
            key, value = line[2:].split(":")
            totals[key.strip()] = float(value.split()[0])
    t_optimize = totals["Total optimization time"]
    return dict(embedding=embedding, t_setup=wall - t_optimize, t_optimize=t_optimize,
                detail=dict(attractive=totals.get("Total P-matvec"), h2_rebuild=totals.get("Total H²-rebuild"),
                            h2_matvec=totals.get("Total H²-matvec"), learning_rate=max(n / 12.0, 200.0),
                            h2_tol=tsne.h2_tol, h2_rebuild_freq=tsne.h2_rebuild_freq,
                            h2_rebuild_freq_early=tsne.h2_rebuild_freq_early, h2_leaf_size=tsne.h2_leaf_size))


# --------------------------------------------------------------------------- openTSNE
def run_opentsne(x, dataset, n, rep, threads, workdir):
    import openTSNE
    from openTSNE.nearest_neighbors import PrecomputedNeighbors

    nbr, nbd = quality.tsne_neighbors(*data.load_graph(dataset, n, rep, threads, config.K_TSNE))
    t0 = time.perf_counter()
    affinities = openTSNE.affinity.PerplexityBasedNN(
        perplexity=config.PERPLEXITY, n_jobs=threads, knn_index=PrecomputedNeighbors(nbr, nbd))
    init = openTSNE.initialization.random(x, n_components=config.DIM, random_state=rep)
    # openTSNE's FFT interpolation is 1D/2D only; Barnes-Hut is its 3D gradient
    embedding = openTSNE.TSNEEmbedding(init, affinities, negative_gradient_method="fft" if config.DIM == 2 else "bh",
                                       learning_rate="auto", n_jobs=threads, random_state=rep)
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    embedding.optimize(config.EARLY_EXAGGERATION_ITER, exaggeration=config.EARLY_EXAGGERATION,
                       momentum=0.5, inplace=True)
    embedding.optimize(config.N_ITER - config.EARLY_EXAGGERATION_ITER, momentum=0.8, inplace=True)
    t_optimize = time.perf_counter() - t0
    return dict(embedding=np.asarray(embedding), t_setup=t_setup, t_optimize=t_optimize, detail={})


# --------------------------------------------------------------------------- FIt-SNE
_FITSNE_CHILD = """
import sys, time, numpy as np, openTSNE
from fitsne import FItSNE
x = np.load(sys.argv[1]).astype(np.float64)
rep, threads, max_iter, mode = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
init = openTSNE.initialization.random(x, n_components={dim}, random_state=rep)
t0 = time.perf_counter()
# FIt-SNE's FFT gradient is hard-coded 2D (the pip wrapper skips the C++ guard and returns
# garbage for no_dims=3), so 3D runs its Barnes-Hut gradient.
emb = FItSNE(x, no_dims={dim}, fft_not_bh={dim} == 2, perplexity={perp}, max_iter=max_iter,
             stop_early_exag_iter={ee_iter}, early_exag_coeff={ee}, initialization=init, nthreads=threads,
             rand_seed=rep, load_affinities=(mode if mode != "native" else None))
print("FITSNE_WALL", time.perf_counter() - t0, flush=True)   # the call alone, not interpreter start-up
np.save(sys.argv[6], np.asarray(emb))
"""


def _fitsne(x_path, rep, threads, max_iter, mode, workdir):
    """FIt-SNE in a child process with cwd=workdir (it reads/writes P_*.dat there).

    Its C++ core reports the time of every 50 iterations on stdout; their sum is the
    optimization time, independent of how long loading or computing P took.
    """
    out = os.path.join(workdir, "fitsne_embedding.npy")
    code = _FITSNE_CHILD.format(perp=config.PERPLEXITY, ee_iter=config.EARLY_EXAGGERATION_ITER,
                                ee=config.EARLY_EXAGGERATION, dim=config.DIM)
    r = subprocess.run([sys.executable, "-c", code, x_path, str(rep), str(threads), str(max_iter), mode, out],
                       cwd=workdir, capture_output=True, text=True, errors="replace")
    wall = re.search(r"FITSNE_WALL ([\d.eE+-]+)", r.stdout)
    if r.returncode != 0 or wall is None or not os.path.exists(out):
        raise RuntimeError(f"FIt-SNE failed: {r.stderr[-1500:]}")
    wall = float(wall.group(1))
    t_optimize = sum(float(t) for t in re.findall(r"iterations in ([\d.]+) seconds", r.stdout))
    return np.load(out), wall, t_optimize


def run_fitsne(x, dataset, n, rep, threads, workdir):
    """FIt-SNE optimizing the SAME P as the others (its wrapper can load affinities)."""
    indptr, indices, p = quality.common_P(dataset, n, rep, threads)
    indptr.astype(np.uint32).tofile(os.path.join(workdir, "P_row.dat"))
    indices.astype(np.uint32).tofile(os.path.join(workdir, "P_col.dat"))
    p.astype(np.float64).tofile(os.path.join(workdir, "P_val.dat"))
    x_path = os.path.join(data.CACHE_DIR, f"{dataset}_x_{n}_r{rep}.npy")
    embedding, _, t_optimize = _fitsne(x_path, rep, threads, config.N_ITER, "load", workdir)
    # Reading P back from disk is an artifact of this harness, not a cost of FIt-SNE,
    # so there is no meaningful setup time; see run_fitsne_native_knn for its own pipeline.
    return dict(embedding=embedding, t_setup=None, t_optimize=t_optimize, detail={})


def run_fitsne_native_knn(x, dataset, n, rep, threads, workdir):
    """FIt-SNE's own front end: bundled Annoy k-NN + perplexity calibration.

    Its wrapper cannot take a k-NN graph (only finished affinities), so this is the
    k-NN cost a FIt-SNE user actually pays. One iteration keeps the run short.
    """
    x_path = os.path.join(data.CACHE_DIR, f"{dataset}_x_{n}_r{rep}.npy")
    _, wall, t_optimize = _fitsne(x_path, rep, threads, 1, "native", workdir)
    return dict(embedding=None, t_setup=wall - t_optimize, t_optimize=None, detail={})


# --------------------------------------------------------------------------- UMAP
def run_umap(x, dataset, n, rep, threads, workdir):
    """UMAP defaults (n_neighbors=15, spectral init, its own epoch count) on a
    pynndescent graph built by the same code path as the t-SNE one.

    No random_state: UMAP silently drops to one thread when a seed is set.
    """
    import umap

    ind, dist = data.load_graph(dataset, n, rep, threads, config.K_UMAP)   # self in column 0, as UMAP expects
    ind, dist = ind[:, :config.K_UMAP], dist[:, :config.K_UMAP]
    # numba compiles UMAP on first use in a process; pay that on a toy input, as for pynndescent
    t0 = time.perf_counter()
    umap.UMAP(n_neighbors=config.K_UMAP, n_jobs=threads).fit_transform(
        np.random.default_rng(0).normal(size=(2000, x.shape[1])).astype(np.float32))
    warmup = time.perf_counter() - t0

    t0 = time.perf_counter()
    embedding = umap.UMAP(n_neighbors=config.K_UMAP, n_components=config.DIM, n_jobs=threads,
                          precomputed_knn=(ind, dist, None)).fit_transform(x)
    wall = time.perf_counter() - t0
    # UMAP does not expose a split between building its fuzzy graph / spectral init and
    # the SGD epochs, so the whole fit is reported as optimization.
    return dict(embedding=embedding, t_setup=0.0, t_optimize=wall, detail=dict(jit_warmup=warmup))


RUNNERS = {
    "tsne-h2pack": run_tsne_h2pack,
    "FIt-SNE": run_fitsne,
    "openTSNE": run_opentsne,
    "UMAP": run_umap,
    "FIt-SNE-native-knn": run_fitsne_native_knn,
}
