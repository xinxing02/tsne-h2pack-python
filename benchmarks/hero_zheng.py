"""The whole Zheng dataset (1,306,127 cells, 50 PCA components) with tsne-h2pack's defaults.

Produces the website's hero figure and the end-to-end time quoted next to it. Everything is
a default except the thread count: PyNNDescent k-NN, perplexity 30, 1000 iterations,
`learning_rate='auto'`. Output in results/hero/: the embedding, a JSON record with the
timing split (k-NN / rest), and the figure colored by the 39 annotated cell types.

    python hero_zheng.py --threads 12
"""
import argparse
import json
import os
import platform
import sys
import time
from os import path

HERE = path.dirname(path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)

    import numpy as np
    sys.path.insert(0, HERE)
    import data
    import tsne_h2pack
    from tsne_h2pack import TSNE

    out = path.join(data.RESULTS_DIR, "hero")
    os.makedirs(out, exist_ok=True)
    tag = f"{args.dim}d_t{args.threads}"

    import pickle
    with open(path.join(data.REPO_ROOT, "data", "10x_mouse_zheng.pkl"), "rb") as f:
        d = pickle.load(f)
    x = np.ascontiguousarray(d["pca_50"], dtype=np.float32)
    labels = np.asarray(d["CellType1"]).astype(str)
    n = x.shape[0]
    print(f"{n:,} cells x {x.shape[1]} features, {len(set(labels))} cell types; {args.threads} threads", flush=True)

    # numba compiles pynndescent on first use (~7 s per process, independent of N); a user
    # pays it once per session, so it is paid here on a toy input and reported separately.
    t0 = time.perf_counter()
    from pynndescent import NNDescent
    NNDescent(np.random.default_rng(0).normal(size=(2000, x.shape[1])).astype(np.float32),
              n_neighbors=91, n_jobs=args.threads, random_state=0).neighbor_graph
    warmup = time.perf_counter() - t0

    tsne = TSNE(n_components=args.dim, n_jobs=args.threads, random_state=args.seed)
    t0 = time.perf_counter()
    y = tsne.fit_transform(x)
    wall = time.perf_counter() - t0
    rec = dict(dataset="zheng", n=int(n), dim=args.dim, threads=args.threads, seed=args.seed,
               t_total=wall, t_knn=float(tsne.knn_time_), t_rest=wall - float(tsne.knn_time_),
               numba_warmup=warmup, kl_self_reported=float(tsne.kl_divergence_),
               version=tsne_h2pack.__version__, machine=platform.platform(),
               python=platform.python_version(), started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(json.dumps(rec), flush=True)
    np.save(path.join(out, f"zheng_full_Y_{tag}.npy"), y)
    json.dump(rec, open(path.join(out, f"zheng_full_{tag}.json"), "w"), indent=1)

    if args.dim == 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        types, inv = np.unique(labels, return_inverse=True)
        # 39 categories: two qualitative maps, ordered so neighbors in the list differ in hue
        palette = [plt.get_cmap("tab20")(i) for i in range(20)] + [plt.get_cmap("tab20b")(i) for i in range(20)]
        colors = np.array(palette)[inv % len(palette)]
        order = np.random.default_rng(0).permutation(n)       # no cell type drawn on top of another
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(y[order, 0], y[order, 1], s=0.6, c=colors[order], linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(path.join(out, f"zheng_hero_t{args.threads}.png"), dpi=160, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
