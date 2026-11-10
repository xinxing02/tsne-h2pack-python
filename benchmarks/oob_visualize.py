"""Figures for the out-of-the-box comparison (results/oob_<dataset>/).

GloVe has no class labels, so both embeddings are colored by the SAME clustering of the
high-dimensional vectors (MiniBatchKMeans on the 100-d data): a cluster gets one color in
both panels, which shows how each library lays out the same structure. Also: log-density
hexbins and a zoom into the densest region of each embedding.

    python oob_visualize.py --dataset glove --threads 8
"""
import argparse
import json
import sys
from os import path

import numpy as np

HERE = path.dirname(path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import data

N_CLUSTERS = 30


def hd_clusters(dataset, n, rep):
    cache = path.join(data.CACHE_DIR, f"{dataset}_kmeans{N_CLUSTERS}_{n}_r{rep}.npy")
    if path.exists(cache):
        return np.load(cache)
    from sklearn.cluster import MiniBatchKMeans
    x = data.load_x(dataset, n, rep)
    labels = MiniBatchKMeans(N_CLUSTERS, batch_size=8192, n_init=3, random_state=0).fit_predict(x)
    np.save(cache, labels)
    return labels


def _palette():
    # two qualitative maps concatenated: 30 distinguishable hues, fixed order
    c = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab10").colors)
    return ListedColormap(c[:N_CLUSTERS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="glove")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rep", type=int, default=0)
    args = ap.parse_args()
    out = path.join(data.RESULTS_DIR, f"oob_{args.dataset}")
    embs = {}
    for lib, fn in (("tsne-h2pack", f"h2pack_Y_t{args.threads}.npy"), ("FIt-SNE", f"fitsne_Y_t{args.threads}.npy")):
        p = path.join(out, fn)
        if path.exists(p):
            embs[lib] = np.load(p)
    if not embs:
        sys.exit("no embeddings found")
    n = next(iter(embs.values())).shape[0]
    cells = {}
    for lib, fn in (("tsne-h2pack", "h2pack"), ("FIt-SNE", "fitsne")):
        p = path.join(out, f"cell_{fn}_t{args.threads}.json")
        if path.exists(p):
            cells[lib] = json.load(open(p))
    labels = hd_clusters(args.dataset, n, args.rep)
    cmap = _palette()
    rng = np.random.default_rng(0)
    order = rng.permutation(n)                  # draw in random order so no cluster sits on top

    def title(lib):
        c = cells.get(lib)
        if not c:
            return lib
        return f"{lib} — {c['wall']:.0f} s end-to-end, KL {c['kl']:.3f}, 10-NN {c['knn_preservation']:.3f}"

    # 1. same high-dimensional clusters, same colors
    fig, axes = plt.subplots(1, len(embs), figsize=(9 * len(embs), 9))
    axes = np.atleast_1d(axes)
    for ax, (lib, Y) in zip(axes, embs.items()):
        ax.scatter(Y[order, 0], Y[order, 1], c=labels[order], cmap=cmap, s=0.15, linewidths=0,
                   rasterized=True, vmin=0, vmax=N_CLUSTERS - 1)
        ax.set_title(title(lib), fontsize=12)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"{args.dataset} (N = {n:,}), {args.threads} threads. Color = k-means cluster of the "
                 f"100-d vectors ({N_CLUSTERS} clusters, identical in both panels)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path.join(out, f"fig_clusters_t{args.threads}.png"), dpi=120)
    plt.close(fig)

    # 2. log-density
    fig, axes = plt.subplots(1, len(embs), figsize=(9 * len(embs), 9))
    axes = np.atleast_1d(axes)
    for ax, (lib, Y) in zip(axes, embs.items()):
        ax.hexbin(Y[:, 0], Y[:, 1], gridsize=500, bins="log", cmap="magma", linewidths=0)
        ax.set_title(title(lib), fontsize=12)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"{args.dataset} (N = {n:,}), {args.threads} threads. Point density (log scale)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path.join(out, f"fig_density_t{args.threads}.png"), dpi=120)
    plt.close(fig)

    # 3. zoom: a window of 12% of the span around the densest hexbin cell of each embedding
    fig, axes = plt.subplots(1, len(embs), figsize=(9 * len(embs), 9))
    axes = np.atleast_1d(axes)
    for ax, (lib, Y) in zip(axes, embs.items()):
        H, xe, ye = np.histogram2d(Y[:, 0], Y[:, 1], bins=60)
        i, j = np.unravel_index(H.argmax(), H.shape)
        cx, cy = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
        w = 0.06 * np.ptp(Y, axis=0).max()
        m = (np.abs(Y[:, 0] - cx) < w) & (np.abs(Y[:, 1] - cy) < w)
        idx = np.flatnonzero(m)
        idx = idx[rng.permutation(len(idx))]
        ax.scatter(Y[idx, 0], Y[idx, 1], c=labels[idx], cmap=cmap, s=1.5, linewidths=0, rasterized=True,
                   vmin=0, vmax=N_CLUSTERS - 1)
        ax.set_xlim(cx - w, cx + w); ax.set_ylim(cy - w, cy + w)
        ax.set_title(f"{lib}: {m.sum():,} points in the densest 12% window", fontsize=12)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Zoom into the densest region of each embedding (same cluster colors)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path.join(out, f"fig_zoom_t{args.threads}.png"), dpi=120)
    plt.close(fig)
    for f in ("clusters", "density", "zoom"):
        p = path.join(out, f"fig_{f}_t{args.threads}.png")
        print(p, f"{path.getsize(p) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
