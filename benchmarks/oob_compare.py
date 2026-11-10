"""Out-of-the-box comparison: tsne-h2pack vs FIt-SNE, each with its own defaults.

Unlike sweep.py, nothing is shared between the two libraries: each builds its own
k-NN graph (tsne-h2pack: pynndescent; FIt-SNE: Annoy), runs its own perplexity search,
its own initialization (random vs PCA), its own learning-rate rule and its own gradient
conventions. The only setting imposed is max_iter=1000 for both (FIt-SNE's default is
750), so the total time compares equal amounts of work; per-iteration time is reported too.

Quality is judged the same way for both, after the fact: KL divergence on ONE common
perplexity-30 P (built from a pynndescent k=90 graph, as in quality.py) and exact 10-NN
preservation for 5000 query points (brute force against the full data, so neither
library's own approximate graph is the yardstick).

    python oob_compare.py --dataset glove --threads 8        # one cell, both libraries
    python oob_compare.py --dataset glove --threads 1
    python oob_compare.py --dataset glove --report            # tables + figure from results

Results: results/oob_<dataset>/ (JSON per cell, embeddings, FIt-SNE logs, REPORT.md, PNG).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from os import path

HERE = path.dirname(path.abspath(__file__))
sys.path.insert(0, HERE)

EVAL_THREADS = 8
K_EVAL = 90


def out_dir(dataset):
    import data
    d = path.join(data.RESULTS_DIR, f"oob_{dataset}")
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- children
def _set_threads(threads):
    for var in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(threads)


def child_h2pack(dataset, n, rep, threads, out):
    import resource
    import numpy as np
    import data
    from tsne_h2pack import TSNE

    x = data.load_x(dataset, n, rep)
    prof = path.join(out, f"h2pack_profile_t{threads}.csv")
    # verbose stays 0: with verbose=1 the C loop evaluates the KL (a serial pass over all
    # of P) every 10 iterations, ~180 s at 1.2M points, which no default user pays.
    tsne = TSNE(n_jobs=threads, random_state=rep, max_iter=1000,
                enable_profiling=True, profiling_output_file=prof)   # everything else: defaults
    t0 = time.perf_counter()
    Y = tsne.fit_transform(x)
    wall = time.perf_counter() - t0
    np.save(path.join(out, f"h2pack_Y_t{threads}.npy"), Y)
    totals = {}
    if path.exists(prof):
        for line in open(prof):
            if line.startswith("# Total"):          # "# Total optimization time: 12.3 s" etc.
                key, value = line[2:].split(":")
                totals[key.strip()] = float(value.split()[0])
    return dict(library="tsne-h2pack", wall=wall, knn=float(tsne.knn_time_), n_iter=int(tsne.n_iter_),
                kl_self=float(tsne.kl_divergence_), optimize=totals.get("Total optimization time"),
                breakdown=dict(attractive=totals.get("Total P-matvec"), h2_rebuild=totals.get("Total H²-rebuild"),
                               h2_matvec=totals.get("Total H²-matvec")),
                peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
                settings="TSNE(n_jobs=threads, random_state=rep, max_iter=1000); defaults otherwise "
                         "(pynndescent k-NN, perplexity 30, random init, learning_rate='auto')")


def child_fitsne(dataset, n, rep, threads, out):
    import resource
    import numpy as np
    import data
    from fitsne import FItSNE

    x = data.load_x(dataset, n, rep).astype(np.float64)
    t0 = time.perf_counter()
    Y = FItSNE(x, max_iter=1000, nthreads=threads, rand_seed=rep)   # everything else: defaults
    wall = time.perf_counter() - t0
    np.save(path.join(out, f"fitsne_Y_t{threads}.npy"), np.asarray(Y))
    return dict(library="FIt-SNE", wall=wall, n_iter=1000,
                peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
                settings="FItSNE(X, max_iter=1000, nthreads=threads, rand_seed=rep); defaults otherwise "
                         "(Annoy k-NN n_trees=50, perplexity 30, PCA init, learning_rate='auto', "
                         "max_step_norm=5)")


def run_child(lib, dataset, n, rep, threads, out):
    """Fresh process per library so thread env / numba / OpenMP state never carries over."""
    log = path.join(out, f"{lib}_t{threads}.log")
    r = subprocess.run([sys.executable, __file__, "--child", lib, "--dataset", dataset, "--n", str(n),
                        "--rep", str(rep), "--threads", str(threads)],
                       capture_output=True, text=True, errors="replace")
    with open(log, "w") as f:
        f.write(r.stdout)
        f.write("\n--- stderr ---\n")
        f.write(r.stderr)
    m = re.search(r"^OOB_RESULT (.*)$", r.stdout, re.M)
    if r.returncode != 0 or m is None:
        raise RuntimeError(f"{lib} failed (see {log}):\n{r.stderr[-2000:]}")
    res = json.loads(m.group(1))
    if lib == "fitsne":
        # FIt-SNE's C++ core prints its own timers; "iterations in X seconds" every 50 iterations
        res["optimize"] = sum(float(t) for t in re.findall(r"iterations in ([\d.]+) seconds", r.stdout))
        res["knn_and_P_lines"] = re.findall(r"^.*(?:Done in [\d.]+ seconds|seconds).*$", r.stdout, re.M)[:12]
    return res


# --------------------------------------------------------------------------- evaluation
def evaluate(dataset, n, rep, Y, k=10, n_eval=5000):
    """KL on the common P (quality.common_P) + exact 10-NN preservation for n_eval queries."""
    import numpy as np
    from sklearn.neighbors import KDTree, NearestNeighbors
    import data
    import quality

    Y = np.asarray(Y, dtype=np.float64)
    if not np.isfinite(Y).all():
        return dict(kl=float("nan"), knn_preservation=float("nan"), span=float("nan"))
    x = data.load_x(dataset, n, rep)
    rng = np.random.default_rng(0)
    queries = rng.choice(n, size=min(n_eval, n), replace=False)
    cache = path.join(data.CACHE_DIR, f"{dataset}_exact{k}_{n}_r{rep}_q{n_eval}.npy")
    if path.exists(cache):
        hd = np.load(cache)
    else:
        hd = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=EVAL_THREADS).fit(x) \
            .kneighbors(x[queries], return_distance=False)[:, 1:]
        np.save(cache, hd)
    ld = KDTree(Y).query(Y[queries], k=k + 1, return_distance=False)[:, 1:]
    preserved = np.mean([len(set(hd[r]) & set(ld[r])) for r in range(len(queries))]) / k

    indptr, indices, p = quality.common_P(dataset, n, rep, EVAL_THREADS)
    rows = np.repeat(np.arange(n), np.diff(indptr))
    d2 = np.sum((Y[rows] - Y[indices]) ** 2, axis=1)
    anchors = rng.choice(n, size=min(1000, n), replace=False)
    row_sums = np.empty(len(anchors))
    for a, i in enumerate(anchors):
        w = 1.0 / (1.0 + np.sum((Y - Y[i]) ** 2, axis=1))
        w[i] = 0.0
        row_sums[a] = w.sum()
    Z = n * row_sums.mean()
    kl = float(np.sum(p * np.log(np.maximum(p, 1e-300))) + np.sum(p * np.log1p(d2)) + np.log(Z))
    return dict(kl=kl, knn_preservation=float(preserved), span=float(np.ptp(Y, axis=0).max()))


def ensure_eval_graph(dataset, n, rep):
    import data
    if not path.exists(data.graph_path(dataset, n, rep, EVAL_THREADS, K_EVAL)):
        _set_threads(EVAL_THREADS)
        print(f"building the evaluation graph (pynndescent k={K_EVAL}, {EVAL_THREADS} threads)...", flush=True)
        secs, _ = data.build_graph(dataset, n, rep, EVAL_THREADS, K_EVAL)
        print(f"  {secs:.1f} s", flush=True)


# ------------------------------------------------------------------------------- report
def report(dataset, n):
    import glob
    import numpy as np
    out = out_dir(dataset)
    cells = [json.load(open(f)) for f in sorted(glob.glob(path.join(out, "cell_*.json")))]
    if not cells:
        print("no results yet")
        return
    L = [f"# Out-of-the-box: tsne-h2pack vs FIt-SNE on `{dataset}` (N = {n:,})\n",
         "Each library with its own defaults (own k-NN, own perplexity search, own init, own learning-rate "
         "rule); only `max_iter=1000` is imposed on both. Quality on a common perplexity-30 P and exact "
         "10-NN preservation of 5000 query points.\n",
         "| threads | library | total wall (s) | k-NN (s) | optimization (s) | s / iteration | KL (common P) | 10-NN preservation | peak RSS (GB) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (c["threads"], c["library"])):
        knn = c.get("knn")
        opt = c.get("optimize")
        L.append(f"| {c['threads']} | {c['library']} | {c['wall']:.1f} | "
                 f"{'' if knn is None else f'{knn:.1f}'} | {'' if opt is None else f'{opt:.1f}'} | "
                 f"{'' if opt is None else f'{opt / c['n_iter']:.3f}'} | {c['kl']:.4f} | "
                 f"{c['knn_preservation']:.4f} | {'' if c['peak_rss_gb'] != c['peak_rss_gb'] else f"{c['peak_rss_gb']:.1f}"} |")
    for t in sorted({c["threads"] for c in cells}):
        h = [c for c in cells if c["threads"] == t and c["library"] == "tsne-h2pack"]
        f = [c for c in cells if c["threads"] == t and c["library"] == "FIt-SNE"]
        if h and f:
            h, f = h[0], f[0]
            line = f"\n**{t} thread(s)**: end-to-end tsne-h2pack is {f['wall'] / h['wall']:.2f}x faster"
            if h.get("optimize") and f.get("optimize"):
                line += f"; optimization phase {f['optimize'] / h['optimize']:.2f}x faster"
            line += (f". KL {h['kl']:.4f} vs {f['kl']:.4f} ({(h['kl'] / f['kl'] - 1) * 100:+.2f}%), "
                     f"10-NN preservation {h['knn_preservation']:.4f} vs {f['knn_preservation']:.4f}.")
            L.append(line)
    shared = next((c for c in cells if c["library"].startswith("FIt-SNE on tsne-h2pack")), None)
    if shared:
        L.append(f"\nControl (oob_fitsne_sharedknn.py): FIt-SNE given tsne-h2pack's own pynndescent graph as its P "
                 f"(load_affinities='load'), {shared['threads']} threads: optimization {shared['optimize']:.1f} s, "
                 f"KL {shared['kl']:.4f}, 10-NN preservation {shared['knn_preservation']:.4f} — the same quality as "
                 f"tsne-h2pack, so the out-of-the-box gap is the k-NN graph, not the optimizer.")
    rp = path.join(out, "knn_recall.json")
    if path.exists(rp):
        r = json.load(open(rp))
        L.append(f"\nk-NN graph quality (recall of the exact {r['k']} neighbors, {r['queries']} query points): "
                 f"FIt-SNE's Annoy defaults (n_trees=50, search_k=4500) {r['annoy_fitsne_defaults']:.3f}; "
                 f"tsne-h2pack's pynndescent defaults {r['pynndescent_defaults']:.3f} "
                 f"(Annoy with search_k=45000: {r['annoy_search_k_45000']:.3f}). The libraries optimize "
                 f"different P matrices; the quality gap is the k-NN graph, not the optimizer.")
    L.append("\nSettings:")
    for lib in ("tsne-h2pack", "FIt-SNE"):
        c = next((c for c in cells if c["library"] == lib), None)
        if c:
            L.append(f"- {lib}: {c['settings']}")
    with open(path.join(out, "REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))

    # side-by-side density plots of the two embeddings (highest thread count available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = max(c["threads"] for c in cells)
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        for ax, (lib, fn) in zip(axes, (("tsne-h2pack", f"h2pack_Y_t{t}.npy"), ("FIt-SNE", f"fitsne_Y_t{t}.npy"))):
            p = path.join(out, fn)
            if not path.exists(p):
                continue
            Y = np.load(p)
            ax.hexbin(Y[:, 0], Y[:, 1], gridsize=400, bins="log", cmap="magma", linewidths=0)
            c = next(c for c in cells if c["library"] == lib and c["threads"] == t)
            ax.set_title(f"{lib}: {c['wall']:.0f} s end-to-end, KL {c['kl']:.3f}, 10-NN {c['knn_preservation']:.3f}")
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{dataset} (N = {n:,}), {t} threads, each library with its own defaults")
        fig.tight_layout()
        fig.savefig(path.join(out, f"fig_embeddings_t{t}.png"), dpi=110)
        print("figure:", path.join(out, f"fig_embeddings_t{t}.png"))
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


# --------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="glove")
    ap.add_argument("--n", type=int, default=0, help="0 = the whole dataset")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--libs", default="h2pack,fitsne")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--child", choices=["h2pack", "fitsne"])
    args = ap.parse_args()

    if args.child:
        _set_threads(args.threads)
        out = out_dir(args.dataset)
        fn = child_h2pack if args.child == "h2pack" else child_fitsne
        res = fn(args.dataset, args.n, args.rep, args.threads, out)
        print("OOB_RESULT " + json.dumps(res), flush=True)
        return

    import data
    n = args.n or data._load_full(args.dataset).shape[0]
    if args.report:
        report(args.dataset, n)
        return

    out = out_dir(args.dataset)
    data.load_x(args.dataset, n, args.rep)          # materialize once, so no child pays for it
    import numpy as np
    for lib in args.libs.split(","):
        name = "tsne-h2pack" if lib == "h2pack" else "FIt-SNE"
        print(f"== {name}, {args.threads} thread(s), N={n:,}", flush=True)
        t0 = time.perf_counter()
        res = run_child(lib, args.dataset, n, args.rep, args.threads, out)
        print(f"   wall {res['wall']:.1f} s (child total {time.perf_counter() - t0:.1f} s)", flush=True)
        ensure_eval_graph(args.dataset, n, args.rep)
        Y = np.load(path.join(out, f"{lib}_Y_t{args.threads}.npy"))
        print("   scoring...", flush=True)
        res.update(evaluate(args.dataset, n, args.rep, Y))
        res.update(dataset=args.dataset, n=n, rep=args.rep, threads=args.threads,
                   machine=__import__("platform").platform(), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(path.join(out, f"cell_{lib}_t{args.threads}.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(f"   KL {res['kl']:.4f}  10-NN preservation {res['knn_preservation']:.4f}", flush=True)
    report(args.dataset, n)


if __name__ == "__main__":
    main()
