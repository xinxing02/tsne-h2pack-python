"""FIt-SNE on tsne-h2pack's k-NN graph: is the out-of-the-box quality gap the graph?

Builds the k-NN graph exactly as tsne-h2pack's default does (PyNNDescentKNN, k = 90),
turns it into the perplexity-30 P with openTSNE's calibration (the formal sweep's
procedure; it matched tsne-h2pack's P to 2e-6), hands it to FIt-SNE through
`load_affinities='load'`, and runs FIt-SNE otherwise as in oob_compare.py (PCA init,
max_iter=1000). Scored like every other cell (common P + exact 10-NN preservation) and
written next to the out-of-the-box cells as cell_fitsne_h2knn_t<threads>.json.

    python oob_fitsne_sharedknn.py --dataset glove --threads 8
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

CHILD = """
import sys, time, numpy as np
from fitsne import FItSNE
x = np.load(sys.argv[1]).astype(np.float64)
threads, rep = int(sys.argv[2]), int(sys.argv[3])
t0 = time.perf_counter()
Y = FItSNE(x, max_iter=1000, nthreads=threads, rand_seed=rep, load_affinities="load")   # defaults otherwise
print("FITSNE_WALL", time.perf_counter() - t0, flush=True)
np.save(sys.argv[4], np.asarray(Y))
"""


def main():
    import numpy as np
    import data
    import oob_compare
    from tsne_h2pack._knn_methods import PyNNDescentKNN

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="glove")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    oob_compare._set_threads(args.threads)
    n = args.n or data._load_full(args.dataset).shape[0]
    out = oob_compare.out_dir(args.dataset)
    x = data.load_x(args.dataset, n, args.rep)

    print(f"== tsne-h2pack's k-NN graph (PyNNDescentKNN, k=90, {args.threads} threads)", flush=True)
    t0 = time.perf_counter()
    g = PyNNDescentKNN().compute(x, 90, n_jobs=args.threads)
    t_knn = time.perf_counter() - t0
    k = int(np.diff(g.indptr)[0])
    ind = g.indices.reshape(n, k)
    dist = g.data.reshape(n, k).astype(np.float64)
    print(f"   {t_knn:.1f} s", flush=True)

    print("== perplexity-30 P from that graph (openTSNE calibration)", flush=True)
    import openTSNE
    import scipy.sparse as sp
    from openTSNE.nearest_neighbors import PrecomputedNeighbors
    t0 = time.perf_counter()
    aff = openTSNE.affinity.PerplexityBasedNN(perplexity=30, n_jobs=args.threads,
                                              knn_index=PrecomputedNeighbors(ind, dist))
    P = sp.csr_matrix(aff.P)
    P.sort_indices()
    P.data = P.data / P.data.sum()
    t_P = time.perf_counter() - t0
    workdir = path.join(out, f"fitsne_h2knn_work_t{args.threads}")
    os.makedirs(workdir, exist_ok=True)
    P.indptr.astype(np.uint32).tofile(path.join(workdir, "P_row.dat"))
    P.indices.astype(np.uint32).tofile(path.join(workdir, "P_col.dat"))
    P.data.astype(np.float64).tofile(path.join(workdir, "P_val.dat"))
    print(f"   {t_P:.1f} s, nnz {P.nnz:,}", flush=True)

    print(f"== FIt-SNE, load_affinities='load', {args.threads} threads", flush=True)
    x_path = path.join(data.CACHE_DIR, f"{args.dataset}_x_{n}_r{args.rep}.npy")
    y_path = path.join(out, f"fitsne_h2knn_Y_t{args.threads}.npy")
    r = subprocess.run([sys.executable, "-c", CHILD, x_path, str(args.threads), str(args.rep), y_path],
                       cwd=workdir, capture_output=True, text=True, errors="replace")
    with open(path.join(out, f"fitsne_h2knn_t{args.threads}.log"), "w") as f:
        f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    m = re.search(r"FITSNE_WALL ([\d.eE+-]+)", r.stdout)
    if r.returncode != 0 or m is None:
        raise RuntimeError(f"FIt-SNE failed:\n{r.stderr[-2000:]}")
    wall = float(m.group(1))
    t_opt = sum(float(t) for t in re.findall(r"iterations in ([\d.]+) seconds", r.stdout))
    print(f"   wall {wall:.1f} s (optimization {t_opt:.1f} s)", flush=True)

    oob_compare.ensure_eval_graph(args.dataset, n, args.rep)
    res = dict(library="FIt-SNE on tsne-h2pack's k-NN graph", wall=wall + t_knn + t_P, wall_fitsne_only=wall,
               knn=t_knn, t_P=t_P, optimize=t_opt, n_iter=1000, dataset=args.dataset, n=n, rep=args.rep,
               threads=args.threads, finished=time.strftime("%Y-%m-%d %H:%M:%S"), peak_rss_gb=float("nan"),
               settings="FItSNE(X, max_iter=1000, nthreads=threads, rand_seed=rep, load_affinities='load'); "
                        "P = perplexity-30 calibration (openTSNE) of tsne-h2pack's PyNNDescentKNN k=90 graph; "
                        "PCA init and all other FIt-SNE defaults")
    res.update(oob_compare.evaluate(args.dataset, n, args.rep, np.load(y_path)))
    json.dump(res, open(path.join(out, f"cell_fitsne_h2knn_t{args.threads}.json"), "w"), indent=1)
    print(f"   KL {res['kl']:.4f}  10-NN preservation {res['knn_preservation']:.4f}", flush=True)
    for fn in ("P_row.dat", "P_col.dat", "P_val.dat"):
        os.remove(path.join(workdir, fn))


if __name__ == "__main__":
    main()
