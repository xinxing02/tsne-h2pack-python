"""Run the whole benchmark: every (size, threads, rep, method) in config.py.

    python sweep.py              # resumes: cells already in results/runs.jsonl are skipped
    python sweep.py --dry-run    # list what would run
    python report.py             # tables + figures from whatever has finished

Order: rep 0 of every size first, then rep 1, ... so a complete single-rep table
exists as early as possible and an interrupted sweep is still usable. Within a
(size, threads, rep): the two k-NN graphs, then each method on them. The large
cached graphs are deleted once their cell is done.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import config
import data

RESULTS = os.path.join(data.RESULTS_DIR, "runs.jsonl")


def done_cells():
    cells = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            r = json.loads(line)
            if "error" not in r:
                cells.add((r["dataset"], r["method"], r["n"], r["threads"], r["rep"], r.get("k")))
    return cells


def plan():
    """[(method, n, threads, rep, k)] in execution order"""
    cells = []
    for rep in range(max(config.reps_for(n) for n in config.SIZES)):
        for n in config.SIZES:
            if rep >= config.reps_for(n):
                continue
            for threads in config.THREADS:
                needs_tsne_graph = any(m in config.METHODS for m in config.TSNE_METHODS)
                if needs_tsne_graph:
                    cells.append(("knn", n, threads, rep, config.K_TSNE))
                if "UMAP" in config.METHODS:
                    cells.append(("knn", n, threads, rep, config.K_UMAP))
                for method in config.METHODS:
                    if not config.skipped(method, n, threads):
                        cells.append((method, n, threads, rep, None))
                if "FIt-SNE" in config.METHODS and rep == 0:
                    # FIt-SNE's own Annoy front end, once per (size, threads): the k-NN
                    # cost its users pay, since its wrapper cannot take a graph
                    cells.append(("FIt-SNE-native-knn", n, threads, rep, None))
    return cells


def cleanup(n, threads, rep):
    """Drop the multi-GB graph / P caches of a finished (size, threads, rep)"""
    for name in (f"{config.DATASET}_knn{config.K_TSNE}_{n}_r{rep}_t{threads}.npz",
                 f"{config.DATASET}_knn{config.K_UMAP}_{n}_r{rep}_t{threads}.npz",
                 f"{config.DATASET}_P_{n}_r{rep}_t{threads}.npz"):
        p = os.path.join(data.CACHE_DIR, name)
        if os.path.exists(p):
            os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-cache", action="store_true", help="keep k-NN graphs and P after each cell group")
    args = ap.parse_args()

    cells, done = plan(), done_cells()
    todo = [c for c in cells if (config.DATASET, c[0], c[1], c[2], c[3], c[4]) not in done]
    print(f"{len(cells)} cells planned, {len(cells) - len(todo)} already done, {len(todo)} to run", flush=True)
    if args.dry_run:
        for c in todo:
            print("  ", c)
        return

    # Keep a laptop awake for the duration (no-op elsewhere)
    prefix = ["caffeinate", "-dims"] if shutil.which("caffeinate") else []
    t_start = time.time()
    for i, (method, n, threads, rep, k) in enumerate(todo):
        graph = data.graph_path(config.DATASET, n, rep, threads, config.K_UMAP if method == "UMAP" else config.K_TSNE)
        if method != "knn" and method != "FIt-SNE-native-knn" and not os.path.exists(graph):
            # resumed after its graph was cleaned up: rebuild it (the timing record already exists)
            subprocess.run(prefix + [sys.executable, os.path.join(data.HERE, "run.py"), "knn", "--n", str(n),
                                     "--threads", str(threads), "--rep", str(rep),
                                     "--k", str(config.K_UMAP if method == "UMAP" else config.K_TSNE),
                                     "--out", os.devnull], cwd=data.HERE)
        cmd = prefix + [sys.executable, os.path.join(data.HERE, "run.py"), method,
                        "--n", str(n), "--threads", str(threads), "--rep", str(rep)]
        if k is not None:
            cmd += ["--k", str(k)]
        print(f"\n[{i + 1}/{len(todo)}  +{(time.time() - t_start) / 60:.0f} min] {' '.join(cmd[len(prefix) + 2:])}", flush=True)
        subprocess.run(cmd, cwd=data.HERE)

        last_of_group = i + 1 == len(todo) or todo[i + 1][1:4] != (n, threads, rep)
        if last_of_group and not args.keep_cache:
            cleanup(n, threads, rep)

    print(f"\nSweep finished in {(time.time() - t_start) / 3600:.1f} h. Now: python report.py", flush=True)


if __name__ == "__main__":
    main()
