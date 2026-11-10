"""Run ONE benchmark cell in this process and append its record to a JSONL file.

    python run.py knn         --n 250000 --threads 8 --rep 0 --k 90
    python run.py tsne-h2pack --n 250000 --threads 8 --rep 0
    python run.py FIt-SNE-native-knn --n 250000 --threads 8 --rep 0

sweep.py calls this once per cell, each in a fresh process: no library sees another's
thread pools, JIT caches or allocator state, and a crash costs one cell.
"""
import argparse
import json
import os
import platform
import resource
import sys
import tempfile
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--k", type=int, default=None, help="neighbors, for method 'knn'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Before any numerical import: numba fixes its thread ceiling at first use
    for var in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(args.threads)

    import config
    import data
    import methods
    import quality

    out = args.out or os.path.join(data.RESULTS_DIR, "runs.jsonl")
    record = dict(dataset=config.DATASET, dim=config.DIM, method=args.method, n=args.n, threads=args.threads, rep=args.rep,
                  started=time.strftime("%Y-%m-%dT%H:%M:%S"), machine=platform.platform(),
                  python=platform.python_version())

    try:
        if args.method == "knn":
            record["k"] = args.k
            record["t_knn"], record["t_jit_warmup"] = data.build_graph(
                config.DATASET, args.n, args.rep, args.threads, args.k)
        else:
            x = data.load_x(config.DATASET, args.n, args.rep)
            with tempfile.TemporaryDirectory(dir=data.CACHE_DIR) as workdir:
                result = methods.RUNNERS[args.method](x, config.DATASET, args.n, args.rep, args.threads, workdir)
            record.update(t_setup=result["t_setup"], t_optimize=result["t_optimize"], detail=result["detail"])
            if result["embedding"] is not None:
                record.update(quality.score(config.DATASET, args.n, args.rep, args.threads, result["embedding"]))
        # FIt-SNE runs in a child process, so take whichever of the two peaked higher
        rss = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        record["peak_rss_gb"] = rss / (2**30 if sys.platform == "darwin" else 2**20)   # bytes on macOS, KB on Linux
    except Exception as e:                       # keep the sweep going; the cell is reported as failed
        record["error"] = f"{type(e).__name__}: {e}"[:2000]

    with open(out, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "detail"}), flush=True)
    return 1 if "error" in record else 0


if __name__ == "__main__":
    sys.exit(main())
