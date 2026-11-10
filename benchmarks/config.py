"""What the benchmark sweep runs. The single place to change sizes, threads or reps.

Every knob can be overridden from the environment for a quick partial run, e.g.
    BENCH_SIZES=50000,100000 BENCH_THREADS=8 BENCH_REPS=1 python sweep.py
"""
import os

# One protocol for every t-SNE library: the settings all three agree on, so the
# only thing that differs between them is how the gradient is computed.
PERPLEXITY = 30
K_TSNE = 3 * PERPLEXITY          # neighbors per point in the shared graph
N_ITER = 1000
EARLY_EXAGGERATION = 12
EARLY_EXAGGERATION_ITER = 250    # momentum 0.5 during it, 0.8 after
# Learning rate: each library's own 'auto' rule (tsne-h2pack, FIt-SNE: N/12 throughout;
# openTSNE: N/12 during exaggeration, then N; see README). A fixed 200 leaves large
# embeddings far from converged and makes timings incomparable.

K_UMAP = 15                      # UMAP's default n_neighbors; it needs a much smaller graph

# Embedding dimension (BENCH_DIM=3 for the 3D sweep -> results_3d/). In 3D the FFT
# libraries have no FFT: FIt-SNE and openTSNE run their Barnes-Hut gradient instead.
DIM = int(os.environ.get("BENCH_DIM", "2"))


DATASET = os.environ.get("BENCH_DATASET", "zheng")

# Smaller inputs are dominated by fixed costs and say nothing about scaling
SIZES = [50_000, 100_000, 250_000, 500_000, 1_000_000]
# 12 = every performance core of the Apple M5 Max the Nov-2026 results were taken on;
# 8 is kept for comparison with the earlier Sept-2026 M1 Pro run;
# 4 fills in the scaling curve.
THREADS = [1, 4, 8, 12]

# Repetitions per size. Rep r uses subsample seed r and init seed r. The big sizes
# get fewer: run-to-run spread is 1-3%, and one 1M openTSNE run takes half an hour.
REPS = {50_000: 3, 100_000: 3, 250_000: 3, 500_000: 2, 1_000_000: 1}

# Order = order of execution within one (size, threads, rep)
METHODS = ["tsne-h2pack", "FIt-SNE", "openTSNE", "UMAP"]
TSNE_METHODS = ["tsne-h2pack", "FIt-SNE", "openTSNE"]

# The three t-SNE embeddings count as "matched quality" if they stay within these
# of the best of the three at the same (size, threads, rep)
KL_TOLERANCE = 0.03              # relative
KNN_PRES_TOLERANCE = 0.015       # absolute


def _env_list(name, default, cast=int):
    raw = os.environ.get(name)
    return [cast(v) for v in raw.split(",")] if raw else default


SIZES = _env_list("BENCH_SIZES", SIZES)
THREADS = _env_list("BENCH_THREADS", THREADS)
METHODS = _env_list("BENCH_METHODS", METHODS, cast=str)
if os.environ.get("BENCH_REPS"):
    REPS = {n: int(os.environ["BENCH_REPS"]) for n in SIZES}


# Cells a method skips: {method: {threads: max_n}} (a missing thread count = no cap).
# 3D: Barnes-Hut at 1M costs ~7 h per run on one thread, so FIt-SNE's 1M cell is run on
# 8 threads only; openTSNE (Barnes-Hut without OpenMP on macOS) is left out of the 3D
# sweep altogether - see README.
SKIP_ABOVE = {} if DIM == 2 else {"FIt-SNE": {1: 500_000}}
if DIM == 3:
    if not os.environ.get("BENCH_METHODS"):
        METHODS = ["tsne-h2pack", "FIt-SNE", "UMAP"]
    if not os.environ.get("BENCH_REPS"):
        REPS = {n: 1 for n in SIZES}


def reps_for(n):
    return REPS.get(n, 1)


def skipped(method, n, threads):
    return n > SKIP_ABOVE.get(method, {}).get(threads, float("inf"))
