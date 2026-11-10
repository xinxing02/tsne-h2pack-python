# Development guide

How to build, test and change tsne-h2pack, and the implementation invariants a change must
not break. User documentation lives in the [website](../website/) sources; this file is for
contributors.

## Project Overview

**tsne-h2pack-python** is a Python wrapper for t-SNE with H²Pack acceleration, built as a drop-in replacement for `sklearn.manifold.TSNE`. Both supported platforms use native C extensions:

- **macOS**: Native C extension built with Homebrew GCC (Apple Silicon only)
- **Linux**: Native C extension with OpenMP support

Releases are published to PyPI; see `docs/PACKAGING_GUIDE.md` for the release workflow.

---

## Environment Setup

### Virtual Environment

```bash
uv venv .venv && source .venv/bin/activate
```

### Package Installation

`uv pip install` or plain `pip install` both work; the examples use uv:

```bash
# Editable install for development
uv pip install -e .

# With optional k-NN methods
uv pip install -e '.[knn]'
```

### Prerequisites

**macOS (Apple Silicon)**:
```bash
brew install gcc openblas
```

Verify Homebrew GCC is installed (Apple Clang will not work — see "Critical Notes" below):
```bash
/opt/homebrew/bin/gcc-15 --version
# Expected: gcc (Homebrew GCC 15.x.x)
ls /opt/homebrew/opt/openblas/lib/libopenblas.dylib
```

**Linux**:
```bash
# Ubuntu/Debian
sudo apt-get install gcc libopenblas-dev liblapack-dev liblapacke-dev

# RHEL/CentOS
sudo yum install gcc openblas-devel lapack-devel
```

### Pre-flight Verification

```bash
which python  # Should be the venv python
python -c "import tsne_h2pack; print(tsne_h2pack.__version__)"
python -c "from tsne_h2pack._backend import get_backend; b = get_backend(); print(f'Backend: {b.backend_type}, Available: {b.is_available()}')"
# Expected: Backend: native, Available: True
```

---

## Repository Structure

```
tsne-h2pack-python/
├── src/                          # C source code
│   ├── tsne_h2pack.c             #   Main t-SNE implementation
│   ├── tsne_h2_matmul.c          #   H² matrix products for the repulsive forces
│   ├── tsne_reorder.c            #   tree-order renumbering of the points
│   ├── tsne_attr_2d.h            #   2D-specialized attractive-force kernel
│   ├── tsne_utils.c              #   k-NN, ball tree, VP-tree utilities
│   └── tsne_kernels.c            #   Kernel functions
├── include/
│   └── tsne_h2pack.h             # Public C API
├── tsne_h2pack/                  # Python package
│   ├── __init__.py
│   ├── manifold.py               #   TSNE class (main user-facing interface)
│   ├── _backend.py               #   ctypes bindings, backend detection
│   ├── _knn_methods.py           #   Unified k-NN interface (5 methods)
│   └── _lib/                     #   Compiled C extension + libH2Pack.so
├── H2Pack/                       # H2Pack submodule (built during install)
├── tests/                        # Pytest suite (at repo root, NOT inside package)
│   ├── test_backend.py
│   ├── test_integration.py
│   ├── test_knn_methods.py
│   ├── test_optimizations.py
│   ├── test_perplexity_calibration.py
│   ├── test_tsne.py
│   └── test_validation.py
├── examples/
│   ├── basic_example.py          #   Main demo (MNIST from OpenML; --n_samples, --n_jobs, ...)
│   └── visualize_embedding.py
├── benchmarks/                   # Benchmark suite: tsne-h2pack vs FIt-SNE vs openTSNE vs UMAP
│   ├── README.md                 #   rules of the comparison, how to run
│   ├── config.py                 #   sizes / threads / reps / protocol (the only file to edit)
│   ├── sweep.py, run.py          #   driver; one cell per fresh process -> results/runs.jsonl
│   ├── methods.py, data.py, quality.py, report.py
│   ├── hero_zheng.py, oob_*.py   #   full-dataset hero run; GloVe out-of-the-box comparison
│   └── results/, results_3d/     #   REPORT.md, summary.csv, figures of the last 2D / 3D sweep
├── data/                         # Datasets, not in git (see benchmarks/data.py for sources)
├── docs/
│   ├── DEVELOPMENT.md            #   this file
│   ├── PACKAGING_GUIDE.md        #   release workflow
│   └── TROUBLESHOOTING.md
├── website/, mkdocs.yml          # Documentation site (MkDocs; builds to _site/)
├── scripts/
│   ├── build_package.sh          #   wheel / sdist build helper
│   ├── release_preflight.sh      #   pre-tag checks: versions, submodule pushed, pytest, build, fresh-venv installs
│   └── test_installation.sh      #   post-install smoke test
├── setup.py                      # Native build (Linux + macOS Apple Silicon)
├── pyproject.toml                # Package metadata
├── MANIFEST.in                   # sdist file inclusion rules
├── CHANGELOG.md
└── README.md
```

---

## Build & Rebuild Commands

Both platforms use native C extensions. Different rebuild scenarios:

### Python code only
Editable installs pick up Python edits immediately. If something doesn't take, force a reinstall:

```bash
uv pip install -e . --reinstall --no-deps
```

### C code in `src/` or `include/`
Recompiles the extension in `tsne_h2pack/_lib/`:

```bash
uv pip install -e . --reinstall --no-deps
```

Time: ~30s–2 min depending on cache state.

### H2Pack submodule changed
Rebuild H2Pack first, then the extension:

```bash
cd H2Pack/src
CC=/opt/homebrew/bin/gcc-15 make -f GCC-OpenBLAS.make clean       # macOS
CC=/opt/homebrew/bin/gcc-15 make -f GCC-OpenBLAS.make
# (Linux: drop the CC= prefix — system gcc works.)
cd ../..
uv pip install -e . --reinstall --no-deps
```

### Clean rebuild (nuclear option)

```bash
uv pip uninstall -y tsne-h2pack
rm -rf build/ dist/ *.egg-info tsne_h2pack/_lib/*.so tsne_h2pack/_lib/*.dylib
cd H2Pack/src && make -f GCC-OpenBLAS.make clean && cd ../..
uv pip install -e .
```

---

## Threading Architecture

The `n_jobs` parameter controls parallelization at all levels:

1. **Python k-NN** (sklearn / pynndescent / faiss / annoy) — receives `n_jobs` directly
2. **C extension** — `params.num_threads = self.n_jobs` in `manifold.py`
3. **H²Pack OpenMP** — set via `omp_set_num_threads()` before `H2P_init()` in C

```python
from tsne_h2pack import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    method='h2pack',
    knn_method='pynndescent',  # NOTE: v2.0+ uses knn_method, NOT use_sklearn_knn
    n_jobs=8,
    random_state=42,
)
Y = tsne.fit_transform(X)
```

---

## Critical Implementation Notes

### 1. H²Pack threading fix (must preserve)

**Location**: `src/tsne_h2pack.c` (before `H2P_init()`, search for `omp_set_num_threads`)

```c
// CRITICAL: Set OpenMP thread count BEFORE H2P_init()
// H2P_init() calls omp_get_max_threads() to initialize h2pack->n_thread.
// Without this, H²Pack runs single-threaded regardless of n_jobs.
omp_set_num_threads(workspace->params->num_threads);
H2P_init(&workspace->h2pack_W, low_dim, 1, QR_REL_NRM, &h2_tol);
```

Impact at 8 threads: H² Rebuild 1.04× → 4.69×, H² Matvec 1.01× → 2.47×, overall optimization 2.06× → 3.54×.

Verify with:
```bash
cd benchmarks && BENCH_SIZES=100000 BENCH_THREADS=1,2,4,8 BENCH_METHODS=tsne-h2pack BENCH_REPS=1 python sweep.py && python report.py
```

### 2. Homebrew GCC required on macOS (not Apple Clang)

`/usr/bin/gcc` on macOS is Apple Clang, not GCC. Apple Clang's OpenMP path:
- requires Homebrew `libomp` and special flags (`-Xpreprocessor -fopenmp -lomp`)
- has known race conditions with H²Pack on Apple Silicon
- forces `OMP_NUM_THREADS=1` workaround, defeating parallelization

`setup.py` enforces Homebrew GCC and refuses to build with Apple Clang. Verify the built library uses GNU OpenMP:

```bash
otool -L tsne_h2pack/_lib/libtsne_h2pack.cpython-*.so | grep omp
# Expect: libgomp.1.dylib
# Reject: libomp.dylib
```

### 3. GCC-15 header workaround on macOS 14.4+

GCC-15's `fixincludes` doesn't handle macOS 14.4 SDK headers (`_bounds.h` not found). `setup.py` already passes `-nostdinc` plus explicit `-isystem` flags to work around this. For manual H²Pack builds:

```bash
SDK_PATH=$(xcrun --show-sdk-path)
CPATH="$SDK_PATH/usr/include" LIBRARY_PATH="$SDK_PATH/usr/lib" \
    CC=/opt/homebrew/bin/gcc-15 make -f GCC-OpenBLAS.make
```

### 4. P matrix construction

`src/tsne_h2pack.c` symmetrizes as `P = P + P.T` (not `(P + P.T) / 2`) and normalizes by total sum. Don't "fix" this — it matches openTSNE / sklearn convention for symmetric P.

### 5. Repulsive force is reused between H² rebuilds (must preserve)

**Location**: `tsne_run()` in `src/tsne_h2pack.c` (`rep_reuse`, `workspace->rep`).

The H² matrix stores kernel weights for the Y it was built from. Multiplying those *stale* weights by the *current* Y makes the repulsion of a pair grow linearly with how far it has separated since the build (instead of decaying) — outliers get flung away and the embedding diverges (Zheng 250K at `lr=N/12`: span 1243 vs 235). This is why `h2_rebuild_freq > 2` used to "diverge". The loop therefore reuses the whole repulsive force from the last rebuild until the next one (a bounded, slightly delayed gradient) and skips the matvec on those iterations. Do not go back to calling the matvec with fresh Y on a stale H² matrix. `TSNE_REP_REUSE=0` exists only for A/B runs.

Quality vs frequency (common-P KL): freq 2 slightly better than FIt-SNE, freq 3 on par (default), freq 6+ visibly worse. Early exaggeration is insensitive (`h2_rebuild_freq_early=10`).

### 6. Other performance invariants

- **Leaf size 100, not H2Pack's 2D default of 200**: the rank is only ~10 at `h2_tol=1e-2`, so the near field (D) dwarfs the far field (B). No accuracy impact.
- **Fused matvec** (`src/tsne_h2_matmul.c`): `W*[1|Y0|Y1]` in one traversal. Do not swap in `H2P_matmul` — it memcpy's every D/B block before each GEMM.
- **Point renumbering** (`src/tsne_reorder.c`): points are renumbered along the H² tree order at iterations 30/100/250/400/600/800 for cache-local gathers; `tsne_get_embedding()` undoes it via `workspace->point_order`. Any new per-point array in the workspace must be added to the permuted list there.
- **2D attractive pass** (`compute_attractive_forces_2d` in `src/tsne_h2pack.c`, kernel in `src/tsne_attr_2d.h` — a template included once per gather type, because `#pragma omp` cannot live in a macro). It is compute-bound (~1.6 cycles per nonzero), not memory-bound, once the points are renumbered. What makes it fast, each measured:
  - P's values are read from a **float32 copy** (`workspace->P_val32`, 4 bytes/nnz; the double `P` stays the source of truth for the renumbering and the KL). `tsne_renumber_points` clears `P_val32_valid`; **anything else that permutes or rescales `P->values` must clear it too.** With a float `DTYPE` the copy is not made (`P->values` is used directly).
  - Neighbors are gathered from a **float32 interleaved copy of Y** (`TSNE_ATTR_F32=2`, default). This computes the exact attractive force of `fl32(Y)`: each point displaced by ≤ half a float ulp (3.8e-6 for |y| < 128; Y is re-centered every iteration), identically in every pair, so forces stay antisymmetric. Force error 2.5e-5 relative vs 1e-2 H² tolerance; KL / kNN-preservation unchanged at 250K/500K/1M. Falls back to the double gather when max|y| ≥ 4096. `=1`: double gather + float divide (error 1.5e-7), `=0`: all double.
  - Each neighbor's (x, y) is copied as **one unit** and de-interleaved inside the `omp simd` loop (two scalar loads + two lane inserts per nonzero were 17% slower); blocks of **16** (24/32 are 22% slower, 8 is 2% slower); scalar tail (a padded tail block is slower); **no software prefetch** (45% slower once renumbered).
  - Loops advance by pointer with `size_t` counters. **setuptools compiles with CPython's `-fno-strict-overflow`**, under which `col_idx[idx + k]` with `int idx` was 30% slower than the same source compiled standalone. When a microbenchmark disagrees with the extension, compile the microbenchmark with `-fno-strict-overflow` first. (Adding `-fstrict-overflow` to the whole extension was tried: no effect on the matvec.)
  - `schedule(dynamic, n/(64*threads) clamped to [64, 1024])`; static is 15–35% slower (row lengths vary by region, and P/E cores differ).
  - Tried and rejected: a symmetric once-per-pair pass with scatter; Hilbert instead of the H² (Z-order) numbering (<1%); other renumbering schedules (time per iteration does not decay between renumberings, and before iteration ~100 the embedding has no neighbor structure to exploit; `TSNE_REORDER_ITERS="30,100,..."` overrides the schedule for experiments).
- **B/D blocks are not stored (JIT)**: with the force reuse above, an H² matrix is multiplied exactly once, so `update_h2_matrix()` builds it with `BD_JIT=1` and `tsne_h2_matmul` evaluates each B/D block into a per-thread cache-resident buffer and applies it at once (bit-identical to the stored path; −2 s and 10.3 → 5.9 GB at 500K/8T). AOT storage is kept only when the matrix is multiplied every iteration (`TSNE_REP_REUSE=0`) or through `H2P_matvec` (`TSNE_NO_FUSED=1`, 3+ dims). `TSNE_H2_JIT=0` forces AOT.
- **Serial parts of the rebuild** (what is left: admissible-pair search, B/D metadata, ~1 s of 13 s at 500K/8T): the H2Pack tree partition runs as OpenMP tasks and is bit-identical to the serial tree (post-order numbering is assigned in a serial post-pass — do not move counters back into the recursion); the root box / NaN check is one fused pass (`tsne_calc_root_enbox`, bit-tests for NaN because of `-ffast-math`); proxy point sets are cached per (box size, depth) in `workspace->pp_cache` (H2Pack never frees them); `H2P_partial_pivot_QR` must not open an `omp parallel if(0)` region per Householder step (a one-thread team costs as much as the step).
- **Reductions are block-wise** (`tsne_sum_blocked`, Z terms): fixed 4096-point blocks summed in parallel, block sums added in order, so rounding does not depend on the thread count. Don't replace them with `reduction(+)`.
- **H2Pack large-block pool** (`malloc_aligned_pool_enable`, submodule): only matters in AOT mode now; without it every rebuild reallocates B/D, macOS keeps the freed blocks resident, and a 500K run peaks at 21 GB instead of 10 GB and crawls under memory pressure.
- **Perplexity search** works on min-shifted squared distances and bisects on beta. Do not reintroduce an exponent clamp: it made P exactly uniform (1/k) for large-magnitude inputs such as MNIST in 0–255 units.
- **`learning_rate='auto'` = max(N/12, 200)**. A fixed 200 badly under-converges at large N; always compare against FIt-SNE/openTSNE at matched learning rate and with a *common* P for the KL (self-reported KL is not comparable).

- **Proxy-point cache**: proxy points never depend on the data, so they are generated once into `~/.cache/tsne_h2pack/proxy_v2_<dim>d_f<bits>_tol<tol>.txt` and copied to a private temp file per run (H2Pack rewrites the file it is given). Bump `v2` in `tsne_acquire_proxy_cache()` if `CACHE_MIN_WIDTH`/`CACHE_MAX_WIDTH` or the kernel change. The selection tolerance is the run's `h2_tol` clamped to [1e-6, 1e-2] (`TSNE_PROXY_TOL` overrides): it was a fixed 1e-4, which in 3D meant 312 proxy points per box instead of 146 and a U-build of 96 s instead of 39 s at 250K / 8 threads (2D: 58 vs fewer, small effect). Quality is unchanged because the ID never resolves the kernel below `h2_tol` anyway.
- **3D**: the H² rank is ~60 at `h2_tol=1e-2` (2D: ~10-17), so the U-build (one pivoted QR per node against the proxy points, cost ~ N x n_proxy x rank, independent of the leaf size) dominates the rebuild. **Defaults are per dimension** (`H2_DEFAULTS` / `h2_defaults()` in `manifold.py`; `None` in the constructor resolves to them): 3D uses `h2_tol=3e-2` (rank 41; 13-20% faster than 1e-2 at 8 threads, common-P KL +0.1-0.2%, 10-NN equal or higher; 1e-1 is ~20% faster again for +0.5% KL), everything else as in 2D. Tuned on Zheng 250K / 1M (M1 Pro, one shared P). `h2_leaf_size` does not help in 3D: flat 50-150, then 100 = 200 < 400 < 800 (GloVe 250K, 8T: 163 / 166 / 193 / 260 s) because the near field grows while the U-build stays flat. Rebuild every 6 falls off the same cliff as in 2D (+2% KL). The 2D-specialized attractive kernel does not apply in 3D (the tree-order renumbering does); 3D optimization is 3-7x the 2D cost at the same N (M5 Max, 12 threads). **Open item**: the 3D attractive pass is irregular in N - about 2x the per-point cost at 100K and 500K compared with 50K/250K/1M, at every thread count (`results_3d/REPORT.md` section 5); the M1 Pro run showed it at 100K only, and its 500K / 1-thread cell (922 s) was as fast as the M5 Max's (930 s) while every other cell gained 1.5-1.9x, so the affected sizes depend on the machine as well as on N. It is tied to N, not to the subsample: 500K with a different rep gives the same 111 us/point at 12T, while 400K gives 58 (the normal value). Suspect the tree partition / renumbering at those sizes interacting with the cache geometry.

Debug/A-B environment switches: `TSNE_NO_FUSED`, `TSNE_ATTR_GENERIC`, `TSNE_ATTR_F32` (attractive pass: 2 = float32 gather + divide, default; 1 = double gather, float32 divide; 0 = all double), `TSNE_REORDER`, `TSNE_REORDER_ITERS` (comma-separated renumbering schedule), `TSNE_REP_REUSE`, `TSNE_H2_JIT` (0 = store B/D blocks as before), `TSNE_H2_TIMERS=1` (prints H2Pack build-phase timers, rebuild count, pool hit rate).

---

## k-NN Methods

| Method         | Description                          | Best for                  |
|----------------|--------------------------------------|---------------------------|
| `pynndescent`  | Approximate (default, ~96% recall); exact brute force below 10K samples | Most use cases |
| `exact`        | sklearn brute force                  | <5K samples               |
| `balltree`     | sklearn ball tree                    | 5K–20K samples            |
| `faiss`        | Facebook FAISS                       | >50K samples              |
| `annoy`        | Spotify Annoy                        | Memory-constrained        |

Implementation: `tsne_h2pack/_knn_methods.py`. Each method is wrapped in a class with a `compute(X, k, n_jobs, ...)` method that returns a CSR matrix of k-NN distances.

**API change (v2.0.0)**: the old `use_sklearn_knn=True` parameter is gone. Use `knn_method='exact'` (or another method name) instead.

---

## Testing

```bash
source .venv/bin/activate
pytest tests/ -v
```

Quick sanity check:
```bash
cd examples
python basic_example.py --n-samples 5000   # ~45-60s
python basic_example.py --n-samples 60000  # ~95-120s, full MNIST
```

---

## Common Errors

| Error                                                    | Likely cause                          | Fix                                                    |
|----------------------------------------------------------|---------------------------------------|--------------------------------------------------------|
| `ModuleNotFoundError: tsne_h2pack`                       | Package not installed in active venv  | `uv pip install -e .`                                  |
| `RuntimeError: Homebrew GCC not found` (macOS)           | Missing prerequisite                  | `brew install gcc openblas`                            |
| `RuntimeError: ... Apple Clang ...`                      | `CC` env var points to wrong compiler | `unset CC` then rebuild                                |
| `backend.is_available() == False`                        | C extension didn't build / link       | Clean rebuild; check `tsne_h2pack/_lib/*.so` exists    |
| `fatal error: _bounds.h: No such file or directory`      | GCC-15 + macOS 14.4+ SDK header issue | See "GCC-15 header workaround" above                   |
| `ImportError: ... pynndescent`                           | Optional dep missing                  | `uv pip install pynndescent` or use `knn_method='exact'` |

For user-facing issues, see `docs/TROUBLESHOOTING.md`.

---

## Performance Reference

**Zheng 1M, 2D** (Apple M5 Max, Nov 2026, shared pynndescent graph, 1000 iterations):
- Optimization: 171.8 s on 1 thread, 56.8 s on 4, 37.5 s on 8, 27.8 s on 12 (FIt-SNE: 384 / 165 / 144 / 131 s)
- End to end on 12 threads: 69 s (35 s k-NN, 7 s P and setup, 28 s optimization); peak RSS 15.7 GB
- Thread scaling, 2D: 1 → 4 = 3.0–3.4×, 1 → 12 = 3.8× at 50K to 6.2× at 1M; 3D 1 → 12: 7.4–10.2×
- Full Zheng (1.3M), defaults, 12 threads: 106 s end to end (`benchmarks/results/hero/`)

**Zheng 1M, 3D**: 1273 / 362 / 219 / 172 s at 1 / 4 / 8 / 12 threads (FIt-SNE Barnes-Hut: 3541 s at 4, 2866 s at 8, 2689 s at 12).

Older reference points (M1 Pro, 8 threads): 1M 2D optimization 59.3 s, 150 s end to end; MNIST 60K ~110–120 s total.

Benchmark sources: `benchmarks/results/REPORT.md`, `benchmarks/results_3d/REPORT.md` (see `benchmarks/README.md` for the method).

---

## Related Files

- **`tsne_h2pack/manifold.py`** — `TSNE` class, parameter validation, backend dispatch
- **`tsne_h2pack/_backend.py`** — ctypes structs (`TSNEParams`), library loading
- **`tsne_h2pack/_knn_methods.py`** — All five k-NN method wrappers
- **`src/tsne_h2pack.c`** — t-SNE algorithm + H²Pack integration
- **`include/tsne_h2pack.h`** — C struct definitions (must stay in sync with `_backend.py`)
- **`setup.py`** — Build logic; macOS rpath fix in `fix_macos_rpath()`, Makefile patches in `patch_h2pack_makefile()`
- **`H2Pack/MACOS_BUILD.md`** — H²Pack-specific macOS build notes

