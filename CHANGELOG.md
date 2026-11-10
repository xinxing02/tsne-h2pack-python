# Changelog

All notable changes to tsne-h2pack will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.3.1] - 2026-11-09

Installs and runs on every CPython from 3.10 to 3.14, checked automatically in CI.

### Added

- Wheels for **Python 3.14** (Linux x86-64 and macOS arm64).
- CI runs the test suite on Python 3.10, 3.11, 3.12, 3.13 and 3.14 on Linux and macOS (was
  3.10 and 3.12), installs the `pynndescent` extra the way users do, and runs the defaults on
  20,000 points, large enough to take PyNNDescent's NN-descent path. Every wheel is tested the
  same way after it is built.

### Changed

- **Python 3.10 or newer is required** (`requires-python = ">=3.10"`; was 3.8). Python 3.8
  and 3.9 are end-of-life (October 2024 and October 2025), and neither was tested before. On
  3.9 the newest PyNNDescent (0.7.0) crashes inside numba 0.60, the newest numba for 3.9,
  which broke the default `knn_method='pynndescent'` above a few thousand points, and the
  test suite failed on GitHub's macOS runner. No more wheels are built for 3.9; 3.3.0 stays
  installable there.

### Fixed

- With the default `init='pca'`, NumPy 2.0-2.2 on macOS (2.2 is the newest NumPy for Python
  3.10) printed about fifty spurious `RuntimeWarning: divide by zero / overflow / invalid value
  encountered in matmul` from scikit-learn's randomized PCA, although the result was exact
  (numpy/numpy#28687). They are silenced around the PCA only; a non-finite PCA result now
  raises a clear error.
- The Annoy self-check in the test suite runs in a child process: a broken Annoy build can
  segfault (GitHub's macOS runner, Python 3.11), which killed the whole pytest run.

## [3.3.0] - 2026-11-05

PCA initialization, now the default as in scikit-learn, and the benchmarks rerun on an
Apple M5 Max at 1, 4, 8 and 12 threads.

### Added

- `init='pca'`, as in scikit-learn: the randomized PCA of X (seeded with `random_state`),
  rescaled so the first coordinate has standard deviation 1e-4. Computed in Python
  (0.5 s at 1M x 50) and handed to C through the new `tsne_params_t.init_Y` field (appended;
  the struct is now 160 bytes).
- `init` also accepts an ndarray of shape (n_samples, n_components), used as is, also with
  `metric='precomputed'`.

### Changed

- **The default initialization is now `init='pca'`**, as in scikit-learn 1.2+ (was `'random'`).
  On Zheng 100K / 250K (same P, 3 seeds) PCA init gives clearly better global structure
  (Spearman correlation of high- vs low-dimensional distances 0.54 / 0.56 vs 0.49 / 0.47, and
  far less seed-dependent), the same 10-NN preservation and a 0.2-0.5% higher common-P KL,
  at no cost in optimization time. Pass `init='random'` for the previous behavior.
- With `metric='precomputed'` there are no features to project, so `init='pca'` (the default)
  falls back to random initialization with a `UserWarning`; scikit-learn raises instead.
  Pass `init='random'` to silence it. The benchmark scripts pin `init='random'`, the protocol
  every library is compared under.

### Benchmarks

- All suites were rerun on an Apple M5 Max at 1, 4, 8 and 12 threads: 2D and 3D sweeps on
  the 10x mouse brain cells, the GloVe out-of-the-box comparison and the full 1.3M-cell run
  (see `benchmarks/`). Headline 2D numbers (optimization on the same P):
  2.1-4.0x faster than FIt-SNE on one thread, 2.9-9.7x on four, 3.9-11.4x on eight and
  4.7-11.6x on twelve; 1M points in 27.8 s of optimization and 69 s end to end on 12
  threads. 3D: 5.3-18.5x faster than FIt-SNE's Barnes-Hut at better quality. Full 1.3M
  cells end to end: 106 s on 12 threads.
- `benchmarks/config.py` runs 1, 4, 8 and 12 threads; `report.py` prints a thread-scaling
  table per thread count; new `benchmarks/hero_zheng.py` for the full-dataset run.

### Build

- **The macOS wheels now require macOS 15** (`macosx_15_0_arm64`; was 14). GitHub retired
  its macOS 14 runners on 2026-11-02, and the Homebrew OpenBLAS and GCC runtime libraries
  bundled into the wheel on the macOS 15 runners are built for macOS 15. On macOS 14, pip
  builds from the source distribution (needs Homebrew `gcc` and `openblas`).
- CI runs its macOS jobs on `macos-15`.
- On macOS 27 the H2Pack makefile needs `MACOSX_DEPLOYMENT_TARGET` set (e.g. 14.0):
  GCC 12 otherwise derives `-mmacosx-version-min=18.0` from the Darwin version and the
  assembler rejects it. `setup.py` already sets it for the extension.

### Repository

- Prepared for the public release: maintainer scripts moved to `scripts/`, a development
  guide in `docs/DEVELOPMENT.md`, superseded pre-September-2026 benchmark results and the
  old hand-written site removed, project links point to GitHub and the documentation site.

## [3.2.0] - 2026-10-07

3D got a real tuning pass: per-dimension H2 defaults, the tree-order renumbering in any
dimension, and proxy points selected at `h2_tol`. Zheng 1M in 3D at 8 threads: 857 s -> 362 s
optimization for the same embedding quality; vs FIt-SNE's Barnes-Hut (its only 3D mode)
9.6-24x faster on identical P (`benchmarks/results_3d/REPORT.md`). 2D results are unchanged
apart from ~10% from the proxy-point change.

### Changed
- H2 settings default to `None` and resolve per embedding dimension (`tsne_h2pack.H2_DEFAULTS`,
  `h2_defaults(n_components)`): 2D unchanged (`h2_tol=1e-2`, rebuild 3/10, leaf 100); **3D uses
  `h2_tol=3e-2`** (rank 41 instead of 58; 13-20% faster at 8 threads, common-P KL +0.1-0.2%,
  10-NN preservation equal or higher on Zheng 250K / 1M). Tuning grid on one shared P: leaf
  size flat 50-150, rebuild every 2 / 4 / 6 = -0.1% / +0.35% / +2% KL, `h2_tol` 1e-1 another
  ~20% faster for +0.5% KL. Explicit values still win.

### Performance
- Points are renumbered along the H2 tree order in any embedding dimension (it was 2D-only).
  The generic attractive pass gathers Y[d*n + j] per neighbor and became cache-miss bound at
  scale in 3D (grew like N^1.7): Zheng 1M / 8 threads, 3D, attractive 562 -> 147 s,
  optimization 856 -> 440 s; embedding unchanged.
- Proxy points are selected at the run's `h2_tol` (clamped to [1e-6, 1e-2]) instead of a fixed
  1e-4 left over from the old default. Fewer proxy points make every node's ID cheaper on every
  rebuild; the ID never resolved the kernel below `h2_tol` anyway, so quality is unchanged.
  3D, GloVe 250K, 8 threads: 312 -> 146 proxy points per box, U-build 96 -> 39 s, optimization
  163 -> 106 s. 2D: 58 -> 33 points, optimization 17.3 -> 15.4 s. Cache files are now
  `proxy_v2_<dim>d_f<bits>_tol<tol>.txt` (`TSNE_PROXY_TOL` overrides for experiments).
- `TSNE_H2_TIMERS=1` also prints `H2RANK max=... mean=...` (H2 rank over all nodes and rebuilds).

### Benchmarks
- 3D, out of the box on GloVe 250K / 8 threads: the H2 rank is
  ~59 (2D: 17) and the U-build dominates the rebuild; `h2_leaf_size` does not help in 3D
  (100 = 200 < 400 < 800), the proxy tolerance does (above). FIt-SNE's FFT path is 1D/2D only
  and returns garbage for `no_dims=3` without an error (negative "cost"); its Barnes-Hut mode
  takes 1138.8 s end-to-end (KL 4.900, 10-NN 0.213) vs 147.9 s for tsne-h2pack (KL 4.616,
  10-NN 0.231): 7.7x end-to-end, 10.3x on the optimization.
- `BENCH_DIM=3 python sweep.py`: the whole suite in 3D (`results_3d/`), FIt-SNE and openTSNE
  on their Barnes-Hut gradient. Zheng 50K-1M, shared P, 1 and 8 threads: tsne-h2pack optimization
  4.5-14x (1 thread) / 9.6-24x (8 threads) faster than FIt-SNE Barnes-Hut, KL 2-4% lower,
  10-NN preservation higher at every size; 1M / 8 threads 362 s vs 5475 s.
- Where the speedup over FIt-SNE comes from (GloVe 250K, one P):
  at 1 thread the advantage over FIt-SNE is the reuse alone (H2 per evaluation ~15% slower than the
  FFT step); at 8 threads 1.7x from H2's parallel scaling x 2.4x from reuse. FIt-SNE's FFT stage is
  serial (Amdahl serial fraction 0.32). In 3D H2 is 3.7-5x faster per evaluation than Barnes-Hut.


## [3.1.1] - 2026-09-20

Quality fix for the default k-NN path; anyone who ran 3.1.0 with the default `knn_method` on
more than 10,000 samples should re-run.

### Fixed
- **`knn_method='pynndescent'` (the default above 10,000 samples) listed every point as its own
  nearest neighbor.** `NNDescent.neighbor_graph` returns the query point in each row (distance 0);
  the wrapper kept it, so P got a distance-0 self entry per row and one real neighbor fewer than
  requested. The self entry absorbed a large share of the row's perplexity budget: on GloVe-100
  (20K), common-P KL 4.49 -> 3.87 and 10-NN preservation 0.16 -> 0.24 after the fix (FIt-SNE:
  3.80 / 0.24). Exact, ball-tree, FAISS and Annoy paths and `metric='precomputed'` were not
  affected. The backend now also strips explicit diagonal entries from any k-NN graph it is given.
- The k-NN CSR is column-sorted before the perplexity search on every path, so the same graph
  gives the same P whether it comes from a k-NN method (rows ordered by distance) or from
  `metric='precomputed'` (rows ordered by index).

### Performance
- `verbose=1` no longer costs ~180 s at 1M points: the progress KL was a serial pass over all of
  P every 10 iterations; it now runs on all threads at the 50-iteration print cadence.

### Benchmarks
- `benchmarks/oob_compare.py` + `oob_visualize.py`: out-of-the-box comparison (each library with
  its own defaults). GloVe-100, 1.18M points: tsne-h2pack 3.02x faster end-to-end at 8 threads
  (276.9 vs 835.8 s), 1.86x at 1 thread; common-P KL 5.93 vs 6.23 and exact 10-NN preservation
  0.151 vs 0.124 in its favor, because FIt-SNE's Annoy defaults reach 48.7% recall on this data
  vs 89.1% for pynndescent. `benchmarks/results/oob_glove/REPORT.md`.


## [3.1.0] - 2026-09-20

First release on PyPI. The optimization phase is 2.0-3.4x faster than FIt-SNE on one thread
and 4.3-8.8x on eight at matched embedding quality (same k-NN graph and P; Zheng 10x,
50K-1M points; `benchmarks/results/REPORT.md`). Two correctness fixes below change results
relative to 3.0.x, as do the new defaults for `learning_rate`, `h2_tol` and `h2_rebuild_freq`.

### Fixed
- **Perplexity calibration collapsed to a uniform P for large-magnitude distances.**
  The sigma search clamped the exponent at -700; when squared k-NN distances were
  large relative to the starting sigma (e.g. MNIST in 0-255 pixel units) every
  neighbor hit the clamp, the row looked uniform, and the bisection walked the wrong
  way, so P came out as exactly `1/k` per neighbor (perplexity = k instead of 30).
  Now bisects on beta with min-shifted distances; P matches openTSNE's to 2e-6.
- **Divergence when the H2 matrix is not rebuilt every iteration.** Between rebuilds
  the stale kernel weights were multiplied by the current Y, so the repulsion of a
  pair grew linearly with how far it had separated since the build. The repulsive
  force from the last rebuild is now reused as a whole until the next rebuild
  (bounded, slightly delayed gradient), which also skips the matvec on those iterations.

### Packaging
- The sdist no longer picks up binaries left in `tsne_h2pack/_lib/` by a local build
  (`MANIFEST.in`); wheels still get the extension and `libH2Pack.so` from `build_ext`.
- `release_preflight.sh`: version consistency, submodule reachability, full pytest,
  sdist + wheel build, `twine check`, delocate, fresh-venv wheel and sdist installs.
- `wheels.yml`: manual dispatch can upload the artifacts to TestPyPI as a dry run.

### Changed
- `metric='precomputed'`: `fit_transform` accepts a sparse matrix of k-NN distances (as sklearn's
  TSNE does) and skips the neighbor search, so runs and libraries can share one graph.
- `knn_method='pynndescent'` (the default) uses exact brute-force search below 10,000
  samples, unless `n_trees` / `n_iters` are passed explicitly. NN-descent is at its worst
  when k = 3 x perplexity is a large share of a small dataset, and the graph was poor
  enough to show in the embedding (Zheng 5K: common-P KL 1.61 -> 1.44, kNN-preservation
  0.40 -> 0.44, now on par with FIt-SNE).
- Benchmark suite rewritten as one suite under `benchmarks/` (`config.py`, `sweep.py`,
  `run.py`, `methods.py`, `data.py`, `quality.py`, `report.py`; see its README). One
  configuration per library (no "fast" variants); every library on its own `'auto'` learning-rate rule;
  all three t-SNE libraries optimize the **same P** from one shared pynndescent k-NN graph
  (FIt-SNE loads it as precomputed affinities); common-P quality gate; k-NN / setup /
  optimization breakdown at 1 and 8 threads; minimum size 50K. `benchmarks_opentsne/` and the
  `examples/benchmark_threading_*` scripts are removed; their old results are not comparable.
- `learning_rate` defaults to `'auto'` = `max(n_samples / 12, 200)` (FIt-SNE's rule; note this gradient carries the sklearn factor 4, FIt-SNE's does not).
- `h2_tol` default `1e-4` -> `1e-2`; `h2_rebuild_freq` default `2` -> `3`.
- New `h2_rebuild_freq_early` (default 10): rebuild frequency during early exaggeration.
- New `h2_leaf_size` (default 100; H2Pack's 2D default was 200).

### Performance
- Proxy points (data-independent, 7 s to select on one thread) are cached per user in
  `~/.cache/tsne_h2pack` (`TSNE_H2PACK_CACHE_DIR` overrides, `TSNE_H2PACK_NO_PROXY_CACHE=1`
  disables); each run works on a private copy. Fixed start-up cost 8.2 s -> 0.2 s on one thread.
- Fused 3-column H2 matvec (`src/tsne_h2_matmul.c`): `W*[1|Y0|Y1]` in one traversal,
  no BLAS calls on the rank-~10 far-field blocks (matvec 2x faster at 1 thread, 3.5x at 8).
- 2D attractive pass: interleaved Y gather, register accumulation, blocked vectorized
  divide (2.4x faster at 1 thread).
- `(1+r^2)^-2` evaluated directly rather than through H2Pack's generic `pow()` kernel.
- Points are renumbered along the H2 tree order six times during the run so the attractive
  pass gathers from nearby memory (1.8x faster attractive pass at 500K); undone on output.
- The attractive pass divides in float32 (differences still formed in double): ~15% faster,
  embedding quality unchanged.
- H2Pack (submodule): pooled large B/D blocks across rebuilds (peak RSS 21.6 -> 10.2 GB at
  500K, which also makes 1M points fit in 32 GB), dynamically scheduled B/D build, inlined
  BLAS-1 in the ID's pivoted QR.
- H2 rebuild, second round (all results bit-identical unless noted; 500K points, 8 threads):
  - H2Pack tree partition built with OpenMP tasks + chunked stable bucket sort, post-order
    numbering in a serial post-pass: same tree for any thread count (8.9 -> 2.4 s).
  - H2Pack ID: no one-thread OpenMP team per Householder step, lock-free `free_aligned`
    for unpooled blocks (U build 12.1 -> 9.0 s); parallel frees in `H2P_destroy` (1.3 -> 0.6 s).
  - B/D blocks are no longer stored: each H2 matrix is multiplied once, so the fused matvec
    evaluates them on the fly into a cache-resident buffer (`TSNE_H2_JIT=0` restores AOT).
    Rebuild + matvec -2.0 s, peak RSS 10.3 -> 5.9 GB.
  - Root enclosing box and NaN/Inf check fused into one pass (1.26 -> 0.24 s); proxy point
    sets cached across rebuilds instead of re-read from / re-written to the text file on
    every rebuild (also fixes a leak of one proxy point set per rebuild).
  - Point renumbering: parallel CSR transposes (six renumberings 4.3 -> 1.6 s).
  - Per-iteration element-wise passes run on all threads; mean of Y and Z are summed
    block-wise in parallel with thread-count-independent rounding (rounding differs from
    the old serial sums: embeddings agree to 3e-15 after 5 iterations, quality unchanged).

- Attractive pass, third round (it had become the largest phase: 41% of the optimization at
  500K / 8 threads, half of it at 1 thread). Kernel now in `src/tsne_attr_2d.h`:
  - float32 copy of P's values (the divide already ran in float32): 12 -> 8 bytes streamed per
    nonzero, no per-nonzero conversion; refreshed after each renumbering. +4 bytes/nnz of memory
    until `DTYPE` is float, when `P->values` is used directly.
  - neighbors gathered from a float32 copy of Y (`TSNE_ATTR_F32=2`, new default; `1` = previous
    arithmetic, `0` = all double). Equivalent to the exact force of the embedding rounded to
    float32 (displacement <= 3.8e-6 per coordinate for |y| < 128, the same in every pair);
    measured force error 2.5e-5 relative, quality unchanged. Falls back to the double gather
    for max|y| >= 4096.
  - one (x, y) copy per neighbor, de-interleaved inside the vector loop; blocks of 16.
  - loops written with pointers / `size_t` counters: setuptools adds CPython's
    `-fno-strict-overflow`, which made the `int`-indexed gather loop 30% slower in the
    extension than in a standalone build of the same source.
  - dynamic chunk `n/(64*threads)` in [64, 1024].

Zheng et al. mouse-brain data, optimization only (k-NN excluded), Apple M1 Pro, 1000 iterations,
same learning-rate rule, quality measured with a common P:

| N    | threads | tsne-h2pack | FIt-SNE | speedup | KL (h2pack / FIt-SNE) | kNN-pres@10 (h2pack / FIt-SNE) |
|------|---------|-------------|---------|---------|-----------------------|--------------------------------|
| 250K | 8       | 26 s        | 81 s    | 3.1x    | 3.92 / 3.93           | 0.205 / 0.203                  |
| 500K | 8       | 53 s        | 137 s   | 2.6x    | 4.48 / 4.50           | 0.158 / 0.155                  |
| 1M   | 8       | 113 s       | 354 s   | 3.1x    | 5.03 / 5.04           | 0.116 / 0.114                  |
| 250K | 1       | 99 s        | 146 s   | 1.5x    | 3.92 / 3.93           | 0.204 / 0.203                  |
| 500K | 1       | 200 s       | 306 s   | 1.5x    | 4.48 / 4.50           | 0.157 / 0.155                  |
| 1M   | 1       | 414 s       | 715 s   | 1.7x    | 5.03 / 5.04           | 0.117 / 0.114                  |

After the second round (same harness and machine; the table above is the state before it;
500K / 1M at 1 thread not re-run):

| N    | threads | opt before -> after | rebuild + matvec before -> after | peak RSS        | KL / kNN-pres@10 after |
|------|---------|---------------------|----------------------------------|-----------------|------------------------|
| 250K | 8       | 26.3 -> 22.4 s      | 14.7 + 3.4 -> 8.3 + 4.9 s        | 4.9 -> 3.4 GB   | 3.92 / 0.203           |
| 500K | 8       | 53.1 -> 39.5 s      | 28.3 + 6.4 -> 13.7 + 8.4 s       | 10.1 -> 5.9 GB  | 4.48 / 0.157           |
| 1M   | 8       | 113.4 -> 82.4 s     | 54.7 + 12.4 -> 23.6 + 16.0 s     | 18.3 -> 10.9 GB | 5.03 / 0.116           |
| 250K | 1       | 98.9 -> 94.1 s      | 46.5 + 12.7 -> 33.8 + 20.4 s     | 4.9 -> 3.4 GB   | 3.92 / 0.205           |

(With JIT blocks the B/D evaluation moves from "rebuild" into "matvec". Wall time improves by
a further ~3 s at 500K: the renumbering is not covered by the per-phase timers.)

After the third round (attractive pass; same harness and machine):

| N    | threads | opt before -> after | attractive before -> after | peak RSS        | KL / kNN-pres@10 after |
|------|---------|---------------------|----------------------------|-----------------|------------------------|
| 250K | 8       | 22.4 -> 15.7 s      | ~7 -> 3.6 s                | 3.4 -> 3.6 GB   | 3.92 / 0.204           |
| 500K | 8       | 38.5 -> 29.9 s      | 16.2 -> 7.7 s              | 5.9 -> 6.2 GB   | 4.48 / 0.159           |
| 1M   | 8       | 82.4 -> 58.7 s      | ~41 -> 17.9 s              | 10.9 -> 11.4 GB | 5.03 / 0.117           |
| 250K | 1       | 94.1 -> 73.7 s      | ~38 -> 18.8 s              | 3.4 -> 3.6 GB   | 3.92 / 0.203           |
| 500K | 1       | -> 144.3 s          | 86 -> 41.4 s               | 6.0 GB          | 4.48 / 0.158           |

(Before this round: 136 s for 250K on 8 threads, at a fixed learning rate of 200 that left
the embedding far less converged -- KL 4.58, kNN-preservation 0.10. openTSNE's FFT takes
282 s at 250K: its macOS wheel is built without OpenMP.)

## [3.0.1] - 2026-07-06

Prepared for PyPI but never uploaded; the first upload was 3.1.0. (Version 3.0.0 and an
earlier 3.0.1 were tagged on 2026-02-11, also never uploaded; their changes are consolidated
into this entry.)

### Added
- GitHub Actions CI:
  - `test.yml` — pytest matrix on Linux + macOS (Apple Silicon) for Python 3.10 and 3.12
  - `wheels.yml` — `cibuildwheel` builds cp39–cp313 wheels for manylinux x86-64 and
    macOS arm64, plus the sdist, and publishes to PyPI via **trusted publishing** on `v*` tags
- `docs/TROUBLESHOOTING.md` — comprehensive troubleshooting guide

### Changed
- Tightened `MANIFEST.in` — sdist trimmed from ~4.8 MB to ~489 KB (excludes H2Pack
  examples/extras and built artifacts; keeps only sources `setup.py` compiles)
- Wheels now bundle their native dependencies (`libgomp`, `libopenblas`, `libH2Pack`)
  via delocate (macOS) / auditwheel (Linux), so they import without a system
  Homebrew/OpenBLAS install
- `_backend.py` loads bundled libraries from `_lib/` first, falling back to system libraries
- Rewrote the developer guide to reflect the current repo layout (correct paths, `tests/` at root,
  `uv pip install`, accurate file tree)
- Simplified README.md; removed broken doc links (`CONTRIBUTING.md`, root-level `PACKAGING_GUIDE.md`)
- Updated `pyproject.toml` classifiers (`Development Status :: 4 - Beta`; added License,
  Operating System, and Python 3.13) and fixed repository URLs (`tsne-h2pack` → `tsne-h2pack-python`)
- Linux install docs now include `liblapack-dev liblapacke-dev` (required by `setup.py`'s linker flags)
- Consolidated packaging docs (folded BUILD_STATUS.md, READY_FOR_PYPI.md, TESTPYPI_GUIDE.md
  into `docs/PACKAGING_GUIDE.md`)

### Fixed
- macOS: added the SDK System library path so H2Pack links correctly
- P matrix construction when the affinity exponent is too negative
- macOS: set `KMP_DUPLICATE_LIB_OK` so the bundled `libgomp` and `libomp` coexist
  in one process without OpenMP's "already initialized" abort
- Plumbed KL divergence through to `kl_divergence_`; hardened flaky k-NN/threading tests

## [3.0.0] - 2026-02-11 (internal)

> Note: This version was tagged internally but never uploaded to PyPI.

### 🚀 Major Changes - Breaking Release

#### Native Builds for All Platforms
- **BREAKING**: Removed Docker/Colima backend (~1,400 lines of code)
- **NEW**: Native C extension builds on macOS (Apple Silicon) and Linux
- **BREAKING**: Intel Mac no longer supported in v3.0 (use v2.x or Rosetta 2)
- Requires Homebrew GCC + OpenBLAS on macOS

#### Benefits
- Faster execution (no Docker overhead)
- Simpler setup (no Docker/Colima installation required)
- Native performance with OpenMP threading

### 🔧 Technical Improvements

#### Build System
- Fixed CC environment variable handling in `setup.py`
- Fixed macOS SDK library path for System library
- Improved H2Pack library loading from `_lib/` directory
- Better compiler detection and validation

#### H2Pack Threading
- Critical fix: Set `omp_set_num_threads()` before `H2P_init()`
- H² Rebuild speedup: 1.04x → 4.69x
- H² Matvec speedup: 1.01x → 2.47x
- Overall optimization speedup: 2.06x → 3.54x (8 threads)

#### Package Structure
- Created `tsne_h2pack/_lib/` for compiled extensions
- Proper packaging of `.so` and `.dylib` files
- Both development and packaged modes supported

### 📦 Distribution
- Local sdist (4.6 MB) and macOS arm64 wheel (267 KB) build successfully
- PyPI upload deferred — see `docs/PACKAGING_GUIDE.md` for remaining work
- Thoroughly tested with 12K dataset smoke tests

### 🧪 Testing
- Automated build script: `build_package.sh`
- Automated installation test: `test_installation.sh`
- 12K dataset smoke test validation
- 5K MNIST example working perfectly (5.2s)

### 📚 Documentation
- Created `PACKAGING_GUIDE.md` - Detailed packaging guide

### 🔄 Migration from v2.x

#### For Apple Silicon Users
```bash
# Remove Docker-based installation
pip uninstall tsne-h2pack

# Install prerequisites
brew install gcc openblas

# Install v3.0.0
pip install tsne-h2pack
```

#### For Intel Mac Users
- v3.0 requires Apple Silicon (ARM64)
- Options:
  1. Use v2.x with Docker support
  2. Use Rosetta 2 to run ARM64 version
  3. Use Linux or remote server

### ⚠️ Known Issues
- macOS wheel shows version warning (cosmetic, works on macOS 12.0+)
- Default k-NN method requires `pip install pynndescent` or use `knn_method='exact'`

### 🎯 Requirements

#### macOS (Apple Silicon)
- Homebrew GCC (gcc-15, gcc-14, gcc-13, or gcc-12)
- OpenBLAS library
- Python 3.8+

#### Linux
- GCC with OpenMP support
- OpenBLAS or other BLAS/LAPACK library
- Python 3.8+

---

## [2.0.0] - 2026-02-01

### Changed - Unified k-NN Interface

#### Breaking API Change
- **BREAKING**: Replaced `use_sklearn_knn` parameter with `knn_method`
- **NEW**: 5 k-NN methods available: 'pynndescent' (default), 'exact', 'balltree', 'faiss', 'annoy'

#### Migration
```python
# Old API (v1.x)
tsne = TSNE(use_sklearn_knn=True, n_jobs=8)

# New API (v2.0+)
tsne = TSNE(knn_method='exact', n_jobs=8)
```

### Added
- PyNNDescent as default k-NN method (96% accuracy, 10-100x faster)
- FAISS support for large datasets (>50K samples)
- Annoy support for memory-constrained environments
- Comprehensive k-NN benchmarks and documentation

### Files
- `tsne_h2pack/_knn_methods.py` - Unified k-NN interface
- `docs/KNN_METHODS.md` - Complete k-NN guide
- `examples/HNSW_BENCHMARK_RESULTS.md` - Performance benchmarks

---

## [1.0.0] - Initial Release

### Features
- H2Pack acceleration for O(N log N) complexity
- Docker/Colima backend for macOS
- Native backend for Linux
- scikit-learn compatible API
- OpenMP parallelization

---

## Version History Summary

- **v3.3.1** (2026-11-09): Python 3.10+ required, 3.10-3.14 tested in CI, 3.14 wheels
- **v3.3.0** (2026-11-05): PCA initialization by default, benchmarks rerun on an Apple M5 Max, macOS wheels need macOS 15
- **v3.2.0** (2026-10-07): Per-dimension H2 defaults, 3D speedups
- **v3.1.1** (2026-09-20): Fix for the PyNNDescent self-neighbor bug of 3.1.0
- **v3.1.0** (2026-09-20): First release on PyPI; optimization round, new benchmark suite
- **v3.0.1** (2026-07-06): CI wheels, trusted publishing, packaging polish (not uploaded)
- **v3.0.0** (2026-02-11): Native builds, Docker removed, P-matrix fix (not uploaded)
- **v2.0.0** (2026-02-01): Unified k-NN interface, 5 methods
- **v1.0.0**: Initial release with Docker backend

---
