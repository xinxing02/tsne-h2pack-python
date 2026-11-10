# Benchmark suite

tsne-h2pack vs FIt-SNE vs openTSNE vs UMAP, on one and on several threads, with a
time breakdown (k-NN graph, setup, optimization) and a quality check that makes the
three t-SNE timings comparable.

```bash
source ../.venv/bin/activate
uv pip install openTSNE fitsne umap-learn pynndescent pandas matplotlib
python sweep.py --dry-run     # what would run
python sweep.py               # resumable: finished cells are skipped
python report.py              # results/REPORT.md, summary.csv, figures (works on a partial sweep)
```

A quick partial run: `BENCH_SIZES=50000 BENCH_THREADS=8 BENCH_REPS=1 python sweep.py`.
One cell by hand: `python run.py tsne-h2pack --n 250000 --threads 8 --rep 0`
(after `python run.py knn --n 250000 --threads 8 --rep 0 --k 90`).

## Layout

| File | Role |
|---|---|
| `config.py` | sizes, thread counts, reps, the common protocol. The only file to edit for a different sweep |
| `data.py` | datasets, deterministic subsamples, the shared pynndescent k-NN graphs (cached in `_cache/`) |
| `methods.py` | one adapter per library, all returning `(embedding, t_setup, t_optimize)` |
| `quality.py` | common-P KL and kNN-preservation |
| `run.py` | one cell in one fresh process → one JSON line in `results/runs.jsonl` |
| `sweep.py` | the driver |
| `report.py` | tables and figures |
| `hero_zheng.py` | the whole 1.3M-cell Zheng dataset with defaults (the website's hero figure) |
| `oob_compare.py`, `oob_fitsne_sharedknn.py`, `oob_visualize.py` | the GloVe out-of-the-box comparison: each library with its own k-NN graph and defaults (`results/oob_glove/`) |

## Rules of the comparison

- **One configuration per library**: its defaults under a common protocol (perplexity 30,
  1000 iterations, 250 of them at exaggeration 12 with momentum 0.5 then 0.8, random
  initialization). There are no "fast" variants.
- **Learning rate**: each library's own `'auto'` rule. For tsne-h2pack and FIt-SNE that is
  max(200, N/12) for the whole run; openTSNE's is N/exaggeration per `optimize()` call, i.e.
  N/12 during early exaggeration and N afterwards. The rules are nominal, not identical steps:
  tsne-h2pack's gradient carries the factor 4 (sklearn convention), FIt-SNE's and openTSNE's do
  not, and both of those clip the update norm at 5. Timings do not depend on the step size
  (every run does exactly 1000 iterations); quality is judged on the common P, and N/48 for
  tsne-h2pack (FIt-SNE's effective step) was measured slightly worse. A fixed 200 leaves
  large embeddings far from converged after 1000 iterations, so timings taken at 200 compare
  different amounts of optimization.
- **Same input**: a given (size, rep) selects the same points in every process; reps draw
  different subsets, so the spread includes data-sampling variance.
- **Same k-NN graph, same P**: one pynndescent graph (k = 90) per (size, rep, threads).
  tsne-h2pack takes it through `metric='precomputed'`, openTSNE through
  `PrecomputedNeighbors`, and **FIt-SNE loads the resulting P** as precomputed affinities
  (`load_affinities='load'`), so all three optimize identical affinities and the
  optimization column differs only in how the gradient is computed.
- **The k-NN constraint**: FIt-SNE's wrapper accepts finished affinities but not a k-NN
  graph, so for the end-to-end column its front end is measured on its own: its bundled
  Annoy search plus perplexity calibration (`FIt-SNE-native-knn`, one rep per size). pip
  `annoy` cannot be the shared method here because it crashes on a second index build
  under Python 3.13.
- **UMAP** runs its defaults (`n_neighbors=15`, spectral initialization, its own epoch
  count) on a k = 15 graph built by the same pynndescent code path. It optimizes a
  different objective; its quality scores are shown for reference only.
- **Quality gate**: every embedding is scored against one common P (KL, with Z estimated
  from 1000 anchor rows) and by kNN-preservation@10. The report flags any size where the
  three t-SNE libraries are not within 3% KL and 0.015 kNN-preservation of the best one.
  A library's self-reported KL uses its own P and is not comparable.
- **Timing hygiene**: one fresh process per cell; thread counts set through the library
  API and `OMP_NUM_THREADS` / `NUMBA_NUM_THREADS`; numba's JIT compilation (pynndescent,
  UMAP: about 7 s per process, independent of N) is paid on a toy input before the timer
  starts and reported separately; tsne-h2pack's proxy points come from its per-user cache,
  as they do for any user after their first run.

## 3D sweep

`BENCH_DIM=3 python sweep.py` runs the same protocol with 3-D embeddings and writes to
`results_3d/`. Only the adapters change: tsne-h2pack and UMAP get `n_components=3`; FIt-SNE
and openTSNE have no FFT gradient beyond 2D (FIt-SNE's pip wrapper skips the C++ guard and
silently returns garbage for `no_dims=3` with FFT), so they run their Barnes-Hut gradient.
3D defaults (`config.py`): one rep per cell (Barnes-Hut is deterministic and 2D spread was
1-3%), no openTSNE (Barnes-Hut without OpenMP on macOS: 830 s at 50K, ~6 h at 1M), and
no FIt-SNE 1M cell on one thread (`SKIP_ABOVE`; that run is ~7 h). Results:
`results_3d/REPORT.md`. Notable in an earlier run on an M1 Pro: FIt-SNE's Barnes-Hut
time is not monotone in N (its 250K cell is slower per point than 500K at both thread
counts), i.e. it depends on the tree the embedding produces, not only on N.

## Platform notes

- openTSNE's macOS wheel is built without OpenMP, so its FFT optimization does not speed
  up with threads on a Mac. On Linux it does.
- **On Linux** the suite runs unchanged: install the build packages from the
  [installation guide](https://xinxing02.github.io/tsne-h2pack-python/installation/)
  (`gcc libopenblas-dev liblapack-dev liblapacke-dev` on Ubuntu / Debian), then
  `pip install -e .` and the benchmark dependencies above. Set `BENCH_THREADS` to match the
  machine (e.g. `BENCH_THREADS=1,8,16,32`). Run the sweep under `nohup` or `tmux` on a
  remote machine; it is resumable either way.
- Apple silicon mixes performance (P) and efficiency (E) cores, and macOS gives no way to
  pin a thread to a core: the scheduler places them. The Nov-2026 results (`results/`,
  `results_3d/`) are from an Apple M5 Max (12 P + 6 E) at 1, 4, 8 and 12 threads. Any work the
  scheduler put on E cores only makes a time longer, so the times are upper bounds.
