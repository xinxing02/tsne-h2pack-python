# Out-of-the-box: tsne-h2pack vs FIt-SNE on `glove` (N = 1,183,514)

Each library with its own defaults (own k-NN, own perplexity search, own init, own learning-rate rule); only `max_iter=1000` is imposed on both. Quality on a common perplexity-30 P and exact 10-NN preservation of 5000 query points.

| threads | library | total wall (s) | k-NN (s) | optimization (s) | s / iteration | KL (common P) | 10-NN preservation | peak RSS (GB) |
|---|---|---|---|---|---|---|---|---|
| 1 | FIt-SNE | 1261.0 |  | 465.4 | 0.465 | 6.2258 | 0.1245 | 6.1 |
| 1 | tsne-h2pack | 788.4 | 548.5 | 211.4 | 0.211 | 5.9393 | 0.1503 | 17.2 |
| 4 | FIt-SNE | 483.8 |  | 204.3 | 0.204 | 6.2261 | 0.1236 | 6.1 |
| 4 | FIt-SNE on tsne-h2pack's k-NN graph | 393.7 | 174.0 | 202.4 | 0.202 | 5.9385 | 0.1485 |  |
| 4 | tsne-h2pack | 259.9 | 167.6 | 75.3 | 0.075 | 5.9371 | 0.1497 | 11.1 |
| 8 | FIt-SNE | 358.2 |  | 159.9 | 0.160 | 6.2261 | 0.1236 | 6.1 |
| 8 | FIt-SNE on tsne-h2pack's k-NN graph | 278.0 | 102.6 | 158.5 | 0.159 | 5.9340 | 0.1496 |  |
| 8 | tsne-h2pack | 158.6 | 98.2 | 46.9 | 0.047 | 5.9325 | 0.1493 | 17.5 |
| 12 | FIt-SNE | 323.1 |  | 150.3 | 0.150 | 6.2261 | 0.1236 | 6.1 |
| 12 | FIt-SNE on tsne-h2pack's k-NN graph | 251.2 | 77.8 | 156.8 | 0.157 | 5.9436 | 0.1490 |  |
| 12 | tsne-h2pack | 126.1 | 75.5 | 38.2 | 0.038 | 5.9365 | 0.1499 | 17.3 |

**1 thread(s)**: end-to-end tsne-h2pack is 1.60x faster; optimization phase 2.20x faster. KL 5.9393 vs 6.2258 (-4.60%), 10-NN preservation 0.1503 vs 0.1245.

**4 thread(s)**: end-to-end tsne-h2pack is 1.86x faster; optimization phase 2.71x faster. KL 5.9371 vs 6.2261 (-4.64%), 10-NN preservation 0.1497 vs 0.1236.

**8 thread(s)**: end-to-end tsne-h2pack is 2.26x faster; optimization phase 3.41x faster. KL 5.9325 vs 6.2261 (-4.72%), 10-NN preservation 0.1493 vs 0.1236.

**12 thread(s)**: end-to-end tsne-h2pack is 2.56x faster; optimization phase 3.93x faster. KL 5.9365 vs 6.2261 (-4.65%), 10-NN preservation 0.1499 vs 0.1236.

Control (oob_fitsne_sharedknn.py): FIt-SNE given tsne-h2pack's own pynndescent graph as its P (load_affinities='load'), 12 threads: optimization 156.8 s, KL 5.9436, 10-NN preservation 0.1490 — the same quality as tsne-h2pack, so the out-of-the-box gap is the k-NN graph, not the optimizer.

Settings:
- tsne-h2pack: TSNE(n_jobs=threads, random_state=rep, max_iter=1000); defaults otherwise (pynndescent k-NN, perplexity 30, random init, learning_rate='auto')
- FIt-SNE: FItSNE(X, max_iter=1000, nthreads=threads, rand_seed=rep); defaults otherwise (Annoy k-NN n_trees=50, perplexity 30, PCA init, learning_rate='auto', max_step_norm=5)
