# How to use tsne-h2pack

## Threads

`n_jobs` defaults to **one thread**, as in scikit-learn. Set it to your number of physical cores, or `n_jobs=-1` for all cores. It controls the k-NN search, the optimizer, and H2Pack's OpenMP threads together.

## Large datasets

tsne-h2pack's advantage grows with N and is largest from about 50K points up. Below about 10K points any t-SNE library is fast, and scikit-learn works fine.

```python
from tsne_h2pack import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",      # max(N / 12, 200)
    knn_method="pynndescent",  # default
    n_jobs=8,
    verbose=1,                 # progress and KL every 50 iterations
)
Y = tsne.fit_transform(X)      # X: (n_samples, n_features), ideally PCA-reduced
```

At 1M points about half of the end-to-end time is the k-NN search, not the optimization. If you embed the same data several times, compute the graph once and reuse it (see [Precomputed k-NN graphs](#precomputed-k-nn-graphs)).

**Memory.** Peak memory grows linearly with N, to about 16 GB at 1M points in both 2D and 3D.

## Perplexity

Perplexity sets roughly how many neighbors each point pays attention to. The default of 30 works for most data; values between 5 and 50 are common, and larger datasets often benefit from larger values. The k-NN search uses 3 × perplexity neighbors, so higher perplexity also costs more time and memory.

## Choosing a k-NN method

| `knn_method` | Kind | Notes |
|---|---|---|
| `"pynndescent"` | approximate, about 90–96% recall | Default. Uses exact search below 10K samples. |
| `"exact"` | scikit-learn brute force | Exact, slow for large N. Useful to rule out graph error. |
| `"balltree"` | scikit-learn ball tree | Exact. Works best in low input dimension. |
| `"faiss"` | FAISS | Approximate. Needs `faiss-cpu`. |
| `"annoy"` | Annoy | Approximate, low memory. Needs `annoy`. |

Keyword arguments for the backend go in `knn_params`. For example, `knn_params={"n_trees": 30, "n_iters": 10}` makes PyNNDescent more accurate and slower. Annoy accepts `n_trees` and `search_k`, and FAISS accepts `nlist` and `nprobe`.

!!! note "The graph matters more than the optimizer"
    A low-recall k-NN graph degrades the embedding more than any of the gradient approximations do. On GloVe word vectors (1.18M points), PyNNDescent's defaults recover 89% of the true 90 nearest neighbors, while a fast Annoy configuration recovers 49%, with visibly worse neighbor preservation in the embedding.

## Precomputed k-NN graphs

Pass a SciPy sparse matrix of k-NN distances with `metric="precomputed"`, as in scikit-learn. The neighbor search is skipped, so several runs, or several libraries, can share one graph.

```python
from sklearn.neighbors import NearestNeighbors
from tsne_h2pack import TSNE

k = 90                                     # 3 x perplexity
knn = NearestNeighbors(n_neighbors=k).fit(X).kneighbors_graph(mode="distance")
Y = TSNE(metric="precomputed", perplexity=30, n_jobs=8, init="random").fit_transform(knn)
```

Each row needs more than `perplexity` neighbors. Three times the perplexity is the usual choice. Do not include the point itself as a neighbor.

A graph has no features for the default PCA initialization. Pass `init="random"`, as above, or your own starting embedding, for example `init=PCA(n_components=2).fit_transform(X) / 1e4`. Otherwise tsne-h2pack falls back to random initialization and warns.

## 3D embeddings

Set `n_components=3`. The same algorithm runs, because the H² matrix works in both 2D and 3D.

```python
Y3 = TSNE(n_components=3, n_jobs=8).fit_transform(X)
```

Expect roughly 3–8× the 2D optimization time at the same N, depending on the size and the thread count. The kernel's numerical rank is higher in 3D: about 40 at the 3D default tolerance, against about 10–17 in 2D. FFT-accelerated t-SNE is limited to 1D and 2D, so in 3D other libraries use Barnes-Hut trees.

## H² parameters

Most users should not change these. In 2D the defaults reach the same embedding quality as FFT-accelerated t-SNE on the same P.

| Parameter | Default | Effect |
|---|---|---|
| `h2_tol` | `1e-2` in 2D, `3e-2` in 3D | Relative accuracy of the H² approximation. Tighter costs time without measurable quality gain. |
| `h2_rebuild_freq` | `3` | Rebuild the H² matrix every this many iterations after early exaggeration, reusing the repulsive force in between. `2` is slightly more accurate; `6` and above is visibly worse. |
| `h2_rebuild_freq_early` | `10` | Rebuild frequency during early exaggeration, which is insensitive to it. |
| `h2_leaf_size` | `100` | Points per leaf box. Speed and memory only. |

Leaving a parameter at `None` selects the default for the embedding dimension. Why rebuilding every few iterations is safe, and why the force is reused rather than recomputed, is explained in [How it works](technical/algorithm.md#reusing-the-repulsive-force).

## Exact repulsion

`method="exact"` computes the repulsive force directly, without the H² matrix. It costs O(N²) per iteration, so use it only for small data or to check results. Datasets below 2,000 points use the direct computation automatically.

## Learning rate

`learning_rate="auto"` is `max(N / 12, 200)`. A fixed 200, the old scikit-learn default, under-converges badly beyond about 100K points. scikit-learn's own `"auto"` uses a smaller step, `max(N / 48, 50)`.

!!! warning "Comparing with other libraries"
    tsne-h2pack follows scikit-learn and keeps the factor 4 in the gradient. Libraries that drop it take a 4× smaller step at the same nominal learning rate.

## Reproducibility

With a fixed `random_state`, single-threaded runs are bit-for-bit reproducible. Multi-threaded runs are not: the result is statistically equivalent but differs between runs. See [Accuracy and reproducibility](technical/accuracy.md#reproducibility).

## Profiling

`enable_profiling=True` records a per-iteration breakdown into attractive forces, H² rebuild, and H² product. `profiling_output_file` writes it to CSV.
