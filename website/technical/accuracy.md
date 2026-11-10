# Accuracy and reproducibility

## Why a loose tolerance is enough

The default `h2_tol` asks for about 1% relative accuracy in the kernel approximation in 2D, and 3% in 3D. That sounds loose, but the gradient is already noisy for other reasons:

- P is built from an approximate k-NN graph with about 90–96% recall.
- Force reuse delays the repulsive force by up to two iterations.
- FFT-accelerated t-SNE also approximates the kernel, by interpolation on a grid.

Tighter tolerances cost time and give no measurable improvement in KL or neighbor preservation. In 2D, at 1e-2, tsne-h2pack's embeddings match those of FFT-accelerated t-SNE on the same P, from 50K to 1M points. In 3D, 3e-2 changes KL by only 0.1–0.2% compared with 1e-2, and the embeddings score better than Barnes-Hut t-SNE on the same P, in both KL and neighbor preservation.

## How quality is measured

Self-reported KL values cannot be compared across libraries: each one builds its own P and evaluates the objective in its own way. The benchmarks therefore score every embedding against **one common P**:

- **KL divergence** \(\sum_{ij} p_{ij} \log (p_{ij} / q_{ij})\) with the common P and the exact Q of the embedding.
- **k-NN preservation**, the fraction of each point's 10 nearest neighbors in the input that are also among its 10 nearest in the embedding.

Two t-SNE embeddings count as matched when each is within 3% in KL and 0.015 in neighbor preservation of the best one.

## Reproducibility

| Setting | Bit-reproducible with fixed `random_state`? |
|---|---|
| `n_jobs=1` | Yes |
| `n_jobs > 1` | No |

Reductions such as Z are summed in fixed blocks of 4096 points and combined in order, so their rounding does not depend on the thread count. Some parallel steps are still order-dependent, so multi-threaded runs differ between runs in the last bits. t-SNE amplifies tiny differences over 1000 iterations, so the final embeddings differ visibly while their quality stays the same. Use `n_jobs=1` when exact reproduction matters.
