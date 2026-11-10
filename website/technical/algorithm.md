# How it works

## The t-SNE gradient

For an embedding \(y_1, \dots, y_N\), t-SNE uses the Cauchy kernel

\[
w_{ij} = \frac{1}{1 + \lVert y_i - y_j \rVert^2}, \qquad
q_{ij} = \frac{w_{ij}}{Z}, \qquad
Z = \sum_{k \neq l} w_{kl}.
\]

The gradient of the KL divergence splits into an attractive and a repulsive part:

\[
\frac{\partial C}{\partial y_i}
= 4 \Big( \underbrace{\sum_j p_{ij}\, w_{ij}\, (y_i - y_j)}_{\text{attractive}}
\;-\; \underbrace{\frac{1}{Z} \sum_{j \neq i} w_{ij}^2\, (y_i - y_j)}_{\text{repulsive}} \Big).
\]

**The attractive part** only involves pairs where \(p_{ij} \neq 0\). P comes from a k-NN graph with k = 3 × perplexity, so this is a sparse pass costing O(kN).

**The repulsive part** couples every pair of points. Computed directly it is O(N²) per iteration, which is the reason fast t-SNE methods exist.

## One kernel gives both force and normalization

Let \(K\) be the \(N \times N\) matrix with \(K_{ij} = (1 + \lVert y_i - y_j \rVert^2)^{-2} = w_{ij}^2\) off the diagonal and zero on it. The repulsive force is a matrix product:

\[
F^{\text{rep}}_i = \frac{1}{Z}\Big( y_i\, s_i - v_i \Big),
\qquad s = K\mathbf{1}, \quad v = K Y.
\]

The normalization \(Z\) needs \(w_{ij}\), not \(w_{ij}^2\). Writing \(w_{ij} = (1 + r_{ij}^2)\, K_{ij}\) and expanding \(r_{ij}^2 = \lVert y_i\rVert^2 + \lVert y_j\rVert^2 - 2\, y_i \cdot y_j\) gives

\[
Z = \sum_i s_i \;+\; 2 \sum_i \lVert y_i \rVert^2 s_i \;-\; 2 \sum_i y_i \cdot v_i .
\]

So a single product \(K \,[\,\mathbf{1} \mid Y\,]\), with \(d+1\) right-hand sides in \(d\) dimensions, yields the full repulsive force and \(Z\).

## H² matrices

\(K\) is dense but smooth away from the diagonal. An **H² matrix**, built here with [H2Pack](https://github.com/scalable-matrix/H2Pack),[^h2pack] exploits that:

1. **Partition.** The points are split recursively into a tree of boxes, down to leaves of about 100 points.
2. **Near field.** Blocks between neighboring boxes are evaluated exactly.
3. **Far field.** A block between two well-separated boxes is low-rank: \(K_{IJ} \approx U_I\, B_{IJ}\, U_J^{\top}\).
4. **Nesting.** Each parent's basis \(U\) is expressed through its children's bases, so the bases are never stored in full.

Nesting is what separates H² matrices from the simpler H-matrices and from Barnes-Hut trees. For a fixed accuracy, both the construction and the matrix-vector product cost **O(N)**, not O(N log N).

## Reusing the repulsive force

The embedding moves slowly, so the H² matrix does not need rebuilding every iteration. tsne-h2pack rebuilds it every iteration for the first 20 iterations, every 10th for the rest of early exaggeration, and every 3rd afterwards.

The subtle part is what to do between rebuilds.

!!! warning "The tempting approach diverges"
    Keeping the old H² matrix and multiplying it by the *current* positions looks harmless. It is not. The matrix holds kernel weights for the old positions, so the repulsion between two points grows linearly with how far they have separated since the rebuild, instead of decaying. Outliers get flung outward and the embedding blows up. On 250K mouse brain cells, the embedding grew to a width of 1243 units instead of 235.

tsne-h2pack instead **reuses the whole repulsive force and Z** from the last rebuild until the next one. The force is slightly delayed but bounded, and the product is skipped entirely on those iterations.

| Rebuild the H² matrix every | Embedding quality |
|---|---|
| 2 iterations | slightly better than the default |
| 3 iterations (default) | matches FFT-accelerated t-SNE on the same P |
| 6+ iterations | visibly worse |

On 250K GloVe word vectors, reuse saves a factor of about 2.4 in optimization time for about 1% in KL.

## Comparison with other methods

| Method | Repulsive force cost per iteration | Embedding dimension | Used by |
|---|---|---|---|
| Exact | O(N²) | any | scikit-learn `method="exact"` |
| Barnes-Hut tree | O(N log N) | any, slow beyond 2D | scikit-learn, openTSNE, FIt-SNE |
| FFT interpolation on a grid | O(N), grid grows with embedding size | 1D and 2D only | FIt-SNE, openTSNE |
| H² matrix | O(N) | any | tsne-h2pack |

FFT-accelerated t-SNE needs a grid that covers the whole embedding, so in 3D its size grows with the cube of the embedding's width. Existing FFT-accelerated libraries implement it only in 1D and 2D and use Barnes-Hut in 3D. The H² matrix adapts to where the points are and has no such limit.

## The rest of the pipeline

- **k-NN graph.** PyNNDescent by default, k = 3 × perplexity.
- **P matrix.** Per-point Gaussian bandwidths are found by bisection on the precision so that each row has the requested perplexity. The search works on min-shifted squared distances, which keeps it stable for inputs with large magnitudes, such as raw MNIST pixels. P is then symmetrized as \(P + P^{\top}\) and normalized to sum to 1.
- **Optimization.** Gradient descent with momentum 0.5, then 0.8 after iteration 250, and per-coordinate adaptive gains. Early exaggeration multiplies P by 12 for the first 250 iterations. The embedding is re-centered each iteration.

[^h2pack]: H. Huang, X. Xing, E. Chow. *H2Pack: High-performance H² matrix package for kernel matrices using the proxy point method.* ACM Transactions on Mathematical Software 47(1), 2020. [doi:10.1145/3412850](https://doi.org/10.1145/3412850)
