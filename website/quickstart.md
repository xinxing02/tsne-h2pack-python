# Quick start

`tsne_h2pack.TSNE` follows the scikit-learn interface: create an estimator, then call `fit_transform`.

```python
from sklearn.datasets import make_blobs
from tsne_h2pack import TSNE

# 20,000 points in 50 dimensions, drawn from 10 clusters
X, labels = make_blobs(n_samples=20_000, n_features=50, centers=10, random_state=0)

tsne = TSNE(n_components=2, perplexity=30, n_jobs=8, random_state=42, verbose=1)
Y = tsne.fit_transform(X)                # (20000, 2)

print(tsne.kl_divergence_, tsne.n_iter_)
```

`verbose=1` prints progress and the KL divergence every 50 iterations. `n_jobs` defaults to one thread, so set it to your core count.

## Differences from scikit-learn

- Use `fit_transform`. `fit` alone is not implemented, and there is no `transform` for new points.
- `init` defaults to `"pca"`, as in scikit-learn: the PCA of `X`, rescaled so the first coordinate
  has standard deviation 1e-4. `"random"` and an `(n_samples, n_components)` array are also
  accepted. With `metric="precomputed"` there is no `X` to project, so `"pca"` falls back to
  random initialization with a warning, where scikit-learn raises an error.
- `learning_rate="auto"` means `max(N / 12, 200)`, a larger step than scikit-learn's own `"auto"`.
- `metric` accepts `"euclidean"`, or `"precomputed"` with a sparse k-NN distance matrix.

## A real dataset

This example downloads MNIST from OpenML the first time it runs, which can take a minute or two.

```python
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tsne_h2pack import TSNE

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = PCA(n_components=50).fit_transform(mnist.data / 255.0)
labels = mnist.target.astype(int)

Y = TSNE(n_jobs=8, random_state=0, verbose=1).fit_transform(X)

plt.figure(figsize=(7, 7))
plt.scatter(Y[:, 0], Y[:, 1], c=labels, s=0.3, cmap="tab10")
plt.axis("off")
plt.show()
```

!!! note "Reduce to about 50 dimensions first"
    Like other t-SNE implementations, tsne-h2pack works best on PCA-reduced input. It speeds up the k-NN search and removes noise without changing the embedding much.

## What to read next

- [How to use tsne-h2pack](user-guide.md) covers threads, large datasets, perplexity, k-NN backends, precomputed graphs, 3D, and parameters.
- [How it works](technical/algorithm.md) explains how the H² matrix makes each iteration linear in N.
- [Benchmarks](benchmarks.md) shows timing and quality from 50K to 1M points.
