# tsne-h2pack: High-Performance t-SNE with H2Pack Acceleration

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, sklearn-compatible implementation of t-SNE with H2Pack acceleration for large-scale dimensionality reduction.

---

## Features

- **Drop-in sklearn replacement**: Same API as `sklearn.manifold.TSNE`
- **Fast on large datasets**: the repulsive forces are computed in O(N) with an H² matrix
  (H2Pack). On the same k-NN graph and P matrix, and at matched embedding quality, the
  optimization phase is 2.1–4.0× faster than FIt-SNE on one thread, 2.9–9.7× on four,
  3.9–11.4× on eight and 4.7–11.6× on twelve (Zheng 10x, 50K–1M points, Apple M5 Max; see
  `benchmarks/results/REPORT.md`)
- **Multiple k-NN methods**: PyNNDescent (default), exact, ball tree, FAISS, and Annoy;
  or pass your own k-NN graph with `metric='precomputed'`
- **Validated at scale**: 1M points in 70 seconds end-to-end (k-NN included) on 12 cores
  of an Apple M5 Max laptop, 2.5 minutes on an 8-core M1 Pro
- **Native performance**: OpenMP parallelization on Linux and macOS

---

## Quick Start

### Installation

```bash
pip install tsne-h2pack
```

Binary wheels are published for Linux x86-64 (manylinux) and macOS Apple Silicon
(macOS 15+), CPython 3.9–3.13. On other platforms, including macOS 14, `pip` builds from the source
distribution, which needs a C compiler and OpenBLAS — see [Prerequisites](#prerequisites).

To install from a source checkout:

```bash
git clone --recursive https://github.com/xinxing02/tsne-h2pack-python.git
cd tsne-h2pack-python
pip install .
```

### Basic Usage

```python
from tsne_h2pack import TSNE
import numpy as np

# Generate sample data
X = np.random.randn(1000, 50)

# Same API as sklearn
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
Y = tsne.fit_transform(X)

print(f"Embedding shape: {Y.shape}")
```


---

## Advanced Usage

### Large Datasets

```python
from tsne_h2pack import TSNE

# For large datasets (>50K samples)
tsne = TSNE(
    n_components=2,
    perplexity=30,
    method='h2pack',         # Use H2Pack acceleration
    knn_method='pynndescent', # Fast approximate k-NN
    n_jobs=8,                # Use 8 CPU cores
    verbose=1
)

Y = tsne.fit_transform(X_large)
```

### k-NN Methods

Choose the best k-NN method for your dataset size:

```python
# Small datasets (<5K): exact k-NN
tsne = TSNE(knn_method='exact', n_jobs=4)

# Medium datasets (5K-50K): PyNNDescent (default)
tsne = TSNE(knn_method='pynndescent')

# Large datasets (>50K): FAISS
tsne = TSNE(knn_method='faiss')
```

**Available k-NN methods:**
- `'pynndescent'` - Fast approximate (default, ~96% accuracy)
- `'exact'` - sklearn brute-force (100% accurate, slow)
- `'balltree'` - sklearn ball tree (100% accurate, medium speed)
- `'faiss'` - Facebook FAISS (very fast approximate)
- `'annoy'` - Spotify Annoy (memory-efficient approximate)

Install optional k-NN methods as extras:
```bash
pip install 'tsne-h2pack[knn]'           # All methods
pip install 'tsne-h2pack[pynndescent]'   # Or a single method
pip install 'tsne-h2pack[faiss]'
pip install 'tsne-h2pack[annoy]'
```
(From a source checkout, replace `tsne-h2pack` with `.`.) Below 10,000 samples the
default `'pynndescent'` falls back to exact brute-force search automatically.

---

## Parameters

All `sklearn.manifold.TSNE` parameters are supported:

```python
TSNE(
    n_components=2,          # Output dimensions
    perplexity=30.0,         # Balance local/global structure
    early_exaggeration=12.0, # Exaggeration factor for the first 250 iterations
    learning_rate='auto',    # 'auto' = max(n_samples / 12, 200), or a float
    max_iter=1000,           # Maximum iterations
    random_state=None,       # Random seed
    init='pca',              # default, as in scikit-learn; or 'random', or an (n_samples, n_components) array
    metric='euclidean',      # 'euclidean', or 'precomputed' (sparse k-NN distances)
    method='h2pack',         # 'h2pack' or 'exact'
    n_jobs=None,             # Number of parallel jobs
    knn_method='pynndescent', # k-NN method (see above)
    verbose=0                # Verbosity level
)
```

`metric='precomputed'` takes a SciPy sparse matrix of k-NN distances (as
`sklearn.manifold.TSNE` does) and skips the neighbor search, so several runs or
several libraries can share one graph. Each row needs more than `perplexity`
neighbors; `3 * perplexity` is the usual choice.

**H2Pack-specific parameters** (default `None` = the tuned value for `n_components`;
`tsne_h2pack.h2_defaults(n_components)` shows them):
- `h2_tol` (float; 1e-2 in 2D, 3e-2 in 3D): relative accuracy of the H² approximation
  of the repulsive kernel. The defaults match FIt-SNE's embedding quality (KL divergence,
  neighbor preservation); tighter tolerances cost time without measurable gain. In 3D the
  kernel's rank is ~3x that of 2D at equal tolerance, so the 3D default is looser; 1e-1 is
  another ~20% faster for ~0.5% KL and suits interactive exploration.
- `h2_rebuild_freq` (int, default 3): the H² matrix is rebuilt every this many
  iterations after early exaggeration; the repulsive force is reused in between.
  2 is slightly more accurate and slower; 6 and above is visibly less accurate (2D and 3D).
- `h2_rebuild_freq_early` (int, default 10): rebuild frequency during early exaggeration.
- `h2_leaf_size` (int, default 100): points per leaf of the H² tree. Only for tuning
  experiments; time is flat between 50 and 150 in both 2D and 3D.

---

## Installation Details

### Prerequisites

**macOS (Apple Silicon):**
```bash
brew install gcc openblas
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install gcc libopenblas-dev liblapack-dev liblapacke-dev

# RHEL/CentOS
sudo yum install gcc openblas-devel lapack-devel
```

**Windows:**
Not supported. Use WSL2 (Windows Subsystem for Linux).

### Building from Source

```bash
git clone --recursive https://github.com/xinxing02/tsne-h2pack-python.git
cd tsne-h2pack-python
pip install .
```

If you cloned without `--recursive`, fetch the H2Pack submodule:
```bash
git submodule update --init --recursive
```

---

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ Fully supported | Recommended for production |
| **macOS (Apple Silicon)** | ✅ Fully supported | Requires Homebrew GCC |
| **macOS (Intel)** | ❌ Not supported | Use Linux, or WSL2 / a Linux VM |
| **Windows** | ❌ Not supported | Use WSL2 |

---

## Troubleshooting

### Common Issues

**ImportError: No module named 'pynndescent'**
```bash
pip install pynndescent
# or use exact k-NN: TSNE(knn_method='exact')
```

**macOS: Build fails**
```bash
brew install gcc openblas
```

**Backend not available**
```bash
pip install --force-reinstall --no-deps .
```

For more issues, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

---

## Documentation

- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Development guide: building, testing, implementation invariants
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Detailed troubleshooting guide
- **[docs/PACKAGING_GUIDE.md](docs/PACKAGING_GUIDE.md)** - Package maintainer guide

---

## For Package Maintainers

### Building Distribution Packages

```bash
# Using automated build script
scripts/build_package.sh

# Or manual build
python -m build
```

### Publishing to PyPI

See [docs/PACKAGING_GUIDE.md](docs/PACKAGING_GUIDE.md) for the full release workflow.

---

## Examples

See the `examples/` directory:

```bash
# Basic demo (configurable via --n_samples)
python examples/basic_example.py

# Full MNIST (60K samples)
python examples/basic_example.py --n_samples 60000

# Benchmark suite (tsne-h2pack vs FIt-SNE vs openTSNE vs UMAP): see benchmarks/README.md
python benchmarks/sweep.py --dry-run
```

---

## Citation & License

### Citation

If you use tsne-h2pack in your research, please cite:

```bibtex
@software{tsne_h2pack,
  title = {tsne-h2pack: High-Performance t-SNE with H2Pack Acceleration},
  author = {Xing, Xin},
  year = {2026},
  url = {https://github.com/xinxing02/tsne-h2pack-python}
}
```

And the H2Pack library:
```bibtex
@article{huang2020toms,
    title = { {H2Pack}: High-performance \textit{{H}} $^{\textrm{2}}$ Matrix Package for Kernel Matrices Using the Proxy Point Method },
    journal = {ACM Transactions on Mathematical Software},
    author = {Huang, Hua and Xing, Xin and Chow, Edmond},
    year = {2020},
    month = {Dec},
    volume = {47},
    pages = {1--29},
    doi = {10.1145/3412850},
    issn = {0098-3500, 1557-7295},
    number = {1},
}
```


### License

MIT License - see [LICENSE](LICENSE) file for details.


---

**Version**: 3.3.0
**Status**: Beta; released on PyPI
