# Installation

## From PyPI

```bash
pip install tsne-h2pack
```

Binary wheels are published for:

| Platform | Python | Notes |
|---|---|---|
| Linux x86-64 (manylinux) | CPython 3.9–3.14 | Recommended for production |
| macOS 15+ on Apple Silicon | CPython 3.9–3.14 | Built with GCC and GNU OpenMP. On macOS 14, pip builds from source |

Only these two platforms are supported. On any other platform pip falls back to the source distribution, and the build fails or is untested:

- **Linux on ARM (aarch64)** does not build yet, because of a compiler issue in H2Pack's ARM vector code.
- **Intel Macs** are not supported.
- **Windows** is not supported natively. Use WSL2, which counts as Linux x86-64.

## Optional k-NN backends

The default k-NN method, PyNNDescent, and the alternatives are optional extras:

```bash
pip install 'tsne-h2pack[knn]'          # all k-NN backends
pip install 'tsne-h2pack[pynndescent]'  # or one at a time
pip install 'tsne-h2pack[faiss]'
pip install 'tsne-h2pack[annoy]'
```

Without PyNNDescent, pass `knn_method='exact'` or `'balltree'`, which only need scikit-learn.

## Building from source

You only need this on a supported platform without a matching wheel, for example a different Python version. First install a C compiler and OpenBLAS:

**macOS (Apple Silicon)**

```bash
brew install gcc openblas
```

The build uses Homebrew GCC and refuses Apple Clang, whose OpenMP runtime has race conditions with H2Pack on Apple Silicon.

**Ubuntu / Debian**

```bash
sudo apt-get install gcc libopenblas-dev liblapack-dev liblapacke-dev
```

**RHEL / CentOS**

```bash
sudo yum install gcc openblas-devel lapack-devel
```

Then build from the source distribution on PyPI, which includes H2Pack:

```bash
pip install --no-binary tsne-h2pack tsne-h2pack
```

## Check the installation

```python
import tsne_h2pack
from tsne_h2pack._backend import get_backend

print(tsne_h2pack.__version__)
print(get_backend().is_available())   # True
```
