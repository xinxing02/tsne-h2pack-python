"""
tsne-h2pack: High-performance t-SNE with H2Pack acceleration

This package provides a drop-in replacement for sklearn.manifold.TSNE
with H2Pack acceleration for improved performance on large datasets.

Platform Support:
- Linux: Native C extension with OpenMP support
- macOS: Native C extension with Homebrew GCC (Apple Silicon only)
- Windows: Not supported (use WSL2)

Example:
    >>> from tsne_h2pack import TSNE
    >>> import numpy as np
    >>> X = np.random.randn(1000, 50)
    >>> tsne = TSNE(n_components=2, perplexity=30)
    >>> Y = tsne.fit_transform(X)
"""

__version__ = "3.3.0"

# On macOS, our wheel ships both libgomp (used by Homebrew GCC for our
# C extension and H2Pack) and libomp (a transitive dep of Homebrew's
# OpenBLAS). When the consumer process has already initialized libomp
# via numpy/scipy/sklearn, loading our copy trips OpenMP's safety
# check ("OMP Error #15: ... already initialized") and SIGABRTs.
# Set the standard workaround flag so the second init is allowed.
# Documented as safe for our case by Intel/numpy/scipy because both
# runtimes implement the same OpenMP v3.1 API. Use setdefault so we
# don't override a user who has explicitly set this.
import os as _os
import sys as _sys
if _sys.platform == "darwin":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .manifold import TSNE, H2_DEFAULTS, h2_defaults

__all__ = ["TSNE", "H2_DEFAULTS", "h2_defaults", "__version__"]
