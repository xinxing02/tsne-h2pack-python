"""
Pytest configuration and shared fixtures for tsne-h2pack tests.
"""

import numpy as np
import pytest
from typing import Tuple


# Dataset size markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "small: tests on small datasets (<=1K samples)")
    config.addinivalue_line("markers", "medium: tests on medium datasets (1K-10K samples)")
    config.addinivalue_line("markers", "large: tests on large datasets (>10K samples)")
    config.addinivalue_line("markers", "requires_backend: tests that require native backend")


# ============================================================================
# Data Fixtures
# ============================================================================

_ANNOY_PROBE = """
import annoy
index = annoy.AnnoyIndex(4, 'euclidean')
for i in range(50):
    index.add_item(i, [float(i), float(i % 7), float(i % 3), 1.0])
index.build(5)
print(len(index.get_nns_by_item(0, 10)))
"""
_annoy_probe_result = None


def _skip_if_annoy_broken():
    """Skip when the installed Annoy crashes or returns too few neighbors (cached)."""
    global _annoy_probe_result
    if _annoy_probe_result is None:
        import subprocess
        import sys
        try:
            r = subprocess.run([sys.executable, "-c", _ANNOY_PROBE],
                               capture_output=True, text=True, timeout=60)
            ok = r.returncode == 0 and r.stdout.strip() == "10"
            _annoy_probe_result = "" if ok else (
                f"exit code {r.returncode}, output {r.stdout.strip()!r}")
        except subprocess.TimeoutExpired:
            _annoy_probe_result = "timed out"
    if _annoy_probe_result:
        pytest.skip("Annoy is installed but broken in this environment "
                    f"({_annoy_probe_result})")


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def small_dataset(random_seed) -> np.ndarray:
    """Small synthetic dataset for quick tests (100 samples, 50 features)."""
    np.random.seed(random_seed)
    return np.random.randn(100, 50).astype(np.float64)


@pytest.fixture
def medium_dataset(random_seed) -> np.ndarray:
    """Medium synthetic dataset (500 samples, 50 features)."""
    np.random.seed(random_seed)
    return np.random.randn(500, 50).astype(np.float64)


@pytest.fixture
def large_dataset(random_seed) -> np.ndarray:
    """Larger synthetic dataset (2000 samples, 100 features)."""
    np.random.seed(random_seed)
    return np.random.randn(2000, 100).astype(np.float64)


@pytest.fixture
def clustered_dataset(random_seed) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic dataset with clear cluster structure.

    Cluster centers placed deterministically on orthogonal axes with a
    large scale, so pairwise center distance is guaranteed (was: random
    N(0,1)*10 centers, which occasionally produced overlapping clusters
    when two random draws landed close — making variance-comparison
    tests flake).

    Returns (X, labels) tuple with 5 clusters.
    """
    np.random.seed(random_seed)
    n_per_cluster = 100
    n_features = 50
    n_clusters = 5
    scale = 20.0  # pairwise inter-center distance = scale * sqrt(2) ≈ 28

    X = []
    labels = []

    for i in range(n_clusters):
        # Center along the i-th coordinate axis: deterministic, well-separated.
        center = np.zeros(n_features)
        center[i] = scale
        cluster_points = center + np.random.randn(n_per_cluster, n_features)
        X.append(cluster_points)
        labels.extend([i] * n_per_cluster)

    X = np.vstack(X).astype(np.float64)
    labels = np.array(labels)

    # Shuffle
    indices = np.random.permutation(len(labels))
    return X[indices], labels[indices]


@pytest.fixture
def tiny_dataset(random_seed) -> np.ndarray:
    """Tiny dataset for edge case tests (10 samples, 5 features)."""
    np.random.seed(random_seed)
    return np.random.randn(10, 5).astype(np.float64)


# ============================================================================
# TSNE Fixtures
# ============================================================================

@pytest.fixture
def default_tsne_params() -> dict:
    """Default parameters for TSNE tests."""
    return {
        'n_components': 2,
        'perplexity': 30.0,
        'learning_rate': 200.0,
        'max_iter': 250,  # Reduced for faster tests
        'random_state': 42,
        'verbose': 0,
    }


@pytest.fixture
def fast_tsne_params() -> dict:
    """Fast parameters for quick smoke tests."""
    return {
        'n_components': 2,
        'perplexity': 5.0,  # Small for tiny datasets
        'learning_rate': 200.0,
        'max_iter': 50,  # Very few iterations
        'random_state': 42,
        'verbose': 0,
    }


# ============================================================================
# k-NN Fixtures
# ============================================================================

@pytest.fixture
def knn_test_data(random_seed) -> Tuple[np.ndarray, int]:
    """
    Data for k-NN testing (200 samples, 20 features).
    Returns (X, k) tuple.
    """
    np.random.seed(random_seed)
    X = np.random.randn(200, 20).astype(np.float64)
    k = 15
    return X, k


# ============================================================================
# Backend Fixtures
# ============================================================================

@pytest.fixture
def backend():
    """Get the tsne-h2pack backend."""
    try:
        from tsne_h2pack._backend import get_backend
        return get_backend()
    except Exception as e:
        pytest.skip(f"Backend not available: {e}")


@pytest.fixture
def backend_available() -> bool:
    """Check if native backend is available."""
    try:
        from tsne_h2pack._backend import get_backend
        backend = get_backend()
        return backend.is_available()
    except Exception:
        return False


# ============================================================================
# Skip Conditions
# ============================================================================

@pytest.fixture
def skip_without_backend(backend_available):
    """Skip test if backend is not available."""
    if not backend_available:
        pytest.skip("Native backend not available")


@pytest.fixture
def skip_without_pynndescent():
    """Skip test if PyNNDescent is not installed."""
    try:
        import pynndescent
    except ImportError:
        pytest.skip("PyNNDescent not installed")


@pytest.fixture
def skip_without_faiss():
    """Skip test if FAISS is not installed."""
    try:
        import faiss
    except ImportError:
        pytest.skip("FAISS not installed")


@pytest.fixture
def skip_without_annoy():
    """Skip test if Annoy is not installed or broken on this Python.

    Note: the previous version of this fixture exercised Annoy with a
    quick sanity build to detect bad wheels. That itself segfaults on
    Python 3.13 because the bad wheel crashes inside `AnnoyIndex.build`
    before Python can return. We now skip statically based on Python
    version, never touching Annoy on 3.13+, until upstream ships a
    compatible wheel.
    """
    import sys
    if sys.version_info >= (3, 13):
        pytest.skip(
            "Annoy has no working wheel for Python 3.13+ (crashes inside "
            "AnnoyIndex.build with bus error). See "
            "https://github.com/spotify/annoy/issues for status."
        )
    try:
        import annoy
    except ImportError:
        pytest.skip("Annoy not installed")
    # Below 3.13 a broken build either returns too few neighbors (GitHub's macos-15
    # runner, Python 3.12) or segfaults (same runner, Python 3.9 / 3.11). Probe it once
    # in a child process, so a crash turns into a skip instead of killing pytest.
    _skip_if_annoy_broken()
