"""
Tests for the perplexity calibration in compute_P_from_knn_distances.

The conditional distribution P_j|i is invariant to a global rescaling of the
input distances (sigma simply rescales with them), and every row must hit the
requested perplexity. Large-magnitude distances (e.g. MNIST in 0-255 pixel
units) used to collapse every row to the uniform distribution 1/k.
"""

import ctypes

import numpy as np
import pytest

from tsne_h2pack._backend import DTYPE, SparseMatrix, get_backend


def _conditional_P(distances, perplexity):
    """Row-wise P_j|i for an (n, k) array of k-NN distances (neighbor ids are irrelevant)."""
    lib = get_backend()._lib
    n, k = distances.shape
    data = np.ascontiguousarray(distances.ravel(), dtype=np.float64)
    indices = np.ascontiguousarray(np.tile(np.arange(k, dtype=np.int32), n))
    indptr = np.arange(0, (n + 1) * k, k, dtype=np.int32)
    ptr = lib.compute_P_from_knn_distances(
        data.ctypes.data_as(ctypes.POINTER(DTYPE)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n, n * k, ctypes.c_double(perplexity), 1,
    )
    mat = ctypes.cast(ptr, ctypes.POINTER(SparseMatrix))
    values = np.ctypeslib.as_array(mat.contents.values, shape=(n * k,)).copy()
    lib.sparse_matrix_destroy(mat)
    return values.reshape(n, k)


def _row_perplexity(P):
    H = -np.sum(np.where(P > 0, P * np.log2(np.maximum(P, 1e-300)), 0.0), axis=1)
    return 2.0 ** H


@pytest.fixture(scope="module")
def knn_distances():
    rng = np.random.default_rng(0)
    return np.sort(rng.uniform(1.0, 3.0, size=(200, 90)), axis=1)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 1e6])
def test_rows_hit_requested_perplexity(knn_distances, scale):
    P = _conditional_P(knn_distances * scale, perplexity=30.0)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=1e-10)
    np.testing.assert_allclose(_row_perplexity(P), 30.0, rtol=1e-3)


@pytest.mark.parametrize("scale", [1e-3, 1e3, 1e6])
def test_calibration_is_scale_invariant(knn_distances, scale):
    P_ref = _conditional_P(knn_distances, perplexity=30.0)
    P_scaled = _conditional_P(knn_distances * scale, perplexity=30.0)
    np.testing.assert_allclose(P_scaled, P_ref, rtol=1e-3, atol=1e-9)
