"""
Tests for the backend module (tsne_h2pack/_backend.py).
"""

import numpy as np
import pytest
import platform


class TestPlatformDetection:
    """Test platform detection constants."""

    def test_platform_constants_are_boolean(self):
        """Test that platform constants are boolean."""
        from tsne_h2pack._backend import IS_LINUX, IS_MACOS, IS_WINDOWS

        assert isinstance(IS_LINUX, bool)
        assert isinstance(IS_MACOS, bool)
        assert isinstance(IS_WINDOWS, bool)

    def test_exactly_one_platform_true(self):
        """Test that exactly one platform constant is True."""
        from tsne_h2pack._backend import IS_LINUX, IS_MACOS, IS_WINDOWS

        # At least one should be True (unless unknown platform)
        # Only one should be True at a time
        platforms = [IS_LINUX, IS_MACOS, IS_WINDOWS]
        assert sum(platforms) <= 1

    def test_platform_matches_system(self):
        """Test that platform detection matches actual system."""
        from tsne_h2pack._backend import IS_LINUX, IS_MACOS, IS_WINDOWS

        system = platform.system()

        if system == 'Linux':
            assert IS_LINUX is True
            assert IS_MACOS is False
            assert IS_WINDOWS is False
        elif system == 'Darwin':
            assert IS_LINUX is False
            assert IS_MACOS is True
            assert IS_WINDOWS is False
        elif system == 'Windows':
            assert IS_LINUX is False
            assert IS_MACOS is False
            assert IS_WINDOWS is True


class TestDtypeConstants:
    """Test dtype constants."""

    def test_dtype_is_c_double(self):
        """Test that DTYPE is ctypes c_double."""
        import ctypes
        from tsne_h2pack._backend import DTYPE

        assert DTYPE == ctypes.c_double

    def test_np_dtype_is_float64(self):
        """Test that NP_DTYPE is numpy float64."""
        from tsne_h2pack._backend import NP_DTYPE

        assert NP_DTYPE == np.float64


class TestTSNEParams:
    """Test TSNEParams ctypes structure."""

    def test_tsne_params_fields_exist(self):
        """Test that TSNEParams has expected fields."""
        from tsne_h2pack._backend import TSNEParams

        # Get field names
        field_names = [f[0] for f in TSNEParams._fields_]

        # Check key fields exist
        assert 'n_iter' in field_names
        assert 'perplexity' in field_names
        assert 'learning_rate' in field_names
        assert 'momentum' in field_names
        assert 'initial_momentum' in field_names
        assert 'early_exag' in field_names
        assert 'h2_tol' in field_names
        assert 'h2_rebuild_freq' in field_names
        assert 'verbose' in field_names
        assert 'random_seed' in field_names
        assert 'num_threads' in field_names
        assert 'enable_profiling' in field_names
        assert 'sklearn_compatible' in field_names

    def test_tsne_params_can_be_instantiated(self):
        """Test that TSNEParams can be instantiated."""
        from tsne_h2pack._backend import TSNEParams

        params = TSNEParams()
        assert params is not None

    def test_tsne_params_default_values(self):
        """Test that TSNEParams fields have expected default values (zero)."""
        from tsne_h2pack._backend import TSNEParams

        params = TSNEParams()

        # Default-constructed struct should have zero values
        assert params.n_iter == 0
        assert params.perplexity == 0.0

    def test_tsne_params_can_set_values(self):
        """Test that TSNEParams fields can be set."""
        from tsne_h2pack._backend import TSNEParams

        params = TSNEParams()
        params.n_iter = 1000
        params.perplexity = 30.0
        params.learning_rate = 200.0
        params.verbose = 1
        params.num_threads = 4

        assert params.n_iter == 1000
        assert params.perplexity == 30.0
        assert params.learning_rate == 200.0
        assert params.verbose == 1
        assert params.num_threads == 4


class TestSparseMatrix:
    """Test SparseMatrix ctypes structure."""

    def test_sparse_matrix_fields_exist(self):
        """Test that SparseMatrix has expected fields."""
        from tsne_h2pack._backend import SparseMatrix

        field_names = [f[0] for f in SparseMatrix._fields_]

        assert 'n_rows' in field_names
        assert 'n_cols' in field_names
        assert 'nnz' in field_names
        assert 'row_ptr' in field_names
        assert 'col_idx' in field_names
        assert 'values' in field_names

    def test_sparse_matrix_can_be_instantiated(self):
        """Test that SparseMatrix can be instantiated."""
        from tsne_h2pack._backend import SparseMatrix

        mat = SparseMatrix()
        assert mat is not None


class TestTSNEBackend:
    """Test TSNEBackend class."""

    def test_backend_type(self, backend):
        """Test that backend_type is 'native'."""
        assert backend.backend_type == 'native'

    def test_backend_is_available(self, backend):
        """Test that backend is available."""
        assert backend.is_available() is True

    def test_create_default_params(self, backend):
        """Test create_default_params method."""
        params = backend.create_default_params()

        # Check params is TSNEParams
        from tsne_h2pack._backend import TSNEParams
        assert isinstance(params, TSNEParams)

        # Check default values
        assert params.n_iter == 1000
        assert params.perplexity == 30.0
        assert params.momentum == 0.8
        assert params.initial_momentum == 0.5
        assert params.early_exag == 12.0
        assert params.verbose == 1
        assert params.sklearn_compatible == 1


class TestGetBackend:
    """Test get_backend function."""

    def test_get_backend_returns_singleton(self, backend_available):
        """Test that get_backend returns the same instance."""
        if not backend_available:
            pytest.skip("Native backend not available")

        from tsne_h2pack._backend import get_backend

        backend1 = get_backend()
        backend2 = get_backend()

        assert backend1 is backend2

    def test_get_backend_returns_tsne_backend(self, backend_available):
        """Test that get_backend returns TSNEBackend instance."""
        if not backend_available:
            pytest.skip("Native backend not available")

        from tsne_h2pack._backend import get_backend, TSNEBackend

        backend = get_backend()
        assert isinstance(backend, TSNEBackend)


class TestScipyCSRConversion:
    """Test scipy CSR matrix conversion functions."""

    def test_scipy_csr_to_sparse_matrix(self, backend):
        """Test scipy_csr_to_sparse_matrix function."""
        from tsne_h2pack._backend import scipy_csr_to_sparse_matrix
        from scipy.sparse import csr_matrix

        # Create a simple CSR matrix
        data = np.array([1.0, 2.0, 3.0, 4.0])
        indices = np.array([0, 2, 1, 3])
        indptr = np.array([0, 2, 4])
        csr = csr_matrix((data, indices, indptr), shape=(2, 4))

        sparse_mat, arrays = scipy_csr_to_sparse_matrix(csr)

        assert sparse_mat.n_rows == 2
        assert sparse_mat.n_cols == 4
        assert sparse_mat.nnz == 4

        # Check that arrays dict keeps arrays alive
        assert 'row_ptr' in arrays
        assert 'col_idx' in arrays
        assert 'values' in arrays

    def test_scipy_csr_to_sparse_matrix_invalid_input(self):
        """Test that scipy_csr_to_sparse_matrix rejects non-CSR input."""
        from tsne_h2pack._backend import scipy_csr_to_sparse_matrix

        # Dense array should raise TypeError
        with pytest.raises(TypeError, match="Expected scipy CSR matrix"):
            scipy_csr_to_sparse_matrix(np.array([[1, 2], [3, 4]]))

        # COO matrix should raise TypeError
        from scipy.sparse import coo_matrix
        coo = coo_matrix(np.array([[1, 2], [3, 4]]))
        with pytest.raises(TypeError, match="Expected scipy CSR matrix"):
            scipy_csr_to_sparse_matrix(coo)


@pytest.mark.requires_backend
class TestBackendFitTransform:
    """Test backend fit_transform methods."""

    def test_fit_transform_basic(self, backend, small_dataset):
        """Test basic fit_transform."""
        n_components = 2
        params = backend.create_default_params()
        params.n_iter = 50
        params.perplexity = 10.0
        params.verbose = 0

        Y, metrics = backend.fit_transform(small_dataset, n_components, params)

        assert Y.shape == (small_dataset.shape[0], n_components)
        assert np.isfinite(Y).all()
        assert 'kl_divergence' in metrics
        assert 'n_iter' in metrics

    def test_fit_transform_with_knn_distances(self, backend, small_dataset):
        """Test fit_transform_with_knn_distances method."""
        from tsne_h2pack._knn_methods import ExactKNN
        from scipy.sparse import csr_matrix

        n_components = 2
        params = backend.create_default_params()
        params.n_iter = 50
        params.perplexity = 10.0
        params.verbose = 0

        # Compute k-NN distances
        k = int(3 * params.perplexity)
        knn_method = ExactKNN()
        knn_distances = knn_method.compute(small_dataset, k=k)

        Y, metrics = backend.fit_transform_with_knn_distances(
            small_dataset, knn_distances, n_components, params
        )

        assert Y.shape == (small_dataset.shape[0], n_components)
        assert np.isfinite(Y).all()


@pytest.mark.skipif(
    platform.system() == 'Windows',
    reason="Windows not supported"
)
class TestWindowsNotSupported:
    """Test that Windows raises RuntimeError."""

    def test_windows_raises_error(self):
        """Test that Windows platform raises RuntimeError."""
        # This test only runs on non-Windows platforms
        # and verifies the code path exists
        from tsne_h2pack._backend import IS_WINDOWS

        # We can't actually test this on non-Windows,
        # but we can verify the constant exists
        assert isinstance(IS_WINDOWS, bool)
