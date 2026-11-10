"""
Tests for k-NN methods (tsne_h2pack/_knn_methods.py).
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix


class TestGetKNNMethod:
    """Test get_knn_method factory function."""

    def test_get_exact_method(self):
        """Test getting exact k-NN method."""
        from tsne_h2pack._knn_methods import get_knn_method, ExactKNN

        method = get_knn_method('exact')
        assert isinstance(method, ExactKNN)

    def test_get_balltree_method(self):
        """Test getting ball tree k-NN method."""
        from tsne_h2pack._knn_methods import get_knn_method, BallTreeKNN

        method = get_knn_method('balltree')
        assert isinstance(method, BallTreeKNN)

    def test_get_pynndescent_method(self):
        """Test getting PyNNDescent k-NN method."""
        from tsne_h2pack._knn_methods import get_knn_method, PyNNDescentKNN

        method = get_knn_method('pynndescent')
        assert isinstance(method, PyNNDescentKNN)

    def test_get_faiss_method(self):
        """Test getting FAISS k-NN method."""
        from tsne_h2pack._knn_methods import get_knn_method, FAISSKNN

        method = get_knn_method('faiss')
        assert isinstance(method, FAISSKNN)

    def test_get_annoy_method(self):
        """Test getting Annoy k-NN method."""
        from tsne_h2pack._knn_methods import get_knn_method, AnnoyKNN

        method = get_knn_method('annoy')
        assert isinstance(method, AnnoyKNN)

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        from tsne_h2pack._knn_methods import get_knn_method

        with pytest.raises(ValueError, match="Unknown knn_method"):
            get_knn_method('invalid')

        with pytest.raises(ValueError, match="Unknown knn_method"):
            get_knn_method('kdtree')


class TestKNNMethodBase:
    """Test KNNMethodBase abstract class."""

    def test_compute_not_implemented(self):
        """Test that base class compute raises NotImplementedError."""
        from tsne_h2pack._knn_methods import KNNMethodBase

        method = KNNMethodBase()
        X = np.random.randn(10, 5)

        with pytest.raises(NotImplementedError):
            method.compute(X, k=3)


class TestExactKNN:
    """Test ExactKNN implementation."""

    def test_compute_returns_csr_matrix(self, knn_test_data):
        """Test that compute returns scipy CSR matrix."""
        from tsne_h2pack._knn_methods import ExactKNN

        X, k = knn_test_data
        method = ExactKNN()

        result = method.compute(X, k=k)

        assert isinstance(result, csr_matrix)

    def test_output_shape(self, knn_test_data):
        """Test output matrix shape."""
        from tsne_h2pack._knn_methods import ExactKNN

        X, k = knn_test_data
        n_samples = X.shape[0]
        method = ExactKNN()

        result = method.compute(X, k=k)

        assert result.shape == (n_samples, n_samples)

    def test_nnz_count(self, knn_test_data):
        """Test that nnz equals n_samples * k."""
        from tsne_h2pack._knn_methods import ExactKNN

        X, k = knn_test_data
        n_samples = X.shape[0]
        method = ExactKNN()

        result = method.compute(X, k=k)

        assert result.nnz == n_samples * k

    def test_distances_non_negative(self, knn_test_data):
        """Test that all distances are non-negative."""
        from tsne_h2pack._knn_methods import ExactKNN

        X, k = knn_test_data
        method = ExactKNN()

        result = method.compute(X, k=k)

        assert (result.data >= 0).all()

    def test_self_exclusion(self, knn_test_data):
        """Test that diagonal (self-distance) is not included."""
        from tsne_h2pack._knn_methods import ExactKNN

        X, k = knn_test_data
        method = ExactKNN()

        result = method.compute(X, k=k)

        # Check that diagonal elements are not in the result
        # (no zero distances from a point to itself)
        assert not np.any((result.data == 0) & (result.indices == np.repeat(np.arange(result.shape[0]), k)))

    def test_knn_time_attribute(self, knn_test_data):
        """Test that knn_time attribute is set."""
        from tsne_h2pack._knn_methods import ExactKNN

        X, k = knn_test_data
        method = ExactKNN()

        result = method.compute(X, k=k)

        assert hasattr(result, 'knn_time')
        assert result.knn_time > 0

    def test_n_jobs_parameter(self, knn_test_data):
        """Test n_jobs parameter."""
        from tsne_h2pack._knn_methods import ExactKNN

        X, k = knn_test_data
        method = ExactKNN()

        # Should work with different n_jobs values
        result1 = method.compute(X, k=k, n_jobs=1)
        result2 = method.compute(X, k=k, n_jobs=2)

        # Results should be identical
        np.testing.assert_array_almost_equal(
            result1.toarray(), result2.toarray(), decimal=10
        )


class TestBallTreeKNN:
    """Test BallTreeKNN implementation."""

    def test_compute_returns_csr_matrix(self, knn_test_data):
        """Test that compute returns scipy CSR matrix."""
        from tsne_h2pack._knn_methods import BallTreeKNN

        X, k = knn_test_data
        method = BallTreeKNN()

        result = method.compute(X, k=k)

        assert isinstance(result, csr_matrix)

    def test_output_shape(self, knn_test_data):
        """Test output matrix shape."""
        from tsne_h2pack._knn_methods import BallTreeKNN

        X, k = knn_test_data
        n_samples = X.shape[0]
        method = BallTreeKNN()

        result = method.compute(X, k=k)

        assert result.shape == (n_samples, n_samples)

    def test_nnz_count(self, knn_test_data):
        """Test that nnz equals n_samples * k."""
        from tsne_h2pack._knn_methods import BallTreeKNN

        X, k = knn_test_data
        n_samples = X.shape[0]
        method = BallTreeKNN()

        result = method.compute(X, k=k)

        assert result.nnz == n_samples * k

    def test_leaf_size_parameter(self, knn_test_data):
        """Test leaf_size parameter."""
        from tsne_h2pack._knn_methods import BallTreeKNN

        X, k = knn_test_data
        method = BallTreeKNN()

        # Should work with different leaf_size values
        result1 = method.compute(X, k=k, leaf_size=30)
        result2 = method.compute(X, k=k, leaf_size=50)

        # Results should be identical (leaf_size affects performance, not results)
        np.testing.assert_array_almost_equal(
            result1.toarray(), result2.toarray(), decimal=10
        )

    def test_matches_exact_knn(self, knn_test_data):
        """Test that ball tree gives same results as exact k-NN."""
        from tsne_h2pack._knn_methods import ExactKNN, BallTreeKNN

        X, k = knn_test_data
        exact_method = ExactKNN()
        balltree_method = BallTreeKNN()

        exact_result = exact_method.compute(X, k=k)
        balltree_result = balltree_method.compute(X, k=k)

        # Both should give the same results
        np.testing.assert_array_almost_equal(
            exact_result.toarray(), balltree_result.toarray(), decimal=10
        )


class TestPyNNDescentKNN:
    """Test PyNNDescentKNN implementation."""

    @pytest.fixture(autouse=True)
    def check_pynndescent(self, skip_without_pynndescent):
        """Skip tests if PyNNDescent is not installed."""
        pass

    def test_compute_returns_csr_matrix(self, knn_test_data):
        """Test that compute returns scipy CSR matrix."""
        from tsne_h2pack._knn_methods import PyNNDescentKNN

        X, k = knn_test_data
        method = PyNNDescentKNN()

        result = method.compute(X, k=k)

        assert isinstance(result, csr_matrix)

    def test_output_shape(self, knn_test_data):
        """Test output matrix shape."""
        from tsne_h2pack._knn_methods import PyNNDescentKNN

        X, k = knn_test_data
        n_samples = X.shape[0]
        method = PyNNDescentKNN()

        result = method.compute(X, k=k)

        assert result.shape == (n_samples, n_samples)

    def test_nnz_count(self, knn_test_data):
        """Test that nnz equals n_samples * k."""
        from tsne_h2pack._knn_methods import PyNNDescentKNN

        X, k = knn_test_data
        n_samples = X.shape[0]
        method = PyNNDescentKNN()

        result = method.compute(X, k=k)

        assert result.nnz == n_samples * k

    def test_nn_descent_path_excludes_self_and_keeps_k(self):
        """The NN-descent search (forced with n_trees; the fixture's 200 points would take
        the exact fallback) lists each point as its own first neighbor. It must be dropped
        and every row must still hold k real neighbors, or P gets a distance-0 self entry."""
        from tsne_h2pack._knn_methods import PyNNDescentKNN

        rng = np.random.default_rng(0)
        X = rng.normal(size=(3000, 10)).astype(np.float32)
        k = 30
        result = PyNNDescentKNN().compute(X, k=k, n_jobs=1, n_trees=8, n_iters=5)
        rows = np.repeat(np.arange(X.shape[0]), np.diff(result.indptr))
        assert not np.any(result.indices == rows)
        assert np.all(np.diff(result.indptr) == k)
        assert result.data.min() > 0

    def test_drop_self_handles_missing_and_misplaced_self(self):
        from tsne_h2pack._knn_methods import _drop_self

        ind = np.array([[0, 5, 6], [7, 1, 8], [4, 5, 6]])      # self first / second / absent
        dist = np.array([[0.0, 1.0, 2.0], [0.5, 0.0, 2.0], [1.0, 2.0, 3.0]])
        i, d = _drop_self(ind, dist)
        assert i.tolist() == [[5, 6], [7, 8], [4, 5]]
        assert d.tolist() == [[1.0, 2.0], [0.5, 2.0], [1.0, 2.0]]

    def test_auto_tune_small_dataset(self):
        """Test auto-tuning for small datasets."""
        from tsne_h2pack._knn_methods import PyNNDescentKNN

        method = PyNNDescentKNN()
        params = method._auto_tune_params(5000, {})

        assert params['n_trees'] == 10  # Small dataset
        assert params['n_iters'] == 5

    def test_auto_tune_medium_dataset(self):
        """Test auto-tuning for medium datasets."""
        from tsne_h2pack._knn_methods import PyNNDescentKNN

        method = PyNNDescentKNN()
        params = method._auto_tune_params(50000, {})

        assert params['n_trees'] == 15  # Medium dataset
        assert params['n_iters'] == 10

    def test_auto_tune_large_dataset(self):
        """Test auto-tuning for large datasets."""
        from tsne_h2pack._knn_methods import PyNNDescentKNN

        method = PyNNDescentKNN()
        params = method._auto_tune_params(150000, {})

        assert params['n_trees'] == 20  # Large dataset
        assert params['n_iters'] == 15

    def test_custom_params_override_auto_tune(self):
        """Test that custom params override auto-tuning."""
        from tsne_h2pack._knn_methods import PyNNDescentKNN

        method = PyNNDescentKNN()
        params = method._auto_tune_params(50000, {'n_trees': 30, 'n_iters': 20})

        assert params['n_trees'] == 30
        assert params['n_iters'] == 20

    def test_high_recall(self, knn_test_data):
        """Test that PyNNDescent achieves high recall (>90%)."""
        from tsne_h2pack._knn_methods import ExactKNN, PyNNDescentKNN

        X, k = knn_test_data
        exact_method = ExactKNN()
        pynndescent_method = PyNNDescentKNN()

        exact_result = exact_method.compute(X, k=k)
        pynndescent_result = pynndescent_method.compute(X, k=k)

        # Calculate recall: proportion of true neighbors found
        n_samples = X.shape[0]
        total_correct = 0

        for i in range(n_samples):
            exact_neighbors = set(exact_result.indices[exact_result.indptr[i]:exact_result.indptr[i+1]])
            pynndescent_neighbors = set(pynndescent_result.indices[pynndescent_result.indptr[i]:pynndescent_result.indptr[i+1]])
            total_correct += len(exact_neighbors & pynndescent_neighbors)

        recall = total_correct / (n_samples * k)
        assert recall > 0.90  # Should have >90% recall


class TestFAISSKNN:
    """Test FAISSKNN implementation."""

    @pytest.fixture(autouse=True)
    def check_faiss(self, skip_without_faiss):
        """Skip tests if FAISS is not installed."""
        pass

    def test_compute_returns_csr_matrix(self, knn_test_data):
        """Test that compute returns scipy CSR matrix."""
        from tsne_h2pack._knn_methods import FAISSKNN

        X, k = knn_test_data
        method = FAISSKNN()

        result = method.compute(X, k=k, index_type='Flat')

        assert isinstance(result, csr_matrix)

    def test_output_shape(self, knn_test_data):
        """Test output matrix shape."""
        from tsne_h2pack._knn_methods import FAISSKNN

        X, k = knn_test_data
        n_samples = X.shape[0]
        method = FAISSKNN()

        result = method.compute(X, k=k, index_type='Flat')

        assert result.shape == (n_samples, n_samples)

    def test_nnz_count(self, knn_test_data):
        """Test that nnz equals n_samples * k."""
        from tsne_h2pack._knn_methods import FAISSKNN

        X, k = knn_test_data
        n_samples = X.shape[0]
        method = FAISSKNN()

        result = method.compute(X, k=k, index_type='Flat')

        assert result.nnz == n_samples * k

    def test_flat_index_exact(self, knn_test_data):
        """Test that Flat index gives exact results."""
        from tsne_h2pack._knn_methods import ExactKNN, FAISSKNN

        X, k = knn_test_data
        exact_method = ExactKNN()
        faiss_method = FAISSKNN()

        exact_result = exact_method.compute(X, k=k)
        faiss_result = faiss_method.compute(X, k=k, index_type='Flat')

        # Flat index should give same neighbors (distances might differ slightly due to float32)
        for i in range(X.shape[0]):
            exact_neighbors = set(exact_result.indices[exact_result.indptr[i]:exact_result.indptr[i+1]])
            faiss_neighbors = set(faiss_result.indices[faiss_result.indptr[i]:faiss_result.indptr[i+1]])
            assert exact_neighbors == faiss_neighbors

    def test_invalid_index_type(self, knn_test_data):
        """Test that invalid index_type raises ValueError."""
        from tsne_h2pack._knn_methods import FAISSKNN

        X, k = knn_test_data
        method = FAISSKNN()

        with pytest.raises(ValueError, match="Unknown FAISS index_type"):
            method.compute(X, k=k, index_type='invalid_index')

    def test_hnsw_index(self, knn_test_data):
        """Test HNSW index."""
        from tsne_h2pack._knn_methods import FAISSKNN

        X, k = knn_test_data
        method = FAISSKNN()

        result = method.compute(X, k=k, index_type='HNSW')

        assert isinstance(result, csr_matrix)
        assert result.shape == (X.shape[0], X.shape[0])


class TestAnnoyKNN:
    """Test AnnoyKNN implementation."""

    @pytest.fixture(autouse=True)
    def check_annoy(self, skip_without_annoy):
        """Skip tests if Annoy is not installed."""
        pass

    def test_compute_returns_csr_matrix(self, knn_test_data):
        """Test that compute returns scipy CSR matrix."""
        from tsne_h2pack._knn_methods import AnnoyKNN

        X, k = knn_test_data
        method = AnnoyKNN()

        result = method.compute(X, k=k)

        assert isinstance(result, csr_matrix)

    def test_output_shape(self, knn_test_data):
        """Test output matrix shape."""
        from tsne_h2pack._knn_methods import AnnoyKNN

        X, k = knn_test_data
        n_samples = X.shape[0]
        method = AnnoyKNN()

        result = method.compute(X, k=k)

        assert result.shape == (n_samples, n_samples)

    def test_nnz_count(self, knn_test_data):
        """Test that nnz equals n_samples * k."""
        from tsne_h2pack._knn_methods import AnnoyKNN

        X, k = knn_test_data
        n_samples = X.shape[0]
        method = AnnoyKNN()

        result = method.compute(X, k=k)

        assert result.nnz == n_samples * k

    def test_n_trees_parameter(self, knn_test_data):
        """Test n_trees parameter."""
        from tsne_h2pack._knn_methods import AnnoyKNN

        X, k = knn_test_data
        method = AnnoyKNN()

        # Should work with different n_trees values
        result1 = method.compute(X, k=k, n_trees=10)
        result2 = method.compute(X, k=k, n_trees=50)

        # Both should produce valid results
        assert result1.shape == result2.shape
        assert result1.nnz == result2.nnz

    def test_distances_non_negative(self, knn_test_data):
        """Test that all distances are non-negative."""
        from tsne_h2pack._knn_methods import AnnoyKNN

        X, k = knn_test_data
        method = AnnoyKNN()

        result = method.compute(X, k=k)

        # Filter out placeholder distances (1e10)
        valid_distances = result.data[result.data < 1e9]
        assert (valid_distances >= 0).all()


class TestKNNImportErrors:
    """Test import errors for optional k-NN libraries."""

    def test_pynndescent_import_error_message(self):
        """Test helpful error message when PyNNDescent not installed."""
        from tsne_h2pack._knn_methods import PyNNDescentKNN
        import sys

        # Temporarily remove pynndescent from imports if present
        pynndescent_module = sys.modules.pop('pynndescent', None)
        try:
            # This should raise ImportError with helpful message
            method = PyNNDescentKNN()
            X = np.random.randn(100, 10)
            # The import error will be raised on compute, not init
            try:
                import pynndescent
                # If import succeeds, we can't test the error message
                pytest.skip("PyNNDescent is installed")
            except ImportError:
                with pytest.raises(ImportError, match="PyNNDescent not installed"):
                    method.compute(X, k=5)
        finally:
            if pynndescent_module is not None:
                sys.modules['pynndescent'] = pynndescent_module

    def test_faiss_import_error_message(self):
        """Test helpful error message when FAISS not installed."""
        from tsne_h2pack._knn_methods import FAISSKNN
        import sys

        # Temporarily remove faiss from imports if present
        faiss_module = sys.modules.pop('faiss', None)
        try:
            method = FAISSKNN()
            X = np.random.randn(100, 10)
            try:
                import faiss
                pytest.skip("FAISS is installed")
            except ImportError:
                with pytest.raises(ImportError, match="FAISS not installed"):
                    method.compute(X, k=5)
        finally:
            if faiss_module is not None:
                sys.modules['faiss'] = faiss_module

    def test_annoy_import_error_message(self):
        """Test helpful error message when Annoy not installed."""
        from tsne_h2pack._knn_methods import AnnoyKNN
        import sys

        # Temporarily remove annoy from imports if present
        annoy_module = sys.modules.pop('annoy', None)
        try:
            method = AnnoyKNN()
            X = np.random.randn(100, 10)
            try:
                import annoy
                pytest.skip("Annoy is installed")
            except ImportError:
                with pytest.raises(ImportError, match="Annoy not installed"):
                    method.compute(X, k=5)
        finally:
            if annoy_module is not None:
                sys.modules['annoy'] = annoy_module


class TestAnnoyKNNValidation:
    """AnnoyKNN must fail loudly on a broken Annoy, not return a garbage graph.

    Uses a mocked Annoy so it runs on every Python (including 3.13, where the
    real annoy wheel crashes) — it exercises AnnoyKNN's defensive check, not annoy.
    """

    def test_degenerate_annoy_output_raises(self):
        """Broken Annoy that returns only the query point (the 3.13 failure
        mode) must raise a clear error pointing to pynndescent, not silently
        pad a diverged graph."""
        import sys
        import types
        from tsne_h2pack._knn_methods import AnnoyKNN

        class _BrokenIndex:
            def __init__(self, dim, metric):
                pass

            def add_item(self, i, v):
                pass

            def build(self, n_trees, **kwargs):
                pass

            def get_nns_by_item(self, i, n, include_distances=False, **kwargs):
                # Simulate the broken wheel: only the query point comes back.
                return ([i], [0.0]) if include_distances else [i]

        fake = types.ModuleType('annoy')
        fake.AnnoyIndex = _BrokenIndex
        saved = sys.modules.get('annoy')
        sys.modules['annoy'] = fake
        try:
            X = np.random.RandomState(0).randn(200, 10)
            with pytest.raises(RuntimeError, match="malfunctioning|pynndescent"):
                AnnoyKNN().compute(X, k=15)
        finally:
            if saved is not None:
                sys.modules['annoy'] = saved
            else:
                del sys.modules['annoy']

    def test_healthy_annoy_output_succeeds(self):
        """A well-behaved Annoy returning k valid neighbors must NOT trip the
        defensive check (guards against false positives)."""
        import sys
        import types
        from scipy.sparse import csr_matrix
        from tsne_h2pack._knn_methods import AnnoyKNN

        n, k = 200, 15
        rng = np.random.RandomState(1)
        X = rng.randn(n, 10)

        class _GoodIndex:
            def __init__(self, dim, metric):
                pass

            def add_item(self, i, v):
                pass

            def build(self, n_trees, **kwargs):
                pass

            def get_nns_by_item(self, i, m, include_distances=False, **kwargs):
                # Return self + k distinct others with increasing distances.
                ids = [i] + [(i + j) % n for j in range(1, m)]
                if include_distances:
                    return ids, [0.0] + [float(j) for j in range(1, m)]
                return ids

        fake = types.ModuleType('annoy')
        fake.AnnoyIndex = _GoodIndex
        saved = sys.modules.get('annoy')
        sys.modules['annoy'] = fake
        try:
            result = AnnoyKNN().compute(X, k=k)
            assert isinstance(result, csr_matrix)
            assert result.shape == (n, n)
            assert np.diff(result.indptr).max() == k
            assert not (result.data > 1e9).any()  # no padding sentinels
        finally:
            if saved is not None:
                sys.modules['annoy'] = saved
            else:
                del sys.modules['annoy']


class TestKNNEdgeCases:
    """Test edge cases for k-NN methods."""

    def test_k_equals_n_minus_one(self):
        """Test with k = n_samples - 1 (maximum k)."""
        from tsne_h2pack._knn_methods import ExactKNN

        np.random.seed(42)
        X = np.random.randn(20, 10)
        k = 19  # n - 1
        method = ExactKNN()

        result = method.compute(X, k=k)

        assert result.shape == (20, 20)
        assert result.nnz == 20 * 19

    def test_small_k(self):
        """Test with small k value."""
        from tsne_h2pack._knn_methods import ExactKNN

        np.random.seed(42)
        X = np.random.randn(100, 10)
        k = 3
        method = ExactKNN()

        result = method.compute(X, k=k)

        assert result.shape == (100, 100)
        assert result.nnz == 100 * 3

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        from tsne_h2pack._knn_methods import ExactKNN

        np.random.seed(42)
        X = np.random.randn(50, 500)  # 500 features
        k = 10
        method = ExactKNN()

        result = method.compute(X, k=k)

        assert result.shape == (50, 50)
        assert result.nnz == 50 * 10

    def test_identical_points(self):
        """Test behavior with identical points."""
        from tsne_h2pack._knn_methods import ExactKNN

        np.random.seed(42)
        X = np.zeros((50, 10))  # All points are identical
        k = 10
        method = ExactKNN()

        result = method.compute(X, k=k)

        # Should still return valid output
        assert result.shape == (50, 50)
        assert result.nnz == 50 * 10
        # All distances should be 0 (for identical points)
        assert np.allclose(result.data, 0)


class TestPyNNDescentSmallData:
    """NN-descent is poor at k = 90 of a few thousand points; small inputs go exact."""

    def test_small_dataset_gets_exact_neighbors(self):
        from tsne_h2pack._knn_methods import ExactKNN, PyNNDescentKNN

        X = np.random.default_rng(0).normal(size=(1500, 20))
        approx = PyNNDescentKNN().compute(X, 90)
        exact = ExactKNN().compute(X, 90)
        assert (approx != exact).nnz == 0

    def test_explicit_params_keep_nn_descent(self):
        pytest.importorskip("pynndescent")
        from tsne_h2pack._knn_methods import PyNNDescentKNN

        X = np.random.default_rng(0).normal(size=(1500, 20))
        graph = PyNNDescentKNN().compute(X, 30, n_trees=10, n_iters=5)
        assert graph.shape == (1500, 1500)
        assert graph.nnz == 1500 * 30
