"""
Tests for input validation across tsne-h2pack.
"""

import numpy as np
import pytest


class TestDataValidation:
    """Test input data validation."""

    def test_accepts_float64_array(self):
        """Test that float64 array is accepted."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10, perplexity=5, knn_method='exact')
        X = np.random.randn(50, 10).astype(np.float64)

        # Should not raise
        try:
            tsne._validate_data(X)
        except Exception as e:
            # Backend might not be available, but validation should pass
            if "Native library not found" not in str(e):
                # Re-raise if it's a validation error
                raise

    def test_converts_float32_to_float64(self):
        """Test that float32 array is converted to float64."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10, perplexity=5)
        X = np.random.randn(50, 10).astype(np.float32)

        X_validated = tsne._validate_data(X)

        assert X_validated.dtype == np.float64

    def test_converts_int_to_float64(self):
        """Test that integer array is converted to float64."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10, perplexity=5)
        X = np.random.randint(0, 100, size=(50, 10))

        X_validated = tsne._validate_data(X)

        assert X_validated.dtype == np.float64

    def test_converts_list_to_array(self):
        """Test that list is converted to numpy array."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10, perplexity=5)
        X_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]] * 20

        X_validated = tsne._validate_data(X_list)

        assert isinstance(X_validated, np.ndarray)
        assert X_validated.dtype == np.float64

    def test_rejects_1d_array(self):
        """Test that 1D array is rejected."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(100)

        with pytest.raises(ValueError, match="must be 2-dimensional"):
            tsne._validate_data(X)

    def test_rejects_3d_array(self):
        """Test that 3D array is rejected."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(10, 10, 10)

        with pytest.raises(ValueError, match="must be 2-dimensional"):
            tsne._validate_data(X)

    def test_rejects_single_sample(self):
        """Test that single sample is rejected."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(1, 50)

        with pytest.raises(ValueError, match="n_samples=1 should be >= 2"):
            tsne._validate_data(X)

    def test_rejects_empty_array(self):
        """Test that empty array is rejected."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(0, 50)

        with pytest.raises(ValueError, match="n_samples=0 should be >= 2"):
            tsne._validate_data(X)

    def test_rejects_nan(self):
        """Test that NaN values are rejected."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(100, 50)
        X[10, 5] = np.nan

        with pytest.raises(ValueError, match="contains NaN or inf"):
            tsne._validate_data(X)

    def test_rejects_positive_inf(self):
        """Test that positive infinity is rejected."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(100, 50)
        X[5, 10] = np.inf

        with pytest.raises(ValueError, match="contains NaN or inf"):
            tsne._validate_data(X)

    def test_rejects_negative_inf(self):
        """Test that negative infinity is rejected."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(100, 50)
        X[5, 10] = -np.inf

        with pytest.raises(ValueError, match="contains NaN or inf"):
            tsne._validate_data(X)

    def test_perplexity_vs_n_samples(self):
        """Test that perplexity >= n_samples is rejected."""
        from tsne_h2pack import TSNE

        tsne = TSNE(perplexity=100, max_iter=10)
        X = np.random.randn(50, 10)  # 50 samples, perplexity=100

        with pytest.raises(ValueError, match="perplexity=.* must be less than"):
            tsne._validate_data(X)

    def test_perplexity_equals_n_samples(self):
        """Test that perplexity == n_samples is rejected."""
        from tsne_h2pack import TSNE

        tsne = TSNE(perplexity=50.0, max_iter=10)
        X = np.random.randn(50, 10)

        with pytest.raises(ValueError, match="perplexity=.* must be less than"):
            tsne._validate_data(X)

    def test_accepts_perplexity_less_than_n_samples(self):
        """Test that perplexity < n_samples is accepted."""
        from tsne_h2pack import TSNE

        tsne = TSNE(perplexity=30.0, max_iter=10)
        X = np.random.randn(100, 10)

        # Should not raise
        X_validated = tsne._validate_data(X)
        assert X_validated.shape == (100, 10)


class TestParameterValidation:
    """Test parameter validation in TSNE init."""

    def test_valid_n_components(self):
        """Test valid n_components values."""
        from tsne_h2pack import TSNE

        # These should not raise
        TSNE(n_components=1)
        TSNE(n_components=2)
        TSNE(n_components=3)
        TSNE(n_components=10)

    def test_invalid_n_components(self):
        """Test invalid n_components values."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="n_components must be >= 1"):
            TSNE(n_components=0)

        with pytest.raises(ValueError, match="n_components must be >= 1"):
            TSNE(n_components=-1)

        with pytest.raises(ValueError, match="n_components must be >= 1"):
            TSNE(n_components=-10)

    def test_valid_perplexity(self):
        """Test valid perplexity values."""
        from tsne_h2pack import TSNE

        # These should not raise
        TSNE(perplexity=1.0)
        TSNE(perplexity=5.0)
        TSNE(perplexity=30.0)
        TSNE(perplexity=100.0)
        TSNE(perplexity=0.1)  # Small but positive

    def test_invalid_perplexity(self):
        """Test invalid perplexity values."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="perplexity must be > 0"):
            TSNE(perplexity=0)

        with pytest.raises(ValueError, match="perplexity must be > 0"):
            TSNE(perplexity=-1)

        with pytest.raises(ValueError, match="perplexity must be > 0"):
            TSNE(perplexity=-30.0)

    def test_valid_learning_rate(self):
        """Test valid learning_rate values."""
        from tsne_h2pack import TSNE

        # These should not raise
        TSNE(learning_rate=1.0)
        TSNE(learning_rate=10.0)
        TSNE(learning_rate=200.0)
        TSNE(learning_rate=1000.0)
        TSNE(learning_rate=0.001)  # Small but positive

    def test_invalid_learning_rate(self):
        """Test invalid learning_rate values."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            TSNE(learning_rate=0)

        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            TSNE(learning_rate=-1)

        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            TSNE(learning_rate=-200.0)

    def test_valid_max_iter(self):
        """Test valid max_iter values."""
        from tsne_h2pack import TSNE

        # These should not raise
        TSNE(max_iter=1)
        TSNE(max_iter=100)
        TSNE(max_iter=1000)
        TSNE(max_iter=10000)

    def test_invalid_max_iter(self):
        """Test invalid max_iter values."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            TSNE(max_iter=0)

        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            TSNE(max_iter=-1)

        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            TSNE(max_iter=-100)

    def test_valid_method(self):
        """Test valid method values."""
        from tsne_h2pack import TSNE

        # These should not raise
        TSNE(method='h2pack')
        TSNE(method='exact')

    def test_invalid_method(self):
        """Test invalid method values."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="method=.*not supported"):
            TSNE(method='barnes_hut')

        with pytest.raises(ValueError, match="method=.*not supported"):
            TSNE(method='invalid')

        with pytest.raises(ValueError, match="method=.*not supported"):
            TSNE(method='')

    def test_valid_metric(self):
        """Test valid metric value."""
        from tsne_h2pack import TSNE

        # Only euclidean is supported
        TSNE(metric='euclidean')

    def test_invalid_metric(self):
        """Test invalid metric values."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="metric=.*not supported"):
            TSNE(metric='manhattan')

        with pytest.raises(ValueError, match="metric=.*not supported"):
            TSNE(metric='cosine')

        with pytest.raises(ValueError, match="metric=.*not supported"):
            TSNE(metric='')

    def test_valid_init(self):
        """Test valid init values."""
        from tsne_h2pack import TSNE

        TSNE(init='random')
        TSNE(init='pca')
        TSNE(init=np.zeros((50, 2)))
        TSNE(init=np.zeros((50, 3)), n_components=3)
        TSNE(init=np.zeros((50, 2)), metric='precomputed')

    def test_pca_init_with_precomputed_is_valid(self):
        """init='pca' with metric='precomputed' is accepted (random fallback at fit)."""
        from tsne_h2pack import TSNE

        TSNE(init='pca', metric='precomputed')

    def test_pca_init_n_components_too_large(self):
        """PCA cannot produce more components than there are features."""
        from tsne_h2pack import TSNE

        X = np.random.default_rng(0).normal(size=(50, 2))
        with pytest.raises(ValueError, match="n_components"):
            TSNE(init='pca', n_components=3, perplexity=5)._initial_embedding(X, 50)

    def test_init_array_validation(self):
        """Init arrays with the wrong number of columns, rows or non-finite values."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="n_components"):
            TSNE(init=np.zeros((50, 3)))
        with pytest.raises(ValueError, match="finite"):
            TSNE(init=np.full((50, 2), np.inf))
        with pytest.raises(ValueError, match="shape"):
            TSNE(init=np.zeros((40, 2)))._initial_embedding(np.zeros((50, 5)), 50)

    def test_invalid_init(self):
        """Test invalid init values."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="init=.*not supported"):
            TSNE(init='spectral')

        with pytest.raises(ValueError, match="init=.*not supported"):
            TSNE(init='invalid')


class TestKNNMethodValidation:
    """Test k-NN method validation."""

    def test_valid_knn_methods(self):
        """Test valid k-NN method values."""
        from tsne_h2pack import TSNE

        # These should not raise
        TSNE(knn_method='exact')
        TSNE(knn_method='balltree')
        TSNE(knn_method='pynndescent')
        TSNE(knn_method='faiss')
        TSNE(knn_method='annoy')

    def test_invalid_knn_method_raises_on_get(self):
        """Test that invalid k-NN method raises error when getting."""
        from tsne_h2pack._knn_methods import get_knn_method

        with pytest.raises(ValueError, match="Unknown knn_method"):
            get_knn_method('invalid')

        with pytest.raises(ValueError, match="Unknown knn_method"):
            get_knn_method('kdtree')

        with pytest.raises(ValueError, match="Unknown knn_method"):
            get_knn_method('')

    def test_knn_params_stored(self):
        """Test that knn_params are stored correctly."""
        from tsne_h2pack import TSNE

        params = {'n_trees': 50, 'metric': 'euclidean'}
        tsne = TSNE(knn_params=params)

        assert tsne.knn_params == params

    def test_knn_params_default_empty_dict(self):
        """Test that knn_params defaults to empty dict."""
        from tsne_h2pack import TSNE

        tsne = TSNE()
        assert tsne.knn_params == {}

    def test_knn_params_none_becomes_empty_dict(self):
        """Test that knn_params=None becomes empty dict."""
        from tsne_h2pack import TSNE

        tsne = TSNE(knn_params=None)
        assert tsne.knn_params == {}


class TestEdgeCases:
    """Test edge cases in validation."""

    def test_minimum_valid_dataset(self):
        """Test minimum valid dataset (2 samples)."""
        from tsne_h2pack import TSNE

        tsne = TSNE(perplexity=0.5, max_iter=10)  # Perplexity must be < 2
        X = np.random.randn(2, 10)

        # Should not raise on validation
        X_validated = tsne._validate_data(X)
        assert X_validated.shape == (2, 10)

    def test_large_dimension(self):
        """Test data with large number of features."""
        from tsne_h2pack import TSNE

        tsne = TSNE(perplexity=10, max_iter=10)
        X = np.random.randn(50, 1000)  # 1000 features

        # Should not raise on validation
        X_validated = tsne._validate_data(X)
        assert X_validated.shape == (50, 1000)

    def test_single_feature(self):
        """Test data with single feature."""
        from tsne_h2pack import TSNE

        tsne = TSNE(perplexity=10, max_iter=10)
        X = np.random.randn(100, 1)  # Single feature

        # Should not raise on validation
        X_validated = tsne._validate_data(X)
        assert X_validated.shape == (100, 1)

    def test_very_large_values(self):
        """Test data with very large values."""
        from tsne_h2pack import TSNE

        tsne = TSNE(perplexity=10, max_iter=10)
        X = np.random.randn(100, 10) * 1e10  # Very large values

        # Should not raise on validation (finite values are OK)
        X_validated = tsne._validate_data(X)
        assert X_validated.shape == (100, 10)

    def test_very_small_values(self):
        """Test data with very small values."""
        from tsne_h2pack import TSNE

        tsne = TSNE(perplexity=10, max_iter=10)
        X = np.random.randn(100, 10) * 1e-10  # Very small values

        # Should not raise on validation (finite values are OK)
        X_validated = tsne._validate_data(X)
        assert X_validated.shape == (100, 10)

    def test_all_zeros(self):
        """Test data with all zeros."""
        from tsne_h2pack import TSNE

        tsne = TSNE(perplexity=10, max_iter=10)
        X = np.zeros((100, 10))

        # Should not raise on validation
        X_validated = tsne._validate_data(X)
        assert X_validated.shape == (100, 10)

    def test_constant_features(self):
        """Test data with constant features."""
        from tsne_h2pack import TSNE

        tsne = TSNE(perplexity=10, max_iter=10)
        X = np.ones((100, 10))  # All features are constant

        # Should not raise on validation
        X_validated = tsne._validate_data(X)
        assert X_validated.shape == (100, 10)
