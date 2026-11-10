"""
Tests for the TSNE class (tsne_h2pack/manifold.py).
"""

import numpy as np
import pytest


class TestTSNEInit:
    """Test TSNE class initialization and parameter storage."""

    def test_default_parameters(self):
        """Test TSNE with default parameters."""
        from tsne_h2pack import TSNE

        tsne = TSNE()

        assert tsne.n_components == 2
        assert tsne.perplexity == 30.0
        assert tsne.early_exaggeration == 12.0
        assert tsne.learning_rate == 'auto'
        assert tsne.max_iter == 1000
        assert tsne.n_iter_without_progress == 300
        assert tsne.min_grad_norm == 1e-7
        assert tsne.metric == 'euclidean'
        assert tsne.init == 'pca'
        assert tsne.verbose == 0
        assert tsne.random_state is None
        assert tsne.method == 'h2pack'
        assert tsne.n_jobs == 1
        assert tsne.h2_tol == 1e-2
        assert tsne.h2_rebuild_freq == 3
        assert tsne.h2_rebuild_freq_early == 10
        assert tsne.h2_leaf_size == 100
        assert tsne.enable_profiling is False
        assert tsne.profiling_output_file is None
        assert tsne.knn_method == 'pynndescent'
        assert tsne.knn_params == {}

    def test_custom_parameters(self):
        """Test TSNE with custom parameters."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=3,
            perplexity=50.0,
            early_exaggeration=15.0,
            learning_rate=500.0,
            max_iter=500,
            n_iter_without_progress=100,
            min_grad_norm=1e-6,
            verbose=1,
            random_state=42,
            method='exact',
            n_jobs=4,
            h2_tol=1e-3,
            h2_rebuild_freq=5,
            enable_profiling=True,
            knn_method='exact',
            knn_params={'leaf_size': 50},
        )

        assert tsne.n_components == 3
        assert tsne.perplexity == 50.0
        assert tsne.early_exaggeration == 15.0
        assert tsne.learning_rate == 500.0
        assert tsne.max_iter == 500
        assert tsne.n_iter_without_progress == 100
        assert tsne.min_grad_norm == 1e-6
        assert tsne.verbose == 1
        assert tsne.random_state == 42
        assert tsne.method == 'exact'
        assert tsne.n_jobs == 4
        assert tsne.h2_tol == 1e-3
        assert tsne.h2_rebuild_freq == 5
        assert tsne.enable_profiling is True
        assert tsne.knn_method == 'exact'
        assert tsne.knn_params == {'leaf_size': 50}

    def test_n_jobs_none_defaults_to_one(self):
        """Test that n_jobs=None defaults to 1."""
        from tsne_h2pack import TSNE

        tsne = TSNE(n_jobs=None)
        assert tsne.n_jobs == 1

    def test_attributes_initialized_to_none(self):
        """Test that result attributes are None before fitting."""
        from tsne_h2pack import TSNE

        tsne = TSNE()
        assert tsne.embedding_ is None
        assert tsne.kl_divergence_ is None
        assert tsne.n_iter_ is None


class TestTSNEParameterValidation:
    """Test TSNE parameter validation."""

    def test_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="metric=.*not supported"):
            TSNE(metric='manhattan')

    def test_invalid_init(self):
        """Test that invalid init raises ValueError."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="init=.*not supported"):
            TSNE(init='spectral')

    def test_pca_init_accepted(self):
        """init='pca' is a valid option."""
        from tsne_h2pack import TSNE

        assert TSNE(init='pca').init == 'pca'

    def test_pca_init_accepted_with_precomputed(self):
        """init='pca' (the default) is valid with metric='precomputed'; it falls back
        to random initialization at fit time, with a warning."""
        from tsne_h2pack import TSNE

        TSNE(init='pca', metric='precomputed')
        TSNE(metric='precomputed')

    def test_invalid_init_array(self):
        """An init array must be 2D with n_components columns and finite."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="init"):
            TSNE(init=np.zeros((10, 3)))
        with pytest.raises(ValueError, match="init"):
            TSNE(init=np.zeros(10))
        bad = np.zeros((10, 2))
        bad[3, 1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            TSNE(init=bad)

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="method=.*not supported"):
            TSNE(method='barnes_hut')

    def test_invalid_n_components(self):
        """Test that invalid n_components raises ValueError."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="n_components must be >= 1"):
            TSNE(n_components=0)

        with pytest.raises(ValueError, match="n_components must be >= 1"):
            TSNE(n_components=-1)

    def test_invalid_perplexity(self):
        """Test that invalid perplexity raises ValueError."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="perplexity must be > 0"):
            TSNE(perplexity=0)

        with pytest.raises(ValueError, match="perplexity must be > 0"):
            TSNE(perplexity=-5)

    def test_invalid_learning_rate(self):
        """Test that invalid learning_rate raises ValueError."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            TSNE(learning_rate=0)

        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            TSNE(learning_rate=-100)

    def test_invalid_max_iter(self):
        """Test that invalid max_iter raises ValueError."""
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            TSNE(max_iter=0)

        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            TSNE(max_iter=-10)


class TestTSNEDataValidation:
    """Test TSNE input data validation."""

    def test_1d_array_raises_error(self):
        """Test that 1D array raises ValueError."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(100)

        with pytest.raises(ValueError, match="must be 2-dimensional"):
            tsne.fit_transform(X)

    def test_3d_array_raises_error(self):
        """Test that 3D array raises ValueError."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(10, 10, 10)

        with pytest.raises(ValueError, match="must be 2-dimensional"):
            tsne.fit_transform(X)

    def test_single_sample_raises_error(self):
        """Test that single sample raises ValueError."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(1, 50)

        with pytest.raises(ValueError, match="n_samples=1 should be >= 2"):
            tsne.fit_transform(X)

    def test_perplexity_too_large_raises_error(self):
        """Test that perplexity >= n_samples raises ValueError."""
        from tsne_h2pack import TSNE

        tsne = TSNE(perplexity=100, max_iter=10)
        X = np.random.randn(50, 10)

        with pytest.raises(ValueError, match="perplexity=.* must be less than"):
            tsne.fit_transform(X)

    def test_nan_values_raise_error(self):
        """Test that NaN values raise ValueError."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(100, 50)
        X[0, 0] = np.nan

        with pytest.raises(ValueError, match="contains NaN or inf"):
            tsne.fit_transform(X)

    def test_inf_values_raise_error(self):
        """Test that inf values raise ValueError."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(100, 50)
        X[0, 0] = np.inf

        with pytest.raises(ValueError, match="contains NaN or inf"):
            tsne.fit_transform(X)

    def test_negative_inf_values_raise_error(self):
        """Test that -inf values raise ValueError."""
        from tsne_h2pack import TSNE

        tsne = TSNE(max_iter=10)
        X = np.random.randn(100, 50)
        X[0, 0] = -np.inf

        with pytest.raises(ValueError, match="contains NaN or inf"):
            tsne.fit_transform(X)


class TestTSNEFit:
    """Test TSNE fit method."""

    def test_fit_not_implemented(self):
        """Test that fit() raises NotImplementedError."""
        from tsne_h2pack import TSNE

        tsne = TSNE()
        X = np.random.randn(100, 50)

        with pytest.raises(NotImplementedError, match="fit\\(\\) is not implemented"):
            tsne.fit(X)


@pytest.mark.requires_backend
class TestTSNEFitTransform:
    """Test TSNE fit_transform method (requires backend)."""

    def test_fit_transform_basic(self, small_dataset, skip_without_backend):
        """Test basic fit_transform functionality."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y = tsne.fit_transform(small_dataset)

        # Check output shape
        assert Y.shape == (small_dataset.shape[0], 2)

        # Check attributes are set
        assert tsne.embedding_ is not None
        assert tsne.embedding_.shape == (small_dataset.shape[0], 2)
        assert tsne.n_iter_ is not None
        assert tsne.kl_divergence_ is not None

        # Check output is finite
        assert np.isfinite(Y).all()

    def test_fit_transform_3_components(self, small_dataset, skip_without_backend):
        """Test fit_transform with 3 components."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=3,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y = tsne.fit_transform(small_dataset)

        assert Y.shape == (small_dataset.shape[0], 3)
        assert tsne.embedding_.shape == (small_dataset.shape[0], 3)
        assert np.isfinite(Y).all()

    def test_fit_transform_with_h2pack(self, medium_dataset, skip_without_backend):
        """Test fit_transform with H2Pack method."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            max_iter=100,
            method='h2pack',
            h2_tol=1e-3,
            h2_rebuild_freq=5,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y = tsne.fit_transform(medium_dataset)

        assert Y.shape == (medium_dataset.shape[0], 2)
        assert np.isfinite(Y).all()

    def test_fit_transform_with_exact_method(self, small_dataset, skip_without_backend):
        """Test fit_transform with exact method."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            method='exact',
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y = tsne.fit_transform(small_dataset)

        assert Y.shape == (small_dataset.shape[0], 2)
        assert np.isfinite(Y).all()

    def test_fit_transform_deterministic_with_seed(self, small_dataset, skip_without_backend):
        """Test that fit_transform is deterministic with random_state."""
        from tsne_h2pack import TSNE

        tsne1 = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        tsne2 = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y1 = tsne1.fit_transform(small_dataset)
        Y2 = tsne2.fit_transform(small_dataset)

        # Results should be identical with same random_state
        np.testing.assert_array_almost_equal(Y1, Y2, decimal=5)

    def test_fit_transform_different_seeds_different_results(
        self, small_dataset, skip_without_backend
    ):
        """Test that different random_state gives different results."""
        from tsne_h2pack import TSNE

        tsne1 = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        tsne2 = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=123,
            verbose=0,
            knn_method='exact',
        )

        Y1 = tsne1.fit_transform(small_dataset)
        Y2 = tsne2.fit_transform(small_dataset)

        # Results should differ with different random_state
        # (at least one element should be different)
        assert not np.allclose(Y1, Y2)

    def test_fit_transform_y_parameter_ignored(self, small_dataset, skip_without_backend):
        """Test that y parameter is ignored in fit_transform."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        # y parameter should be ignored
        y = np.arange(small_dataset.shape[0])
        Y = tsne.fit_transform(small_dataset, y=y)

        assert Y.shape == (small_dataset.shape[0], 2)

    def test_fit_transform_converts_dtype(self, skip_without_backend):
        """Test that fit_transform converts input to correct dtype."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        # Test with float32 input
        np.random.seed(42)
        X_float32 = np.random.randn(100, 50).astype(np.float32)
        Y = tsne.fit_transform(X_float32)

        assert Y.shape == (100, 2)
        assert np.isfinite(Y).all()

    def test_fit_transform_list_input(self, skip_without_backend):
        """Test that fit_transform works with list input."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        # Test with list input
        np.random.seed(42)
        X_list = np.random.randn(100, 50).tolist()
        Y = tsne.fit_transform(X_list)

        assert Y.shape == (100, 2)
        assert np.isfinite(Y).all()


@pytest.mark.requires_backend
class TestInitialization:
    """init='pca' and ndarray initializations."""

    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.default_rng(0)
        centers = rng.normal(scale=5.0, size=(5, 10))
        return centers[rng.integers(0, 5, size=600)] + rng.normal(size=(600, 10))

    def test_pca_initial_embedding_follows_sklearn(self, data):
        """PCA projection to n_components, first column rescaled to std 1e-4."""
        from sklearn.decomposition import PCA
        from tsne_h2pack import TSNE

        Y0 = TSNE(init='pca', random_state=3)._initial_embedding(data, data.shape[0])
        ref = PCA(2, svd_solver='randomized', random_state=3).fit_transform(data)
        ref = ref / np.std(ref[:, 0]) * 1e-4
        np.testing.assert_allclose(Y0, ref, rtol=1e-10, atol=1e-16)
        assert abs(np.std(Y0[:, 0]) - 1e-4) < 1e-12

    def test_pca_fit_transform(self, data):
        """init='pca' runs, has the right shape and is reproducible for a fixed seed."""
        from tsne_h2pack import TSNE

        kw = dict(init='pca', perplexity=20, max_iter=250, n_jobs=1, random_state=0)
        Y1 = TSNE(**kw).fit_transform(data)
        Y2 = TSNE(**kw).fit_transform(data)
        assert Y1.shape == (data.shape[0], 2)
        assert np.isfinite(Y1).all()
        np.testing.assert_array_equal(Y1, Y2)
        Y3 = TSNE(**{**kw, 'n_components': 3}).fit_transform(data)
        assert Y3.shape == (data.shape[0], 3)

    def test_pca_differs_from_random(self, data):
        from tsne_h2pack import TSNE

        kw = dict(perplexity=20, max_iter=50, n_jobs=1, random_state=0)
        Yp = TSNE(init='pca', **kw).fit_transform(data)
        Yr = TSNE(init='random', **kw).fit_transform(data)
        assert not np.allclose(Yp, Yr)

    def test_array_init_is_used(self, data):
        """An ndarray init is the starting point: with a negligible step the
        embedding is the (re-centered) init array."""
        from tsne_h2pack import TSNE

        Y0 = np.random.default_rng(1).normal(size=(data.shape[0], 2))
        Y = TSNE(init=Y0, max_iter=1, learning_rate=1e-10, perplexity=20,
                 n_jobs=1, random_state=0).fit_transform(data)
        np.testing.assert_allclose(Y - Y.mean(0), Y0 - Y0.mean(0), atol=1e-8)
        assert np.array_equal(Y0, np.random.default_rng(1).normal(size=Y0.shape))  # not modified

    def test_array_init_with_precomputed(self, data):
        """An ndarray init works without features (metric='precomputed')."""
        from sklearn.neighbors import kneighbors_graph
        from tsne_h2pack import TSNE

        graph = kneighbors_graph(data, n_neighbors=60, mode='distance')
        Y0 = np.random.default_rng(1).normal(size=(data.shape[0], 2))
        Y = TSNE(metric='precomputed', init=Y0, max_iter=1, learning_rate=1e-10,
                 perplexity=20, n_jobs=1).fit_transform(graph)
        np.testing.assert_allclose(Y - Y.mean(0), Y0 - Y0.mean(0), atol=1e-8)
        Y = TSNE(metric='precomputed', init=Y0 * 1e-4, perplexity=20, max_iter=250,
                 n_jobs=1).fit_transform(graph)
        assert Y.shape == (data.shape[0], 2) and np.isfinite(Y).all()

    def test_pca_with_precomputed_falls_back_to_random(self, data):
        """With metric='precomputed' there are no features for PCA: init='pca' warns and
        gives exactly the init='random' result; an explicit init='random' does not warn."""
        import warnings
        from sklearn.neighbors import kneighbors_graph
        from tsne_h2pack import TSNE

        graph = kneighbors_graph(data, n_neighbors=60, mode='distance')
        kw = dict(metric='precomputed', perplexity=20, max_iter=50, n_jobs=1, random_state=0)
        with pytest.warns(UserWarning, match="init='random'"):
            got = TSNE(**kw).fit_transform(graph)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            ref = TSNE(init='random', **kw).fit_transform(graph)
        np.testing.assert_array_equal(got, ref)

    def test_array_init_wrong_n_samples(self, data):
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="shape"):
            TSNE(init=np.zeros((10, 2)), perplexity=20).fit_transform(data)


class TestTSNEKNNMethods:
    """Test TSNE with different k-NN methods."""

    def test_exact_knn(self, small_dataset, skip_without_backend):
        """Test TSNE with exact k-NN method."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            knn_method='exact',
            random_state=42,
            verbose=0,
        )

        Y = tsne.fit_transform(small_dataset)
        assert Y.shape == (small_dataset.shape[0], 2)
        assert np.isfinite(Y).all()

    def test_balltree_knn(self, small_dataset, skip_without_backend):
        """Test TSNE with ball tree k-NN method."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            knn_method='balltree',
            random_state=42,
            verbose=0,
        )

        Y = tsne.fit_transform(small_dataset)
        assert Y.shape == (small_dataset.shape[0], 2)
        assert np.isfinite(Y).all()

    def test_pynndescent_knn(self, small_dataset, skip_without_backend, skip_without_pynndescent):
        """Test TSNE with PyNNDescent k-NN method."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            knn_method='pynndescent',
            random_state=42,
            verbose=0,
        )

        Y = tsne.fit_transform(small_dataset)
        assert Y.shape == (small_dataset.shape[0], 2)
        assert np.isfinite(Y).all()

    def test_faiss_knn(self, small_dataset, skip_without_backend, skip_without_faiss):
        """Test TSNE with FAISS k-NN method."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            knn_method='faiss',
            knn_params={'index_type': 'Flat'},  # Use flat index for small dataset
            random_state=42,
            verbose=0,
        )

        Y = tsne.fit_transform(small_dataset)
        assert Y.shape == (small_dataset.shape[0], 2)
        assert np.isfinite(Y).all()

    def test_invalid_knn_method(self):
        """Test that invalid k-NN method raises error during fit_transform."""
        from tsne_h2pack import TSNE
        from tsne_h2pack._knn_methods import get_knn_method

        # The error is raised when get_knn_method is called during fit_transform
        with pytest.raises(ValueError, match="Unknown knn_method"):
            get_knn_method('invalid_method')


@pytest.mark.requires_backend
class TestTSNENJobs:
    """Test TSNE with different n_jobs values."""

    def test_single_thread(self, small_dataset, skip_without_backend):
        """Test TSNE with single thread."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            n_jobs=1,
            knn_method='exact',
            random_state=42,
            verbose=0,
        )

        Y = tsne.fit_transform(small_dataset)
        assert Y.shape == (small_dataset.shape[0], 2)

    def test_multiple_threads(self, small_dataset, skip_without_backend):
        """Test TSNE with multiple threads."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            n_jobs=2,
            knn_method='exact',
            random_state=42,
            verbose=0,
        )

        Y = tsne.fit_transform(small_dataset)
        assert Y.shape == (small_dataset.shape[0], 2)


class TestLearningRateAuto:
    """learning_rate='auto' follows the FIt-SNE / openTSNE rule max(N / 12, 200)."""

    def test_auto_scales_with_n_samples(self):
        from tsne_h2pack import TSNE

        tsne = TSNE()
        assert tsne._map_params_to_c_struct(n_samples=120000).learning_rate == pytest.approx(10000.0)
        assert tsne._map_params_to_c_struct(n_samples=600).learning_rate == pytest.approx(200.0)

    def test_explicit_learning_rate_is_kept(self):
        from tsne_h2pack import TSNE

        assert TSNE(learning_rate=350.0)._map_params_to_c_struct(n_samples=120000).learning_rate == 350.0

    def test_invalid_string_raises(self):
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="learning_rate"):
            TSNE(learning_rate='fast')


class TestH2TuningParams:
    def test_mapped_to_c_struct(self):
        from tsne_h2pack import TSNE

        params = TSNE(h2_rebuild_freq=4, h2_rebuild_freq_early=2, h2_leaf_size=64)._map_params_to_c_struct(1000)
        assert params.h2_leaf_size == 64
        assert params.h2_rebuild_freq == 4
        assert params.h2_rebuild_freq_early == 4   # never lower than h2_rebuild_freq

    def test_c_defaults(self):
        from tsne_h2pack._backend import get_backend

        params = get_backend().create_default_params()
        assert params.h2_leaf_size == 100
        assert params.h2_rebuild_freq_early == 10


class TestH2DefaultsByDimension:
    """H2 settings left at None resolve per embedding dimension (H2_DEFAULTS)."""

    def test_2d_and_3d_defaults(self):
        from tsne_h2pack import TSNE, H2_DEFAULTS, h2_defaults
        assert h2_defaults(2) == H2_DEFAULTS[2] and h2_defaults(3) == H2_DEFAULTS[3] == h2_defaults(5)
        t2, t3 = TSNE(), TSNE(n_components=3)
        assert (t2.h2_tol, t2.h2_leaf_size) == (1e-2, 100)
        assert t3.h2_tol == H2_DEFAULTS[3]["h2_tol"] == 3e-2
        assert t3.h2_rebuild_freq == 3 and t3.h2_rebuild_freq_early == 10 and t3.h2_leaf_size == 100

    def test_explicit_values_win(self):
        from tsne_h2pack import TSNE
        t = TSNE(n_components=3, h2_tol=1e-3, h2_leaf_size=64, h2_rebuild_freq=2)
        assert (t.h2_tol, t.h2_leaf_size, t.h2_rebuild_freq) == (1e-3, 64, 2)
        p = t._map_params_to_c_struct(n_samples=1000)
        assert (p.h2_tol, p.h2_leaf_size, p.h2_rebuild_freq) == (1e-3, 64, 2)

    def test_3d_default_reaches_the_c_struct(self):
        from tsne_h2pack import TSNE
        p = TSNE(n_components=3)._map_params_to_c_struct(n_samples=1000)
        assert abs(p.h2_tol - 3e-2) < 1e-12


class TestPrecomputedGraph:
    """metric='precomputed': embed from a sparse matrix of k-NN distances."""

    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.default_rng(0)
        centers = rng.normal(scale=5.0, size=(5, 10))
        return centers[rng.integers(0, 5, size=600)] + rng.normal(size=(600, 10))

    def test_matches_the_feature_array_path(self, data):
        from sklearn.neighbors import NearestNeighbors
        from tsne_h2pack import TSNE

        # Few iterations: t-SNE amplifies rounding differences exponentially. The graph is
        # built exactly as knn_method='exact' builds it (brute force, float32 distances):
        # a tree search rounds the same distances differently, and on some numpy/sklearn
        # versions that difference grows past 1e-8 within ten iterations.
        kw = dict(perplexity=20, max_iter=10, n_jobs=1, random_state=0, init='random')
        ref = TSNE(knn_method='exact', **kw).fit_transform(data)
        graph = NearestNeighbors(n_neighbors=60, algorithm='brute').fit(data).kneighbors_graph(mode='distance')
        graph.data = graph.data.astype(np.float32)
        got = TSNE(metric='precomputed', **kw).fit_transform(graph)
        assert np.abs(ref - got).max() / np.abs(ref).max() < 1e-8

    def test_self_entries_in_a_graph_are_ignored(self, data):
        """A k-NN graph that lists each point as its own (distance-0) neighbor gives the
        same embedding as the clean graph: the backend strips the diagonal."""
        from scipy.sparse import csr_matrix
        from sklearn.neighbors import NearestNeighbors
        from tsne_h2pack import TSNE
        from tsne_h2pack._backend import get_backend

        n = data.shape[0]
        graph = NearestNeighbors(n_neighbors=60, algorithm='brute').fit(data).kneighbors_graph(mode='distance')
        graph.data = graph.data.astype(np.float32)
        k = 60                                           # prepend an explicit 0 at (i, i) to every row
        ind = np.concatenate([np.arange(n)[:, None], graph.indices.reshape(n, k)], axis=1).ravel()
        dat = np.concatenate([np.zeros((n, 1), np.float32), graph.data.reshape(n, k)], axis=1).ravel()
        with_self = csr_matrix((dat, ind, np.arange(0, (n + 1) * (k + 1), k + 1)), shape=(n, n))
        assert with_self.nnz == graph.nnz + n            # explicit zeros on the diagonal
        kw = dict(perplexity=20, max_iter=10, n_jobs=1, random_state=0, init="random")
        params_a = TSNE(metric='precomputed', **kw)
        ref = params_a.fit_transform(graph)
        # bypass manifold's own setdiag(0)/eliminate_zeros to test the backend guard itself
        p = params_a._map_params_to_c_struct(n_samples=n)
        got, _ = get_backend().fit_transform_with_knn_distances(None, with_self, 2, p)
        assert np.abs(ref - got).max() / np.abs(ref).max() < 1e-8

    def test_rejects_dense_input_and_short_rows(self, data):
        from sklearn.neighbors import kneighbors_graph
        from tsne_h2pack import TSNE

        with pytest.raises(ValueError, match="sparse"):
            TSNE(metric='precomputed').fit_transform(data)
        graph = kneighbors_graph(data, n_neighbors=10, mode='distance')
        with pytest.raises(ValueError, match="neighbors"):
            TSNE(metric='precomputed', perplexity=30).fit_transform(graph)
