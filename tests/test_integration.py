"""
Integration tests for tsne-h2pack.

These tests verify end-to-end functionality with realistic data and scenarios.
"""

import numpy as np
import pytest


@pytest.mark.requires_backend
class TestClusterSeparation:
    """Test that t-SNE separates clusters correctly."""

    def test_separates_simple_clusters(self, clustered_dataset, skip_without_backend):
        """Test that t-SNE separates well-defined clusters."""
        from tsne_h2pack import TSNE

        X, labels = clustered_dataset
        n_clusters = len(np.unique(labels))

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            max_iter=250,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y = tsne.fit_transform(X)

        # Calculate cluster centroids in embedding space
        centroids = np.array([Y[labels == i].mean(axis=0) for i in range(n_clusters)])

        # Check that centroids are reasonably separated
        from scipy.spatial.distance import pdist
        centroid_distances = pdist(centroids)

        # All centroid pairs should have some minimum distance
        min_distance = centroid_distances.min()
        assert min_distance > 0.1, "Cluster centroids should be separated"

    def test_within_cluster_variance_lower_than_between(
        self, clustered_dataset, skip_without_backend
    ):
        """Test that within-cluster variance is lower than between-cluster variance."""
        from tsne_h2pack import TSNE

        X, labels = clustered_dataset
        n_clusters = len(np.unique(labels))

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            max_iter=500,  # More iterations for better cluster separation
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y = tsne.fit_transform(X)

        # Calculate within-cluster variance
        within_variances = []
        for i in range(n_clusters):
            cluster_points = Y[labels == i]
            centroid = cluster_points.mean(axis=0)
            within_var = np.mean(np.sum((cluster_points - centroid) ** 2, axis=1))
            within_variances.append(within_var)

        avg_within_variance = np.mean(within_variances)

        # Calculate between-cluster variance
        overall_centroid = Y.mean(axis=0)
        cluster_centroids = np.array([Y[labels == i].mean(axis=0) for i in range(n_clusters)])
        between_variance = np.mean(np.sum((cluster_centroids - overall_centroid) ** 2, axis=1))

        # Between-cluster variance should be larger than within-cluster variance
        assert between_variance > avg_within_variance, (
            f"Between-cluster variance ({between_variance:.4f}) should be larger "
            f"than within-cluster variance ({avg_within_variance:.4f})"
        )


@pytest.mark.requires_backend
class TestReproducibility:
    """Test reproducibility of results."""

    def test_same_seed_same_result(self, small_dataset, skip_without_backend):
        """Test that same random_state produces identical results."""
        from tsne_h2pack import TSNE

        params = dict(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        tsne1 = TSNE(**params)
        tsne2 = TSNE(**params)

        Y1 = tsne1.fit_transform(small_dataset)
        Y2 = tsne2.fit_transform(small_dataset)

        np.testing.assert_array_almost_equal(Y1, Y2, decimal=5)

    def test_different_seed_different_result(self, small_dataset, skip_without_backend):
        """Test that different random_state produces different results."""
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

        # Results should differ
        assert not np.allclose(Y1, Y2)


@pytest.mark.requires_backend
class TestSklearnCompatibility:
    """Test sklearn API compatibility."""

    def test_fit_transform_returns_embedding(self, small_dataset, skip_without_backend):
        """Test that fit_transform returns embedding array."""
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

        assert Y is not None
        assert isinstance(Y, np.ndarray)
        assert Y.shape == (small_dataset.shape[0], 2)

    def test_embedding_attribute_set(self, small_dataset, skip_without_backend):
        """Test that embedding_ attribute is set after fit_transform."""
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

        assert tsne.embedding_ is not None
        np.testing.assert_array_equal(Y, tsne.embedding_)

    def test_n_iter_attribute_set(self, small_dataset, skip_without_backend):
        """Test that n_iter_ attribute is set after fit_transform."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        tsne.fit_transform(small_dataset)

        assert tsne.n_iter_ is not None
        assert isinstance(tsne.n_iter_, int)

    def test_kl_divergence_attribute_set(self, small_dataset, skip_without_backend):
        """Test that kl_divergence_ attribute is set after fit_transform."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        tsne.fit_transform(small_dataset)

        assert tsne.kl_divergence_ is not None
        assert isinstance(tsne.kl_divergence_, float)

    def test_y_parameter_ignored(self, small_dataset, skip_without_backend):
        """Test that y parameter is silently ignored (sklearn convention)."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        # Should work with y parameter
        y_labels = np.arange(small_dataset.shape[0])
        Y = tsne.fit_transform(small_dataset, y=y_labels)

        assert Y.shape == (small_dataset.shape[0], 2)


@pytest.mark.requires_backend
class TestParameterEffect:
    """Test that parameters affect the output as expected."""

    def test_perplexity_affects_result(self, medium_dataset, skip_without_backend):
        """Test that different perplexity values give different results."""
        from tsne_h2pack import TSNE

        tsne_low = TSNE(
            n_components=2,
            perplexity=5,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        tsne_high = TSNE(
            n_components=2,
            perplexity=50,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y_low = tsne_low.fit_transform(medium_dataset)
        Y_high = tsne_high.fit_transform(medium_dataset)

        # Results should differ
        assert not np.allclose(Y_low, Y_high)

    def test_n_components_affects_output_shape(self, small_dataset, skip_without_backend):
        """Test that n_components changes output dimensionality."""
        from tsne_h2pack import TSNE

        for n_components in [2, 3]:
            tsne = TSNE(
                n_components=n_components,
                perplexity=10,
                max_iter=50,
                random_state=42,
                verbose=0,
                knn_method='exact',
            )

            Y = tsne.fit_transform(small_dataset)

            assert Y.shape == (small_dataset.shape[0], n_components)


@pytest.mark.requires_backend
@pytest.mark.slow
class TestLargerDatasets:
    """Test with larger datasets (marked slow)."""

    def test_medium_dataset(self, medium_dataset, skip_without_backend):
        """Test with medium-sized dataset (500 samples)."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            max_iter=100,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y = tsne.fit_transform(medium_dataset)

        assert Y.shape == (medium_dataset.shape[0], 2)
        assert np.isfinite(Y).all()

    def test_large_dataset(self, large_dataset, skip_without_backend):
        """Test with larger dataset (2000 samples)."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            max_iter=100,
            method='h2pack',  # Use H2Pack for efficiency
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y = tsne.fit_transform(large_dataset)

        assert Y.shape == (large_dataset.shape[0], 2)
        assert np.isfinite(Y).all()


@pytest.mark.requires_backend
class TestMethodComparison:
    """Test and compare different methods."""

    def test_h2pack_vs_exact_similar_results(self, small_dataset, skip_without_backend):
        """Test that H2Pack and exact methods produce similar results."""
        from tsne_h2pack import TSNE

        tsne_h2pack = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=100,
            method='h2pack',
            h2_tol=1e-4,  # High precision
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        tsne_exact = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=100,
            method='exact',
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y_h2pack = tsne_h2pack.fit_transform(small_dataset)
        Y_exact = tsne_exact.fit_transform(small_dataset)

        # Results should be similar (not necessarily identical due to approximation)
        # Check that correlation is high
        correlation = np.corrcoef(Y_h2pack.flatten(), Y_exact.flatten())[0, 1]
        assert correlation > 0.5, f"Correlation between methods: {correlation}"


@pytest.mark.requires_backend
class TestKNNMethodsIntegration:
    """Test integration with different k-NN methods."""

    def test_exact_and_balltree_same_result(self, small_dataset, skip_without_backend):
        """Test that exact and balltree k-NN give same results."""
        from tsne_h2pack import TSNE

        tsne_exact = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            knn_method='exact',
            random_state=42,
            verbose=0,
        )

        tsne_balltree = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            knn_method='balltree',
            random_state=42,
            verbose=0,
        )

        Y_exact = tsne_exact.fit_transform(small_dataset)
        Y_balltree = tsne_balltree.fit_transform(small_dataset)

        # Should be identical (both are exact methods)
        np.testing.assert_array_almost_equal(Y_exact, Y_balltree, decimal=5)


@pytest.mark.requires_backend
class TestOutputQuality:
    """Test output quality metrics."""

    def test_output_centered(self, small_dataset, skip_without_backend):
        """Test that output is approximately centered."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=100,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y = tsne.fit_transform(small_dataset)

        # Mean should be close to zero
        mean = Y.mean(axis=0)
        assert np.allclose(mean, 0, atol=1.0), f"Mean: {mean}"

    def test_output_finite(self, small_dataset, skip_without_backend):
        """Test that all output values are finite."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=100,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        Y = tsne.fit_transform(small_dataset)

        assert np.isfinite(Y).all(), "All output values should be finite"
        assert not np.isnan(Y).any(), "No NaN values in output"
        assert not np.isinf(Y).any(), "No inf values in output"


@pytest.mark.requires_backend
class TestProfiler:
    """Test profiling functionality."""

    def test_profiling_enabled(self, tiny_dataset, skip_without_backend):
        """Test that profiling can be enabled."""
        from tsne_h2pack import TSNE
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            profiling_file = f.name

        try:
            tsne = TSNE(
                n_components=2,
                perplexity=3,
                max_iter=10,
                enable_profiling=True,
                profiling_output_file=profiling_file,
                random_state=42,
                verbose=0,
                knn_method='exact',
            )

            Y = tsne.fit_transform(tiny_dataset)

            # Should complete without error
            assert Y.shape == (tiny_dataset.shape[0], 2)
        finally:
            # Clean up
            if os.path.exists(profiling_file):
                os.remove(profiling_file)

    def test_knn_time_attribute(self, small_dataset, skip_without_backend):
        """Test that knn_time_ attribute is set."""
        from tsne_h2pack import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=50,
            random_state=42,
            verbose=0,
            knn_method='exact',
        )

        tsne.fit_transform(small_dataset)

        # knn_time_ should be set
        assert hasattr(tsne, 'knn_time_')
        assert tsne.knn_time_ >= 0


@pytest.mark.requires_backend
class TestKLDivergenceConvergence:
    """Verify that KL divergence is actually computed and decreases.

    Pre-fix this test would have caught a real bug: the C extension
    computed KL but the Python wrapper hardcoded `kl_divergence_=0.0`
    because `compute_kl_divergence` was declared in the header but
    never defined, so dlsym couldn't find it.
    """

    def test_kl_divergence_is_positive(self, skip_without_backend):
        """KL divergence on a real run should be > 0 (not the 0.0 default)."""
        from tsne_h2pack import TSNE

        rng = np.random.default_rng(0)
        n_dim = 20
        centers = rng.standard_normal((5, n_dim)) * 8
        X = np.vstack([
            rng.standard_normal((100, n_dim)) + centers[i]
            for i in range(5)
        ])

        tsne = TSNE(
            n_components=2, perplexity=15, max_iter=300,
            knn_method='exact', random_state=0, verbose=0,
        )
        tsne.fit_transform(X)
        assert tsne.kl_divergence_ > 0.0, (
            f"kl_divergence_ should be positive after a real run, "
            f"got {tsne.kl_divergence_!r}. If this is exactly 0.0, the "
            f"C-side `compute_kl_divergence` symbol is likely missing "
            f"from the built library."
        )

    def test_kl_divergence_decreases_with_more_iters(self, skip_without_backend):
        """KL after 50 iters should be > KL after 500 iters on the same data."""
        from tsne_h2pack import TSNE

        rng = np.random.default_rng(1)
        n_dim = 15
        centers = rng.standard_normal((4, n_dim)) * 10
        X = np.vstack([
            rng.standard_normal((80, n_dim)) + centers[i]
            for i in range(4)
        ])

        kls = []
        for max_iter in (50, 500):
            tsne = TSNE(
                n_components=2, perplexity=12, max_iter=max_iter,
                knn_method='exact', random_state=0, verbose=0,
            )
            tsne.fit_transform(X)
            kls.append(tsne.kl_divergence_)

        kl_short, kl_long = kls
        assert kl_short > 0 and kl_long > 0
        # Long run should converge to lower KL than short run.
        assert kl_long < kl_short, (
            f"KL did not decrease with more iterations: "
            f"kl(50)={kl_short:.4f}, kl(500)={kl_long:.4f}"
        )


@pytest.mark.slow
@pytest.mark.requires_backend
class TestLargerScaleIntegration:
    """Smoke-test t-SNE at a size representative of real-world use.

    Marked `slow` so it can be excluded with `-m "not slow"` for quick
    iteration; CI runs the full suite on push to main.
    """

    def test_10k_samples_clustered(self, skip_without_backend):
        """t-SNE should run cleanly and separate clusters at 10K samples."""
        import os
        from tsne_h2pack import TSNE

        rng = np.random.default_rng(42)
        n_clusters = 5
        n_per_cluster = 2000  # 10,000 total
        n_dim = 50
        centers = rng.standard_normal((n_clusters, n_dim)) * 8

        X = np.vstack([
            rng.standard_normal((n_per_cluster, n_dim)) + centers[i]
            for i in range(n_clusters)
        ])
        labels = np.repeat(np.arange(n_clusters), n_per_cluster)

        # Cap n_jobs at the runner's CPU count. GitHub macos-14 runners
        # only have 3 cores; pynndescent rejects n_jobs > cpu_count.
        n_jobs = min(4, os.cpu_count() or 1)

        tsne = TSNE(
            n_components=2, perplexity=30, max_iter=300,
            method='h2pack', knn_method='pynndescent',
            n_jobs=n_jobs, random_state=42, verbose=0,
        )
        Y = tsne.fit_transform(X)

        assert Y.shape == (n_clusters * n_per_cluster, 2)
        assert np.isfinite(Y).all()
        assert tsne.kl_divergence_ > 0

        # Cluster separation: each cluster's centroid should be further
        # from other centroids than its own scatter.
        centroids = np.array([Y[labels == i].mean(axis=0) for i in range(n_clusters)])
        intra_scatters = np.array([
            np.linalg.norm(Y[labels == i] - centroids[i], axis=1).mean()
            for i in range(n_clusters)
        ])
        # Pairwise inter-centroid distances (excluding self)
        from itertools import combinations
        inter_distances = [
            np.linalg.norm(centroids[i] - centroids[j])
            for i, j in combinations(range(n_clusters), 2)
        ]
        # The smallest inter-cluster distance should still exceed the
        # largest intra-cluster scatter — i.e., clusters don't overlap.
        assert min(inter_distances) > max(intra_scatters), (
            f"Clusters overlap at scale: "
            f"min inter-centroid = {min(inter_distances):.2f}, "
            f"max intra scatter = {max(intra_scatters):.2f}"
        )
