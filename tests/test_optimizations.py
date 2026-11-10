"""
Equivalence tests for the performance paths in the optimization loop.

Each fast path has an environment switch that restores the reference behavior.
t-SNE amplifies rounding differences exponentially (on this problem: 1e-15 after
10 iterations, 1e-10 after 20, 1e-5 after 40), so the comparisons run only a few
iterations past the point where the two paths first differ. A genuine bug (wrong
permutation, wrong block, wrong column) shows up as an O(1) difference at once.
"""

import numpy as np
import pytest

from tsne_h2pack import TSNE


@pytest.fixture(scope="module")
def clustered_data():
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=6.0, size=(12, 20))
    return (centers[rng.integers(0, 12, size=4000)] + rng.normal(size=(4000, 20))).astype(np.float64)


def _embed(X, monkeypatch, max_iter, **env):
    env.setdefault("TSNE_ATTR_F32", "0")   # exact comparisons: keep the divide in double
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=max_iter, knn_method='exact',
                n_jobs=1, random_state=0)
    return tsne.fit_transform(X)


def _rel_diff(a, b):
    return np.abs(a - b).max() / np.abs(a).max()


def test_fused_matvec_matches_h2p_matvec(clustered_data, monkeypatch):
    """One fused traversal for W*[1|Y0|Y1] == three H2P_matvec calls."""
    ref = _embed(clustered_data, monkeypatch, 10, TSNE_NO_FUSED="1")
    fused = _embed(clustered_data, monkeypatch, 10, TSNE_NO_FUSED="0")
    assert _rel_diff(ref, fused) < 1e-10


def test_attractive_fast_path_matches_generic(clustered_data, monkeypatch):
    ref = _embed(clustered_data, monkeypatch, 10, TSNE_ATTR_GENERIC="1")
    fast = _embed(clustered_data, monkeypatch, 10, TSNE_ATTR_GENERIC="0")
    assert _rel_diff(ref, fast) < 1e-10


def test_single_precision_divide_is_a_small_perturbation(clustered_data, monkeypatch):
    """The attractive pass divides in float32: ~1e-7 per pair force, not a different force."""
    ref = _embed(clustered_data, monkeypatch, 10, TSNE_ATTR_F32="0")
    mixed = _embed(clustered_data, monkeypatch, 10, TSNE_ATTR_F32="1")
    assert _rel_diff(ref, mixed) < 1e-4


def test_float32_gather_is_a_small_perturbation(clustered_data, monkeypatch):
    """Default path: neighbors gathered from a float32 copy of Y, i.e. the exact force
    of an embedding displaced by at most half a float ulp per coordinate."""
    ref = _embed(clustered_data, monkeypatch, 10, TSNE_ATTR_F32="0")
    single = _embed(clustered_data, monkeypatch, 10, TSNE_ATTR_F32="2")
    assert _rel_diff(ref, single) < 1e-3


@pytest.mark.parametrize("level", ["1", "2"])
def test_float_p_copy_follows_renumbering(clustered_data, monkeypatch, level):
    """The float32 attractive paths read a float copy of P's values, which must be
    refreshed when the renumbering at iteration 30 permutes P. Both runs use the same
    float32 arithmetic; the renumbering only changes the summation order within a row
    (~1e-7, amplified over 4 iterations), whereas a stale copy pairs every p_ij with
    the wrong neighbor and changes the embedding at once."""
    ref = _embed(clustered_data, monkeypatch, 34, TSNE_ATTR_F32=level, TSNE_REORDER="0")
    renumbered = _embed(clustered_data, monkeypatch, 34, TSNE_ATTR_F32=level, TSNE_REORDER="1")
    assert _rel_diff(ref, renumbered) < 1e-5


def test_point_renumbering_is_transparent(clustered_data, monkeypatch):
    """The first renumbering happens at iteration 30: the embedding must still come
    back in the caller's point order, and match a run without renumbering."""
    ref = _embed(clustered_data, monkeypatch, 34, TSNE_REORDER="0")
    renumbered = _embed(clustered_data, monkeypatch, 34, TSNE_REORDER="1")
    assert _rel_diff(ref, renumbered) < 1e-10


def test_point_renumbering_is_transparent_in_3d(clustered_data, monkeypatch):
    """Same as above with n_components=3: the renumbering permutes every per-point array
    for any embedding dimension and the output comes back in the caller's order."""
    def embed(reorder):
        monkeypatch.setenv("TSNE_ATTR_F32", "0")
        monkeypatch.setenv("TSNE_REORDER", reorder)
        return TSNE(n_components=3, perplexity=30, max_iter=34, knn_method='exact',
                    n_jobs=1, random_state=0).fit_transform(clustered_data)
    ref, renumbered = embed("0"), embed("1")
    assert ref.shape == (clustered_data.shape[0], 3)
    assert _rel_diff(ref, renumbered) < 1e-10


def test_renumbering_keeps_cluster_structure(clustered_data, monkeypatch):
    """Through all six renumberings the rows of the output still line up with the
    rows of the input (a mis-tracked permutation would scatter every cluster)."""
    from sklearn.neighbors import NearestNeighbors

    Y = _embed(clustered_data, monkeypatch, 1000, TSNE_REORDER="1")
    k = 10
    # kneighbors() without a query excludes each point itself
    hd = NearestNeighbors(n_neighbors=k).fit(clustered_data).kneighbors(return_distance=False)
    ld = NearestNeighbors(n_neighbors=k).fit(Y).kneighbors(return_distance=False)
    overlap = np.mean([len(set(h) & set(l)) for h, l in zip(hd, ld)]) / k
    # ~0.3 when the rows line up (it varies by a few percent with the numpy / sklearn
    # version); a scattered permutation gives about k / n = 0.0025
    assert overlap > 0.15


def test_proxy_points_are_cached_per_user(clustered_data, monkeypatch, tmp_path):
    """Proxy points are data-independent: generated once, then loaded, with identical results."""
    monkeypatch.setenv("TSNE_H2PACK_CACHE_DIR", str(tmp_path))
    cold = _embed(clustered_data, monkeypatch, 5)
    cached = list(tmp_path.iterdir())
    assert len(cached) == 1 and cached[0].stat().st_size > 0
    warm = _embed(clustered_data, monkeypatch, 5)
    assert np.array_equal(cold, warm)

    monkeypatch.setenv("TSNE_H2PACK_NO_PROXY_CACHE", "1")
    assert np.array_equal(cold, _embed(clustered_data, monkeypatch, 5))
