"""
sklearn-compatible TSNE implementation using tsne-h2pack backend.

This module provides a drop-in replacement for sklearn.manifold.TSNE.
"""

import os
import warnings
import numpy as np
from typing import Optional, Union
from ._backend import get_backend, TSNEParams, NP_DTYPE


def _resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """Resolve n_jobs to a concrete positive thread count (joblib convention).

    None -> 1, -1 -> all cores, -2 -> all but one, etc. A concrete positive
    count passes through. Without this, a negative n_jobs would reach the C
    extension's ``num_threads`` field, which clamps any value <= 0 back to 1 —
    silently running the entire optimization single-threaded (the C optimization
    is 50-64%% of runtime and threads 2-2.6x, so ``n_jobs=-1`` otherwise loses
    most of the available speedup)."""
    if n_jobs is None:
        return 1
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    if n_jobs == 0:
        return 1
    return n_jobs


# H2 defaults by embedding dimension. The far-field rank of (1 + r^2)^-2 grows with the
# dimension (max 17 in 2D vs 58 in 3D at h2_tol=1e-2), and the ID of every node against
# the proxy points (the U-build) then dominates the 3D rebuild. Tuned on Zheng 250K / 1M,
# 1 and 8 threads of an M1 Pro, one shared P:
#   - h2_tol 3e-2 in 3D: rank 41, 13-20% faster than 1e-2 at 8 threads with the same
#     quality (common-P KL +0.1-0.2%, 10-NN preservation equal or higher). 1e-1 (rank 25)
#     is another ~20% faster for +0.5% KL - a reasonable choice for interactive work.
#   - h2_leaf_size: flat between 50 and 150 at every tolerance (the U-build cost does
#     not depend on it), 400+ slower; 100 as in 2D.
#   - rebuild schedule as in 2D: every 2 costs 35% more time for -0.1% KL, every 4 is
#     +0.35% KL, every 6 falls off a cliff (+2% KL, 10-NN preservation drops).
# 3 and more dimensions share a row.
H2_DEFAULTS = {
    2: dict(h2_tol=1e-2, h2_rebuild_freq=3, h2_rebuild_freq_early=10, h2_leaf_size=100),
    3: dict(h2_tol=3e-2, h2_rebuild_freq=3, h2_rebuild_freq_early=10, h2_leaf_size=100),
}


def h2_defaults(n_components: int) -> dict:
    """The default H2 settings for an embedding dimension (see H2_DEFAULTS)."""
    return dict(H2_DEFAULTS[2 if n_components <= 2 else 3])


class TSNE:
    """
    t-SNE: t-Distributed Stochastic Neighbor Embedding with H2Pack acceleration.

    Drop-in replacement for sklearn.manifold.TSNE with identical API.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly
        different results.

    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them.

    learning_rate : float or 'auto', default='auto'
        The learning rate. 'auto' uses ``max(n_samples / 12, 200)``, the rule
        FIt-SNE and openTSNE use: a fixed rate of a few hundred badly
        under-converges once n_samples reaches the tens of thousands.

    max_iter : int, default=1000
        Maximum number of iterations for the optimization.

    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization.

    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be stopped.

    metric : {'euclidean', 'precomputed'}, default='euclidean'
        'euclidean': X is a feature array and the k-NN graph is built with
        ``knn_method``. 'precomputed': X is a sparse (n_samples, n_samples)
        matrix of k-NN distances (not squared), as returned by
        ``sklearn.neighbors.kneighbors_graph(..., mode='distance')`` or any
        other k-NN library; every row needs more than ``perplexity`` neighbors
        (3 * perplexity is what the 'euclidean' path uses). The k-NN search is
        skipped, so several runs, or several libraries, can share one graph.

    init : {'random', 'pca'} or ndarray of shape (n_samples, n_components), default='pca'
        Initialization of the embedding. 'pca' follows scikit-learn: the PCA
        projection of X (``svd_solver='randomized'``, seeded with
        ``random_state``) rescaled so that the first coordinate has standard
        deviation 1e-4; it usually preserves the global structure better.
        'random' draws each coordinate from N(0, 1e-4) in the C extension.
        PCA needs features, so with ``metric='precomputed'`` an init of 'pca'
        falls back to 'random' with a warning (pass ``init='random'`` to
        silence it). An ndarray is used as the initial embedding as is (no
        rescaling).

        .. versionchanged:: 3.3.0
           The default changed from 'random' to 'pca', as in scikit-learn 1.2.

    verbose : int, default=0
        Verbosity level. If > 0, prints progress information.

    random_state : int or None, default=None
        Random seed for reproducibility.

    method : str, default='h2pack'
        Method for repulsive force computation:
        - 'h2pack': Use H2Pack for O(N) computation (default)
        - 'exact': Use direct O(N²) computation

    n_jobs : int or None, default=None
        Number of threads for the k-NN search, the optimizer and H2Pack.
        None means 1 thread; -1 means all cores (joblib convention).

    h2_tol : float, default=None
        H2Pack low-rank approximation tolerance. Lower values = more accurate
        but slower. None selects the default for ``n_components`` (see
        ``H2_DEFAULTS``; 1e-2 in 2D, which already reproduces the embedding
        quality of an FFT (openTSNE) run). Only used when method='h2pack'.

    h2_rebuild_freq : int, default=None
        The H2 matrix is rebuilt, and the repulsive force recomputed, every
        this many iterations; in between, the last repulsive force is reused.
        Lower values = more accurate but slower (in 2D, 2 is slightly better
        than FIt-SNE's quality, 3 is on par). None selects the default for
        ``n_components``. Only used when method='h2pack'.

    h2_rebuild_freq_early : int, default=None
        Rebuild frequency during early exaggeration (after the first 20
        iterations), a phase that is insensitive to a delayed repulsive force.
        Never lower than ``h2_rebuild_freq``. None selects the default for
        ``n_components``.

    h2_leaf_size : int, default=None
        Maximum number of points per H2 leaf box. Affects speed and memory
        only, not accuracy. None selects the default for ``n_components``.

    enable_profiling : bool, default=False
        Enable detailed profiling of computation time breakdown.

    profiling_output_file : str or None, default=None
        Path to save profiling CSV data. If None, uses default name.

    knn_method : str, default='pynndescent'
        k-NN method for finding nearest neighbors:
        - 'pynndescent': PyNNDescent approximate k-NN (default, ~96% recall, auto-tuning)
        - 'exact': sklearn brute-force exact k-NN
        - 'balltree': sklearn ball tree exact k-NN
        - 'faiss': FAISS approximate k-NN (CPU-only, very fast)
        - 'annoy': Annoy approximate k-NN (memory-efficient)

    knn_params : dict or None, default=None
        Method-specific parameters for k-NN computation. See _knn_methods.py
        for details on each method's parameters. Common parameters:

        PyNNDescent (auto-tuning default):
            - n_trees: Number of RP trees (None=auto, default varies by dataset size)
            - n_iters: Search iterations (None=auto, default varies by dataset size)
            - metric: Distance metric (default='euclidean')

        FAISS:
            - index_type: 'Flat', 'IVFFlat', 'IVFPQ', 'HNSW' (default='IVFFlat')
            - nlist: Number of clusters for IVF (default=100)
            - nprobe: Clusters to search (default=10)
            - use_gpu: Use GPU if available (default=False)

        Annoy:
            - n_trees: Number of trees (default=50)
            - search_k: Search depth (default=-1, auto)

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the embedding vectors.

    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.

    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> from tsne_h2pack import TSNE
    >>> import numpy as np
    >>> X = np.random.randn(1000, 50)
    >>> tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    >>> Y = tsne.fit_transform(X)
    >>> print(Y.shape)
    (1000, 2)
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: Union[float, str] = 'auto',
        max_iter: int = 1000,
        n_iter_without_progress: int = 300,
        min_grad_norm: float = 1e-7,
        metric: str = 'euclidean',
        init: Union[str, np.ndarray] = 'pca',
        verbose: int = 0,
        random_state: Optional[int] = None,
        method: str = 'h2pack',
        n_jobs: Optional[int] = None,
        h2_tol: Optional[float] = None,
        h2_rebuild_freq: Optional[int] = None,
        enable_profiling: bool = False,
        profiling_output_file: Optional[str] = None,
        knn_method: str = 'pynndescent',
        knn_params: Optional[dict] = None,
        h2_rebuild_freq_early: Optional[int] = None,
        h2_leaf_size: Optional[int] = None,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.n_jobs = _resolve_n_jobs(n_jobs)
        # None -> the default for this embedding dimension (resolved here so that
        # the attributes always hold the values the run will use)
        d = h2_defaults(n_components if isinstance(n_components, int) and n_components >= 1 else 2)
        self.h2_tol = d["h2_tol"] if h2_tol is None else h2_tol
        self.h2_rebuild_freq = d["h2_rebuild_freq"] if h2_rebuild_freq is None else h2_rebuild_freq
        self.enable_profiling = enable_profiling
        self.profiling_output_file = profiling_output_file
        self.knn_method = knn_method
        self.knn_params = knn_params or {}
        self.h2_rebuild_freq_early = d["h2_rebuild_freq_early"] if h2_rebuild_freq_early is None else h2_rebuild_freq_early
        self.h2_leaf_size = d["h2_leaf_size"] if h2_leaf_size is None else h2_leaf_size

        # Attributes set after fitting
        self.embedding_: Optional[np.ndarray] = None
        self.kl_divergence_: Optional[float] = None
        self.n_iter_: Optional[int] = None

        # Validate parameters
        self._validate_params()

    def _validate_params(self):
        """Validate input parameters"""
        if self.metric not in ('euclidean', 'precomputed'):
            raise ValueError(
                f"metric='{self.metric}' not supported. "
                "Use 'euclidean' or 'precomputed'."
            )

        if isinstance(self.init, str):
            if self.init not in ('random', 'pca'):
                raise ValueError(
                    f"init='{self.init}' not supported. "
                    "Must be 'random', 'pca' or an ndarray of shape "
                    "(n_samples, n_components)."
                )
        else:
            init = np.asarray(self.init)
            if init.ndim != 2 or init.shape[1] != self.n_components:
                raise ValueError(
                    "init not supported: must be 'random', 'pca' or an ndarray of "
                    f"shape (n_samples, n_components={self.n_components}), got "
                    f"{'an array of shape ' + str(init.shape) if init.ndim else repr(self.init)}"
                )
            if not np.issubdtype(init.dtype, np.number) or not np.isfinite(init).all():
                raise ValueError("init array must contain only finite numbers")

        if self.method not in ['h2pack', 'exact']:
            raise ValueError(
                f"method='{self.method}' not supported. "
                "Must be 'h2pack' or 'exact'."
            )

        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")

        if self.perplexity <= 0:
            raise ValueError("perplexity must be > 0")

        if isinstance(self.learning_rate, str):
            if self.learning_rate != 'auto':
                raise ValueError("learning_rate must be a positive number or 'auto'")
        elif self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        if self.h2_rebuild_freq < 1 or self.h2_rebuild_freq_early < 1:
            raise ValueError("h2_rebuild_freq and h2_rebuild_freq_early must be >= 1")

        if self.h2_leaf_size < 1:
            raise ValueError("h2_leaf_size must be >= 1")

        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")

    def _map_params_to_c_struct(self, n_samples: Optional[int] = None) -> TSNEParams:
        """
        Map sklearn parameters to C struct.

        NOTE: With pointer-based params allocation, we can now safely set ALL fields!
        No more heap corruption issues.
        """
        backend = get_backend()
        params = backend.create_default_params()

        # Core algorithm parameters
        params.n_iter = self.max_iter
        params.perplexity = self.perplexity
        if self.learning_rate == 'auto':
            # FIt-SNE / openTSNE rule; without n_samples fall back to the floor
            params.learning_rate = max((n_samples or 0) / 12.0, 200.0)
        else:
            params.learning_rate = self.learning_rate
        params.early_exag = self.early_exaggeration
        params.early_exag_iter = 250

        # Momentum parameters
        params.initial_momentum = 0.5
        params.momentum = 0.8
        params.momentum_switch_iter = 250

        # Output control
        params.verbose = 1 if self.verbose > 0 else 0
        params.print_freq = 50

        # H2Pack settings
        if self.method == 'exact':
            params.h2_rebuild_freq = 999999  # Never rebuild H2
            params.h2_tol = 1e-4
        else:
            params.h2_rebuild_freq = self.h2_rebuild_freq
            params.h2_tol = self.h2_tol
        params.h2_rebuild_freq_early = max(self.h2_rebuild_freq_early, self.h2_rebuild_freq)
        params.h2_leaf_size = self.h2_leaf_size

        # Profiling
        params.enable_profiling = 1 if self.enable_profiling else 0
        if self.profiling_output_file is not None:
            params.profiling_output_file = self.profiling_output_file.encode('utf-8')

        # Random seed
        if self.random_state is not None:
            params.random_seed = self.random_state

        # Threading
        params.num_threads = self.n_jobs

        # Convergence
        params.convergence_patience = self.n_iter_without_progress
        params.min_iter = 250

        # Sklearn compatibility mode
        params.sklearn_compatible = 1

        return params

    def _initial_embedding(self, X, n_samples: int) -> Optional[np.ndarray]:
        """Initial Y as an (n_samples, n_components) array, or None for 'random'
        (which the C extension draws itself from random_state)."""
        self._validate_params()   # init / metric may have been changed after construction
        if isinstance(self.init, str):
            if self.init == 'random':
                return None
            if X is None:   # metric='precomputed': no features to project
                warnings.warn(
                    "init='pca' needs the input features, which metric='precomputed' "
                    "does not provide; using init='random' instead. Pass init='random' "
                    "(or an ndarray) to silence this warning.", UserWarning, stacklevel=4)
                return None
            n_features = X.shape[1]
            if self.n_components > min(n_samples, n_features):
                raise ValueError(
                    f"init='pca' needs n_components={self.n_components} <= "
                    f"min(n_samples, n_features)={min(n_samples, n_features)}"
                )
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=self.random_state)
            # NumPy 2.0-2.2 on macOS (the newest NumPy for Python 3.9 / 3.10) raises
            # spurious divide/overflow/invalid warnings from the matmuls inside
            # scikit-learn's randomized SVD even when the result is exact. Silence
            # them here only, and check the result explicitly instead.
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                Y = pca.fit_transform(np.asarray(X, dtype=np.float64))
            if not np.isfinite(Y).all():
                raise ValueError("init='pca' failed: the PCA of X is not finite; "
                                 "check X for inf/NaN or pass init='random'")
            std = np.std(Y[:, 0])
            if not np.isfinite(std) or std == 0:
                raise ValueError("init='pca' failed: X has no variance along "
                                 "its first principal component")
            # scikit-learn's scaling: first coordinate std 1e-4, like 'random'
            return Y / std * 1e-4

        Y = np.array(self.init, dtype=np.float64)
        if Y.shape != (n_samples, self.n_components):
            raise ValueError(
                f"init array has shape {Y.shape}, expected "
                f"(n_samples, n_components)={(n_samples, self.n_components)}"
            )
        return Y

    def _fit_transform_precomputed(self, X):
        """Embed from a sparse matrix of k-NN distances (metric='precomputed')"""
        from scipy.sparse import issparse, csr_matrix

        if not issparse(X):
            raise ValueError(
                "metric='precomputed' expects a sparse matrix of k-NN distances, "
                "e.g. kneighbors_graph(X, n_neighbors=90, mode='distance')"
            )
        if X.shape[0] != X.shape[1]:
            raise ValueError(f"precomputed k-NN graph must be square, got {X.shape}")

        graph = csr_matrix(X, copy=True)
        graph.setdiag(0)              # a point is not its own neighbor
        graph.eliminate_zeros()
        graph.sort_indices()
        if not np.isfinite(graph.data).all() or (graph.data < 0).any():
            raise ValueError("precomputed distances must be finite and non-negative")

        min_neighbors = int(np.diff(graph.indptr).min())
        if min_neighbors <= self.perplexity:
            raise ValueError(
                f"every row needs more than perplexity={self.perplexity} neighbors, "
                f"but one row has {min_neighbors}"
            )

        backend = get_backend()
        params = self._map_params_to_c_struct(n_samples=graph.shape[0])
        if self.verbose:
            print(f"Running t-SNE with backend: {backend.backend_type}")
            print(f"Precomputed k-NN graph: {graph.shape[0]} points, {graph.nnz} distances")

        init_Y = self._initial_embedding(None, graph.shape[0])
        self.knn_time_ = 0.0
        Y, metrics = backend.fit_transform_with_knn_distances(
            None, graph, self.n_components, params, init_Y=init_Y
        )
        self.embedding_ = Y
        self.n_iter_ = metrics.get('n_iter', params.n_iter)
        self.kl_divergence_ = metrics.get('kl_divergence', 0.0)
        return self.embedding_

    def fit(self, X: np.ndarray, y=None):
        """
        Fit X into an embedded space (not implemented).

        Use fit_transform instead.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : Ignored
            Not used, present for API consistency by sklearn convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        raise NotImplementedError(
            "fit() is not implemented. Use fit_transform() instead."
        )

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : Ignored
            Not used, present for API consistency by sklearn convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        # Input validation
        if self.metric == 'precomputed':
            return self._fit_transform_precomputed(X)
        X = self._validate_data(X)
        init_Y = self._initial_embedding(X, X.shape[0])

        # Get backend
        backend = get_backend()

        # Map parameters
        params = self._map_params_to_c_struct(n_samples=X.shape[0])

        # Run t-SNE
        if self.verbose:
            print(f"Running t-SNE with backend: {backend.backend_type}")
            print(f"Input shape: {X.shape}")
            print(f"Perplexity: {self.perplexity}, max_iter: {self.max_iter}")
            print(f"k-NN method: {self.knn_method}")

        # Use Python-based k-NN method (C ball tree removed)
        from ._knn_methods import get_knn_method

        knn_method = get_knn_method(self.knn_method)
        k = min(X.shape[0] - 1, int(3 * self.perplexity))

        if self.verbose:
            print(f"Computing {k} nearest neighbors using {self.knn_method}...")

        knn_distances_csr = knn_method.compute(
            X, k=k, n_jobs=self.n_jobs,
            verbose=(self.verbose > 0), **self.knn_params
        )

        # Store k-NN timing
        self.knn_time_ = getattr(knn_distances_csr, 'knn_time', 0.0)

        # C code computes P matrix from distances
        Y, metrics = backend.fit_transform_with_knn_distances(
            X, knn_distances_csr, self.n_components, params, init_Y=init_Y
        )

        # Store results
        self.embedding_ = Y
        self.n_iter_ = metrics.get('n_iter', params.n_iter)
        self.kl_divergence_ = metrics.get('kl_divergence', 0.0)

        return self.embedding_

    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        """Validate and prepare input data"""
        # Convert to numpy array if needed
        X = np.asarray(X, dtype=NP_DTYPE)

        # Check shape
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional array, got {X.ndim} dimensions"
            )

        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError(
                f"n_samples={n_samples} should be >= 2"
            )

        # Check perplexity
        if self.perplexity >= n_samples:
            raise ValueError(
                f"perplexity={self.perplexity} must be less than "
                f"n_samples={n_samples}"
            )

        # Check for NaN/inf
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf values")

        return X
