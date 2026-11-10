"""
k-NN method implementations for t-SNE.

This module provides a unified interface for different k-nearest neighbor methods.
All methods return scipy.sparse.csr_matrix of L2 distances (NOT squared).

Available methods:
- exact: sklearn brute-force exact kNN (algorithm='brute')
- balltree: sklearn ball tree exact kNN (algorithm='ball_tree')
- pynndescent: PyNNDescent approximate kNN (recommended, ~96% recall, auto-tuning)
- faiss: FAISS approximate kNN (CPU-only, very fast)
- annoy: Annoy approximate kNN (memory-efficient)
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Optional
import time


def _drop_self(indices, distances):
    """(n, k + 1) neighbor arrays -> (n, k) without the point itself.

    A search over the data returns the query point as its own nearest neighbor, but
    not always in column 0 (exact duplicates tie at distance 0, and an approximate
    search can miss it). The self entry is located per row; where absent, the last
    (farthest) column is dropped instead, so every row keeps k real neighbors.
    """
    n, k1 = indices.shape
    is_self = indices == np.arange(n)[:, None]
    drop = np.where(is_self.any(axis=1), is_self.argmax(axis=1), k1 - 1)
    keep = np.ones_like(indices, dtype=bool)
    keep[np.arange(n), drop] = False
    return indices[keep].reshape(n, k1 - 1), distances[keep].reshape(n, k1 - 1)


class KNNMethodBase:
    """Base class for k-NN methods"""

    def compute(
        self,
        X: np.ndarray,
        k: int,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs
    ) -> csr_matrix:
        """
        Compute k-nearest neighbors and return CSR distance matrix.

        Args:
            X: Input data (n_samples, n_features)
            k: Number of nearest neighbors
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Print progress information
            **kwargs: Method-specific parameters

        Returns:
            distances_csr: Sparse CSR matrix of L2 distances (NOT squared)
                          Shape: (n_samples, n_samples), nnz = n_samples * k
                          Attributes: .knn_time (float) - timing in seconds

        Notes:
            - Returns L2 distances (NOT squared) - C code will square them
            - Point is NOT its own neighbor (self-exclusion)
            - CSR format for memory efficiency
        """
        raise NotImplementedError


class ExactKNN(KNNMethodBase):
    """sklearn brute-force exact kNN (algorithm='brute')"""

    def compute(
        self,
        X: np.ndarray,
        k: int,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs
    ) -> csr_matrix:
        """Compute exact k-NN using sklearn brute-force search"""
        from sklearn.neighbors import NearestNeighbors

        n_samples = X.shape[0]

        if verbose:
            print(f"[t-SNE kNN] Using exact brute-force search for {k} neighbors...")

        t_start = time.time()

        # Use brute-force algorithm (exact)
        nn = NearestNeighbors(
            n_neighbors=k,
            algorithm='brute',
            metric='euclidean',
            n_jobs=n_jobs
        )
        nn.fit(X)

        # kneighbors_graph automatically excludes self
        distances_csr = nn.kneighbors_graph(mode='distance')

        # Convert to float32 for memory efficiency
        distances_csr.data = distances_csr.data.astype(np.float32, copy=False)

        t_elapsed = time.time() - t_start

        if verbose:
            print(f"[t-SNE kNN] Exact search completed in {t_elapsed:.3f}s")
            memory_mb = (distances_csr.data.nbytes +
                        distances_csr.indices.nbytes +
                        distances_csr.indptr.nbytes) / (1024 * 1024)
            print(f"[t-SNE kNN] CSR memory: {memory_mb:.2f} MB")

        # Store timing
        distances_csr.knn_time = t_elapsed

        return distances_csr


class BallTreeKNN(KNNMethodBase):
    """sklearn ball tree exact kNN (algorithm='ball_tree')"""

    def compute(
        self,
        X: np.ndarray,
        k: int,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs
    ) -> csr_matrix:
        """Compute exact k-NN using sklearn ball tree"""
        from sklearn.neighbors import NearestNeighbors

        n_samples = X.shape[0]

        if verbose:
            print(f"[t-SNE kNN] Using ball tree search for {k} neighbors...")

        t_start = time.time()

        # Use ball tree algorithm (exact, optimized for low-medium dimensions)
        nn = NearestNeighbors(
            n_neighbors=k,
            algorithm='ball_tree',
            metric='euclidean',
            n_jobs=n_jobs,
            leaf_size=kwargs.get('leaf_size', 30)
        )
        nn.fit(X)

        # kneighbors_graph automatically excludes self
        distances_csr = nn.kneighbors_graph(mode='distance')

        # Convert to float32 for memory efficiency
        distances_csr.data = distances_csr.data.astype(np.float32, copy=False)

        t_elapsed = time.time() - t_start

        if verbose:
            print(f"[t-SNE kNN] Ball tree search completed in {t_elapsed:.3f}s")
            memory_mb = (distances_csr.data.nbytes +
                        distances_csr.indices.nbytes +
                        distances_csr.indptr.nbytes) / (1024 * 1024)
            print(f"[t-SNE kNN] CSR memory: {memory_mb:.2f} MB")

        # Store timing
        distances_csr.knn_time = t_elapsed

        return distances_csr


class PyNNDescentKNN(KNNMethodBase):
    """PyNNDescent approximate kNN (recommended, ~96% recall, auto-tuning)"""

    EXACT_BELOW = 10000   # smaller datasets use exact search, see compute()

    def _auto_tune_params(self, n_samples: int, kwargs: dict) -> dict:
        """Auto-tune parameters based on dataset size"""
        params = {}

        # Auto-tune n_trees
        if 'n_trees' in kwargs and kwargs['n_trees'] is not None:
            params['n_trees'] = kwargs['n_trees']
        else:
            if n_samples < 10000:
                params['n_trees'] = 10  # Small datasets
            elif n_samples < 100000:
                params['n_trees'] = 15  # Medium datasets (default target)
            else:
                params['n_trees'] = 20  # Large datasets

        # Auto-tune n_iters
        if 'n_iters' in kwargs and kwargs['n_iters'] is not None:
            params['n_iters'] = kwargs['n_iters']
        else:
            if n_samples < 10000:
                params['n_iters'] = 5
            elif n_samples < 100000:
                params['n_iters'] = 10
            else:
                params['n_iters'] = 15

        # Other parameters
        params['metric'] = kwargs.get('metric', 'euclidean')
        params['max_candidates'] = kwargs.get('max_candidates', 60)
        params['pruning_degree_multiplier'] = kwargs.get('pruning_degree_multiplier', 1.5)
        params['leaf_size'] = kwargs.get('leaf_size', 30)

        return params

    def compute(
        self,
        X: np.ndarray,
        k: int,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs
    ) -> csr_matrix:
        """Compute approximate k-NN using PyNNDescent"""
        # Below ~10K points brute force takes about a second, and NN-descent is at
        # its worst: t-SNE asks for k = 3 * perplexity (90) neighbors, a large
        # fraction of a small dataset, and the graph it returns is poor enough to
        # show in the embedding (Zheng 5K: common-P KL 1.61 vs 1.44 with exact
        # neighbors, kNN-preservation 0.40 vs 0.44). Explicit n_trees / n_iters
        # mean the caller really wants NN-descent.
        if (X.shape[0] < self.EXACT_BELOW and kwargs.get('n_trees') is None
                and kwargs.get('n_iters') is None):
            if verbose:
                print(f"[t-SNE kNN] {X.shape[0]} samples < {self.EXACT_BELOW}: "
                      f"using exact search instead of PyNNDescent")
            return ExactKNN().compute(X, k, n_jobs=n_jobs, verbose=verbose)

        try:
            from pynndescent import NNDescent
        except ImportError:
            raise ImportError(
                "PyNNDescent not installed. Install with:\n"
                "  pip install tsne-h2pack[pynndescent]\n"
                "Or for all k-NN methods:\n"
                "  pip install tsne-h2pack[knn]"
            )

        n_samples = X.shape[0]

        # Auto-tune parameters
        params = self._auto_tune_params(n_samples, kwargs)

        if verbose:
            print(f"[t-SNE kNN] Using PyNNDescent for {k} neighbors...")
            print(f"[t-SNE kNN] Auto-tuned params: n_trees={params['n_trees']}, "
                  f"n_iters={params['n_iters']}")

        t_start = time.time()

        # Build index with auto-tuned parameters. neighbor_graph lists each point as
        # its own first neighbor (distance 0), so ask for k + 1 and drop that entry.
        index = NNDescent(
            X,
            metric=params['metric'],
            n_neighbors=k + 1,
            n_trees=params['n_trees'],
            n_iters=params['n_iters'],
            max_candidates=params['max_candidates'],
            pruning_degree_multiplier=params['pruning_degree_multiplier'],
            leaf_size=params['leaf_size'],
            n_jobs=n_jobs,
            verbose=verbose
        )
        indices, distances = index.neighbor_graph
        indices, distances = _drop_self(indices, distances)

        # Convert to CSR format: (n_samples, k) arrays
        indptr = np.arange(0, (n_samples + 1) * k, k, dtype=np.int32)
        distances_csr = csr_matrix(
            (distances.ravel().astype(np.float32),
             indices.ravel().astype(np.int32),
             indptr),
            shape=(n_samples, n_samples)
        )

        t_elapsed = time.time() - t_start

        if verbose:
            print(f"[t-SNE kNN] PyNNDescent completed in {t_elapsed:.3f}s")
            memory_mb = (distances_csr.data.nbytes +
                        distances_csr.indices.nbytes +
                        distances_csr.indptr.nbytes) / (1024 * 1024)
            print(f"[t-SNE kNN] CSR memory: {memory_mb:.2f} MB")

        # Store timing
        distances_csr.knn_time = t_elapsed

        return distances_csr


class FAISSKNN(KNNMethodBase):
    """FAISS approximate kNN (CPU-only, very fast for large datasets)"""

    def _auto_tune_hnsw_params(self, n_samples: int, kwargs: dict) -> dict:
        """Auto-tune HNSW parameters based on dataset size"""
        params = {}

        # Auto-tune M (graph connectivity) based on dataset size
        # Larger datasets need higher M to ensure connectivity
        if 'M' in kwargs and kwargs['M'] is not None:
            params['M'] = kwargs['M']
        else:
            if n_samples < 10000:
                params['M'] = 16
            elif n_samples < 50000:
                params['M'] = 32
            elif n_samples < 200000:
                params['M'] = 48
            else:
                params['M'] = 64  # Large datasets (200K+)

        # ef_construction: controls index build quality
        params['ef_construction'] = kwargs.get('ef_construction', max(200, params['M'] * 2))

        # ef_search: controls search quality at query time
        # FAISS doesn't expose this directly, we'll set it after index creation
        params['ef_search'] = kwargs.get('ef_search', max(128, params['M'] * 2))

        return params

    def compute(
        self,
        X: np.ndarray,
        k: int,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs
    ) -> csr_matrix:
        """Compute approximate k-NN using FAISS"""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with:\n"
                "  pip install tsne-h2pack[faiss]\n"
                "Or for all k-NN methods:\n"
                "  pip install tsne-h2pack[knn]\n"
                "For GPU support:\n"
                "  pip install faiss-gpu"
            )

        # Set number of threads for FAISS operations
        # FAISS uses OpenMP for CPU parallelization
        if n_jobs == -1:
            # -1 means use all available cores
            import multiprocessing
            n_threads = multiprocessing.cpu_count()
        else:
            n_threads = max(1, n_jobs)

        faiss.omp_set_num_threads(n_threads)

        if verbose:
            print(f"[t-SNE kNN] FAISS using {n_threads} threads")

        n_samples, n_features = X.shape

        # Get parameters
        index_type = kwargs.get('index_type', 'IVFFlat')
        nlist = kwargs.get('nlist', min(max(100, n_samples // 100), n_samples // 10))
        nprobe = kwargs.get('nprobe', min(20, nlist // 5))
        use_gpu = kwargs.get('use_gpu', False)

        if verbose:
            print(f"[t-SNE kNN] Using FAISS {index_type} for {k} neighbors...")
            if index_type in ['IVFFlat', 'IVFPQ']:
                print(f"[t-SNE kNN] IVF params: nlist={nlist}, nprobe={nprobe}")

        t_start = time.time()

        # Ensure float32 (FAISS requirement)
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)

        # Create index based on type
        if index_type == 'Flat':
            index = faiss.IndexFlatL2(n_features)
        elif index_type == 'IVFFlat':
            quantizer = faiss.IndexFlatL2(n_features)
            index = faiss.IndexIVFFlat(quantizer, n_features, nlist)
            index.train(X_f32)
            index.nprobe = nprobe
        elif index_type == 'IVFPQ':
            quantizer = faiss.IndexFlatL2(n_features)
            m = kwargs.get('m', 8)  # Number of subquantizers
            index = faiss.IndexIVFPQ(quantizer, n_features, nlist, m, 8)
            index.train(X_f32)
            index.nprobe = nprobe
        elif index_type == 'HNSW':
            # Auto-tune HNSW parameters for large datasets
            hnsw_params = self._auto_tune_hnsw_params(n_samples, kwargs)
            M = hnsw_params['M']
            ef_construction = hnsw_params['ef_construction']
            ef_search = hnsw_params['ef_search']

            if verbose:
                print(f"[t-SNE kNN] HNSW params: M={M}, ef_construction={ef_construction}, ef_search={ef_search}")

            index = faiss.IndexHNSWFlat(n_features, M)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
        else:
            raise ValueError(f"Unknown FAISS index_type: {index_type}")

        # GPU support (if requested and available)
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                if verbose:
                    print(f"[t-SNE kNN] Using GPU acceleration")
            except Exception as e:
                if verbose:
                    print(f"[t-SNE kNN] GPU not available, using CPU: {e}")

        # Add vectors to index
        index.add(X_f32)

        # Search for k+1 neighbors (includes self)
        distances_sq, indices = index.search(X_f32, k + 1)

        # CRITICAL: Validate indices (FAISS may return -1 for unfound neighbors)
        # This happens when FAISS cannot find k neighbors for some samples
        # Observed in 0.0004% of samples for 240K datasets with HNSW M=32
        invalid_mask = indices[:, 1:] < 0  # Check non-self neighbors
        n_invalid = invalid_mask.sum()

        if n_invalid > 0:
            import warnings
            affected_samples = invalid_mask.any(axis=1).sum()
            warnings.warn(
                f"FAISS {index_type} returned {n_invalid} invalid neighbors (index = -1) "
                f"for {affected_samples}/{n_samples} samples. "
                f"This indicates FAISS couldn't find k={k} neighbors for some points.\n"
                f"Solutions:\n"
                f"  1. Use PyNNDescent instead: knn_method='pynndescent' (recommended)\n"
                f"  2. For FAISS HNSW, increase M: knn_params={{'M': 64}} or higher\n"
                f"  3. For FAISS IVF, increase nprobe: knn_params={{'nprobe': 40}}\n"
                f"Replacing invalid entries with duplicates of furthest valid neighbor.",
                UserWarning
            )

            # Fix invalid neighbors row by row
            for i in range(n_samples):
                invalid_cols = np.where(indices[i, 1:] < 0)[0]
                if len(invalid_cols) > 0:
                    # Find last valid neighbor (excluding self at position 0)
                    valid_cols = np.where(indices[i, 1:] >= 0)[0]
                    if len(valid_cols) > 0:
                        # Get last valid neighbor's data
                        last_valid_idx = valid_cols[-1] + 1  # +1 to account for self at 0
                        last_valid_neighbor = indices[i, last_valid_idx]
                        last_valid_dist_sq = distances_sq[i, last_valid_idx]

                        # Replace all invalid entries with last valid neighbor
                        for inv_col in invalid_cols:
                            actual_col = inv_col + 1  # +1 to account for self at 0
                            indices[i, actual_col] = last_valid_neighbor
                            # Use distance slightly larger than last valid (10% increase)
                            distances_sq[i, actual_col] = last_valid_dist_sq * 1.1
                    else:
                        # Extreme edge case: no valid neighbors found at all
                        # Fall back to self-loop (will be removed below)
                        for inv_col in invalid_cols:
                            actual_col = inv_col + 1
                            indices[i, actual_col] = i  # Self
                            distances_sq[i, actual_col] = 0.0

        # CRITICAL: Remove self (first neighbor)
        distances_sq = distances_sq[:, 1:]
        indices = indices[:, 1:]

        # CRITICAL: Convert squared L2 to L2 (FAISS returns L2²)
        distances = np.sqrt(distances_sq)

        # Convert to CSR format
        indptr = np.arange(0, (n_samples + 1) * k, k, dtype=np.int32)
        distances_csr = csr_matrix(
            (distances.ravel().astype(np.float32),
             indices.ravel().astype(np.int32),
             indptr),
            shape=(n_samples, n_samples)
        )

        t_elapsed = time.time() - t_start

        if verbose:
            print(f"[t-SNE kNN] FAISS completed in {t_elapsed:.3f}s")
            memory_mb = (distances_csr.data.nbytes +
                        distances_csr.indices.nbytes +
                        distances_csr.indptr.nbytes) / (1024 * 1024)
            print(f"[t-SNE kNN] CSR memory: {memory_mb:.2f} MB")

        # Store timing
        distances_csr.knn_time = t_elapsed

        return distances_csr


class AnnoyKNN(KNNMethodBase):
    """Annoy approximate kNN (memory-efficient)"""

    def compute(
        self,
        X: np.ndarray,
        k: int,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs
    ) -> csr_matrix:
        """Compute approximate k-NN using Annoy"""
        try:
            from annoy import AnnoyIndex
        except ImportError:
            raise ImportError(
                "Annoy not installed. Install with:\n"
                "  pip install tsne-h2pack[annoy]\n"
                "Or for all k-NN methods:\n"
                "  pip install tsne-h2pack[knn]"
            )

        n_samples, n_features = X.shape

        # Get parameters
        n_trees = kwargs.get('n_trees', 50)
        search_k = kwargs.get('search_k', -1)
        metric = kwargs.get('metric', 'euclidean')

        if verbose:
            print(f"[t-SNE kNN] Using Annoy with {n_trees} trees for {k} neighbors...")

        t_start = time.time()

        # Build index
        index = AnnoyIndex(n_features, metric)

        for i in range(n_samples):
            index.add_item(i, X[i])

        # Build with specified number of trees (parallel)
        index.build(n_trees, n_jobs=n_jobs)

        # Query neighbors for all points
        # Note: Annoy queries are very fast and memory-bound; parallelizing
        # with joblib/threads adds overhead that exceeds the query time itself.
        # Only the build step benefits from parallelization.
        all_indices = []
        all_distances = []
        row_counts = []
        degenerate_rows = 0

        for i in range(n_samples):
            idx, dist = index.get_nns_by_item(
                i, k + 1, search_k=search_k, include_distances=True
            )
            idx_arr = np.array(idx, dtype=np.int32)
            dist_arr = np.array(dist, dtype=np.float32)

            # Remove self only (matches ExactKNN; do not drop near-duplicates —
            # the C code handles zero distances).
            mask = (idx_arr != i)
            idx_arr = idx_arr[mask][:k]
            dist_arr = dist_arr[mask][:k]

            if len(idx_arr) < k:
                # A healthy Annoy index returns k neighbors for every point when
                # n_samples > k. A short row means Annoy returned too few — track
                # it; the check after the loop fails loudly if it is widespread.
                degenerate_rows += 1
                pad_size = k - len(idx_arr)
                pad_idx = idx_arr[-1] if len(idx_arr) else i
                idx_arr = np.concatenate([idx_arr, np.full(pad_size, pad_idx, dtype=np.int32)])
                dist_arr = np.concatenate([dist_arr, np.full(pad_size, 1e10, dtype=np.float32)])

            all_indices.extend(idx_arr)
            all_distances.extend(dist_arr)
            row_counts.append(len(idx_arr))

        # Defensive check: if Annoy failed to return k neighbors for a meaningful
        # fraction of points (when the dataset is large enough that it should),
        # the `annoy` package is malfunctioning in this environment — a known
        # failure mode of some annoy builds (e.g. on Python 3.13 it can return
        # only the query point itself, or SIGBUS). Fail loudly with a clear
        # remedy instead of silently producing a diverged embedding.
        if n_samples > k + 1 and degenerate_rows > 0.01 * n_samples:
            raise RuntimeError(
                f"Annoy returned fewer than k={k} neighbors for "
                f"{degenerate_rows}/{n_samples} points — the 'annoy' package "
                f"appears to be malfunctioning in this environment (a known "
                f"issue with some annoy builds, notably on Python 3.13). "
                f"Use knn_method='pynndescent' (the default) or 'exact' instead."
            )

        # Convert to CSR format
        indices = np.array(all_indices, dtype=np.int32)
        distances = np.array(all_distances, dtype=np.float32)

        # Build indptr: cumulative sum of row counts
        indptr = np.zeros(n_samples + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(row_counts)

        distances_csr = csr_matrix(
            (distances, indices, indptr),
            shape=(n_samples, n_samples)
        )

        t_elapsed = time.time() - t_start

        if verbose:
            print(f"[t-SNE kNN] Annoy completed in {t_elapsed:.3f}s")
            memory_mb = (distances_csr.data.nbytes +
                        distances_csr.indices.nbytes +
                        distances_csr.indptr.nbytes) / (1024 * 1024)
            print(f"[t-SNE kNN] CSR memory: {memory_mb:.2f} MB")

        # Store timing
        distances_csr.knn_time = t_elapsed

        return distances_csr


def get_knn_method(method: str) -> KNNMethodBase:
    """
    Factory function to get k-NN method by name.

    Args:
        method: Method name ('exact', 'balltree', 'pynndescent', 'faiss', 'annoy')

    Returns:
        KNNMethodBase instance

    Raises:
        ValueError: If method name is unknown
    """
    METHODS = {
        'exact': ExactKNN,
        'balltree': BallTreeKNN,
        'pynndescent': PyNNDescentKNN,
        'faiss': FAISSKNN,
        'annoy': AnnoyKNN
    }

    if method not in METHODS:
        raise ValueError(
            f"Unknown knn_method='{method}'. "
            f"Valid options: {list(METHODS.keys())}"
        )

    return METHODS[method]()
