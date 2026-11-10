"""
Backend abstraction layer for tsne-h2pack.

Handles platform detection and loading of appropriate backend:
- Linux: Native C extension via ctypes
- macOS: Native C extension with Homebrew GCC (Apple Silicon only)
- Windows: Not supported (use WSL2)
"""

import ctypes
import sys
import platform
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# Platform detection
IS_LINUX = platform.system() == 'Linux'
IS_MACOS = platform.system() == 'Darwin'
IS_WINDOWS = platform.system() == 'Windows'


# Define DTYPE for C library (double precision)
DTYPE = ctypes.c_double
NP_DTYPE = np.float64


# C structure definitions (must match tsne_h2pack.h)

class TSNEParams(ctypes.Structure):
    """
    Matches tsne_params_t struct from include/tsne_h2pack.h (v2.0.0+)

    CRITICAL: This struct MUST match exactly the C struct layout!
    Field order and types must be identical to avoid memory corruption.

    The C compiler adds padding for natural alignment (doubles/pointers aligned to 8 bytes).
    We must add explicit padding fields to match this layout.

    REMOVED in v2.0.0 (parameter cleanup):
      - enable_vp_comparison (debug-only, always 0)
      - pp_fname (deprecated, replaced by temp cache)
      - enable_h2_comparison (debug-only, always 0)
      - init_Y_file (never used, always NULL)
      - parallel_knn (consolidated into num_threads)
      - parallel_p_matvec (consolidated into num_threads)

    New struct size: ~160 bytes (down from 184 bytes)
    """
    _fields_ = [
        # Algorithm parameters (unchanged)
        ('n_iter', ctypes.c_int),                    # offset 0
        ('_pad1', ctypes.c_int),                     # offset 4
        ('perplexity', DTYPE),                       # offset 8
        ('learning_rate', DTYPE),                    # offset 16
        ('momentum', DTYPE),                         # offset 24
        ('initial_momentum', DTYPE),                 # offset 32
        ('momentum_switch_iter', ctypes.c_int),      # offset 40
        ('_pad2', ctypes.c_int),                     # offset 44
        ('early_exag', DTYPE),                       # offset 48
        ('early_exag_iter', ctypes.c_int),           # offset 56

        # k-NN method (REMOVED enable_vp_comparison)
        ('use_vptree', ctypes.c_int),                # offset 60

        # H2Pack parameters (REMOVED pp_fname, enable_h2_comparison)
        ('h2_tol', DTYPE),                           # offset 64 (C: 64, NO padding!)
        ('h2_rebuild_freq', ctypes.c_int),           # offset 72 (C: 72)
        ('h2pack_print_level', ctypes.c_int),        # offset 76 (C: 76)

        # Initialization (REMOVED init_Y_file)
        ('init_method', ctypes.c_int),               # offset 80 (C: 80)
        ('random_seed', ctypes.c_int),               # offset 84 (C: 84)
        ('verbose', ctypes.c_int),                   # offset 88 (C: 88)
        ('print_freq', ctypes.c_int),                # offset 92 (C: 92)

        # Convergence detection (unchanged)
        ('enable_early_stopping', ctypes.c_int),     # offset 96 (C: 96)
        ('_pad4', ctypes.c_int),                     # offset 100
        ('convergence_tol', DTYPE),                  # offset 104 (C: 104)
        ('convergence_patience', ctypes.c_int),      # offset 112 (C: 112)
        ('min_iter', ctypes.c_int),                  # offset 116 (C: 116)

        # Compatibility mode (unchanged)
        ('sklearn_compatible', ctypes.c_int),        # offset 120 (C: 120)

        # Profiling and parallelization (REMOVED parallel_knn, parallel_p_matvec)
        ('enable_profiling', ctypes.c_int),          # offset 124 (C: 124)
        ('profiling_output_file', ctypes.c_char_p),  # offset 128 (C: 128)
        ('num_threads', ctypes.c_int),               # offset 136 (C: 136)
        ('h2_leaf_size', ctypes.c_int),              # offset 140 (C: 140)
        ('h2_rebuild_freq_early', ctypes.c_int),     # offset 144 (C: 144)
        ('_pad5', ctypes.c_int),                     # offset 148
        ('init_Y', ctypes.POINTER(DTYPE)),           # offset 152 (C: 152), struct size 160
    ]


def _optional_ndpointer(**kwargs):
    """np.ctypeslib.ndpointer that also accepts None (passed to C as NULL)."""
    base = np.ctypeslib.ndpointer(**kwargs)

    def from_param(cls, obj):
        return None if obj is None else base.from_param(obj)

    return type(base.__name__, (base,), {'from_param': classmethod(from_param)})


class SparseMatrix(ctypes.Structure):
    """
    Matches sparse_matrix_t struct (CSR format)
    """
    _fields_ = [
        ('n_rows', ctypes.c_int),
        ('n_cols', ctypes.c_int),
        ('nnz', ctypes.c_int),
        ('row_ptr', ctypes.POINTER(ctypes.c_int)),
        ('col_idx', ctypes.POINTER(ctypes.c_int)),
        ('values', ctypes.POINTER(DTYPE)),
    ]


class TSNEWorkspace(ctypes.Structure):
    """
    Opaque workspace structure (only pointer used from Python)
    """
    pass


class TSNEBackend:
    """
    Backend abstraction for tsne-h2pack.

    Automatically selects appropriate backend based on platform:
    - Linux: Native C extension
    - macOS: Native C extension (Homebrew GCC, Apple Silicon only)
    - Windows: Not supported (use WSL2)
    """

    def __init__(self):
        """Initialize backend based on platform"""
        if IS_WINDOWS:
            raise RuntimeError(
                "Windows is not currently supported. "
                "Please use WSL2 (Windows Subsystem for Linux) instead."
            )

        # Both Linux and macOS use native backend
        self._lib = self._load_native_library()
        self._backend_type = 'native'

    @staticmethod
    def _homebrew_libgomp_candidates():
        """Enumerate Homebrew libgomp install paths in preference order."""
        candidates = []
        gcc_cellar = Path('/opt/homebrew/Cellar/gcc')
        if gcc_cellar.exists():
            for gcc_ver in sorted(gcc_cellar.glob('*'), reverse=True):
                candidates.extend([
                    gcc_ver / 'lib' / 'gcc' / 'current' / 'libgomp.1.dylib',
                    gcc_ver / 'lib' / 'gcc' / 'current' / 'libgomp.dylib',
                ])
        candidates.extend([
            Path('/opt/homebrew/lib/libgomp.1.dylib'),
            Path('/opt/homebrew/lib/libgomp.dylib'),
        ])
        return candidates

    @staticmethod
    def _preload_first_match(label, bundle_paths=(), system_paths=(),
                             system_soname_fallbacks=(), hint='', required=False):
        """Try to dlopen a shared library from the first location that works.

        Search order: in-wheel ``bundle_paths`` first, then ``system_paths``
        (absolute paths), then ``system_soname_fallbacks`` (bare sonames
        resolved via the platform loader / ldconfig).

        On success, returns the loaded ``CDLL`` handle. On failure, raises
        ``RuntimeError`` if ``required`` else prints a warning and returns
        ``None``.
        """
        # Try absolute paths first (bundle, then system).
        for candidate in list(bundle_paths) + list(system_paths):
            if candidate.exists():
                try:
                    return ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                except OSError as exc:
                    print(f"Warning: could not load {candidate}: {exc}")
        # Fall back to soname lookup via the loader.
        for soname in system_soname_fallbacks:
            try:
                return ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue

        msg = f"Could not load {label} from bundle or system locations."
        if hint:
            msg += f"\n  Hint: {hint}"
        if required:
            raise RuntimeError(msg)
        print(f"Warning: {msg}")
        return None

    def _load_native_library(self):
        """Load native C library on Linux and macOS.

        Lookup order for each dependency: bundled-in-wheel first
        (`tsne_h2pack/_lib/`), then system locations (Homebrew on macOS,
        ldconfig on Linux). This lets a delocated/auditwheel'd PyPI wheel
        work on machines without Homebrew or matching distro packages,
        while still working in editable-install development mode.
        """
        bundle_dir = Path(__file__).parent / '_lib'

        if IS_MACOS:
            # Pre-load order matters: OpenMP and OpenBLAS before H2Pack.
            self._preload_first_match(
                'libgomp',
                bundle_paths=[
                    bundle_dir / 'libgomp.1.dylib',
                    bundle_dir / 'libgomp.dylib',
                ],
                system_paths=self._homebrew_libgomp_candidates(),
                hint="Install Homebrew GCC with: brew install gcc",
                required=True,
            )
            self._preload_first_match(
                'libopenblas',
                bundle_paths=[
                    bundle_dir / 'libopenblas.dylib',
                    bundle_dir / 'libopenblas.0.dylib',
                ],
                system_paths=[
                    Path('/opt/homebrew/opt/openblas/lib/libopenblas.dylib'),
                    Path('/opt/homebrew/opt/openblas/lib/libopenblas.0.dylib'),
                ],
                hint="Install with: brew install openblas",
                required=True,
            )
            # LAPACKE symbols ship inside libopenblas on macOS — nothing to load.

            # H2Pack — also try development location (H2Pack/lib/) and static (.a).
            self._preload_first_match(
                'libH2Pack',
                bundle_paths=[bundle_dir / 'libH2Pack.so'],
                system_paths=[
                    Path(__file__).parent.parent / 'H2Pack' / 'lib' / 'libH2Pack.dylib',
                    Path(__file__).parent.parent / 'H2Pack' / 'lib' / 'libH2Pack.so',
                    Path(__file__).parent.parent / 'H2Pack' / 'lib' / 'libH2Pack.a',
                ],
                hint="Reinstall the package: pip install --force-reinstall tsne-h2pack",
                required=True,
            )

        else:  # Linux
            # libgomp — bundled by auditwheel for manylinux, otherwise from system.
            self._preload_first_match(
                'libgomp',
                bundle_paths=[
                    bundle_dir / 'libgomp.so.1',
                    bundle_dir / 'libgomp.so',
                ],
                system_soname_fallbacks=['libgomp.so.1', 'libgomp.so'],
                hint="Install with: sudo apt-get install libgomp1",
                required=False,  # warn-only; some distros static-link gomp
            )
            # OpenBLAS — provides cblas + lapacke symbols.
            self._preload_first_match(
                'libopenblas',
                bundle_paths=[
                    bundle_dir / 'libopenblas.so.0',
                    bundle_dir / 'libopenblas.so',
                ],
                system_soname_fallbacks=['libopenblas.so.0', 'libopenblas.so'],
                hint="Install with: sudo apt-get install libopenblas0",
                required=False,
            )
            # LAPACKE — sometimes a separate library (Debian/Ubuntu).
            self._preload_first_match(
                'liblapacke',
                bundle_paths=[
                    bundle_dir / 'liblapacke.so.3',
                    bundle_dir / 'liblapacke.so',
                ],
                system_soname_fallbacks=['liblapacke.so.3', 'liblapacke.so'],
                hint="Install with: sudo apt-get install liblapacke",
                required=False,
            )
            self._preload_first_match(
                'libH2Pack',
                bundle_paths=[bundle_dir / 'libH2Pack.so'],
                system_paths=[
                    Path(__file__).parent.parent / 'H2Pack' / 'lib' / 'libH2Pack.so',
                ],
                system_soname_fallbacks=['libH2Pack.so'],
                hint="Reinstall the package: pip install --force-reinstall tsne-h2pack",
                required=False,
            )

        # Try to find the compiled library
        if IS_MACOS:
            # macOS library naming
            lib_candidates = [
                # Installed package location (pip install)
                Path(__file__).parent / '_lib' / f'libtsne_h2pack.cpython-{sys.version_info.major}{sys.version_info.minor}-darwin.so',
                # Generic installed location
                Path(__file__).parent / '_lib' / 'libtsne_h2pack.so',
                # Development build location
                Path(__file__).parent.parent / 'build' / 'lib' / 'libtsne_h2pack.so',
            ]
        else:
            # Linux library naming
            lib_candidates = [
                # Installed package location (pip install)
                Path(__file__).parent / '_lib' / f'libtsne_h2pack.cpython-{sys.version_info.major}{sys.version_info.minor}-{platform.machine()}-linux-gnu.so',
                # Generic installed location
                Path(__file__).parent / '_lib' / 'libtsne_h2pack.so',
                # Development build location
                Path(__file__).parent.parent / 'build' / 'lib' / 'libtsne_h2pack.so',
            ]

        lib_path = None
        for candidate in lib_candidates:
            if candidate.exists():
                lib_path = candidate
                break

        if lib_path is None:
            raise RuntimeError(
                "Native library not found. "
                "Please build the package with: pip install -e ."
            )

        # Load library
        try:
            lib = ctypes.CDLL(str(lib_path))
        except OSError as e:
            raise RuntimeError(f"Failed to load native library from {lib_path}: {e}")

        # Configure function signatures
        self._configure_ctypes_signatures(lib)

        return lib

    def _configure_ctypes_signatures(self, lib):
        """Configure ctypes function signatures for C API"""

        # tsne_create_default_params
        # CRITICAL: On ARM64, returning 184-byte struct by value violates ABI
        # We must allocate the struct in Python and have C fill it
        # Change signature to void and manually create+initialize
        lib.tsne_create_default_params.argtypes = []
        lib.tsne_create_default_params.restype = TSNEParams  # Returns by value - works on x86_64 but may have issues on ARM64

        # tsne_init
        # CRITICAL: Pass params by pointer not by value on ARM64
        # The 184-byte struct violates ARM64 ABI when passed by value
        lib.tsne_init.argtypes = [
            np.ctypeslib.ndpointer(dtype=NP_DTYPE, flags='C_CONTIGUOUS'),  # X (C-contiguous)
            ctypes.c_int,  # n_point
            ctypes.c_int,  # pt_dim
            ctypes.c_int,  # low_dim
            ctypes.POINTER(TSNEParams),    # params - BY POINTER
        ]
        lib.tsne_init.restype = ctypes.POINTER(TSNEWorkspace)

        # tsne_init_with_P
        lib.tsne_init_with_P.argtypes = [
            _optional_ndpointer(dtype=NP_DTYPE, flags='C_CONTIGUOUS', ndim=2),  # X (can be None)
            ctypes.c_int,  # n_point
            ctypes.c_int,  # pt_dim
            ctypes.c_int,  # low_dim
            ctypes.POINTER(TSNEParams),    # params - BY POINTER
            ctypes.POINTER(SparseMatrix),  # P matrix - BY POINTER
        ]
        lib.tsne_init_with_P.restype = ctypes.POINTER(TSNEWorkspace)

        # tsne_run
        lib.tsne_run.argtypes = [ctypes.POINTER(TSNEWorkspace)]
        lib.tsne_run.restype = ctypes.c_int

        # tsne_get_embedding
        lib.tsne_get_embedding.argtypes = [
            ctypes.POINTER(TSNEWorkspace),
            np.ctypeslib.ndpointer(dtype=NP_DTYPE, flags='C_CONTIGUOUS'),  # Y_out (C-contiguous)
        ]
        lib.tsne_get_embedding.restype = None

        # tsne_destroy
        lib.tsne_destroy.argtypes = [ctypes.POINTER(TSNEWorkspace)]
        lib.tsne_destroy.restype = None

        # tsne_fit_transform (convenience function)
        # CRITICAL: Pass params by pointer not by value on ARM64
        lib.tsne_fit_transform.argtypes = [
            np.ctypeslib.ndpointer(dtype=NP_DTYPE, flags='C_CONTIGUOUS'),  # X (C-contiguous)
            ctypes.c_int,  # n_point
            ctypes.c_int,  # pt_dim
            ctypes.c_int,  # low_dim
            ctypes.POINTER(TSNEParams),    # params - BY POINTER
            np.ctypeslib.ndpointer(dtype=NP_DTYPE, flags='C_CONTIGUOUS'),  # Y_out (C-contiguous)
        ]
        lib.tsne_fit_transform.restype = ctypes.c_int

        # tsne_fit_transform_with_P (convenience function with pre-computed P)
        lib.tsne_fit_transform_with_P.argtypes = [
            np.ctypeslib.ndpointer(dtype=NP_DTYPE, flags='C_CONTIGUOUS', ndim=2),  # X (can be None)
            ctypes.c_int,  # n_point
            ctypes.c_int,  # pt_dim
            ctypes.c_int,  # low_dim
            ctypes.POINTER(TSNEParams),    # params - BY POINTER
            ctypes.POINTER(SparseMatrix),  # P matrix - BY POINTER
            np.ctypeslib.ndpointer(dtype=NP_DTYPE, flags='C_CONTIGUOUS'),  # Y_out (C-contiguous)
        ]
        lib.tsne_fit_transform_with_P.restype = ctypes.c_int

        # compute_P_from_knn_distances (compute P matrix from pre-computed k-NN distances)
        lib.compute_P_from_knn_distances.argtypes = [
            ctypes.POINTER(DTYPE),           # knn_distances (L2 distances, NOT squared)
            ctypes.POINTER(ctypes.c_int),    # knn_indices
            ctypes.POINTER(ctypes.c_int),    # knn_indptr
            ctypes.c_int,                    # n_samples
            ctypes.c_int,                    # nnz
            ctypes.c_double,                 # perplexity
            ctypes.c_int,                    # num_threads
        ]
        lib.compute_P_from_knn_distances.restype = ctypes.POINTER(SparseMatrix)

        # sparse_matrix_destroy (free a sparse matrix)
        lib.sparse_matrix_destroy.argtypes = [ctypes.POINTER(SparseMatrix)]
        lib.sparse_matrix_destroy.restype = None

        # sparse_matrix_create (allocate a sparse matrix)
        lib.sparse_matrix_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.sparse_matrix_create.restype = ctypes.POINTER(SparseMatrix)

        # tsne_set_init_profiling_times (set k-NN and P construction times)
        lib.tsne_set_init_profiling_times.argtypes = [
            ctypes.POINTER(TSNEWorkspace),  # workspace
            ctypes.c_double,                 # knn_time
            ctypes.c_double,                 # p_construction_time
        ]
        lib.tsne_set_init_profiling_times.restype = None

        # compute_kl_divergence (returns the most recent KL value from the
        # workspace; safe to call after tsne_run, before tsne_destroy)
        lib.compute_kl_divergence.argtypes = [ctypes.POINTER(TSNEWorkspace)]
        lib.compute_kl_divergence.restype = DTYPE

    def create_default_params(self) -> TSNEParams:
        """Create default parameters"""
        # Manually initialize to avoid ARM64 ABI issues with struct return by value
        params = TSNEParams()

        # Algorithm parameters (matching C defaults)
        params.n_iter = 1000
        params.perplexity = 30.0
        params.learning_rate = -1.0  # Adaptive
        params.momentum = 0.8
        params.initial_momentum = 0.5
        params.momentum_switch_iter = 250
        params.early_exag = 12.0
        params.early_exag_iter = 250

        # k-NN method
        params.use_vptree = 2  # Ball tree

        # H2Pack parameters
        params.h2_tol = 1e-3
        params.h2_rebuild_freq = 10
        params.h2pack_print_level = 0

        # Initialization
        params.init_method = 0
        params.random_seed = -1  # Will use time() in C

        # Verbosity
        params.verbose = 1
        params.print_freq = 50

        # Convergence detection
        params.enable_early_stopping = 0
        params.convergence_tol = 1e-5
        params.convergence_patience = 10
        params.min_iter = 250

        # Compatibility mode
        params.sklearn_compatible = 1

        # Profiling and parallelization
        params.enable_profiling = 0
        params.profiling_output_file = None
        params.num_threads = 1  # Controls all parallel operations

        # H2 tuning
        params.h2_leaf_size = 100
        params.h2_rebuild_freq_early = 10

        # Caller-supplied initial embedding (init_method = 1); NULL = random
        params.init_Y = None

        return params

    def fit_transform(self, X: np.ndarray, n_components: int,
                     params: TSNEParams) -> Tuple[np.ndarray, dict]:
        """
        Fit and transform X into low-dimensional embedding.

        Args:
            X: Input data (n_samples, n_features) in row-major format
            n_components: Number of embedding dimensions
            params: Algorithm parameters

        Returns:
            Tuple of (Y: Embedding (n_samples, n_components), metrics: dict)
        """
        n_samples, n_features = X.shape

        # Ensure contiguous array in correct dtype
        X = np.ascontiguousarray(X, dtype=NP_DTYPE)

        # C library stores X as X[feature_idx * n_points + point_idx]
        # This means all values of feature 0, then all values of feature 1, etc.
        # X.T in C-contiguous (row-major) gives exactly this layout
        X_col_major = np.ascontiguousarray(X.T, dtype=NP_DTYPE)

        # Allocate output in same format
        Y_col_major = np.zeros((n_components, n_samples),
                               dtype=NP_DTYPE, order='C')

        # Use native backend
        ret = self._lib.tsne_fit_transform(
            X_col_major, n_samples, n_features, n_components,
            params, Y_col_major
        )

        if ret != 0:
            raise RuntimeError(f"tsne_fit_transform failed with code {ret}")

        # KL divergence is not extracted via the convenience-function path
        # (the C function destroys the workspace internally). Callers that
        # need real KL should use fit_transform_with_knn_distances.
        metrics = {
            'kl_divergence': 0.0,
            'n_iter': params.n_iter,
        }

        # Convert back to row-major
        Y = np.ascontiguousarray(Y_col_major.T)

        return Y, metrics

    def fit_transform_with_P(self, X: np.ndarray, P_scipy, n_components: int,
                            params: TSNEParams) -> Tuple[np.ndarray, dict]:
        """
        Fit and transform X using pre-computed P matrix.

        Args:
            X: Input data (n_samples, n_features) in row-major format (can be None)
            P_scipy: Pre-computed sparse P matrix (scipy CSR format)
            n_components: Number of embedding dimensions
            params: Algorithm parameters

        Returns:
            Tuple of (Y: Embedding (n_samples, n_components), metrics: dict)
        """
        from scipy.sparse import csr_matrix

        if not isinstance(P_scipy, csr_matrix):
            raise TypeError(f"P must be scipy CSR matrix, got {type(P_scipy)}")

        n_samples = P_scipy.shape[0]

        # The optimization only needs P: the features are not passed down, which
        # saves two n_samples x n_features double copies (0.8 GB at 1M x 50).
        n_features = X.shape[1] if X is not None else 0
        X_col_major = None

        # Convert scipy CSR to ctypes structure
        sparse_mat, arrays = scipy_csr_to_sparse_matrix(P_scipy)

        # Allocate output
        Y_col_major = np.zeros((n_components, n_samples),
                               dtype=NP_DTYPE, order='C')

        # Use native backend with P matrix
        ret = self._lib.tsne_fit_transform_with_P(
            X_col_major, n_samples, n_features, n_components,
            ctypes.byref(params), ctypes.byref(sparse_mat), Y_col_major
        )

        if ret != 0:
            raise RuntimeError(f"tsne_fit_transform_with_P failed with code {ret}")

        # KL divergence is not extracted via the convenience-function path
        # (the C function destroys the workspace internally). Callers that
        # need real KL should use fit_transform_with_knn_distances.
        metrics = {
            'kl_divergence': 0.0,
            'n_iter': params.n_iter,
        }

        # Convert back to row-major
        Y = np.ascontiguousarray(Y_col_major.T)

        # Keep arrays alive until we're done
        del arrays

        return Y, metrics

    def fit_transform_with_knn_distances(self, X: np.ndarray, knn_distances_csr,
                                        n_components: int, params: TSNEParams,
                                        init_Y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        Fit and transform X using pre-computed k-NN distance CSR matrix.

        This accepts k-NN distances from ANY source (sklearn, FAISS, Annoy, etc.).

        Args:
            X: Input data (n_samples, n_features) in row-major format (can be None)
            knn_distances_csr: Sparse CSR matrix of k-NN distances (L2, not squared)
            n_components: Number of embedding dimensions
            params: Algorithm parameters
            init_Y: Optional initial embedding (n_samples, n_components); None = random
                initialization in C (params.random_seed)

        Returns:
            Tuple of (Y: Embedding (n_samples, n_components), metrics: dict)
        """
        from scipy.sparse import csr_matrix

        if not isinstance(knn_distances_csr, csr_matrix):
            raise TypeError(f"knn_distances must be scipy CSR matrix, got {type(knn_distances_csr)}")

        n_samples = knn_distances_csr.shape[0]

        if X is not None:
            n_features = X.shape[1]
            # Ensure contiguous array in correct dtype
            X = np.ascontiguousarray(X, dtype=NP_DTYPE)
            # Convert to column-major
            X_col_major = np.ascontiguousarray(X.T, dtype=NP_DTYPE)
        else:
            n_features = 0
            X_col_major = None

        # Neighbors in column order, whatever the k-NN method produced (sklearn orders
        # them by distance, a precomputed graph by index). The perplexity search sums
        # each row in storage order, and t-SNE amplifies a last-bit difference in P to
        # ~1e-5 within ten iterations, so the order has to be the same for every path.
        if not knn_distances_csr.has_sorted_indices:
            knn_distances_csr = knn_distances_csr.copy()
            knn_distances_csr.sort_indices()
        # A point must not be its own neighbor: a distance-0 self entry takes a large
        # share of the row's perplexity budget in the P matrix and visibly degrades the
        # embedding. Every k-NN method drops it, but guard the C side regardless.
        rows = np.repeat(np.arange(n_samples), np.diff(knn_distances_csr.indptr))
        if np.any(knn_distances_csr.indices == rows):
            keep = knn_distances_csr.indices != rows
            knn_distances_csr = csr_matrix(
                (knn_distances_csr.data[keep], knn_distances_csr.indices[keep],
                 np.concatenate([[0], np.cumsum(np.bincount(rows[keep], minlength=n_samples))])),
                shape=knn_distances_csr.shape)

        # Use native backend with k-NN distances
        # NOTE: C function expects L2 distances (NOT squared) and will square them internally
        knn_data = knn_distances_csr.data.astype(np.float64)  # Don't square!
        knn_indices = knn_distances_csr.indices.astype(np.int32)
        knn_indptr = knn_distances_csr.indptr.astype(np.int32)

        # Ensure arrays are C-contiguous
        knn_data = np.ascontiguousarray(knn_data)
        knn_indices = np.ascontiguousarray(knn_indices)
        knn_indptr = np.ascontiguousarray(knn_indptr)

        # Measure P construction time
        import time
        t_p_start = time.time()

        # Call C function to compute P from k-NN distances
        P_ptr = self._lib.compute_P_from_knn_distances(
            knn_data.ctypes.data_as(ctypes.POINTER(DTYPE)),
            knn_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            knn_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n_samples,
            knn_distances_csr.nnz,
            ctypes.c_double(params.perplexity),
            params.num_threads
        )

        if not P_ptr:
            raise RuntimeError("compute_P_from_knn_distances returned NULL")

        # Cast to SparseMatrix pointer and extract data
        P_sparse_mat = ctypes.cast(P_ptr, ctypes.POINTER(SparseMatrix))
        P_mat = P_sparse_mat.contents

        # Extract CSR arrays from C structure
        P_row_ptr = np.ctypeslib.as_array(P_mat.row_ptr, shape=(P_mat.n_rows + 1,)).copy()
        P_col_idx = np.ctypeslib.as_array(P_mat.col_idx, shape=(P_mat.nnz,)).copy()
        P_values = np.ctypeslib.as_array(P_mat.values, shape=(P_mat.nnz,)).copy()

        # Convert to scipy CSR
        P_cond = csr_matrix((P_values, P_col_idx, P_row_ptr), shape=(n_samples, n_samples))

        # Free the original C matrix (we have a copy now)
        self._lib.sparse_matrix_destroy(P_sparse_mat)

        # CRITICAL FIX: Symmetrize and normalize P matrix
        # P = (P + P.T) / sum(P + P.T)
        # This matches sklearn's _joint_probabilities_nn()
        P_symmetric = P_cond + P_cond.T
        P_sum = P_symmetric.sum()
        if P_sum > 0:
            P_symmetric.data /= P_sum

        # Convert back to C-allocated SparseMatrix (C can free it properly)
        P_c_ptr = scipy_csr_to_c_sparse_matrix(P_symmetric, self._lib)

        # Measure P construction time (includes symmetrization and normalization)
        p_construction_time = time.time() - t_p_start

        # Get k-NN time from CSR matrix attribute (set by k-NN method)
        knn_time = getattr(knn_distances_csr, 'knn_time', 0.0)

        # Allocate output
        Y_col_major = np.zeros((n_components, n_samples), dtype=NP_DTYPE, order='C')

        # Initial embedding: C reads it (low_dim x n_point, like Y_col_major) only
        # inside tsne_init_with_P, so this local reference keeps it alive long enough.
        if init_Y is not None:
            init_Y = np.asarray(init_Y)
            if init_Y.shape != (n_samples, n_components):
                raise ValueError(
                    f"init_Y must have shape {(n_samples, n_components)}, got {init_Y.shape}")
            init_Y_col_major = np.ascontiguousarray(init_Y.T, dtype=NP_DTYPE)
            params.init_method = 1
            params.init_Y = init_Y_col_major.ctypes.data_as(ctypes.POINTER(DTYPE))
        else:
            init_Y_col_major = None
            params.init_method = 0
            params.init_Y = None

        # Always go through the explicit-workspace path so we can read
        # the final KL divergence before tsne_destroy. The profiling
        # branch additionally records k-NN and P-construction timings.
        workspace = self._lib.tsne_init_with_P(
            X_col_major, n_samples, n_features, n_components,
            ctypes.byref(params), P_c_ptr
        )
        params.init_Y = None          # C no longer needs it
        del init_Y_col_major
        if not workspace:
            raise RuntimeError("tsne_init_with_P returned NULL")

        if params.enable_profiling:
            self._lib.tsne_set_init_profiling_times(
                workspace,
                ctypes.c_double(knn_time),
                ctypes.c_double(p_construction_time),
            )

        try:
            ret = self._lib.tsne_run(workspace)
            if ret != 0:
                raise RuntimeError(f"tsne_run failed with code {ret}")

            self._lib.tsne_get_embedding(workspace, Y_col_major)
            kl_value = float(self._lib.compute_kl_divergence(workspace))
        finally:
            self._lib.tsne_destroy(workspace)

        metrics = {
            'kl_divergence': kl_value,
            'n_iter': params.n_iter,
        }

        # Convert back to row-major
        Y = np.ascontiguousarray(Y_col_major.T)

        return Y, metrics

    @property
    def backend_type(self) -> str:
        """Get backend type (always 'native')"""
        return self._backend_type

    def is_available(self) -> bool:
        """Check if backend is available (always True for native backend)"""
        return True


def scipy_csr_to_sparse_matrix(P_scipy) -> Tuple[SparseMatrix, dict]:
    """
    Convert scipy CSR matrix to ctypes SparseMatrix structure.

    Args:
        P_scipy: Scipy CSR matrix

    Returns:
        Tuple of (SparseMatrix struct, dict of numpy arrays to keep alive)

    Note:
        The dict of arrays MUST be kept alive as long as the SparseMatrix is in use,
        because the SparseMatrix contains pointers to these arrays.
    """
    from scipy.sparse import csr_matrix

    if not isinstance(P_scipy, csr_matrix):
        raise TypeError(f"Expected scipy CSR matrix, got {type(P_scipy)}")

    # Convert to correct dtypes
    row_ptr = P_scipy.indptr.astype(np.int32)
    col_idx = P_scipy.indices.astype(np.int32)
    values = P_scipy.data.astype(np.float64)

    # Ensure C-contiguous
    if not row_ptr.flags['C_CONTIGUOUS']:
        row_ptr = np.ascontiguousarray(row_ptr)
    if not col_idx.flags['C_CONTIGUOUS']:
        col_idx = np.ascontiguousarray(col_idx)
    if not values.flags['C_CONTIGUOUS']:
        values = np.ascontiguousarray(values)

    # Create ctypes structure
    sparse_mat = SparseMatrix()
    sparse_mat.n_rows = P_scipy.shape[0]
    sparse_mat.n_cols = P_scipy.shape[1]
    sparse_mat.nnz = P_scipy.nnz

    # Get pointers to numpy arrays
    sparse_mat.row_ptr = row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    sparse_mat.col_idx = col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    sparse_mat.values = values.ctypes.data_as(ctypes.POINTER(DTYPE))

    # Return both the struct and the arrays to keep them alive
    arrays = {'row_ptr': row_ptr, 'col_idx': col_idx, 'values': values}

    return sparse_mat, arrays


def scipy_csr_to_c_sparse_matrix(P_scipy, lib) -> ctypes.POINTER(SparseMatrix):
    """
    Convert scipy CSR matrix to C-allocated SparseMatrix.

    Unlike scipy_csr_to_sparse_matrix, this allocates memory using C malloc
    so the C code can properly free it with sparse_matrix_destroy.

    Args:
        P_scipy: Scipy CSR matrix
        lib: ctypes library with sparse_matrix_create function

    Returns:
        ctypes pointer to C-allocated SparseMatrix
    """
    from scipy.sparse import csr_matrix

    if not isinstance(P_scipy, csr_matrix):
        raise TypeError(f"Expected scipy CSR matrix, got {type(P_scipy)}")

    n_rows = P_scipy.shape[0]
    n_cols = P_scipy.shape[1]
    nnz = P_scipy.nnz

    # Create C sparse matrix using C allocator
    P_ptr = lib.sparse_matrix_create(n_rows, n_cols, nnz)
    if not P_ptr:
        raise MemoryError("Failed to allocate sparse matrix in C")

    P_mat = P_ptr.contents

    # Copy data from scipy to C arrays
    row_ptr = P_scipy.indptr.astype(np.int32)
    col_idx = P_scipy.indices.astype(np.int32)
    values = P_scipy.data.astype(np.float64)

    # Copy using ctypes memmove
    ctypes.memmove(P_mat.row_ptr, row_ptr.ctypes.data, (n_rows + 1) * ctypes.sizeof(ctypes.c_int))
    ctypes.memmove(P_mat.col_idx, col_idx.ctypes.data, nnz * ctypes.sizeof(ctypes.c_int))
    ctypes.memmove(P_mat.values, values.ctypes.data, nnz * ctypes.sizeof(DTYPE))

    return P_ptr


# Singleton backend instance
_backend: Optional[TSNEBackend] = None


def get_backend() -> TSNEBackend:
    """Get singleton backend instance"""
    global _backend
    if _backend is None:
        _backend = TSNEBackend()
    return _backend
