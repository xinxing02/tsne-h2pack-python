#ifndef __TSNE_H2PACK_H__
#define __TSNE_H2PACK_H__

#include "H2Pack.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Use DTYPE from H2Pack (double or float)
#ifndef DTYPE
#define DTYPE double
#endif

// Sparse matrix structure (CSR format)
typedef struct {
    int n_rows;           // Number of rows
    int n_cols;           // Number of columns
    int nnz;              // Number of non-zeros
    int *row_ptr;         // Row pointers (size: n_rows + 1)
    int *col_idx;         // Column indices (size: nnz)
    DTYPE *values;        // Non-zero values (size: nnz)
} sparse_matrix_t;

// t-SNE parameters structure
typedef struct {
    // Algorithm parameters
    int n_iter;           // Number of iterations (default: 1000)
    DTYPE perplexity;     // Perplexity (default: 30.0)
    DTYPE learning_rate;  // Learning rate (default: 200.0)
    DTYPE momentum;       // Final momentum (default: 0.8)
    DTYPE initial_momentum; // Initial momentum (default: 0.5)
    int momentum_switch_iter; // When to switch momentum (default: 250)
    DTYPE early_exag;     // Early exaggeration factor (default: 12.0)
    int early_exag_iter;  // Early exaggeration iterations (default: 250)

    // k-NN method
    int use_vptree;       // Use VP-tree (2=ball tree, 1=vp-tree, 0=brute-force) (default: 2)

    // H2Pack parameters
    DTYPE h2_tol;         // H² approximation tolerance (default: 1e-3)
    int h2_rebuild_freq;  // H² rebuild frequency (default: 10)
    int h2pack_print_level;   // tsne-h2pack debug print level (0: silent, 1: statistics, 2: full debug, default: 0)
                              // Note: H2Pack's internal print_timers and print_dbginfo are always disabled

    // Initialization
    int init_method;      // 0: random N(0, 1e-4); 1: copy init_Y (PCA or a user array, computed by the caller)
    int random_seed;      // Random seed (default: time(NULL))

    // Verbosity
    int verbose;          // Print progress (0: silent, 1: basic, 2: detailed)
    int print_freq;       // Print frequency (default: 50)

    // Convergence detection
    int enable_early_stopping;  // Enable convergence detection (default: 0)
    DTYPE convergence_tol;      // Convergence tolerance (default: 1e-5)
    int convergence_patience;   // Number of iterations to check for convergence (default: 10)
    int min_iter;               // Minimum iterations before checking convergence (default: 250)

    // Compatibility mode
    int sklearn_compatible;     // Use pure sklearn behavior (default: 1, enabled)
                                // Set to 0 to enable extra stabilizations:
                                //   - Gradual exaggeration decay
                                //   - Learning rate decay
                                //   - Gradient clipping
                                //   - Z-based early stopping

    // Profiling and parallelization
    int enable_profiling;         // Enable fine-grained timing (default: 0)
    char *profiling_output_file;  // CSV output file (default: "tsne_profiling.csv")
    int num_threads;              // OpenMP threads for all parallel operations (default: 1, -1 for auto)
                                  // Controls k-NN search, P-matvec, and all other parallel operations

    // H2 tuning (appended: keep in sync with TSNEParams in _backend.py)
    int h2_leaf_size;             // Max points per H2 leaf box (default: 100). H2Pack's own 2D default of
                                  // 200 is sized for rank ~50-100 kernels; here the rank is ~10, so smaller
                                  // leaves cut the (dominant) near-field storage/work at no accuracy cost.
    int h2_rebuild_freq_early;    // Rebuild frequency during early exaggeration, after the first 20
                                  // iterations (default: 10). That phase is insensitive to a delayed
                                  // repulsive force; h2_rebuild_freq applies afterwards.
    const DTYPE *init_Y;          // init_method == 1: initial embedding (low_dim x n_point, the layout
                                  // of tsne_get_embedding), in the caller's point order. Read only by
                                  // tsne_init / tsne_init_with_P; it need not outlive them.
} tsne_params_t;

// Fine-grained profiling data (allocated only if profiling enabled)
typedef struct {
    // Per-iteration breakdown (seconds)
    double *knn_time;             // k-NN search (iteration 0 only)
    double *p_construction_time;  // P matrix construction (iteration 0 only)
    double *p_matvec_time;        // Sparse P-matvec per iteration
    double *h2_rebuild_time;      // H² matrix rebuild time
    double *h2_matvec_s_time;     // H² matvec for s vector
    double *h2_matvec_v_time;     // H² matvec for v vectors
    double *z_computation_time;   // Z normalization constant
    double *gradient_time;        // Gradient descent update
    double *other_time;           // Misc (centering, convergence check)
} profiling_data_t;

// t-SNE workspace structure
typedef struct {
    // Input data
    int n_point;          // Number of points
    int pt_dim;           // Input dimension
    int low_dim;          // Output dimension (2 or 3)
    DTYPE *X;             // Input data (pt_dim × n_point)

    // Sparse P matrix
    sparse_matrix_t *P;   // High-dimensional affinities

    // Embedding
    DTYPE *Y;             // Current embedding (low_dim × n_point)
    DTYPE *Y_prev;        // Previous embedding for momentum
    DTYPE *grad;          // Gradient (low_dim × n_point)
    DTYPE *gains;         // Adaptive gains (low_dim × n_point)
    DTYPE *uY;            // Velocity/update (low_dim × n_point)

    // H2Pack for repulsive forces
    H2Pack_p h2pack_W;    // H² matrix for (1 + r²)^{-2} kernel
    H2P_dense_mat_p *pp;  // Proxy points (owned by pp_cache)
    void  *pp_cache;      // Proxy point sets of earlier rebuilds, see tsne_pp_cache_get()

    // Temporary buffers
    DTYPE *s;             // Row sums (size: n_point)
    DTYPE *v;             // Matvec results (size: low_dim * n_point, flattened)
    DTYPE *ones;          // Vector of ones (size: n_point)
    DTYPE *Y_il;          // Interleaved (x0,y0,x1,y1,...) copy of Y for the 2D attractive pass
    float *Y_il32;        // float32 version of Y_il (gather source of the default attractive path)
    float *P_val32;       // P->values as float32 for the 2D attractive pass (NULL if DTYPE is float)
    int   P_val32_valid;  // 0: P has been permuted since P_val32 was filled
    void  *mm_ws;         // Scratch owned by tsne_h2_matmul (fused multi-vector H2 matvec)
    DTYPE *rep;           // Repulsive force from the last H2 rebuild (low_dim * n_point), for reuse
    DTYPE rep_Z;          // Z that goes with rep
    int   rep_valid;      // rep/rep_Z hold a usable force
    int   *point_order;   // Current point number -> caller's number (NULL: identity); see tsne_reorder.c
    sparse_matrix_t *P_scratch;  // P-sized scratch reused by every renumbering

    // Statistics
    DTYPE kl_divergence;  // Current KL divergence
    double *iter_times;   // Time per iteration

    // Convergence tracking
    DTYPE *Y_prev_for_convergence;  // Previous Y for convergence check
    DTYPE *delta_Y;                 // Change in Y per iteration
    int convergence_count;          // Consecutive iterations below threshold
    int converged;                  // Flag indicating convergence

    // Profiling
    profiling_data_t *profiling;  // NULL if profiling disabled

    // Temporary proxy cache
    char *temp_cache_file;  // Temporary cache file path (cleaned up on destroy)

    // Parameters (pointer to avoid heap corruption from large embedded struct)
    tsne_params_t *params;
} tsne_workspace_t;

// =============================================================================
// Public API
// =============================================================================

/**
 * Create default t-SNE parameters
 * @return Default parameters structure
 */
tsne_params_t tsne_create_default_params(void);

/**
 * Initialize t-SNE workspace
 * @param X         Input data (pt_dim × n_point, column-major)
 * @param n_point   Number of points
 * @param pt_dim    Input dimension
 * @param low_dim   Output dimension (2 or 3)
 * @param params    t-SNE parameters (passed by pointer for ARM64 ABI compliance)
 * @return          Initialized workspace (NULL on failure)
 */
tsne_workspace_t* tsne_init(
    const DTYPE *X,
    int n_point,
    int pt_dim,
    int low_dim,
    const tsne_params_t *params
);

/**
 * Initialize t-SNE workspace with pre-computed P matrix
 * @param X         Input data (pt_dim × n_point, column-major) - can be NULL if only P is used
 * @param n_point   Number of points
 * @param pt_dim    Input dimension
 * @param low_dim   Output dimension (2 or 3)
 * @param params    t-SNE parameters (passed by pointer for ARM64 ABI compliance)
 * @param P         Pre-computed sparse P matrix (ownership transferred to workspace)
 * @return          Initialized workspace (NULL on failure)
 */
tsne_workspace_t* tsne_init_with_P(
    const DTYPE *X,
    int n_point,
    int pt_dim,
    int low_dim,
    const tsne_params_t *params,
    sparse_matrix_t *P
);

/**
 * Run t-SNE optimization
 * @param workspace  t-SNE workspace
 * @return           0 on success, error code otherwise
 */
int tsne_run(tsne_workspace_t *workspace);

/**
 * Get embedding result
 * @param workspace  t-SNE workspace
 * @param Y_out      Output buffer (low_dim × n_point, column-major)
 */
void tsne_get_embedding(tsne_workspace_t *workspace, DTYPE *Y_out);

/**
 * Destroy t-SNE workspace and free memory
 * @param workspace  t-SNE workspace
 */
void tsne_destroy(tsne_workspace_t *workspace);

/**
 * Convenience function: run complete t-SNE
 * @param X         Input data (pt_dim × n_point, column-major)
 * @param n_point   Number of points
 * @param pt_dim    Input dimension
 * @param low_dim   Output dimension (2 or 3)
 * @param params    t-SNE parameters (passed by pointer for ARM64 ABI compliance)
 * @param Y_out     Output embedding (low_dim × n_point, column-major)
 * @return          0 on success, error code otherwise
 */
int tsne_fit_transform(
    const DTYPE *X,
    int n_point,
    int pt_dim,
    int low_dim,
    const tsne_params_t *params,
    DTYPE *Y_out
);

/**
 * Convenience function: run complete t-SNE with pre-computed P matrix
 * @param X         Input data (pt_dim × n_point, column-major) - can be NULL if only P is used
 * @param n_point   Number of points
 * @param pt_dim    Input dimension
 * @param low_dim   Output dimension (2 or 3)
 * @param params    t-SNE parameters (passed by pointer for ARM64 ABI compliance)
 * @param P         Pre-computed sparse P matrix (ownership transferred)
 * @param Y_out     Output embedding (low_dim × n_point, column-major)
 * @return          0 on success, error code otherwise
 */
int tsne_fit_transform_with_P(
    const DTYPE *X,
    int n_point,
    int pt_dim,
    int low_dim,
    const tsne_params_t *params,
    sparse_matrix_t *P,
    DTYPE *Y_out
);

// =============================================================================
// Sparse matrix utilities
// =============================================================================

/**
 * Create sparse matrix
 * @param n_rows  Number of rows
 * @param n_cols  Number of columns
 * @param nnz     Number of non-zeros
 * @return        Allocated sparse matrix (NULL on failure)
 */
sparse_matrix_t* sparse_matrix_create(int n_rows, int n_cols, int nnz);

/**
 * Destroy sparse matrix
 * @param mat  Sparse matrix to free
 */
void sparse_matrix_destroy(sparse_matrix_t *mat);

/**
 * Sparse matrix-vector multiplication: y = A * x
 * @param A  Sparse matrix
 * @param x  Input vector (size: A->n_cols)
 * @param y  Output vector (size: A->n_rows)
 */
void sparse_matvec(const sparse_matrix_t *A, const DTYPE *x, DTYPE *y);

// =============================================================================
// Internal functions (exposed for testing)
// =============================================================================

/**
 * Compute sparse P matrix using k-nearest neighbors
 * @param X          Input data (pt_dim × n_point)
 * @param n_point    Number of points
 * @param pt_dim     Input dimension
 * @param perplexity Perplexity parameter
 * @param use_vptree Use VP-tree (2=ball tree, 1=vp-tree, 0=brute-force) for k-NN
 * @param verbose    Verbosity level
 * @param num_threads Number of OpenMP threads to use (controls all parallel operations)
 * @return           Sparse P matrix (NULL on failure)
 */
sparse_matrix_t* compute_sparse_P(
    const DTYPE *X,
    int n_point,
    int pt_dim,
    DTYPE perplexity,
    int use_vptree,
    int verbose,
    int num_threads
);

/**
 * Compute conditional P matrix from pre-computed k-NN distances (from ANY k-NN source).
 *
 * This function computes CONDITIONAL probabilities P_j|i (asymmetric matrix).
 * Python wrapper should symmetrize using scipy: P = (P + P.T) / sum(P + P.T)
 * This matches sklearn's approach in _joint_probabilities_nn().
 *
 * @param knn_distances  CSR data array (L2 distances, NOT squared)
 * @param knn_indices    CSR column indices
 * @param knn_indptr     CSR row pointers (length n_samples + 1)
 * @param n_samples      Number of samples
 * @param nnz            Number of non-zeros in CSR matrix
 * @param perplexity     Target perplexity for binary search
 * @param num_threads    Number of OpenMP threads
 * @return               Conditional P matrix (asymmetric, CSR format, NULL on failure)
 *                       Python must symmetrize and normalize
 */
sparse_matrix_t* compute_P_from_knn_distances(
    const DTYPE *knn_distances,
    const int *knn_indices,
    const int *knn_indptr,
    int n_samples,
    int nnz,
    DTYPE perplexity,
    int num_threads
);

/**
 * Initialize embedding Y
 * @param Y          Output buffer (low_dim × n_point)
 * @param n_point    Number of points
 * @param low_dim    Output dimension
 * @param method     Initialization method (0: random N(0, 1e-4), 1: copy Y0)
 * @param Y0         Initial embedding for method 1 (low_dim × n_point), e.g. the PCA
 *                   initialization computed by the Python wrapper; NULL for random
 * @param seed       Random seed (method 0)
 */
void initialize_embedding(
    DTYPE *Y,
    int n_point,
    int low_dim,
    int method,
    const DTYPE *Y0,
    unsigned int seed
);

/**
 * Compute attractive forces (sparse)
 * @param P          Sparse P matrix
 * @param Y          Current embedding (low_dim × n_point)
 * @param low_dim    Output dimension
 * @param n_point    Number of points
 * @param early_exag Early exaggeration factor
 * @param grad       Gradient (low_dim × n_point, accumulated)
 * @param num_threads Number of OpenMP threads to use (controls all parallel operations)
 */
void compute_attractive_forces(
    const sparse_matrix_t *P,
    const DTYPE *Y,
    int low_dim,
    int n_point,
    DTYPE early_exag,
    DTYPE *grad,
    int num_threads
);

/**
 * Compute repulsive forces using H2Pack
 * @param h2pack_W   H² matrix for (1 + r²)^{-2} kernel
 * @param Y          Current embedding (low_dim × n_point)
 * @param low_dim    Output dimension
 * @param n_point    Number of points
 * @param workspace  Temporary buffers (s, v, ones)
 * @param grad       Gradient (low_dim × n_point, accumulated)
 * @param iter       Current iteration (for profiling)
 * @return           Normalization constant Z
 */
DTYPE compute_repulsive_forces_h2pack(
    H2Pack_p h2pack_W,
    const DTYPE *Y,
    int low_dim,
    int n_point,
    tsne_workspace_t *workspace,
    DTYPE *grad,
    int iter
);

/**
 * Update or rebuild H² matrix for current embedding
 * @param workspace  t-SNE workspace
 * @param force_rebuild  Force rebuild even if not at rebuild frequency
 * @return           0 on success, error code otherwise
 */
int update_h2_matrix(tsne_workspace_t *workspace, int force_rebuild);

/**
 * Compute KL divergence (for monitoring)
 * @param workspace  t-SNE workspace
 * @return           KL divergence value
 */
DTYPE compute_kl_divergence(tsne_workspace_t *workspace);

/**
 * Write profiling data to CSV file
 * @param workspace  t-SNE workspace
 * @param filename   Output CSV filename
 */
void tsne_write_profiling_data(tsne_workspace_t *workspace, const char *filename);

/**
 * Set initialization profiling times (k-NN and P construction)
 * Called from Python wrapper when these operations happen in Python
 * @param workspace           t-SNE workspace
 * @param knn_time            k-NN search time in seconds
 * @param p_construction_time P matrix construction time in seconds
 */
void tsne_set_init_profiling_times(
    tsne_workspace_t *workspace,
    double knn_time,
    double p_construction_time
);

// Fused multi-vector H2 matvec (src/tsne_h2_matmul.c): Y[c] = W * X[c] for nv columns
// in one traversal of the H2 representation. Returns -1 if unsupported.
int  tsne_h2_matmul(H2Pack_p h2, const int nv, const DTYPE *const *X, DTYPE *const *Y, void **ws);
void tsne_h2_matmul_free(void **ws);

// Renumber the points so that new point k is old point ord[k] (src/tsne_reorder.c).
// Permutes P and all per-point state; undone transparently by tsne_get_embedding.
int tsne_renumber_points(tsne_workspace_t *workspace, const int *ord, int num_threads);

#ifdef __cplusplus
}
#endif

#endif // __TSNE_H2PACK_H__
