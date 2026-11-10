#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <unistd.h>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tsne_h2pack.h"
#include "utils.h"

#define CACHE_MIN_WIDTH 1e-4
#define CACHE_MAX_WIDTH 1e3

// Experiment knob: integer from the environment (read once per call site use).
static int tsne_env_int(const char *name, int dflt)
{
    const char *v = getenv(name);
    return (v != NULL && *v != '\0') ? atoi(v) : dflt;
}

// Comparison function for qsort (used in gradient clipping)
static int compare_double(const void *a, const void *b) {
    double diff = *(const double*)a - *(const double*)b;
    if (diff < 0.0) return -1;
    if (diff > 0.0) return 1;
    return 0;
}

// Forward declarations of kernels
extern void tsne_repulsive_kernel_eval(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const void *krnl_param, DTYPE *mat, const int ldm
);

extern void tsne_repulsive_kernel_bimv(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const void *krnl_param,
    const DTYPE *x_in_0, const DTYPE *x_in_1,
    DTYPE *x_out_0, DTYPE *x_out_1
);

extern int tsne_repulsive_kernel_bimv_flop(int pt_dim, int n0, int n1);

// =============================================================================
// Create default parameters
// =============================================================================

tsne_params_t tsne_create_default_params(void)
{
    tsne_params_t params;

    // Algorithm parameters
    params.n_iter = 1000;
    params.perplexity = 30.0;
    params.learning_rate = -1.0;  // Negative = adaptive (like sklearn 'auto')
    params.initial_momentum = 0.5;
    params.momentum = 0.8;
    params.momentum_switch_iter = 250;
    params.early_exag = 12.0;
    params.early_exag_iter = 250;

    // k-NN method (ball tree enabled by default)
    params.use_vptree = 2;  // 2 = ball tree, 0 = brute force

    // H2Pack parameters
    params.h2_tol = 1e-3;
    params.h2_rebuild_freq = 10;
    params.h2pack_print_level = 0;    // Silent by default (0=silent, 1=statistics, 2=full debug)
                                       // Note: H2Pack's internal print_timers/print_dbginfo are always disabled

    // Initialization
    params.init_method = 0;  // Random
    params.init_Y = NULL;
    params.random_seed = (unsigned int)time(NULL);

    // Verbosity
    params.verbose = 1;
    params.print_freq = 50;

    // Convergence detection
    params.enable_early_stopping = 0;  // Disabled by default
    params.convergence_tol = 1e-5;
    params.convergence_patience = 10;
    params.min_iter = 250;

    // Compatibility mode
    params.sklearn_compatible = 1;  // ENABLED by default (sklearn-compatible behavior)

    // Profiling and parallelization
    params.enable_profiling = 0;           // Disabled by default
    params.profiling_output_file = NULL;   // Will use default if profiling enabled
    params.num_threads = 1;                // Serial by default (controls all parallel operations)
    params.h2_leaf_size = 100;
    params.h2_rebuild_freq_early = 10;

    return params;
}

// Experiment aid (TSNE_H2_TIMERS=1): accumulate H2Pack's internal per-phase
// timers across rebuilds, since each H2P_destroy discards them.
static double g_h2_timers[9];
static int    g_h2_max_rank = 0;        // largest U column count seen over all rebuilds
static double g_h2_rank_sum = 0.0, g_h2_rank_cnt = 0.0;
static void tsne_accum_h2_timers(H2Pack_p h2pack)
{
    if (h2pack == NULL) return;
    for (int i = 0; i < 9; i++) g_h2_timers[i] += h2pack->timers[i];
    if (h2pack->U != NULL) {
        for (int i = 0; i < h2pack->n_node; i++) {
            if (h2pack->U[i] == NULL || h2pack->U[i]->ncol <= 0) continue;   // root / unused nodes
            const int r = h2pack->U[i]->ncol;
            if (r > g_h2_max_rank) g_h2_max_rank = r;
            g_h2_rank_sum += r; g_h2_rank_cnt += 1.0;
        }
    }
}
static void tsne_print_h2_timers(void)
{
    static const char *names[9] = {"partition", "U_build", "B_build", "D_build",
        "mv_fwd", "mv_mid", "mv_bwd", "mv_dense", "mv_vop"};
    printf("H2TIMERS");
    for (int i = 0; i < 9; i++) { printf(" %s=%.2f", names[i], g_h2_timers[i]); g_h2_timers[i] = 0.0; }
    printf("\n");
    printf("H2RANK max=%d mean=%.1f\n", g_h2_max_rank,
           g_h2_rank_cnt > 0 ? g_h2_rank_sum / g_h2_rank_cnt : 0.0);
    g_h2_max_rank = 0; g_h2_rank_sum = 0.0; g_h2_rank_cnt = 0.0;
    fflush(stdout);
}

// =============================================================================
// Attractive forces (sparse)
// =============================================================================

// 2D fast path. The generic routine below is dominated by (a) two cache misses
// per neighbor (Y is stored one array per dimension), (b) a store/reload of
// grad[] on every nonzero because grad may alias Y, and (c) one scalar divide
// per nonzero. Here Y is gathered from an interleaved copy (one cache line per
// neighbor), the row's force is accumulated in registers, and the divide runs
// over blocks of gathered neighbors so the compiler can vectorize it.
//
// TSNE_ATTR_F32 selects the arithmetic (A/B switch):
//   0  everything in DTYPE (reference; blocks of 8)
//   1  differences formed in DTYPE, divide and force sums in float32: ~1.5e-7
//      relative error of the attractive force.
//   2  (default) the neighbors are also gathered from a float32 copy of Y: half the
//      bytes per gather, and no double subtract + double->float conversion per
//      nonzero, which cost more than the divide (kernel 64 -> 47 ms per pass at 500K
//      points on 1 thread). What this computes is the *exact* attractive force of the
//      embedding rounded to float32, fl(Y): every point is displaced by at most half
//      a float ulp (3.8e-6 for |y| < 128; Y is re-centered every iteration and spans
//      ~250 at 1M points), the same displacement in every pair it appears in, so
//      pair forces stay antisymmetric and nothing drifts. Measured force error 2.5e-5
//      relative -- the H2 tolerance on the repulsive side is 1e-2 -- and common-P KL /
//      kNN-preservation are unchanged. The error grows with |y|, so beyond
//      ATTR_F32_GATHER_MAX (displacement up to 2.4e-4) the pass falls back to 1.
#define ATTR_BLK 16
#define ATTR_F32_GATHER_MAX 4096.0
#define ATTR_FN  attr_2d_gather_dtype
#define ATTR_YT  DTYPE
#include "tsne_attr_2d.h"
#define ATTR_FN  attr_2d_gather_f32
#define ATTR_YT  float
#include "tsne_attr_2d.h"

// P's values as float32, aligned with P->col_idx (NULL: out of memory). With a double
// DTYPE this is a copy, refreshed after every renumbering (which permutes the double P,
// the source of truth for the renumbering itself and for the KL divergence); 4 bytes per
// nonzero, 0.6 GB at 1M points.
static const float *attr_p_values_f32(tsne_workspace_t *ws, const int nthr)
{
    const sparse_matrix_t *P = ws->P;
    if (sizeof(DTYPE) == sizeof(float)) return (const float*)(const void*)P->values;
    const int nnz = P->row_ptr[P->n_rows];
    if (ws->P_val32 == NULL) {
        ws->P_val32 = (float*)malloc(((size_t)nnz + 1) * sizeof(float));
        ws->P_val32_valid = 0;
        if (ws->P_val32 == NULL) return NULL;
    }
    if (!ws->P_val32_valid) {
        float *pv = ws->P_val32;
        const DTYPE *values = P->values;
        #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(static)
        for (int idx = 0; idx < nnz; idx++) pv[idx] = (float)values[idx];
        ws->P_val32_valid = 1;
    }
    return ws->P_val32;
}

// Returns 0 when the forces have been added to ws->grad, -1 when this path cannot
// run (out of memory, or TSNE_ATTR_GENERIC=1) and the caller must use the generic one.
static int compute_attractive_forces_2d(tsne_workspace_t *ws, DTYPE early_exag, int num_threads)
{
    if (tsne_env_int("TSNE_ATTR_GENERIC", 0)) return -1;
    const int n_point = ws->n_point;
    const int nthr = (num_threads > 1) ? num_threads : 1;
    const sparse_matrix_t *P = ws->P;
    const int *row_ptr = P->row_ptr, *col_idx = P->col_idx;
    const DTYPE *Y0 = ws->Y, *Y1 = ws->Y + n_point;
    DTYPE *grad = ws->grad;
    const DTYPE scale = 4.0 * early_exag;

    int level = tsne_env_int("TSNE_ATTR_F32", 2);
    const float *pv = (level >= 1) ? attr_p_values_f32(ws, nthr) : NULL;
    if (pv == NULL) level = 0;

    // Rows differ in length (and cores in speed), so the schedule is dynamic; the chunk
    // keeps >= 64 chunks per thread for balance and is otherwise as large as possible
    // (1024 measured 2.5% faster than 256 at 500K points).
    int chunk = n_point / (nthr * 64);
    if (chunk < 64)   chunk = 64;
    if (chunk > 1024) chunk = 1024;

    // With a float DTYPE the gather below is already single precision
    if (level >= 2 && sizeof(DTYPE) > sizeof(float)) {
        if (ws->Y_il32 == NULL) ws->Y_il32 = (float*)malloc((size_t)2 * n_point * sizeof(float));
        float *Y_il32 = ws->Y_il32;
        if (Y_il32 != NULL) {
            DTYPE y_max = 0.0;
            #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(static) reduction(max:y_max)
            for (int i = 0; i < n_point; i++) {
                const DTYPE x = Y0[i], y = Y1[i];
                Y_il32[2 * (size_t)i]     = (float)x;
                Y_il32[2 * (size_t)i + 1] = (float)y;
                const DTYPE a = (DABS(x) > DABS(y)) ? DABS(x) : DABS(y);
                if (a > y_max) y_max = a;
            }
            if (y_max < ATTR_F32_GATHER_MAX) {
                attr_2d_gather_f32(row_ptr, col_idx, pv, Y_il32, n_point, scale, grad, nthr, chunk);
                return 0;
            }
        }
    }

    if (ws->Y_il == NULL) ws->Y_il = (DTYPE*)malloc((size_t)2 * n_point * sizeof(DTYPE));
    DTYPE *Y_il = ws->Y_il;
    if (Y_il == NULL) return -1;
    #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(static)
    for (int i = 0; i < n_point; i++) {
        Y_il[2 * (size_t)i]     = Y0[i];
        Y_il[2 * (size_t)i + 1] = Y1[i];
    }

    if (level >= 1) {
        attr_2d_gather_dtype(row_ptr, col_idx, pv, Y_il, n_point, scale, grad, nthr, chunk);
        return 0;
    }

    const DTYPE *values = P->values;
    #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(dynamic, chunk)
    for (int i = 0; i < n_point; i++) {
        const DTYPE xi = Y_il[2 * (size_t)i], yi = Y_il[2 * (size_t)i + 1];
        DTYPE gx = 0.0, gy = 0.0;
        int idx = row_ptr[i];
        const int end = row_ptr[i + 1];
        for (; idx + 8 <= end; idx += 8) {
            DTYPE dx[8], dy[8];
            for (int k = 0; k < 8; k++) {
                const size_t j = (size_t)col_idx[idx + k];
                dx[k] = xi - Y_il[2 * j];
                dy[k] = yi - Y_il[2 * j + 1];
            }
            #pragma omp simd reduction(+:gx,gy)
            for (int k = 0; k < 8; k++) {
                const DTYPE q = values[idx + k] / (1.0 + dx[k] * dx[k] + dy[k] * dy[k]);
                gx += q * dx[k];
                gy += q * dy[k];
            }
        }
        for (; idx < end; idx++) {
            const size_t j = (size_t)col_idx[idx];
            const DTYPE ddx = xi - Y_il[2 * j], ddy = yi - Y_il[2 * j + 1];
            const DTYPE q = values[idx] / (1.0 + ddx * ddx + ddy * ddy);
            gx += q * ddx;
            gy += q * ddy;
        }
        grad[i]           += scale * gx;
        grad[n_point + i] += scale * gy;
    }
    return 0;
}

void compute_attractive_forces(
    const sparse_matrix_t *P,
    const DTYPE *Y,
    int low_dim,
    int n_point,
    DTYPE early_exag,
    DTYPE *grad,
    int num_threads
)
{
    // Gradient contribution from attractive forces:
    // grad[i] += 4 * early_exag * p_ij * (y_i - y_j) / (1 + ||y_i - y_j||^2)

    // Parallel execution with OpenMP (if enabled and num_threads > 1)
#ifdef _OPENMP
    if (num_threads > 1) {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 64)
        for (int i = 0; i < n_point; i++) {
            for (int idx = P->row_ptr[i]; idx < P->row_ptr[i + 1]; idx++) {
                int j = P->col_idx[idx];
                DTYPE pij = P->values[idx] * early_exag;

                // Compute ||y_i - y_j||^2
                DTYPE dist2 = 0.0;
                for (int d = 0; d < low_dim; d++) {
                    DTYPE diff = Y[d * n_point + i] - Y[d * n_point + j];
                    dist2 += diff * diff;
                }

                // Compute (1 + ||y_i - y_j||^2)^{-1}
                DTYPE inv_dist = 1.0 / (1.0 + dist2);

                // Add gradient contribution
                for (int d = 0; d < low_dim; d++) {
                    DTYPE diff = Y[d * n_point + i] - Y[d * n_point + j];
                    grad[d * n_point + i] += 4.0 * pij * diff * inv_dist;
                }
            }
        }
    } else
#endif
    {
        // Serial execution (fallback or when OpenMP not available)
        for (int i = 0; i < n_point; i++) {
            for (int idx = P->row_ptr[i]; idx < P->row_ptr[i + 1]; idx++) {
                int j = P->col_idx[idx];
                DTYPE pij = P->values[idx] * early_exag;

                // Compute ||y_i - y_j||^2
                DTYPE dist2 = 0.0;
                for (int d = 0; d < low_dim; d++) {
                    DTYPE diff = Y[d * n_point + i] - Y[d * n_point + j];
                    dist2 += diff * diff;
                }

                // Compute (1 + ||y_i - y_j||^2)^{-1}
                DTYPE inv_dist = 1.0 / (1.0 + dist2);

                // Add gradient contribution
                for (int d = 0; d < low_dim; d++) {
                    DTYPE diff = Y[d * n_point + i] - Y[d * n_point + j];
                    grad[d * n_point + i] += 4.0 * pij * diff * inv_dist;
                }
            }
        }
    }
}

// Reductions over all points (the mean of Y, the normalization Z) used to be serial
// scalar add chains, ~1 ms each per iteration at 500K points. A plain OpenMP reduction
// would make their rounding -- and with it the whole trajectory -- depend on the number
// of threads, so the points are cut into fixed blocks instead: the blocks are summed in
// parallel and the block sums are added up in order.
#define TSNE_SUM_BLK   4096
#define TSNE_SUM_NBLK  256
static DTYPE tsne_sum_blocked(const DTYPE *x, const int n, const int nthr)
{
    DTYPE total = 0.0;
    for (int s0 = 0; s0 < n; s0 += TSNE_SUM_BLK * TSNE_SUM_NBLK) {
        DTYPE blk_sum[TSNE_SUM_NBLK];
        const int n_s = (n - s0 < TSNE_SUM_BLK * TSNE_SUM_NBLK) ? (n - s0) : TSNE_SUM_BLK * TSNE_SUM_NBLK;
        const int n_blk = (n_s + TSNE_SUM_BLK - 1) / TSNE_SUM_BLK;
        #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(static)
        for (int b = 0; b < n_blk; b++) {
            const int i_s = s0 + b * TSNE_SUM_BLK;
            const int i_e = (i_s + TSNE_SUM_BLK < s0 + n_s) ? (i_s + TSNE_SUM_BLK) : (s0 + n_s);
            DTYPE acc = 0.0;
            for (int i = i_s; i < i_e; i++) acc += x[i];
            blk_sum[b] = acc;
        }
        for (int b = 0; b < n_blk; b++) total += blk_sum[b];
    }
    return total;
}

// =============================================================================
// Repulsive forces with H2Pack
// =============================================================================

DTYPE compute_repulsive_forces_h2pack(
    H2Pack_p h2pack_W,
    const DTYPE *Y,
    int low_dim,
    int n_point,
    tsne_workspace_t *workspace,
    DTYPE *grad,
    int iter
)
{
    double t_start, t_end;  // For profiling sub-timings

    // Check if H2Pack is disabled (NULL when skipped for small/shallow problems)
    // NOTE: h2_rebuild_freq may be 0 due to heap corruption workaround, treat as 10
    int h2_rebuild_freq = (workspace->params->h2_rebuild_freq > 0) ? workspace->params->h2_rebuild_freq : 10;
    int never_use_h2 = (h2_rebuild_freq > 100000) || (h2pack_W == NULL);

    // If we never use H2, just do direct computation
    if (never_use_h2) {
        // Allocate arrays for direct computation only
        DTYPE *s_direct = (DTYPE*)malloc(n_point * sizeof(DTYPE));
        DTYPE *v_direct = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));

        // Compute s_direct[i] = sum_{j≠i} W_ij with explicit diagonal skip
        memset(s_direct, 0, n_point * sizeof(DTYPE));
        for (int i = 0; i < n_point; i++) {
            for (int j = 0; j < n_point; j++) {
                if (i == j) continue;  // Skip diagonal
                DTYPE r2 = 0.0;
                for (int d = 0; d < low_dim; d++) {
                    DTYPE diff = Y[d * n_point + i] - Y[d * n_point + j];
                    r2 += diff * diff;
                }
                DTYPE denom = 1.0 + r2;
                DTYPE W_ij = 1.0 / (denom * denom);
                s_direct[i] += W_ij;
            }
        }

        // Compute v_direct[d * n_point + i] = sum_{j≠i} W_ij * y_j^(d) with explicit diagonal skip
        memset(v_direct, 0, low_dim * n_point * sizeof(DTYPE));
        for (int d = 0; d < low_dim; d++) {
            for (int i = 0; i < n_point; i++) {
                for (int j = 0; j < n_point; j++) {
                    if (i == j) continue;  // Skip diagonal
                    DTYPE r2 = 0.0;
                    for (int dd = 0; dd < low_dim; dd++) {
                        DTYPE diff = Y[dd * n_point + i] - Y[dd * n_point + j];
                        r2 += diff * diff;
                    }
                    DTYPE denom = 1.0 + r2;
                    DTYPE W_ij = 1.0 / (denom * denom);
                    v_direct[d * n_point + i] += W_ij * Y[d * n_point + j];
                }
            }
        }

        // Compute Z using reformulated formula
        // Z = sum_i s_i + 2*sum_i ||y_i||^2 * s_i - 2*sum_d sum_i y_i^(d) * v_i^(d)
        DTYPE Z_term1 = 0.0;
        for (int i = 0; i < n_point; i++)
            Z_term1 += s_direct[i];

        DTYPE Z_term2 = 0.0;
        for (int i = 0; i < n_point; i++) {
            DTYPE norm_y_i_sq = 0.0;
            for (int d = 0; d < low_dim; d++) {
                DTYPE y_id = Y[d * n_point + i];
                norm_y_i_sq += y_id * y_id;
            }
            Z_term2 += norm_y_i_sq * s_direct[i];
        }
        Z_term2 *= 2.0;

        DTYPE Z_term3 = 0.0;
        for (int d = 0; d < low_dim; d++) {
            for (int i = 0; i < n_point; i++) {
                Z_term3 += Y[d * n_point + i] * v_direct[d * n_point + i];
            }
        }
        Z_term3 *= 2.0;

        DTYPE Z_direct = Z_term1 + Z_term2 - Z_term3;

        // Use direct results
        DTYPE *s = workspace->s;
        DTYPE *v = workspace->v;
        memcpy(s, s_direct, n_point * sizeof(DTYPE));
        memcpy(v, v_direct, low_dim * n_point * sizeof(DTYPE));

        // Assemble repulsive force
        for (int d = 0; d < low_dim; d++) {
            for (int i = 0; i < n_point; i++) {
                grad[d * n_point + i] -= (4.0 / Z_direct) * (Y[d * n_point + i] * s[i] - v[d * n_point + i]);
            }
        }

        // Clean up
        free(s_direct);
        free(v_direct);

        return Z_direct;
    }

    // Normal path: use H2Pack (with or without comparison)

    // Write H2Pack results directly into the preallocated workspace buffers
    // (workspace->s, workspace->v) instead of malloc'ing fresh scratch and
    // memcpy'ing every iteration. This is the hottest function in the loop
    // (48-64% of optimization time), so removing the per-iteration alloc/free
    // churn and the 3*n_point memcpy is free savings.
    DTYPE *s_h2 = workspace->s;
    DTYPE *v_h2 = workspace->v;

    // ==================== H2PACK COMPUTATION ====================
    // H2P_matvec handles permutation internally via pmt_x/pmt_y
    // Input and output are in original point order

    // Safety check: ensure H2Pack matrix is valid
    if (h2pack_W == NULL) {
        fprintf(stderr, "ERROR: h2pack_W is NULL in compute_repulsive_forces_h2pack\n");
        return 1.0;
    }

    // Compute [s | v] = W * [ones | Y], then remove the diagonal (W_ii = 1).
    if (workspace->profiling) t_start = get_wtime_sec();

    // One fused traversal of the H2 matrix for all 1 + low_dim columns (see
    // tsne_h2_matmul.c); falls back to one H2P_matvec per column if unsupported.
    int fused = 0;
    if (low_dim <= 3 && !tsne_env_int("TSNE_NO_FUSED", 0)) {
        const DTYPE *x_cols[4];
        DTYPE *y_cols[4];
        x_cols[0] = workspace->ones;
        y_cols[0] = s_h2;
        for (int d = 0; d < low_dim; d++) {
            x_cols[1 + d] = &Y[d * n_point];
            y_cols[1 + d] = &v_h2[d * n_point];
        }
        fused = (tsne_h2_matmul(h2pack_W, 1 + low_dim, x_cols, y_cols, &workspace->mm_ws) == 0);
    }
    if (!fused) {
        H2P_matvec(h2pack_W, workspace->ones, s_h2);
        for (int d = 0; d < low_dim; d++)
            H2P_matvec(h2pack_W, &Y[d * n_point], &v_h2[d * n_point]);
    }

    // The element-wise passes of this routine are memory-bound; left serial they add up
    // to a visible share of an iteration once the matvec itself runs on many threads.
    const int nthr = (workspace->params->num_threads > 1) ? workspace->params->num_threads : 1;
    #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(static)
    for (int i = 0; i < n_point; i++) s_h2[i] -= 1.0;
    for (int d = 0; d < low_dim; d++) {
        const DTYPE *Y_d = &Y[(size_t)d * n_point];
        DTYPE *v_d = &v_h2[(size_t)d * n_point];
        #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(static)
        for (int i = 0; i < n_point; i++) v_d[i] -= Y_d[i];
    }

    if (workspace->profiling) {
        t_end = get_wtime_sec();
        workspace->profiling->h2_matvec_s_time[iter] = t_end - t_start;
        workspace->profiling->h2_matvec_v_time[iter] = 0.0;
    }

    // ==================== COMPUTE Z VALUE ====================
    // REFORMULATED FORMULA: Z = sum_i s_i + 2*sum_i ||y_i||^2 * s_i - 2*sum_d sum_i y_i^(d) * v_i^(d)
    // See tsne_h2pack_integration.md lines 196-229 for derivation

    if (workspace->profiling) t_start = get_wtime_sec();

    // The three sums of the formula above in one pass, block-wise (see tsne_sum_blocked)
    DTYPE Z_term1 = 0.0, Z_term2 = 0.0, Z_term3 = 0.0;
    for (int s0 = 0; s0 < n_point; s0 += TSNE_SUM_BLK * TSNE_SUM_NBLK) {
        DTYPE blk1[TSNE_SUM_NBLK], blk2[TSNE_SUM_NBLK], blk3[TSNE_SUM_NBLK];
        const int n_s = (n_point - s0 < TSNE_SUM_BLK * TSNE_SUM_NBLK) ? (n_point - s0) : TSNE_SUM_BLK * TSNE_SUM_NBLK;
        const int n_blk = (n_s + TSNE_SUM_BLK - 1) / TSNE_SUM_BLK;
        #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(static)
        for (int b = 0; b < n_blk; b++) {
            const int i_s = s0 + b * TSNE_SUM_BLK;
            const int i_e = (i_s + TSNE_SUM_BLK < s0 + n_s) ? (i_s + TSNE_SUM_BLK) : (s0 + n_s);
            DTYPE a1 = 0.0, a2 = 0.0, a3 = 0.0;
            for (int i = i_s; i < i_e; i++) {
                DTYPE norm_y_i_sq = 0.0, y_dot_v = 0.0;
                for (int d = 0; d < low_dim; d++) {
                    const DTYPE y_id = Y[d * n_point + i];
                    norm_y_i_sq += y_id * y_id;
                    y_dot_v += y_id * v_h2[d * n_point + i];
                }
                a1 += s_h2[i];
                a2 += norm_y_i_sq * s_h2[i];
                a3 += y_dot_v;
            }
            blk1[b] = a1; blk2[b] = a2; blk3[b] = a3;
        }
        for (int b = 0; b < n_blk; b++) { Z_term1 += blk1[b]; Z_term2 += blk2[b]; Z_term3 += blk3[b]; }
    }
    Z_term2 *= 2.0;
    Z_term3 *= 2.0;

    // Final Z value
    DTYPE Z = Z_term1 + Z_term2 - Z_term3;

    if (workspace->profiling) {
        t_end = get_wtime_sec();
        workspace->profiling->z_computation_time[iter] = t_end - t_start;
    }

    // ==================== RESULTS ALREADY IN WORKSPACE ====================
    // s_h2/v_h2 alias workspace->s / workspace->v, so the matvec results are
    // already where the gradient assembly expects them — no copy needed.
    DTYPE *s = workspace->s;
    DTYPE *v = workspace->v;

    // Safeguard against numerical instability
    if (Z <= 0.0 || !isfinite(Z)) {
        fprintf(stderr, "ERROR: Z = %.6e (should be > 0) at iter %d\n", Z, iter);
        fprintf(stderr, "  Z_term1 (sum s_i) = %.6e\n", Z_term1);
        fprintf(stderr, "  Z_term2 (2*sum ||y||²*s) = %.6e\n", Z_term2);
        fprintf(stderr, "  Z_term3 (2*sum y*v) = %.6e\n", Z_term3);
        fflush(stderr);
        return 1.0;
    }

    // 4. Assemble repulsive force using H2Pack results
    const DTYPE inv_Z4 = 4.0 / Z;
    DTYPE *rep = workspace->rep;   // non-NULL when the force is kept for reuse
    for (int d = 0; d < low_dim; d++) {
        #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(static)
        for (int i = 0; i < n_point; i++) {
            const DTYPE f = inv_Z4 * (Y[d * n_point + i] * s[i] - v[d * n_point + i]);
            grad[d * n_point + i] -= f;
            if (rep != NULL) rep[d * n_point + i] = f;
        }
    }
    if (rep != NULL) {
        workspace->rep_Z = Z;
        workspace->rep_valid = 1;
    }

    return Z;
}

// This file is compiled with -ffast-math, under which isnan() / isinf() and comparisons
// against NaN may be folded away: test the exponent bits instead.
static inline int tsne_is_finite(const DTYPE x)
{
#if DTYPE_SIZE == 8
    uint64_t u; memcpy(&u, &x, sizeof(u));
    return (u & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
#else
    uint32_t u; memcpy(&u, &x, sizeof(u));
    return (u & 0x7f800000U) != 0x7f800000U;
#endif
}

// Root enclosing box of the embedding: same box as H2P_calc_enclosing_box() up to the
// rounding of the center. That routine makes two serial, non-vectorized passes over Y on
// every rebuild, after a third serial pass here that looked for NaN / Inf. Here the center
// is one (vectorized) pass whose result does not depend on the number of threads, and the
// radius -- a max, exact in any order -- is computed in parallel and checks the
// coordinates on the way.
static DTYPE *tsne_calc_root_enbox(const int pt_dim, const int n_point, const DTYPE *coord,
                                   const char *pp_fname, int *all_finite)
{
    DTYPE *enbox = (DTYPE*) malloc(sizeof(DTYPE) * 2 * pt_dim);
    if (enbox == NULL) return NULL;
    DTYPE *center = enbox;   // enbox[0 : pt_dim-1] holds the center until the end
    for (int j = 0; j < pt_dim; j++) {
        const DTYPE *coord_dim_j = coord + (size_t)j * n_point;
        DTYPE s = 0.0;
        for (int i = 0; i < n_point; i++) s += coord_dim_j[i];
        center[j] = s;
    }
    for (int j = 0; j < pt_dim; j++) center[j] /= (DTYPE) n_point;

    DTYPE radius = 0.0;
    int n_bad = 0;
    for (int j = 0; j < pt_dim; j++) {
        const DTYPE *coord_dim_j = coord + (size_t)j * n_point;
        const DTYPE center_j = center[j];
        #pragma omp parallel for schedule(static) reduction(max:radius) reduction(+:n_bad) if(n_point >= 50000)
        for (int i = 0; i < n_point; i++) {
            DTYPE tmp = DABS(coord_dim_j[i] - center_j);
            n_bad += !tsne_is_finite(coord_dim_j[i]);
            radius = MAX(radius, tmp);
        }
    }
    *all_finite = (n_bad == 0);

    // Align the radius with the box sizes of the proxy point file (minL * 2^k)
    FILE *inf = (pp_fname != NULL) ? fopen(pp_fname, "r") : NULL;
    if (inf != NULL) {
        int pt_dim_, L3_nlayer, num_pp;
        DTYPE reltol, minL;
        const char *fmt_str = (DTYPE_SIZE == 8) ? "%d %lf %d %lf %d" : "%d %f %d %f %d";
        if (fscanf(inf, fmt_str, &pt_dim_, &reltol, &L3_nlayer, &minL, &num_pp) == 5 && pt_dim_ == pt_dim) {
            DTYPE k = DCEIL(DLOG2(radius / minL));
            radius = minL * DPOW(2.0, k);
        }
        fclose(inf);
    }

    for (int j = 0; j < pt_dim; j++) {
        DTYPE center_j = center[j];
        enbox[j] = center_j - radius;
        enbox[pt_dim + j] = 2 * radius;
    }
    return enbox;
}

// Proxy point sets by (root box size, tree depth). H2P_generate_proxy_point_ID_file()
// parses the whole proxy point text file and writes it back on every call, although the
// root box size is quantized to powers of two and the depth barely moves, so a run only
// ever asks for a handful of distinct sets. H2Pack does not take ownership of a set (it
// is only read during H2P_build), so the cache owns them and hands the same set out again.
// A set enters the cache the second time H2Pack returns it for a key: by then every level
// has been read back from the file, whereas a level generated by the first call still has
// the digits that the file's %.12f drops. Cached results are thus exactly the uncached ones.
#define PP_CACHE_SLOTS 8
typedef struct {
    DTYPE box_size;
    int   max_level, n_seen;
    H2P_dense_mat_p *pp;    // max_level + 1 matrices
} tsne_pp_entry_t;
typedef struct {
    tsne_pp_entry_t e[PP_CACHE_SLOTS];
    int next_evict;
} tsne_pp_cache_t;

static void tsne_pp_free(H2P_dense_mat_p *pp, const int max_level)
{
    if (pp == NULL) return;
    for (int i = 0; i <= max_level; i++) H2P_dense_mat_destroy(&pp[i]);
    free(pp);
}

static void tsne_pp_cache_free(void **cache_)
{
    tsne_pp_cache_t *cache = (tsne_pp_cache_t*) *cache_;
    if (cache == NULL) return;
    for (int i = 0; i < PP_CACHE_SLOTS; i++) tsne_pp_free(cache->e[i].pp, cache->e[i].max_level);
    free(cache);
    *cache_ = NULL;
}

// Proxy points for the freshly partitioned h2pack, from the cache or from H2Pack.
// Must not be called while an H2 matrix built from an earlier set is still being built.
static H2P_dense_mat_p *tsne_pp_cache_get(tsne_workspace_t *workspace, H2Pack_p h2pack, const char *pp_fname)
{
    if (workspace->pp_cache == NULL) workspace->pp_cache = calloc(1, sizeof(tsne_pp_cache_t));
    tsne_pp_cache_t *cache = (tsne_pp_cache_t*) workspace->pp_cache;
    const DTYPE box_size = h2pack->root_enbox[h2pack->pt_dim];
    const int max_level = h2pack->max_level;

    tsne_pp_entry_t *entry = NULL;
    for (int i = 0; cache != NULL && i < PP_CACHE_SLOTS; i++) {
        tsne_pp_entry_t *e = &cache->e[i];
        if (e->pp != NULL && e->box_size == box_size && e->max_level == max_level) entry = e;
    }
    if (entry != NULL && entry->n_seen >= 2) return entry->pp;

    H2P_dense_mat_p *pp = NULL;
    H2P_generate_proxy_point_ID_file(h2pack, &workspace->low_dim, tsne_repulsive_kernel_eval, pp_fname, &pp);
    if (pp == NULL || cache == NULL) return pp;   // (without a cache the set is leaked, as before)

    if (entry == NULL) {
        entry = &cache->e[cache->next_evict];
        cache->next_evict = (cache->next_evict + 1) % PP_CACHE_SLOTS;
        entry->n_seen = 0;
    }
    tsne_pp_free(entry->pp, entry->max_level);
    entry->pp = pp;
    entry->box_size = box_size;
    entry->max_level = max_level;
    entry->n_seen++;
    return pp;
}

// =============================================================================
// Update H² matrix
// =============================================================================

int update_h2_matrix(tsne_workspace_t *workspace, int force_rebuild)
{

    int low_dim = workspace->low_dim;
    int n_point = workspace->n_point;
    DTYPE *Y = workspace->Y;

    // CRITICAL FIX: Skip all H2Pack operations for small datasets
    // Small datasets (N < 2000) use direct O(N²) computation to avoid heap corruption
    if (n_point < 2000) {
        // Ensure h2pack is NULL (should already be NULL from tsne_init)
        if (workspace->h2pack_W != NULL) {
            H2P_destroy(&workspace->h2pack_W);
            workspace->h2pack_W = NULL;
            workspace->pp = NULL;
        }
        return 0;  // Success - using direct computation
    }

    // NOTE: h2_tol may be 0 due to heap corruption workaround, use 1e-6 as fallback
    DTYPE h2_tol = (workspace->params->h2_tol > 0.0) ? workspace->params->h2_tol : 1e-6;
    char *pp_fname = workspace->temp_cache_file;  // Use temporary cache file

    // Note: Temporary cache file will be automatically cleaned up when tsne_destroy() is called

    // Destroy old H² matrix if exists (this also cleans up proxy points internally)
    if (workspace->h2pack_W != NULL) {
        tsne_accum_h2_timers(workspace->h2pack_W);
        H2P_destroy(&workspace->h2pack_W);
        workspace->h2pack_W = NULL;
    }

    // Reset proxy points pointer (H2P_destroy may have freed it)
    workspace->pp = NULL;

    // CRITICAL: Set OpenMP thread count BEFORE H2P_init()
    // H2P_init() calls omp_get_max_threads() to initialize h2pack->n_thread
    // Without this, H2Pack always uses system default (typically 1 or max hardware threads)
    omp_set_num_threads(workspace->params->num_threads);

    // The H2 matrix is torn down and rebuilt hundreds of times at nearly the same
    // size: let H2Pack recycle its large B/D blocks instead of page-faulting fresh
    // ones in on every build.
    malloc_aligned_pool_enable(1);

    // Initialize H2Pack from scratch
    H2P_init(&workspace->h2pack_W, low_dim, 1, QR_REL_NRM, &h2_tol);
    if (workspace->h2pack_W == NULL) {
        fprintf(stderr, "ERROR: H2P_init failed - workspace->h2pack_W is NULL\n");
        return -1;
    }

    // Always silence H2Pack's internal debug output
    // The h2pack_print_level flag only controls tsne-h2pack's own debug output
    workspace->h2pack_W->print_timers = 0;
    workspace->h2pack_W->print_dbginfo = 0;

    // Calculate enclosing box for current Y positions (pp_fname: the box size is
    // aligned with the cached proxy points); also rejects NaN / Inf coordinates
    int y_finite = 1;
    workspace->h2pack_W->root_enbox = tsne_calc_root_enbox(low_dim, n_point, Y, pp_fname, &y_finite);
    if (!y_finite) {
        fprintf(stderr, "ERROR: Y contains NaN or Inf!\n");
        H2P_destroy(&workspace->h2pack_W);
        return -1;
    }
    if (workspace->h2pack_W->root_enbox == NULL) {
        fprintf(stderr, "ERROR: failed to allocate the root enclosing box\n");
        H2P_destroy(&workspace->h2pack_W);
        return -1;
    }

    // DEBUG: Print enclosing box information (if enabled)
    if (workspace->params->h2pack_print_level >= 2) {
        DTYPE box_radius = workspace->h2pack_W->root_enbox[low_dim];
        printf("\n========== H2 Matrix Rebuild Debug ==========\n");
        printf("  Enclosing box radius: %.6e\n", box_radius);
        printf("  Enclosing box diameter: %.6e\n", 2.0 * box_radius);

        // Calculate and print ||Y|| for reference
        DTYPE Y_norm = 0.0;
        for (int i = 0; i < n_point * low_dim; i++) {
            Y_norm += Y[i] * Y[i];
        }
        Y_norm = sqrt(Y_norm);
        printf("  ||Y|| (Frobenius norm): %.6e\n", Y_norm);
    }

    // Partition points based on current Y positions
    // 0 lets H2Pack pick its default (200 points per leaf in 2D)
    const int leaf_size = (workspace->params->h2_leaf_size > 0) ? workspace->params->h2_leaf_size : 0;
    H2P_partition_points(workspace->h2pack_W, n_point, Y, leaf_size, 0.0);
    if (workspace->h2pack_W->n_leaf_node <= 0) {
        fprintf(stderr, "ERROR: H2P_partition_points failed - no leaf nodes created\n");
        H2P_destroy(&workspace->h2pack_W);
        return -1;
    }

    // FIX: Use direct O(N²) computation for small problems or shallow trees
    // Conditions for skipping H2Pack:
    // 1. n_point < 2000: H2Pack overhead not worth it for small datasets
    // 2. max_level <= 2: Too few levels means no admissible pairs, causes heap corruption in proxy point generation
    int skip_h2 = (n_point < 2000) || (workspace->h2pack_W->max_level <= 2);

    if (skip_h2) {
        if (workspace->params->verbose || workspace->params->h2pack_print_level >= 1) {
            printf("  Skipping H2Pack approximation:\n");
            if (n_point < 2000) {
                printf("    - Small dataset (N=%d < 2000)\n", n_point);
            }
            if (workspace->h2pack_W->max_level <= 2) {
                printf("    - Shallow tree (max_level=%d <= 2, no admissible pairs)\n",
                       workspace->h2pack_W->max_level);
            }
            printf("    - Using direct O(N²) computation instead\n");
        }
        H2P_destroy(&workspace->h2pack_W);
        workspace->h2pack_W = NULL;
        workspace->pp = NULL;
        return 0;  // Success, but H2Pack disabled
    }

    // Proxy points for this root box size and tree depth: from an earlier rebuild, or
    // loaded by H2Pack from the cache file (which avoids the expensive generation)
    workspace->pp = tsne_pp_cache_get(workspace, workspace->h2pack_W, pp_fname);
    if (workspace->pp == NULL) {
        fprintf(stderr, "ERROR: H2P_generate_proxy_point_ID_file failed - pp is NULL\n");
        H2P_destroy(&workspace->h2pack_W);
        return -1;
    }

    // Build H² matrix. With the repulsive force reused until the next rebuild, a matrix
    // serves exactly one (fused) product, so its B & D blocks are not stored: writing them
    // out and streaming them back once costs more than evaluating them inside the product
    // (tsne_h2_matmul.c), and they are nearly all of the matrix' memory. They are stored
    // (AOT) when the matrix is multiplied every iteration or by H2P_matvec.
    const int bd_jit = tsne_env_int("TSNE_H2_JIT", 1) && tsne_env_int("TSNE_REP_REUSE", 1)
                       && low_dim <= 3 && !tsne_env_int("TSNE_NO_FUSED", 0);
    int krnl_bimv_flop = tsne_repulsive_kernel_bimv_flop(low_dim, n_point, n_point);
    H2P_build(
        workspace->h2pack_W,
        workspace->pp,
        bd_jit,
        &workspace->low_dim,  // FIX: Use persistent workspace member, not local variable
        tsne_repulsive_kernel_eval,
        tsne_repulsive_kernel_bimv,
        krnl_bimv_flop
    );
    if (workspace->h2pack_W->n_UJ <= 0) {
        fprintf(stderr, "ERROR: H2P_build failed - no U/J blocks created\n");
        H2P_destroy(&workspace->h2pack_W);
        return -1;
    }

    // DEBUG: Print tree structure and rank summary (if enabled)
    if (workspace->params->h2pack_print_level >= 2) {
        printf("  Tree max_level: %d, n_nodes: %d\n",
               workspace->h2pack_W->max_level, workspace->h2pack_W->n_node);
        int max_rank = 0;
        double sum_rank = 0.0;
        int non_empty = 0;
        for (int i = 0; i < workspace->h2pack_W->n_UJ; i++) {
            int rank_i = workspace->h2pack_W->U[i]->ncol;
            if (rank_i > 0) {
                sum_rank += (double)rank_i;
                non_empty++;
                if (rank_i > max_rank) max_rank = rank_i;
            }
        }
        printf("  Rank summary: max=%d, avg=%.1f, non_empty_nodes=%d\n",
               max_rank, sum_rank / non_empty, non_empty);
        printf("=============================================\n\n");
    }

    // Print full H2Pack statistics including proxy points (if enabled)
    if (workspace->params->h2pack_print_level >= 1) {
        H2P_print_statistic(workspace->h2pack_W);
        printf("\n");
    }

    return 0;
}

// =============================================================================
// Trim proxy point cache to practical range
// =============================================================================

// Helper function to generate proxy cache for specified range
// Relative tolerance of the proxy point selection: the run's h2_tol (the ID never
// resolves the kernel better than that), clamped to [1e-6, 1e-2]. It used to be a
// fixed 1e-4 from when h2_tol defaulted to 1e-4; the ID of every node on every rebuild
// pays for the proxy count, and at 1e-4 a 3D box carried 312 proxy points against 146
// at 1e-2 (U-build 96 -> 39 s at 250K / 8 threads, quality unchanged). TSNE_PROXY_TOL
// overrides it for experiments.
static DTYPE tsne_proxy_tol(DTYPE h2_tol)
{
    const char *s = getenv("TSNE_PROXY_TOL");
    if (s != NULL && s[0] != '\0') {
        const double v = atof(s);
        if (v > 0.0 && v < 1.0) return (DTYPE)v;
    }
    if (!(h2_tol > 0.0)) h2_tol = 1e-4;
    if (h2_tol > 1e-2) h2_tol = 1e-2;
    if (h2_tol < 1e-6) h2_tol = 1e-6;
    return h2_tol;
}

static int generate_proxy_cache_range(const char *cache_file, int pt_dim, DTYPE proxy_tol,
                                       double min_width, double max_width, int verbose)
{
    // Skip cache generation if no cache file specified
    if (cache_file == NULL) {
        if (verbose) {
            printf("Skipping proxy cache generation (no cache file specified)\n");
        }
        return 0;
    }

    if (verbose) {
        printf("Generating proxy cache: [%.2e, %.2e] for %dD\n",
               min_width, max_width, pt_dim);
    }

    // Calculate range parameters
    // Note: H2Pack stores levels from finest to coarsest in the cache file
    // We need to ensure the generated range covers [min_width, max_width]
    DTYPE minL = min_width / 2.0;  // H2Pack uses half-width
    DTYPE maxL = max_width / 2.0;

    // Calculate number of levels needed, with extra margin for safety
    int num_levels = (int)ceil(log2(maxL / minL)) + 2;  // +2 for safety margin

    if (verbose) {
        printf("  Calculating %d levels from half-width %.2e to %.2e\n",
               num_levels, minL, maxL);
    }

    // Create dummy H2Pack structure
    H2Pack_p h2pack;
    DTYPE h2_tol = proxy_tol;
    H2P_init(&h2pack, pt_dim, 1, QR_REL_NRM, &h2_tol);
    if (h2pack == NULL) {
        fprintf(stderr, "Error: H2P_init failed in cache generation\n");
        return -1;
    }

    // Disable printing for cache generation (internal operation)
    h2pack->print_timers = 0;
    h2pack->print_dbginfo = 0;

    // Set up enclosing box
    h2pack->root_enbox = (DTYPE*)malloc((2 * pt_dim) * sizeof(DTYPE));
    for (int d = 0; d < pt_dim; d++) {
        h2pack->root_enbox[d] = -maxL;
        h2pack->root_enbox[pt_dim + d] = 2.0 * maxL;
    }
    h2pack->max_level = num_levels - 1;

    // Generate proxy points
    H2P_dense_mat_p *pp = NULL;
    int pt_dim_kernel = pt_dim;
    H2P_generate_proxy_point_ID_file(
        h2pack, &pt_dim_kernel, tsne_repulsive_kernel_eval,
        cache_file, &pp
    );

    if (pp == NULL) {
        fprintf(stderr, "Error: Failed to generate proxy points\n");
        H2P_destroy(&h2pack);
        return -1;
    }

    if (verbose) {
        int total_pp = 0;
        for (int i = 0; i < h2pack->max_level + 1; i++) {
            if (pp[i] != NULL) total_pp += pp[i]->ncol;
        }
        printf("  Generated %d proxy points across %d levels\n", total_pp, num_levels);
    }

    // Cleanup
    for (int i = 0; i < h2pack->max_level + 1; i++) {
        if (pp[i]) H2P_dense_mat_destroy(&pp[i]);
    }
    free(pp);
    H2P_destroy(&h2pack);

    return 0;
}

// =============================================================================
// Proxy point cache
// =============================================================================

static int tsne_copy_file(const char *src, const char *dst)
{
    FILE *in = fopen(src, "rb");
    if (in == NULL) return -1;
    FILE *out = fopen(dst, "wb");
    if (out == NULL) { fclose(in); return -1; }
    char buf[65536];
    size_t got;
    int rc = 0;
    while ((got = fread(buf, 1, sizeof(buf), in)) > 0)
        if (fwrite(buf, 1, got, out) != got) { rc = -1; break; }
    fclose(in);
    if (fclose(out) != 0) rc = -1;
    return rc;
}

// Returns the (malloc'd) path of this run's private proxy-point file, or NULL.
//
// Proxy points depend on the kernel, the dimension, the precision and the range of
// box widths -- never on the data -- yet selecting them takes several seconds (7 s on
// one thread), which at 50K points rivals the whole optimization. They are therefore
// kept in a per-user cache: $TSNE_H2PACK_CACHE_DIR, else $XDG_CACHE_HOME/tsne_h2pack,
// else ~/.cache/tsne_h2pack. Each run still works on a private COPY in $TMPDIR,
// because H2Pack rewrites the file it is given on every call; the shared file is only
// ever replaced by an atomic rename, so concurrent processes cannot see a torn one.
// TSNE_H2PACK_NO_PROXY_CACHE=1 disables the shared file. Bump the version in the name
// whenever CACHE_MIN_WIDTH, CACHE_MAX_WIDTH, the kernel or the proxy tolerance change.
static char *tsne_acquire_proxy_cache(int low_dim, DTYPE run_h2_tol, int verbose)
{
    const DTYPE proxy_tol = tsne_proxy_tol(run_h2_tol);
    static int n_acquired = 0;
    int serial;
    #pragma omp atomic capture
    serial = n_acquired++;

    const char *tmpdir = getenv("TMPDIR");   // POSIX; fall back to /tmp when unset
    if (tmpdir == NULL || tmpdir[0] == '\0') tmpdir = "/tmp";
    char temp_path[1024];
    snprintf(temp_path, sizeof(temp_path), "%s/tsne_proxy_cache_%d_%d.txt", tmpdir, (int)getpid(), serial);

    char shared[1024] = "";
    if (!tsne_env_int("TSNE_H2PACK_NO_PROXY_CACHE", 0)) {
        char dir[900] = "";
        const char *env_dir = getenv("TSNE_H2PACK_CACHE_DIR"), *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
        if (env_dir != NULL && env_dir[0] != '\0') {
            snprintf(dir, sizeof(dir), "%s", env_dir);
        } else if (xdg != NULL && xdg[0] != '\0') {
            snprintf(dir, sizeof(dir), "%s/tsne_h2pack", xdg);
        } else if (home != NULL && home[0] != '\0') {
            snprintf(dir, sizeof(dir), "%s/.cache", home);
            mkdir(dir, 0700);
            snprintf(dir, sizeof(dir), "%s/.cache/tsne_h2pack", home);
        }
        if (dir[0] != '\0') {
            mkdir(dir, 0700);   // fine if it exists; if it cannot be created the opens below fail
            // v2: the file name carries the tolerance (v1 files were always 1e-4)
            snprintf(shared, sizeof(shared), "%s/proxy_v2_%dd_f%d_tol%.0e.txt", dir, low_dim,
                     (int)(8 * sizeof(DTYPE)), (double)proxy_tol);
        }
    }

    if (shared[0] != '\0' && tsne_copy_file(shared, temp_path) == 0) {
        if (verbose) printf("Proxy points: loaded from %s\n", shared);
        return strdup(temp_path);
    }

    if (verbose) {
        printf("Generating proxy points: [%.0e, %.0e] box widths for %dD embedding, tolerance %.0e\n",
               CACHE_MIN_WIDTH, CACHE_MAX_WIDTH, low_dim, (double)proxy_tol);
    }
    if (generate_proxy_cache_range(temp_path, low_dim, proxy_tol, CACHE_MIN_WIDTH, CACHE_MAX_WIDTH, verbose) != 0) {
        unlink(temp_path);
        return NULL;
    }
    if (shared[0] != '\0') {
        char staging[1100];
        snprintf(staging, sizeof(staging), "%s.%d.tmp", shared, (int)getpid());
        if (tsne_copy_file(temp_path, staging) != 0 || rename(staging, shared) != 0) unlink(staging);
    }
    return strdup(temp_path);
}

// =============================================================================
// Initialize workspace
// =============================================================================

tsne_workspace_t* tsne_init(
    const DTYPE *X,
    int n_point,
    int pt_dim,
    int low_dim,
    const tsne_params_t *params
)
{
    if (low_dim != 2 && low_dim != 3) {
        printf("Error: low_dim must be 2 or 3 (got %d)\n", low_dim);
        return NULL;
    }

    // For small datasets, skip H2Pack entirely and use direct O(N²) computation
    int use_h2pack = (n_point >= 2000);

    // Declare temp_cache_file variable
    char *temp_cache_file = NULL;

    // Skip proxy cache generation for small datasets (will use direct computation)
    // Small datasets (N < 2000) will be detected later in update_h2_matrix and use O(N²) direct computation
    // This avoids H2Pack heap corruption bugs with small datasets
    if (use_h2pack) {
        temp_cache_file = tsne_acquire_proxy_cache(low_dim, params->h2_tol, params->verbose);
        if (temp_cache_file == NULL) {
            fprintf(stderr, "Error: Failed to generate temporary proxy cache\n");
            return NULL;
        }
    } else {
        if (params->verbose) {
            printf("Skipping proxy cache generation for small dataset (N=%d < 2000)\n", n_point);
            printf("  Will use direct O(N²) computation instead of H2Pack\n");
        }
        // NOTE: No heap state fix needed with pointer-based params!
    }

    // Adaptive learning rate
    DTYPE learning_rate = params->learning_rate;
    if (learning_rate < 0.0) {
        learning_rate = fmax(200.0, n_point / 12.0);
        if (params->verbose) {
            printf("Using adaptive learning rate: %.1f (N=%d)\n",
                   learning_rate, n_point);
        }
    }

    // Allocate workspace
    tsne_workspace_t *workspace = (tsne_workspace_t*)calloc(1, sizeof(tsne_workspace_t));
    if (workspace == NULL) {
        return NULL;
    }

    // Allocate params separately to avoid heap corruption
    // (Embedding 184-byte struct at offset 168-351 in 352-byte allocation caused corruption)
    workspace->params = (tsne_params_t*)malloc(sizeof(tsne_params_t));
    if (workspace->params == NULL) {
        free(workspace);
        return NULL;
    }

    // Copy ALL 30 parameter fields - no more heap corruption!
    memcpy(workspace->params, params, sizeof(tsne_params_t));

    // Override learning_rate with adaptive value if needed
    if (learning_rate != params->learning_rate) {
        workspace->params->learning_rate = learning_rate;
    }

    // Store temporary cache file path for cleanup on destroy (may be NULL for small datasets)
    workspace->temp_cache_file = temp_cache_file;  // May be NULL, which is fine

    // Store dimensions
    workspace->n_point = n_point;
    workspace->pt_dim = pt_dim;
    workspace->low_dim = low_dim;

    // Copy input data
    workspace->X = (DTYPE*)malloc(pt_dim * n_point * sizeof(DTYPE));
    memcpy(workspace->X, X, pt_dim * n_point * sizeof(DTYPE));

    // Allocate embedding and gradients
    workspace->Y = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));
    workspace->Y_prev = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));
    workspace->grad = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));
    workspace->gains = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));
    workspace->uY = (DTYPE*)calloc(low_dim * n_point, sizeof(DTYPE));  // Initialize to 0

    // Initialize gains to 1.0
    for (int i = 0; i < low_dim * n_point; i++) {
        workspace->gains[i] = 1.0;
    }

    // Allocate temporary buffers
    workspace->s = (DTYPE*)malloc(n_point * sizeof(DTYPE));
    workspace->ones = (DTYPE*)malloc(n_point * sizeof(DTYPE));
    workspace->v = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));

    // Initialize ones vector
    for (int i = 0; i < n_point; i++)
        workspace->ones[i] = 1.0;

    // Allocate timing array
    workspace->iter_times = (double*)malloc(params->n_iter * sizeof(double));

    // Allocate convergence tracking arrays
    if (params->enable_early_stopping) {
        workspace->Y_prev_for_convergence = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));
        workspace->delta_Y = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));
        workspace->convergence_count = 0;
        workspace->converged = 0;
    } else {
        workspace->Y_prev_for_convergence = NULL;
        workspace->delta_Y = NULL;
        workspace->convergence_count = 0;
        workspace->converged = 0;
    }

    // Allocate profiling data if enabled
    if (params->enable_profiling) {
        workspace->profiling = (profiling_data_t*)calloc(1, sizeof(profiling_data_t));
        workspace->profiling->knn_time = (double*)calloc(1, sizeof(double));
        workspace->profiling->p_construction_time = (double*)calloc(1, sizeof(double));
        workspace->profiling->p_matvec_time = (double*)calloc(params->n_iter, sizeof(double));
        workspace->profiling->h2_rebuild_time = (double*)calloc(params->n_iter, sizeof(double));
        workspace->profiling->h2_matvec_s_time = (double*)calloc(params->n_iter, sizeof(double));
        workspace->profiling->h2_matvec_v_time = (double*)calloc(params->n_iter, sizeof(double));
        workspace->profiling->z_computation_time = (double*)calloc(params->n_iter, sizeof(double));
        workspace->profiling->gradient_time = (double*)calloc(params->n_iter, sizeof(double));
        workspace->profiling->other_time = (double*)calloc(params->n_iter, sizeof(double));
    } else {
        workspace->profiling = NULL;
    }

    if (params->verbose) {
        printf("Initializing t-SNE (n=%d, pt_dim=%d, low_dim=%d)\n",
               n_point, pt_dim, low_dim);
        if (params->enable_early_stopping) {
            printf("  Early stopping enabled (tol=%.2e, patience=%d, min_iter=%d)\n",
                   params->convergence_tol, params->convergence_patience, params->min_iter);
        }
    }

    // Compute sparse P matrix
    double t_p_start = 0.0;
    if (workspace->profiling) t_p_start = get_wtime_sec();

    workspace->P = compute_sparse_P(X, n_point, pt_dim, params->perplexity,
                                     params->use_vptree,
                                     params->verbose, params->num_threads);
    if (workspace->P == NULL) {
        printf("Error: Failed to compute sparse P matrix\n");
        tsne_destroy(workspace);
        return NULL;
    }

    if (workspace->profiling) {
        workspace->profiling->p_construction_time[0] = get_wtime_sec() - t_p_start;
        // Note: k-NN time is included in p_construction_time
        // For separate k-NN timing, would need to modify compute_sparse_P
        workspace->profiling->knn_time[0] = workspace->profiling->p_construction_time[0];
    }

    // Initialize embedding (REMOVED init_Y_file support)
    initialize_embedding(workspace->Y, n_point, low_dim, params->init_method,
                         params->init_Y, params->random_seed);
    workspace->params->init_Y = NULL;   // the caller's buffer need not outlive tsne_init*
    memcpy(workspace->Y_prev, workspace->Y, low_dim * n_point * sizeof(DTYPE));

    // Build initial H² matrix only for large datasets
    if (use_h2pack) {
        if (params->verbose) printf("Building initial H² matrix for comparison...\n");
        if (update_h2_matrix(workspace, 1) != 0) {
            printf("Error: Failed to build initial H² matrix\n");
            tsne_destroy(workspace);
            return NULL;
        }
    } else {
        // For small datasets, explicitly set h2pack_W to NULL to signal direct computation
        workspace->h2pack_W = NULL;
        workspace->pp = NULL;
        if (params->verbose) {
            printf("Skipping H² matrix build for small dataset (N=%d < 2000)\n", n_point);
            printf("  Using direct O(N²) computation for repulsive forces\n");
        }
    }

    if (params->verbose) {
        if (workspace->h2pack_W != NULL) {
            printf("Initialization complete (H² matrix built successfully)\n");
            if (params->h2pack_print_level >= 1) {
                H2P_print_statistic(workspace->h2pack_W);
            }
        } else {
            printf("Initialization complete (using direct O(N²) computation)\n");
        }
    }

    return workspace;
}

// =============================================================================
// tsne_init_with_P: Initialize workspace with pre-computed P matrix
// =============================================================================

tsne_workspace_t* tsne_init_with_P(
    const DTYPE *X,
    int n_point,
    int pt_dim,
    int low_dim,
    const tsne_params_t *params,
    sparse_matrix_t *P
)
{
    if (low_dim != 2 && low_dim != 3) {
        printf("Error: low_dim must be 2 or 3 (got %d)\n", low_dim);
        return NULL;
    }

    // Validate P matrix dimensions
    if (P == NULL || P->n_rows != n_point || P->n_cols != n_point) {
        printf("Error: Invalid P matrix (expected %dx%d, got %dx%d)\n",
               n_point, n_point, P ? P->n_rows : 0, P ? P->n_cols : 0);
        return NULL;
    }

    // For small datasets, skip H2Pack entirely and use direct O(N²) computation
    int use_h2pack = (n_point >= 2000);

    // Declare temp_cache_file variable
    char *temp_cache_file = NULL;

    // Skip proxy cache generation for small datasets
    if (use_h2pack) {
        temp_cache_file = tsne_acquire_proxy_cache(low_dim, params->h2_tol, params->verbose);
        if (temp_cache_file == NULL) {
            fprintf(stderr, "Error: Failed to generate temporary proxy cache\n");
            return NULL;
        }
    } else {
        if (params->verbose) {
            printf("Skipping proxy cache generation for small dataset (N=%d < 2000)\n", n_point);
            printf("  Will use direct O(N²) computation instead of H2Pack\n");
        }
    }

    // Adaptive learning rate
    DTYPE learning_rate = params->learning_rate;
    if (learning_rate < 0.0) {
        learning_rate = fmax(200.0, n_point / 12.0);
        if (params->verbose) {
            printf("Using adaptive learning rate: %.1f (N=%d)\n",
                   learning_rate, n_point);
        }
    }

    // Allocate workspace
    tsne_workspace_t *workspace = (tsne_workspace_t*)calloc(1, sizeof(tsne_workspace_t));
    if (workspace == NULL) {
        return NULL;
    }

    // Allocate params separately
    workspace->params = (tsne_params_t*)malloc(sizeof(tsne_params_t));
    if (workspace->params == NULL) {
        free(workspace);
        return NULL;
    }

    // Copy parameters
    memcpy(workspace->params, params, sizeof(tsne_params_t));

    // Override learning_rate with adaptive value if needed
    if (learning_rate != params->learning_rate) {
        workspace->params->learning_rate = learning_rate;
    }

    // Store temporary cache file path
    workspace->temp_cache_file = temp_cache_file;

    // Store dimensions
    workspace->n_point = n_point;
    workspace->pt_dim = pt_dim;
    workspace->low_dim = low_dim;

    // Copy input data (if provided - can be NULL if only P is used)
    if (X != NULL) {
        workspace->X = (DTYPE*)malloc(pt_dim * n_point * sizeof(DTYPE));
        memcpy(workspace->X, X, pt_dim * n_point * sizeof(DTYPE));
    } else {
        workspace->X = NULL;
    }

    // Allocate embedding and gradients
    workspace->Y = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));
    workspace->Y_prev = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));
    workspace->grad = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));
    workspace->gains = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));
    workspace->uY = (DTYPE*)calloc(low_dim * n_point, sizeof(DTYPE));

    // Initialize gains to 1.0
    for (int i = 0; i < low_dim * n_point; i++) {
        workspace->gains[i] = 1.0;
    }

    // Allocate temporary buffers
    workspace->s = (DTYPE*)malloc(n_point * sizeof(DTYPE));
    workspace->ones = (DTYPE*)malloc(n_point * sizeof(DTYPE));
    workspace->v = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));

    // Initialize ones vector
    for (int i = 0; i < n_point; i++)
        workspace->ones[i] = 1.0;

    // Allocate timing array
    workspace->iter_times = (double*)malloc(params->n_iter * sizeof(double));

    // Allocate convergence tracking arrays
    if (params->enable_early_stopping) {
        workspace->Y_prev_for_convergence = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));
        workspace->delta_Y = (DTYPE*)malloc(low_dim * n_point * sizeof(DTYPE));
        workspace->convergence_count = 0;
        workspace->converged = 0;
    } else {
        workspace->Y_prev_for_convergence = NULL;
        workspace->delta_Y = NULL;
        workspace->convergence_count = 0;
        workspace->converged = 0;
    }

    // Allocate profiling data if enabled
    if (params->enable_profiling) {
        workspace->profiling = (profiling_data_t*)calloc(1, sizeof(profiling_data_t));
        workspace->profiling->knn_time = (double*)calloc(1, sizeof(double));
        workspace->profiling->p_construction_time = (double*)calloc(1, sizeof(double));
        workspace->profiling->p_matvec_time = (double*)calloc(params->n_iter, sizeof(double));
        workspace->profiling->h2_rebuild_time = (double*)calloc(params->n_iter, sizeof(double));
        workspace->profiling->h2_matvec_s_time = (double*)calloc(params->n_iter, sizeof(double));
        workspace->profiling->h2_matvec_v_time = (double*)calloc(params->n_iter, sizeof(double));
        workspace->profiling->z_computation_time = (double*)calloc(params->n_iter, sizeof(double));
        workspace->profiling->gradient_time = (double*)calloc(params->n_iter, sizeof(double));
        workspace->profiling->other_time = (double*)calloc(params->n_iter, sizeof(double));
    } else {
        workspace->profiling = NULL;
    }

    if (params->verbose) {
        printf("Initializing t-SNE with pre-computed P matrix (n=%d, pt_dim=%d, low_dim=%d)\n",
               n_point, pt_dim, low_dim);
        printf("  P matrix: %dx%d sparse with %d non-zeros (%.2f%% sparse)\n",
               P->n_rows, P->n_cols, P->nnz, 100.0 * (1.0 - (double)P->nnz / (n_point * n_point)));
        if (params->enable_early_stopping) {
            printf("  Early stopping enabled (tol=%.2e, patience=%d, min_iter=%d)\n",
                   params->convergence_tol, params->convergence_patience, params->min_iter);
        }
    }

    // Use pre-computed P matrix (transfer ownership)
    workspace->P = P;

    // Initialize embedding (REMOVED init_Y_file support)
    initialize_embedding(workspace->Y, n_point, low_dim, params->init_method,
                         params->init_Y, params->random_seed);
    workspace->params->init_Y = NULL;   // the caller's buffer need not outlive tsne_init*
    memcpy(workspace->Y_prev, workspace->Y, low_dim * n_point * sizeof(DTYPE));

    // Build initial H² matrix only for large datasets
    if (use_h2pack) {
        if (params->verbose) printf("Building initial H² matrix for comparison...\n");
        if (update_h2_matrix(workspace, 1) != 0) {
            printf("Error: Failed to build initial H² matrix\n");
            tsne_destroy(workspace);
            return NULL;
        }
    } else {
        // For small datasets, explicitly set h2pack_W to NULL
        workspace->h2pack_W = NULL;
        workspace->pp = NULL;
        if (params->verbose) {
            printf("Skipping H² matrix build for small dataset (N=%d < 2000)\n", n_point);
            printf("  Using direct O(N²) computation for repulsive forces\n");
        }
    }

    if (params->verbose) {
        if (workspace->h2pack_W != NULL) {
            printf("Initialization complete (H² matrix built successfully)\n");
            if (params->h2pack_print_level >= 1) {
                H2P_print_statistic(workspace->h2pack_W);
            }
        } else {
            printf("Initialization complete (using direct O(N²) computation)\n");
        }
    }

    return workspace;
}

// =============================================================================
// Profiling utilities
// =============================================================================

void tsne_write_profiling_data(tsne_workspace_t *workspace, const char *filename)
{
    if (!workspace->profiling) return;

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot write profiling data to %s\n", filename);
        return;
    }

    // CSV header
    fprintf(fp, "iteration,p_matvec,h2_rebuild,h2_matvec_s,h2_matvec_v,");
    fprintf(fp, "z_computation,gradient,other,total\n");

    // Per-iteration data
    int n_iter = workspace->params->n_iter;
    for (int i = 0; i < n_iter; i++) {
        double total = workspace->profiling->p_matvec_time[i]
                     + workspace->profiling->h2_rebuild_time[i]
                     + workspace->profiling->h2_matvec_s_time[i]
                     + workspace->profiling->h2_matvec_v_time[i]
                     + workspace->profiling->z_computation_time[i]
                     + workspace->profiling->gradient_time[i]
                     + workspace->profiling->other_time[i];

        fprintf(fp, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                i, workspace->profiling->p_matvec_time[i],
                workspace->profiling->h2_rebuild_time[i],
                workspace->profiling->h2_matvec_s_time[i],
                workspace->profiling->h2_matvec_v_time[i],
                workspace->profiling->z_computation_time[i],
                workspace->profiling->gradient_time[i],
                workspace->profiling->other_time[i],
                total);
    }

    // Summary statistics
    fprintf(fp, "\n# Summary Statistics\n");
    fprintf(fp, "# k-NN time: %.3f seconds\n", workspace->profiling->knn_time[0]);
    fprintf(fp, "# P construction: %.3f seconds\n", workspace->profiling->p_construction_time[0]);

    // Calculate totals
    double total_p_matvec = 0.0, total_h2_rebuild = 0.0;
    double total_h2_matvec = 0.0, total_gradient = 0.0;
    double total_time = 0.0;
    for (int i = 0; i < n_iter; i++) {
        total_p_matvec += workspace->profiling->p_matvec_time[i];
        total_h2_rebuild += workspace->profiling->h2_rebuild_time[i];
        total_h2_matvec += workspace->profiling->h2_matvec_s_time[i]
                        + workspace->profiling->h2_matvec_v_time[i];
        total_gradient += workspace->profiling->gradient_time[i];
        total_time += workspace->profiling->p_matvec_time[i]
                   + workspace->profiling->h2_rebuild_time[i]
                   + workspace->profiling->h2_matvec_s_time[i]
                   + workspace->profiling->h2_matvec_v_time[i]
                   + workspace->profiling->z_computation_time[i]
                   + workspace->profiling->gradient_time[i]
                   + workspace->profiling->other_time[i];
    }

    fprintf(fp, "# Total P-matvec: %.3f seconds (%.1f%%)\n",
            total_p_matvec, 100.0 * total_p_matvec / total_time);
    fprintf(fp, "# Total H²-rebuild: %.3f seconds (%.1f%%)\n",
            total_h2_rebuild, 100.0 * total_h2_rebuild / total_time);
    fprintf(fp, "# Total H²-matvec: %.3f seconds (%.1f%%)\n",
            total_h2_matvec, 100.0 * total_h2_matvec / total_time);
    fprintf(fp, "# Total gradient: %.3f seconds (%.1f%%)\n",
            total_gradient, 100.0 * total_gradient / total_time);
    fprintf(fp, "# Total optimization time: %.3f seconds\n", total_time);

    fclose(fp);

    if (workspace->params->verbose) {
        printf("\nProfiling data written to: %s\n", filename);
    }
}

// =============================================================================
// Set initialization profiling times (called from Python wrapper)
// =============================================================================

void tsne_set_init_profiling_times(
    tsne_workspace_t *workspace,
    double knn_time,
    double p_construction_time
)
{
    if (!workspace || !workspace->profiling) {
        return;
    }

    workspace->profiling->knn_time[0] = knn_time;
    workspace->profiling->p_construction_time[0] = p_construction_time;
}

// =============================================================================
// Main optimization loop
// =============================================================================

int tsne_run(tsne_workspace_t *workspace)
{
    int n_point = workspace->n_point;
    int low_dim = workspace->low_dim;
    int n_iter = workspace->params->n_iter;
    DTYPE learning_rate = workspace->params->learning_rate;
    DTYPE initial_momentum = workspace->params->initial_momentum;
    DTYPE final_momentum = workspace->params->momentum;
    int momentum_switch_iter = workspace->params->momentum_switch_iter;
    int early_exag_iter = workspace->params->early_exag_iter;
    DTYPE early_exag = workspace->params->early_exag;
    int verbose = workspace->params->verbose;
    int print_freq = workspace->params->print_freq;

    // Use actual sklearn_compatible value (pointer-based params fix resolved heap corruption)
    int sklearn_compatible = workspace->params->sklearn_compatible;

    // Start with initial momentum
    DTYPE momentum = initial_momentum;

    if (verbose) {
        printf("\nStarting t-SNE optimization (%d iterations)\n", n_iter);
        printf("  Learning rate: %.1f\n", learning_rate);
        printf("  Momentum: %.2f -> %.2f (switch at iter %d)\n",
               initial_momentum, final_momentum, momentum_switch_iter);
        printf("  Early exaggeration: %.1f (first %d iterations)\n",
               early_exag, early_exag_iter);

        if (sklearn_compatible) {
            printf("  Mode: Sklearn-compatible (stabilizations disabled)\n");
            printf("    - Abrupt exaggeration switch at iter %d\n", early_exag_iter);
            printf("    - Constant learning rate (no decay)\n");
            printf("    - No gradient clipping\n");
            printf("    - No Z-based early stopping\n");
        } else {
            printf("  Mode: Enhanced (stabilizations enabled)\n");
            printf("    - Gradual exaggeration decay (iters %d-%d)\n",
                   early_exag_iter, early_exag_iter + 100);
            printf("    - Learning rate decay after iter %d (0.5%% per iter)\n",
                   momentum_switch_iter);
            printf("    - Gradient clipping (5× median norm)\n");
            printf("    - Z-based early stopping (threshold: 10.0)\n");
        }
    }

    // Reuse the repulsive force between H2 rebuilds (see the call site below).
    // TSNE_REP_REUSE=0 restores the old stale-weights matvec, for A/B runs only.
    const int rep_reuse = tsne_env_int("TSNE_REP_REUSE", 1) && n_point >= 2000;
    int n_rebuild = 0;

    // Iterations at (or right after) which the points are renumbered; TSNE_REORDER=0 disables.
    // Any dimension: the generic attractive pass gathers Y[d*n + j] for every neighbor, and
    // without the renumbering those gathers are random at 1M points (3D, 1 thread: the
    // attractive pass took 2043 s of 3711 s and grew like N^1.7).
    int reorder_iters[16] = {30, 100, 250, 400, 600, 800};
    int n_reorder_iters = tsne_env_int("TSNE_REORDER", 1) ? 6 : 0;
    // TSNE_REORDER_ITERS="40,70,100,...": another (ascending) schedule, for experiments
    const char *reorder_env = getenv("TSNE_REORDER_ITERS");
    if (n_reorder_iters > 0 && reorder_env != NULL && *reorder_env != '\0') {
        n_reorder_iters = 0;
        for (const char *c = reorder_env; *c != '\0' && n_reorder_iters < 16; ) {
            reorder_iters[n_reorder_iters++] = atoi(c);
            while (*c != '\0' && *c != ',') c++;
            if (*c == ',') c++;
        }
    }
    int next_reorder = 0;

    for (int iter = 0; iter < n_iter; iter++) {
        double iter_start = get_wtime_sec();
        double t_start, t_end;  // For profiling timing points

        // Switch momentum if needed
        if (iter == momentum_switch_iter) {
            momentum = final_momentum;
            if (verbose >= 2) printf("  [iter %d] Switched to final momentum %.2f\n",
                                     iter, momentum);
        }

        // Gradual early exaggeration decay (prevents abrupt transition)
        // Phase 1 (0-250): Full exaggeration (12.0)
        // Phase 2 (250-350): Linear decay from 12.0 to 1.0
        // Phase 3 (350+): No exaggeration (1.0)
        // NOTE: sklearn uses abrupt switch at early_exag_iter
        DTYPE exag;
        if (sklearn_compatible) {
            // Sklearn mode: abrupt switch at early_exag_iter
            exag = (iter < early_exag_iter) ? early_exag : 1.0;
        } else {
            // Our mode: gradual decay for stability
            if (iter < early_exag_iter) {
                exag = early_exag;
            } else if (iter < early_exag_iter + 100) {
                // Gradual decay over 100 iterations
                DTYPE progress = (DTYPE)(iter - early_exag_iter) / 100.0;
                exag = early_exag - (early_exag - 1.0) * progress;
            } else {
                exag = 1.0;
            }
        }

        // Learning rate decay after momentum switch (prevents overshooting)
        // NOTE: sklearn uses constant learning rate
        DTYPE current_lr = learning_rate;
        if (!sklearn_compatible && iter >= momentum_switch_iter) {
            // Our mode: decay by 0.5% per iteration after momentum switch
            int decay_iters = iter - momentum_switch_iter;
            current_lr = learning_rate * pow(0.995, decay_iters);
        }
        // Sklearn mode: current_lr = learning_rate (no decay)

        // Rebuild H² matrix periodically BEFORE computing gradient
        // NOTE: h2_rebuild_freq may be 0 due to heap corruption workaround, use 10 as fallback
        int h2_rebuild_freq = (workspace->params->h2_rebuild_freq > 0) ? workspace->params->h2_rebuild_freq : 10;
        // For the first 20 iterations, always rebuild every iteration (freq=1) to track rapid movement
        int effective_rebuild_freq = (iter < 20) ? 1 : h2_rebuild_freq;
        // Early exaggeration is dominated by the (12x) attractive term and is
        // insensitive to a delayed repulsive force, so rebuild sparsely there.
        if (iter >= 20 && iter < early_exag_iter && rep_reuse) {
            const int freq_early = workspace->params->h2_rebuild_freq_early;
            if (freq_early > effective_rebuild_freq) effective_rebuild_freq = freq_early;
        }
        int rebuilt = (iter == 0);   // tsne_init built the H2 matrix for the initial Y
        const int want_rebuild = (iter > 0 && iter % effective_rebuild_freq == 0);
        if (n_point >= 2000 && want_rebuild) {
            // Renumber the points along the (previous) H2 tree order before
            // rebuilding, so the attractive pass gathers from nearby memory. Done
            // a handful of times: early on the embedding reorganizes quickly,
            // later the numbering stays good for hundreds of iterations.
            if (next_reorder < n_reorder_iters && iter >= reorder_iters[next_reorder] &&
                workspace->h2pack_W != NULL) {
                while (next_reorder < n_reorder_iters && iter >= reorder_iters[next_reorder]) next_reorder++;
                const int nthr_ = (workspace->params->num_threads > 0) ? workspace->params->num_threads : 1;
                if (tsne_renumber_points(workspace, workspace->h2pack_W->fwd_pmt_idx, nthr_) != 0)
                    fprintf(stderr, "WARNING: point renumbering skipped (out of memory)\n");
                if (next_reorder >= n_reorder_iters && workspace->P_scratch != NULL) {
                    sparse_matrix_destroy(workspace->P_scratch);   // last one: drop the scratch
                    workspace->P_scratch = NULL;
                }
            }

            if (verbose >= 2 || (verbose && h2_rebuild_freq == 1 && iter < 5))
                printf("  [iter %d] Rebuilding H² matrix with current Y positions\n", iter);

            if (workspace->profiling) t_start = get_wtime_sec();

            if (update_h2_matrix(workspace, 1) != 0) {
                fprintf(stderr, "ERROR: Failed to rebuild H² matrix at iteration %d\n", iter);
                return -1;
            }
            rebuilt = 1;
            n_rebuild++;

            if (workspace->profiling) {
                t_end = get_wtime_sec();
                workspace->profiling->h2_rebuild_time[iter] = t_end - t_start;
            }
        }

        // Zero gradient
        memset(workspace->grad, 0, low_dim * n_point * sizeof(DTYPE));

        // Compute attractive forces (sparse)
        if (workspace->profiling) t_start = get_wtime_sec();

        // NOTE: num_threads may be 0 due to heap corruption workaround, use 1 as fallback
        int num_threads = (workspace->params->num_threads > 0) ? workspace->params->num_threads : 1;
        if (low_dim != 2 || compute_attractive_forces_2d(workspace, exag, num_threads) != 0) {
            compute_attractive_forces(workspace->P, workspace->Y, low_dim, n_point,
                                       exag, workspace->grad, num_threads);
        }

        if (workspace->profiling) {
            t_end = get_wtime_sec();
            workspace->profiling->p_matvec_time[iter] = t_end - t_start;
        }

        // Compute repulsive forces (H²)
        // Between rebuilds the H2 matrix holds kernel weights for the Y it was
        // built from. Multiplying those stale weights by the *current* Y gives a
        // repulsion that grows linearly with how far a pair has since separated
        // (instead of decaying), which is what makes infrequent rebuilds diverge.
        // Instead the whole force from the last rebuild is reused -- a consistent,
        // bounded, slightly delayed gradient -- which also skips the matvec on
        // those iterations.
        DTYPE Z;
        if (rep_reuse && workspace->rep == NULL)
            workspace->rep = (DTYPE*)malloc((size_t)low_dim * n_point * sizeof(DTYPE));
        if (rep_reuse && !rebuilt && workspace->rep_valid) {
            const DTYPE *rep = workspace->rep;
            DTYPE *g = workspace->grad;
            #pragma omp parallel for if (num_threads > 1) num_threads(num_threads) schedule(static)
            for (int i = 0; i < low_dim * n_point; i++) g[i] -= rep[i];
            Z = workspace->rep_Z;
        } else {
            Z = compute_repulsive_forces_h2pack(workspace->h2pack_W, workspace->Y,
                                                low_dim, n_point, workspace,
                                                workspace->grad, iter);
        }

        // Z-based early stopping (prevents complete collapse)
        // If Z drops below 10 after momentum switch, optimization is diverging
        // NOTE: sklearn does not have this check
        if (!sklearn_compatible && iter > momentum_switch_iter && Z < 10.0) {
            if (verbose) {
                printf("\n*** STOPPING: Z collapsed to %.3e (threshold: 10.0) ***\n", Z);
                printf("    Iteration: %d/%d\n", iter + 1, n_iter);
                printf("    This prevents further divergence\n");
            }
            break;
        }

        // KL divergence for the verbose progress print and, at the last
        // iteration, for kl_divergence_. It is a full pass over P with a
        // log() per nonzero (1.8 s serial at 1.2M points) and is not counted
        // by the profiling timers, so it runs only at the 50-iteration print
        // cadence when verbose, always at the end, and on all threads with
        // per-row sums added block-wise (thread-count-independent rounding).
        if ((verbose && iter % 50 == 0) || iter == n_iter - 1) {
            // KL(P||Q) = Σ_ij p_ij log(p_ij / q_ij), q_ij = (1 + ||y_i - y_j||²)^-1 / Z
            const sparse_matrix_t *P = workspace->P;
            const DTYPE *Y = workspace->Y;
            DTYPE *row_kl = (DTYPE*)malloc((size_t)n_point * sizeof(DTYPE));
            #pragma omp parallel for schedule(dynamic, 1024) num_threads(num_threads)
            for (int i = 0; i < n_point; i++) {
                DTYPE acc = 0.0;
                for (int idx = P->row_ptr[i]; idx < P->row_ptr[i + 1]; idx++) {
                    const int j = P->col_idx[idx];
                    const DTYPE p_ij = P->values[idx] * exag;   // the exaggerated P is what is optimized
                    DTYPE dist2 = 0.0;
                    for (int d = 0; d < low_dim; d++) {
                        const DTYPE diff = Y[d * n_point + i] - Y[d * n_point + j];
                        dist2 += diff * diff;
                    }
                    const DTYPE q_ij = 1.0 / ((1.0 + dist2) * Z);
                    if (p_ij > 1e-12 && q_ij > 1e-12) {
                        acc += p_ij * log(p_ij / q_ij);
                    }
                }
                row_kl[i] = acc;
            }
            workspace->kl_divergence = tsne_sum_blocked(row_kl, n_point, num_threads);
            free(row_kl);
        }

        // Gradient clipping (prevents rare blow-ups)
        // NOTE: sklearn does not use gradient clipping
        // DISABLED: Gradient clipping removed for performance (O(N log N) overhead)
        if (0) {  // Always skip gradient clipping
            // Compute per-point gradient norms
            DTYPE *grad_norms = (DTYPE*)malloc(n_point * sizeof(DTYPE));
            for (int i = 0; i < n_point; i++) {
                DTYPE norm_sq = 0.0;
                for (int d = 0; d < low_dim; d++) {
                    DTYPE g = workspace->grad[d * n_point + i];
                    norm_sq += g * g;
                }
                grad_norms[i] = sqrt(norm_sq);
            }

            // Find median gradient norm
            DTYPE *sorted_norms = (DTYPE*)malloc(n_point * sizeof(DTYPE));
            memcpy(sorted_norms, grad_norms, n_point * sizeof(DTYPE));
            // Use qsort for O(N log N) instead of O(N²) selection sort
            qsort(sorted_norms, n_point, sizeof(DTYPE), compare_double);
            DTYPE median_norm = sorted_norms[n_point / 2];
            free(sorted_norms);

            // Clip gradient norms to 5× median (prevents blow-ups)
            DTYPE clip_threshold = 5.0 * median_norm;
            if (clip_threshold > 0.0) {  // Avoid division by zero
                for (int i = 0; i < n_point; i++) {
                    if (grad_norms[i] > clip_threshold) {
                        DTYPE scale = clip_threshold / grad_norms[i];
                        for (int d = 0; d < low_dim; d++) {
                            workspace->grad[d * n_point + i] *= scale;
                        }
                    }
                }
            }
            free(grad_norms);
        }

        // Gradient descent with momentum and adaptive gains
        // Match sklearn's exact algorithm:
        // 1. Detect sign changes: inc = (update * grad < 0)
        // 2. Update gains
        // 3. Apply gains to gradient
        // 4. Update velocity with momentum

        if (workspace->profiling) t_start = get_wtime_sec();

        #pragma omp parallel for if (num_threads > 1) num_threads(num_threads) schedule(static)
        for (int i = 0; i < low_dim * n_point; i++) {
            // Detect if gradient and velocity have opposite signs
            int sign_changed = (workspace->uY[i] * workspace->grad[i] < 0.0);

            // Update gains
            if (sign_changed) {
                workspace->gains[i] += 0.2;  // Increase gain when oscillating
            } else {
                workspace->gains[i] *= 0.8;  // Decrease gain when consistent
            }
            // Floor at minimum gain
            if (workspace->gains[i] < 0.01) {
                workspace->gains[i] = 0.01;
            }

            // Apply gains to gradient
            workspace->grad[i] *= workspace->gains[i];

            // Update velocity: uY = momentum * uY - current_lr * (grad * gains)
            workspace->uY[i] = momentum * workspace->uY[i] - current_lr * workspace->grad[i];

            // Update position
            workspace->Y[i] += workspace->uY[i];
        }

        if (workspace->profiling) {
            t_end = get_wtime_sec();
            workspace->profiling->gradient_time[iter] = t_end - t_start;
        }

        // Center embedding (remove mean)
        if (workspace->profiling) t_start = get_wtime_sec();

        for (int d = 0; d < low_dim; d++) {
            DTYPE *Y_d = workspace->Y + (size_t)d * n_point;
            const DTYPE mean = tsne_sum_blocked(Y_d, n_point, num_threads) / n_point;
            #pragma omp parallel for if (num_threads > 1) num_threads(num_threads) schedule(static)
            for (int i = 0; i < n_point; i++)
                Y_d[i] -= mean;
        }

        // Convergence detection
        if (workspace->params->enable_early_stopping && iter >= workspace->params->min_iter) {
            // Compute change in Y since last convergence check
            DTYPE max_change = 0.0;
            DTYPE sum_change = 0.0;

            for (int i = 0; i < n_point; i++) {
                DTYPE point_change_sq = 0.0;
                for (int d = 0; d < low_dim; d++) {
                    int idx = d * n_point + i;
                    DTYPE diff = workspace->Y[idx] - workspace->Y_prev_for_convergence[idx];
                    point_change_sq += diff * diff;
                }
                DTYPE point_change = sqrt(point_change_sq);
                sum_change += point_change;
                if (point_change > max_change) {
                    max_change = point_change;
                }
            }
            DTYPE avg_change = sum_change / n_point;

            // Check if converged
            if (max_change < workspace->params->convergence_tol) {
                workspace->convergence_count++;
                if (verbose >= 2) {
                    printf("  [iter %d] Convergence check: max_change=%.3e < tol=%.3e (%d/%d)\n",
                           iter, max_change, workspace->params->convergence_tol,
                           workspace->convergence_count, workspace->params->convergence_patience);
                }

                // Check if we've been below threshold for patience iterations
                if (workspace->convergence_count >= workspace->params->convergence_patience) {
                    workspace->converged = 1;
                    if (verbose) {
                        printf("\n*** CONVERGED at iteration %d ***\n", iter + 1);
                        printf("    Max change: %.3e (threshold: %.3e)\n",
                               max_change, workspace->params->convergence_tol);
                        printf("    Avg change: %.3e\n", avg_change);
                    }
                    break;  // Exit optimization loop
                }
            } else {
                // Reset counter if change exceeds threshold
                workspace->convergence_count = 0;
            }

            // Update previous Y for next convergence check
            memcpy(workspace->Y_prev_for_convergence, workspace->Y,
                   low_dim * n_point * sizeof(DTYPE));
        } else if (workspace->params->enable_early_stopping && iter == workspace->params->min_iter - 1) {
            // Initialize Y_prev_for_convergence at min_iter - 1
            memcpy(workspace->Y_prev_for_convergence, workspace->Y,
                   low_dim * n_point * sizeof(DTYPE));
        }

        // Verbose debugging: Check Y magnitude periodically
        if (verbose >= 2 && (iter < 5 || iter % 10 == 0)) {
            DTYPE y_norm = 0.0;
            for (int i = 0; i < low_dim * n_point; i++)
                y_norm += workspace->Y[i] * workspace->Y[i];
            y_norm = sqrt(y_norm);
            printf("  [DEBUG iter %d] ||Y||=%.6e, grad_norm=%.6e, uY_norm=%.6e\n",
                   iter, y_norm,
                   sqrt(workspace->grad[0]*workspace->grad[0] + workspace->grad[1]*workspace->grad[1]),
                   sqrt(workspace->uY[0]*workspace->uY[0] + workspace->uY[1]*workspace->uY[1]));
        }

        if (workspace->profiling) {
            t_end = get_wtime_sec();
            workspace->profiling->other_time[iter] = t_end - t_start;
        }

        double iter_end = get_wtime_sec();
        workspace->iter_times[iter] = iter_end - iter_start;

        // Print progress (print_freq for regular updates, every 10 for KL)
        if (verbose && (iter % print_freq == 0 || iter == n_iter - 1)) {
            printf("  Iteration %4d/%d: Z=%.3e, time=%.3f sec",
                   iter + 1, n_iter, Z, workspace->iter_times[iter]);
            // Show stabilization parameters during decay phase
            if (iter >= early_exag_iter && iter < early_exag_iter + 100) {
                printf(" [exag=%.2f, lr=%.1f]", exag, current_lr);
            } else if (iter >= momentum_switch_iter && iter < momentum_switch_iter + 50) {
                printf(" [lr=%.1f]", current_lr);
            }
            printf("\n");
        }

        // Print KL divergence at the iteration-print cadence
        if (verbose && iter % 50 == 0) {
            printf("  [KL iter %4d] KL divergence: %.5f\n", iter, workspace->kl_divergence);
        }
    }

    if (verbose) {
        if (workspace->converged) {
            printf("\nOptimization complete (converged)\n");
        } else {
            printf("\nOptimization complete\n");
        }
    }

    if (tsne_env_int("TSNE_H2_TIMERS", 0)) {
        long pool_hits = 0, pool_misses = 0;
        malloc_aligned_pool_stats(&pool_hits, &pool_misses);
        printf("H2REBUILDS %d  pool hits=%ld misses=%ld\n", n_rebuild, pool_hits, pool_misses);
        tsne_accum_h2_timers(workspace->h2pack_W);
        tsne_print_h2_timers();
    }

    // Write profiling data if enabled
    if (workspace->profiling && workspace->params->profiling_output_file) {
        tsne_write_profiling_data(workspace, workspace->params->profiling_output_file);
    }

    return 0;
}

// =============================================================================
// Get embedding result
// =============================================================================

void tsne_get_embedding(tsne_workspace_t *workspace, DTYPE *Y_out)
{
    const int n_point = workspace->n_point;
    const int *order = workspace->point_order;
    if (order == NULL) {
        memcpy(Y_out, workspace->Y, (size_t)workspace->low_dim * n_point * sizeof(DTYPE));
        return;
    }
    // Points were renumbered during the optimization: report them in the caller's order
    for (int d = 0; d < workspace->low_dim; d++) {
        const DTYPE *src = workspace->Y + (size_t)d * n_point;
        DTYPE *dst = Y_out + (size_t)d * n_point;
        for (int k = 0; k < n_point; k++) dst[order[k]] = src[k];
    }
}

// =============================================================================
// Destroy workspace
// =============================================================================

void tsne_destroy(tsne_workspace_t *workspace)
{
    if (workspace == NULL) return;

    // Clean up temporary proxy cache file
    if (workspace->temp_cache_file) {
        unlink(workspace->temp_cache_file);  // Delete temporary file
        free(workspace->temp_cache_file);
    }

    if (workspace->X) free(workspace->X);
    if (workspace->Y) free(workspace->Y);
    if (workspace->Y_prev) free(workspace->Y_prev);
    if (workspace->grad) free(workspace->grad);
    if (workspace->gains) free(workspace->gains);
    if (workspace->Y_il) free(workspace->Y_il);
    if (workspace->P_val32) free(workspace->P_val32);
    if (workspace->Y_il32) free(workspace->Y_il32);
    tsne_h2_matmul_free(&workspace->mm_ws);
    tsne_pp_cache_free(&workspace->pp_cache);
    if (workspace->rep) free(workspace->rep);
    if (workspace->point_order) free(workspace->point_order);
    if (workspace->P_scratch) sparse_matrix_destroy(workspace->P_scratch);
    if (workspace->uY) free(workspace->uY);
    if (workspace->s) free(workspace->s);
    if (workspace->ones) free(workspace->ones);

    if (workspace->v) free(workspace->v);

    if (workspace->iter_times) free(workspace->iter_times);

    if (workspace->Y_prev_for_convergence) free(workspace->Y_prev_for_convergence);
    if (workspace->delta_Y) free(workspace->delta_Y);

    // Free profiling data
    if (workspace->profiling) {
        if (workspace->profiling->knn_time) free(workspace->profiling->knn_time);
        if (workspace->profiling->p_construction_time) free(workspace->profiling->p_construction_time);
        if (workspace->profiling->p_matvec_time) free(workspace->profiling->p_matvec_time);
        if (workspace->profiling->h2_rebuild_time) free(workspace->profiling->h2_rebuild_time);
        if (workspace->profiling->h2_matvec_s_time) free(workspace->profiling->h2_matvec_s_time);
        if (workspace->profiling->h2_matvec_v_time) free(workspace->profiling->h2_matvec_v_time);
        if (workspace->profiling->z_computation_time) free(workspace->profiling->z_computation_time);
        if (workspace->profiling->gradient_time) free(workspace->profiling->gradient_time);
        if (workspace->profiling->other_time) free(workspace->profiling->other_time);
        free(workspace->profiling);
    }

    if (workspace->P) sparse_matrix_destroy(workspace->P);
    if (workspace->h2pack_W) H2P_destroy(&workspace->h2pack_W);
    malloc_aligned_pool_enable(0);   // release the blocks parked by the last H2P_destroy

    // Free params (allocated separately to avoid heap corruption)
    if (workspace->params) free(workspace->params);

    free(workspace);
}

// =============================================================================
// Convenience function
// =============================================================================

int tsne_fit_transform(
    const DTYPE *X,
    int n_point,
    int pt_dim,
    int low_dim,
    const tsne_params_t *params,
    DTYPE *Y_out
)
{
    tsne_workspace_t *workspace = tsne_init(X, n_point, pt_dim, low_dim, params);
    if (workspace == NULL) return -1;

    int ret = tsne_run(workspace);
    if (ret != 0) {
        tsne_destroy(workspace);
        return ret;
    }

    tsne_get_embedding(workspace, Y_out);
    tsne_destroy(workspace);

    return 0;
}

int tsne_fit_transform_with_P(
    const DTYPE *X,
    int n_point,
    int pt_dim,
    int low_dim,
    const tsne_params_t *params,
    sparse_matrix_t *P,
    DTYPE *Y_out
)
{
    tsne_workspace_t *workspace = tsne_init_with_P(X, n_point, pt_dim, low_dim, params, P);
    if (workspace == NULL) return -1;

    int ret = tsne_run(workspace);
    if (ret != 0) {
        tsne_destroy(workspace);
        return ret;
    }

    tsne_get_embedding(workspace, Y_out);
    tsne_destroy(workspace);

    return 0;
}

// =============================================================================
// Public accessor: read KL divergence from workspace
// (declared in tsne_h2pack.h:396 — used by Python bindings)
// =============================================================================
DTYPE compute_kl_divergence(tsne_workspace_t *workspace)
{
    if (workspace == NULL) return 0.0;
    return workspace->kl_divergence;
}
