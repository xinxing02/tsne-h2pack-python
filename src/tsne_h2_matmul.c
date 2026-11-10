// Fused multi-vector H2 matvec for the t-SNE repulsive force.
//
// Every iteration needs W * [1 | Y0 | Y1] for the same H2 matrix W. Calling
// H2P_matvec once per column streams the near-field blocks D (by far the largest
// part of the H2 representation) through the cores once per column, and that
// dense phase is memory-bandwidth bound. H2P_matmul does not help: it memcpy's
// each D/B block into a thread buffer before every GEMM. This routine applies all
// columns in a single traversal, touching each U/B/D block exactly once per
// iteration, with no BLAS calls on the tiny (rank ~10) far-field blocks.
//
// It only reads the H2Pack representation (non-HSS); the H2 build is untouched.
//
// Just-in-time blocks. t-SNE uses each H2 matrix for exactly one such product (the
// repulsive force is then reused until the next rebuild), so storing the B and D blocks
// means writing ~2 GB (500K points) to memory during the build only to stream it back
// once. When the matrix was built with BD_JIT = 1 the blocks are not stored: each one is
// evaluated here into a small per-thread buffer that stays in cache and is applied at
// once. Same kernel routine, same block layout and same apply kernels as the stored
// path, so the product is the same.

#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "tsne_h2pack.h"

typedef struct {
    size_t cap_n, cap_rank, cap_node;
    int    cap_thread, cap_nv;
    DTYPE  *xp;      // nv * n            permuted input, one array per column
    DTYPE  *yp;      // n_thread * nv * n per-thread permuted output
    DTYPE  *y0;      // nv * sum(rank)    upward-pass coefficients
    DTYPE  *y1;      // n_thread * nv * sum(rank) per-thread downward-pass coefficients
    size_t *off;     // n_node            offset of each node's block in y0/y1 (units of DTYPE)
} mm_ws_t;

#define MM_INLINE static inline __attribute__((always_inline))

// out0 (nrow) += M * in0 (ncol), for nv columns; column c of a vector block lives at base + c * ld
MM_INLINE void mm_apply(
    const int nrow, const int ncol, const DTYPE * restrict M, const int ldM,
    const DTYPE * restrict in0, const int ld_in0, DTYPE * restrict out0, const int ld_out0, const int nv
)
{
    if (nv == 3) {
        const DTYPE *x0 = in0, *x1 = in0 + ld_in0, *x2 = in0 + 2 * ld_in0;
        DTYPE *o0 = out0, *o1 = out0 + ld_out0, *o2 = out0 + 2 * ld_out0;
        for (int i = 0; i < nrow; i++) {
            const DTYPE *m = M + (size_t)i * ldM;
            DTYPE s0 = 0, s1 = 0, s2 = 0;
            #pragma omp simd reduction(+:s0,s1,s2)
            for (int j = 0; j < ncol; j++) {
                s0 += m[j] * x0[j];
                s1 += m[j] * x1[j];
                s2 += m[j] * x2[j];
            }
            o0[i] += s0; o1[i] += s1; o2[i] += s2;
        }
        return;
    }
    for (int c = 0; c < nv; c++) {
        const DTYPE *x = in0 + (size_t)c * ld_in0;
        DTYPE *o = out0 + (size_t)c * ld_out0;
        for (int i = 0; i < nrow; i++) {
            const DTYPE *m = M + (size_t)i * ldM;
            DTYPE s = 0;
            #pragma omp simd reduction(+:s)
            for (int j = 0; j < ncol; j++) s += m[j] * x[j];
            o[i] += s;
        }
    }
}

// out1 (ncol) += M^T * in1 (nrow)
MM_INLINE void mm_apply_trans(
    const int nrow, const int ncol, const DTYPE * restrict M, const int ldM,
    const DTYPE * restrict in1, const int ld_in1, DTYPE * restrict out1, const int ld_out1, const int nv
)
{
    if (nv == 3) {
        const DTYPE *a0 = in1, *a1 = in1 + ld_in1, *a2 = in1 + 2 * ld_in1;
        DTYPE *o0 = out1, *o1 = out1 + ld_out1, *o2 = out1 + 2 * ld_out1;
        for (int i = 0; i < nrow; i++) {
            const DTYPE *m = M + (size_t)i * ldM;
            const DTYPE b0 = a0[i], b1 = a1[i], b2 = a2[i];
            #pragma omp simd
            for (int j = 0; j < ncol; j++) {
                o0[j] += m[j] * b0;
                o1[j] += m[j] * b1;
                o2[j] += m[j] * b2;
            }
        }
        return;
    }
    for (int c = 0; c < nv; c++) {
        const DTYPE *a = in1 + (size_t)c * ld_in1;
        DTYPE *o = out1 + (size_t)c * ld_out1;
        for (int i = 0; i < nrow; i++) {
            const DTYPE *m = M + (size_t)i * ldM;
            const DTYPE b = a[i];
            #pragma omp simd
            for (int j = 0; j < ncol; j++) o[j] += m[j] * b;
        }
    }
}

// out0 (nrow) += M * in0 (ncol)  and  out1 (ncol) += M^T * in1 (nrow), one pass over M
MM_INLINE void mm_apply_bi(
    const int nrow, const int ncol, const DTYPE * restrict M, const int ldM,
    const DTYPE * restrict in0, const int ld_in0, const DTYPE * restrict in1, const int ld_in1,
    DTYPE * restrict out0, const int ld_out0, DTYPE * restrict out1, const int ld_out1, const int nv
)
{
    if (nv != 3) {
        mm_apply(nrow, ncol, M, ldM, in0, ld_in0, out0, ld_out0, nv);
        mm_apply_trans(nrow, ncol, M, ldM, in1, ld_in1, out1, ld_out1, nv);
        return;
    }
    const DTYPE *x0 = in0, *x1 = in0 + ld_in0, *x2 = in0 + 2 * ld_in0;
    const DTYPE *a0 = in1, *a1 = in1 + ld_in1, *a2 = in1 + 2 * ld_in1;
    DTYPE *p0 = out0, *p1 = out0 + ld_out0, *p2 = out0 + 2 * ld_out0;
    DTYPE *q0 = out1, *q1 = out1 + ld_out1, *q2 = out1 + 2 * ld_out1;
    int i = 0;
    // Two rows of M per sweep halves the load/store traffic on out1
    for (; i + 2 <= nrow; i += 2) {
        const DTYPE *m0 = M + (size_t)i * ldM, *m1 = m0 + ldM;
        const DTYPE b00 = a0[i], b01 = a1[i], b02 = a2[i];
        const DTYPE b10 = a0[i + 1], b11 = a1[i + 1], b12 = a2[i + 1];
        DTYPE s00 = 0, s01 = 0, s02 = 0, s10 = 0, s11 = 0, s12 = 0;
        #pragma omp simd reduction(+:s00,s01,s02,s10,s11,s12)
        for (int j = 0; j < ncol; j++) {
            const DTYPE r0 = m0[j], r1 = m1[j];
            const DTYPE u0 = x0[j], u1 = x1[j], u2 = x2[j];
            s00 += r0 * u0; s01 += r0 * u1; s02 += r0 * u2;
            s10 += r1 * u0; s11 += r1 * u1; s12 += r1 * u2;
            q0[j] += r0 * b00 + r1 * b10;
            q1[j] += r0 * b01 + r1 * b11;
            q2[j] += r0 * b02 + r1 * b12;
        }
        p0[i] += s00; p1[i] += s01; p2[i] += s02;
        p0[i + 1] += s10; p1[i + 1] += s11; p2[i + 1] += s12;
    }
    for (; i < nrow; i++) {
        const DTYPE *m = M + (size_t)i * ldM;
        const DTYPE b0 = a0[i], b1 = a1[i], b2 = a2[i];
        DTYPE s0 = 0, s1 = 0, s2 = 0;
        #pragma omp simd reduction(+:s0,s1,s2)
        for (int j = 0; j < ncol; j++) {
            const DTYPE r = m[j];
            s0 += r * x0[j]; s1 += r * x1[j]; s2 += r * x2[j];
            q0[j] += r * b0; q1[j] += r * b1; q2[j] += r * b2;
        }
        p0[i] += s0; p1[i] += s1; p2[i] += s2;
    }
}

static int mm_ws_reserve(mm_ws_t *ws, size_t n, size_t rank_sum, int n_node, int n_thread, int nv)
{
    if (n > ws->cap_n || n_thread > ws->cap_thread || nv > ws->cap_nv) {
        free(ws->xp); free(ws->yp);
        ws->xp = (DTYPE*)malloc(sizeof(DTYPE) * nv * n);
        ws->yp = (DTYPE*)malloc(sizeof(DTYPE) * nv * n * n_thread);
        if (ws->xp == NULL || ws->yp == NULL) return -1;
    }
    if (rank_sum > ws->cap_rank || n_thread > ws->cap_thread || nv > ws->cap_nv) {
        // Head-room so that small rank changes between rebuilds do not reallocate
        size_t cap = rank_sum + rank_sum / 4 + 64;
        free(ws->y0); free(ws->y1);
        ws->y0 = (DTYPE*)malloc(sizeof(DTYPE) * nv * cap);
        ws->y1 = (DTYPE*)malloc(sizeof(DTYPE) * nv * cap * n_thread);
        if (ws->y0 == NULL || ws->y1 == NULL) return -1;
        ws->cap_rank = cap;
    }
    if ((size_t)n_node > ws->cap_node) {
        free(ws->off);
        ws->off = (size_t*)malloc(sizeof(size_t) * n_node);
        if (ws->off == NULL) return -1;
        ws->cap_node = n_node;
    }
    if (n > ws->cap_n) ws->cap_n = n;
    if (n_thread > ws->cap_thread) ws->cap_thread = n_thread;
    if (nv > ws->cap_nv) ws->cap_nv = nv;
    return 0;
}

void tsne_h2_matmul_free(void **ws_)
{
    mm_ws_t *ws = (mm_ws_t*)(*ws_);
    if (ws == NULL) return;
    free(ws->xp); free(ws->yp); free(ws->y0); free(ws->y1); free(ws->off);
    free(ws);
    *ws_ = NULL;
}

// B block of admissible pair i, evaluated as in H2P_build_B_AOT()
MM_INLINE void mm_eval_B(H2Pack_p h2, const int node0, const int node1, const int lvl0, const int lvl1, DTYPE *Bi)
{
    const int n_point = h2->n_point;
    const int *pt_cluster = h2->pt_cluster;
    H2P_dense_mat_p J0 = h2->J_coord[node0], J1 = h2->J_coord[node1];
    if (lvl0 == lvl1) {
        h2->krnl_eval(J0->data, J0->ncol, J0->ncol, J1->data, J1->ncol, J1->ncol, h2->krnl_param, Bi, J1->ncol);
    } else if (lvl0 > lvl1) {
        const int pt_s1 = pt_cluster[2 * node1], npt1 = pt_cluster[2 * node1 + 1] - pt_s1 + 1;
        h2->krnl_eval(J0->data, J0->ncol, J0->ncol, h2->coord + pt_s1, n_point, npt1, h2->krnl_param, Bi, npt1);
    } else {
        const int pt_s0 = pt_cluster[2 * node0], npt0 = pt_cluster[2 * node0 + 1] - pt_s0 + 1;
        h2->krnl_eval(h2->coord + pt_s0, n_point, npt0, J1->data, J1->ncol, J1->ncol, h2->krnl_param, Bi, J1->ncol);
    }
}

// D block of the point clusters of node0 x node1, evaluated as in H2P_build_D_AOT()
MM_INLINE void mm_eval_D(H2Pack_p h2, const int node0, const int node1, DTYPE *Di)
{
    const int n_point = h2->n_point;
    const int *pt_cluster = h2->pt_cluster;
    const int pt_s0 = pt_cluster[2 * node0], npt0 = pt_cluster[2 * node0 + 1] - pt_s0 + 1;
    const int pt_s1 = pt_cluster[2 * node1], npt1 = pt_cluster[2 * node1 + 1] - pt_s1 + 1;
    h2->krnl_eval(h2->coord + pt_s0, n_point, npt0, h2->coord + pt_s1, n_point, npt1, h2->krnl_param, Di, npt1);
}

// Y (n x nv) = W * X (n x nv); X[c] / Y[c] point to the length-n column c, in the
// caller's original point order. Returns 0 on success, -1 if this H2 matrix is not
// supported (caller falls back to H2P_matvec).
int tsne_h2_matmul(H2Pack_p h2, const int nv, const DTYPE *const *X, DTYPE *const *Y, void **ws_)
{
    if (h2 == NULL || h2->is_HSS != 0 || h2->krnl_dim != 1) return -1;
    const int jit = (h2->BD_JIT != 0);
    if (jit && (h2->krnl_eval == NULL || h2->J_coord == NULL)) return -1;

    const int n            = h2->krnl_mat_size;
    const int n_node       = h2->n_node;
    const int n_leaf       = h2->n_leaf_node;
    const int n_thread     = h2->n_thread;
    const int max_level    = h2->max_level;
    const int min_adm_lvl  = h2->min_adm_level;
    const int max_child    = h2->max_child;
    const int *children    = h2->children;
    const int *n_child     = h2->n_child;
    const int *level_nodes = h2->level_nodes;
    const int *level_n_node = h2->level_n_node;
    const int *node_level  = h2->node_level;
    const int *leaf_nodes  = h2->height_nodes;
    const int *mat_cluster = h2->mat_cluster;
    const int *adm_pairs   = h2->r_adm_pairs;
    const int *inadm_pairs = h2->r_inadm_pairs;
    const int n_adm        = h2->n_r_adm_pair;
    const int n_inadm      = h2->n_r_inadm_pair;
    H2P_dense_mat_p *U     = h2->U;

    if (*ws_ == NULL) {
        *ws_ = calloc(1, sizeof(mm_ws_t));
        if (*ws_ == NULL) return -1;
    }
    mm_ws_t *ws = (mm_ws_t*)(*ws_);

    size_t rank_sum = 0;
    for (int i = 0; i < n_node; i++) rank_sum += (size_t)U[i]->ncol;
    if (mm_ws_reserve(ws, n, rank_sum, n_node, n_thread, nv) != 0) return -1;
    size_t *off = ws->off;
    {
        size_t acc = 0;
        for (int i = 0; i < n_node; i++) { off[i] = acc; acc += (size_t)nv * U[i]->ncol; }
    }
    // Largest block a thread has to hold in JIT mode
    size_t blk_max = 0;
    if (jit) {
        for (int i = 0; i < n_adm; i++) {
            const size_t sz = (size_t)h2->B_nrow[i] * h2->B_ncol[i];
            if (sz > blk_max) blk_max = sz;
        }
        for (int i = 0; i < n_leaf + n_inadm; i++) {
            const size_t sz = (size_t)h2->D_nrow[i] * h2->D_ncol[i];
            if (sz > blk_max) blk_max = sz;
        }
    }
    int jit_failed = 0;

    const size_t y_len  = (size_t)nv * n;
    const size_t y1_len = (size_t)nv * rank_sum;
    DTYPE *xp = ws->xp, *yp = ws->yp, *y0 = ws->y0, *y1 = ws->y1;
    const int *fwd_pmt = h2->fwd_pmt_idx, *bwd_pmt = h2->bwd_pmt_idx;

    #pragma omp parallel num_threads(n_thread)
    {
        const int tid = omp_get_thread_num();
        DTYPE *yp_t = yp + (size_t)tid * y_len;
        DTYPE *y1_t = y1 + (size_t)tid * y1_len;
        DTYPE *blk = NULL;
        if (jit) {
            blk = (DTYPE*)malloc(sizeof(DTYPE) * (blk_max + 1));
            if (blk == NULL) {
                #pragma omp atomic write
                jit_failed = 1;
            }
        }

        // 1. Permute the input into H2Pack's tree order; clear this thread's outputs
        for (int c = 0; c < nv; c++) {
            const DTYPE *src = X[c];
            DTYPE *dst = xp + (size_t)c * n;
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < n; i++) dst[i] = src[fwd_pmt[i]];
        }
        memset(yp_t, 0, sizeof(DTYPE) * y_len);
        memset(y1_t, 0, sizeof(DTYPE) * y1_len);
        #pragma omp barrier

        // 2. Upward pass: y0[node] = U[node]^T * (x rows | children's y0)
        for (int lvl = max_level; lvl >= min_adm_lvl; lvl--) {
            const int *lvl_nodes = level_nodes + (size_t)lvl * n_leaf;
            #pragma omp for schedule(dynamic, 8)
            for (int j = 0; j < level_n_node[lvl]; j++) {
                const int node = lvl_nodes[j];
                const int rank = U[node]->ncol;
                if (rank == 0) continue;
                DTYPE *y0_node = y0 + off[node];
                memset(y0_node, 0, sizeof(DTYPE) * nv * rank);
                if (n_child[node] == 0) {
                    mm_apply_trans(U[node]->nrow, rank, U[node]->data, U[node]->ld,
                                   xp + mat_cluster[2 * node], n, y0_node, rank, nv);
                } else {
                    const int *ch = children + (size_t)node * max_child;
                    int srow = 0;
                    for (int k = 0; k < n_child[node]; k++) {
                        const int rank_k = U[ch[k]]->ncol;
                        if (rank_k == 0) continue;
                        mm_apply_trans(rank_k, rank, U[node]->data + (size_t)srow * U[node]->ld, U[node]->ld,
                                       y0 + off[ch[k]], rank_k, y0_node, rank, nv);
                        srow += rank_k;
                    }
                }
            }
        }

        // 3. Far-field coupling: y1[i] += B_ij * y0[j], y1[j] += B_ij^T * y0[i]
        #pragma omp for schedule(dynamic, 32) nowait
        for (int i = 0; i < n_adm; i++) {
            const int node0 = adm_pairs[2 * i], node1 = adm_pairs[2 * i + 1];
            const int lvl0 = node_level[node0], lvl1 = node_level[node1];
            const DTYPE *B;
            if (jit) {
                if (blk == NULL) continue;
                mm_eval_B(h2, node0, node1, lvl0, lvl1, blk);
                B = blk;
            } else {
                B = h2->B_data + h2->B_ptr[i];
            }
            const int B_nrow = h2->B_nrow[i], B_ncol = h2->B_ncol[i];
            if (lvl0 == lvl1) {
                mm_apply_bi(B_nrow, B_ncol, B, B_ncol,
                            y0 + off[node1], U[node1]->ncol, y0 + off[node0], U[node0]->ncol,
                            y1_t + off[node0], U[node0]->ncol, y1_t + off[node1], U[node1]->ncol, nv);
            } else if (lvl0 > lvl1) {
                // node1 is a leaf on a coarser level: uncompressed on node1's side
                const int s1 = mat_cluster[2 * node1];
                mm_apply_bi(B_nrow, B_ncol, B, B_ncol,
                            xp + s1, n, y0 + off[node0], U[node0]->ncol,
                            y1_t + off[node0], U[node0]->ncol, yp_t + s1, n, nv);
            } else {
                const int s0 = mat_cluster[2 * node0];
                mm_apply_bi(B_nrow, B_ncol, B, B_ncol,
                            y0 + off[node1], U[node1]->ncol, xp + s0, n,
                            yp_t + s0, n, y1_t + off[node1], U[node1]->ncol, nv);
            }
        }

        // 4. Near field: diagonal leaf blocks, then inadmissible leaf pairs
        #pragma omp for schedule(dynamic, 16) nowait
        for (int i = 0; i < n_leaf; i++) {
            const int s = mat_cluster[2 * leaf_nodes[i]];
            const DTYPE *D;
            if (jit) {
                if (blk == NULL) continue;
                mm_eval_D(h2, leaf_nodes[i], leaf_nodes[i], blk);
                D = blk;
            } else {
                D = h2->D_data + h2->D_ptr[i];
            }
            mm_apply(h2->D_nrow[i], h2->D_ncol[i], D, h2->D_ncol[i],
                     xp + s, n, yp_t + s, n, nv);
        }
        #pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < n_inadm; i++) {
            const int s0 = mat_cluster[2 * inadm_pairs[2 * i]];
            const int s1 = mat_cluster[2 * inadm_pairs[2 * i + 1]];
            const int k = n_leaf + i;
            const DTYPE *D;
            if (jit) {
                if (blk == NULL) continue;
                mm_eval_D(h2, inadm_pairs[2 * i], inadm_pairs[2 * i + 1], blk);
                D = blk;
            } else {
                D = h2->D_data + h2->D_ptr[k];
            }
            mm_apply_bi(h2->D_nrow[k], h2->D_ncol[k], D, h2->D_ncol[k],
                        xp + s1, n, xp + s0, n, yp_t + s0, n, yp_t + s1, n, nv);
        }

        // 5. Reduce the per-thread buffers into thread 0's
        if (n_thread > 1) {
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < y_len; i++) {
                DTYPE s = yp[i];
                for (int t = 1; t < n_thread; t++) s += yp[(size_t)t * y_len + i];
                yp[i] = s;
            }
            #pragma omp for schedule(static)
            for (size_t i = 0; i < y1_len; i++) {
                DTYPE s = y1[i];
                for (int t = 1; t < n_thread; t++) s += y1[(size_t)t * y1_len + i];
                y1[i] = s;
            }
        }

        // 6. Downward pass: push y1 to the children, expand onto the points at the leaves
        for (int lvl = min_adm_lvl; lvl <= max_level; lvl++) {
            const int *lvl_nodes = level_nodes + (size_t)lvl * n_leaf;
            #pragma omp for schedule(dynamic, 8)
            for (int j = 0; j < level_n_node[lvl]; j++) {
                const int node = lvl_nodes[j];
                const int rank = U[node]->ncol;
                if (rank == 0) continue;
                const DTYPE *y1_node = y1 + off[node];
                if (n_child[node] == 0) {
                    mm_apply(U[node]->nrow, rank, U[node]->data, U[node]->ld,
                             y1_node, rank, yp + mat_cluster[2 * node], n, nv);
                } else {
                    const int *ch = children + (size_t)node * max_child;
                    int srow = 0;
                    for (int k = 0; k < n_child[node]; k++) {
                        const int rank_k = U[ch[k]]->ncol;
                        if (rank_k == 0) continue;
                        mm_apply(rank_k, rank, U[node]->data + (size_t)srow * U[node]->ld, U[node]->ld,
                                 y1_node, rank, y1 + off[ch[k]], rank_k, nv);
                        srow += rank_k;
                    }
                }
            }
        }

        // 7. Permute the result back to the caller's point order
        for (int c = 0; c < nv; c++) {
            const DTYPE *src = yp + (size_t)c * n;
            DTYPE *dst = Y[c];
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < n; i++) dst[i] = src[bwd_pmt[i]];
        }
        free(blk);
    }
    return jit_failed ? -1 : 0;   // (out of memory: the caller falls back to H2P_matvec)
}
