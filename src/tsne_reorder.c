// Spatial renumbering of the points during the optimization.
//
// The attractive pass gathers y_j for every nonzero P_ij. With the input numbering
// those gathers are random over the whole embedding, and once the embedding no longer
// fits in cache (a few 100K points) the pass is dominated by cache misses. Points
// that are neighbors in P end up close together in the embedding, so renumbering them
// along a space-filling order of the *current* embedding makes the gathers local. The
// H2 tree already provides such an order (its leaf-by-leaf point permutation) for
// free. The cumulative renumbering is kept in workspace->point_order and undone in
// tsne_get_embedding, so callers never see it.

#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "tsne_h2pack.h"

// dst = transpose(src), with source row i renumbered to row_new[i] (NULL: identity)
// and visited in the order row_visit[0..n) (NULL: 0..n). Visiting the source rows in
// ascending *new* number leaves every destination row sorted by column. dst must have
// room for the same n and nnz; `next` is an n-sized scratch.
//
// With nthr > 1 the source rows are split (in visiting order) into one chunk per thread.
// Each thread histograms the destination rows of its chunk; from those, every thread
// gets its own starting offset inside every destination row and scatters its chunk
// there. Chunk t lands before chunk t+1 in each row, so the result is identical to the
// serial counting sort. Costs nthr * n ints of scratch; without it, runs serially.
static void csr_transpose(
    const sparse_matrix_t *src, const int *row_visit, const int *row_new,
    const int *col_new, sparse_matrix_t *dst, int *next, const int nthr
)
{
    const int n = src->n_rows, nnz = src->row_ptr[n];
    int *row_ptr = dst->row_ptr, *col_idx = dst->col_idx;
    DTYPE *values = dst->values;

    int *cnt = (nthr > 1 && n >= 16 * nthr) ? (int*)malloc((size_t)nthr * n * sizeof(int)) : NULL;
    if (cnt != NULL) {
        #pragma omp parallel num_threads(nthr)
        {
            const int t = omp_get_thread_num(), nt = omp_get_num_threads();
            const int k_s = (int)((long long)n * t / nt), k_e = (int)((long long)n * (t + 1) / nt);
            int *cnt_t = cnt + (size_t)t * n;
            memset(cnt_t, 0, (size_t)n * sizeof(int));
            for (int k = k_s; k < k_e; k++) {
                const int i = row_visit ? row_visit[k] : k;
                for (int idx = src->row_ptr[i]; idx < src->row_ptr[i + 1]; idx++) {
                    const int j = src->col_idx[idx];
                    cnt_t[col_new ? col_new[j] : j]++;
                }
            }
            #pragma omp barrier

            #pragma omp for schedule(static)
            for (int r = 0; r < n; r++) {
                int tot = 0;
                for (int u = 0; u < nt; u++) tot += cnt[(size_t)u * n + r];
                row_ptr[r + 1] = tot;
            }
            #pragma omp single
            {
                row_ptr[0] = 0;
                for (int r = 0; r < n; r++) row_ptr[r + 1] += row_ptr[r];
            }
            #pragma omp for schedule(static)
            for (int r = 0; r < n; r++) {
                int pos = row_ptr[r];
                for (int u = 0; u < nt; u++) {
                    const int c = cnt[(size_t)u * n + r];
                    cnt[(size_t)u * n + r] = pos;
                    pos += c;
                }
            }

            for (int k = k_s; k < k_e; k++) {
                const int i = row_visit ? row_visit[k] : k;
                const int i_new = row_new ? row_new[i] : i;
                for (int idx = src->row_ptr[i]; idx < src->row_ptr[i + 1]; idx++) {
                    const int j = src->col_idx[idx];
                    const int dst_idx = cnt_t[col_new ? col_new[j] : j]++;
                    col_idx[dst_idx] = i_new;
                    values[dst_idx]  = src->values[idx];
                }
            }
        }
        free(cnt);
        return;
    }

    memset(row_ptr, 0, ((size_t)n + 1) * sizeof(int));
    for (int idx = 0; idx < nnz; idx++) {
        const int j = src->col_idx[idx];
        row_ptr[(col_new ? col_new[j] : j) + 1]++;
    }
    for (int i = 0; i < n; i++) row_ptr[i + 1] += row_ptr[i];
    memcpy(next, row_ptr, (size_t)n * sizeof(int));

    for (int k = 0; k < n; k++) {
        const int i = row_visit ? row_visit[k] : k;
        const int i_new = row_new ? row_new[i] : i;
        for (int idx = src->row_ptr[i]; idx < src->row_ptr[i + 1]; idx++) {
            const int j = src->col_idx[idx];
            const int dst_idx = next[col_new ? col_new[j] : j]++;
            col_idx[dst_idx] = i_new;
            values[dst_idx]  = src->values[idx];
        }
    }
}

// Renumber so that new point k is old point ord[k]. Returns 0 on success; on failure
// (out of memory) nothing has been changed.
int tsne_renumber_points(tsne_workspace_t *ws, const int *ord, int num_threads)
{
    const int n = ws->n_point, low_dim = ws->low_dim;
    const int nthr = (num_threads > 1) ? num_threads : 1;
    sparse_matrix_t *P = ws->P;

    // One P-sized scratch matrix, kept for the following renumberings: allocating and
    // freeing buffers of this size every time leaves them parked (and resident) in the
    // system allocator's large-block cache.
    if (ws->P_scratch == NULL) {
        ws->P_scratch = sparse_matrix_create(n, n, P->row_ptr[n]);
        if (ws->P_scratch == NULL) return -1;
    }
    sparse_matrix_t *T = ws->P_scratch;

    int   *inv     = (int*)malloc((size_t)n * sizeof(int));
    int   *order   = (int*)malloc((size_t)n * sizeof(int));
    int   *next    = (int*)malloc((size_t)n * sizeof(int));
    DTYPE *scratch = (DTYPE*)malloc((size_t)n * sizeof(DTYPE));
    if (inv == NULL || order == NULL || next == NULL || scratch == NULL) {
        free(inv); free(order); free(next); free(scratch);
        return -1;
    }
    #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(static)
    for (int k = 0; k < n; k++) inv[ord[k]] = k;

    // P: two transposes, P -> T -> P. The first renumbers (and sorts each row by new
    // column), the second restores the orientation, so this is correct for any P, not
    // only a symmetric one.
    csr_transpose(P, ord, inv, inv, T, next, nthr);
    csr_transpose(T, NULL, NULL, NULL, P, next, nthr);
    ws->P_val32_valid = 0;   // float copy of P->values (attractive pass) is now stale

    // Per-point state
    DTYPE *arrays[] = { ws->Y, ws->Y_prev, ws->uY, ws->gains,
                        ws->rep_valid ? ws->rep : NULL,
                        ws->Y_prev_for_convergence, ws->delta_Y };
    for (size_t a = 0; a < sizeof(arrays) / sizeof(arrays[0]); a++) {
        if (arrays[a] == NULL) continue;
        for (int d = 0; d < low_dim; d++) {
            DTYPE *col = arrays[a] + (size_t)d * n;
            #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(static)
            for (int k = 0; k < n; k++) scratch[k] = col[ord[k]];
            memcpy(col, scratch, (size_t)n * sizeof(DTYPE));
        }
    }

    // Cumulative map: current number -> caller's number
    #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(static)
    for (int k = 0; k < n; k++) order[k] = ws->point_order ? ws->point_order[ord[k]] : ord[k];
    free(ws->point_order);
    ws->point_order = order;

    free(inv); free(next); free(scratch);
    return 0;
}
