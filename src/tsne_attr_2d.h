// Blocked 2D attractive kernel. This is a template, not a header: tsne_h2pack.c includes
// it once per coordinate type of the gather, with
//   ATTR_FN  name of the generated function
//   ATTR_YT  type of the interleaved (x0,y0,x1,y1,...) coordinates that are gathered
// (a template because `#pragma omp` cannot appear inside a macro body).
//
// Per row: gather the neighbors' coordinates ATTR_BLK at a time, then with a constant
// trip count -- straight vector code -- form the differences in ATTR_YT and evaluate
// q = p / (1 + dx^2 + dy^2) and the two force sums in float32. The row's
// force is accumulated in registers (in DTYPE, block sums in float).
//
// pv holds P's values as float32, aligned with col_idx. The divide has always run in
// float32, so reading p as float loses nothing, and the stream of P shrinks from 12 to
// 8 bytes per nonzero.
//
// ATTR_BLK = 16: measured ~2% faster than 8 (fewer block epilogues per row of ~150
// nonzeros) and 22% faster than 24 or 32. The tail of a row stays scalar: padding it to
// a full block (variable trip count in the gather) measured slower than the scalar loop
// it replaces. Software prefetch of the gather targets measured 45% slower once the
// points are renumbered (the gathers hit L1/L2 already).

static void ATTR_FN(
    const int *row_ptr, const int *col_idx, const float *pv, const ATTR_YT *Yg,
    const int n_point, const DTYPE scale, DTYPE *grad, const int nthr, const int chunk
)
{
    #pragma omp parallel for if (nthr > 1) num_threads(nthr) schedule(dynamic, chunk)
    for (int i = 0; i < n_point; i++) {
        const ATTR_YT xi = Yg[2 * (size_t)i], yi = Yg[2 * (size_t)i + 1];
        DTYPE gx = 0.0, gy = 0.0;
        // Pointer-bumped blocks with constant inner offsets, no `int` index arithmetic:
        // setuptools compiles with Python's -fno-strict-overflow, under which
        // col_idx[idx + k] with int idx has to honor wrap-around and the gather loop
        // came out 30% slower.
        const int *cj = col_idx + row_ptr[i];
        const float *pj = pv + row_ptr[i];
        const size_t row_nnz = (size_t)(row_ptr[i + 1] - row_ptr[i]);
        for (size_t nblk = row_nnz / ATTR_BLK; nblk > 0; nblk--, cj += ATTR_BLK, pj += ATTR_BLK) {
            // One 2-coordinate copy per neighbor (a single load + store); the simd loop
            // de-interleaves. Gathering x and y separately costs two scalar loads and two
            // vector-lane inserts per neighbor and measured 17% slower.
            ATTR_YT xy[2 * ATTR_BLK];
            for (size_t k = 0; k < ATTR_BLK; k++)
                memcpy(&xy[2 * k], &Yg[2 * (size_t)cj[k]], 2 * sizeof(ATTR_YT));
            float bx = 0.0f, by = 0.0f;
            #pragma omp simd reduction(+:bx,by)
            for (size_t k = 0; k < ATTR_BLK; k++) {
                const float dx = (float)(xi - xy[2 * k]), dy = (float)(yi - xy[2 * k + 1]);
                const float q = pj[k] / (1.0f + dx * dx + dy * dy);
                bx += q * dx;
                by += q * dy;
            }
            gx += bx;
            gy += by;
        }
        float tx = 0.0f, ty = 0.0f;
        for (size_t k = 0; k < row_nnz % ATTR_BLK; k++) {
            const size_t j = (size_t)cj[k];
            const float ddx = (float)(xi - Yg[2 * j]), ddy = (float)(yi - Yg[2 * j + 1]);
            const float q = pj[k] / (1.0f + ddx * ddx + ddy * ddy);
            tx += q * ddx;
            ty += q * ddy;
        }
        grad[i]           += scale * (gx + tx);
        grad[n_point + i] += scale * (gy + ty);
    }
}

#undef ATTR_FN
#undef ATTR_YT
