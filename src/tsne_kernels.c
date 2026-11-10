#include <string.h>
#include <math.h>
#include "tsne_h2pack.h"

// Include H2Pack's optimized kernel implementations
#include "../../H2Pack/src/H2Pack_config.h"
#include "../../H2Pack/src/ASTER/include/aster.h"
#include "../../H2Pack/src/H2Pack_2D_kernels.h"
#include "../../H2Pack/src/H2Pack_3D_kernels.h"

// Static parameters for t-SNE quadratic kernel: (1 + c*r²)^a
// For t-SNE kernel (1 + r²)^{-2}, we use c=1.0 and a=-2.0
static const DTYPE tsne_quad_params[2] = {1.0, -2.0};

/**
 * t-SNE repulsive force kernel: (1 + ||y_i - y_j||²)^{-2}
 *
 * This implementation uses H2Pack's optimized quadratic kernel for 2D/3D
 * and falls back to generic implementation for other dimensions.
 *
 * The t-SNE kernel (1 + r²)^{-2} is the quadratic kernel (1 + c*r²)^a
 * with parameters c=1.0 and a=-2.0.
 *
 * @param coord0     First point set (pt_dim × n0)
 * @param ld0        Leading dimension of coord0
 * @param n0         Number of points in coord0
 * @param coord1     Second point set (pt_dim × n1)
 * @param ld1        Leading dimension of coord1
 * @param n1         Number of points in coord1
 * @param krnl_param Kernel parameter (int* pointing to pt_dim)
 * @param mat        Output kernel matrix (n0 × n1)
 * @param ldm        Leading dimension of mat
 */
void tsne_repulsive_kernel_eval(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const void *krnl_param,
    DTYPE * __restrict mat, const int ldm
)
{
    // Extract point dimension from parameter
    const int pt_dim = krnl_param ? *(const int*)krnl_param : 2;

    if (pt_dim == 2) {
        // Direct (1 + r^2)^-2 = 1 / (d * d). H2Pack's generic quadratic kernel
        // evaluates (1 + c r^2)^a through a vector pow(), which on NEON falls back
        // to scalar log/exp per entry -- an order of magnitude slower than one
        // divide, and this routine fills every D/B block and every ID sample
        // matrix on each H2 rebuild. The plain loop below auto-vectorizes.
        const DTYPE *x0 = coord0, *y0 = coord0 + ld0;
        const DTYPE *x1 = coord1, *y1 = coord1 + ld1;
        for (int i = 0; i < n0; i++) {
            DTYPE * __restrict mat_irow = mat + (size_t)i * ldm;
            const DTYPE x0_i = x0[i], y0_i = y0[i];
            #pragma omp simd
            for (int j = 0; j < n1; j++) {
                const DTYPE dx = x0_i - x1[j];
                const DTYPE dy = y0_i - y1[j];
                const DTYPE d = 1.0 + dx * dx + dy * dy;
                mat_irow[j] = 1.0 / (d * d);
            }
        }
    } else if (pt_dim == 3) {
        // Use H2Pack's SIMD-optimized 3D quadratic kernel
        Quadratic_3D_eval_intrin_t(
            coord0, ld0, n0,
            coord1, ld1, n1,
            (const void*)tsne_quad_params,
            mat, ldm
        );
    } else {
        // Fallback to generic dimension-independent implementation
        // (For arbitrary dimensions not supported by H2Pack)
        for (int i = 0; i < n0; i++) {
            for (int j = 0; j < n1; j++) {
                // Compute squared distance
                DTYPE r2 = 0.0;
                for (int d = 0; d < pt_dim; d++) {
                    DTYPE diff = coord0[d * ld0 + i] - coord1[d * ld1 + j];
                    r2 += diff * diff;
                }

                // Compute (1 + r²)^{-2}
                DTYPE denom = 1.0 + r2;
                mat[i * ldm + j] = 1.0 / (denom * denom);
            }
        }
    }
}

/**
 * Optimized vectorized version of t-SNE repulsive kernel
 * This is the same as the standard version since we already use optimized kernels
 */
void tsne_repulsive_kernel_eval_intrin(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const void *krnl_param,
    DTYPE * __restrict mat, const int ldm
)
{
    // The standard version already uses H2Pack's intrinsic implementations
    tsne_repulsive_kernel_eval(coord0, ld0, n0, coord1, ld1, n1,
                                krnl_param, mat, ldm);
}

/**
 * Kernel bi-matvec for t-SNE repulsive kernel
 * Uses H2Pack's SIMD-optimized quadratic kernel bimv
 *
 * Computes:
 *   x_out_0 = kernel_matrix(coord0, coord1) * x_in_0
 *   x_out_1 = kernel_matrix(coord1, coord0) * x_in_1
 */
void tsne_repulsive_kernel_bimv(
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const void *krnl_param,
    const DTYPE *x_in_0, const DTYPE *x_in_1,
    DTYPE * __restrict x_out_0, DTYPE * __restrict x_out_1
)
{
    const int pt_dim = krnl_param ? *(const int*)krnl_param : 2;

    // Call H2Pack's optimized quadratic kernel bimv based on dimension
    if (pt_dim == 2) {
        // Use H2Pack's SIMD-optimized 2D quadratic kernel bimv
        Quadratic_2D_krnl_bimv_intrin_t(
            coord0, ld0, n0,
            coord1, ld1, n1,
            (const void*)tsne_quad_params,
            x_in_0, x_in_1,
            x_out_0, x_out_1
        );
    } else if (pt_dim == 3) {
        // Use H2Pack's SIMD-optimized 3D quadratic kernel bimv
        Quadratic_3D_krnl_bimv_intrin_t(
            coord0, ld0, n0,
            coord1, ld1, n1,
            (const void*)tsne_quad_params,
            x_in_0, x_in_1,
            x_out_0, x_out_1
        );
    } else {
        // Fallback to generic implementation
        // Forward direction: x_out_0 += K(coord0, coord1) * x_in_0
        for (int i = 0; i < n0; i++) {
            DTYPE sum = 0.0;
            for (int j = 0; j < n1; j++) {
                DTYPE r2 = 0.0;
                for (int d = 0; d < pt_dim; d++) {
                    DTYPE diff = coord0[d * ld0 + i] - coord1[d * ld1 + j];
                    r2 += diff * diff;
                }
                DTYPE denom = 1.0 + r2;
                DTYPE kval = 1.0 / (denom * denom);
                sum += kval * x_in_0[j];
            }
            x_out_0[i] += sum;
        }

        // Backward direction: x_out_1 += K(coord1, coord0) * x_in_1
        for (int j = 0; j < n1; j++) {
            DTYPE sum = 0.0;
            for (int i = 0; i < n0; i++) {
                DTYPE r2 = 0.0;
                for (int d = 0; d < pt_dim; d++) {
                    DTYPE diff = coord1[d * ld1 + j] - coord0[d * ld0 + i];
                    r2 += diff * diff;
                }
                DTYPE denom = 1.0 + r2;
                DTYPE kval = 1.0 / (denom * denom);
                sum += kval * x_in_1[i];
            }
            x_out_1[j] += sum;
        }
    }
}

/**
 * Estimate FLOPs for kernel bi-matvec
 * Uses H2Pack's FLOP estimates for 2D/3D quadratic kernels
 */
int tsne_repulsive_kernel_bimv_flop(int pt_dim, int n0, int n1)
{
    if (pt_dim == 2) {
        // Use H2Pack's 2D quadratic kernel FLOP count
        // Defined in H2Pack_2D_kernels.h
        return 2 * n0 * n1 * Quadratic_2D_krnl_bimv_flop;
    } else if (pt_dim == 3) {
        // Use H2Pack's 3D quadratic kernel FLOP count
        // Defined in H2Pack_3D_kernels.h
        return 2 * n0 * n1 * Quadratic_3D_krnl_bimv_flop;
    } else {
        // Generic fallback: same as original estimate
        // Per element: pt_dim subtractions + pt_dim multiplications + pt_dim additions (for r²)
        //              + 1 add (1 + r²) + 1 multiply (denom*denom) + 1 divide + 1 multiply (kval * x)
        // Total per pair: (3*pt_dim + 5) FLOPs
        // Two directions: 2 * n0 * n1
        return 2 * n0 * n1 * (3 * pt_dim + 5);
    }
}
