#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <pthread.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tsne_h2pack.h"
#include "utils.h"

// =============================================================================
// Sparse matrix operations
// =============================================================================

sparse_matrix_t* sparse_matrix_create(int n_rows, int n_cols, int nnz)
{
    sparse_matrix_t *mat = (sparse_matrix_t*)malloc(sizeof(sparse_matrix_t));
    if (mat == NULL) {
        return NULL;
    }

    mat->n_rows = n_rows;
    mat->n_cols = n_cols;
    mat->nnz = nnz;

    mat->row_ptr = (int*)malloc((n_rows + 1) * sizeof(int));
    mat->col_idx = (int*)malloc(nnz * sizeof(int));
    mat->values = (DTYPE*)malloc(nnz * sizeof(DTYPE));

    if (mat->row_ptr == NULL || mat->col_idx == NULL || mat->values == NULL) {
        sparse_matrix_destroy(mat);
        return NULL;
    }

    return mat;
}

void sparse_matrix_destroy(sparse_matrix_t *mat)
{
    if (mat == NULL) return;

    if (mat->row_ptr) free(mat->row_ptr);
    if (mat->col_idx) free(mat->col_idx);
    if (mat->values) free(mat->values);
    free(mat);
}

void sparse_matvec(const sparse_matrix_t *A, const DTYPE *x, DTYPE *y)
{
    for (int i = 0; i < A->n_rows; i++) {
        DTYPE sum = 0.0;
        for (int idx = A->row_ptr[i]; idx < A->row_ptr[i + 1]; idx++) {
            sum += A->values[idx] * x[A->col_idx[idx]];
        }
        y[i] = sum;
    }
}

// =============================================================================
// Distance and neighbor computations
// =============================================================================

// =============================================================================
// k-NN Search Utilities (Shared by Ball Tree)
// =============================================================================

typedef struct {
    int index;
    DTYPE dist;
} neighbor_t;

/**
 * Comparison function for sorting neighbors by distance
 */
static int compare_neighbors(const void *a, const void *b)
{
    DTYPE diff = ((neighbor_t*)a)->dist - ((neighbor_t*)b)->dist;
    return (diff > 0) - (diff < 0);
}

/**
 * Push a neighbor onto the max-heap (Phase 3 optimized)
 * Assumes heap is pre-initialized to size k with DBL_MAX values
 * This eliminates the "heap not full" branch for better CPU prediction
 * @param neighbors: Max-heap array (size k)
 * @param k: Number of neighbors
 * @param index: Candidate point index
 * @param dist2: SQUARED distance to candidate
 */
static inline void heap_push(neighbor_t *neighbors, int k, int index, DTYPE dist2)
{
    // Only push if distance is better than current max
    if (dist2 >= neighbors[0].dist) return;

    // Replace root with new value
    neighbors[0].index = index;
    neighbors[0].dist = dist2;

    // Heapify down to restore max-heap property (O(log k))
    int pos = 0;
    while (1) {
        int left = 2 * pos + 1;
        int right = 2 * pos + 2;
        int largest = pos;

        if (left < k && neighbors[left].dist > neighbors[largest].dist)
            largest = left;
        if (right < k && neighbors[right].dist > neighbors[largest].dist)
            largest = right;

        if (largest == pos) break;

        // Swap with largest child
        neighbor_t tmp = neighbors[pos];
        neighbors[pos] = neighbors[largest];
        neighbors[largest] = tmp;
        pos = largest;
    }
}

// =============================================================================
// Ball Tree for k-NN Search (pthread-parallelized, PHASE 2 OPTIMIZED)
// =============================================================================

/**
 * Transpose data from column-major to row-major layout
 * Column-major: X[d * n_point + i] = i-th point, d-th dimension
 * Row-major:    X[i * pt_dim + d] = i-th point, d-th dimension
 *
 * Row-major provides MUCH better cache locality for distance computation
 */
static DTYPE* transpose_to_rowmajor(const DTYPE *X_colmajor, int n_point, int pt_dim)
{
    DTYPE *X_rowmajor = (DTYPE*)malloc(n_point * pt_dim * sizeof(DTYPE));
    if (X_rowmajor == NULL) return NULL;

    for (int i = 0; i < n_point; i++) {
        for (int d = 0; d < pt_dim; d++) {
            X_rowmajor[i * pt_dim + d] = X_colmajor[d * n_point + i];
        }
    }

    return X_rowmajor;
}

/**
 * Ball tree node structure
 * Uses pointer-based tree for simplicity (can optimize to array-based later)
 */
typedef struct ball_node {
    // Ball properties
    DTYPE *centroid;       // Centroid position [pt_dim]
    DTYPE radius;          // Radius of enclosing ball

    // Tree structure
    struct ball_node *left;
    struct ball_node *right;

    // Leaf node data
    int *point_indices;    // Indices of points in this leaf
    int n_points;          // Number of points in leaf
    int is_leaf;           // 1 if leaf, 0 if interior node
} ball_node_t;

/**
 * Helper: swap two elements
 */
static void swap_int(int *a, int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

/**
 * Helper: compute SQUARED Euclidean distance between two points (contiguous)
 * OPTIMIZED: Returns dist^2 to avoid expensive sqrt() operations
 * Relies on compiler auto-vectorization with -O3 -march=native
 */
static inline DTYPE euclidean_distance_squared(const DTYPE *a, const DTYPE *b, int dim)
{
    DTYPE dist2 = 0.0;
    for (int d = 0; d < dim; d++) {
        DTYPE diff = a[d] - b[d];
        dist2 += diff * diff;
    }
    return dist2;
}

/**
 * Helper: compute SQUARED Euclidean distance from contiguous query to point in ROW-MAJOR X
 * OPTIMIZED: Row-major layout provides contiguous access for better cache locality
 */
static inline DTYPE euclidean_distance_to_point_squared_rowmajor(
    const DTYPE *query,  // Contiguous [pt_dim]
    const DTYPE *X_rowmajor,  // Row-major [n_point × pt_dim]
    int idx,             // Point index
    int pt_dim
)
{
    const DTYPE *point = &X_rowmajor[idx * pt_dim];
    return euclidean_distance_squared(query, point, pt_dim);
}

/**
 * Helper: compute min SQUARED distance from query point to ball
 * OPTIMIZED: Returns squared distance for comparison with tau
 */
static inline DTYPE min_distance_to_ball_squared(
    const DTYPE *query,
    const ball_node_t *node,
    int pt_dim
)
{
    DTYPE dist2_to_center = euclidean_distance_squared(query, node->centroid, pt_dim);
    DTYPE dist_to_center = sqrt(dist2_to_center);
    DTYPE min_dist = fmax(0.0, dist_to_center - node->radius);
    return min_dist * min_dist;  // Return squared min distance
}

/**
 * Helper: find dimension with largest spread
 */
static int find_widest_dimension(
    const DTYPE *X,      // Column-major
    const int *indices,
    int n,
    int pt_dim,
    int n_point
)
{
    int best_dim = 0;
    DTYPE max_spread = 0.0;

    for (int d = 0; d < pt_dim; d++) {
        DTYPE min_val = DBL_MAX;
        DTYPE max_val = -DBL_MAX;

        for (int i = 0; i < n; i++) {
            DTYPE val = X[d * n_point + indices[i]];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        DTYPE spread = max_val - min_val;
        if (spread > max_spread) {
            max_spread = spread;
            best_dim = d;
        }
    }

    return best_dim;
}

/**
 * Helper: partition indices around median along given dimension using quickselect (OPTIMIZED)
 * O(N) expected time instead of O(N log N) with qsort
 */
static void partition_indices_at_median(
    const DTYPE *X,
    int *indices,
    int n,
    int split_dim,
    int n_point
)
{
    if (n <= 1) return;

    int median_idx = n / 2;
    int left = 0;
    int right = n - 1;

    // Quickselect algorithm to partition around median
    while (left < right) {
        // Choose pivot (median-of-three for better performance)
        int mid = (left + right) / 2;
        DTYPE val_left = X[split_dim * n_point + indices[left]];
        DTYPE val_mid = X[split_dim * n_point + indices[mid]];
        DTYPE val_right = X[split_dim * n_point + indices[right]];

        // Swap median value to position left
        if ((val_left <= val_mid && val_mid <= val_right) || (val_right <= val_mid && val_mid <= val_left)) {
            swap_int(&indices[left], &indices[mid]);
        } else if ((val_mid <= val_left && val_left <= val_right) || (val_right <= val_left && val_left <= val_mid)) {
            // Already at left, do nothing
        } else {
            swap_int(&indices[left], &indices[right]);
        }

        DTYPE pivot = X[split_dim * n_point + indices[left]];
        int i = left;
        int j = right + 1;

        // Partition
        while (1) {  // Infinite loop, break explicitly
            do { i++; } while (i <= right && X[split_dim * n_point + indices[i]] < pivot);
            do { j--; } while (X[split_dim * n_point + indices[j]] > pivot);

            if (i >= j) break;
            swap_int(&indices[i], &indices[j]);
        }

        swap_int(&indices[left], &indices[j]);

        // Narrow search range
        if (j == median_idx) {
            break;  // Found median, done
        } else if (j > median_idx) {
            right = j - 1;  // Search left partition
        } else {
            left = j + 1;   // Search right partition
        }
    }
}

/**
 * Build ball tree recursively
 */
static ball_node_t* balltree_build_recursive(
    const DTYPE *X,      // Column-major [pt_dim × n_point]
    int *indices,        // Points to include in this subtree
    int n,               // Number of points
    int pt_dim,
    int n_point,
    int max_leaf_size
)
{
    if (n == 0) return NULL;

    // Allocate node
    ball_node_t *node = (ball_node_t*)malloc(sizeof(ball_node_t));
    if (node == NULL) return NULL;

    // Compute centroid (mean of all points in this subset)
    node->centroid = (DTYPE*)malloc(pt_dim * sizeof(DTYPE));
    if (node->centroid == NULL) {
        free(node);
        return NULL;
    }

    memset(node->centroid, 0, pt_dim * sizeof(DTYPE));
    for (int i = 0; i < n; i++) {
        int idx = indices[i];
        for (int d = 0; d < pt_dim; d++) {
            node->centroid[d] += X[d * n_point + idx];
        }
    }
    for (int d = 0; d < pt_dim; d++) {
        node->centroid[d] /= n;
    }

    // Compute radius (max distance from centroid to any point)
    node->radius = 0.0;
    for (int i = 0; i < n; i++) {
        int idx = indices[i];
        DTYPE dist2 = 0.0;
        for (int d = 0; d < pt_dim; d++) {
            DTYPE diff = X[d * n_point + idx] - node->centroid[d];
            dist2 += diff * diff;
        }
        DTYPE dist = sqrt(dist2);
        if (dist > node->radius) {
            node->radius = dist;
        }
    }

    // Base case: create leaf node
    if (n <= max_leaf_size) {
        node->is_leaf = 1;
        node->n_points = n;
        node->point_indices = (int*)malloc(n * sizeof(int));
        if (node->point_indices == NULL) {
            free(node->centroid);
            free(node);
            return NULL;
        }
        memcpy(node->point_indices, indices, n * sizeof(int));
        node->left = NULL;
        node->right = NULL;
        return node;
    }

    // Interior node: split along dimension with largest spread
    node->is_leaf = 0;
    node->point_indices = NULL;
    node->n_points = 0;

    // Find widest dimension
    int split_dim = find_widest_dimension(X, indices, n, pt_dim, n_point);

    // Partition indices at median using quickselect (O(N) instead of O(N log N))
    partition_indices_at_median(X, indices, n, split_dim, n_point);
    int median = n / 2;

    // Recursively build left and right subtrees
    node->left = balltree_build_recursive(X, indices, median,
                                         pt_dim, n_point, max_leaf_size);
    node->right = balltree_build_recursive(X, &indices[median], n - median,
                                          pt_dim, n_point, max_leaf_size);

    return node;
}

/**
 * Build ball tree (public wrapper)
 */
static ball_node_t* balltree_build(
    const DTYPE *X,
    int n_point,
    int pt_dim,
    int max_leaf_size
)
{
    // Create index array
    int *indices = (int*)malloc(n_point * sizeof(int));
    if (indices == NULL) return NULL;

    for (int i = 0; i < n_point; i++) {
        indices[i] = i;
    }

    // Build tree
    ball_node_t *root = balltree_build_recursive(X, indices, n_point,
                                                 pt_dim, n_point, max_leaf_size);

    free(indices);
    return root;
}

/**
 * Destroy ball tree (post-order traversal)
 */
static void balltree_destroy(ball_node_t *node)
{
    if (node == NULL) return;

    // Recursively destroy children
    balltree_destroy(node->left);
    balltree_destroy(node->right);

    // Free node data
    if (node->centroid) free(node->centroid);
    if (node->point_indices) free(node->point_indices);
    free(node);
}

/**
 * Search ball tree for k nearest neighbors of a single query point (PHASE 3 OPTIMIZED)
 * Uses recursive dual-ball pruning strategy with row-major memory layout
 * Simplified heap logic: assumes heap is pre-initialized to size k
 */
static void balltree_search_single_rowmajor(
    const ball_node_t *node,
    const DTYPE *X_rowmajor,     // Row-major [n_point × pt_dim]
    int query_idx,               // Query point index (for self-exclusion)
    const DTYPE *query_point,    // Query point (contiguous, for cache efficiency)
    int pt_dim,
    neighbor_t *neighbors,       // k-element max-heap (pre-initialized)
    int k,
    DTYPE *tau2                  // Current search radius SQUARED (max dist^2 in heap)
)
{
    if (node == NULL) return;

    // Compute SQUARED min distance from query to ball
    DTYPE min_possible_dist2 = min_distance_to_ball_squared(query_point, node, pt_dim);

    // Prune: if minimum possible squared distance exceeds tau^2, skip entire subtree
    if (min_possible_dist2 > *tau2) {
        return;
    }

    // Leaf node: check all points
    if (node->is_leaf) {
        for (int i = 0; i < node->n_points; i++) {
            int idx = node->point_indices[i];

            // Skip self
            if (idx == query_idx) continue;

            // Compute exact SQUARED distance with ROW-MAJOR access (much faster!)
            DTYPE dist2 = euclidean_distance_to_point_squared_rowmajor(
                query_point, X_rowmajor, idx, pt_dim);

            // Add to heap (Phase 3 optimized: no heap_size tracking needed)
            heap_push(neighbors, k, idx, dist2);

            // Update search radius: heap[0] always contains max distance
            *tau2 = neighbors[0].dist;
        }
        return;
    }

    // Interior node: compute distances to both children and visit in optimal order
    DTYPE left_min_dist2 = min_distance_to_ball_squared(query_point, node->left, pt_dim);
    DTYPE right_min_dist2 = min_distance_to_ball_squared(query_point, node->right, pt_dim);

    // Visit closer child first (heuristic for better pruning)
    if (left_min_dist2 < right_min_dist2) {
        // Left is closer
        if (left_min_dist2 <= *tau2) {
            balltree_search_single_rowmajor(node->left, X_rowmajor, query_idx, query_point,
                                           pt_dim, neighbors, k, tau2);
        }
        // Check right only if it could contain closer neighbors
        if (right_min_dist2 <= *tau2) {
            balltree_search_single_rowmajor(node->right, X_rowmajor, query_idx, query_point,
                                           pt_dim, neighbors, k, tau2);
        }
    } else {
        // Right is closer
        if (right_min_dist2 <= *tau2) {
            balltree_search_single_rowmajor(node->right, X_rowmajor, query_idx, query_point,
                                           pt_dim, neighbors, k, tau2);
        }
        // Check left only if it could contain closer neighbors
        if (left_min_dist2 <= *tau2) {
            balltree_search_single_rowmajor(node->left, X_rowmajor, query_idx, query_point,
                                           pt_dim, neighbors, k, tau2);
        }
    }
}

/**
 * Thread worker data structure for ball tree k-NN search
 */
typedef struct {
    // Shared read-only data
    const ball_node_t *tree;
    const DTYPE *X;
    int n_point;
    int pt_dim;
    int k;

    // Thread assignment (static partitioning)
    int thread_id;
    int start_idx;
    int end_idx;

    // Output (disjoint writes, no synchronization needed)
    int *knn_indices;
    DTYPE *knn_dists2;
} ball_tree_thread_data_t;

/**
 * Thread worker function for ball tree k-NN search (PHASE 3 OPTIMIZED)
 * Uses ROW-MAJOR data layout for contiguous memory access
 * Simplified heap logic: pre-initializes heap, no heap_size tracking
 */
static void* balltree_search_worker_rowmajor(void *arg)
{
    ball_tree_thread_data_t *data = (ball_tree_thread_data_t*)arg;

    // Per-thread allocations (reused for all queries in this thread)
    neighbor_t *neighbors = (neighbor_t*)malloc(data->k * sizeof(neighbor_t));

    // Process assigned points
    for (int i = data->start_idx; i < data->end_idx; i++) {
        // PHASE 3: Pre-initialize heap with infinity values (sklearn-style)
        // This eliminates the "heap not full" branch in heap_push for better CPU prediction
        for (int j = 0; j < data->k; j++) {
            neighbors[j].dist = DBL_MAX;
            neighbors[j].index = -1;
        }

        // Query point is already contiguous in row-major layout!
        const DTYPE *query_point = &data->X[i * data->pt_dim];

        // Search ball tree with squared distances and row-major data
        DTYPE tau2 = DBL_MAX;  // Squared search radius
        balltree_search_single_rowmajor(data->tree, data->X, i, query_point,
                                       data->pt_dim, neighbors, data->k, &tau2);

        // Sort neighbors by distance (ascending) - neighbors already contain squared distances
        qsort(neighbors, data->k, sizeof(neighbor_t), compare_neighbors);

        // Store results in output arrays - no need to square, already squared
        for (int j = 0; j < data->k; j++) {
            data->knn_indices[i * data->k + j] = neighbors[j].index;
            data->knn_dists2[i * data->k + j] = neighbors[j].dist;  // Already squared
        }
    }

    free(neighbors);
    return NULL;
}

/**
 * Find k-nearest neighbors using ball tree (PHASE 2 OPTIMIZED)
 * Uses row-major data layout for 2-3x better cache performance
 */
static void compute_knn_balltree(
    const DTYPE *X,
    int n_point,
    int pt_dim,
    int k,
    int *knn_indices,
    DTYPE *knn_dists2,
    int num_threads
)
{
    int max_leaf_size = 40;  // Optimal for most datasets

    // PHASE 2: Transpose X from column-major to row-major for better cache locality
    printf("[Ball Tree] Transposing data to row-major layout...\n");
    double t_transpose_start = get_wtime_sec();
    DTYPE *X_rowmajor = transpose_to_rowmajor(X, n_point, pt_dim);
    double t_transpose = get_wtime_sec() - t_transpose_start;

    if (X_rowmajor == NULL) {
        printf("[ERROR] Failed to transpose data matrix\n");
        return;
    }

    // Build ball tree (serial, done once) - still uses column-major for tree build
    double t_build_start = get_wtime_sec();
    ball_node_t *tree = balltree_build(X, n_point, pt_dim, max_leaf_size);
    double t_build = get_wtime_sec() - t_build_start;

    if (tree == NULL) {
        printf("[ERROR] Failed to build ball tree\n");
        free(X_rowmajor);
        return;
    }

    double t_search_start = get_wtime_sec();

    if (num_threads <= 1) {
        // Serial search with row-major data
        ball_tree_thread_data_t data = {
            .tree = tree,
            .X = X_rowmajor,  // Use row-major data!
            .n_point = n_point,
            .pt_dim = pt_dim,
            .k = k,
            .thread_id = 0,
            .start_idx = 0,
            .end_idx = n_point,
            .knn_indices = knn_indices,
            .knn_dists2 = knn_dists2
        };
        balltree_search_worker_rowmajor(&data);
    } else {
        // Parallel search with static partitioning
        pthread_t *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
        ball_tree_thread_data_t *thread_data = (ball_tree_thread_data_t*)malloc(
            num_threads * sizeof(ball_tree_thread_data_t));

        int points_per_thread = (n_point + num_threads - 1) / num_threads;

        // Create threads
        for (int t = 0; t < num_threads; t++) {
            thread_data[t] = (ball_tree_thread_data_t){
                .tree = tree,
                .X = X_rowmajor,  // Use row-major data!
                .n_point = n_point,
                .pt_dim = pt_dim,
                .k = k,
                .thread_id = t,
                .start_idx = t * points_per_thread,
                .end_idx = (t + 1) * points_per_thread,
                .knn_indices = knn_indices,
                .knn_dists2 = knn_dists2
            };

            // Ensure last thread doesn't exceed n_point
            if (thread_data[t].end_idx > n_point) {
                thread_data[t].end_idx = n_point;
            }

            pthread_create(&threads[t], NULL, balltree_search_worker_rowmajor, &thread_data[t]);
        }

        // Wait for all threads to complete
        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }

        free(threads);
        free(thread_data);
    }

    double t_search = get_wtime_sec() - t_search_start;

    // Report timing breakdown
    printf("[Ball Tree Timing] Transpose: %.3f sec, Build: %.3f sec, Search: %.3f sec, Total: %.3f sec\n",
           t_transpose, t_build, t_search, t_transpose + t_build + t_search);

    // Cleanup
    free(X_rowmajor);
    balltree_destroy(tree);
}

/**
 * Find k-nearest neighbors for all points using brute force
 * Returns: array of size n_point × k with neighbor indices
 *          array of size n_point × k with squared distances
 */
static void compute_knn_brute_force(
    const DTYPE *X,      // Input data (pt_dim × n_point)
    int n_point,
    int pt_dim,
    int k,
    int *knn_indices,    // Output: n_point × k
    DTYPE *knn_dists2    // Output: n_point × k (squared distances)
)
{
    // For each point
    for (int i = 0; i < n_point; i++) {
        // Compute distances to all other points
        DTYPE *dists = (DTYPE*)malloc(n_point * sizeof(DTYPE));
        int *indices = (int*)malloc(n_point * sizeof(int));

        for (int j = 0; j < n_point; j++) {
            if (i == j) {
                dists[j] = DBL_MAX;  // Exclude self
            } else {
                // Access points in column-major order
                const DTYPE *xi = X + i;
                const DTYPE *xj = X + j;
                DTYPE dist2 = 0.0;
                for (int d = 0; d < pt_dim; d++) {
                    DTYPE diff = xi[d * n_point] - xj[d * n_point];
                    dist2 += diff * diff;
                }
                dists[j] = dist2;
            }
            indices[j] = j;
        }

        // Partial sort to find k nearest neighbors
        for (int kk = 0; kk < k; kk++) {
            int min_idx = kk;
            for (int j = kk + 1; j < n_point; j++) {
                if (dists[j] < dists[min_idx]) {
                    min_idx = j;
                }
            }
            // Swap
            DTYPE tmp_dist = dists[kk];
            dists[kk] = dists[min_idx];
            dists[min_idx] = tmp_dist;

            int tmp_idx = indices[kk];
            indices[kk] = indices[min_idx];
            indices[min_idx] = tmp_idx;
        }

        // Store k nearest neighbors
        for (int kk = 0; kk < k; kk++) {
            knn_indices[i * k + kk] = indices[kk];
            knn_dists2[i * k + kk] = dists[kk];
        }

        free(dists);
        free(indices);
    }
}

/**
 * Binary search for sigma_i to match target perplexity
 * Returns sigma that gives perplexity closest to target
 */
static DTYPE search_sigma_for_perplexity(
    const DTYPE *dists2,  // Squared distances to neighbors (size: k)
    int k,
    DTYPE target_perplexity,
    DTYPE *p_conditional  // Output: conditional probabilities (size: k)
)
{
    const DTYPE tol = 1e-5;
    const int max_iter = 200;

    DTYPE sigma_min = 1e-20;
    DTYPE sigma_max = 1e20;
    DTYPE sigma = 1.0;

    for (int iter = 0; iter < max_iter; iter++) {
        // Compute conditional probabilities with current sigma
        DTYPE sum_exp = 0.0;
        for (int j = 0; j < k; j++) {
            p_conditional[j] = exp(-dists2[j] / (2.0 * sigma * sigma));
            sum_exp += p_conditional[j];
        }

        // Normalize
        if (sum_exp == 0.0) sum_exp = DBL_MIN;
        for (int j = 0; j < k; j++) {
            p_conditional[j] /= sum_exp;
        }

        // Compute entropy: H = -sum(p * log2(p))
        DTYPE entropy = 0.0;
        for (int j = 0; j < k; j++) {
            if (p_conditional[j] > DBL_MIN) {
                entropy -= p_conditional[j] * log2(p_conditional[j]);
            }
        }

        // Perplexity = 2^H
        DTYPE perplexity = pow(2.0, entropy);

        // Check convergence
        if (fabs(perplexity - target_perplexity) < tol) {
            break;
        }

        // Binary search update
        if (perplexity > target_perplexity) {
            sigma_max = sigma;
            sigma = (sigma + sigma_min) / 2.0;
        } else {
            sigma_min = sigma;
            sigma = (sigma + sigma_max) / 2.0;
        }
    }

    return sigma;
}

// =============================================================================
// Compute sparse P matrix
// =============================================================================

sparse_matrix_t* compute_sparse_P(
    const DTYPE *X,
    int n_point,
    int pt_dim,
    DTYPE perplexity,
    int use_vptree,
    int verbose,
    int num_threads
)
{
    // Determine number of neighbors: k = 3 * perplexity
    int k = (int)(3.0 * perplexity);
    if (k >= n_point) k = n_point - 1;

    if (verbose) {
        printf("Computing sparse P matrix (n=%d, perplexity=%.1f, k=%d)...\n",
               n_point, perplexity, k);
        const char *method_name = (use_vptree == 2) ? "ball tree" : "brute-force";
        printf("  Using %s for k-NN computation\n", method_name);
    }

    // 1. Compute k-nearest neighbors
    int *knn_indices = (int*)malloc(n_point * k * sizeof(int));
    DTYPE *knn_dists2 = (DTYPE*)malloc(n_point * k * sizeof(DTYPE));

    if (verbose) printf("  Computing k-NN...\n");

    // Use single method based on use_vptree
    if (use_vptree == 2) {
        // Ball tree method: O(N log N) expected, better for high-dimensional data
        compute_knn_balltree(X, n_point, pt_dim, k, knn_indices, knn_dists2,
                            num_threads);
    } else {
        // Brute-force method: O(N^2) complexity, but exact
        compute_knn_brute_force(X, n_point, pt_dim, k, knn_indices, knn_dists2);
    }

    // 2. Compute conditional probabilities p_{j|i}
    DTYPE **p_conditional = (DTYPE**)malloc(n_point * sizeof(DTYPE*));
    for (int i = 0; i < n_point; i++) {
        p_conditional[i] = (DTYPE*)malloc(k * sizeof(DTYPE));
    }

    if (verbose) printf("  Computing conditional probabilities...\n");
    for (int i = 0; i < n_point; i++) {
        search_sigma_for_perplexity(&knn_dists2[i * k], k, perplexity,
                                     p_conditional[i]);
    }

    // 3. Symmetrize: p_ij = (p_{j|i} + p_{i|j}) / (2N)
    if (verbose) printf("  Symmetrizing...\n");

    // Create sparse matrix (upper bound: each point has k neighbors)
    int nnz_max = n_point * k;
    sparse_matrix_t *P = sparse_matrix_create(n_point, n_point, nnz_max);

    // Build CSR structure
    int nnz = 0;
    P->row_ptr[0] = 0;

    for (int i = 0; i < n_point; i++) {
        // For each neighbor of i
        for (int kk = 0; kk < k; kk++) {
            int j = knn_indices[i * k + kk];
            DTYPE p_ji = p_conditional[i][kk];  // p_{j|i}

            // Find p_{i|j} (if j has i as neighbor)
            DTYPE p_ij = 0.0;
            for (int ll = 0; ll < k; ll++) {
                if (knn_indices[j * k + ll] == i) {
                    p_ij = p_conditional[j][ll];
                    break;
                }
            }

            // Symmetrized probability
            DTYPE p_sym = (p_ji + p_ij) / (2.0 * n_point);

            // Only store if non-zero
            if (p_sym > DBL_MIN) {
                P->col_idx[nnz] = j;
                P->values[nnz] = p_sym;
                nnz++;
            }
        }
        P->row_ptr[i + 1] = nnz;
    }

    // Update actual nnz
    P->nnz = nnz;

    if (verbose) {
        printf("  Sparse P matrix: %d non-zeros (%.2f%% sparse)\n",
               nnz, 100.0 * (1.0 - (double)nnz / (n_point * n_point)));
    }

    // Cleanup
    free(knn_indices);
    free(knn_dists2);
    for (int i = 0; i < n_point; i++) {
        free(p_conditional[i]);
    }
    free(p_conditional);

    return P;
}

// =============================================================================
// Binary search for P matrix from pre-computed k-NN distances
// (sklearn-compatible, uses natural log for entropy)
// =============================================================================

/**
 * Binary search for sigma to match target perplexity (sigma-based for better numerical range).
 *
 * This uses sigma-based formula: exp(-d² / (2*sigma²)) which handles large unnormalized
 * distances better than beta-based formula exp(-beta * d²).
 *
 * @param sqdistances  Squared distances to k neighbors (size: k)
 * @param k            Number of neighbors
 * @param desired_perplexity  Target perplexity
 * @param p_out        Output: conditional probabilities (size: k)
 * @return sigma       Bandwidth parameter
 */
static DTYPE binary_search_perplexity_sklearn(
    const DTYPE *sqdistances,
    int k,
    DTYPE desired_perplexity,
    DTYPE *p_out
)
{
    // Bisection on beta = 1 / (2 sigma^2), with the bracket grown by doubling /
    // halving while a side is still unbounded (as in sklearn / van der Maaten).
    //
    // The squared distances are shifted by their minimum first. P_j|i is invariant
    // to that shift, and it keeps the nearest neighbor's weight at exp(0) = 1, so
    // the row never underflows to all-zeros. (Clamping the exponent instead makes
    // every neighbor hit the clamp when d^2 >> sigma^2 -- e.g. MNIST in 0-255 pixel
    // units -- so the row looks *uniform*, the search walks sigma the wrong way,
    // and P comes out as exactly 1/k for every neighbor.)
    const DTYPE tol = 1e-5;
    const int max_iter = 200;
    const DTYPE desired_entropy = log(desired_perplexity);   // nats

    DTYPE d_min = sqdistances[0];
    for (int j = 1; j < k; j++)
        if (sqdistances[j] < d_min) d_min = sqdistances[j];

    DTYPE beta = 1.0, beta_min = 0.0, beta_max = 0.0;
    int has_min = 0, has_max = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        DTYPE sum_p = 0.0, sum_dp = 0.0;
        for (int j = 0; j < k; j++) {
            const DTYPE d = sqdistances[j] - d_min;
            p_out[j] = exp(-beta * d);
            sum_p += p_out[j];
            sum_dp += d * p_out[j];
        }
        // sum_p >= 1 (the nearest neighbor contributes exp(0)), so this is safe
        const DTYPE entropy = log(sum_p) + beta * sum_dp / sum_p;
        const DTYPE inv_sum = 1.0 / sum_p;
        for (int j = 0; j < k; j++) p_out[j] *= inv_sum;

        const DTYPE diff = entropy - desired_entropy;
        if (fabs(diff) < tol) break;

        if (diff > 0.0) {            // too flat -> sharpen (larger beta)
            beta_min = beta; has_min = 1;
            beta = has_max ? 0.5 * (beta + beta_max) : 2.0 * beta;
        } else {                     // too peaked -> flatten (smaller beta)
            beta_max = beta; has_max = 1;
            beta = has_min ? 0.5 * (beta + beta_min) : 0.5 * beta;
        }
    }

    return sqrt(1.0 / (2.0 * beta));
}

/**
 * Compute P matrix from pre-computed k-NN distances (from ANY k-NN source).
 *
 * This function accepts k-NN distances from any source (sklearn, FAISS, Annoy, etc.)
 * and computes the symmetric, normalized P matrix using sklearn-compatible binary search.
 *
 * @param knn_distances  CSR data array (L2 distances, NOT squared)
 * @param knn_indices    CSR column indices
 * @param knn_indptr     CSR row pointers (length n_samples + 1)
 * @param n_samples      Number of samples
 * @param nnz            Number of non-zeros in CSR matrix
 * @param perplexity     Target perplexity for binary search
 * @param num_threads    Number of OpenMP threads
 * @return               Symmetric, normalized P matrix (CSR format)
 */
sparse_matrix_t* compute_P_from_knn_distances(
    const DTYPE *knn_distances,
    const int *knn_indices,
    const int *knn_indptr,
    int n_samples,
    int nnz,
    DTYPE perplexity,
    int num_threads
)
{
    // Set OpenMP thread count if specified
    #ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    #else
    (void)num_threads;  // Suppress unused parameter warning when OpenMP disabled
    #endif

    // Step 1: Square distances (input is L2, we need squared for computation)
    DTYPE *knn_distances_sq = (DTYPE*)malloc(nnz * sizeof(DTYPE));
    if (knn_distances_sq == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for squared distances\n");
        return NULL;
    }

    // Parallelize distance squaring (simple, uniform work)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < nnz; i++) {
        knn_distances_sq[i] = knn_distances[i] * knn_distances[i];
    }

    // Step 2: Allocate CSR matrix for conditional probabilities P_j|i
    sparse_matrix_t *P_cond = sparse_matrix_create(n_samples, n_samples, nnz);
    if (P_cond == NULL) {
        free(knn_distances_sq);
        fprintf(stderr, "Error: Failed to create sparse matrix for P\n");
        return NULL;
    }

    // Step 3: Compute conditional probabilities for each sample
    // This is the main computational bottleneck and highly parallelizable:
    // - Each sample is processed independently
    // - Each writes to distinct memory locations (CSR structure guarantees no overlap)
    // - Binary search iterations may vary, so use dynamic scheduling
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i = 0; i < n_samples; i++) {
        int row_start = knn_indptr[i];
        int row_end = knn_indptr[i + 1];
        int k = row_end - row_start;

        const DTYPE *dists_sq_i = &knn_distances_sq[row_start];
        const int *indices_i = &knn_indices[row_start];

        // Binary search for sigma (using squared distances)
        // Probabilities are computed directly in p_out
        DTYPE *p_out = &P_cond->values[row_start];
        binary_search_perplexity_sklearn(dists_sq_i, k, perplexity, p_out);

        // Store column indices
        for (int j = 0; j < k; j++) {
            P_cond->col_idx[row_start + j] = indices_i[j];
        }
        P_cond->row_ptr[i] = row_start;
    }
    P_cond->row_ptr[n_samples] = nnz;
    P_cond->nnz = nnz;

    free(knn_distances_sq);

    // Step 4: Return conditional P matrix (asymmetric)
    // Let Python handle symmetrization using scipy: P = P + P.T
    // This matches sklearn's approach and avoids complex C symmetrization logic

    // NOTE: We return the CONDITIONAL probabilities P_j|i (asymmetric)
    // Python wrapper will compute: P = (P + P.T) / sum(P + P.T)
    // This is exactly what sklearn does in _joint_probabilities_nn()

    return P_cond;
}

// =============================================================================
// Embedding initialization
// =============================================================================

void initialize_embedding(
    DTYPE *Y,
    int n_point,
    int low_dim,
    int method,
    const DTYPE *Y0,
    unsigned int seed
)
{
    if (method == 1 && Y0 != NULL) {
        // Caller-supplied initialization (init='pca' or an ndarray, computed in Python)
        memcpy(Y, Y0, (size_t)low_dim * n_point * sizeof(DTYPE));
        return;
    }
    if (method == 1) {
        printf("Warning: init_method=1 without init_Y, using random initialization\n");
    }

    srand(seed);
    {
        // Random initialization: N(0, 1e-4) using Box-Muller transform
        for (int i = 0; i < n_point * low_dim; i += 2) {
            // Box-Muller transform to generate pairs of normal random variables
            DTYPE u1 = (DTYPE)rand() / RAND_MAX;
            DTYPE u2 = (DTYPE)rand() / RAND_MAX;

            // Avoid log(0)
            if (u1 < 1e-10) u1 = 1e-10;

            DTYPE z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            DTYPE z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

            // Scale to N(0, 1e-4)
            Y[i] = 1e-4 * z0;
            if (i + 1 < n_point * low_dim) {
                Y[i + 1] = 1e-4 * z1;
            }
        }
    }
}
