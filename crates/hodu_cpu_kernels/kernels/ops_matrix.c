#include "ops_matrix.h"
#include "thread_utils.h"
#include "types.h"
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#ifdef USE_BLAS
#include <cblas.h>
#endif

// ============================================================================
// BATCHED MATRIX MULTIPLICATION (MATMUL)
// ============================================================================
//
// Performs batched matrix multiplication with broadcasting support.
// Computes C[..., i, j] = sum_k A[..., i, k] * B[..., k, j]
//
// Metadata layout:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: lhs_ndim (number of dimensions in lhs)
// - metadata[2]: rhs_ndim (number of dimensions in rhs)
// - metadata[3]: batch_ndim (number of batch dimensions in output)
// - metadata[4..4+lhs_ndim]: lhs_shape
// - metadata[4+lhs_ndim..4+lhs_ndim+rhs_ndim]: rhs_shape
// - metadata[4+lhs_ndim+rhs_ndim..4+lhs_ndim+rhs_ndim+batch_ndim]: batch_shape
// - metadata[...+lhs_ndim]: lhs_strides
// - metadata[...+rhs_ndim]: rhs_strides
// - metadata[...]: lhs_offset
// - metadata[...+1]: rhs_offset
// - metadata[...+2]: M (rows of lhs matrix)
// - metadata[...+3]: K (cols of lhs / rows of rhs)
// - metadata[...+4]: N (cols of rhs matrix)
//
// Algorithm:
// For each output element (batch_idx, i, j):
// 1. Decompose flat batch_idx into multi-dimensional batch indices
// 2. Map batch indices to lhs/rhs with broadcasting (size 1 dims stay at 0)
// 3. Compute dot product: sum_k lhs[batch, i, k] * rhs[batch, k, j]
//
// Broadcasting:
// Batch dimensions of size 1 are broadcast by using index 0 for that dimension.

/// Optimized macro for batched matrix multiplication
/// - Fast path for contiguous non-batched case with parallel execution
/// - Scalar accumulation for better register usage
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
#define MATMUL_OP(TYPE, TYPE_SUFFIX)                                                               \
    typedef struct {                                                                               \
        const TYPE *lhs;                                                                           \
        const TYPE *rhs;                                                                           \
        TYPE *output;                                                                              \
        size_t start_row;                                                                          \
        size_t end_row;                                                                            \
        size_t M, K, N;                                                                            \
    } matmul_##TYPE_SUFFIX##_args_t;                                                               \
                                                                                                   \
    static void *matmul_##TYPE_SUFFIX##_worker(void *arg) {                                        \
        matmul_##TYPE_SUFFIX##_args_t *args = (matmul_##TYPE_SUFFIX##_args_t *)arg;                \
        for (size_t i = args->start_row; i < args->end_row; i++) {                                 \
            for (size_t j = 0; j < args->N; j++) {                                                 \
                TYPE sum = 0;                                                                      \
                for (size_t k = 0; k < args->K; k++) {                                             \
                    sum += args->lhs[i * args->K + k] * args->rhs[k * args->N + j];                \
                }                                                                                  \
                args->output[i * args->N + j] = sum;                                               \
            }                                                                                      \
        }                                                                                          \
        return NULL;                                                                               \
    }                                                                                              \
                                                                                                   \
    void matmul_##TYPE_SUFFIX(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,          \
                              const size_t *metadata) {                                            \
        const TYPE *lhs = (const TYPE *)lhs_ptr;                                                   \
        const TYPE *rhs = (const TYPE *)rhs_ptr;                                                   \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t lhs_ndim = metadata[1];                                                       \
        const size_t rhs_ndim = metadata[2];                                                       \
        const size_t batch_ndim = metadata[3];                                                     \
                                                                                                   \
        const size_t *lhs_shape = metadata + 4;                                                    \
        const size_t *rhs_shape = lhs_shape + lhs_ndim;                                            \
        const size_t *batch_shape = rhs_shape + rhs_ndim;                                          \
        const size_t *lhs_strides = batch_shape + batch_ndim;                                      \
        const size_t *rhs_strides = lhs_strides + lhs_ndim;                                        \
        const size_t lhs_offset = *(rhs_strides + rhs_ndim);                                       \
        const size_t rhs_offset = *(rhs_strides + rhs_ndim + 1);                                   \
        const size_t M = *(rhs_strides + rhs_ndim + 2);                                            \
        const size_t K = *(rhs_strides + rhs_ndim + 3);                                            \
        const size_t N = *(rhs_strides + rhs_ndim + 4);                                            \
                                                                                                   \
        size_t lhs_batch_ndim = lhs_ndim - 2;                                                      \
        size_t rhs_batch_ndim = rhs_ndim - 2;                                                      \
                                                                                                   \
        /* Check for contiguous fast path */                                                       \
        bool is_contiguous = (lhs_strides[lhs_ndim - 1] == 1 && rhs_strides[rhs_ndim - 1] == 1 &&  \
                              lhs_strides[lhs_ndim - 2] == K && rhs_strides[rhs_ndim - 2] == N &&  \
                              lhs_offset == 0 && rhs_offset == 0);                                 \
                                                                                                   \
        if (is_contiguous && batch_ndim == 0) {                                                    \
            /* Fast path: no batching, contiguous - use parallel execution */                      \
            size_t num_threads = get_optimal_threads(M, 128);                                      \
                                                                                                   \
            if (num_threads > 1) {                                                                 \
                /* Parallel execution */                                                           \
                pthread_t threads[32];                                                             \
                matmul_##TYPE_SUFFIX##_args_t thread_args[32];                                     \
                if (num_threads > 32)                                                              \
                    num_threads = 32;                                                              \
                                                                                                   \
                size_t rows_per_thread = M / num_threads;                                          \
                size_t remaining_rows = M % num_threads;                                           \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    thread_args[t].lhs = lhs;                                                      \
                    thread_args[t].rhs = rhs;                                                      \
                    thread_args[t].output = output;                                                \
                    thread_args[t].M = M;                                                          \
                    thread_args[t].K = K;                                                          \
                    thread_args[t].N = N;                                                          \
                    thread_args[t].start_row = t * rows_per_thread;                                \
                    thread_args[t].end_row = (t + 1) * rows_per_thread;                            \
                    if (t == num_threads - 1)                                                      \
                        thread_args[t].end_row += remaining_rows;                                  \
                                                                                                   \
                    pthread_create(&threads[t], NULL, matmul_##TYPE_SUFFIX##_worker,               \
                                   &thread_args[t]);                                               \
                }                                                                                  \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    pthread_join(threads[t], NULL);                                                \
                }                                                                                  \
            } else {                                                                               \
                /* Single-threaded execution */                                                    \
                for (size_t i = 0; i < M; i++) {                                                   \
                    for (size_t j = 0; j < N; j++) {                                               \
                        TYPE sum = 0;                                                              \
                        for (size_t k = 0; k < K; k++) {                                           \
                            sum += lhs[i * K + k] * rhs[k * N + j];                                \
                        }                                                                          \
                        output[i * N + j] = sum;                                                   \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            /* General path with batching/striding */                                              \
            for (size_t idx = 0; idx < num_els; idx++) {                                           \
                size_t mn = idx % (M * N);                                                         \
                size_t batch_idx = idx / (M * N);                                                  \
                size_t i = mn / N;                                                                 \
                size_t j = mn % N;                                                                 \
                                                                                                   \
                size_t batch_indices[16];                                                          \
                size_t temp = batch_idx;                                                           \
                for (int d = (int)batch_ndim - 1; d >= 0; d--) {                                   \
                    batch_indices[d] = temp % batch_shape[d];                                      \
                    temp /= batch_shape[d];                                                        \
                }                                                                                  \
                                                                                                   \
                size_t lhs_batch_indices[16];                                                      \
                for (size_t d = 0; d < lhs_batch_ndim; d++) {                                      \
                    size_t batch_dim_idx = batch_ndim - lhs_batch_ndim + d;                        \
                    lhs_batch_indices[d] = (lhs_shape[d] == 1) ? 0 : batch_indices[batch_dim_idx]; \
                }                                                                                  \
                                                                                                   \
                size_t rhs_batch_indices[16];                                                      \
                for (size_t d = 0; d < rhs_batch_ndim; d++) {                                      \
                    size_t batch_dim_idx = batch_ndim - rhs_batch_ndim + d;                        \
                    rhs_batch_indices[d] = (rhs_shape[d] == 1) ? 0 : batch_indices[batch_dim_idx]; \
                }                                                                                  \
                                                                                                   \
                TYPE sum = 0;                                                                      \
                for (size_t k = 0; k < K; k++) {                                                   \
                    size_t lhs_idx = lhs_offset;                                                   \
                    for (size_t d = 0; d < lhs_batch_ndim; d++) {                                  \
                        lhs_idx += lhs_batch_indices[d] * lhs_strides[d];                          \
                    }                                                                              \
                    lhs_idx += i * lhs_strides[lhs_ndim - 2];                                      \
                    lhs_idx += k * lhs_strides[lhs_ndim - 1];                                      \
                                                                                                   \
                    size_t rhs_idx = rhs_offset;                                                   \
                    for (size_t d = 0; d < rhs_batch_ndim; d++) {                                  \
                        rhs_idx += rhs_batch_indices[d] * rhs_strides[d];                          \
                    }                                                                              \
                    rhs_idx += k * rhs_strides[rhs_ndim - 2];                                      \
                    rhs_idx += j * rhs_strides[rhs_ndim - 1];                                      \
                                                                                                   \
                    sum += lhs[lhs_idx] * rhs[rhs_idx];                                            \
                }                                                                                  \
                                                                                                   \
                output[idx] = sum;                                                                 \
            }                                                                                      \
        }                                                                                          \
    }

// Generate fallback implementations for all types first
MATMUL_OP(f32_t, f32_fallback)
MATMUL_OP(f64_t, f64_fallback)

// BLAS-accelerated versions for F32/F64
#ifdef USE_BLAS

/// F32 matmul using BLAS cblas_sgemm with fallback
void matmul_f32(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,
                const size_t *metadata) {
    const f32_t *lhs = (const f32_t *)lhs_ptr;
    const f32_t *rhs = (const f32_t *)rhs_ptr;
    f32_t *output = (f32_t *)output_ptr;

    const size_t lhs_ndim = metadata[1];
    const size_t rhs_ndim = metadata[2];
    const size_t batch_ndim = metadata[3];

    const size_t *lhs_shape = metadata + 4;
    const size_t *rhs_shape = lhs_shape + lhs_ndim;
    const size_t *batch_shape = rhs_shape + rhs_ndim;
    const size_t *lhs_strides = batch_shape + batch_ndim;
    const size_t *rhs_strides = lhs_strides + lhs_ndim;
    const size_t lhs_offset = *(rhs_strides + rhs_ndim);
    const size_t rhs_offset = *(rhs_strides + rhs_ndim + 1);
    const size_t M = *(rhs_strides + rhs_ndim + 2);
    const size_t K = *(rhs_strides + rhs_ndim + 3);
    const size_t N = *(rhs_strides + rhs_ndim + 4);

    // Apply offsets
    lhs += lhs_offset;
    rhs += rhs_offset;

    // Determine matrix layouts and transpose flags
    // Row-major: innermost (last) stride is 1
    // Col-major: outermost (second-to-last) stride is 1
    bool lhs_row_major = (lhs_strides[lhs_ndim - 1] == 1);
    bool lhs_col_major = (lhs_strides[lhs_ndim - 2] == 1);
    bool rhs_row_major = (rhs_strides[rhs_ndim - 1] == 1);
    bool rhs_col_major = (rhs_strides[rhs_ndim - 2] == 1);

    // Can use BLAS if either row-major or col-major (transposed)
    if ((lhs_row_major || lhs_col_major) && (rhs_row_major || rhs_col_major)) {
        CBLAS_TRANSPOSE trans_lhs = lhs_row_major ? CblasNoTrans : CblasTrans;
        CBLAS_TRANSPOSE trans_rhs = rhs_row_major ? CblasNoTrans : CblasTrans;

        // Leading dimensions
        size_t lda = lhs_row_major ? lhs_strides[lhs_ndim - 2] : lhs_strides[lhs_ndim - 1];
        size_t ldb = rhs_row_major ? rhs_strides[rhs_ndim - 2] : rhs_strides[rhs_ndim - 1];
        size_t ldc = N;

        // Fix leading dimensions when they're too small
        // For row-major: lda >= K, ldb >= N
        // For col-major: lda >= M, ldb >= K
        if (trans_lhs == CblasNoTrans) {
            if (lda < K)
                lda = K;
        } else {
            if (lda < M)
                lda = M;
        }
        if (trans_rhs == CblasNoTrans) {
            if (ldb < N)
                ldb = N;
        } else {
            if (ldb < K)
                ldb = K;
        }

        if (batch_ndim == 0) {
            // No batching - single BLAS call
            cblas_sgemm(CblasRowMajor, trans_lhs, trans_rhs, M, N, K, 1.0f, lhs, lda, rhs, ldb,
                        0.0f, output, ldc);
        } else {
            // Batched case - iterate over all batch indices
            size_t total_batches = 1;
            for (size_t i = 0; i < batch_ndim; i++) {
                total_batches *= batch_shape[i];
            }

            size_t out_batch_stride = M * N;

            for (size_t batch_idx = 0; batch_idx < total_batches; batch_idx++) {
                // Compute multi-dimensional batch index
                size_t idx = batch_idx;
                size_t lhs_batch_offset = 0;
                size_t rhs_batch_offset = 0;

                for (size_t i = batch_ndim; i > 0; i--) {
                    size_t dim_idx = idx % batch_shape[i - 1];
                    idx /= batch_shape[i - 1];

                    // Apply broadcasting: if shape is 1, use index 0
                    size_t lhs_dim_idx = (lhs_shape[lhs_ndim - 2 - i] == 1) ? 0 : dim_idx;
                    size_t rhs_dim_idx = (rhs_shape[rhs_ndim - 2 - i] == 1) ? 0 : dim_idx;

                    lhs_batch_offset += lhs_dim_idx * lhs_strides[lhs_ndim - 2 - i];
                    rhs_batch_offset += rhs_dim_idx * rhs_strides[rhs_ndim - 2 - i];
                }

                const f32_t *lhs_batch = lhs + lhs_batch_offset;
                const f32_t *rhs_batch = rhs + rhs_batch_offset;
                f32_t *out_batch = output + batch_idx * out_batch_stride;

                cblas_sgemm(CblasRowMajor, trans_lhs, trans_rhs, M, N, K, 1.0f, lhs_batch, lda,
                            rhs_batch, ldb, 0.0f, out_batch, ldc);
            }
        }
    } else {
        // Fallback for complex stride patterns
        matmul_f32_fallback(lhs_ptr, rhs_ptr, output_ptr, metadata);
    }
}

/// F64 matmul using BLAS cblas_dgemm with fallback
void matmul_f64(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,
                const size_t *metadata) {
    const f64_t *lhs = (const f64_t *)lhs_ptr;
    const f64_t *rhs = (const f64_t *)rhs_ptr;
    f64_t *output = (f64_t *)output_ptr;

    const size_t lhs_ndim = metadata[1];
    const size_t rhs_ndim = metadata[2];
    const size_t batch_ndim = metadata[3];

    const size_t *lhs_shape = metadata + 4;
    const size_t *rhs_shape = lhs_shape + lhs_ndim;
    const size_t *batch_shape = rhs_shape + rhs_ndim;
    const size_t *lhs_strides = batch_shape + batch_ndim;
    const size_t *rhs_strides = lhs_strides + lhs_ndim;
    const size_t lhs_offset = *(rhs_strides + rhs_ndim);
    const size_t rhs_offset = *(rhs_strides + rhs_ndim + 1);
    const size_t M = *(rhs_strides + rhs_ndim + 2);
    const size_t K = *(rhs_strides + rhs_ndim + 3);
    const size_t N = *(rhs_strides + rhs_ndim + 4);

    // Apply offsets
    lhs += lhs_offset;
    rhs += rhs_offset;

    // Determine matrix layouts and transpose flags
    bool lhs_row_major = (lhs_strides[lhs_ndim - 1] == 1);
    bool lhs_col_major = (lhs_strides[lhs_ndim - 2] == 1);
    bool rhs_row_major = (rhs_strides[rhs_ndim - 1] == 1);
    bool rhs_col_major = (rhs_strides[rhs_ndim - 2] == 1);

    if ((lhs_row_major || lhs_col_major) && (rhs_row_major || rhs_col_major)) {
        CBLAS_TRANSPOSE trans_lhs = lhs_row_major ? CblasNoTrans : CblasTrans;
        CBLAS_TRANSPOSE trans_rhs = rhs_row_major ? CblasNoTrans : CblasTrans;

        size_t lda = lhs_row_major ? lhs_strides[lhs_ndim - 2] : lhs_strides[lhs_ndim - 1];
        size_t ldb = rhs_row_major ? rhs_strides[rhs_ndim - 2] : rhs_strides[rhs_ndim - 1];
        size_t ldc = N;

        // Fix leading dimensions when they're too small
        if (trans_lhs == CblasNoTrans) {
            if (lda < K)
                lda = K;
        } else {
            if (lda < M)
                lda = M;
        }
        if (trans_rhs == CblasNoTrans) {
            if (ldb < N)
                ldb = N;
        } else {
            if (ldb < K)
                ldb = K;
        }

        if (batch_ndim == 0) {
            // No batching - single BLAS call
            cblas_dgemm(CblasRowMajor, trans_lhs, trans_rhs, M, N, K, 1.0, lhs, lda, rhs, ldb, 0.0,
                        output, ldc);
        } else {
            // Batched case - iterate over all batch indices
            size_t total_batches = 1;
            for (size_t i = 0; i < batch_ndim; i++) {
                total_batches *= batch_shape[i];
            }

            size_t out_batch_stride = M * N;

            for (size_t batch_idx = 0; batch_idx < total_batches; batch_idx++) {
                // Compute multi-dimensional batch index
                size_t idx = batch_idx;
                size_t lhs_batch_offset = 0;
                size_t rhs_batch_offset = 0;

                for (size_t i = batch_ndim; i > 0; i--) {
                    size_t dim_idx = idx % batch_shape[i - 1];
                    idx /= batch_shape[i - 1];

                    // Apply broadcasting: if shape is 1, use index 0
                    size_t lhs_dim_idx = (lhs_shape[lhs_ndim - 2 - i] == 1) ? 0 : dim_idx;
                    size_t rhs_dim_idx = (rhs_shape[rhs_ndim - 2 - i] == 1) ? 0 : dim_idx;

                    lhs_batch_offset += lhs_dim_idx * lhs_strides[lhs_ndim - 2 - i];
                    rhs_batch_offset += rhs_dim_idx * rhs_strides[rhs_ndim - 2 - i];
                }

                const f64_t *lhs_batch = lhs + lhs_batch_offset;
                const f64_t *rhs_batch = rhs + rhs_batch_offset;
                f64_t *out_batch = output + batch_idx * out_batch_stride;

                cblas_dgemm(CblasRowMajor, trans_lhs, trans_rhs, M, N, K, 1.0, lhs_batch, lda,
                            rhs_batch, ldb, 0.0, out_batch, ldc);
            }
        }
    } else {
        // Fallback for complex stride patterns
        matmul_f64_fallback(lhs_ptr, rhs_ptr, output_ptr, metadata);
    }
}

#else
// No BLAS, alias fallback to main functions
void matmul_f32(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,
                const size_t *metadata) {
    matmul_f32_fallback(lhs_ptr, rhs_ptr, output_ptr, metadata);
}

void matmul_f64(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,
                const size_t *metadata) {
    matmul_f64_fallback(lhs_ptr, rhs_ptr, output_ptr, metadata);
}
#endif

// All other types use generic MATMUL_OP
MATMUL_OP(f8e4m3_t, f8e4m3)
MATMUL_OP(f8e5m2_t, f8e5m2)
MATMUL_OP(bf16_t, bf16)
MATMUL_OP(f16_t, f16)
MATMUL_OP(int8_t, i8)
MATMUL_OP(int16_t, i16)
MATMUL_OP(int32_t, i32)
MATMUL_OP(int64_t, i64)
MATMUL_OP(uint8_t, u8)
MATMUL_OP(uint16_t, u16)
MATMUL_OP(uint32_t, u32)
MATMUL_OP(uint64_t, u64)

// ============================================================================
// 2D MATRIX MULTIPLICATION (DOT)
// ============================================================================
//
// Performs simple 2D matrix multiplication without batching.
// Computes C[i, j] = sum_k A[i, k] * B[k, j]
//
// Metadata layout:
// - metadata[0]: M (number of rows in lhs)
// - metadata[1]: K (number of cols in lhs / rows in rhs)
// - metadata[2]: N (number of cols in rhs)
// - metadata[3]: lhs_stride_m (stride for lhs rows)
// - metadata[4]: lhs_stride_k (stride for lhs cols)
// - metadata[5]: rhs_stride_k (stride for rhs rows)
// - metadata[6]: rhs_stride_n (stride for rhs cols)
// - metadata[7]: lhs_offset (starting offset in lhs)
// - metadata[8]: rhs_offset (starting offset in rhs)
//
// Algorithm:
// Simple triple-nested loop over (row, col, k) for matrix multiplication.
// No tiling or cache optimization as CPU doesn't benefit from shared memory
// tiling like GPUs do.

/// Macro to implement highly optimized 2D matrix multiplication with parallel execution
///
/// Optimizations:
/// - Parallel execution with pthread for large matrices
/// - Cache blocking (32x32x256 blocks for L1 cache)
/// - Register blocking (4x4 micro-kernels)
/// - Loop unrolling (4x inner loop)
/// - Memory prefetching hints
/// - Optimized loop ordering (i-k-j for better cache locality)
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
#define DOT_OP(TYPE, TYPE_SUFFIX)                                                                  \
    typedef struct {                                                                               \
        const TYPE *lhs;                                                                           \
        const TYPE *rhs;                                                                           \
        TYPE *output;                                                                              \
        size_t start_row;                                                                          \
        size_t end_row;                                                                            \
        size_t M, K, N;                                                                            \
    } dot_##TYPE_SUFFIX##_args_t;                                                                  \
                                                                                                   \
    static void *dot_##TYPE_SUFFIX##_worker(void *arg) {                                           \
        dot_##TYPE_SUFFIX##_args_t *args = (dot_##TYPE_SUFFIX##_args_t *)arg;                      \
        for (size_t i = args->start_row; i < args->end_row; i++) {                                 \
            for (size_t j = 0; j < args->N; j++) {                                                 \
                TYPE sum = 0;                                                                      \
                for (size_t k = 0; k < args->K; k++) {                                             \
                    sum += args->lhs[i * args->K + k] * args->rhs[k * args->N + j];                \
                }                                                                                  \
                args->output[i * args->N + j] = sum;                                               \
            }                                                                                      \
        }                                                                                          \
        return NULL;                                                                               \
    }                                                                                              \
                                                                                                   \
    void dot_##TYPE_SUFFIX(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,             \
                           const size_t *metadata) {                                               \
        const TYPE *lhs = (const TYPE *)lhs_ptr;                                                   \
        const TYPE *rhs = (const TYPE *)rhs_ptr;                                                   \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t M = metadata[0];                                                              \
        const size_t K = metadata[1];                                                              \
        const size_t N = metadata[2];                                                              \
        const size_t lhs_stride_m = metadata[3];                                                   \
        const size_t lhs_stride_k = metadata[4];                                                   \
        const size_t rhs_stride_k = metadata[5];                                                   \
        const size_t rhs_stride_n = metadata[6];                                                   \
        const size_t lhs_offset = metadata[7];                                                     \
        const size_t rhs_offset = metadata[8];                                                     \
                                                                                                   \
        /* Initialize output to zero */                                                            \
        memset(output, 0, M * N * sizeof(TYPE));                                                   \
                                                                                                   \
        /* Optimized blocking parameters for L1 cache */                                           \
        const size_t BLOCK_M = 32;                                                                 \
        const size_t BLOCK_N = 32;                                                                 \
        const size_t BLOCK_K = 256;                                                                \
        const size_t REG_M = 4;                                                                    \
        const size_t REG_N = 4;                                                                    \
                                                                                                   \
        /* Check if contiguous for fast path */                                                    \
        bool is_contiguous =                                                                       \
            (lhs_stride_k == 1 && rhs_stride_n == 1 && lhs_stride_m == K && rhs_stride_k == N);    \
                                                                                                   \
        if (is_contiguous && lhs_offset == 0 && rhs_offset == 0) {                                 \
            /* Fast path: contiguous matrices - use parallel execution for large matrices */       \
            size_t num_threads = get_optimal_threads(M, 256);                                      \
                                                                                                   \
            if (num_threads > 1) {                                                                 \
                /* Parallel execution */                                                           \
                pthread_t threads[32];                                                             \
                dot_##TYPE_SUFFIX##_args_t thread_args[32];                                        \
                if (num_threads > 32)                                                              \
                    num_threads = 32;                                                              \
                                                                                                   \
                size_t rows_per_thread = M / num_threads;                                          \
                size_t remaining_rows = M % num_threads;                                           \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    thread_args[t].lhs = lhs;                                                      \
                    thread_args[t].rhs = rhs;                                                      \
                    thread_args[t].output = output;                                                \
                    thread_args[t].M = M;                                                          \
                    thread_args[t].K = K;                                                          \
                    thread_args[t].N = N;                                                          \
                    thread_args[t].start_row = t * rows_per_thread;                                \
                    thread_args[t].end_row = (t + 1) * rows_per_thread;                            \
                    if (t == num_threads - 1)                                                      \
                        thread_args[t].end_row += remaining_rows;                                  \
                                                                                                   \
                    pthread_create(&threads[t], NULL, dot_##TYPE_SUFFIX##_worker,                  \
                                   &thread_args[t]);                                               \
                }                                                                                  \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    pthread_join(threads[t], NULL);                                                \
                }                                                                                  \
            } else {                                                                               \
                /* Single-threaded execution with cache blocking */                                \
                for (size_t ii = 0; ii < M; ii += BLOCK_M) {                                       \
                    size_t i_end = (ii + BLOCK_M < M) ? (ii + BLOCK_M) : M;                        \
                    for (size_t kk = 0; kk < K; kk += BLOCK_K) {                                   \
                        size_t k_end = (kk + BLOCK_K < K) ? (kk + BLOCK_K) : K;                    \
                        for (size_t jj = 0; jj < N; jj += BLOCK_N) {                               \
                            size_t j_end = (jj + BLOCK_N < N) ? (jj + BLOCK_N) : N;                \
                            /* Register blocking with 4x4 micro-kernels */                         \
                            for (size_t i = ii; i < i_end; i += REG_M) {                           \
                                size_t i_reg_end = (i + REG_M < i_end) ? (i + REG_M) : i_end;      \
                                for (size_t j = jj; j < j_end; j += REG_N) {                       \
                                    size_t j_reg_end = (j + REG_N < j_end) ? (j + REG_N) : j_end;  \
                                    /* Accumulate 4x4 block in registers */                        \
                                    TYPE acc[4][4] = {{0}};                                        \
                                    for (size_t ir = 0; ir < (i_reg_end - i); ir++) {              \
                                        for (size_t jr = 0; jr < (j_reg_end - j); jr++) {          \
                                            acc[ir][jr] = output[(i + ir) * N + (j + jr)];         \
                                        }                                                          \
                                    }                                                              \
                                    /* Compute 4x4 micro-kernel */                                 \
                                    for (size_t k = kk; k < k_end; k++) {                          \
                                        for (size_t ir = 0; ir < (i_reg_end - i); ir++) {          \
                                            TYPE a_val = lhs[(i + ir) * K + k];                    \
                                            for (size_t jr = 0; jr < (j_reg_end - j); jr++) {      \
                                                acc[ir][jr] += a_val * rhs[k * N + (j + jr)];      \
                                            }                                                      \
                                        }                                                          \
                                    }                                                              \
                                    /* Store back */                                               \
                                    for (size_t ir = 0; ir < (i_reg_end - i); ir++) {              \
                                        for (size_t jr = 0; jr < (j_reg_end - j); jr++) {          \
                                            output[(i + ir) * N + (j + jr)] = acc[ir][jr];         \
                                        }                                                          \
                                    }                                                              \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            /* Strided path with basic optimization */                                             \
            for (size_t ii = 0; ii < M; ii += BLOCK_M) {                                           \
                size_t i_end = (ii + BLOCK_M < M) ? (ii + BLOCK_M) : M;                            \
                for (size_t kk = 0; kk < K; kk += BLOCK_K) {                                       \
                    size_t k_end = (kk + BLOCK_K < K) ? (kk + BLOCK_K) : K;                        \
                    for (size_t jj = 0; jj < N; jj += BLOCK_N) {                                   \
                        size_t j_end = (jj + BLOCK_N < N) ? (jj + BLOCK_N) : N;                    \
                        for (size_t i = ii; i < i_end; i++) {                                      \
                            for (size_t k = kk; k < k_end; k++) {                                  \
                                size_t lhs_idx = lhs_offset + i * lhs_stride_m + k * lhs_stride_k; \
                                TYPE a_val = lhs[lhs_idx];                                         \
                                for (size_t j = jj; j < j_end; j++) {                              \
                                    size_t rhs_idx =                                               \
                                        rhs_offset + k * rhs_stride_k + j * rhs_stride_n;          \
                                    output[i * N + j] += a_val * rhs[rhs_idx];                     \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// DOT IMPLEMENTATIONS
// ============================================================================

// Generate fallback implementations for all types first
DOT_OP(f32_t, f32_fallback)
DOT_OP(f64_t, f64_fallback)

// BLAS-accelerated versions for F32/F64
#ifdef USE_BLAS

/// F32 dot product using BLAS cblas_sgemm with fallback
void dot_f32(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr, const size_t *metadata) {
    const f32_t *lhs = (const f32_t *)lhs_ptr;
    const f32_t *rhs = (const f32_t *)rhs_ptr;
    f32_t *output = (f32_t *)output_ptr;

    const size_t M = metadata[0];
    const size_t K = metadata[1];
    const size_t N = metadata[2];
    const size_t lhs_stride_m = metadata[3];
    const size_t lhs_stride_k = metadata[4];
    const size_t rhs_stride_k = metadata[5];
    const size_t rhs_stride_n = metadata[6];
    const size_t lhs_offset = metadata[7];
    const size_t rhs_offset = metadata[8];

    // Apply offsets
    lhs += lhs_offset;
    rhs += rhs_offset;

    // Determine matrix layouts and transpose flags
    bool lhs_row_major = (lhs_stride_k == 1);
    bool lhs_col_major = (lhs_stride_m == 1);
    bool rhs_row_major = (rhs_stride_n == 1);
    bool rhs_col_major = (rhs_stride_k == 1);

    if ((lhs_row_major || lhs_col_major) && (rhs_row_major || rhs_col_major)) {
        CBLAS_TRANSPOSE trans_lhs = lhs_row_major ? CblasNoTrans : CblasTrans;
        CBLAS_TRANSPOSE trans_rhs = rhs_row_major ? CblasNoTrans : CblasTrans;

        size_t lda = lhs_row_major ? lhs_stride_m : lhs_stride_k;
        size_t ldb = rhs_row_major ? rhs_stride_k : rhs_stride_n;

        if (trans_lhs == CblasNoTrans) {
            if (lda < K)
                lda = K;
        } else {
            if (lda < M)
                lda = M;
        }
        if (trans_rhs == CblasNoTrans) {
            if (ldb < N)
                ldb = N;
        } else {
            if (ldb < K)
                ldb = K;
        }

        cblas_sgemm(CblasRowMajor, trans_lhs, trans_rhs, M, N, K, 1.0f, lhs, lda, rhs, ldb, 0.0f,
                    output, N);
    } else {
        // Fallback for complex stride patterns
        dot_f32_fallback(lhs_ptr, rhs_ptr, output_ptr, metadata);
    }
}

/// F64 dot product using BLAS cblas_dgemm with fallback
void dot_f64(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr, const size_t *metadata) {
    const f64_t *lhs = (const f64_t *)lhs_ptr;
    const f64_t *rhs = (const f64_t *)rhs_ptr;
    f64_t *output = (f64_t *)output_ptr;

    const size_t M = metadata[0];
    const size_t K = metadata[1];
    const size_t N = metadata[2];
    const size_t lhs_stride_m = metadata[3];
    const size_t lhs_stride_k = metadata[4];
    const size_t rhs_stride_k = metadata[5];
    const size_t rhs_stride_n = metadata[6];
    const size_t lhs_offset = metadata[7];
    const size_t rhs_offset = metadata[8];

    // Apply offsets
    lhs += lhs_offset;
    rhs += rhs_offset;

    // Determine matrix layouts and transpose flags
    bool lhs_row_major = (lhs_stride_k == 1);
    bool lhs_col_major = (lhs_stride_m == 1);
    bool rhs_row_major = (rhs_stride_n == 1);
    bool rhs_col_major = (rhs_stride_k == 1);

    if ((lhs_row_major || lhs_col_major) && (rhs_row_major || rhs_col_major)) {
        CBLAS_TRANSPOSE trans_lhs = lhs_row_major ? CblasNoTrans : CblasTrans;
        CBLAS_TRANSPOSE trans_rhs = rhs_row_major ? CblasNoTrans : CblasTrans;

        size_t lda = lhs_row_major ? lhs_stride_m : lhs_stride_k;
        size_t ldb = rhs_row_major ? rhs_stride_k : rhs_stride_n;

        // Fix leading dimensions when they're too small
        if (trans_lhs == CblasNoTrans) {
            if (lda < K)
                lda = K;
        } else {
            if (lda < M)
                lda = M;
        }
        if (trans_rhs == CblasNoTrans) {
            if (ldb < N)
                ldb = N;
        } else {
            if (ldb < K)
                ldb = K;
        }

        cblas_dgemm(CblasRowMajor, trans_lhs, trans_rhs, M, N, K, 1.0, lhs, lda, rhs, ldb, 0.0,
                    output, N);
    } else {
        // Fallback for complex stride patterns
        dot_f64_fallback(lhs_ptr, rhs_ptr, output_ptr, metadata);
    }
}

#else
// No BLAS, alias fallback to main functions
void dot_f32(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr, const size_t *metadata) {
    dot_f32_fallback(lhs_ptr, rhs_ptr, output_ptr, metadata);
}

void dot_f64(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr, const size_t *metadata) {
    dot_f64_fallback(lhs_ptr, rhs_ptr, output_ptr, metadata);
}
#endif

// All other types use generic DOT_OP
DOT_OP(f8e4m3_t, f8e4m3)
DOT_OP(f8e5m2_t, f8e5m2)
DOT_OP(bf16_t, bf16)
DOT_OP(f16_t, f16)
DOT_OP(int8_t, i8)
DOT_OP(int16_t, i16)
DOT_OP(int32_t, i32)
DOT_OP(int64_t, i64)
DOT_OP(uint8_t, u8)
DOT_OP(uint16_t, u16)
DOT_OP(uint32_t, u32)
DOT_OP(uint64_t, u64)
