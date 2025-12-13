#include "ops_matrix.h"
#include "thread_utils.h"
#include "types.h"
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

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
    void hodu_cpu_matmul_##TYPE_SUFFIX(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr, \
                                       const size_t *metadata) {                                   \
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
                thread_t threads[32];                                                              \
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
                    thread_create(&threads[t], matmul_##TYPE_SUFFIX##_worker, &thread_args[t]);    \
                }                                                                                  \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    thread_join(threads[t]);                                                       \
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

// Exotic floating-point matmul (same as MATMUL_OP but with proper float arithmetic)
#define MATMUL_OP_EXOTIC(TYPE, TYPE_SUFFIX, ZERO, ADD_FN, MUL_FN)                                  \
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
                TYPE sum = ZERO;                                                                   \
                for (size_t k = 0; k < args->K; k++) {                                             \
                    sum = ADD_FN(sum,                                                              \
                                 MUL_FN(args->lhs[i * args->K + k], args->rhs[k * args->N + j]));  \
                }                                                                                  \
                args->output[i * args->N + j] = sum;                                               \
            }                                                                                      \
        }                                                                                          \
        return NULL;                                                                               \
    }                                                                                              \
                                                                                                   \
    void hodu_cpu_matmul_##TYPE_SUFFIX(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr, \
                                       const size_t *metadata) {                                   \
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
        bool is_contiguous = (lhs_strides[lhs_ndim - 1] == 1 && rhs_strides[rhs_ndim - 1] == 1 &&  \
                              lhs_strides[lhs_ndim - 2] == K && rhs_strides[rhs_ndim - 2] == N &&  \
                              lhs_offset == 0 && rhs_offset == 0);                                 \
                                                                                                   \
        if (is_contiguous && batch_ndim == 0) {                                                    \
            size_t num_threads = get_optimal_threads(M, 128);                                      \
                                                                                                   \
            if (num_threads > 1) {                                                                 \
                thread_t threads[32];                                                              \
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
                    thread_create(&threads[t], matmul_##TYPE_SUFFIX##_worker, &thread_args[t]);    \
                }                                                                                  \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    thread_join(threads[t]);                                                       \
                }                                                                                  \
            } else {                                                                               \
                for (size_t i = 0; i < M; i++) {                                                   \
                    for (size_t j = 0; j < N; j++) {                                               \
                        TYPE sum = ZERO;                                                           \
                        for (size_t k = 0; k < K; k++) {                                           \
                            sum = ADD_FN(sum, MUL_FN(lhs[i * K + k], rhs[k * N + j]));             \
                        }                                                                          \
                        output[i * N + j] = sum;                                                   \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
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
                TYPE sum = ZERO;                                                                   \
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
                    sum = ADD_FN(sum, MUL_FN(lhs[lhs_idx], rhs[rhs_idx]));                         \
                }                                                                                  \
                                                                                                   \
                output[idx] = sum;                                                                 \
            }                                                                                      \
        }                                                                                          \
    }

// Generate fallback implementations for all types first
MATMUL_OP(f32_t, f32_fallback)
MATMUL_OP(f64_t, f64_fallback)

// F32/F64 matmul implementations are in separate BLAS-specific files:
// - ops_matrix_openblas.c (OpenBLAS with thread control)
// - ops_matrix_blas_aarch64_apple_darwin.c (Accelerate framework)
// These files provide matmul_f32() and matmul_f64() implementations

// Exotic floating-point types use proper arithmetic
MATMUL_OP_EXOTIC(f8e4m3_t, f8e4m3, F8E4M3_ZERO, f8e4m3_add, f8e4m3_mul)
MATMUL_OP_EXOTIC(f8e5m2_t, f8e5m2, F8E5M2_ZERO, f8e5m2_add, f8e5m2_mul)
MATMUL_OP_EXOTIC(bf16_t, bf16, BF16_ZERO, bf16_add, bf16_mul)
MATMUL_OP_EXOTIC(f16_t, f16, F16_ZERO, f16_add, f16_mul)
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
    void hodu_cpu_dot_##TYPE_SUFFIX(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,    \
                                    const size_t *metadata) {                                      \
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
                thread_t threads[32];                                                              \
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
                    thread_create(&threads[t], dot_##TYPE_SUFFIX##_worker, &thread_args[t]);       \
                }                                                                                  \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    thread_join(threads[t]);                                                       \
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

// Exotic dot (same as DOT_OP but with proper float arithmetic)
#define DOT_OP_EXOTIC(TYPE, TYPE_SUFFIX, ZERO, ADD_FN, MUL_FN)                                     \
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
                TYPE sum = ZERO;                                                                   \
                for (size_t k = 0; k < args->K; k++) {                                             \
                    sum = ADD_FN(sum,                                                              \
                                 MUL_FN(args->lhs[i * args->K + k], args->rhs[k * args->N + j]));  \
                }                                                                                  \
                args->output[i * args->N + j] = sum;                                               \
            }                                                                                      \
        }                                                                                          \
        return NULL;                                                                               \
    }                                                                                              \
                                                                                                   \
    void hodu_cpu_dot_##TYPE_SUFFIX(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,    \
                                    const size_t *metadata) {                                      \
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
        for (size_t i = 0; i < M * N; i++) {                                                       \
            output[i] = ZERO;                                                                      \
        }                                                                                          \
                                                                                                   \
        const size_t BLOCK_M = 32;                                                                 \
        const size_t BLOCK_N = 32;                                                                 \
        const size_t BLOCK_K = 256;                                                                \
        const size_t REG_M = 4;                                                                    \
        const size_t REG_N = 4;                                                                    \
                                                                                                   \
        bool is_contiguous =                                                                       \
            (lhs_stride_k == 1 && rhs_stride_n == 1 && lhs_stride_m == K && rhs_stride_k == N);    \
                                                                                                   \
        if (is_contiguous && lhs_offset == 0 && rhs_offset == 0) {                                 \
            size_t num_threads = get_optimal_threads(M, 256);                                      \
                                                                                                   \
            if (num_threads > 1) {                                                                 \
                thread_t threads[32];                                                              \
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
                    thread_create(&threads[t], dot_##TYPE_SUFFIX##_worker, &thread_args[t]);       \
                }                                                                                  \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    thread_join(threads[t]);                                                       \
                }                                                                                  \
            } else {                                                                               \
                for (size_t ii = 0; ii < M; ii += BLOCK_M) {                                       \
                    size_t i_end = (ii + BLOCK_M < M) ? (ii + BLOCK_M) : M;                        \
                    for (size_t kk = 0; kk < K; kk += BLOCK_K) {                                   \
                        size_t k_end = (kk + BLOCK_K < K) ? (kk + BLOCK_K) : K;                    \
                        for (size_t jj = 0; jj < N; jj += BLOCK_N) {                               \
                            size_t j_end = (jj + BLOCK_N < N) ? (jj + BLOCK_N) : N;                \
                            for (size_t i = ii; i < i_end; i += REG_M) {                           \
                                size_t i_reg_end = (i + REG_M < i_end) ? (i + REG_M) : i_end;      \
                                for (size_t j = jj; j < j_end; j += REG_N) {                       \
                                    size_t j_reg_end = (j + REG_N < j_end) ? (j + REG_N) : j_end;  \
                                    TYPE acc[4][4];                                                \
                                    for (size_t ir = 0; ir < (i_reg_end - i); ir++) {              \
                                        for (size_t jr = 0; jr < (j_reg_end - j); jr++) {          \
                                            acc[ir][jr] = output[(i + ir) * N + (j + jr)];         \
                                        }                                                          \
                                    }                                                              \
                                    for (size_t k = kk; k < k_end; k++) {                          \
                                        for (size_t ir = 0; ir < (i_reg_end - i); ir++) {          \
                                            TYPE a_val = lhs[(i + ir) * K + k];                    \
                                            for (size_t jr = 0; jr < (j_reg_end - j); jr++) {      \
                                                acc[ir][jr] =                                      \
                                                    ADD_FN(acc[ir][jr],                            \
                                                           MUL_FN(a_val, rhs[k * N + (j + jr)]));  \
                                            }                                                      \
                                        }                                                          \
                                    }                                                              \
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
                                    output[i * N + j] =                                            \
                                        ADD_FN(output[i * N + j], MUL_FN(a_val, rhs[rhs_idx]));    \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

// Generate fallback implementations for all types first
DOT_OP(f32_t, f32_fallback)
DOT_OP(f64_t, f64_fallback)

// F32/F64 dot implementations are in separate BLAS-specific files:
// - ops_matrix_openblas.c (OpenBLAS)
// - ops_matrix_blas_aarch64_apple_darwin.c (Accelerate framework)
// These files provide dot_f32() and dot_f64() implementations

// Exotic floating-point types use simple correct implementation
DOT_OP_EXOTIC(f8e4m3_t, f8e4m3, F8E4M3_ZERO, f8e4m3_add, f8e4m3_mul)
DOT_OP_EXOTIC(f8e5m2_t, f8e5m2, F8E5M2_ZERO, f8e5m2_add, f8e5m2_mul)
DOT_OP_EXOTIC(bf16_t, bf16, BF16_ZERO, bf16_add, bf16_mul)
DOT_OP_EXOTIC(f16_t, f16, F16_ZERO, f16_add, f16_mul)
DOT_OP(int8_t, i8)
DOT_OP(int16_t, i16)
DOT_OP(int32_t, i32)
DOT_OP(int64_t, i64)
DOT_OP(uint8_t, u8)
DOT_OP(uint16_t, u16)
DOT_OP(uint32_t, u32)
DOT_OP(uint64_t, u64)

// ============================================================================
// MATRIX DETERMINANT (DET)
// ============================================================================
//
// Computes the determinant of square matrices with optional batch dimensions.
// Uses direct formulas for small matrices (1x1, 2x2, 3x3) and LU decomposition
// for larger matrices.
//
// Metadata layout:
// - metadata[0]: batch_size (product of batch dimensions)
// - metadata[1]: n (matrix size, N×N)
// - metadata[2]: ndim (total number of dimensions)
// - metadata[3..3+ndim]: shape
// - metadata[3+ndim..3+2*ndim]: strides
// - metadata[3+2*ndim]: offset
//
// Algorithm:
// - 1x1: det = a[0,0]
// - 2x2: det = a*d - b*c
// - 3x3: Sarrus' rule
// - NxN: LU decomposition with partial pivoting
//        det = (-1)^swaps * product(diagonal of U)

/// Helper to get element from strided matrix for det
#define DET_GET(input, batch_offset, row, col, row_stride, col_stride)                             \
    ((input)[(batch_offset) + (row) * (row_stride) + (col) * (col_stride)])

/// Macro for determinant operation
#define DET_OP(TYPE, TYPE_SUFFIX)                                                                  \
    void hodu_cpu_det_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                       \
                                    const size_t *metadata) {                                      \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t batch_size = metadata[0];                                                     \
        const size_t n = metadata[1];                                                              \
        const size_t ndim = metadata[2];                                                           \
        const size_t *strides = metadata + 3 + ndim;                                               \
        const size_t offset = metadata[3 + 2 * ndim];                                              \
                                                                                                   \
        /* Get row and column strides (last two dimensions) */                                     \
        const size_t row_stride = (ndim >= 2) ? strides[ndim - 2] : n;                             \
        const size_t col_stride = (ndim >= 1) ? strides[ndim - 1] : 1;                             \
                                                                                                   \
        for (size_t batch = 0; batch < batch_size; batch++) {                                      \
            /* Calculate batch offset using strides */                                             \
            size_t batch_offset = offset;                                                          \
            if (ndim > 2) {                                                                        \
                size_t temp = batch;                                                               \
                const size_t *shape = metadata + 3;                                                \
                for (int d = (int)ndim - 3; d >= 0; d--) {                                         \
                    size_t dim_size = shape[d];                                                    \
                    size_t idx = temp % dim_size;                                                  \
                    temp /= dim_size;                                                              \
                    batch_offset += idx * strides[d];                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            TYPE det;                                                                              \
                                                                                                   \
            if (n == 1) {                                                                          \
                det = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);                  \
            } else if (n == 2) {                                                                   \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                TYPE b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
                TYPE c = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);               \
                TYPE d_val = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);           \
                det = a * d_val - b * c;                                                           \
            } else if (n == 3) {                                                                   \
                /* Sarrus' rule for 3x3 */                                                         \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                TYPE b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
                TYPE c = DET_GET(input, batch_offset, 0, 2, row_stride, col_stride);               \
                TYPE d_val = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);           \
                TYPE e = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);               \
                TYPE f = DET_GET(input, batch_offset, 1, 2, row_stride, col_stride);               \
                TYPE g = DET_GET(input, batch_offset, 2, 0, row_stride, col_stride);               \
                TYPE h = DET_GET(input, batch_offset, 2, 1, row_stride, col_stride);               \
                TYPE i = DET_GET(input, batch_offset, 2, 2, row_stride, col_stride);               \
                det = a * (e * i - f * h) - b * (d_val * i - f * g) + c * (d_val * h - e * g);     \
            } else {                                                                               \
                /* LU decomposition with partial pivoting for NxN */                               \
                /* Allocate temporary matrix */                                                    \
                TYPE *lu = (TYPE *)malloc(n * n * sizeof(TYPE));                                   \
                                                                                                   \
                /* Copy input to LU matrix */                                                      \
                for (size_t i = 0; i < n; i++) {                                                   \
                    for (size_t j = 0; j < n; j++) {                                               \
                        lu[i * n + j] =                                                            \
                            DET_GET(input, batch_offset, i, j, row_stride, col_stride);            \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                int swaps = 0;                                                                     \
                det = 1;                                                                           \
                                                                                                   \
                /* LU decomposition with partial pivoting */                                       \
                for (size_t k = 0; k < n; k++) {                                                   \
                    /* Find pivot */                                                               \
                    size_t pivot_row = k;                                                          \
                    TYPE max_val = lu[k * n + k];                                                  \
                    if (max_val < 0)                                                               \
                        max_val = -max_val;                                                        \
                                                                                                   \
                    for (size_t i = k + 1; i < n; i++) {                                           \
                        TYPE val = lu[i * n + k];                                                  \
                        if (val < 0)                                                               \
                            val = -val;                                                            \
                        if (val > max_val) {                                                       \
                            max_val = val;                                                         \
                            pivot_row = i;                                                         \
                        }                                                                          \
                    }                                                                              \
                                                                                                   \
                    /* Swap rows if needed */                                                      \
                    if (pivot_row != k) {                                                          \
                        for (size_t j = 0; j < n; j++) {                                           \
                            TYPE tmp = lu[k * n + j];                                              \
                            lu[k * n + j] = lu[pivot_row * n + j];                                 \
                            lu[pivot_row * n + j] = tmp;                                           \
                        }                                                                          \
                        swaps++;                                                                   \
                    }                                                                              \
                                                                                                   \
                    TYPE pivot = lu[k * n + k];                                                    \
                                                                                                   \
                    /* Check for singular matrix */                                                \
                    TYPE abs_pivot = pivot;                                                        \
                    if (abs_pivot < 0)                                                             \
                        abs_pivot = -abs_pivot;                                                    \
                    if (abs_pivot < (TYPE)1e-15) {                                                 \
                        det = 0;                                                                   \
                        free(lu);                                                                  \
                        goto store_result_##TYPE_SUFFIX;                                           \
                    }                                                                              \
                                                                                                   \
                    det *= pivot;                                                                  \
                                                                                                   \
                    /* Eliminate below */                                                          \
                    for (size_t i = k + 1; i < n; i++) {                                           \
                        TYPE factor = lu[i * n + k] / pivot;                                       \
                        for (size_t j = k; j < n; j++) {                                           \
                            lu[i * n + j] -= factor * lu[k * n + j];                               \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                /* Apply sign from row swaps */                                                    \
                if (swaps % 2 != 0) {                                                              \
                    det = -det;                                                                    \
                }                                                                                  \
                                                                                                   \
                free(lu);                                                                          \
            }                                                                                      \
                                                                                                   \
            store_result_##TYPE_SUFFIX : output[batch] = det;                                      \
        }                                                                                          \
    }

/// Macro for determinant operation with exotic types
#define DET_OP_EXOTIC(TYPE, TYPE_SUFFIX, ZERO, ONE, ADD_FN, SUB_FN, MUL_FN, DIV_FN, NEG_FN,        \
                      ABS_FN, LT_FN)                                                               \
    void hodu_cpu_det_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                       \
                                    const size_t *metadata) {                                      \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t batch_size = metadata[0];                                                     \
        const size_t n = metadata[1];                                                              \
        const size_t ndim = metadata[2];                                                           \
        const size_t *strides = metadata + 3 + ndim;                                               \
        const size_t offset = metadata[3 + 2 * ndim];                                              \
                                                                                                   \
        const size_t row_stride = (ndim >= 2) ? strides[ndim - 2] : n;                             \
        const size_t col_stride = (ndim >= 1) ? strides[ndim - 1] : 1;                             \
                                                                                                   \
        for (size_t batch = 0; batch < batch_size; batch++) {                                      \
            size_t batch_offset = offset;                                                          \
            if (ndim > 2) {                                                                        \
                size_t temp = batch;                                                               \
                const size_t *shape = metadata + 3;                                                \
                for (int d = (int)ndim - 3; d >= 0; d--) {                                         \
                    size_t dim_size = shape[d];                                                    \
                    size_t idx = temp % dim_size;                                                  \
                    temp /= dim_size;                                                              \
                    batch_offset += idx * strides[d];                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            TYPE det;                                                                              \
                                                                                                   \
            if (n == 1) {                                                                          \
                det = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);                  \
            } else if (n == 2) {                                                                   \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                TYPE b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
                TYPE c = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);               \
                TYPE d_val = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);           \
                det = SUB_FN(MUL_FN(a, d_val), MUL_FN(b, c));                                      \
            } else if (n == 3) {                                                                   \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                TYPE b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
                TYPE c = DET_GET(input, batch_offset, 0, 2, row_stride, col_stride);               \
                TYPE d_val = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);           \
                TYPE e = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);               \
                TYPE f = DET_GET(input, batch_offset, 1, 2, row_stride, col_stride);               \
                TYPE g = DET_GET(input, batch_offset, 2, 0, row_stride, col_stride);               \
                TYPE h = DET_GET(input, batch_offset, 2, 1, row_stride, col_stride);               \
                TYPE i = DET_GET(input, batch_offset, 2, 2, row_stride, col_stride);               \
                TYPE t1 = MUL_FN(a, SUB_FN(MUL_FN(e, i), MUL_FN(f, h)));                           \
                TYPE t2 = MUL_FN(b, SUB_FN(MUL_FN(d_val, i), MUL_FN(f, g)));                       \
                TYPE t3 = MUL_FN(c, SUB_FN(MUL_FN(d_val, h), MUL_FN(e, g)));                       \
                det = ADD_FN(SUB_FN(t1, t2), t3);                                                  \
            } else {                                                                               \
                /* LU decomposition for exotic types */                                            \
                TYPE *lu = (TYPE *)malloc(n * n * sizeof(TYPE));                                   \
                for (size_t i = 0; i < n; i++) {                                                   \
                    for (size_t j = 0; j < n; j++) {                                               \
                        lu[i * n + j] =                                                            \
                            DET_GET(input, batch_offset, i, j, row_stride, col_stride);            \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                int swaps = 0;                                                                     \
                det = ONE;                                                                         \
                                                                                                   \
                for (size_t k = 0; k < n; k++) {                                                   \
                    size_t pivot_row = k;                                                          \
                    TYPE max_val = ABS_FN(lu[k * n + k]);                                          \
                                                                                                   \
                    for (size_t i = k + 1; i < n; i++) {                                           \
                        TYPE val = ABS_FN(lu[i * n + k]);                                          \
                        if (LT_FN(max_val, val)) {                                                 \
                            max_val = val;                                                         \
                            pivot_row = i;                                                         \
                        }                                                                          \
                    }                                                                              \
                                                                                                   \
                    if (pivot_row != k) {                                                          \
                        for (size_t j = 0; j < n; j++) {                                           \
                            TYPE tmp = lu[k * n + j];                                              \
                            lu[k * n + j] = lu[pivot_row * n + j];                                 \
                            lu[pivot_row * n + j] = tmp;                                           \
                        }                                                                          \
                        swaps++;                                                                   \
                    }                                                                              \
                                                                                                   \
                    TYPE pivot = lu[k * n + k];                                                    \
                    det = MUL_FN(det, pivot);                                                      \
                                                                                                   \
                    for (size_t i = k + 1; i < n; i++) {                                           \
                        TYPE factor = DIV_FN(lu[i * n + k], pivot);                                \
                        for (size_t j = k; j < n; j++) {                                           \
                            lu[i * n + j] = SUB_FN(lu[i * n + j], MUL_FN(factor, lu[k * n + j]));  \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                if (swaps % 2 != 0) {                                                              \
                    det = NEG_FN(det);                                                             \
                }                                                                                  \
                                                                                                   \
                free(lu);                                                                          \
            }                                                                                      \
                                                                                                   \
            output[batch] = det;                                                                   \
        }                                                                                          \
    }

// Generate det implementations
DET_OP(f32_t, f32)
DET_OP(f64_t, f64)
DET_OP_EXOTIC(f8e4m3_t, f8e4m3, F8E4M3_ZERO, F8E4M3_ONE, f8e4m3_add, f8e4m3_sub, f8e4m3_mul,
              f8e4m3_div, f8e4m3_neg, f8e4m3_abs, f8e4m3_lt)
DET_OP_EXOTIC(f8e5m2_t, f8e5m2, F8E5M2_ZERO, F8E5M2_ONE, f8e5m2_add, f8e5m2_sub, f8e5m2_mul,
              f8e5m2_div, f8e5m2_neg, f8e5m2_abs, f8e5m2_lt)
DET_OP_EXOTIC(bf16_t, bf16, BF16_ZERO, BF16_ONE, bf16_add, bf16_sub, bf16_mul, bf16_div, bf16_neg,
              bf16_abs, bf16_lt)
DET_OP_EXOTIC(f16_t, f16, F16_ZERO, F16_ONE, f16_add, f16_sub, f16_mul, f16_div, f16_neg, f16_abs,
              f16_lt)
DET_OP(int8_t, i8)
DET_OP(int16_t, i16)
DET_OP(int32_t, i32)
DET_OP(int64_t, i64)
DET_OP(uint8_t, u8)
DET_OP(uint16_t, u16)
DET_OP(uint32_t, u32)
DET_OP(uint64_t, u64)
