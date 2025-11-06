#include "ops_matrix.h"
#include "simd_utils.h"
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

/// Macro to implement batched matrix multiplication
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
#define MATMUL_OP(TYPE, TYPE_SUFFIX)                                                               \
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
        for (size_t idx = 0; idx < num_els; idx++) {                                               \
            /* Calculate output position: batch_idx, i, j */                                       \
            size_t mn = idx % (M * N);                                                             \
            size_t batch_idx = idx / (M * N);                                                      \
            size_t i = mn / N;                                                                     \
            size_t j = mn % N;                                                                     \
                                                                                                   \
            /* Compute batch indices from flat batch_idx */                                        \
            size_t batch_indices[16];                                                              \
            size_t temp = batch_idx;                                                               \
            for (int d = (int)batch_ndim - 1; d >= 0; d--) {                                       \
                batch_indices[d] = temp % batch_shape[d];                                          \
                temp /= batch_shape[d];                                                            \
            }                                                                                      \
                                                                                                   \
            /* Map batch indices to lhs indices (with broadcasting) */                             \
            size_t lhs_batch_ndim = lhs_ndim - 2;                                                  \
            size_t lhs_batch_indices[16];                                                          \
            for (size_t d = 0; d < lhs_batch_ndim; d++) {                                          \
                size_t batch_dim_idx = batch_ndim - lhs_batch_ndim + d;                            \
                lhs_batch_indices[d] = (lhs_shape[d] == 1) ? 0 : batch_indices[batch_dim_idx];     \
            }                                                                                      \
                                                                                                   \
            /* Map batch indices to rhs indices (with broadcasting) */                             \
            size_t rhs_batch_ndim = rhs_ndim - 2;                                                  \
            size_t rhs_batch_indices[16];                                                          \
            for (size_t d = 0; d < rhs_batch_ndim; d++) {                                          \
                size_t batch_dim_idx = batch_ndim - rhs_batch_ndim + d;                            \
                rhs_batch_indices[d] = (rhs_shape[d] == 1) ? 0 : batch_indices[batch_dim_idx];     \
            }                                                                                      \
                                                                                                   \
            /* Compute matrix multiplication for this output element */                            \
            TYPE sum = 0;                                                                          \
            for (size_t k = 0; k < K; k++) {                                                       \
                /* Calculate lhs index: batch_indices + [i, k] */                                  \
                size_t lhs_idx = lhs_offset;                                                       \
                for (size_t d = 0; d < lhs_batch_ndim; d++) {                                      \
                    lhs_idx += lhs_batch_indices[d] * lhs_strides[d];                              \
                }                                                                                  \
                lhs_idx += i * lhs_strides[lhs_ndim - 2];                                          \
                lhs_idx += k * lhs_strides[lhs_ndim - 1];                                          \
                                                                                                   \
                /* Calculate rhs index: batch_indices + [k, j] */                                  \
                size_t rhs_idx = rhs_offset;                                                       \
                for (size_t d = 0; d < rhs_batch_ndim; d++) {                                      \
                    rhs_idx += rhs_batch_indices[d] * rhs_strides[d];                              \
                }                                                                                  \
                rhs_idx += k * rhs_strides[rhs_ndim - 2];                                          \
                rhs_idx += j * rhs_strides[rhs_ndim - 1];                                          \
                                                                                                   \
                sum += lhs[lhs_idx] * rhs[rhs_idx];                                                \
            }                                                                                      \
                                                                                                   \
            output[idx] = sum;                                                                     \
        }                                                                                          \
    }

// ============================================================================
// SIMD-OPTIMIZED MATMUL for F32/F64
// ============================================================================

#if SIMD_F32_WIDTH > 1

void matmul_f32(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,
                const size_t *metadata) {
    const f32_t *lhs = (const f32_t *)lhs_ptr;
    const f32_t *rhs = (const f32_t *)rhs_ptr;
    f32_t *output = (f32_t *)output_ptr;

    const size_t num_els = metadata[0];
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

    for (size_t idx = 0; idx < num_els; idx++) {
        size_t mn = idx % (M * N);
        size_t batch_idx = idx / (M * N);
        size_t i = mn / N;
        size_t j = mn % N;

        size_t batch_indices[16];
        size_t temp = batch_idx;
        for (int d = (int)batch_ndim - 1; d >= 0; d--) {
            batch_indices[d] = temp % batch_shape[d];
            temp /= batch_shape[d];
        }

        size_t lhs_batch_ndim = lhs_ndim - 2;
        size_t lhs_batch_indices[16];
        for (size_t d = 0; d < lhs_batch_ndim; d++) {
            size_t batch_dim_idx = batch_ndim - lhs_batch_ndim + d;
            lhs_batch_indices[d] = (lhs_shape[d] == 1) ? 0 : batch_indices[batch_dim_idx];
        }

        size_t rhs_batch_ndim = rhs_ndim - 2;
        size_t rhs_batch_indices[16];
        for (size_t d = 0; d < rhs_batch_ndim; d++) {
            size_t batch_dim_idx = batch_ndim - rhs_batch_ndim + d;
            rhs_batch_indices[d] = (rhs_shape[d] == 1) ? 0 : batch_indices[batch_dim_idx];
        }

        /* SIMD dot product along K dimension */
        simd_f32_t vsum = simd_f32_set1(0.0f);
        size_t k = 0;
        const size_t simd_end = (K / SIMD_F32_WIDTH) * SIMD_F32_WIDTH;

        for (; k < simd_end; k += SIMD_F32_WIDTH) {
            simd_f32_t va, vb;

            /* Load lhs values */
            f32_t lhs_vals[SIMD_F32_WIDTH];
            for (size_t kk = 0; kk < SIMD_F32_WIDTH; kk++) {
                size_t lhs_idx = lhs_offset;
                for (size_t d = 0; d < lhs_batch_ndim; d++) {
                    lhs_idx += lhs_batch_indices[d] * lhs_strides[d];
                }
                lhs_idx += i * lhs_strides[lhs_ndim - 2];
                lhs_idx += (k + kk) * lhs_strides[lhs_ndim - 1];
                lhs_vals[kk] = lhs[lhs_idx];
            }
            va = simd_f32_load(lhs_vals);

            /* Load rhs values */
            f32_t rhs_vals[SIMD_F32_WIDTH];
            for (size_t kk = 0; kk < SIMD_F32_WIDTH; kk++) {
                size_t rhs_idx = rhs_offset;
                for (size_t d = 0; d < rhs_batch_ndim; d++) {
                    rhs_idx += rhs_batch_indices[d] * rhs_strides[d];
                }
                rhs_idx += (k + kk) * rhs_strides[rhs_ndim - 2];
                rhs_idx += j * rhs_strides[rhs_ndim - 1];
                rhs_vals[kk] = rhs[rhs_idx];
            }
            vb = simd_f32_load(rhs_vals);

            vsum = simd_f32_fmadd(va, vb, vsum);
        }

        /* Horizontal reduction + scalar remainder */
        f32_t sum = simd_f32_reduce_add(vsum);

        for (; k < K; k++) {
            size_t lhs_idx = lhs_offset;
            for (size_t d = 0; d < lhs_batch_ndim; d++) {
                lhs_idx += lhs_batch_indices[d] * lhs_strides[d];
            }
            lhs_idx += i * lhs_strides[lhs_ndim - 2];
            lhs_idx += k * lhs_strides[lhs_ndim - 1];

            size_t rhs_idx = rhs_offset;
            for (size_t d = 0; d < rhs_batch_ndim; d++) {
                rhs_idx += rhs_batch_indices[d] * rhs_strides[d];
            }
            rhs_idx += k * rhs_strides[rhs_ndim - 2];
            rhs_idx += j * rhs_strides[rhs_ndim - 1];

            sum += lhs[lhs_idx] * rhs[rhs_idx];
        }

        output[idx] = sum;
    }
}

#else
MATMUL_OP(f32_t, f32)
#endif

#if SIMD_F64_WIDTH > 1

void matmul_f64(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,
                const size_t *metadata) {
    const f64_t *lhs = (const f64_t *)lhs_ptr;
    const f64_t *rhs = (const f64_t *)rhs_ptr;
    f64_t *output = (f64_t *)output_ptr;

    const size_t num_els = metadata[0];
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

    for (size_t idx = 0; idx < num_els; idx++) {
        size_t mn = idx % (M * N);
        size_t batch_idx = idx / (M * N);
        size_t i = mn / N;
        size_t j = mn % N;

        size_t batch_indices[16];
        size_t temp = batch_idx;
        for (int d = (int)batch_ndim - 1; d >= 0; d--) {
            batch_indices[d] = temp % batch_shape[d];
            temp /= batch_shape[d];
        }

        size_t lhs_batch_ndim = lhs_ndim - 2;
        size_t lhs_batch_indices[16];
        for (size_t d = 0; d < lhs_batch_ndim; d++) {
            size_t batch_dim_idx = batch_ndim - lhs_batch_ndim + d;
            lhs_batch_indices[d] = (lhs_shape[d] == 1) ? 0 : batch_indices[batch_dim_idx];
        }

        size_t rhs_batch_ndim = rhs_ndim - 2;
        size_t rhs_batch_indices[16];
        for (size_t d = 0; d < rhs_batch_ndim; d++) {
            size_t batch_dim_idx = batch_ndim - rhs_batch_ndim + d;
            rhs_batch_indices[d] = (rhs_shape[d] == 1) ? 0 : batch_indices[batch_dim_idx];
        }

        simd_f64_t vsum = simd_f64_set1(0.0);
        size_t k = 0;
        const size_t simd_end = (K / SIMD_F64_WIDTH) * SIMD_F64_WIDTH;

        for (; k < simd_end; k += SIMD_F64_WIDTH) {
            simd_f64_t va, vb;

            f64_t lhs_vals[SIMD_F64_WIDTH];
            for (size_t kk = 0; kk < SIMD_F64_WIDTH; kk++) {
                size_t lhs_idx = lhs_offset;
                for (size_t d = 0; d < lhs_batch_ndim; d++) {
                    lhs_idx += lhs_batch_indices[d] * lhs_strides[d];
                }
                lhs_idx += i * lhs_strides[lhs_ndim - 2];
                lhs_idx += (k + kk) * lhs_strides[lhs_ndim - 1];
                lhs_vals[kk] = lhs[lhs_idx];
            }
            va = simd_f64_load(lhs_vals);

            f64_t rhs_vals[SIMD_F64_WIDTH];
            for (size_t kk = 0; kk < SIMD_F64_WIDTH; kk++) {
                size_t rhs_idx = rhs_offset;
                for (size_t d = 0; d < rhs_batch_ndim; d++) {
                    rhs_idx += rhs_batch_indices[d] * rhs_strides[d];
                }
                rhs_idx += (k + kk) * rhs_strides[rhs_ndim - 2];
                rhs_idx += j * rhs_strides[rhs_ndim - 1];
                rhs_vals[kk] = rhs[rhs_idx];
            }
            vb = simd_f64_load(rhs_vals);

            vsum = simd_f64_fmadd(va, vb, vsum);
        }

        f64_t sum = simd_f64_reduce_add(vsum);

        for (; k < K; k++) {
            size_t lhs_idx = lhs_offset;
            for (size_t d = 0; d < lhs_batch_ndim; d++) {
                lhs_idx += lhs_batch_indices[d] * lhs_strides[d];
            }
            lhs_idx += i * lhs_strides[lhs_ndim - 2];
            lhs_idx += k * lhs_strides[lhs_ndim - 1];

            size_t rhs_idx = rhs_offset;
            for (size_t d = 0; d < rhs_batch_ndim; d++) {
                rhs_idx += rhs_batch_indices[d] * rhs_strides[d];
            }
            rhs_idx += k * rhs_strides[rhs_ndim - 2];
            rhs_idx += j * rhs_strides[rhs_ndim - 1];

            sum += lhs[lhs_idx] * rhs[rhs_idx];
        }

        output[idx] = sum;
    }
}

#else
MATMUL_OP(f64_t, f64)
#endif

// Non-SIMD types use regular MATMUL_OP
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

/// Macro to implement 2D matrix multiplication with cache blocking
///
/// Uses tiling/blocking to improve cache locality. Cache-aware algorithm
/// processes matrix in blocks that fit in L1/L2 cache, significantly
/// reducing cache misses.
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
#define DOT_OP(TYPE, TYPE_SUFFIX)                                                                  \
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
        for (size_t i = 0; i < M * N; i++) {                                                       \
            output[i] = 0;                                                                         \
        }                                                                                          \
                                                                                                   \
        /* Cache blocking parameters - tuned for L1/L2 cache */                                    \
        const size_t BLOCK_M = 64;                                                                 \
        const size_t BLOCK_N = 64;                                                                 \
        const size_t BLOCK_K = 64;                                                                 \
                                                                                                   \
        /* Check if contiguous for fast path */                                                    \
        bool is_contiguous =                                                                       \
            (lhs_stride_k == 1 && rhs_stride_n == 1 && lhs_stride_m == K && rhs_stride_k == N);    \
                                                                                                   \
        if (is_contiguous && lhs_offset == 0 && rhs_offset == 0) {                                 \
            /* Fast path: contiguous matrices with cache blocking */                               \
            for (size_t ii = 0; ii < M; ii += BLOCK_M) {                                           \
                size_t i_end = (ii + BLOCK_M < M) ? (ii + BLOCK_M) : M;                            \
                for (size_t jj = 0; jj < N; jj += BLOCK_N) {                                       \
                    size_t j_end = (jj + BLOCK_N < N) ? (jj + BLOCK_N) : N;                        \
                    for (size_t kk = 0; kk < K; kk += BLOCK_K) {                                   \
                        size_t k_end = (kk + BLOCK_K < K) ? (kk + BLOCK_K) : K;                    \
                        /* Process block */                                                        \
                        for (size_t i = ii; i < i_end; i++) {                                      \
                            for (size_t k = kk; k < k_end; k++) {                                  \
                                TYPE a_val = lhs[i * K + k];                                       \
                                for (size_t j = jj; j < j_end; j++) {                              \
                                    output[i * N + j] += a_val * rhs[k * N + j];                   \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            /* Strided path with cache blocking */                                                 \
            for (size_t ii = 0; ii < M; ii += BLOCK_M) {                                           \
                size_t i_end = (ii + BLOCK_M < M) ? (ii + BLOCK_M) : M;                            \
                for (size_t jj = 0; jj < N; jj += BLOCK_N) {                                       \
                    size_t j_end = (jj + BLOCK_N < N) ? (jj + BLOCK_N) : N;                        \
                    for (size_t kk = 0; kk < K; kk += BLOCK_K) {                                   \
                        size_t k_end = (kk + BLOCK_K < K) ? (kk + BLOCK_K) : K;                    \
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
// SIMD-OPTIMIZED DOT for F32/F64
// ============================================================================

#if SIMD_F32_WIDTH > 1

/// SIMD-optimized F32 dot product with cache blocking
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

    /* Initialize output */
    for (size_t i = 0; i < M * N; i++) {
        output[i] = 0;
    }

    const size_t BLOCK_M = 64;
    const size_t BLOCK_N = 64;
    const size_t BLOCK_K = 64;

    bool is_contiguous =
        (lhs_stride_k == 1 && rhs_stride_n == 1 && lhs_stride_m == K && rhs_stride_k == N);

    if (is_contiguous && lhs_offset == 0 && rhs_offset == 0) {
        /* SIMD fast path */
        for (size_t ii = 0; ii < M; ii += BLOCK_M) {
            size_t i_end = (ii + BLOCK_M < M) ? (ii + BLOCK_M) : M;
            for (size_t jj = 0; jj < N; jj += BLOCK_N) {
                size_t j_end = (jj + BLOCK_N < N) ? (jj + BLOCK_N) : N;
                for (size_t kk = 0; kk < K; kk += BLOCK_K) {
                    size_t k_end = (kk + BLOCK_K < K) ? (kk + BLOCK_K) : K;

                    for (size_t i = ii; i < i_end; i++) {
                        for (size_t k = kk; k < k_end; k++) {
                            f32_t a_val = lhs[i * K + k];
                            simd_f32_t va = simd_f32_set1(a_val);

                            size_t j = jj;
                            const size_t simd_end =
                                jj + ((j_end - jj) / SIMD_F32_WIDTH) * SIMD_F32_WIDTH;

                            /* SIMD loop */
                            for (; j < simd_end; j += SIMD_F32_WIDTH) {
                                simd_f32_t vb = simd_f32_load(&rhs[k * N + j]);
                                simd_f32_t vc = simd_f32_load(&output[i * N + j]);
                                vc = simd_f32_fmadd(va, vb, vc);
                                simd_f32_store(&output[i * N + j], vc);
                            }

                            /* Scalar remainder */
                            for (; j < j_end; j++) {
                                output[i * N + j] += a_val * rhs[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        /* Fallback to non-SIMD blocked version */
        for (size_t ii = 0; ii < M; ii += BLOCK_M) {
            size_t i_end = (ii + BLOCK_M < M) ? (ii + BLOCK_M) : M;
            for (size_t jj = 0; jj < N; jj += BLOCK_N) {
                size_t j_end = (jj + BLOCK_N < N) ? (jj + BLOCK_N) : N;
                for (size_t kk = 0; kk < K; kk += BLOCK_K) {
                    size_t k_end = (kk + BLOCK_K < K) ? (kk + BLOCK_K) : K;
                    for (size_t i = ii; i < i_end; i++) {
                        for (size_t k = kk; k < k_end; k++) {
                            size_t lhs_idx = lhs_offset + i * lhs_stride_m + k * lhs_stride_k;
                            f32_t a_val = lhs[lhs_idx];
                            for (size_t j = jj; j < j_end; j++) {
                                size_t rhs_idx = rhs_offset + k * rhs_stride_k + j * rhs_stride_n;
                                output[i * N + j] += a_val * rhs[rhs_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

#else
DOT_OP(f32_t, f32)
#endif

#if SIMD_F64_WIDTH > 1

/// SIMD-optimized F64 dot product with cache blocking
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

    for (size_t i = 0; i < M * N; i++) {
        output[i] = 0;
    }

    const size_t BLOCK_M = 64;
    const size_t BLOCK_N = 64;
    const size_t BLOCK_K = 64;

    bool is_contiguous =
        (lhs_stride_k == 1 && rhs_stride_n == 1 && lhs_stride_m == K && rhs_stride_k == N);

    if (is_contiguous && lhs_offset == 0 && rhs_offset == 0) {
        for (size_t ii = 0; ii < M; ii += BLOCK_M) {
            size_t i_end = (ii + BLOCK_M < M) ? (ii + BLOCK_M) : M;
            for (size_t jj = 0; jj < N; jj += BLOCK_N) {
                size_t j_end = (jj + BLOCK_N < N) ? (jj + BLOCK_N) : N;
                for (size_t kk = 0; kk < K; kk += BLOCK_K) {
                    size_t k_end = (kk + BLOCK_K < K) ? (kk + BLOCK_K) : K;

                    for (size_t i = ii; i < i_end; i++) {
                        for (size_t k = kk; k < k_end; k++) {
                            f64_t a_val = lhs[i * K + k];
                            simd_f64_t va = simd_f64_set1(a_val);

                            size_t j = jj;
                            const size_t simd_end =
                                jj + ((j_end - jj) / SIMD_F64_WIDTH) * SIMD_F64_WIDTH;

                            for (; j < simd_end; j += SIMD_F64_WIDTH) {
                                simd_f64_t vb = simd_f64_load(&rhs[k * N + j]);
                                simd_f64_t vc = simd_f64_load(&output[i * N + j]);
                                vc = simd_f64_fmadd(va, vb, vc);
                                simd_f64_store(&output[i * N + j], vc);
                            }

                            for (; j < j_end; j++) {
                                output[i * N + j] += a_val * rhs[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (size_t ii = 0; ii < M; ii += BLOCK_M) {
            size_t i_end = (ii + BLOCK_M < M) ? (ii + BLOCK_M) : M;
            for (size_t jj = 0; jj < N; jj += BLOCK_N) {
                size_t j_end = (jj + BLOCK_N < N) ? (jj + BLOCK_N) : N;
                for (size_t kk = 0; kk < K; kk += BLOCK_K) {
                    size_t k_end = (kk + BLOCK_K < K) ? (kk + BLOCK_K) : K;
                    for (size_t i = ii; i < i_end; i++) {
                        for (size_t k = kk; k < k_end; k++) {
                            size_t lhs_idx = lhs_offset + i * lhs_stride_m + k * lhs_stride_k;
                            f64_t a_val = lhs[lhs_idx];
                            for (size_t j = jj; j < j_end; j++) {
                                size_t rhs_idx = rhs_offset + k * rhs_stride_k + j * rhs_stride_n;
                                output[i * N + j] += a_val * rhs[rhs_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

#else
DOT_OP(f64_t, f64)
#endif

// Non-SIMD types use regular DOT_OP
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
