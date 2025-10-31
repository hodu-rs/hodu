#include "ops_matrix.h"
#include "types.h"
#include <stdint.h>
#include <string.h>

// ============================================================================
// BATCHED MATRIX MULTIPLICATION (MATMUL)
// ============================================================================

#define MATMUL_OP(TYPE, TYPE_SUFFIX)                                                               \
    void matmul_##TYPE_SUFFIX(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,          \
                              size_t num_els, size_t num_dims, const size_t *metadata) {           \
        (void)num_dims; /* Unused parameter */                                                     \
        const TYPE *lhs = (const TYPE *)lhs_ptr;                                                   \
        const TYPE *rhs = (const TYPE *)rhs_ptr;                                                   \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t lhs_ndim = metadata[0];                                                       \
        const size_t rhs_ndim = metadata[1];                                                       \
        const size_t batch_ndim = metadata[2];                                                     \
                                                                                                   \
        const size_t *lhs_shape = metadata + 3;                                                    \
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

// Define matmul for all types
MATMUL_OP(f8e4m3_t, f8e4m3)
MATMUL_OP(f8e5m2_t, f8e5m2)
MATMUL_OP(bf16_t, bf16)
MATMUL_OP(f16_t, f16)
MATMUL_OP(f32_t, f32)
MATMUL_OP(f64_t, f64)
MATMUL_OP(int8_t, i8)
MATMUL_OP(int16_t, i16)
MATMUL_OP(int32_t, i32)
MATMUL_OP(int64_t, i64)
MATMUL_OP(uint8_t, u8)
MATMUL_OP(uint16_t, u16)
MATMUL_OP(uint32_t, u32)
MATMUL_OP(uint64_t, u64)

// ============================================================================
// TILED 2D DOT PRODUCT
// ============================================================================

#define DOT_OP(TYPE, TYPE_SUFFIX)                                                                  \
    void dot_##TYPE_SUFFIX(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,             \
                           size_t num_els, size_t num_dims, const size_t *metadata) {              \
        (void)num_els;  /* Unused parameter */                                                     \
        (void)num_dims; /* Unused parameter */                                                     \
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
        /* Simple matrix multiplication without tiling (CPU doesn't benefit from shared memory) */ \
        for (size_t row = 0; row < M; row++) {                                                     \
            for (size_t col = 0; col < N; col++) {                                                 \
                TYPE sum = 0;                                                                      \
                for (size_t k = 0; k < K; k++) {                                                   \
                    size_t lhs_idx = lhs_offset + row * lhs_stride_m + k * lhs_stride_k;           \
                    size_t rhs_idx = rhs_offset + k * rhs_stride_k + col * rhs_stride_n;           \
                    sum += lhs[lhs_idx] * rhs[rhs_idx];                                            \
                }                                                                                  \
                output[row * N + col] = sum;                                                       \
            }                                                                                      \
        }                                                                                          \
    }

// Define dot for all types
DOT_OP(f8e4m3_t, f8e4m3)
DOT_OP(f8e5m2_t, f8e5m2)
DOT_OP(bf16_t, bf16)
DOT_OP(f16_t, f16)
DOT_OP(f32_t, f32)
DOT_OP(f64_t, f64)
DOT_OP(int8_t, i8)
DOT_OP(int16_t, i16)
DOT_OP(int32_t, i32)
DOT_OP(int64_t, i64)
DOT_OP(uint8_t, u8)
DOT_OP(uint16_t, u16)
DOT_OP(uint32_t, u32)
DOT_OP(uint64_t, u64)
