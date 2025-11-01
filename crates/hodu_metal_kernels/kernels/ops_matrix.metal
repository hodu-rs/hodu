#include "./headers/utils.metal"
#include <metal_stdlib>

using namespace metal;

// Matrix multiplication operations for tensors
// Supports both batched matmul with broadcasting and tiled 2D dot product

// ============================================================================
// BATCHED MATRIX MULTIPLICATION (MATMUL)
// ============================================================================

// Batched matrix multiplication with broadcasting
// LHS: [...batch_dims..., M, K]
// RHS: [...batch_dims..., K, N]
// Output: [...batch_dims..., M, N]
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

#define MATMUL_OP(TYPENAME, FN_NAME)                                                               \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *lhs [[buffer(0)]], const device TYPENAME *rhs [[buffer(1)]],        \
        device TYPENAME *output [[buffer(2)]], constant size_t *metadata [[buffer(3)]],            \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t lhs_ndim = metadata[1];                                                       \
        const size_t rhs_ndim = metadata[2];                                                       \
        const size_t batch_ndim = metadata[3];                                                     \
                                                                                                   \
        const constant size_t *lhs_shape = metadata + 4;                                           \
        const constant size_t *rhs_shape = lhs_shape + lhs_ndim;                                   \
        const constant size_t *batch_shape = rhs_shape + rhs_ndim;                                 \
        const constant size_t *lhs_strides = batch_shape + batch_ndim;                             \
        const constant size_t *rhs_strides = lhs_strides + lhs_ndim;                               \
        const size_t lhs_offset = *(rhs_strides + rhs_ndim);                                       \
        const size_t rhs_offset = *(rhs_strides + rhs_ndim + 1);                                   \
        const size_t M = *(rhs_strides + rhs_ndim + 2);                                            \
        const size_t K = *(rhs_strides + rhs_ndim + 3);                                            \
        const size_t N = *(rhs_strides + rhs_ndim + 4);                                            \
                                                                                                   \
        /* Grid-stride loop for better GPU utilization */                                          \
        for (uint idx = thread_index; idx < num_els; idx += threads_per_grid) {                    \
                                                                                                   \
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
            TYPENAME sum = 0;                                                                      \
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

// Define matmul operations for all types
MATMUL_OP(bfloat, matmul_bf16)
MATMUL_OP(half, matmul_f16)
MATMUL_OP(float, matmul_f32)
MATMUL_OP(int8_t, matmul_i8)
MATMUL_OP(int16_t, matmul_i16)
MATMUL_OP(int32_t, matmul_i32)
MATMUL_OP(int64_t, matmul_i64)
MATMUL_OP(uint8_t, matmul_u8)
MATMUL_OP(uint16_t, matmul_u16)
MATMUL_OP(uint32_t, matmul_u32)
MATMUL_OP(uint64_t, matmul_u64)

// ============================================================================
// TILED 2D DOT PRODUCT (Optimized with threadgroup memory)
// ============================================================================

// Tiled matrix multiplication using threadgroup memory
// This version provides better performance for larger matrices by:
// 1. Using shared memory (threadgroup) to cache tiles
// 2. Reducing global memory access
// 3. Better memory coalescing
//
// LHS: (M, K)
// RHS: (K, N)
// Output: (M, N)
//
// Metadata layout:
// - lhs_shape: [M, K]
// - rhs_shape: [K, N]
// - lhs_strides: [stride_m, stride_k]
// - rhs_strides: [stride_k, stride_n]
// - lhs_offset: scalar offset for LHS
// - rhs_offset: scalar offset for RHS

#define TILE_SIZE 16

#define DOT_OP(TYPENAME, FN_NAME)                                                                  \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *lhs [[buffer(0)]], const device TYPENAME *rhs [[buffer(1)]],        \
        device TYPENAME *output [[buffer(2)]], constant size_t &num_els [[buffer(3)]],             \
        constant size_t *metadata [[buffer(4)]],                                                   \
        uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],                   \
        uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]],                       \
        uint2 threads_per_threadgroup [[threads_per_threadgroup]]) {                               \
                                                                                                   \
        const size_t M = metadata[0];                                                              \
        const size_t K = metadata[1];                                                              \
        const size_t N = metadata[3];                                                              \
        const size_t lhs_stride_m = metadata[4];                                                   \
        const size_t lhs_stride_k = metadata[5];                                                   \
        const size_t rhs_stride_k = metadata[6];                                                   \
        const size_t rhs_stride_n = metadata[7];                                                   \
        const size_t lhs_offset = metadata[8];                                                     \
        const size_t rhs_offset = metadata[9];                                                     \
                                                                                                   \
        /* Shared memory for tiles */                                                              \
        threadgroup TYPENAME lhs_tile[TILE_SIZE][TILE_SIZE];                                       \
        threadgroup TYPENAME rhs_tile[TILE_SIZE][TILE_SIZE];                                       \
                                                                                                   \
        /* Calculate global row and column */                                                      \
        size_t row =                                                                               \
            threadgroup_position_in_grid.y * TILE_SIZE + thread_position_in_threadgroup.y;         \
        size_t col =                                                                               \
            threadgroup_position_in_grid.x * TILE_SIZE + thread_position_in_threadgroup.x;         \
                                                                                                   \
        TYPENAME sum = 0;                                                                          \
                                                                                                   \
        /* Loop over tiles */                                                                      \
        size_t num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;                                        \
        for (size_t tile = 0; tile < num_tiles; tile++) {                                          \
            /* Load LHS tile */                                                                    \
            size_t lhs_k = tile * TILE_SIZE + thread_position_in_threadgroup.x;                    \
            if (row < M && lhs_k < K) {                                                            \
                size_t lhs_idx = lhs_offset + row * lhs_stride_m + lhs_k * lhs_stride_k;           \
                lhs_tile[thread_position_in_threadgroup.y][thread_position_in_threadgroup.x] =     \
                    lhs[lhs_idx];                                                                  \
            } else {                                                                               \
                lhs_tile[thread_position_in_threadgroup.y][thread_position_in_threadgroup.x] = 0;  \
            }                                                                                      \
                                                                                                   \
            /* Load RHS tile */                                                                    \
            size_t rhs_k = tile * TILE_SIZE + thread_position_in_threadgroup.y;                    \
            if (rhs_k < K && col < N) {                                                            \
                size_t rhs_idx = rhs_offset + rhs_k * rhs_stride_k + col * rhs_stride_n;           \
                rhs_tile[thread_position_in_threadgroup.y][thread_position_in_threadgroup.x] =     \
                    rhs[rhs_idx];                                                                  \
            } else {                                                                               \
                rhs_tile[thread_position_in_threadgroup.y][thread_position_in_threadgroup.x] = 0;  \
            }                                                                                      \
                                                                                                   \
            /* Synchronize to ensure tiles are loaded */                                           \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
                                                                                                   \
            /* Compute partial dot product for this tile */                                        \
            for (size_t k = 0; k < TILE_SIZE; k++) {                                               \
                sum += lhs_tile[thread_position_in_threadgroup.y][k] *                             \
                       rhs_tile[k][thread_position_in_threadgroup.x];                              \
            }                                                                                      \
                                                                                                   \
            /* Synchronize before loading next tile */                                             \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
        }                                                                                          \
                                                                                                   \
        /* Write result */                                                                         \
        if (row < M && col < N) {                                                                  \
            output[row * N + col] = sum;                                                           \
        }                                                                                          \
    }

// Define tiled dot operations for all types
DOT_OP(bfloat, dot_bf16)
DOT_OP(half, dot_f16)
DOT_OP(float, dot_f32)
DOT_OP(int8_t, dot_i8)
DOT_OP(int16_t, dot_i16)
DOT_OP(int32_t, dot_i32)
DOT_OP(int64_t, dot_i64)
DOT_OP(uint8_t, dot_u8)
DOT_OP(uint16_t, dot_u16)
DOT_OP(uint32_t, dot_u32)
DOT_OP(uint64_t, dot_u64)
