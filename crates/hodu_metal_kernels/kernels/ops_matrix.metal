#include "./headers/utils.metal"
#include <metal_stdlib>

using namespace metal;

// Matrix multiplication operations for tensors
// Supports both batched matmul with broadcasting and tiled 2D dot product

#define TILE_SIZE 16

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
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *lhs [[buffer(0)]], const device TYPENAME *rhs [[buffer(1)]],        \
        device TYPENAME *output [[buffer(2)]], constant size_t *metadata [[buffer(3)]],            \
        uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],                   \
        uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],                       \
        uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {                               \
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
        /* Shared memory for tiles */                                                              \
        threadgroup TYPENAME lhs_tile[TILE_SIZE][TILE_SIZE];                                       \
        threadgroup TYPENAME rhs_tile[TILE_SIZE][TILE_SIZE];                                       \
                                                                                                   \
        /* Calculate global row, column, and batch index */                                        \
        size_t batch_idx = threadgroup_position_in_grid.z;                                         \
        size_t row =                                                                               \
            threadgroup_position_in_grid.y * TILE_SIZE + thread_position_in_threadgroup.y;         \
        size_t col =                                                                               \
            threadgroup_position_in_grid.x * TILE_SIZE + thread_position_in_threadgroup.x;         \
                                                                                                   \
        /* Compute batch indices from flat batch_idx */                                            \
        size_t batch_indices[16];                                                                  \
        size_t temp = batch_idx;                                                                   \
        for (int d = (int)batch_ndim - 1; d >= 0; d--) {                                           \
            batch_indices[d] = temp % batch_shape[d];                                              \
            temp /= batch_shape[d];                                                                \
        }                                                                                          \
                                                                                                   \
        /* Map batch indices to lhs indices (with broadcasting) */                                 \
        size_t lhs_batch_ndim = lhs_ndim - 2;                                                      \
        size_t lhs_batch_indices[16];                                                              \
        for (size_t d = 0; d < lhs_batch_ndim; d++) {                                              \
            size_t batch_dim_idx = batch_ndim - lhs_batch_ndim + d;                                \
            lhs_batch_indices[d] = (lhs_shape[d] == 1) ? 0 : batch_indices[batch_dim_idx];         \
        }                                                                                          \
                                                                                                   \
        /* Map batch indices to rhs indices (with broadcasting) */                                 \
        size_t rhs_batch_ndim = rhs_ndim - 2;                                                      \
        size_t rhs_batch_indices[16];                                                              \
        for (size_t d = 0; d < rhs_batch_ndim; d++) {                                              \
            size_t batch_dim_idx = batch_ndim - rhs_batch_ndim + d;                                \
            rhs_batch_indices[d] = (rhs_shape[d] == 1) ? 0 : batch_indices[batch_dim_idx];         \
        }                                                                                          \
                                                                                                   \
        /* Calculate base offsets for this batch */                                                \
        size_t lhs_base_offset = lhs_offset;                                                       \
        for (size_t d = 0; d < lhs_batch_ndim; d++) {                                              \
            lhs_base_offset += lhs_batch_indices[d] * lhs_strides[d];                              \
        }                                                                                          \
                                                                                                   \
        size_t rhs_base_offset = rhs_offset;                                                       \
        for (size_t d = 0; d < rhs_batch_ndim; d++) {                                              \
            rhs_base_offset += rhs_batch_indices[d] * rhs_strides[d];                              \
        }                                                                                          \
                                                                                                   \
        TYPENAME sum = 0;                                                                          \
                                                                                                   \
        /* Loop over tiles */                                                                      \
        size_t num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;                                        \
        for (size_t tile = 0; tile < num_tiles; tile++) {                                          \
            /* Load LHS tile */                                                                    \
            size_t lhs_k = tile * TILE_SIZE + thread_position_in_threadgroup.x;                    \
            if (row < M && lhs_k < K) {                                                            \
                size_t lhs_idx = lhs_base_offset + row * lhs_strides[lhs_ndim - 2] +               \
                                 lhs_k * lhs_strides[lhs_ndim - 1];                                \
                lhs_tile[thread_position_in_threadgroup.y][thread_position_in_threadgroup.x] =     \
                    lhs[lhs_idx];                                                                  \
            } else {                                                                               \
                lhs_tile[thread_position_in_threadgroup.y][thread_position_in_threadgroup.x] = 0;  \
            }                                                                                      \
                                                                                                   \
            /* Load RHS tile */                                                                    \
            size_t rhs_k = tile * TILE_SIZE + thread_position_in_threadgroup.y;                    \
            if (rhs_k < K && col < N) {                                                            \
                size_t rhs_idx = rhs_base_offset + rhs_k * rhs_strides[rhs_ndim - 2] +             \
                                 col * rhs_strides[rhs_ndim - 1];                                  \
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
            size_t output_idx = batch_idx * (M * N) + row * N + col;                               \
            output[output_idx] = sum;                                                              \
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
// Metadata layout (9 elements):
// - metadata[0]: M (rows of A)
// - metadata[1]: K (cols of A / rows of B)
// - metadata[2]: N (cols of B)
// - metadata[3]: lhs_stride_m (stride for row dimension of A)
// - metadata[4]: lhs_stride_k (stride for col dimension of A)
// - metadata[5]: rhs_stride_k (stride for row dimension of B)
// - metadata[6]: rhs_stride_n (stride for col dimension of B)
// - metadata[7]: lhs_offset (starting offset in lhs buffer)
// - metadata[8]: rhs_offset (starting offset in rhs buffer)

// Optimized DOT with register blocking, loop unrolling, and larger tiles
// Each thread computes a 4x4 block of output elements
#define BLOCK_M 4
#define BLOCK_N 4
#define DOT_TILE_SIZE 32
#define THREADS_PER_TILE (DOT_TILE_SIZE / BLOCK_M)

#define DOT_OP(TYPENAME, FN_NAME)                                                                  \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *lhs [[buffer(0)]], const device TYPENAME *rhs [[buffer(1)]],        \
        device TYPENAME *output [[buffer(2)]], constant size_t &num_els [[buffer(3)]],             \
        constant size_t *metadata [[buffer(4)]],                                                   \
        uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],                   \
        uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]],                       \
        uint2 threads_per_threadgroup [[threads_per_threadgroup]]) {                               \
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
        /* Shared memory for larger tiles */                                                       \
        threadgroup TYPENAME lhs_tile[DOT_TILE_SIZE][DOT_TILE_SIZE];                               \
        threadgroup TYPENAME rhs_tile[DOT_TILE_SIZE][DOT_TILE_SIZE];                               \
                                                                                                   \
        /* Each thread computes a BLOCK_M x BLOCK_N sub-block */                                   \
        size_t base_row = threadgroup_position_in_grid.y * DOT_TILE_SIZE +                         \
                          thread_position_in_threadgroup.y * BLOCK_M;                              \
        size_t base_col = threadgroup_position_in_grid.x * DOT_TILE_SIZE +                         \
                          thread_position_in_threadgroup.x * BLOCK_N;                              \
                                                                                                   \
        /* Register blocking: accumulate 4x4 block */                                              \
        TYPENAME sums[BLOCK_M][BLOCK_N];                                                           \
        for (size_t i = 0; i < BLOCK_M; i++) {                                                     \
            for (size_t j = 0; j < BLOCK_N; j++) {                                                 \
                sums[i][j] = 0;                                                                    \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        /* Loop over tiles */                                                                      \
        size_t num_tiles = (K + DOT_TILE_SIZE - 1) / DOT_TILE_SIZE;                                \
        for (size_t tile = 0; tile < num_tiles; tile++) {                                          \
            /* Load LHS tile: each thread loads 4x4 elements */                                    \
            for (size_t i = 0; i < BLOCK_M; i++) {                                                 \
                for (size_t j = 0; j < BLOCK_N; j++) {                                             \
                    size_t row = thread_position_in_threadgroup.y * BLOCK_M + i;                   \
                    size_t col = thread_position_in_threadgroup.x * BLOCK_N + j;                   \
                    size_t global_row = base_row + i;                                              \
                    size_t k_idx = tile * DOT_TILE_SIZE + col;                                     \
                    if (global_row < M && k_idx < K) {                                             \
                        size_t lhs_idx =                                                           \
                            lhs_offset + global_row * lhs_stride_m + k_idx * lhs_stride_k;         \
                        lhs_tile[row][col] = lhs[lhs_idx];                                         \
                    } else {                                                                       \
                        lhs_tile[row][col] = 0;                                                    \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            /* Load RHS tile: each thread loads 4x4 elements */                                    \
            for (size_t i = 0; i < BLOCK_M; i++) {                                                 \
                for (size_t j = 0; j < BLOCK_N; j++) {                                             \
                    size_t row = thread_position_in_threadgroup.y * BLOCK_M + i;                   \
                    size_t col = thread_position_in_threadgroup.x * BLOCK_N + j;                   \
                    size_t k_idx = tile * DOT_TILE_SIZE + row;                                     \
                    size_t global_col = base_col + j;                                              \
                    if (k_idx < K && global_col < N) {                                             \
                        size_t rhs_idx =                                                           \
                            rhs_offset + k_idx * rhs_stride_k + global_col * rhs_stride_n;         \
                        rhs_tile[row][col] = rhs[rhs_idx];                                         \
                    } else {                                                                       \
                        rhs_tile[row][col] = 0;                                                    \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            /* Synchronize to ensure tiles are loaded */                                           \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
                                                                                                   \
            /* Compute partial dot product with loop unrolling */                                  \
            _Pragma("unroll") for (size_t k = 0; k < DOT_TILE_SIZE; k++) {                         \
                TYPENAME lhs_vals[BLOCK_M];                                                        \
                TYPENAME rhs_vals[BLOCK_N];                                                        \
                                                                                                   \
                /* Load values into registers */                                                   \
                _Pragma("unroll") for (size_t i = 0; i < BLOCK_M; i++) {                           \
                    lhs_vals[i] = lhs_tile[thread_position_in_threadgroup.y * BLOCK_M + i][k];     \
                }                                                                                  \
                _Pragma("unroll") for (size_t j = 0; j < BLOCK_N; j++) {                           \
                    rhs_vals[j] = rhs_tile[k][thread_position_in_threadgroup.x * BLOCK_N + j];     \
                }                                                                                  \
                                                                                                   \
                /* Outer product */                                                                \
                _Pragma("unroll") for (size_t i = 0; i < BLOCK_M; i++) {                           \
                    _Pragma("unroll") for (size_t j = 0; j < BLOCK_N; j++) {                       \
                        sums[i][j] += lhs_vals[i] * rhs_vals[j];                                   \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            /* Synchronize before loading next tile */                                             \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
        }                                                                                          \
                                                                                                   \
        /* Write results */                                                                        \
        for (size_t i = 0; i < BLOCK_M; i++) {                                                     \
            for (size_t j = 0; j < BLOCK_N; j++) {                                                 \
                size_t global_row = base_row + i;                                                  \
                size_t global_col = base_col + j;                                                  \
                if (global_row < M && global_col < N) {                                            \
                    output[global_row * N + global_col] = sums[i][j];                              \
                }                                                                                  \
            }                                                                                      \
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
