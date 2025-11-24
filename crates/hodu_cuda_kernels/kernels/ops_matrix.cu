#include "math.cuh"
#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define TILE_SIZE 16

#define MATMUL_OP(TYPENAME, FN_NAME)                                                               \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *lhs, const TYPENAME *rhs,       \
                                                   TYPENAME *out, const size_t *metadata) {        \
        const size_t lhs_ndim = metadata[1];                                                       \
        const size_t rhs_ndim = metadata[2];                                                       \
        const size_t batch_ndim = metadata[3];                                                     \
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
        __shared__ TYPENAME lhs_tile[TILE_SIZE][TILE_SIZE];                                        \
        __shared__ TYPENAME rhs_tile[TILE_SIZE][TILE_SIZE];                                        \
        size_t batch_idx = blockIdx.z;                                                             \
        size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;                                         \
        size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;                                         \
        size_t batch_indices[16];                                                                  \
        size_t temp = batch_idx;                                                                   \
        for (int d = (int)batch_ndim - 1; d >= 0; d--) {                                           \
            batch_indices[d] = temp % batch_shape[d];                                              \
            temp /= batch_shape[d];                                                                \
        }                                                                                          \
        size_t lhs_batch_ndim = lhs_ndim - 2;                                                      \
        size_t lhs_batch_indices[16];                                                              \
        for (size_t d = 0; d < lhs_batch_ndim; d++) {                                              \
            size_t batch_dim_idx = batch_ndim - lhs_batch_ndim + d;                                \
            lhs_batch_indices[d] = (lhs_shape[d] == 1) ? 0 : batch_indices[batch_dim_idx];         \
        }                                                                                          \
        size_t rhs_batch_ndim = rhs_ndim - 2;                                                      \
        size_t rhs_batch_indices[16];                                                              \
        for (size_t d = 0; d < rhs_batch_ndim; d++) {                                              \
            size_t batch_dim_idx = batch_ndim - rhs_batch_ndim + d;                                \
            rhs_batch_indices[d] = (rhs_shape[d] == 1) ? 0 : batch_indices[batch_dim_idx];         \
        }                                                                                          \
        size_t lhs_base_offset = lhs_offset;                                                       \
        for (size_t d = 0; d < lhs_batch_ndim; d++) {                                              \
            lhs_base_offset += lhs_batch_indices[d] * lhs_strides[d];                              \
        }                                                                                          \
        size_t rhs_base_offset = rhs_offset;                                                       \
        for (size_t d = 0; d < rhs_batch_ndim; d++) {                                              \
            rhs_base_offset += rhs_batch_indices[d] * rhs_strides[d];                              \
        }                                                                                          \
        TYPENAME sum = from_float<TYPENAME>(0.0f);                                                 \
        size_t num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;                                        \
        for (size_t tile = 0; tile < num_tiles; tile++) {                                          \
            size_t lhs_k = tile * TILE_SIZE + threadIdx.x;                                         \
            if (row < M && lhs_k < K) {                                                            \
                size_t lhs_idx = lhs_base_offset + row * lhs_strides[lhs_ndim - 2] +               \
                                 lhs_k * lhs_strides[lhs_ndim - 1];                                \
                lhs_tile[threadIdx.y][threadIdx.x] = lhs[lhs_idx];                                 \
            } else {                                                                               \
                lhs_tile[threadIdx.y][threadIdx.x] = from_float<TYPENAME>(0.0f);                   \
            }                                                                                      \
            size_t rhs_k = tile * TILE_SIZE + threadIdx.y;                                         \
            if (rhs_k < K && col < N) {                                                            \
                size_t rhs_idx = rhs_base_offset + rhs_k * rhs_strides[rhs_ndim - 2] +             \
                                 col * rhs_strides[rhs_ndim - 1];                                  \
                rhs_tile[threadIdx.y][threadIdx.x] = rhs[rhs_idx];                                 \
            } else {                                                                               \
                rhs_tile[threadIdx.y][threadIdx.x] = from_float<TYPENAME>(0.0f);                   \
            }                                                                                      \
            __syncthreads();                                                                       \
            for (size_t k = 0; k < TILE_SIZE; k++) {                                               \
                float lhs_val = to_float(lhs_tile[threadIdx.y][k]);                                \
                float rhs_val = to_float(rhs_tile[k][threadIdx.x]);                                \
                sum = from_float<TYPENAME>(to_float(sum) + lhs_val * rhs_val);                     \
            }                                                                                      \
            __syncthreads();                                                                       \
        }                                                                                          \
        if (row < M && col < N) {                                                                  \
            size_t output_idx = batch_idx * (M * N) + row * N + col;                               \
            out[output_idx] = sum;                                                                 \
        }                                                                                          \
    }

MATMUL_OP(__nv_fp8_e4m3, matmul_f8e4m3)
MATMUL_OP(__nv_fp8_e5m2, matmul_f8e5m2)
MATMUL_OP(__nv_bfloat16, matmul_bf16)
MATMUL_OP(__half, matmul_f16)
MATMUL_OP(float, matmul_f32)
MATMUL_OP(double, matmul_f64)
MATMUL_OP(int8_t, matmul_i8)
MATMUL_OP(int16_t, matmul_i16)
MATMUL_OP(int32_t, matmul_i32)
MATMUL_OP(int64_t, matmul_i64)
MATMUL_OP(uint8_t, matmul_u8)
MATMUL_OP(uint16_t, matmul_u16)
MATMUL_OP(uint32_t, matmul_u32)
MATMUL_OP(uint64_t, matmul_u64)

#define DOT_TILE_SIZE 32

#define DOT_OP(TYPENAME, FN_NAME)                                                                  \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *lhs, const TYPENAME *rhs,       \
                                                   TYPENAME *out, const size_t *metadata) {        \
        const size_t M = metadata[0];                                                              \
        const size_t K = metadata[1];                                                              \
        const size_t N = metadata[2];                                                              \
        const size_t lhs_stride_m = metadata[3];                                                   \
        const size_t lhs_stride_k = metadata[4];                                                   \
        const size_t rhs_stride_k = metadata[5];                                                   \
        const size_t rhs_stride_n = metadata[6];                                                   \
        const size_t lhs_offset = metadata[7];                                                     \
        const size_t rhs_offset = metadata[8];                                                     \
        __shared__ TYPENAME lhs_tile[DOT_TILE_SIZE][DOT_TILE_SIZE];                                \
        __shared__ TYPENAME rhs_tile[DOT_TILE_SIZE][DOT_TILE_SIZE];                                \
        size_t row = blockIdx.y * DOT_TILE_SIZE + threadIdx.y;                                     \
        size_t col = blockIdx.x * DOT_TILE_SIZE + threadIdx.x;                                     \
        TYPENAME sum = from_float<TYPENAME>(0.0f);                                                 \
        size_t num_tiles = (K + DOT_TILE_SIZE - 1) / DOT_TILE_SIZE;                                \
        for (size_t tile = 0; tile < num_tiles; tile++) {                                          \
            size_t lhs_k = tile * DOT_TILE_SIZE + threadIdx.x;                                     \
            if (row < M && lhs_k < K) {                                                            \
                size_t lhs_idx = lhs_offset + row * lhs_stride_m + lhs_k * lhs_stride_k;           \
                lhs_tile[threadIdx.y][threadIdx.x] = lhs[lhs_idx];                                 \
            } else {                                                                               \
                lhs_tile[threadIdx.y][threadIdx.x] = from_float<TYPENAME>(0.0f);                   \
            }                                                                                      \
            size_t rhs_k = tile * DOT_TILE_SIZE + threadIdx.y;                                     \
            if (rhs_k < K && col < N) {                                                            \
                size_t rhs_idx = rhs_offset + rhs_k * rhs_stride_k + col * rhs_stride_n;           \
                rhs_tile[threadIdx.y][threadIdx.x] = rhs[rhs_idx];                                 \
            } else {                                                                               \
                rhs_tile[threadIdx.y][threadIdx.x] = from_float<TYPENAME>(0.0f);                   \
            }                                                                                      \
            __syncthreads();                                                                       \
            for (size_t k = 0; k < DOT_TILE_SIZE; k++) {                                           \
                float lhs_val = to_float(lhs_tile[threadIdx.y][k]);                                \
                float rhs_val = to_float(rhs_tile[k][threadIdx.x]);                                \
                sum = from_float<TYPENAME>(to_float(sum) + lhs_val * rhs_val);                     \
            }                                                                                      \
            __syncthreads();                                                                       \
        }                                                                                          \
        if (row < M && col < N) {                                                                  \
            out[row * N + col] = sum;                                                              \
        }                                                                                          \
    }

DOT_OP(__nv_fp8_e4m3, dot_f8e4m3)
DOT_OP(__nv_fp8_e5m2, dot_f8e5m2)
DOT_OP(__nv_bfloat16, dot_bf16)
DOT_OP(__half, dot_f16)
DOT_OP(float, dot_f32)
DOT_OP(double, dot_f64)
DOT_OP(int8_t, dot_i8)
DOT_OP(int16_t, dot_i16)
DOT_OP(int32_t, dot_i32)
DOT_OP(int64_t, dot_i64)
DOT_OP(uint8_t, dot_u8)
DOT_OP(uint16_t, dot_u16)
DOT_OP(uint32_t, dot_u32)
DOT_OP(uint64_t, dot_u64)
