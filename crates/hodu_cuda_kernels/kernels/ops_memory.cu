#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CONTIGUOUS_OP(TYPENAME, FN_NAME)                                                           \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, TYPENAME *out,                       \
                                       const size_t *metadata) {                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *dims = metadata + 2;                                                         \
        const size_t *strides = metadata + 2 + num_dims;                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        bool cont = is_contiguous(num_dims, dims, strides);                                        \
        if (cont) {                                                                                \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;              \
                 i += blockDim.x * gridDim.x) {                                                    \
                out[i] = input[offset + i];                                                        \
            }                                                                                      \
        } else {                                                                                   \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;              \
                 i += blockDim.x * gridDim.x) {                                                    \
                unsigned int idx = offset + get_strided_index(i, num_dims, dims, strides);         \
                out[i] = input[idx];                                                               \
            }                                                                                      \
        }                                                                                          \
    }

#define COPY_OP(TYPENAME, FN_NAME)                                                                 \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, TYPENAME *out,                       \
                                       const size_t *metadata) {                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *dims = metadata + 2;                                                         \
        const size_t *strides = metadata + 2 + num_dims;                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        bool cont = is_contiguous(num_dims, dims, strides);                                        \
        if (cont) {                                                                                \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;              \
                 i += blockDim.x * gridDim.x) {                                                    \
                out[i] = input[offset + i];                                                        \
            }                                                                                      \
        } else {                                                                                   \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;              \
                 i += blockDim.x * gridDim.x) {                                                    \
                unsigned int idx = offset + get_strided_index(i, num_dims, dims, strides);         \
                out[i] = input[idx];                                                               \
            }                                                                                      \
        }                                                                                          \
    }

CONTIGUOUS_OP(bool, contiguous_bool)
CONTIGUOUS_OP(__nv_fp8_e4m3, contiguous_f8e4m3)
CONTIGUOUS_OP(__nv_fp8_e5m2, contiguous_f8e5m2)
CONTIGUOUS_OP(__nv_bfloat16, contiguous_bf16)
CONTIGUOUS_OP(__half, contiguous_f16)
CONTIGUOUS_OP(float, contiguous_f32)
CONTIGUOUS_OP(double, contiguous_f64)
CONTIGUOUS_OP(uint8_t, contiguous_u8)
CONTIGUOUS_OP(uint16_t, contiguous_u16)
CONTIGUOUS_OP(uint32_t, contiguous_u32)
CONTIGUOUS_OP(uint64_t, contiguous_u64)
CONTIGUOUS_OP(int8_t, contiguous_i8)
CONTIGUOUS_OP(int16_t, contiguous_i16)
CONTIGUOUS_OP(int32_t, contiguous_i32)
CONTIGUOUS_OP(int64_t, contiguous_i64)

COPY_OP(bool, copy_bool)
COPY_OP(__nv_fp8_e4m3, copy_f8e4m3)
COPY_OP(__nv_fp8_e5m2, copy_f8e5m2)
COPY_OP(__nv_bfloat16, copy_bf16)
COPY_OP(__half, copy_f16)
COPY_OP(float, copy_f32)
COPY_OP(double, copy_f64)
COPY_OP(uint8_t, copy_u8)
COPY_OP(uint16_t, copy_u16)
COPY_OP(uint32_t, copy_u32)
COPY_OP(uint64_t, copy_u64)
COPY_OP(int8_t, copy_i8)
COPY_OP(int16_t, copy_i16)
COPY_OP(int32_t, copy_i32)
COPY_OP(int64_t, copy_i64)
