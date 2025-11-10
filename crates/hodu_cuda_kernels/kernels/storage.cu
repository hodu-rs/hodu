#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CONST_SET_OP(TYPENAME, FN_NAME)                                                            \
    extern "C" __global__ void FN_NAME(TYPENAME *output, const TYPENAME const_val,                 \
                                       const size_t *metadata) {                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *dims = metadata + 2;                                                         \
        const size_t *strides = metadata + 2 + num_dims;                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        bool cont = is_contiguous(num_dims, dims, strides);                                        \
        if (cont) {                                                                                \
            for (uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_els;                \
                 id += blockDim.x * gridDim.x) {                                                   \
                output[offset + id] = const_val;                                                   \
            }                                                                                      \
        } else {                                                                                   \
            for (uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_els;                \
                 id += blockDim.x * gridDim.x) {                                                   \
                uint32_t strided_i = offset + get_strided_index(id, num_dims, dims, strides);      \
                output[strided_i] = const_val;                                                     \
            }                                                                                      \
        }                                                                                          \
    }

CONST_SET_OP(bool, const_set_bool)
CONST_SET_OP(__nv_fp8_e4m3, const_set_f8e4m3)
CONST_SET_OP(__nv_fp8_e5m2, const_set_f8e5m2)
CONST_SET_OP(__nv_bfloat16, const_set_bf16)
CONST_SET_OP(__half, const_set_f16)
CONST_SET_OP(float, const_set_f32)
CONST_SET_OP(double, const_set_f64)
CONST_SET_OP(uint8_t, const_set_u8)
CONST_SET_OP(uint16_t, const_set_u16)
CONST_SET_OP(uint32_t, const_set_u32)
CONST_SET_OP(uint64_t, const_set_u64)
CONST_SET_OP(int8_t, const_set_i8)
CONST_SET_OP(int16_t, const_set_i16)
CONST_SET_OP(int32_t, const_set_i32)
CONST_SET_OP(int64_t, const_set_i64)
