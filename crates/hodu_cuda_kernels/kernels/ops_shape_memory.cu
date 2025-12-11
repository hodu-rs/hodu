#include "cuda_fp8.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Flip operation: reverses tensor along specified dimensions
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: flip_mask (1 = flip this dim, 0 = don't flip)

#define FLIP_OP(TYPE, TYPE_SUFFIX)                                                                 \
    extern "C" __global__ void hodu_cuda_flip_##TYPE_SUFFIX(const TYPE *input, TYPE *output,       \
                                                            const size_t *metadata) {              \
        const size_t num_els = metadata[0];                                                        \
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;                                  \
        if (tid >= num_els)                                                                        \
            return;                                                                                \
                                                                                                   \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *shape = metadata + 2;                                                        \
        const size_t *flip_mask = metadata + 2 + num_dims;                                         \
                                                                                                   \
        size_t remaining = tid;                                                                    \
        size_t input_idx = 0;                                                                      \
        size_t in_stride = 1;                                                                      \
                                                                                                   \
        for (size_t d = num_dims; d-- > 0;) {                                                      \
            size_t coord = remaining % shape[d];                                                   \
            remaining /= shape[d];                                                                 \
                                                                                                   \
            size_t in_coord = flip_mask[d] ? (shape[d] - 1 - coord) : coord;                       \
            input_idx += in_coord * in_stride;                                                     \
            in_stride *= shape[d];                                                                 \
        }                                                                                          \
                                                                                                   \
        output[tid] = input[input_idx];                                                            \
    }

FLIP_OP(bool, bool)
FLIP_OP(__nv_fp8_e4m3, f8e4m3)
FLIP_OP(__nv_fp8_e5m2, f8e5m2)
FLIP_OP(__nv_bfloat16, bf16)
FLIP_OP(__half, f16)
FLIP_OP(float, f32)
FLIP_OP(double, f64)
FLIP_OP(uint8_t, u8)
FLIP_OP(uint16_t, u16)
FLIP_OP(uint32_t, u32)
FLIP_OP(uint64_t, u64)
FLIP_OP(int8_t, i8)
FLIP_OP(int16_t, i16)
FLIP_OP(int32_t, i32)
FLIP_OP(int64_t, i64)
