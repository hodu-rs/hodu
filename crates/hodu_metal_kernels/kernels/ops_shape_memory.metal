#include <metal_stdlib>
using namespace metal;

// Flip operation: reverses tensor along specified dimensions
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: flip_mask (1 = flip this dim, 0 = don't flip)

#define FLIP_OP(TYPE, TYPE_SUFFIX)                                                                 \
    kernel void hodu_metal_flip_##TYPE_SUFFIX(                                                     \
        device const TYPE *input [[buffer(0)]], device TYPE *output [[buffer(1)]],                 \
        constant size_t *metadata [[buffer(2)]], uint tid [[thread_position_in_grid]]) {           \
        const size_t num_els = metadata[0];                                                        \
        if (tid >= num_els)                                                                        \
            return;                                                                                \
                                                                                                   \
        const size_t num_dims = metadata[1];                                                       \
        constant size_t *shape = metadata + 2;                                                     \
        constant size_t *flip_mask = metadata + 2 + num_dims;                                      \
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
FLIP_OP(bfloat, bf16)
FLIP_OP(half, f16)
FLIP_OP(float, f32)
FLIP_OP(uint8_t, u8)
FLIP_OP(uint16_t, u16)
FLIP_OP(uint32_t, u32)
FLIP_OP(uint64_t, u64)
FLIP_OP(int8_t, i8)
FLIP_OP(int16_t, i16)
FLIP_OP(int32_t, i32)
FLIP_OP(int64_t, i64)
