#include "./headers/utils.metal"
#include <metal_stdlib>

using namespace metal;

// Macro to define const_set kernel that fills a buffer with a constant value
#define CONST_SET_OP(TYPENAME, FN_NAME)                                                            \
    kernel void FN_NAME(                                                                           \
        device TYPENAME *output [[buffer(0)]], constant size_t &num_els [[buffer(1)]],             \
        constant size_t &num_dims [[buffer(2)]], constant size_t *metadata [[buffer(3)]],          \
        constant TYPENAME &const_val [[buffer(4)]], uint thread_index [[thread_position_in_grid]], \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
            const constant size_t *dims = metadata;                                                \
            const constant size_t *strides = metadata + num_dims;                                  \
            const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;                       \
                                                                                                   \
            if (metadata == nullptr || is_contiguous(num_dims, dims, strides)) {                   \
                output[offset + id] = const_val;                                                   \
            } else {                                                                               \
                unsigned strided_i = offset + get_strided_index(id, num_dims, dims, strides);      \
                output[strided_i] = const_val;                                                     \
            }                                                                                      \
        }                                                                                          \
    }

// Define const_set kernels for all supported types
CONST_SET_OP(bool, const_set_bool);
CONST_SET_OP(bfloat, const_set_bf16);
CONST_SET_OP(half, const_set_f16);
CONST_SET_OP(float, const_set_f32);
CONST_SET_OP(uint8_t, const_set_u8);
CONST_SET_OP(uint16_t, const_set_u16);
CONST_SET_OP(uint32_t, const_set_u32);
CONST_SET_OP(uint64_t, const_set_u64);
CONST_SET_OP(int8_t, const_set_i8);
CONST_SET_OP(int16_t, const_set_i16);
CONST_SET_OP(int32_t, const_set_i32);
CONST_SET_OP(int64_t, const_set_i64);
