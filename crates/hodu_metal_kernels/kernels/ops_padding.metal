#include "./headers/utils.metal"
#include <metal_stdlib>

using namespace metal;

static inline size_t reflect_index(long idx, size_t size) {
    if (idx < 0) {
        idx = -idx;
    }
    if ((size_t)idx >= size) {
        size_t period = 2 * (size - 1);
        if (period > 0) {
            idx = idx % period;
            if ((size_t)idx >= size) {
                idx = period - idx;
            }
        } else {
            idx = 0;
        }
    }
    return (size_t)idx;
}

static inline size_t replicate_index(long idx, size_t size) {
    if (idx < 0)
        return 0;
    if ((size_t)idx >= size)
        return size - 1;
    return (size_t)idx;
}

static inline size_t circular_index(long idx, size_t size) {
    long result = idx % (long)size;
    if (result < 0)
        result += (long)size;
    return (size_t)result;
}

#define PAD_CONSTANT_OP(TYPENAME, FN_NAME)                                                         \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        const device TYPENAME *pad_value [[buffer(2)]], constant size_t *metadata [[buffer(3)]],   \
        uint id [[thread_position_in_grid]]) {                                                     \
        const size_t num_els = metadata[0];                                                        \
        if (id >= num_els)                                                                         \
            return;                                                                                \
                                                                                                   \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *input_shape = metadata + 2;                                         \
        const constant size_t *output_shape = metadata + 2 + num_dims;                             \
        const constant size_t *pad_before = metadata + 2 + 2 * num_dims;                           \
                                                                                                   \
        TYPENAME pv = pad_value[0];                                                                \
        size_t remaining = id;                                                                     \
        bool in_bounds = true;                                                                     \
        size_t input_idx = 0;                                                                      \
                                                                                                   \
        for (size_t d = num_dims; d-- > 0;) {                                                      \
            size_t out_coord = remaining % output_shape[d];                                        \
            remaining /= output_shape[d];                                                          \
                                                                                                   \
            long in_coord = (long)out_coord - (long)pad_before[d];                                 \
            if (in_coord < 0 || (size_t)in_coord >= input_shape[d]) {                              \
                in_bounds = false;                                                                 \
                break;                                                                             \
            }                                                                                      \
                                                                                                   \
            size_t in_stride = 1;                                                                  \
            for (size_t k = d + 1; k < num_dims; k++) {                                            \
                in_stride *= input_shape[k];                                                       \
            }                                                                                      \
            input_idx += (size_t)in_coord * in_stride;                                             \
        }                                                                                          \
                                                                                                   \
        output[id] = in_bounds ? input[input_idx] : pv;                                            \
    }

#define PAD_REFLECT_OP(TYPENAME, FN_NAME)                                                          \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t *metadata [[buffer(2)]], uint id [[thread_position_in_grid]]) {            \
        const size_t num_els = metadata[0];                                                        \
        if (id >= num_els)                                                                         \
            return;                                                                                \
                                                                                                   \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *input_shape = metadata + 2;                                         \
        const constant size_t *output_shape = metadata + 2 + num_dims;                             \
        const constant size_t *pad_before = metadata + 2 + 2 * num_dims;                           \
                                                                                                   \
        size_t remaining = id;                                                                     \
        size_t input_idx = 0;                                                                      \
                                                                                                   \
        for (size_t d = num_dims; d-- > 0;) {                                                      \
            size_t out_coord = remaining % output_shape[d];                                        \
            remaining /= output_shape[d];                                                          \
                                                                                                   \
            long in_coord = (long)out_coord - (long)pad_before[d];                                 \
            size_t reflected = reflect_index(in_coord, input_shape[d]);                            \
                                                                                                   \
            size_t in_stride = 1;                                                                  \
            for (size_t k = d + 1; k < num_dims; k++) {                                            \
                in_stride *= input_shape[k];                                                       \
            }                                                                                      \
            input_idx += reflected * in_stride;                                                    \
        }                                                                                          \
                                                                                                   \
        output[id] = input[input_idx];                                                             \
    }

#define PAD_REPLICATE_OP(TYPENAME, FN_NAME)                                                        \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t *metadata [[buffer(2)]], uint id [[thread_position_in_grid]]) {            \
        const size_t num_els = metadata[0];                                                        \
        if (id >= num_els)                                                                         \
            return;                                                                                \
                                                                                                   \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *input_shape = metadata + 2;                                         \
        const constant size_t *output_shape = metadata + 2 + num_dims;                             \
        const constant size_t *pad_before = metadata + 2 + 2 * num_dims;                           \
                                                                                                   \
        size_t remaining = id;                                                                     \
        size_t input_idx = 0;                                                                      \
                                                                                                   \
        for (size_t d = num_dims; d-- > 0;) {                                                      \
            size_t out_coord = remaining % output_shape[d];                                        \
            remaining /= output_shape[d];                                                          \
                                                                                                   \
            long in_coord = (long)out_coord - (long)pad_before[d];                                 \
            size_t clamped = replicate_index(in_coord, input_shape[d]);                            \
                                                                                                   \
            size_t in_stride = 1;                                                                  \
            for (size_t k = d + 1; k < num_dims; k++) {                                            \
                in_stride *= input_shape[k];                                                       \
            }                                                                                      \
            input_idx += clamped * in_stride;                                                      \
        }                                                                                          \
                                                                                                   \
        output[id] = input[input_idx];                                                             \
    }

#define PAD_CIRCULAR_OP(TYPENAME, FN_NAME)                                                         \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t *metadata [[buffer(2)]], uint id [[thread_position_in_grid]]) {            \
        const size_t num_els = metadata[0];                                                        \
        if (id >= num_els)                                                                         \
            return;                                                                                \
                                                                                                   \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *input_shape = metadata + 2;                                         \
        const constant size_t *output_shape = metadata + 2 + num_dims;                             \
        const constant size_t *pad_before = metadata + 2 + 2 * num_dims;                           \
                                                                                                   \
        size_t remaining = id;                                                                     \
        size_t input_idx = 0;                                                                      \
                                                                                                   \
        for (size_t d = num_dims; d-- > 0;) {                                                      \
            size_t out_coord = remaining % output_shape[d];                                        \
            remaining /= output_shape[d];                                                          \
                                                                                                   \
            long in_coord = (long)out_coord - (long)pad_before[d];                                 \
            size_t wrapped = circular_index(in_coord, input_shape[d]);                             \
                                                                                                   \
            size_t in_stride = 1;                                                                  \
            for (size_t k = d + 1; k < num_dims; k++) {                                            \
                in_stride *= input_shape[k];                                                       \
            }                                                                                      \
            input_idx += wrapped * in_stride;                                                      \
        }                                                                                          \
                                                                                                   \
        output[id] = input[input_idx];                                                             \
    }

PAD_CONSTANT_OP(bool, pad_constant_bool);
PAD_CONSTANT_OP(bfloat, pad_constant_bf16);
PAD_CONSTANT_OP(half, pad_constant_f16);
PAD_CONSTANT_OP(float, pad_constant_f32);
PAD_CONSTANT_OP(uint8_t, pad_constant_u8);
PAD_CONSTANT_OP(uint16_t, pad_constant_u16);
PAD_CONSTANT_OP(uint32_t, pad_constant_u32);
PAD_CONSTANT_OP(uint64_t, pad_constant_u64);
PAD_CONSTANT_OP(int8_t, pad_constant_i8);
PAD_CONSTANT_OP(int16_t, pad_constant_i16);
PAD_CONSTANT_OP(int32_t, pad_constant_i32);
PAD_CONSTANT_OP(int64_t, pad_constant_i64);

PAD_REFLECT_OP(bool, pad_reflect_bool);
PAD_REFLECT_OP(bfloat, pad_reflect_bf16);
PAD_REFLECT_OP(half, pad_reflect_f16);
PAD_REFLECT_OP(float, pad_reflect_f32);
PAD_REFLECT_OP(uint8_t, pad_reflect_u8);
PAD_REFLECT_OP(uint16_t, pad_reflect_u16);
PAD_REFLECT_OP(uint32_t, pad_reflect_u32);
PAD_REFLECT_OP(uint64_t, pad_reflect_u64);
PAD_REFLECT_OP(int8_t, pad_reflect_i8);
PAD_REFLECT_OP(int16_t, pad_reflect_i16);
PAD_REFLECT_OP(int32_t, pad_reflect_i32);
PAD_REFLECT_OP(int64_t, pad_reflect_i64);

PAD_REPLICATE_OP(bool, pad_replicate_bool);
PAD_REPLICATE_OP(bfloat, pad_replicate_bf16);
PAD_REPLICATE_OP(half, pad_replicate_f16);
PAD_REPLICATE_OP(float, pad_replicate_f32);
PAD_REPLICATE_OP(uint8_t, pad_replicate_u8);
PAD_REPLICATE_OP(uint16_t, pad_replicate_u16);
PAD_REPLICATE_OP(uint32_t, pad_replicate_u32);
PAD_REPLICATE_OP(uint64_t, pad_replicate_u64);
PAD_REPLICATE_OP(int8_t, pad_replicate_i8);
PAD_REPLICATE_OP(int16_t, pad_replicate_i16);
PAD_REPLICATE_OP(int32_t, pad_replicate_i32);
PAD_REPLICATE_OP(int64_t, pad_replicate_i64);

PAD_CIRCULAR_OP(bool, pad_circular_bool);
PAD_CIRCULAR_OP(bfloat, pad_circular_bf16);
PAD_CIRCULAR_OP(half, pad_circular_f16);
PAD_CIRCULAR_OP(float, pad_circular_f32);
PAD_CIRCULAR_OP(uint8_t, pad_circular_u8);
PAD_CIRCULAR_OP(uint16_t, pad_circular_u16);
PAD_CIRCULAR_OP(uint32_t, pad_circular_u32);
PAD_CIRCULAR_OP(uint64_t, pad_circular_u64);
PAD_CIRCULAR_OP(int8_t, pad_circular_i8);
PAD_CIRCULAR_OP(int16_t, pad_circular_i16);
PAD_CIRCULAR_OP(int32_t, pad_circular_i32);
PAD_CIRCULAR_OP(int64_t, pad_circular_i64);
