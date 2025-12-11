#include "ops_padding.h"
#include "types.h"

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

#define IMPL_PAD_CONSTANT_OP(TYPE, TYPE_SUFFIX)                                                    \
    void hodu_cpu_pad_constant_##TYPE_SUFFIX(const void *input, void *output,                      \
                                             const void *pad_value, const size_t *metadata) {      \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *output_shape = metadata + 2 + num_dims;                                      \
        const size_t *pad_before = metadata + 2 + 2 * num_dims;                                    \
                                                                                                   \
        const TYPE *in = (const TYPE *)input;                                                      \
        TYPE *out = (TYPE *)output;                                                                \
        TYPE pv = *(const TYPE *)pad_value;                                                        \
                                                                                                   \
        for (size_t i = 0; i < num_els; i++) {                                                     \
            size_t remaining = i;                                                                  \
            bool in_bounds = true;                                                                 \
            size_t input_idx = 0;                                                                  \
                                                                                                   \
            for (size_t d = num_dims; d-- > 0;) {                                                  \
                size_t out_coord = remaining % output_shape[d];                                    \
                remaining /= output_shape[d];                                                      \
                                                                                                   \
                long in_coord = (long)out_coord - (long)pad_before[d];                             \
                if (in_coord < 0 || (size_t)in_coord >= input_shape[d]) {                          \
                    in_bounds = false;                                                             \
                    break;                                                                         \
                }                                                                                  \
                                                                                                   \
                size_t in_stride = 1;                                                              \
                for (size_t k = d + 1; k < num_dims; k++) {                                        \
                    in_stride *= input_shape[k];                                                   \
                }                                                                                  \
                input_idx += (size_t)in_coord * in_stride;                                         \
            }                                                                                      \
                                                                                                   \
            out[i] = in_bounds ? in[input_idx] : pv;                                               \
        }                                                                                          \
    }

#define IMPL_PAD_REFLECT_OP(TYPE, TYPE_SUFFIX)                                                     \
    void hodu_cpu_pad_reflect_##TYPE_SUFFIX(const void *input, void *output,                       \
                                            const size_t *metadata) {                              \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *output_shape = metadata + 2 + num_dims;                                      \
        const size_t *pad_before = metadata + 2 + 2 * num_dims;                                    \
                                                                                                   \
        const TYPE *in = (const TYPE *)input;                                                      \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        for (size_t i = 0; i < num_els; i++) {                                                     \
            size_t remaining = i;                                                                  \
            size_t input_idx = 0;                                                                  \
                                                                                                   \
            for (size_t d = num_dims; d-- > 0;) {                                                  \
                size_t out_coord = remaining % output_shape[d];                                    \
                remaining /= output_shape[d];                                                      \
                                                                                                   \
                long in_coord = (long)out_coord - (long)pad_before[d];                             \
                size_t reflected = reflect_index(in_coord, input_shape[d]);                        \
                                                                                                   \
                size_t in_stride = 1;                                                              \
                for (size_t k = d + 1; k < num_dims; k++) {                                        \
                    in_stride *= input_shape[k];                                                   \
                }                                                                                  \
                input_idx += reflected * in_stride;                                                \
            }                                                                                      \
                                                                                                   \
            out[i] = in[input_idx];                                                                \
        }                                                                                          \
    }

#define IMPL_PAD_REPLICATE_OP(TYPE, TYPE_SUFFIX)                                                   \
    void hodu_cpu_pad_replicate_##TYPE_SUFFIX(const void *input, void *output,                     \
                                              const size_t *metadata) {                            \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *output_shape = metadata + 2 + num_dims;                                      \
        const size_t *pad_before = metadata + 2 + 2 * num_dims;                                    \
                                                                                                   \
        const TYPE *in = (const TYPE *)input;                                                      \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        for (size_t i = 0; i < num_els; i++) {                                                     \
            size_t remaining = i;                                                                  \
            size_t input_idx = 0;                                                                  \
                                                                                                   \
            for (size_t d = num_dims; d-- > 0;) {                                                  \
                size_t out_coord = remaining % output_shape[d];                                    \
                remaining /= output_shape[d];                                                      \
                                                                                                   \
                long in_coord = (long)out_coord - (long)pad_before[d];                             \
                size_t clamped = replicate_index(in_coord, input_shape[d]);                        \
                                                                                                   \
                size_t in_stride = 1;                                                              \
                for (size_t k = d + 1; k < num_dims; k++) {                                        \
                    in_stride *= input_shape[k];                                                   \
                }                                                                                  \
                input_idx += clamped * in_stride;                                                  \
            }                                                                                      \
                                                                                                   \
            out[i] = in[input_idx];                                                                \
        }                                                                                          \
    }

#define IMPL_PAD_CIRCULAR_OP(TYPE, TYPE_SUFFIX)                                                    \
    void hodu_cpu_pad_circular_##TYPE_SUFFIX(const void *input, void *output,                      \
                                             const size_t *metadata) {                             \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *output_shape = metadata + 2 + num_dims;                                      \
        const size_t *pad_before = metadata + 2 + 2 * num_dims;                                    \
                                                                                                   \
        const TYPE *in = (const TYPE *)input;                                                      \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        for (size_t i = 0; i < num_els; i++) {                                                     \
            size_t remaining = i;                                                                  \
            size_t input_idx = 0;                                                                  \
                                                                                                   \
            for (size_t d = num_dims; d-- > 0;) {                                                  \
                size_t out_coord = remaining % output_shape[d];                                    \
                remaining /= output_shape[d];                                                      \
                                                                                                   \
                long in_coord = (long)out_coord - (long)pad_before[d];                             \
                size_t wrapped = circular_index(in_coord, input_shape[d]);                         \
                                                                                                   \
                size_t in_stride = 1;                                                              \
                for (size_t k = d + 1; k < num_dims; k++) {                                        \
                    in_stride *= input_shape[k];                                                   \
                }                                                                                  \
                input_idx += wrapped * in_stride;                                                  \
            }                                                                                      \
                                                                                                   \
            out[i] = in[input_idx];                                                                \
        }                                                                                          \
    }

IMPL_PAD_CONSTANT_OP(uint8_t, bool)
IMPL_PAD_CONSTANT_OP(f8e4m3_t, f8e4m3)
IMPL_PAD_CONSTANT_OP(f8e5m2_t, f8e5m2)
IMPL_PAD_CONSTANT_OP(bf16_t, bf16)
IMPL_PAD_CONSTANT_OP(f16_t, f16)
IMPL_PAD_CONSTANT_OP(f32_t, f32)
IMPL_PAD_CONSTANT_OP(f64_t, f64)
IMPL_PAD_CONSTANT_OP(u8_t, u8)
IMPL_PAD_CONSTANT_OP(u16_t, u16)
IMPL_PAD_CONSTANT_OP(u32_t, u32)
IMPL_PAD_CONSTANT_OP(u64_t, u64)
IMPL_PAD_CONSTANT_OP(i8_t, i8)
IMPL_PAD_CONSTANT_OP(i16_t, i16)
IMPL_PAD_CONSTANT_OP(i32_t, i32)
IMPL_PAD_CONSTANT_OP(i64_t, i64)

IMPL_PAD_REFLECT_OP(uint8_t, bool)
IMPL_PAD_REFLECT_OP(f8e4m3_t, f8e4m3)
IMPL_PAD_REFLECT_OP(f8e5m2_t, f8e5m2)
IMPL_PAD_REFLECT_OP(bf16_t, bf16)
IMPL_PAD_REFLECT_OP(f16_t, f16)
IMPL_PAD_REFLECT_OP(f32_t, f32)
IMPL_PAD_REFLECT_OP(f64_t, f64)
IMPL_PAD_REFLECT_OP(u8_t, u8)
IMPL_PAD_REFLECT_OP(u16_t, u16)
IMPL_PAD_REFLECT_OP(u32_t, u32)
IMPL_PAD_REFLECT_OP(u64_t, u64)
IMPL_PAD_REFLECT_OP(i8_t, i8)
IMPL_PAD_REFLECT_OP(i16_t, i16)
IMPL_PAD_REFLECT_OP(i32_t, i32)
IMPL_PAD_REFLECT_OP(i64_t, i64)

IMPL_PAD_REPLICATE_OP(uint8_t, bool)
IMPL_PAD_REPLICATE_OP(f8e4m3_t, f8e4m3)
IMPL_PAD_REPLICATE_OP(f8e5m2_t, f8e5m2)
IMPL_PAD_REPLICATE_OP(bf16_t, bf16)
IMPL_PAD_REPLICATE_OP(f16_t, f16)
IMPL_PAD_REPLICATE_OP(f32_t, f32)
IMPL_PAD_REPLICATE_OP(f64_t, f64)
IMPL_PAD_REPLICATE_OP(u8_t, u8)
IMPL_PAD_REPLICATE_OP(u16_t, u16)
IMPL_PAD_REPLICATE_OP(u32_t, u32)
IMPL_PAD_REPLICATE_OP(u64_t, u64)
IMPL_PAD_REPLICATE_OP(i8_t, i8)
IMPL_PAD_REPLICATE_OP(i16_t, i16)
IMPL_PAD_REPLICATE_OP(i32_t, i32)
IMPL_PAD_REPLICATE_OP(i64_t, i64)

IMPL_PAD_CIRCULAR_OP(uint8_t, bool)
IMPL_PAD_CIRCULAR_OP(f8e4m3_t, f8e4m3)
IMPL_PAD_CIRCULAR_OP(f8e5m2_t, f8e5m2)
IMPL_PAD_CIRCULAR_OP(bf16_t, bf16)
IMPL_PAD_CIRCULAR_OP(f16_t, f16)
IMPL_PAD_CIRCULAR_OP(f32_t, f32)
IMPL_PAD_CIRCULAR_OP(f64_t, f64)
IMPL_PAD_CIRCULAR_OP(u8_t, u8)
IMPL_PAD_CIRCULAR_OP(u16_t, u16)
IMPL_PAD_CIRCULAR_OP(u32_t, u32)
IMPL_PAD_CIRCULAR_OP(u64_t, u64)
IMPL_PAD_CIRCULAR_OP(i8_t, i8)
IMPL_PAD_CIRCULAR_OP(i16_t, i16)
IMPL_PAD_CIRCULAR_OP(i32_t, i32)
IMPL_PAD_CIRCULAR_OP(i64_t, i64)
