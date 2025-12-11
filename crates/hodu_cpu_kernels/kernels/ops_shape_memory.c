#include "ops_shape_memory.h"
#include "types.h"
#include <stdbool.h>

// Flip operation: reverses tensor along specified dimensions
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: flip_mask (1 = flip this dim, 0 = don't flip)

#define IMPL_FLIP_OP(TYPE, TYPE_SUFFIX)                                                            \
    void hodu_cpu_flip_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata) {    \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *shape = metadata + 2;                                                        \
        const size_t *flip_mask = metadata + 2 + num_dims;                                         \
                                                                                                   \
        const TYPE *in = (const TYPE *)input;                                                      \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        for (size_t i = 0; i < num_els; i++) {                                                     \
            size_t remaining = i;                                                                  \
            size_t input_idx = 0;                                                                  \
            size_t in_stride = 1;                                                                  \
                                                                                                   \
            for (size_t d = num_dims; d-- > 0;) {                                                  \
                size_t coord = remaining % shape[d];                                               \
                remaining /= shape[d];                                                             \
                                                                                                   \
                size_t in_coord = flip_mask[d] ? (shape[d] - 1 - coord) : coord;                   \
                input_idx += in_coord * in_stride;                                                 \
                in_stride *= shape[d];                                                             \
            }                                                                                      \
                                                                                                   \
            out[i] = in[input_idx];                                                                \
        }                                                                                          \
    }

IMPL_FLIP_OP(bool, bool)
IMPL_FLIP_OP(f8e4m3_t, f8e4m3)
IMPL_FLIP_OP(f8e5m2_t, f8e5m2)
IMPL_FLIP_OP(bf16_t, bf16)
IMPL_FLIP_OP(f16_t, f16)
IMPL_FLIP_OP(float, f32)
IMPL_FLIP_OP(double, f64)
IMPL_FLIP_OP(uint8_t, u8)
IMPL_FLIP_OP(uint16_t, u16)
IMPL_FLIP_OP(uint32_t, u32)
IMPL_FLIP_OP(uint64_t, u64)
IMPL_FLIP_OP(int8_t, i8)
IMPL_FLIP_OP(int16_t, i16)
IMPL_FLIP_OP(int32_t, i32)
IMPL_FLIP_OP(int64_t, i64)
