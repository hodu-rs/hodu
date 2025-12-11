#ifndef HODU_CPU_KERNELS_OPS_PADDING_H
#define HODU_CPU_KERNELS_OPS_PADDING_H

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// Metadata layout for all padding operations:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: output_shape
// - metadata[2+2*num_dims..2+3*num_dims]: pad_before

#define DECLARE_PAD_CONSTANT_OP(TYPE_SUFFIX)                                                       \
    void hodu_cpu_pad_constant_##TYPE_SUFFIX(const void *input, void *output,                      \
                                             const void *pad_value, const size_t *metadata);

#define DECLARE_PAD_REFLECT_OP(TYPE_SUFFIX)                                                        \
    void hodu_cpu_pad_reflect_##TYPE_SUFFIX(const void *input, void *output,                       \
                                            const size_t *metadata);

#define DECLARE_PAD_REPLICATE_OP(TYPE_SUFFIX)                                                      \
    void hodu_cpu_pad_replicate_##TYPE_SUFFIX(const void *input, void *output,                     \
                                              const size_t *metadata);

#define DECLARE_PAD_CIRCULAR_OP(TYPE_SUFFIX)                                                       \
    void hodu_cpu_pad_circular_##TYPE_SUFFIX(const void *input, void *output,                      \
                                             const size_t *metadata);

DECLARE_PAD_CONSTANT_OP(bool)
DECLARE_PAD_CONSTANT_OP(f8e4m3)
DECLARE_PAD_CONSTANT_OP(f8e5m2)
DECLARE_PAD_CONSTANT_OP(bf16)
DECLARE_PAD_CONSTANT_OP(f16)
DECLARE_PAD_CONSTANT_OP(f32)
DECLARE_PAD_CONSTANT_OP(f64)
DECLARE_PAD_CONSTANT_OP(u8)
DECLARE_PAD_CONSTANT_OP(u16)
DECLARE_PAD_CONSTANT_OP(u32)
DECLARE_PAD_CONSTANT_OP(u64)
DECLARE_PAD_CONSTANT_OP(i8)
DECLARE_PAD_CONSTANT_OP(i16)
DECLARE_PAD_CONSTANT_OP(i32)
DECLARE_PAD_CONSTANT_OP(i64)

DECLARE_PAD_REFLECT_OP(bool)
DECLARE_PAD_REFLECT_OP(f8e4m3)
DECLARE_PAD_REFLECT_OP(f8e5m2)
DECLARE_PAD_REFLECT_OP(bf16)
DECLARE_PAD_REFLECT_OP(f16)
DECLARE_PAD_REFLECT_OP(f32)
DECLARE_PAD_REFLECT_OP(f64)
DECLARE_PAD_REFLECT_OP(u8)
DECLARE_PAD_REFLECT_OP(u16)
DECLARE_PAD_REFLECT_OP(u32)
DECLARE_PAD_REFLECT_OP(u64)
DECLARE_PAD_REFLECT_OP(i8)
DECLARE_PAD_REFLECT_OP(i16)
DECLARE_PAD_REFLECT_OP(i32)
DECLARE_PAD_REFLECT_OP(i64)

DECLARE_PAD_REPLICATE_OP(bool)
DECLARE_PAD_REPLICATE_OP(f8e4m3)
DECLARE_PAD_REPLICATE_OP(f8e5m2)
DECLARE_PAD_REPLICATE_OP(bf16)
DECLARE_PAD_REPLICATE_OP(f16)
DECLARE_PAD_REPLICATE_OP(f32)
DECLARE_PAD_REPLICATE_OP(f64)
DECLARE_PAD_REPLICATE_OP(u8)
DECLARE_PAD_REPLICATE_OP(u16)
DECLARE_PAD_REPLICATE_OP(u32)
DECLARE_PAD_REPLICATE_OP(u64)
DECLARE_PAD_REPLICATE_OP(i8)
DECLARE_PAD_REPLICATE_OP(i16)
DECLARE_PAD_REPLICATE_OP(i32)
DECLARE_PAD_REPLICATE_OP(i64)

DECLARE_PAD_CIRCULAR_OP(bool)
DECLARE_PAD_CIRCULAR_OP(f8e4m3)
DECLARE_PAD_CIRCULAR_OP(f8e5m2)
DECLARE_PAD_CIRCULAR_OP(bf16)
DECLARE_PAD_CIRCULAR_OP(f16)
DECLARE_PAD_CIRCULAR_OP(f32)
DECLARE_PAD_CIRCULAR_OP(f64)
DECLARE_PAD_CIRCULAR_OP(u8)
DECLARE_PAD_CIRCULAR_OP(u16)
DECLARE_PAD_CIRCULAR_OP(u32)
DECLARE_PAD_CIRCULAR_OP(u64)
DECLARE_PAD_CIRCULAR_OP(i8)
DECLARE_PAD_CIRCULAR_OP(i16)
DECLARE_PAD_CIRCULAR_OP(i32)
DECLARE_PAD_CIRCULAR_OP(i64)

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_OPS_PADDING_H
