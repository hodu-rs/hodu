#ifndef HODU_OPS_SHAPE_MEMORY_H
#define HODU_OPS_SHAPE_MEMORY_H

#include <stddef.h>

#define DECLARE_FLIP_OP(TYPE_SUFFIX)                                                               \
    void hodu_cpu_flip_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);

DECLARE_FLIP_OP(bool)
DECLARE_FLIP_OP(f8e4m3)
DECLARE_FLIP_OP(f8e5m2)
DECLARE_FLIP_OP(bf16)
DECLARE_FLIP_OP(f16)
DECLARE_FLIP_OP(f32)
DECLARE_FLIP_OP(f64)
DECLARE_FLIP_OP(u8)
DECLARE_FLIP_OP(u16)
DECLARE_FLIP_OP(u32)
DECLARE_FLIP_OP(u64)
DECLARE_FLIP_OP(i8)
DECLARE_FLIP_OP(i16)
DECLARE_FLIP_OP(i32)
DECLARE_FLIP_OP(i64)

#endif
