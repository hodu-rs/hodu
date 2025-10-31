#ifndef HODU_CPU_KERNELS_OPS_BINARY_H
#define HODU_CPU_KERNELS_OPS_BINARY_H

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// BINARY OPERATION FUNCTION SIGNATURES
// ============================================================================

#define DECLARE_BINARY_OP(TYPE_SUFFIX)                                                             \
    void add_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,         \
                           size_t num_dims, const size_t *metadata);                               \
    void sub_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,         \
                           size_t num_dims, const size_t *metadata);                               \
    void mul_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,         \
                           size_t num_dims, const size_t *metadata);                               \
    void div_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,         \
                           size_t num_dims, const size_t *metadata);                               \
    void pow_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,         \
                           size_t num_dims, const size_t *metadata);                               \
    void maximum_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,     \
                               size_t num_dims, const size_t *metadata);                           \
    void minimum_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,     \
                               size_t num_dims, const size_t *metadata);

#define DECLARE_BINARY_LOGICAL(TYPE_SUFFIX)                                                        \
    void logical_and_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els, \
                                   size_t num_dims, const size_t *metadata);                       \
    void logical_or_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,  \
                                  size_t num_dims, const size_t *metadata);                        \
    void logical_xor_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els, \
                                   size_t num_dims, const size_t *metadata);

#define DECLARE_BINARY_CMP(TYPE_SUFFIX)                                                            \
    void eq_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,          \
                          size_t num_dims, const size_t *metadata);                                \
    void ne_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,          \
                          size_t num_dims, const size_t *metadata);                                \
    void lt_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,          \
                          size_t num_dims, const size_t *metadata);                                \
    void le_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,          \
                          size_t num_dims, const size_t *metadata);                                \
    void gt_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,          \
                          size_t num_dims, const size_t *metadata);                                \
    void ge_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,          \
                          size_t num_dims, const size_t *metadata);

// Bool type
DECLARE_BINARY_OP(bool)
DECLARE_BINARY_LOGICAL(bool)
DECLARE_BINARY_CMP(bool)

// Float types (f8e4m3, f8e5m2, bf16, f16, f32, f64)
DECLARE_BINARY_OP(f8e4m3)
DECLARE_BINARY_LOGICAL(f8e4m3)
DECLARE_BINARY_CMP(f8e4m3)

DECLARE_BINARY_OP(f8e5m2)
DECLARE_BINARY_LOGICAL(f8e5m2)
DECLARE_BINARY_CMP(f8e5m2)

DECLARE_BINARY_OP(bf16)
DECLARE_BINARY_LOGICAL(bf16)
DECLARE_BINARY_CMP(bf16)

DECLARE_BINARY_OP(f16)
DECLARE_BINARY_LOGICAL(f16)
DECLARE_BINARY_CMP(f16)

DECLARE_BINARY_OP(f32)
DECLARE_BINARY_LOGICAL(f32)
DECLARE_BINARY_CMP(f32)

DECLARE_BINARY_OP(f64)
DECLARE_BINARY_LOGICAL(f64)
DECLARE_BINARY_CMP(f64)

// Integer types (u8, u16, u32, u64, i8, i16, i32, i64)
DECLARE_BINARY_OP(u8)
DECLARE_BINARY_LOGICAL(u8)
DECLARE_BINARY_CMP(u8)

DECLARE_BINARY_OP(u16)
DECLARE_BINARY_LOGICAL(u16)
DECLARE_BINARY_CMP(u16)

DECLARE_BINARY_OP(u32)
DECLARE_BINARY_LOGICAL(u32)
DECLARE_BINARY_CMP(u32)

DECLARE_BINARY_OP(u64)
DECLARE_BINARY_LOGICAL(u64)
DECLARE_BINARY_CMP(u64)

DECLARE_BINARY_OP(i8)
DECLARE_BINARY_LOGICAL(i8)
DECLARE_BINARY_CMP(i8)

DECLARE_BINARY_OP(i16)
DECLARE_BINARY_LOGICAL(i16)
DECLARE_BINARY_CMP(i16)

DECLARE_BINARY_OP(i32)
DECLARE_BINARY_LOGICAL(i32)
DECLARE_BINARY_CMP(i32)

DECLARE_BINARY_OP(i64)
DECLARE_BINARY_LOGICAL(i64)
DECLARE_BINARY_CMP(i64)

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_OPS_BINARY_H
