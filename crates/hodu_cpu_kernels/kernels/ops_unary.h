#ifndef HODU_CPU_KERNELS_OPS_UNARY_H
#define HODU_CPU_KERNELS_OPS_UNARY_H

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// UNARY OPERATION FUNCTION SIGNATURES
// ============================================================================

// Macro to declare unary operation functions
// Each function processes num_els elements with optional strided access
#define DECLARE_UNARY_OP(TYPE_SUFFIX)                                                              \
    void neg_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,       \
                           const size_t *metadata);                                                \
    void abs_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,       \
                           const size_t *metadata);                                                \
    void sign_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,      \
                            const size_t *metadata);                                               \
    void square_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,    \
                              const size_t *metadata);                                             \
    void sqrt_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,      \
                            const size_t *metadata);                                               \
    void recip_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,     \
                             const size_t *metadata);

#define DECLARE_UNARY_ACTIVATION(TYPE_SUFFIX)                                                      \
    void relu_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,      \
                            const size_t *metadata);                                               \
    void sigmoid_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,   \
                               const size_t *metadata);                                            \
    void tanh_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,      \
                            const size_t *metadata);                                               \
    void gelu_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,      \
                            const size_t *metadata);                                               \
    void softplus_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,  \
                                const size_t *metadata);                                           \
    void silu_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,      \
                            const size_t *metadata);                                               \
    void mish_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,      \
                            const size_t *metadata);

#define DECLARE_UNARY_TRIG(TYPE_SUFFIX)                                                            \
    void sin_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,       \
                           const size_t *metadata);                                                \
    void cos_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,       \
                           const size_t *metadata);                                                \
    void tan_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,       \
                           const size_t *metadata);

#define DECLARE_UNARY_EXP(TYPE_SUFFIX)                                                             \
    void exp_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,       \
                           const size_t *metadata);                                                \
    void exp2_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,      \
                            const size_t *metadata);                                               \
    void exp10_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,     \
                             const size_t *metadata);                                              \
    void ln_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,        \
                          const size_t *metadata);                                                 \
    void log2_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,      \
                            const size_t *metadata);                                               \
    void log10_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims,     \
                             const size_t *metadata);

#define DECLARE_UNARY_LOGICAL(TYPE_SUFFIX)                                                         \
    void logical_not_##TYPE_SUFFIX(const void *input, void *output, size_t num_els,                \
                                   size_t num_dims, const size_t *metadata);

#define DECLARE_UNARY_WITH_SCALAR(TYPE_SUFFIX)                                                     \
    void add_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els,                 \
                                  size_t num_dims, const size_t *metadata, const void *scalar);    \
    void sub_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els,                 \
                                  size_t num_dims, const size_t *metadata, const void *scalar);    \
    void mul_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els,                 \
                                  size_t num_dims, const size_t *metadata, const void *scalar);    \
    void div_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els,                 \
                                  size_t num_dims, const size_t *metadata, const void *scalar);    \
    void pow_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els,                 \
                                  size_t num_dims, const size_t *metadata, const void *scalar);    \
    void maximum_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els,             \
                                      size_t num_dims, const size_t *metadata,                     \
                                      const void *scalar);                                         \
    void minimum_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els,             \
                                      size_t num_dims, const size_t *metadata,                     \
                                      const void *scalar);

#define DECLARE_UNARY_CMP_SCALAR(TYPE_SUFFIX)                                                      \
    void eq_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata, const void *scalar);                      \
    void ne_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata, const void *scalar);                      \
    void lt_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata, const void *scalar);                      \
    void le_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata, const void *scalar);                      \
    void gt_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata, const void *scalar);                      \
    void ge_scalar_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata, const void *scalar);

// Bool type
DECLARE_UNARY_OP(bool)
DECLARE_UNARY_ACTIVATION(bool)
DECLARE_UNARY_TRIG(bool)
DECLARE_UNARY_EXP(bool)
DECLARE_UNARY_LOGICAL(bool)
DECLARE_UNARY_WITH_SCALAR(bool)
DECLARE_UNARY_CMP_SCALAR(bool)

// Float types (f8e4m3, f8e5m2, bf16, f16, f32, f64)
DECLARE_UNARY_OP(f8e4m3)
DECLARE_UNARY_ACTIVATION(f8e4m3)
DECLARE_UNARY_TRIG(f8e4m3)
DECLARE_UNARY_EXP(f8e4m3)
DECLARE_UNARY_LOGICAL(f8e4m3)
DECLARE_UNARY_WITH_SCALAR(f8e4m3)
DECLARE_UNARY_CMP_SCALAR(f8e4m3)

DECLARE_UNARY_OP(f8e5m2)
DECLARE_UNARY_ACTIVATION(f8e5m2)
DECLARE_UNARY_TRIG(f8e5m2)
DECLARE_UNARY_EXP(f8e5m2)
DECLARE_UNARY_LOGICAL(f8e5m2)
DECLARE_UNARY_WITH_SCALAR(f8e5m2)
DECLARE_UNARY_CMP_SCALAR(f8e5m2)

DECLARE_UNARY_OP(bf16)
DECLARE_UNARY_ACTIVATION(bf16)
DECLARE_UNARY_TRIG(bf16)
DECLARE_UNARY_EXP(bf16)
DECLARE_UNARY_LOGICAL(bf16)
DECLARE_UNARY_WITH_SCALAR(bf16)
DECLARE_UNARY_CMP_SCALAR(bf16)

DECLARE_UNARY_OP(f16)
DECLARE_UNARY_ACTIVATION(f16)
DECLARE_UNARY_TRIG(f16)
DECLARE_UNARY_EXP(f16)
DECLARE_UNARY_LOGICAL(f16)
DECLARE_UNARY_WITH_SCALAR(f16)
DECLARE_UNARY_CMP_SCALAR(f16)

DECLARE_UNARY_OP(f32)
DECLARE_UNARY_ACTIVATION(f32)
DECLARE_UNARY_TRIG(f32)
DECLARE_UNARY_EXP(f32)
DECLARE_UNARY_LOGICAL(f32)
DECLARE_UNARY_WITH_SCALAR(f32)
DECLARE_UNARY_CMP_SCALAR(f32)

DECLARE_UNARY_OP(f64)
DECLARE_UNARY_ACTIVATION(f64)
DECLARE_UNARY_TRIG(f64)
DECLARE_UNARY_EXP(f64)
DECLARE_UNARY_LOGICAL(f64)
DECLARE_UNARY_WITH_SCALAR(f64)
DECLARE_UNARY_CMP_SCALAR(f64)

// Integer types (u8, u16, u32, u64, i8, i16, i32, i64)
DECLARE_UNARY_OP(u8)
DECLARE_UNARY_LOGICAL(u8)
DECLARE_UNARY_WITH_SCALAR(u8)
DECLARE_UNARY_CMP_SCALAR(u8)

DECLARE_UNARY_OP(u16)
DECLARE_UNARY_LOGICAL(u16)
DECLARE_UNARY_WITH_SCALAR(u16)
DECLARE_UNARY_CMP_SCALAR(u16)

DECLARE_UNARY_OP(u32)
DECLARE_UNARY_LOGICAL(u32)
DECLARE_UNARY_WITH_SCALAR(u32)
DECLARE_UNARY_CMP_SCALAR(u32)

DECLARE_UNARY_OP(u64)
DECLARE_UNARY_LOGICAL(u64)
DECLARE_UNARY_WITH_SCALAR(u64)
DECLARE_UNARY_CMP_SCALAR(u64)

DECLARE_UNARY_OP(i8)
DECLARE_UNARY_LOGICAL(i8)
DECLARE_UNARY_WITH_SCALAR(i8)
DECLARE_UNARY_CMP_SCALAR(i8)

DECLARE_UNARY_OP(i16)
DECLARE_UNARY_LOGICAL(i16)
DECLARE_UNARY_WITH_SCALAR(i16)
DECLARE_UNARY_CMP_SCALAR(i16)

DECLARE_UNARY_OP(i32)
DECLARE_UNARY_LOGICAL(i32)
DECLARE_UNARY_WITH_SCALAR(i32)
DECLARE_UNARY_CMP_SCALAR(i32)

DECLARE_UNARY_OP(i64)
DECLARE_UNARY_LOGICAL(i64)
DECLARE_UNARY_WITH_SCALAR(i64)
DECLARE_UNARY_CMP_SCALAR(i64)

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_OPS_UNARY_H
