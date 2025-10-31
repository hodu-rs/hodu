#include "ops_unary.h"
#include <math.h>

// ============================================================================
// UNARY OPERATION IMPLEMENTATION MACROS
// ============================================================================

#define IMPL_UNARY_OP(TYPE, TYPE_SUFFIX, OP_NAME, FUNC)                                            \
    void OP_NAME##_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata) {                                         \
        const TYPE *in = (const TYPE *)input;                                                      \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        const size_t *dims = metadata;                                                             \
        const size_t *strides = metadata ? metadata + num_dims : NULL;                             \
        const size_t offset = (metadata && num_dims > 0) ? metadata[2 * num_dims] : 0;             \
                                                                                                   \
        bool contiguous = (metadata == NULL) || is_contiguous(num_dims, dims, strides);            \
                                                                                                   \
        if (contiguous) {                                                                          \
            if (in) {                                                                              \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    TYPE x = in[offset + i];                                                       \
                    out[i] = FUNC;                                                                 \
                }                                                                                  \
            } else {                                                                               \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    TYPE x = out[i];                                                               \
                    out[i] = FUNC;                                                                 \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t strided_i = offset + get_strided_index(i, num_dims, dims, strides);         \
                TYPE x = in ? in[strided_i] : out[i];                                              \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        }                                                                                          \
    }

#define IMPL_UNARY_WITH_SCALAR(TYPE, TYPE_SUFFIX, OP_NAME, FUNC)                                   \
    void OP_NAME##_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata, const void *scalar) {                     \
        const TYPE *in = (const TYPE *)input;                                                      \
        TYPE *out = (TYPE *)output;                                                                \
        TYPE const_val = *(const TYPE *)scalar;                                                    \
                                                                                                   \
        const size_t *dims = metadata;                                                             \
        const size_t *strides = metadata ? metadata + num_dims : NULL;                             \
        const size_t offset = (metadata && num_dims > 0) ? metadata[2 * num_dims] : 0;             \
                                                                                                   \
        bool contiguous = (metadata == NULL) || is_contiguous(num_dims, dims, strides);            \
                                                                                                   \
        if (contiguous) {                                                                          \
            if (in) {                                                                              \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    TYPE x = in[offset + i];                                                       \
                    out[i] = FUNC;                                                                 \
                }                                                                                  \
            } else {                                                                               \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    TYPE x = out[i];                                                               \
                    out[i] = FUNC;                                                                 \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t strided_i = offset + get_strided_index(i, num_dims, dims, strides);         \
                TYPE x = in ? in[strided_i] : out[i];                                              \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        }                                                                                          \
    }

#define IMPL_UNARY_TO_BOOL(TYPE, TYPE_SUFFIX, OP_NAME, FUNC)                                       \
    void OP_NAME##_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata) {                                         \
        const TYPE *in = (const TYPE *)input;                                                      \
        uint8_t *out = (uint8_t *)output;                                                          \
                                                                                                   \
        const size_t *dims = metadata;                                                             \
        const size_t *strides = metadata ? metadata + num_dims : NULL;                             \
        const size_t offset = (metadata && num_dims > 0) ? metadata[2 * num_dims] : 0;             \
                                                                                                   \
        bool contiguous = (metadata == NULL) || is_contiguous(num_dims, dims, strides);            \
                                                                                                   \
        if (contiguous) {                                                                          \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                TYPE x = in[offset + i];                                                           \
                out[i] = (FUNC) ? 1 : 0;                                                           \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t strided_i = offset + get_strided_index(i, num_dims, dims, strides);         \
                TYPE x = in[strided_i];                                                            \
                out[i] = (FUNC) ? 1 : 0;                                                           \
            }                                                                                      \
        }                                                                                          \
    }

#define IMPL_UNARY_CMP_SCALAR_TO_BOOL(TYPE, TYPE_SUFFIX, OP_NAME, FUNC)                            \
    void OP_NAME##_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata, const void *scalar) {                     \
        const TYPE *in = (const TYPE *)input;                                                      \
        uint8_t *out = (uint8_t *)output;                                                          \
        TYPE const_val = *(const TYPE *)scalar;                                                    \
                                                                                                   \
        const size_t *dims = metadata;                                                             \
        const size_t *strides = metadata ? metadata + num_dims : NULL;                             \
        const size_t offset = (metadata && num_dims > 0) ? metadata[2 * num_dims] : 0;             \
                                                                                                   \
        bool contiguous = (metadata == NULL) || is_contiguous(num_dims, dims, strides);            \
                                                                                                   \
        if (contiguous) {                                                                          \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                TYPE x = in ? in[offset + i] : ((TYPE *)out)[i];                                   \
                out[i] = (FUNC) ? 1 : 0;                                                           \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t strided_i = offset + get_strided_index(i, num_dims, dims, strides);         \
                TYPE x = in ? in[strided_i] : ((TYPE *)out)[i];                                    \
                out[i] = (FUNC) ? 1 : 0;                                                           \
            }                                                                                      \
        }                                                                                          \
    }

// Macros for FP8/FP16/BF16 with float conversion
#define IMPL_UNARY_OP_CONVERT(TYPE, TYPE_SUFFIX, OP_NAME, FUNC, TO_FLOAT, FROM_FLOAT)              \
    void OP_NAME##_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata) {                                         \
        const TYPE *in = (const TYPE *)input;                                                      \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        const size_t *dims = metadata;                                                             \
        const size_t *strides = metadata ? metadata + num_dims : NULL;                             \
        const size_t offset = (metadata && num_dims > 0) ? metadata[2 * num_dims] : 0;             \
                                                                                                   \
        bool contiguous = (metadata == NULL) || is_contiguous(num_dims, dims, strides);            \
                                                                                                   \
        if (contiguous) {                                                                          \
            if (in) {                                                                              \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    float x = TO_FLOAT(in[offset + i]);                                            \
                    out[i] = FROM_FLOAT(FUNC);                                                     \
                }                                                                                  \
            } else {                                                                               \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    float x = TO_FLOAT(out[i]);                                                    \
                    out[i] = FROM_FLOAT(FUNC);                                                     \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t strided_i = offset + get_strided_index(i, num_dims, dims, strides);         \
                float x = in ? TO_FLOAT(in[strided_i]) : TO_FLOAT(out[i]);                         \
                out[i] = FROM_FLOAT(FUNC);                                                         \
            }                                                                                      \
        }                                                                                          \
    }

#define IMPL_UNARY_WITH_SCALAR_CONVERT(TYPE, TYPE_SUFFIX, OP_NAME, FUNC, TO_FLOAT, FROM_FLOAT)     \
    void OP_NAME##_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata, const void *scalar) {                     \
        const TYPE *in = (const TYPE *)input;                                                      \
        TYPE *out = (TYPE *)output;                                                                \
        float const_val = TO_FLOAT(*(const TYPE *)scalar);                                         \
                                                                                                   \
        const size_t *dims = metadata;                                                             \
        const size_t *strides = metadata ? metadata + num_dims : NULL;                             \
        const size_t offset = (metadata && num_dims > 0) ? metadata[2 * num_dims] : 0;             \
                                                                                                   \
        bool contiguous = (metadata == NULL) || is_contiguous(num_dims, dims, strides);            \
                                                                                                   \
        if (contiguous) {                                                                          \
            if (in) {                                                                              \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    float x = TO_FLOAT(in[offset + i]);                                            \
                    out[i] = FROM_FLOAT(FUNC);                                                     \
                }                                                                                  \
            } else {                                                                               \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    float x = TO_FLOAT(out[i]);                                                    \
                    out[i] = FROM_FLOAT(FUNC);                                                     \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t strided_i = offset + get_strided_index(i, num_dims, dims, strides);         \
                float x = in ? TO_FLOAT(in[strided_i]) : TO_FLOAT(out[i]);                         \
                out[i] = FROM_FLOAT(FUNC);                                                         \
            }                                                                                      \
        }                                                                                          \
    }

#define IMPL_UNARY_TO_BOOL_CONVERT(TYPE, TYPE_SUFFIX, OP_NAME, FUNC, TO_FLOAT)                     \
    void OP_NAME##_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata) {                                         \
        const TYPE *in = (const TYPE *)input;                                                      \
        uint8_t *out = (uint8_t *)output;                                                          \
                                                                                                   \
        const size_t *dims = metadata;                                                             \
        const size_t *strides = metadata ? metadata + num_dims : NULL;                             \
        const size_t offset = (metadata && num_dims > 0) ? metadata[2 * num_dims] : 0;             \
                                                                                                   \
        bool contiguous = (metadata == NULL) || is_contiguous(num_dims, dims, strides);            \
                                                                                                   \
        if (contiguous) {                                                                          \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                float x = TO_FLOAT(in[offset + i]);                                                \
                out[i] = (FUNC) ? 1 : 0;                                                           \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t strided_i = offset + get_strided_index(i, num_dims, dims, strides);         \
                float x = TO_FLOAT(in[strided_i]);                                                 \
                out[i] = (FUNC) ? 1 : 0;                                                           \
            }                                                                                      \
        }                                                                                          \
    }

#define IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(TYPE, TYPE_SUFFIX, OP_NAME, FUNC, TO_FLOAT)          \
    void OP_NAME##_##TYPE_SUFFIX(const void *input, void *output, size_t num_els, size_t num_dims, \
                                 const size_t *metadata, const void *scalar) {                     \
        const TYPE *in = (const TYPE *)input;                                                      \
        uint8_t *out = (uint8_t *)output;                                                          \
        float const_val = TO_FLOAT(*(const TYPE *)scalar);                                         \
                                                                                                   \
        const size_t *dims = metadata;                                                             \
        const size_t *strides = metadata ? metadata + num_dims : NULL;                             \
        const size_t offset = (metadata && num_dims > 0) ? metadata[2 * num_dims] : 0;             \
                                                                                                   \
        bool contiguous = (metadata == NULL) || is_contiguous(num_dims, dims, strides);            \
                                                                                                   \
        if (contiguous) {                                                                          \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                float x = in ? TO_FLOAT(in[offset + i]) : TO_FLOAT(((TYPE *)out)[i]);              \
                out[i] = (FUNC) ? 1 : 0;                                                           \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t strided_i = offset + get_strided_index(i, num_dims, dims, strides);         \
                float x = in ? TO_FLOAT(in[strided_i]) : TO_FLOAT(((TYPE *)out)[i]);               \
                out[i] = (FUNC) ? 1 : 0;                                                           \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// F32 OPERATIONS
// ============================================================================

IMPL_UNARY_OP(f32_t, f32, neg, -x)
IMPL_UNARY_OP(f32_t, f32, abs, fabsf(x))
IMPL_UNARY_OP(f32_t, f32, sign, (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))
IMPL_UNARY_OP(f32_t, f32, square, x *x)
IMPL_UNARY_OP(f32_t, f32, sqrt, sqrtf(x))
IMPL_UNARY_OP(f32_t, f32, recip, 1.0f / x)

IMPL_UNARY_OP(f32_t, f32, relu, (x > 0.0f) ? x : 0.0f)
IMPL_UNARY_OP(f32_t, f32, sigmoid, 1.0f / (1.0f + expf(-x)))
IMPL_UNARY_OP(f32_t, f32, tanh, tanhf(x))
IMPL_UNARY_OP(f32_t, f32, gelu, gelu_helper_f32(x))
IMPL_UNARY_OP(f32_t, f32, softplus, softplus_helper_f32(x))
IMPL_UNARY_OP(f32_t, f32, silu, silu_helper_f32(x))
IMPL_UNARY_OP(f32_t, f32, mish, mish_helper_f32(x))

IMPL_UNARY_OP(f32_t, f32, sin, sinf(x))
IMPL_UNARY_OP(f32_t, f32, cos, cosf(x))
IMPL_UNARY_OP(f32_t, f32, tan, tanf_opt(x))

IMPL_UNARY_OP(f32_t, f32, exp, expf(x))
IMPL_UNARY_OP(f32_t, f32, exp2, exp2f(x))
IMPL_UNARY_OP(f32_t, f32, exp10, exp10f_opt(x))
IMPL_UNARY_OP(f32_t, f32, ln, logf(x))
IMPL_UNARY_OP(f32_t, f32, log2, log2f(x))
IMPL_UNARY_OP(f32_t, f32, log10, log10f(x))

IMPL_UNARY_TO_BOOL(f32_t, f32, logical_not, x == 0.0f)

IMPL_UNARY_WITH_SCALAR(f32_t, f32, add_scalar, x + const_val)
IMPL_UNARY_WITH_SCALAR(f32_t, f32, sub_scalar, x - const_val)
IMPL_UNARY_WITH_SCALAR(f32_t, f32, mul_scalar, x *const_val)
IMPL_UNARY_WITH_SCALAR(f32_t, f32, div_scalar, x / const_val)
IMPL_UNARY_WITH_SCALAR(f32_t, f32, pow_scalar, powf_opt(x, const_val))
IMPL_UNARY_WITH_SCALAR(f32_t, f32, maximum_scalar, MAXIMUM(x, const_val))
IMPL_UNARY_WITH_SCALAR(f32_t, f32, minimum_scalar, MINIMUM(x, const_val))

IMPL_UNARY_CMP_SCALAR_TO_BOOL(f32_t, f32, eq_scalar, x == const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(f32_t, f32, ne_scalar, x != const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(f32_t, f32, lt_scalar, x < const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(f32_t, f32, le_scalar, x <= const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(f32_t, f32, gt_scalar, x > const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(f32_t, f32, ge_scalar, x >= const_val)

// ============================================================================
// F64 OPERATIONS
// ============================================================================

IMPL_UNARY_OP(f64_t, f64, neg, -x)
IMPL_UNARY_OP(f64_t, f64, abs, fabs(x))
IMPL_UNARY_OP(f64_t, f64, sign, (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0))
IMPL_UNARY_OP(f64_t, f64, square, x *x)
IMPL_UNARY_OP(f64_t, f64, sqrt, sqrt(x))
IMPL_UNARY_OP(f64_t, f64, recip, 1.0 / x)

IMPL_UNARY_OP(f64_t, f64, relu, (x > 0.0) ? x : 0.0)
IMPL_UNARY_OP(f64_t, f64, sigmoid, 1.0 / (1.0 + exp(-x)))
IMPL_UNARY_OP(f64_t, f64, tanh, tanh(x))
IMPL_UNARY_OP(f64_t, f64, gelu, gelu_helper_f64(x))
IMPL_UNARY_OP(f64_t, f64, softplus, softplus_helper_f64(x))
IMPL_UNARY_OP(f64_t, f64, silu, silu_helper_f64(x))
IMPL_UNARY_OP(f64_t, f64, mish, mish_helper_f64(x))

IMPL_UNARY_OP(f64_t, f64, sin, sin(x))
IMPL_UNARY_OP(f64_t, f64, cos, cos(x))
IMPL_UNARY_OP(f64_t, f64, tan, tan_opt(x))

IMPL_UNARY_OP(f64_t, f64, exp, exp(x))
IMPL_UNARY_OP(f64_t, f64, exp2, exp2(x))
IMPL_UNARY_OP(f64_t, f64, exp10, exp10_opt(x))
IMPL_UNARY_OP(f64_t, f64, ln, log(x))
IMPL_UNARY_OP(f64_t, f64, log2, log2(x))
IMPL_UNARY_OP(f64_t, f64, log10, log10(x))

IMPL_UNARY_TO_BOOL(f64_t, f64, logical_not, x == 0.0)

IMPL_UNARY_WITH_SCALAR(f64_t, f64, add_scalar, x + const_val)
IMPL_UNARY_WITH_SCALAR(f64_t, f64, sub_scalar, x - const_val)
IMPL_UNARY_WITH_SCALAR(f64_t, f64, mul_scalar, x *const_val)
IMPL_UNARY_WITH_SCALAR(f64_t, f64, div_scalar, x / const_val)
IMPL_UNARY_WITH_SCALAR(f64_t, f64, pow_scalar, pow_opt(x, const_val))
IMPL_UNARY_WITH_SCALAR(f64_t, f64, maximum_scalar, MAXIMUM(x, const_val))
IMPL_UNARY_WITH_SCALAR(f64_t, f64, minimum_scalar, MINIMUM(x, const_val))

IMPL_UNARY_CMP_SCALAR_TO_BOOL(f64_t, f64, eq_scalar, x == const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(f64_t, f64, ne_scalar, x != const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(f64_t, f64, lt_scalar, x < const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(f64_t, f64, le_scalar, x <= const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(f64_t, f64, gt_scalar, x > const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(f64_t, f64, ge_scalar, x >= const_val)

// ============================================================================
// BOOL OPERATIONS
// ============================================================================

IMPL_UNARY_OP(uint8_t, bool, neg, !x)
IMPL_UNARY_OP(uint8_t, bool, abs, x)
IMPL_UNARY_OP(uint8_t, bool, sign, x ? 1 : 0)
IMPL_UNARY_OP(uint8_t, bool, square, x)
IMPL_UNARY_OP(uint8_t, bool, sqrt, x)
IMPL_UNARY_OP(uint8_t, bool, recip, x)

IMPL_UNARY_OP(uint8_t, bool, relu, x)
IMPL_UNARY_OP(uint8_t, bool, sigmoid, x)
IMPL_UNARY_OP(uint8_t, bool, tanh, x)
IMPL_UNARY_OP(uint8_t, bool, gelu, x)
IMPL_UNARY_OP(uint8_t, bool, softplus, x)
IMPL_UNARY_OP(uint8_t, bool, silu, x)
IMPL_UNARY_OP(uint8_t, bool, mish, x)

IMPL_UNARY_OP(uint8_t, bool, sin, x ? 1 : 0)
IMPL_UNARY_OP(uint8_t, bool, cos, x ? 0 : 1)
IMPL_UNARY_OP(uint8_t, bool, tan, x ? 1 : 0)

IMPL_UNARY_OP(uint8_t, bool, exp, x ? 1 : 0)
IMPL_UNARY_OP(uint8_t, bool, exp2, x ? 1 : 0)
IMPL_UNARY_OP(uint8_t, bool, exp10, x ? 1 : 0)
IMPL_UNARY_OP(uint8_t, bool, ln, x ? 0 : 0)
IMPL_UNARY_OP(uint8_t, bool, log2, x ? 0 : 0)
IMPL_UNARY_OP(uint8_t, bool, log10, x ? 0 : 0)

IMPL_UNARY_TO_BOOL(uint8_t, bool, logical_not, !x)

IMPL_UNARY_WITH_SCALAR(uint8_t, bool, add_scalar, x || const_val)
IMPL_UNARY_WITH_SCALAR(uint8_t, bool, sub_scalar, x ^ const_val)
IMPL_UNARY_WITH_SCALAR(uint8_t, bool, mul_scalar, x &&const_val)
IMPL_UNARY_WITH_SCALAR(uint8_t, bool, div_scalar, x &&const_val)
IMPL_UNARY_WITH_SCALAR(uint8_t, bool, pow_scalar, x && (const_val != 0))
IMPL_UNARY_WITH_SCALAR(uint8_t, bool, maximum_scalar, x || const_val)
IMPL_UNARY_WITH_SCALAR(uint8_t, bool, minimum_scalar, x &&const_val)

IMPL_UNARY_CMP_SCALAR_TO_BOOL(uint8_t, bool, eq_scalar, x == const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(uint8_t, bool, ne_scalar, x != const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(uint8_t, bool, lt_scalar, !x && const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(uint8_t, bool, le_scalar, !x || const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(uint8_t, bool, gt_scalar, x && !const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(uint8_t, bool, ge_scalar, x || !const_val)

// ============================================================================
// FP8 E4M3 OPERATIONS
// ============================================================================

IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, neg, -x, fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, abs, fabsf(x), fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, sign, (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f),
                      fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, square, x *x, fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, sqrt, sqrtf(x), fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, recip, 1.0f / x, fp8_e4m3_to_float, float_to_fp8_e4m3)

IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, relu, (x > 0.0f) ? x : 0.0f, fp8_e4m3_to_float,
                      float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, sigmoid, 1.0f / (1.0f + expf(-x)), fp8_e4m3_to_float,
                      float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, tanh, tanhf(x), fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, gelu, gelu_helper_f32(x), fp8_e4m3_to_float,
                      float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, softplus, softplus_helper_f32(x), fp8_e4m3_to_float,
                      float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, silu, silu_helper_f32(x), fp8_e4m3_to_float,
                      float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, mish, mish_helper_f32(x), fp8_e4m3_to_float,
                      float_to_fp8_e4m3)

IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, sin, sinf(x), fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, cos, cosf(x), fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, tan, tanf(x), fp8_e4m3_to_float, float_to_fp8_e4m3)

IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, exp, expf(x), fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, exp2, exp2f(x), fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, exp10, exp10f(x), fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, ln, logf(x), fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, log2, log2f(x), fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e4m3, log10, log10f(x), fp8_e4m3_to_float, float_to_fp8_e4m3)

IMPL_UNARY_TO_BOOL_CONVERT(uint8_t, f8e4m3, logical_not, x == 0.0f, fp8_e4m3_to_float)

IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e4m3, add_scalar, x + const_val, fp8_e4m3_to_float,
                               float_to_fp8_e4m3)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e4m3, sub_scalar, x - const_val, fp8_e4m3_to_float,
                               float_to_fp8_e4m3)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e4m3, mul_scalar, x *const_val, fp8_e4m3_to_float,
                               float_to_fp8_e4m3)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e4m3, div_scalar, x / const_val, fp8_e4m3_to_float,
                               float_to_fp8_e4m3)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e4m3, pow_scalar, powf(x, const_val), fp8_e4m3_to_float,
                               float_to_fp8_e4m3)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e4m3, maximum_scalar, MAXIMUM(x, const_val),
                               fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e4m3, minimum_scalar, MINIMUM(x, const_val),
                               fp8_e4m3_to_float, float_to_fp8_e4m3)

IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint8_t, f8e4m3, eq_scalar, x == const_val, fp8_e4m3_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint8_t, f8e4m3, ne_scalar, x != const_val, fp8_e4m3_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint8_t, f8e4m3, lt_scalar, x < const_val, fp8_e4m3_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint8_t, f8e4m3, le_scalar, x <= const_val, fp8_e4m3_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint8_t, f8e4m3, gt_scalar, x > const_val, fp8_e4m3_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint8_t, f8e4m3, ge_scalar, x >= const_val, fp8_e4m3_to_float)

// ============================================================================
// FP8 E5M2 OPERATIONS
// ============================================================================

IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, neg, -x, fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, abs, fabsf(x), fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, sign, (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f),
                      fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, square, x *x, fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, sqrt, sqrtf(x), fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, recip, 1.0f / x, fp8_e5m2_to_float, float_to_fp8_e5m2)

IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, relu, (x > 0.0f) ? x : 0.0f, fp8_e5m2_to_float,
                      float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, sigmoid, 1.0f / (1.0f + expf(-x)), fp8_e5m2_to_float,
                      float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, tanh, tanhf(x), fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, gelu, gelu_helper_f32(x), fp8_e5m2_to_float,
                      float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, softplus, softplus_helper_f32(x), fp8_e5m2_to_float,
                      float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, silu, silu_helper_f32(x), fp8_e5m2_to_float,
                      float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, mish, mish_helper_f32(x), fp8_e5m2_to_float,
                      float_to_fp8_e5m2)

IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, sin, sinf(x), fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, cos, cosf(x), fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, tan, tanf(x), fp8_e5m2_to_float, float_to_fp8_e5m2)

IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, exp, expf(x), fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, exp2, exp2f(x), fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, exp10, exp10f(x), fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, ln, logf(x), fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, log2, log2f(x), fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_OP_CONVERT(uint8_t, f8e5m2, log10, log10f(x), fp8_e5m2_to_float, float_to_fp8_e5m2)

IMPL_UNARY_TO_BOOL_CONVERT(uint8_t, f8e5m2, logical_not, x == 0.0f, fp8_e5m2_to_float)

IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e5m2, add_scalar, x + const_val, fp8_e5m2_to_float,
                               float_to_fp8_e5m2)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e5m2, sub_scalar, x - const_val, fp8_e5m2_to_float,
                               float_to_fp8_e5m2)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e5m2, mul_scalar, x *const_val, fp8_e5m2_to_float,
                               float_to_fp8_e5m2)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e5m2, div_scalar, x / const_val, fp8_e5m2_to_float,
                               float_to_fp8_e5m2)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e5m2, pow_scalar, powf(x, const_val), fp8_e5m2_to_float,
                               float_to_fp8_e5m2)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e5m2, maximum_scalar, MAXIMUM(x, const_val),
                               fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint8_t, f8e5m2, minimum_scalar, MINIMUM(x, const_val),
                               fp8_e5m2_to_float, float_to_fp8_e5m2)

IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint8_t, f8e5m2, eq_scalar, x == const_val, fp8_e5m2_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint8_t, f8e5m2, ne_scalar, x != const_val, fp8_e5m2_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint8_t, f8e5m2, lt_scalar, x < const_val, fp8_e5m2_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint8_t, f8e5m2, le_scalar, x <= const_val, fp8_e5m2_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint8_t, f8e5m2, gt_scalar, x > const_val, fp8_e5m2_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint8_t, f8e5m2, ge_scalar, x >= const_val, fp8_e5m2_to_float)

// ============================================================================
// BF16 OPERATIONS
// ============================================================================

IMPL_UNARY_OP_CONVERT(uint16_t, bf16, neg, -x, bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, abs, fabsf(x), bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, sign, (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f),
                      bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, square, x *x, bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, sqrt, sqrtf(x), bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, recip, 1.0f / x, bf16_to_float, float_to_bf16)

IMPL_UNARY_OP_CONVERT(uint16_t, bf16, relu, (x > 0.0f) ? x : 0.0f, bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, sigmoid, 1.0f / (1.0f + expf(-x)), bf16_to_float,
                      float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, tanh, tanhf(x), bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, gelu, gelu_helper_f32(x), bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, softplus, softplus_helper_f32(x), bf16_to_float,
                      float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, silu, silu_helper_f32(x), bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, mish, mish_helper_f32(x), bf16_to_float, float_to_bf16)

IMPL_UNARY_OP_CONVERT(uint16_t, bf16, sin, sinf(x), bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, cos, cosf(x), bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, tan, tanf(x), bf16_to_float, float_to_bf16)

IMPL_UNARY_OP_CONVERT(uint16_t, bf16, exp, expf(x), bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, exp2, exp2f(x), bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, exp10, exp10f(x), bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, ln, logf(x), bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, log2, log2f(x), bf16_to_float, float_to_bf16)
IMPL_UNARY_OP_CONVERT(uint16_t, bf16, log10, log10f(x), bf16_to_float, float_to_bf16)

IMPL_UNARY_TO_BOOL_CONVERT(uint16_t, bf16, logical_not, x == 0.0f, bf16_to_float)

IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, bf16, add_scalar, x + const_val, bf16_to_float,
                               float_to_bf16)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, bf16, sub_scalar, x - const_val, bf16_to_float,
                               float_to_bf16)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, bf16, mul_scalar, x *const_val, bf16_to_float,
                               float_to_bf16)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, bf16, div_scalar, x / const_val, bf16_to_float,
                               float_to_bf16)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, bf16, pow_scalar, powf(x, const_val), bf16_to_float,
                               float_to_bf16)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, bf16, maximum_scalar, MAXIMUM(x, const_val), bf16_to_float,
                               float_to_bf16)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, bf16, minimum_scalar, MINIMUM(x, const_val), bf16_to_float,
                               float_to_bf16)

IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint16_t, bf16, eq_scalar, x == const_val, bf16_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint16_t, bf16, ne_scalar, x != const_val, bf16_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint16_t, bf16, lt_scalar, x < const_val, bf16_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint16_t, bf16, le_scalar, x <= const_val, bf16_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint16_t, bf16, gt_scalar, x > const_val, bf16_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint16_t, bf16, ge_scalar, x >= const_val, bf16_to_float)

// ============================================================================
// FP16 OPERATIONS
// ============================================================================

IMPL_UNARY_OP_CONVERT(uint16_t, f16, neg, -x, fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, abs, fabsf(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, sign, (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f),
                      fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, square, x *x, fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, sqrt, sqrtf(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, recip, 1.0f / x, fp16_to_float, float_to_fp16)

IMPL_UNARY_OP_CONVERT(uint16_t, f16, relu, (x > 0.0f) ? x : 0.0f, fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, sigmoid, 1.0f / (1.0f + expf(-x)), fp16_to_float,
                      float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, tanh, tanhf(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, gelu, gelu_helper_f32(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, softplus, softplus_helper_f32(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, silu, silu_helper_f32(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, mish, mish_helper_f32(x), fp16_to_float, float_to_fp16)

IMPL_UNARY_OP_CONVERT(uint16_t, f16, sin, sinf(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, cos, cosf(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, tan, tanf(x), fp16_to_float, float_to_fp16)

IMPL_UNARY_OP_CONVERT(uint16_t, f16, exp, expf(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, exp2, exp2f(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, exp10, exp10f(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, ln, logf(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, log2, log2f(x), fp16_to_float, float_to_fp16)
IMPL_UNARY_OP_CONVERT(uint16_t, f16, log10, log10f(x), fp16_to_float, float_to_fp16)

IMPL_UNARY_TO_BOOL_CONVERT(uint16_t, f16, logical_not, x == 0.0f, fp16_to_float)

IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, f16, add_scalar, x + const_val, fp16_to_float,
                               float_to_fp16)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, f16, sub_scalar, x - const_val, fp16_to_float,
                               float_to_fp16)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, f16, mul_scalar, x *const_val, fp16_to_float,
                               float_to_fp16)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, f16, div_scalar, x / const_val, fp16_to_float,
                               float_to_fp16)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, f16, pow_scalar, powf(x, const_val), fp16_to_float,
                               float_to_fp16)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, f16, maximum_scalar, MAXIMUM(x, const_val), fp16_to_float,
                               float_to_fp16)
IMPL_UNARY_WITH_SCALAR_CONVERT(uint16_t, f16, minimum_scalar, MINIMUM(x, const_val), fp16_to_float,
                               float_to_fp16)

IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint16_t, f16, eq_scalar, x == const_val, fp16_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint16_t, f16, ne_scalar, x != const_val, fp16_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint16_t, f16, lt_scalar, x < const_val, fp16_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint16_t, f16, le_scalar, x <= const_val, fp16_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint16_t, f16, gt_scalar, x > const_val, fp16_to_float)
IMPL_UNARY_CMP_SCALAR_TO_BOOL_CONVERT(uint16_t, f16, ge_scalar, x >= const_val, fp16_to_float)

// ============================================================================
// UNSIGNED INTEGER OPERATIONS (U8, U16, U32, U64)
// ============================================================================

// U8 (already defined in original code)
IMPL_UNARY_OP(u8_t, u8, sign, (x > 0) ? 1 : 0)
IMPL_UNARY_OP(u8_t, u8, square, x *x)
IMPL_UNARY_OP(u8_t, u8, sqrt, (u8_t)sqrtf((float)x))
IMPL_UNARY_OP(u8_t, u8, abs, x)
IMPL_UNARY_OP(u8_t, u8, recip, (x != 0) ? (u8_t)(1.0f / (float)x) : 0)
IMPL_UNARY_TO_BOOL(u8_t, u8, logical_not, x == 0)
IMPL_UNARY_WITH_SCALAR(u8_t, u8, add_scalar, x + const_val)
IMPL_UNARY_WITH_SCALAR(u8_t, u8, sub_scalar, (x > const_val) ? (x - const_val) : 0)
IMPL_UNARY_WITH_SCALAR(u8_t, u8, mul_scalar, x *const_val)
IMPL_UNARY_WITH_SCALAR(u8_t, u8, div_scalar, (const_val != 0) ? (x / const_val) : 0)
IMPL_UNARY_WITH_SCALAR(u8_t, u8, pow_scalar, (u8_t)powf_opt((float)x, (float)const_val))
IMPL_UNARY_WITH_SCALAR(u8_t, u8, maximum_scalar, MAXIMUM(x, const_val))
IMPL_UNARY_WITH_SCALAR(u8_t, u8, minimum_scalar, MINIMUM(x, const_val))
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u8_t, u8, eq_scalar, x == const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u8_t, u8, ne_scalar, x != const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u8_t, u8, lt_scalar, x < const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u8_t, u8, le_scalar, x <= const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u8_t, u8, gt_scalar, x > const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u8_t, u8, ge_scalar, x >= const_val)

// U16
IMPL_UNARY_OP(u16_t, u16, sign, (x > 0) ? 1 : 0)
IMPL_UNARY_OP(u16_t, u16, square, x *x)
IMPL_UNARY_OP(u16_t, u16, sqrt, (u16_t)sqrtf((float)x))
IMPL_UNARY_OP(u16_t, u16, abs, x)
IMPL_UNARY_OP(u16_t, u16, recip, (x != 0) ? (u16_t)(1.0f / (float)x) : 0)
IMPL_UNARY_TO_BOOL(u16_t, u16, logical_not, x == 0)
IMPL_UNARY_WITH_SCALAR(u16_t, u16, add_scalar, x + const_val)
IMPL_UNARY_WITH_SCALAR(u16_t, u16, sub_scalar, (x > const_val) ? (x - const_val) : 0)
IMPL_UNARY_WITH_SCALAR(u16_t, u16, mul_scalar, x *const_val)
IMPL_UNARY_WITH_SCALAR(u16_t, u16, div_scalar, (const_val != 0) ? (x / const_val) : 0)
IMPL_UNARY_WITH_SCALAR(u16_t, u16, pow_scalar, (u16_t)powf_opt((float)x, (float)const_val))
IMPL_UNARY_WITH_SCALAR(u16_t, u16, maximum_scalar, MAXIMUM(x, const_val))
IMPL_UNARY_WITH_SCALAR(u16_t, u16, minimum_scalar, MINIMUM(x, const_val))
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u16_t, u16, eq_scalar, x == const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u16_t, u16, ne_scalar, x != const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u16_t, u16, lt_scalar, x < const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u16_t, u16, le_scalar, x <= const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u16_t, u16, gt_scalar, x > const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u16_t, u16, ge_scalar, x >= const_val)

// U32
IMPL_UNARY_OP(u32_t, u32, sign, (x > 0) ? 1 : 0)
IMPL_UNARY_OP(u32_t, u32, square, x *x)
IMPL_UNARY_OP(u32_t, u32, sqrt, (u32_t)sqrtf((float)x))
IMPL_UNARY_OP(u32_t, u32, abs, x)
IMPL_UNARY_OP(u32_t, u32, recip, (x != 0) ? (u32_t)(1.0f / (float)x) : 0)
IMPL_UNARY_TO_BOOL(u32_t, u32, logical_not, x == 0)
IMPL_UNARY_WITH_SCALAR(u32_t, u32, add_scalar, x + const_val)
IMPL_UNARY_WITH_SCALAR(u32_t, u32, sub_scalar, (x > const_val) ? (x - const_val) : 0)
IMPL_UNARY_WITH_SCALAR(u32_t, u32, mul_scalar, x *const_val)
IMPL_UNARY_WITH_SCALAR(u32_t, u32, div_scalar, (const_val != 0) ? (x / const_val) : 0)
IMPL_UNARY_WITH_SCALAR(u32_t, u32, pow_scalar, (u32_t)powf_opt((float)x, (float)const_val))
IMPL_UNARY_WITH_SCALAR(u32_t, u32, maximum_scalar, MAXIMUM(x, const_val))
IMPL_UNARY_WITH_SCALAR(u32_t, u32, minimum_scalar, MINIMUM(x, const_val))
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u32_t, u32, eq_scalar, x == const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u32_t, u32, ne_scalar, x != const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u32_t, u32, lt_scalar, x < const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u32_t, u32, le_scalar, x <= const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u32_t, u32, gt_scalar, x > const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u32_t, u32, ge_scalar, x >= const_val)

// U64
IMPL_UNARY_OP(u64_t, u64, sign, (x > 0) ? 1 : 0)
IMPL_UNARY_OP(u64_t, u64, square, x *x)
IMPL_UNARY_OP(u64_t, u64, sqrt, (u64_t)sqrt((double)x))
IMPL_UNARY_OP(u64_t, u64, abs, x)
IMPL_UNARY_OP(u64_t, u64, recip, (x != 0) ? (u64_t)(1.0 / (double)x) : 0)
IMPL_UNARY_TO_BOOL(u64_t, u64, logical_not, x == 0)
IMPL_UNARY_WITH_SCALAR(u64_t, u64, add_scalar, x + const_val)
IMPL_UNARY_WITH_SCALAR(u64_t, u64, sub_scalar, (x > const_val) ? (x - const_val) : 0)
IMPL_UNARY_WITH_SCALAR(u64_t, u64, mul_scalar, x *const_val)
IMPL_UNARY_WITH_SCALAR(u64_t, u64, div_scalar, (const_val != 0) ? (x / const_val) : 0)
IMPL_UNARY_WITH_SCALAR(u64_t, u64, pow_scalar, (u64_t)pow_opt((double)x, (double)const_val))
IMPL_UNARY_WITH_SCALAR(u64_t, u64, maximum_scalar, MAXIMUM(x, const_val))
IMPL_UNARY_WITH_SCALAR(u64_t, u64, minimum_scalar, MINIMUM(x, const_val))
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u64_t, u64, eq_scalar, x == const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u64_t, u64, ne_scalar, x != const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u64_t, u64, lt_scalar, x < const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u64_t, u64, le_scalar, x <= const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u64_t, u64, gt_scalar, x > const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(u64_t, u64, ge_scalar, x >= const_val)

// ============================================================================
// SIGNED INTEGER OPERATIONS (I8, I16, I32, I64)
// ============================================================================

// I8
IMPL_UNARY_OP(i8_t, i8, neg, -x)
IMPL_UNARY_OP(i8_t, i8, abs, (x < 0) ? -x : x)
IMPL_UNARY_OP(i8_t, i8, sign, (x > 0) ? 1 : ((x < 0) ? -1 : 0))
IMPL_UNARY_OP(i8_t, i8, square, x *x)
IMPL_UNARY_OP(i8_t, i8, sqrt, (i8_t)sqrtf((float)((x < 0) ? -x : x)))
IMPL_UNARY_OP(i8_t, i8, recip, (x != 0) ? (i8_t)(1.0f / (float)x) : 0)
IMPL_UNARY_TO_BOOL(i8_t, i8, logical_not, x == 0)
IMPL_UNARY_WITH_SCALAR(i8_t, i8, add_scalar, x + const_val)
IMPL_UNARY_WITH_SCALAR(i8_t, i8, sub_scalar, x - const_val)
IMPL_UNARY_WITH_SCALAR(i8_t, i8, mul_scalar, x *const_val)
IMPL_UNARY_WITH_SCALAR(i8_t, i8, div_scalar, (const_val != 0) ? (x / const_val) : 0)
IMPL_UNARY_WITH_SCALAR(i8_t, i8, pow_scalar, (i8_t)powf_opt((float)x, (float)const_val))
IMPL_UNARY_WITH_SCALAR(i8_t, i8, maximum_scalar, MAXIMUM(x, const_val))
IMPL_UNARY_WITH_SCALAR(i8_t, i8, minimum_scalar, MINIMUM(x, const_val))
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i8_t, i8, eq_scalar, x == const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i8_t, i8, ne_scalar, x != const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i8_t, i8, lt_scalar, x < const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i8_t, i8, le_scalar, x <= const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i8_t, i8, gt_scalar, x > const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i8_t, i8, ge_scalar, x >= const_val)

// I16
IMPL_UNARY_OP(i16_t, i16, neg, -x)
IMPL_UNARY_OP(i16_t, i16, abs, (x < 0) ? -x : x)
IMPL_UNARY_OP(i16_t, i16, sign, (x > 0) ? 1 : ((x < 0) ? -1 : 0))
IMPL_UNARY_OP(i16_t, i16, square, x *x)
IMPL_UNARY_OP(i16_t, i16, sqrt, (i16_t)sqrtf((float)((x < 0) ? -x : x)))
IMPL_UNARY_OP(i16_t, i16, recip, (x != 0) ? (i16_t)(1.0f / (float)x) : 0)
IMPL_UNARY_TO_BOOL(i16_t, i16, logical_not, x == 0)
IMPL_UNARY_WITH_SCALAR(i16_t, i16, add_scalar, x + const_val)
IMPL_UNARY_WITH_SCALAR(i16_t, i16, sub_scalar, x - const_val)
IMPL_UNARY_WITH_SCALAR(i16_t, i16, mul_scalar, x *const_val)
IMPL_UNARY_WITH_SCALAR(i16_t, i16, div_scalar, (const_val != 0) ? (x / const_val) : 0)
IMPL_UNARY_WITH_SCALAR(i16_t, i16, pow_scalar, (i16_t)powf_opt((float)x, (float)const_val))
IMPL_UNARY_WITH_SCALAR(i16_t, i16, maximum_scalar, MAXIMUM(x, const_val))
IMPL_UNARY_WITH_SCALAR(i16_t, i16, minimum_scalar, MINIMUM(x, const_val))
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i16_t, i16, eq_scalar, x == const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i16_t, i16, ne_scalar, x != const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i16_t, i16, lt_scalar, x < const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i16_t, i16, le_scalar, x <= const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i16_t, i16, gt_scalar, x > const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i16_t, i16, ge_scalar, x >= const_val)

// I32
IMPL_UNARY_OP(i32_t, i32, neg, -x)
IMPL_UNARY_OP(i32_t, i32, abs, (x < 0) ? -x : x)
IMPL_UNARY_OP(i32_t, i32, sign, (x > 0) ? 1 : ((x < 0) ? -1 : 0))
IMPL_UNARY_OP(i32_t, i32, square, x *x)
IMPL_UNARY_OP(i32_t, i32, sqrt, (i32_t)sqrtf((float)((x < 0) ? -x : x)))
IMPL_UNARY_OP(i32_t, i32, recip, (x != 0) ? (i32_t)(1.0f / (float)x) : 0)
IMPL_UNARY_TO_BOOL(i32_t, i32, logical_not, x == 0)
IMPL_UNARY_WITH_SCALAR(i32_t, i32, add_scalar, x + const_val)
IMPL_UNARY_WITH_SCALAR(i32_t, i32, sub_scalar, x - const_val)
IMPL_UNARY_WITH_SCALAR(i32_t, i32, mul_scalar, x *const_val)
IMPL_UNARY_WITH_SCALAR(i32_t, i32, div_scalar, (const_val != 0) ? (x / const_val) : 0)
IMPL_UNARY_WITH_SCALAR(i32_t, i32, pow_scalar, (i32_t)powf_opt((float)x, (float)const_val))
IMPL_UNARY_WITH_SCALAR(i32_t, i32, maximum_scalar, MAXIMUM(x, const_val))
IMPL_UNARY_WITH_SCALAR(i32_t, i32, minimum_scalar, MINIMUM(x, const_val))
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i32_t, i32, eq_scalar, x == const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i32_t, i32, ne_scalar, x != const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i32_t, i32, lt_scalar, x < const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i32_t, i32, le_scalar, x <= const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i32_t, i32, gt_scalar, x > const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i32_t, i32, ge_scalar, x >= const_val)

// I64
IMPL_UNARY_OP(i64_t, i64, neg, -x)
IMPL_UNARY_OP(i64_t, i64, abs, (x < 0) ? -x : x)
IMPL_UNARY_OP(i64_t, i64, sign, (x > 0) ? 1 : ((x < 0) ? -1 : 0))
IMPL_UNARY_OP(i64_t, i64, square, x *x)
IMPL_UNARY_OP(i64_t, i64, sqrt, (i64_t)sqrt((double)((x < 0) ? -x : x)))
IMPL_UNARY_OP(i64_t, i64, recip, (x != 0) ? (i64_t)(1.0 / (double)x) : 0)
IMPL_UNARY_TO_BOOL(i64_t, i64, logical_not, x == 0)
IMPL_UNARY_WITH_SCALAR(i64_t, i64, add_scalar, x + const_val)
IMPL_UNARY_WITH_SCALAR(i64_t, i64, sub_scalar, x - const_val)
IMPL_UNARY_WITH_SCALAR(i64_t, i64, mul_scalar, x *const_val)
IMPL_UNARY_WITH_SCALAR(i64_t, i64, div_scalar, (const_val != 0) ? (x / const_val) : 0)
IMPL_UNARY_WITH_SCALAR(i64_t, i64, pow_scalar, (i64_t)pow_opt((double)x, (double)const_val))
IMPL_UNARY_WITH_SCALAR(i64_t, i64, maximum_scalar, MAXIMUM(x, const_val))
IMPL_UNARY_WITH_SCALAR(i64_t, i64, minimum_scalar, MINIMUM(x, const_val))
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i64_t, i64, eq_scalar, x == const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i64_t, i64, ne_scalar, x != const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i64_t, i64, lt_scalar, x < const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i64_t, i64, le_scalar, x <= const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i64_t, i64, gt_scalar, x > const_val)
IMPL_UNARY_CMP_SCALAR_TO_BOOL(i64_t, i64, ge_scalar, x >= const_val)
