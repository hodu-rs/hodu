#include "ops_binary.h"
#include <math.h>

// ============================================================================
// BINARY OPERATION IMPLEMENTATION MACROS
// ============================================================================

// Metadata layout:
// - dims: num_dims size_t values
// - lhs_strides: num_dims size_t values
// - rhs_strides: num_dims size_t values
// - lhs_offset: 1 size_t value
// - rhs_offset: 1 size_t value

#define IMPL_BINARY_OP(TYPE, TYPE_SUFFIX, OP_NAME, FUNC)                                           \
    void OP_NAME##_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,   \
                                 size_t num_dims, const size_t *metadata) {                        \
        const TYPE *l = (const TYPE *)lhs;                                                         \
        const TYPE *r = (const TYPE *)rhs;                                                         \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        const size_t *dims = metadata;                                                             \
        const size_t *lhs_strides = metadata ? metadata + num_dims : NULL;                         \
        const size_t *rhs_strides = metadata ? metadata + 2 * num_dims : NULL;                     \
        const size_t lhs_offset = (metadata && num_dims > 0) ? metadata[3 * num_dims] : 0;         \
        const size_t rhs_offset = (metadata && num_dims > 0) ? metadata[3 * num_dims + 1] : 0;     \
                                                                                                   \
        bool lhs_cont = (metadata == NULL) || is_contiguous(num_dims, dims, lhs_strides);          \
        bool rhs_cont = (metadata == NULL) || is_contiguous(num_dims, dims, rhs_strides);          \
                                                                                                   \
        if (lhs_cont && rhs_cont) {                                                                \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                TYPE x = l[lhs_offset + i];                                                        \
                TYPE y = r[rhs_offset + i];                                                        \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        } else if (lhs_cont) {                                                                     \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t rhs_i = rhs_offset + get_strided_index(i, num_dims, dims, rhs_strides);     \
                TYPE x = l[lhs_offset + i];                                                        \
                TYPE y = r[rhs_i];                                                                 \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        } else if (rhs_cont) {                                                                     \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t lhs_i = lhs_offset + get_strided_index(i, num_dims, dims, lhs_strides);     \
                TYPE x = l[lhs_i];                                                                 \
                TYPE y = r[rhs_offset + i];                                                        \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t lhs_i = lhs_offset + get_strided_index(i, num_dims, dims, lhs_strides);     \
                size_t rhs_i = rhs_offset + get_strided_index(i, num_dims, dims, rhs_strides);     \
                TYPE x = l[lhs_i];                                                                 \
                TYPE y = r[rhs_i];                                                                 \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        }                                                                                          \
    }

#define IMPL_BINARY_TO_BOOL(TYPE, TYPE_SUFFIX, OP_NAME, FUNC)                                      \
    void OP_NAME##_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,   \
                                 size_t num_dims, const size_t *metadata) {                        \
        const TYPE *l = (const TYPE *)lhs;                                                         \
        const TYPE *r = (const TYPE *)rhs;                                                         \
        uint8_t *out = (uint8_t *)output;                                                          \
                                                                                                   \
        const size_t *dims = metadata;                                                             \
        const size_t *lhs_strides = metadata ? metadata + num_dims : NULL;                         \
        const size_t *rhs_strides = metadata ? metadata + 2 * num_dims : NULL;                     \
        const size_t lhs_offset = (metadata && num_dims > 0) ? metadata[3 * num_dims] : 0;         \
        const size_t rhs_offset = (metadata && num_dims > 0) ? metadata[3 * num_dims + 1] : 0;     \
                                                                                                   \
        bool lhs_cont = (metadata == NULL) || is_contiguous(num_dims, dims, lhs_strides);          \
        bool rhs_cont = (metadata == NULL) || is_contiguous(num_dims, dims, rhs_strides);          \
                                                                                                   \
        if (lhs_cont && rhs_cont) {                                                                \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                TYPE x = l[lhs_offset + i];                                                        \
                TYPE y = r[rhs_offset + i];                                                        \
                out[i] = (FUNC) ? 1 : 0;                                                           \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t lhs_i =                                                                     \
                    lhs_cont ? (lhs_offset + i)                                                    \
                             : (lhs_offset + get_strided_index(i, num_dims, dims, lhs_strides));   \
                size_t rhs_i =                                                                     \
                    rhs_cont ? (rhs_offset + i)                                                    \
                             : (rhs_offset + get_strided_index(i, num_dims, dims, rhs_strides));   \
                TYPE x = l[lhs_i];                                                                 \
                TYPE y = r[rhs_i];                                                                 \
                out[i] = (FUNC) ? 1 : 0;                                                           \
            }                                                                                      \
        }                                                                                          \
    }

// Macros for FP8/FP16/BF16 with float conversion
#define IMPL_BINARY_OP_CONVERT(TYPE, TYPE_SUFFIX, OP_NAME, FUNC, TO_FLOAT, FROM_FLOAT)             \
    void OP_NAME##_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,   \
                                 size_t num_dims, const size_t *metadata) {                        \
        const TYPE *l = (const TYPE *)lhs;                                                         \
        const TYPE *r = (const TYPE *)rhs;                                                         \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        const size_t *dims = metadata;                                                             \
        const size_t *lhs_strides = metadata ? metadata + num_dims : NULL;                         \
        const size_t *rhs_strides = metadata ? metadata + 2 * num_dims : NULL;                     \
        const size_t lhs_offset = (metadata && num_dims > 0) ? metadata[3 * num_dims] : 0;         \
        const size_t rhs_offset = (metadata && num_dims > 0) ? metadata[3 * num_dims + 1] : 0;     \
                                                                                                   \
        bool lhs_cont = (metadata == NULL) || is_contiguous(num_dims, dims, lhs_strides);          \
        bool rhs_cont = (metadata == NULL) || is_contiguous(num_dims, dims, rhs_strides);          \
                                                                                                   \
        if (lhs_cont && rhs_cont) {                                                                \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                float x = TO_FLOAT(l[lhs_offset + i]);                                             \
                float y = TO_FLOAT(r[rhs_offset + i]);                                             \
                out[i] = FROM_FLOAT(FUNC);                                                         \
            }                                                                                      \
        } else if (lhs_cont) {                                                                     \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t rhs_i = rhs_offset + get_strided_index(i, num_dims, dims, rhs_strides);     \
                float x = TO_FLOAT(l[lhs_offset + i]);                                             \
                float y = TO_FLOAT(r[rhs_i]);                                                      \
                out[i] = FROM_FLOAT(FUNC);                                                         \
            }                                                                                      \
        } else if (rhs_cont) {                                                                     \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t lhs_i = lhs_offset + get_strided_index(i, num_dims, dims, lhs_strides);     \
                float x = TO_FLOAT(l[lhs_i]);                                                      \
                float y = TO_FLOAT(r[rhs_offset + i]);                                             \
                out[i] = FROM_FLOAT(FUNC);                                                         \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t lhs_i = lhs_offset + get_strided_index(i, num_dims, dims, lhs_strides);     \
                size_t rhs_i = rhs_offset + get_strided_index(i, num_dims, dims, rhs_strides);     \
                float x = TO_FLOAT(l[lhs_i]);                                                      \
                float y = TO_FLOAT(r[rhs_i]);                                                      \
                out[i] = FROM_FLOAT(FUNC);                                                         \
            }                                                                                      \
        }                                                                                          \
    }

#define IMPL_BINARY_TO_BOOL_CONVERT(TYPE, TYPE_SUFFIX, OP_NAME, FUNC, TO_FLOAT)                    \
    void OP_NAME##_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output, size_t num_els,   \
                                 size_t num_dims, const size_t *metadata) {                        \
        const TYPE *l = (const TYPE *)lhs;                                                         \
        const TYPE *r = (const TYPE *)rhs;                                                         \
        uint8_t *out = (uint8_t *)output;                                                          \
                                                                                                   \
        const size_t *dims = metadata;                                                             \
        const size_t *lhs_strides = metadata ? metadata + num_dims : NULL;                         \
        const size_t *rhs_strides = metadata ? metadata + 2 * num_dims : NULL;                     \
        const size_t lhs_offset = (metadata && num_dims > 0) ? metadata[3 * num_dims] : 0;         \
        const size_t rhs_offset = (metadata && num_dims > 0) ? metadata[3 * num_dims + 1] : 0;     \
                                                                                                   \
        bool lhs_cont = (metadata == NULL) || is_contiguous(num_dims, dims, lhs_strides);          \
        bool rhs_cont = (metadata == NULL) || is_contiguous(num_dims, dims, rhs_strides);          \
                                                                                                   \
        if (lhs_cont && rhs_cont) {                                                                \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                float x = TO_FLOAT(l[lhs_offset + i]);                                             \
                float y = TO_FLOAT(r[rhs_offset + i]);                                             \
                out[i] = (FUNC) ? 1 : 0;                                                           \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t lhs_i =                                                                     \
                    lhs_cont ? (lhs_offset + i)                                                    \
                             : (lhs_offset + get_strided_index(i, num_dims, dims, lhs_strides));   \
                size_t rhs_i =                                                                     \
                    rhs_cont ? (rhs_offset + i)                                                    \
                             : (rhs_offset + get_strided_index(i, num_dims, dims, rhs_strides));   \
                float x = TO_FLOAT(l[lhs_i]);                                                      \
                float y = TO_FLOAT(r[rhs_i]);                                                      \
                out[i] = (FUNC) ? 1 : 0;                                                           \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// F32 OPERATIONS
// ============================================================================

IMPL_BINARY_OP(f32_t, f32, add, x + y)
IMPL_BINARY_OP(f32_t, f32, sub, x - y)
IMPL_BINARY_OP(f32_t, f32, mul, x *y)
IMPL_BINARY_OP(f32_t, f32, div, x / y)
IMPL_BINARY_OP(f32_t, f32, pow, powf_opt(x, y))
IMPL_BINARY_OP(f32_t, f32, maximum, MAXIMUM(x, y))
IMPL_BINARY_OP(f32_t, f32, minimum, MINIMUM(x, y))

IMPL_BINARY_TO_BOOL(f32_t, f32, logical_and, (x != 0.0f && y != 0.0f))
IMPL_BINARY_TO_BOOL(f32_t, f32, logical_or, (x != 0.0f || y != 0.0f))
IMPL_BINARY_TO_BOOL(f32_t, f32, logical_xor, (x != 0.0f) != (y != 0.0f))

IMPL_BINARY_TO_BOOL(f32_t, f32, eq, x == y)
IMPL_BINARY_TO_BOOL(f32_t, f32, ne, x != y)
IMPL_BINARY_TO_BOOL(f32_t, f32, lt, x < y)
IMPL_BINARY_TO_BOOL(f32_t, f32, le, x <= y)
IMPL_BINARY_TO_BOOL(f32_t, f32, gt, x > y)
IMPL_BINARY_TO_BOOL(f32_t, f32, ge, x >= y)

// ============================================================================
// F64 OPERATIONS
// ============================================================================

IMPL_BINARY_OP(f64_t, f64, add, x + y)
IMPL_BINARY_OP(f64_t, f64, sub, x - y)
IMPL_BINARY_OP(f64_t, f64, mul, x *y)
IMPL_BINARY_OP(f64_t, f64, div, x / y)
IMPL_BINARY_OP(f64_t, f64, pow, pow_opt(x, y))
IMPL_BINARY_OP(f64_t, f64, maximum, MAXIMUM(x, y))
IMPL_BINARY_OP(f64_t, f64, minimum, MINIMUM(x, y))

IMPL_BINARY_TO_BOOL(f64_t, f64, logical_and, (x != 0.0 && y != 0.0))
IMPL_BINARY_TO_BOOL(f64_t, f64, logical_or, (x != 0.0 || y != 0.0))
IMPL_BINARY_TO_BOOL(f64_t, f64, logical_xor, (x != 0.0) != (y != 0.0))

IMPL_BINARY_TO_BOOL(f64_t, f64, eq, x == y)
IMPL_BINARY_TO_BOOL(f64_t, f64, ne, x != y)
IMPL_BINARY_TO_BOOL(f64_t, f64, lt, x < y)
IMPL_BINARY_TO_BOOL(f64_t, f64, le, x <= y)
IMPL_BINARY_TO_BOOL(f64_t, f64, gt, x > y)
IMPL_BINARY_TO_BOOL(f64_t, f64, ge, x >= y)

// ============================================================================
// BOOL OPERATIONS
// ============================================================================

IMPL_BINARY_OP(uint8_t, bool, add, x || y)
IMPL_BINARY_OP(uint8_t, bool, sub, x && !y)
IMPL_BINARY_OP(uint8_t, bool, mul, x &&y)
IMPL_BINARY_OP(uint8_t, bool, div, x &&y)
IMPL_BINARY_OP(uint8_t, bool, pow, x || !y)
IMPL_BINARY_OP(uint8_t, bool, maximum, x || y)
IMPL_BINARY_OP(uint8_t, bool, minimum, x &&y)

IMPL_BINARY_TO_BOOL(uint8_t, bool, logical_and, x &&y)
IMPL_BINARY_TO_BOOL(uint8_t, bool, logical_or, x || y)
IMPL_BINARY_TO_BOOL(uint8_t, bool, logical_xor, x != y)

IMPL_BINARY_TO_BOOL(uint8_t, bool, eq, x == y)
IMPL_BINARY_TO_BOOL(uint8_t, bool, ne, x != y)
IMPL_BINARY_TO_BOOL(uint8_t, bool, lt, !x && y)
IMPL_BINARY_TO_BOOL(uint8_t, bool, le, !x || y)
IMPL_BINARY_TO_BOOL(uint8_t, bool, gt, x && !y)
IMPL_BINARY_TO_BOOL(uint8_t, bool, ge, x || !y)

// ============================================================================
// FP8 E4M3 OPERATIONS
// ============================================================================

IMPL_BINARY_OP_CONVERT(uint8_t, f8e4m3, add, x + y, fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_BINARY_OP_CONVERT(uint8_t, f8e4m3, sub, x - y, fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_BINARY_OP_CONVERT(uint8_t, f8e4m3, mul, x *y, fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_BINARY_OP_CONVERT(uint8_t, f8e4m3, div, x / y, fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_BINARY_OP_CONVERT(uint8_t, f8e4m3, pow, powf(x, y), fp8_e4m3_to_float, float_to_fp8_e4m3)
IMPL_BINARY_OP_CONVERT(uint8_t, f8e4m3, maximum, MAXIMUM(x, y), fp8_e4m3_to_float,
                       float_to_fp8_e4m3)
IMPL_BINARY_OP_CONVERT(uint8_t, f8e4m3, minimum, MINIMUM(x, y), fp8_e4m3_to_float,
                       float_to_fp8_e4m3)

IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e4m3, logical_and, (x != 0.0f && y != 0.0f),
                            fp8_e4m3_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e4m3, logical_or, (x != 0.0f || y != 0.0f),
                            fp8_e4m3_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e4m3, logical_xor, (x != 0.0f) != (y != 0.0f),
                            fp8_e4m3_to_float)

IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e4m3, eq, x == y, fp8_e4m3_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e4m3, ne, x != y, fp8_e4m3_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e4m3, lt, x < y, fp8_e4m3_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e4m3, le, x <= y, fp8_e4m3_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e4m3, gt, x > y, fp8_e4m3_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e4m3, ge, x >= y, fp8_e4m3_to_float)

// ============================================================================
// FP8 E5M2 OPERATIONS
// ============================================================================

IMPL_BINARY_OP_CONVERT(uint8_t, f8e5m2, add, x + y, fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_BINARY_OP_CONVERT(uint8_t, f8e5m2, sub, x - y, fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_BINARY_OP_CONVERT(uint8_t, f8e5m2, mul, x *y, fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_BINARY_OP_CONVERT(uint8_t, f8e5m2, div, x / y, fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_BINARY_OP_CONVERT(uint8_t, f8e5m2, pow, powf(x, y), fp8_e5m2_to_float, float_to_fp8_e5m2)
IMPL_BINARY_OP_CONVERT(uint8_t, f8e5m2, maximum, MAXIMUM(x, y), fp8_e5m2_to_float,
                       float_to_fp8_e5m2)
IMPL_BINARY_OP_CONVERT(uint8_t, f8e5m2, minimum, MINIMUM(x, y), fp8_e5m2_to_float,
                       float_to_fp8_e5m2)

IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e5m2, logical_and, (x != 0.0f && y != 0.0f),
                            fp8_e5m2_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e5m2, logical_or, (x != 0.0f || y != 0.0f),
                            fp8_e5m2_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e5m2, logical_xor, (x != 0.0f) != (y != 0.0f),
                            fp8_e5m2_to_float)

IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e5m2, eq, x == y, fp8_e5m2_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e5m2, ne, x != y, fp8_e5m2_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e5m2, lt, x < y, fp8_e5m2_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e5m2, le, x <= y, fp8_e5m2_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e5m2, gt, x > y, fp8_e5m2_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint8_t, f8e5m2, ge, x >= y, fp8_e5m2_to_float)

// ============================================================================
// BF16 OPERATIONS
// ============================================================================

IMPL_BINARY_OP_CONVERT(uint16_t, bf16, add, x + y, bf16_to_float, float_to_bf16)
IMPL_BINARY_OP_CONVERT(uint16_t, bf16, sub, x - y, bf16_to_float, float_to_bf16)
IMPL_BINARY_OP_CONVERT(uint16_t, bf16, mul, x *y, bf16_to_float, float_to_bf16)
IMPL_BINARY_OP_CONVERT(uint16_t, bf16, div, x / y, bf16_to_float, float_to_bf16)
IMPL_BINARY_OP_CONVERT(uint16_t, bf16, pow, powf(x, y), bf16_to_float, float_to_bf16)
IMPL_BINARY_OP_CONVERT(uint16_t, bf16, maximum, MAXIMUM(x, y), bf16_to_float, float_to_bf16)
IMPL_BINARY_OP_CONVERT(uint16_t, bf16, minimum, MINIMUM(x, y), bf16_to_float, float_to_bf16)

IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, bf16, logical_and, (x != 0.0f && y != 0.0f), bf16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, bf16, logical_or, (x != 0.0f || y != 0.0f), bf16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, bf16, logical_xor, (x != 0.0f) != (y != 0.0f), bf16_to_float)

IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, bf16, eq, x == y, bf16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, bf16, ne, x != y, bf16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, bf16, lt, x < y, bf16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, bf16, le, x <= y, bf16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, bf16, gt, x > y, bf16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, bf16, ge, x >= y, bf16_to_float)

// ============================================================================
// FP16 OPERATIONS
// ============================================================================

IMPL_BINARY_OP_CONVERT(uint16_t, f16, add, x + y, fp16_to_float, float_to_fp16)
IMPL_BINARY_OP_CONVERT(uint16_t, f16, sub, x - y, fp16_to_float, float_to_fp16)
IMPL_BINARY_OP_CONVERT(uint16_t, f16, mul, x *y, fp16_to_float, float_to_fp16)
IMPL_BINARY_OP_CONVERT(uint16_t, f16, div, x / y, fp16_to_float, float_to_fp16)
IMPL_BINARY_OP_CONVERT(uint16_t, f16, pow, powf(x, y), fp16_to_float, float_to_fp16)
IMPL_BINARY_OP_CONVERT(uint16_t, f16, maximum, MAXIMUM(x, y), fp16_to_float, float_to_fp16)
IMPL_BINARY_OP_CONVERT(uint16_t, f16, minimum, MINIMUM(x, y), fp16_to_float, float_to_fp16)

IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, f16, logical_and, (x != 0.0f && y != 0.0f), fp16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, f16, logical_or, (x != 0.0f || y != 0.0f), fp16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, f16, logical_xor, (x != 0.0f) != (y != 0.0f), fp16_to_float)

IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, f16, eq, x == y, fp16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, f16, ne, x != y, fp16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, f16, lt, x < y, fp16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, f16, le, x <= y, fp16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, f16, gt, x > y, fp16_to_float)
IMPL_BINARY_TO_BOOL_CONVERT(uint16_t, f16, ge, x >= y, fp16_to_float)

// ============================================================================
// UNSIGNED INTEGER OPERATIONS (U8, U16, U32, U64)
// ============================================================================

// U8
IMPL_BINARY_OP(u8_t, u8, add, x + y)
IMPL_BINARY_OP(u8_t, u8, sub, (x > y) ? (x - y) : 0)
IMPL_BINARY_OP(u8_t, u8, mul, x *y)
IMPL_BINARY_OP(u8_t, u8, div, (y != 0) ? (x / y) : 0)
IMPL_BINARY_OP(u8_t, u8, pow, (u8_t)ipow_u64(x, y))
IMPL_BINARY_OP(u8_t, u8, maximum, MAXIMUM(x, y))
IMPL_BINARY_OP(u8_t, u8, minimum, MINIMUM(x, y))

IMPL_BINARY_TO_BOOL(u8_t, u8, logical_and, (x != 0 && y != 0))
IMPL_BINARY_TO_BOOL(u8_t, u8, logical_or, (x != 0 || y != 0))
IMPL_BINARY_TO_BOOL(u8_t, u8, logical_xor, (x != 0) != (y != 0))

IMPL_BINARY_TO_BOOL(u8_t, u8, eq, x == y)
IMPL_BINARY_TO_BOOL(u8_t, u8, ne, x != y)
IMPL_BINARY_TO_BOOL(u8_t, u8, lt, x < y)
IMPL_BINARY_TO_BOOL(u8_t, u8, le, x <= y)
IMPL_BINARY_TO_BOOL(u8_t, u8, gt, x > y)
IMPL_BINARY_TO_BOOL(u8_t, u8, ge, x >= y)

// U16
IMPL_BINARY_OP(u16_t, u16, add, x + y)
IMPL_BINARY_OP(u16_t, u16, sub, (x > y) ? (x - y) : 0)
IMPL_BINARY_OP(u16_t, u16, mul, x *y)
IMPL_BINARY_OP(u16_t, u16, div, (y != 0) ? (x / y) : 0)
IMPL_BINARY_OP(u16_t, u16, pow, (u16_t)ipow_u64(x, y))
IMPL_BINARY_OP(u16_t, u16, maximum, MAXIMUM(x, y))
IMPL_BINARY_OP(u16_t, u16, minimum, MINIMUM(x, y))

IMPL_BINARY_TO_BOOL(u16_t, u16, logical_and, (x != 0 && y != 0))
IMPL_BINARY_TO_BOOL(u16_t, u16, logical_or, (x != 0 || y != 0))
IMPL_BINARY_TO_BOOL(u16_t, u16, logical_xor, (x != 0) != (y != 0))

IMPL_BINARY_TO_BOOL(u16_t, u16, eq, x == y)
IMPL_BINARY_TO_BOOL(u16_t, u16, ne, x != y)
IMPL_BINARY_TO_BOOL(u16_t, u16, lt, x < y)
IMPL_BINARY_TO_BOOL(u16_t, u16, le, x <= y)
IMPL_BINARY_TO_BOOL(u16_t, u16, gt, x > y)
IMPL_BINARY_TO_BOOL(u16_t, u16, ge, x >= y)

// U32
IMPL_BINARY_OP(u32_t, u32, add, x + y)
IMPL_BINARY_OP(u32_t, u32, sub, (x > y) ? (x - y) : 0)
IMPL_BINARY_OP(u32_t, u32, mul, x *y)
IMPL_BINARY_OP(u32_t, u32, div, (y != 0) ? (x / y) : 0)
IMPL_BINARY_OP(u32_t, u32, pow, (u32_t)ipow_u64(x, y))
IMPL_BINARY_OP(u32_t, u32, maximum, MAXIMUM(x, y))
IMPL_BINARY_OP(u32_t, u32, minimum, MINIMUM(x, y))

IMPL_BINARY_TO_BOOL(u32_t, u32, logical_and, (x != 0 && y != 0))
IMPL_BINARY_TO_BOOL(u32_t, u32, logical_or, (x != 0 || y != 0))
IMPL_BINARY_TO_BOOL(u32_t, u32, logical_xor, (x != 0) != (y != 0))

IMPL_BINARY_TO_BOOL(u32_t, u32, eq, x == y)
IMPL_BINARY_TO_BOOL(u32_t, u32, ne, x != y)
IMPL_BINARY_TO_BOOL(u32_t, u32, lt, x < y)
IMPL_BINARY_TO_BOOL(u32_t, u32, le, x <= y)
IMPL_BINARY_TO_BOOL(u32_t, u32, gt, x > y)
IMPL_BINARY_TO_BOOL(u32_t, u32, ge, x >= y)

// U64
IMPL_BINARY_OP(u64_t, u64, add, x + y)
IMPL_BINARY_OP(u64_t, u64, sub, (x > y) ? (x - y) : 0)
IMPL_BINARY_OP(u64_t, u64, mul, x *y)
IMPL_BINARY_OP(u64_t, u64, div, (y != 0) ? (x / y) : 0)
IMPL_BINARY_OP(u64_t, u64, pow, ipow_u64(x, y))
IMPL_BINARY_OP(u64_t, u64, maximum, MAXIMUM(x, y))
IMPL_BINARY_OP(u64_t, u64, minimum, MINIMUM(x, y))

IMPL_BINARY_TO_BOOL(u64_t, u64, logical_and, (x != 0 && y != 0))
IMPL_BINARY_TO_BOOL(u64_t, u64, logical_or, (x != 0 || y != 0))
IMPL_BINARY_TO_BOOL(u64_t, u64, logical_xor, (x != 0) != (y != 0))

IMPL_BINARY_TO_BOOL(u64_t, u64, eq, x == y)
IMPL_BINARY_TO_BOOL(u64_t, u64, ne, x != y)
IMPL_BINARY_TO_BOOL(u64_t, u64, lt, x < y)
IMPL_BINARY_TO_BOOL(u64_t, u64, le, x <= y)
IMPL_BINARY_TO_BOOL(u64_t, u64, gt, x > y)
IMPL_BINARY_TO_BOOL(u64_t, u64, ge, x >= y)

// ============================================================================
// SIGNED INTEGER OPERATIONS (I8, I16, I32, I64)
// ============================================================================

// I8
IMPL_BINARY_OP(i8_t, i8, add, x + y)
IMPL_BINARY_OP(i8_t, i8, sub, x - y)
IMPL_BINARY_OP(i8_t, i8, mul, x *y)
IMPL_BINARY_OP(i8_t, i8, div, (y != 0) ? (x / y) : 0)
IMPL_BINARY_OP(i8_t, i8, pow, (i8_t)ipow_i64(x, y))
IMPL_BINARY_OP(i8_t, i8, maximum, MAXIMUM(x, y))
IMPL_BINARY_OP(i8_t, i8, minimum, MINIMUM(x, y))

IMPL_BINARY_TO_BOOL(i8_t, i8, logical_and, (x != 0 && y != 0))
IMPL_BINARY_TO_BOOL(i8_t, i8, logical_or, (x != 0 || y != 0))
IMPL_BINARY_TO_BOOL(i8_t, i8, logical_xor, (x != 0) != (y != 0))

IMPL_BINARY_TO_BOOL(i8_t, i8, eq, x == y)
IMPL_BINARY_TO_BOOL(i8_t, i8, ne, x != y)
IMPL_BINARY_TO_BOOL(i8_t, i8, lt, x < y)
IMPL_BINARY_TO_BOOL(i8_t, i8, le, x <= y)
IMPL_BINARY_TO_BOOL(i8_t, i8, gt, x > y)
IMPL_BINARY_TO_BOOL(i8_t, i8, ge, x >= y)

// I16
IMPL_BINARY_OP(i16_t, i16, add, x + y)
IMPL_BINARY_OP(i16_t, i16, sub, x - y)
IMPL_BINARY_OP(i16_t, i16, mul, x *y)
IMPL_BINARY_OP(i16_t, i16, div, (y != 0) ? (x / y) : 0)
IMPL_BINARY_OP(i16_t, i16, pow, (i16_t)ipow_i64(x, y))
IMPL_BINARY_OP(i16_t, i16, maximum, MAXIMUM(x, y))
IMPL_BINARY_OP(i16_t, i16, minimum, MINIMUM(x, y))

IMPL_BINARY_TO_BOOL(i16_t, i16, logical_and, (x != 0 && y != 0))
IMPL_BINARY_TO_BOOL(i16_t, i16, logical_or, (x != 0 || y != 0))
IMPL_BINARY_TO_BOOL(i16_t, i16, logical_xor, (x != 0) != (y != 0))

IMPL_BINARY_TO_BOOL(i16_t, i16, eq, x == y)
IMPL_BINARY_TO_BOOL(i16_t, i16, ne, x != y)
IMPL_BINARY_TO_BOOL(i16_t, i16, lt, x < y)
IMPL_BINARY_TO_BOOL(i16_t, i16, le, x <= y)
IMPL_BINARY_TO_BOOL(i16_t, i16, gt, x > y)
IMPL_BINARY_TO_BOOL(i16_t, i16, ge, x >= y)

// I32
IMPL_BINARY_OP(i32_t, i32, add, x + y)
IMPL_BINARY_OP(i32_t, i32, sub, x - y)
IMPL_BINARY_OP(i32_t, i32, mul, x *y)
IMPL_BINARY_OP(i32_t, i32, div, (y != 0) ? (x / y) : 0)
IMPL_BINARY_OP(i32_t, i32, pow, (i32_t)ipow_i64(x, y))
IMPL_BINARY_OP(i32_t, i32, maximum, MAXIMUM(x, y))
IMPL_BINARY_OP(i32_t, i32, minimum, MINIMUM(x, y))

IMPL_BINARY_TO_BOOL(i32_t, i32, logical_and, (x != 0 && y != 0))
IMPL_BINARY_TO_BOOL(i32_t, i32, logical_or, (x != 0 || y != 0))
IMPL_BINARY_TO_BOOL(i32_t, i32, logical_xor, (x != 0) != (y != 0))

IMPL_BINARY_TO_BOOL(i32_t, i32, eq, x == y)
IMPL_BINARY_TO_BOOL(i32_t, i32, ne, x != y)
IMPL_BINARY_TO_BOOL(i32_t, i32, lt, x < y)
IMPL_BINARY_TO_BOOL(i32_t, i32, le, x <= y)
IMPL_BINARY_TO_BOOL(i32_t, i32, gt, x > y)
IMPL_BINARY_TO_BOOL(i32_t, i32, ge, x >= y)

// I64
IMPL_BINARY_OP(i64_t, i64, add, x + y)
IMPL_BINARY_OP(i64_t, i64, sub, x - y)
IMPL_BINARY_OP(i64_t, i64, mul, x *y)
IMPL_BINARY_OP(i64_t, i64, div, (y != 0) ? (x / y) : 0)
IMPL_BINARY_OP(i64_t, i64, pow, ipow_i64(x, y))
IMPL_BINARY_OP(i64_t, i64, maximum, MAXIMUM(x, y))
IMPL_BINARY_OP(i64_t, i64, minimum, MINIMUM(x, y))

IMPL_BINARY_TO_BOOL(i64_t, i64, logical_and, (x != 0 && y != 0))
IMPL_BINARY_TO_BOOL(i64_t, i64, logical_or, (x != 0 || y != 0))
IMPL_BINARY_TO_BOOL(i64_t, i64, logical_xor, (x != 0) != (y != 0))

IMPL_BINARY_TO_BOOL(i64_t, i64, eq, x == y)
IMPL_BINARY_TO_BOOL(i64_t, i64, ne, x != y)
IMPL_BINARY_TO_BOOL(i64_t, i64, lt, x < y)
IMPL_BINARY_TO_BOOL(i64_t, i64, le, x <= y)
IMPL_BINARY_TO_BOOL(i64_t, i64, gt, x > y)
IMPL_BINARY_TO_BOOL(i64_t, i64, ge, x >= y)
