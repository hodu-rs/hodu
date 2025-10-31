/**
 * @file ops_unary.h
 * @brief Unary tensor operations header
 *
 * Provides element-wise unary operations for tensors:
 * - Basic arithmetic: neg, abs, sign, square, sqrt, recip
 * - Activation functions: relu, sigmoid, tanh, gelu, softplus, silu, mish
 * - Trigonometric: sin, cos, tan
 * - Exponential/logarithmic: exp, exp2, exp10, ln, log2, log10
 * - Logical: logical_not
 * - Scalar operations: arithmetic (add, sub, mul, div, pow, max, min) and comparison (eq, ne, lt,
 * le, gt, ge)
 *
 * All operations support strided tensor access and multiple data types.
 */

#ifndef HODU_CPU_KERNELS_OPS_UNARY_H
#define HODU_CPU_KERNELS_OPS_UNARY_H

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// UNARY OPERATION FUNCTION SIGNATURES
// ============================================================================
//
// All unary operations follow consistent signatures:
//   void op_type(const void *input, void *output, const size_t *metadata)
//   void op_scalar_type(const void *input, void *output, const size_t *metadata, const void
//   *scalar)
//
// Parameters:
//   input    - Pointer to input tensor data (may be strided)
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing tensor layout (see below)
//   scalar   - Scalar value for scalar operations (same type as tensor elements)
//
// Metadata layout (same for all operations):
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: strides
// - metadata[2+2*num_dims]: offset
//
// Type support:
// - Basic operations (neg, abs, sign, square, sqrt, recip): all numeric types
// - Activation functions: float types only (f8e4m3, f8e5m2, bf16, f16, f32, f64)
// - Trigonometric: float types only
// - Exponential/logarithmic: float types only
// - Logical: all types (including bool)
// - Scalar operations: all types

/**
 * @brief Macro to declare basic unary operations
 *
 * Declares functions for element-wise basic arithmetic operations:
 * - neg: Negate elements (-x)
 * - abs: Absolute value (|x|)
 * - sign: Sign function (-1, 0, or 1)
 * - square: Square elements (x²)
 * - sqrt: Square root (√x)
 * - recip: Reciprocal (1/x)
 */
#define DECLARE_UNARY_OP(TYPE_SUFFIX)                                                              \
    void neg_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);               \
    void abs_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);               \
    void sign_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);              \
    void square_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);            \
    void sqrt_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);              \
    void recip_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);

/**
 * @brief Macro to declare activation function operations
 *
 * Declares functions for neural network activation functions:
 * - relu: Rectified Linear Unit (max(0, x))
 * - sigmoid: Sigmoid function (1 / (1 + e^(-x)))
 * - tanh: Hyperbolic tangent
 * - gelu: Gaussian Error Linear Unit
 * - softplus: Softplus function (ln(1 + e^x))
 * - silu: Sigmoid Linear Unit (x * sigmoid(x))
 * - mish: Mish activation (x * tanh(softplus(x)))
 *
 * Note: Only available for float types (f8e4m3, f8e5m2, bf16, f16, f32, f64)
 */
#define DECLARE_UNARY_ACTIVATION(TYPE_SUFFIX)                                                      \
    void relu_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);              \
    void sigmoid_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);           \
    void tanh_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);              \
    void gelu_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);              \
    void softplus_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);          \
    void silu_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);              \
    void mish_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);

/**
 * @brief Macro to declare trigonometric operations
 *
 * Declares functions for trigonometric operations:
 * - sin: Sine function
 * - cos: Cosine function
 * - tan: Tangent function
 *
 * Note: Only available for float types (f8e4m3, f8e5m2, bf16, f16, f32, f64)
 */
#define DECLARE_UNARY_TRIG(TYPE_SUFFIX)                                                            \
    void sin_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);               \
    void cos_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);               \
    void tan_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);

/**
 * @brief Macro to declare exponential and logarithmic operations
 *
 * Declares functions for exponential and logarithmic operations:
 * - exp: Natural exponential (e^x)
 * - exp2: Base-2 exponential (2^x)
 * - exp10: Base-10 exponential (10^x)
 * - ln: Natural logarithm (log_e(x))
 * - log2: Base-2 logarithm (log_2(x))
 * - log10: Base-10 logarithm (log_10(x))
 *
 * Note: Only available for float types (f8e4m3, f8e5m2, bf16, f16, f32, f64)
 */
#define DECLARE_UNARY_EXP(TYPE_SUFFIX)                                                             \
    void exp_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);               \
    void exp2_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);              \
    void exp10_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);             \
    void ln_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);                \
    void log2_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);              \
    void log10_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);

/**
 * @brief Macro to declare logical operations
 *
 * Declares functions for logical operations:
 * - logical_not: Logical NOT (returns 1 if x==0, else 0)
 *
 * Output type is always bool (uint8_t), regardless of input type.
 * Available for all types including bool.
 */
#define DECLARE_UNARY_LOGICAL(TYPE_SUFFIX)                                                         \
    void logical_not_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);

/**
 * @brief Macro to declare scalar arithmetic operations
 *
 * Declares functions that combine each tensor element with a scalar value:
 * - add_scalar: Add scalar to each element (x + s)
 * - sub_scalar: Subtract scalar from each element (x - s)
 * - mul_scalar: Multiply each element by scalar (x * s)
 * - div_scalar: Divide each element by scalar (x / s)
 * - pow_scalar: Raise each element to scalar power (x^s)
 * - maximum_scalar: Element-wise maximum with scalar (max(x, s))
 * - minimum_scalar: Element-wise minimum with scalar (min(x, s))
 *
 * The scalar parameter must be the same type as the tensor elements.
 * Available for all types.
 */
#define DECLARE_UNARY_WITH_SCALAR(TYPE_SUFFIX)                                                     \
    void add_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,         \
                                  const void *scalar);                                             \
    void sub_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,         \
                                  const void *scalar);                                             \
    void mul_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,         \
                                  const void *scalar);                                             \
    void div_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,         \
                                  const void *scalar);                                             \
    void pow_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,         \
                                  const void *scalar);                                             \
    void maximum_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,     \
                                      const void *scalar);                                         \
    void minimum_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,     \
                                      const void *scalar);

/**
 * @brief Macro to declare scalar comparison operations
 *
 * Declares functions that compare each tensor element with a scalar value:
 * - eq_scalar: Equal to (x == s)
 * - ne_scalar: Not equal to (x != s)
 * - lt_scalar: Less than (x < s)
 * - le_scalar: Less than or equal to (x <= s)
 * - gt_scalar: Greater than (x > s)
 * - ge_scalar: Greater than or equal to (x >= s)
 *
 * Output type is always bool (uint8_t), regardless of input type.
 * The scalar parameter must be the same type as the tensor elements.
 * Available for all types.
 */
#define DECLARE_UNARY_CMP_SCALAR(TYPE_SUFFIX)                                                      \
    void eq_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,          \
                                 const void *scalar);                                              \
    void ne_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,          \
                                 const void *scalar);                                              \
    void lt_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,          \
                                 const void *scalar);                                              \
    void le_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,          \
                                 const void *scalar);                                              \
    void gt_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,          \
                                 const void *scalar);                                              \
    void ge_scalar_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata,          \
                                 const void *scalar);

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
