#include "./headers/utils.metal"
#include <metal_math>
#include <metal_stdlib>

using namespace metal;

template <typename T> T m_pow_int(T base, unsigned int exponent) {
    T result = 1;
    while (exponent > 0) {
        if (exponent & 1) {
            result *= base;
        }
        exponent >>= 1;
        base *= base;
    }
    return result;
}

float m_pow_float(float base, float exponent) {
    if (exponent == 0.0f) {
        return 1.0f;
    }

    if (base == 0.0f) {
        return (exponent > 0.0f) ? 0.0f : INFINITY;
    }

    if (base == 1.0f) {
        return 1.0f;
    }

    if (exponent == 1.0f) {
        return base;
    }

    if (floor(exponent) == exponent) {
        if (exponent >= 0.0f) {
            return m_pow_int(base, (unsigned int)exponent);
        } else {
            return 1.0f / m_pow_int(base, (unsigned int)(-exponent));
        }
    }

    if (base < 0.0f) {
        return NAN;
    }

    return pow(base, exponent);
}

float m_tan(float x) {
    x = fmod(x, 2 * M_PI_F);
    if (x > M_PI_F)
        x -= 2 * M_PI_F;
    else if (x < -M_PI_F)
        x += 2 * M_PI_F;

    float halfPi = M_PI_F / 2;
    float eps = 1e-6f;

    if (fabs(fabs(x) - halfPi) < eps) {
        return x > 0 ? 1e6f : -1e6f;
    }

    return sin(x) / cos(x);
}

float m_exp10(float x) {
    return exp(x * 2.3025850929940456f); // ln(10) ≈ 2.302585...
}

// Error function (erf) approximation using Abramowitz and Stegun formula 7.1.26
// Maximum error: 1.5×10^−7
float m_erf(float x) {
    // Constants
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    const float p = 0.3275911f;

    // Save the sign of x
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    x = abs(x);

    // A&S formula 7.1.26
    float t = 1.0f / (1.0f + p * x);
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t3 * t;
    float t5 = t4 * t;
    float y = 1.0f - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp(-x * x);

    return sign * y;
}

float m_hardsigmoid(float x) { return fmax(0.0f, fmin(1.0f, (x + 3.0f) / 6.0f)); }

float m_hardsilu(float x) { return x * m_hardsigmoid(x); }

// Metadata layout:
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: dims (shape)
// - metadata[2+num_dims..2+2*num_dims]: strides
// - metadata[2+2*num_dims]: offset (starting offset in input)

#define UNARY_OP_OUTPUT(IN_TYPENAME, OUT_TYPENAME, FN_NAME, FUNC)                                  \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
            const constant size_t *dims = metadata + 2;                                            \
            const constant size_t *strides = metadata + 2 + num_dims;                              \
            const size_t offset = metadata[2 + 2 * num_dims];                                      \
                                                                                                   \
            if (is_contiguous(num_dims, dims, strides)) {                                          \
                IN_TYPENAME x = input ? input[offset + id] : (IN_TYPENAME)output[id];              \
                output[id] = FUNC;                                                                 \
            } else {                                                                               \
                unsigned strided_i = offset + get_strided_index(id, num_dims, dims, strides);      \
                IN_TYPENAME x = input ? input[strided_i] : (IN_TYPENAME)output[id];                \
                output[id] = FUNC;                                                                 \
            }                                                                                      \
        }                                                                                          \
    }

#define UNARY_OP(TYPENAME, FN_NAME, FUNC) UNARY_OP_OUTPUT(TYPENAME, TYPENAME, FN_NAME, FUNC)

#define UNARY_OP_WITH_CONSTANT(IN_TYPENAME, OUT_TYPENAME, FN_NAME, FUNC)                           \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t *metadata [[buffer(2)]], constant IN_TYPENAME &const_val [[buffer(3)]],    \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
            const constant size_t *dims = metadata + 2;                                            \
            const constant size_t *strides = metadata + 2 + num_dims;                              \
            const size_t offset = metadata[2 + 2 * num_dims];                                      \
                                                                                                   \
            if (is_contiguous(num_dims, dims, strides)) {                                          \
                IN_TYPENAME x;                                                                     \
                if (input) {                                                                       \
                    x = input[offset + id];                                                        \
                } else {                                                                           \
                    x = output[id];                                                                \
                }                                                                                  \
                output[id] = FUNC;                                                                 \
            } else {                                                                               \
                unsigned strided_i = offset + get_strided_index(id, num_dims, dims, strides);      \
                IN_TYPENAME x;                                                                     \
                if (input) {                                                                       \
                    x = input[strided_i];                                                          \
                } else {                                                                           \
                    x = output[id];                                                                \
                }                                                                                  \
                output[id] = FUNC;                                                                 \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// BOOL OPERATIONS
// ============================================================================

// cmp with scalar
UNARY_OP_WITH_CONSTANT(bool, bool, eq_scalar_bool, x == const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, ne_scalar_bool, x != const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, lt_scalar_bool, !x && const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, le_scalar_bool, !x || const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, gt_scalar_bool, x && !const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, ge_scalar_bool, x || !const_val);

// unary - basic
UNARY_OP(bool, neg_bool, !x);
UNARY_OP(bool, abs_bool, x);
UNARY_OP(bool, sign_bool, x ? 1 : 0);
UNARY_OP(bool, square_bool, x);
UNARY_OP(bool, sqrt_bool, x);

// unary - activation
UNARY_OP(bool, relu_bool, x);
UNARY_OP(bool, sigmoid_bool, x);

// unary logical
UNARY_OP_OUTPUT(bool, bool, logical_not_bool, !x);

UNARY_OP(bool, tanh_bool, x);

// unary with scalar - arithmetic
UNARY_OP_WITH_CONSTANT(bool, bool, add_scalar_bool, x || const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, sub_scalar_bool, x ^ const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, mul_scalar_bool, x &&const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, div_scalar_bool, x &&const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, rem_scalar_bool, x && !const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, pow_scalar_bool, x && (const_val != 0));
UNARY_OP_WITH_CONSTANT(bool, bool, maximum_scalar_bool, x || const_val);
UNARY_OP_WITH_CONSTANT(bool, bool, minimum_scalar_bool, x &&const_val);

// ============================================================================
// BFLOAT16 OPERATIONS
// ============================================================================

// cmp with scalar
UNARY_OP_WITH_CONSTANT(bfloat, bool, eq_scalar_bf16, float(x) == float(const_val));
UNARY_OP_WITH_CONSTANT(bfloat, bool, ne_scalar_bf16, float(x) != float(const_val));
UNARY_OP_WITH_CONSTANT(bfloat, bool, lt_scalar_bf16, float(x) < float(const_val));
UNARY_OP_WITH_CONSTANT(bfloat, bool, le_scalar_bf16, float(x) <= float(const_val));
UNARY_OP_WITH_CONSTANT(bfloat, bool, gt_scalar_bf16, float(x) > float(const_val));
UNARY_OP_WITH_CONSTANT(bfloat, bool, ge_scalar_bf16, float(x) >= float(const_val));

// unary - basic
UNARY_OP(bfloat, neg_bf16, -x);
UNARY_OP(bfloat, abs_bf16, bfloat(abs(float(x))));
UNARY_OP(bfloat, sign_bf16, (x > 0.0bf) ? 1.0bf : ((x < 0.0bf) ? -1.0bf : 0.0bf));
UNARY_OP(bfloat, square_bf16, x *x);
UNARY_OP(bfloat, sqrt_bf16, bfloat(sqrt(float(x))));
UNARY_OP(bfloat, recip_bf16, bfloat(1.0f / float(x)));

// unary - activation
UNARY_OP(bfloat, relu_bf16, x > 0.0bf ? x : 0.0bf);
UNARY_OP(bfloat, sigmoid_bf16, bfloat(1.0f / (1.0f + exp(-float(x)))));
UNARY_OP(bfloat, hardsigmoid_bf16, bfloat(m_hardsigmoid(float(x))));
UNARY_OP(bfloat, gelu_bf16,
         bfloat(0.5f * float(x) *
                (1.0f + tanh(0.7978845608028654f *
                             (float(x) + 0.044715f * float(x) * float(x) * float(x))))));
UNARY_OP(bfloat, softplus_bf16, bfloat(log(1.0f + exp(float(x)))));
UNARY_OP(bfloat, silu_bf16, bfloat(float(x) / (1.0f + exp(-float(x)))));
UNARY_OP(bfloat, hardsilu_bf16, bfloat(m_hardsilu(float(x))));
UNARY_OP(bfloat, mish_bf16, bfloat(float(x) * tanh(log(1.0f + exp(float(x))))));

// unary - trigonometric
UNARY_OP(bfloat, sin_bf16, bfloat(sin(float(x))));
UNARY_OP(bfloat, cos_bf16, bfloat(cos(float(x))));
UNARY_OP(bfloat, tan_bf16, bfloat(m_tan(float(x))));
UNARY_OP(bfloat, asin_bf16, bfloat(asin(float(x))));
UNARY_OP(bfloat, acos_bf16, bfloat(acos(float(x))));
UNARY_OP(bfloat, atan_bf16, bfloat(atan(float(x))));

// unary - hyperbolic
UNARY_OP(bfloat, sinh_bf16, bfloat(sinh(float(x))));
UNARY_OP(bfloat, cosh_bf16, bfloat(cosh(float(x))));
UNARY_OP(bfloat, tanh_bf16, bfloat(tanh(float(x))));
UNARY_OP(bfloat, asinh_bf16, bfloat(asinh(float(x))));
UNARY_OP(bfloat, acosh_bf16, bfloat(acosh(float(x))));
UNARY_OP(bfloat, atanh_bf16, bfloat(atanh(float(x))));

// unary - exp
UNARY_OP(bfloat, exp_bf16, bfloat(exp(float(x))));
UNARY_OP(bfloat, exp2_bf16, bfloat(exp2(float(x))));
UNARY_OP(bfloat, exp10_bf16, bfloat(m_exp10(float(x))));
UNARY_OP(bfloat, ln_bf16, bfloat(log(float(x))));
UNARY_OP(bfloat, log2_bf16, bfloat(log2(float(x))));
UNARY_OP(bfloat, log10_bf16, bfloat(log10(float(x))));
UNARY_OP(bfloat, ceil_bf16, bfloat(ceil(float(x))));
UNARY_OP(bfloat, floor_bf16, bfloat(floor(float(x))));
UNARY_OP(bfloat, round_bf16, bfloat(round(float(x))));

UNARY_OP(bfloat, erf_bf16, bfloat(m_erf(float(x))));

// unary logical
UNARY_OP_OUTPUT(bfloat, bool, logical_not_bf16, float(x) == 0.0f);

// unary with scalar - arithmetic
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, add_scalar_bf16, x + const_val);
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, sub_scalar_bf16, x - const_val);
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, mul_scalar_bf16, x *const_val);
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, div_scalar_bf16, x / const_val);
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, rem_scalar_bf16, bfloat(fmod(float(x), float(const_val))));
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, pow_scalar_bf16,
                       bfloat(m_pow_float(float(x), float(const_val))));
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, maximum_scalar_bf16,
                       bfloat(maximum(float(x), float(const_val))));
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, minimum_scalar_bf16,
                       bfloat(minimum(float(x), float(const_val))));

// unary with scalar - activation
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, leaky_relu_bf16,
                       x > 0.0bf ? x : bfloat(float(const_val) * float(x)));
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, elu_bf16,
                       bfloat(float(x) > 0.0f ? float(x)
                                              : float(const_val) * (exp(float(x)) - 1.0f)));
UNARY_OP_WITH_CONSTANT(bfloat, bfloat, prelu_bf16,
                       x > 0.0bf ? x : bfloat(float(const_val) * float(x)));

// ============================================================================
// HALF (F16) OPERATIONS
// ============================================================================

// cmp with scalar
UNARY_OP_WITH_CONSTANT(half, bool, eq_scalar_f16, float(x) == float(const_val));
UNARY_OP_WITH_CONSTANT(half, bool, ne_scalar_f16, float(x) != float(const_val));
UNARY_OP_WITH_CONSTANT(half, bool, lt_scalar_f16, float(x) < float(const_val));
UNARY_OP_WITH_CONSTANT(half, bool, le_scalar_f16, float(x) <= float(const_val));
UNARY_OP_WITH_CONSTANT(half, bool, gt_scalar_f16, float(x) > float(const_val));
UNARY_OP_WITH_CONSTANT(half, bool, ge_scalar_f16, float(x) >= float(const_val));

// unary - basic
UNARY_OP(half, neg_f16, -x);
UNARY_OP(half, abs_f16, abs(x));
UNARY_OP(half, sign_f16, (x > 0.0h) ? 1.0h : ((x < 0.0h) ? -1.0h : 0.0h));
UNARY_OP(half, square_f16, x *x);
UNARY_OP(half, sqrt_f16, sqrt(x));
UNARY_OP(half, recip_f16, half(1.0f / float(x)));

// unary - activation
UNARY_OP(half, relu_f16, float(x) > 0.0f ? x : 0.0h);
UNARY_OP(half, sigmoid_f16, half(1.0f / (1.0f + exp(float(-x)))));
UNARY_OP(half, hardsigmoid_f16, half(m_hardsigmoid(float(x))));
UNARY_OP(half, gelu_f16,
         half(0.5f * float(x) *
              (1.0f + tanh(0.7978845608028654f *
                           (float(x) + 0.044715f * float(x) * float(x) * float(x))))));
UNARY_OP(half, softplus_f16, half(log(1.0f + exp(float(x)))));
UNARY_OP(half, silu_f16, half(float(x) / (1.0f + exp(-float(x)))));
UNARY_OP(half, hardsilu_f16, half(m_hardsilu(float(x))));
UNARY_OP(half, mish_f16, half(float(x) * tanh(log(1.0f + exp(float(x))))));

// unary - trigonometric
UNARY_OP(half, sin_f16, half(sin(float(x))));
UNARY_OP(half, cos_f16, half(cos(float(x))));
UNARY_OP(half, tan_f16, half(m_tan(float(x))));
UNARY_OP(half, asin_f16, half(asin(float(x))));
UNARY_OP(half, acos_f16, half(acos(float(x))));
UNARY_OP(half, atan_f16, half(atan(float(x))));

// unary - hyperbolic
UNARY_OP(half, sinh_f16, half(sinh(float(x))));
UNARY_OP(half, cosh_f16, half(cosh(float(x))));
UNARY_OP(half, tanh_f16, half(tanh(float(x))));
UNARY_OP(half, asinh_f16, half(asinh(float(x))));
UNARY_OP(half, acosh_f16, half(acosh(float(x))));
UNARY_OP(half, atanh_f16, half(atanh(float(x))));

// unary - exp
UNARY_OP(half, exp_f16, half(exp(float(x))));
UNARY_OP(half, exp2_f16, half(exp2(float(x))));
UNARY_OP(half, exp10_f16, half(m_exp10(float(x))));
UNARY_OP(half, ln_f16, half(log(float(x))));
UNARY_OP(half, log2_f16, half(log2(float(x))));
UNARY_OP(half, log10_f16, half(log10(float(x))));
UNARY_OP(half, ceil_f16, half(ceil(float(x))));
UNARY_OP(half, floor_f16, half(floor(float(x))));
UNARY_OP(half, round_f16, half(round(float(x))));

UNARY_OP(half, erf_f16, half(m_erf(float(x))));

// unary logical
UNARY_OP_OUTPUT(half, bool, logical_not_f16, float(x) == 0.0f);

// unary with scalar - arithmetic
UNARY_OP_WITH_CONSTANT(half, half, add_scalar_f16, x + const_val);
UNARY_OP_WITH_CONSTANT(half, half, sub_scalar_f16, x - const_val);
UNARY_OP_WITH_CONSTANT(half, half, mul_scalar_f16, x *const_val);
UNARY_OP_WITH_CONSTANT(half, half, div_scalar_f16, x / const_val);
UNARY_OP_WITH_CONSTANT(half, half, rem_scalar_f16, half(fmod(float(x), float(const_val))));
UNARY_OP_WITH_CONSTANT(half, half, pow_scalar_f16, half(m_pow_float(float(x), float(const_val))));
UNARY_OP_WITH_CONSTANT(half, half, maximum_scalar_f16, half(maximum(float(x), float(const_val))));
UNARY_OP_WITH_CONSTANT(half, half, minimum_scalar_f16, half(minimum(float(x), float(const_val))));

// unary with scalar - activation
UNARY_OP_WITH_CONSTANT(half, half, leaky_relu_f16,
                       x > 0.0h ? x : half(float(const_val) * float(x)));
UNARY_OP_WITH_CONSTANT(half, half, elu_f16,
                       half(float(x) > 0.0f ? float(x)
                                            : float(const_val) * (exp(float(x)) - 1.0f)));
UNARY_OP_WITH_CONSTANT(half, half, prelu_f16, x > 0.0h ? x : half(float(const_val) * float(x)));

// ============================================================================
// FLOAT32 OPERATIONS
// ============================================================================

// cmp with scalar
UNARY_OP_WITH_CONSTANT(float, bool, eq_scalar_f32, x == const_val);
UNARY_OP_WITH_CONSTANT(float, bool, ne_scalar_f32, x != const_val);
UNARY_OP_WITH_CONSTANT(float, bool, lt_scalar_f32, x < const_val);
UNARY_OP_WITH_CONSTANT(float, bool, le_scalar_f32, x <= const_val);
UNARY_OP_WITH_CONSTANT(float, bool, gt_scalar_f32, x > const_val);
UNARY_OP_WITH_CONSTANT(float, bool, ge_scalar_f32, x >= const_val);

// unary - basic
UNARY_OP(float, neg_f32, -x);
UNARY_OP(float, abs_f32, abs(x));
UNARY_OP(float, sign_f32, (x > 0) ? 1.0f : ((x < 0) ? -1.0f : 0.0f));
UNARY_OP(float, square_f32, x *x);
UNARY_OP(float, sqrt_f32, sqrt(x));
UNARY_OP(float, recip_f32, 1.0f / x);

// unary - activation
UNARY_OP(float, relu_f32, x > 0 ? x : 0);
UNARY_OP(float, sigmoid_f32, 1.0f / (1.0f + exp(-x)));
UNARY_OP(float, hardsigmoid_f32, m_hardsigmoid(x));
UNARY_OP(float, gelu_f32,
         0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))));
UNARY_OP(float, softplus_f32, log(1.0f + exp(x)));
UNARY_OP(float, silu_f32, x / (1.0f + exp(-x)));
UNARY_OP(float, hardsilu_f32, m_hardsilu(x));
UNARY_OP(float, mish_f32, x *tanh(log(1.0f + exp(x))));

// unary - trigonometric
UNARY_OP(float, sin_f32, sin(x));
UNARY_OP(float, cos_f32, cos(x));
UNARY_OP(float, tan_f32, m_tan(x));
UNARY_OP(float, asin_f32, asin(x));
UNARY_OP(float, acos_f32, acos(x));
UNARY_OP(float, atan_f32, atan(x));

// unary - hyperbolic
UNARY_OP(float, sinh_f32, sinh(x));
UNARY_OP(float, cosh_f32, cosh(x));
UNARY_OP(float, tanh_f32, tanh(x));
UNARY_OP(float, asinh_f32, asinh(x));
UNARY_OP(float, acosh_f32, acosh(x));
UNARY_OP(float, atanh_f32, atanh(x));

// unary - exp
UNARY_OP(float, exp_f32, exp(x));
UNARY_OP(float, exp2_f32, exp2(x));
UNARY_OP(float, exp10_f32, m_exp10(x));
UNARY_OP(float, ln_f32, log(x));
UNARY_OP(float, log2_f32, log2(x));
UNARY_OP(float, log10_f32, log10(x));
UNARY_OP(float, ceil_f32, ceil(x));
UNARY_OP(float, floor_f32, floor(x));
UNARY_OP(float, round_f32, round(x));

UNARY_OP(float, erf_f32, m_erf(x));

// unary logical
UNARY_OP_OUTPUT(float, bool, logical_not_f32, x == 0.0f);

// unary with scalar - arithmetic
UNARY_OP_WITH_CONSTANT(float, float, add_scalar_f32, x + const_val);
UNARY_OP_WITH_CONSTANT(float, float, sub_scalar_f32, x - const_val);
UNARY_OP_WITH_CONSTANT(float, float, mul_scalar_f32, x *const_val);
UNARY_OP_WITH_CONSTANT(float, float, div_scalar_f32, x / const_val);
UNARY_OP_WITH_CONSTANT(float, float, rem_scalar_f32, fmod(x, const_val));
UNARY_OP_WITH_CONSTANT(float, float, pow_scalar_f32, m_pow_float(x, const_val));
UNARY_OP_WITH_CONSTANT(float, float, maximum_scalar_f32, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(float, float, minimum_scalar_f32, minimum(x, const_val));

// unary with scalar - activation
UNARY_OP_WITH_CONSTANT(float, float, leaky_relu_f32, x > 0 ? x : const_val * x);
UNARY_OP_WITH_CONSTANT(float, float, elu_f32, x > 0 ? x : const_val * (exp(x) - 1.0f));
UNARY_OP_WITH_CONSTANT(float, float, prelu_f32, x > 0 ? x : const_val * x);

// ============================================================================
// UINT8 OPERATIONS
// ============================================================================

// cmp with scalar
UNARY_OP_WITH_CONSTANT(uint8_t, bool, eq_scalar_u8, x == const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, ne_scalar_u8, x != const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, lt_scalar_u8, x < const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, le_scalar_u8, x <= const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, gt_scalar_u8, x > const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, bool, ge_scalar_u8, x >= const_val);

// unary - basic
UNARY_OP(uint8_t, sign_u8, (x > 0) ? 1 : 0);
UNARY_OP(uint8_t, square_u8, x *x);
UNARY_OP(uint8_t, sqrt_u8, (uint8_t)sqrt(float(x)));

// unary - activation
UNARY_OP(uint8_t, relu_u8, x > 0 ? x : 0);

// unary logical
UNARY_OP_OUTPUT(uint8_t, bool, logical_not_u8, x == 0u);

// unary with scalar - arithmetic
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, add_scalar_u8, x + const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, sub_scalar_u8, (x > const_val) ? x - const_val : 0);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, mul_scalar_u8, x *const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, div_scalar_u8, x / const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, rem_scalar_u8, x % const_val);
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, pow_scalar_u8,
                       (uint8_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, maximum_scalar_u8, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(uint8_t, uint8_t, minimum_scalar_u8, minimum(x, const_val));

// ============================================================================
// UINT16 OPERATIONS
// ============================================================================

// cmp with scalar
UNARY_OP_WITH_CONSTANT(uint16_t, bool, eq_scalar_u16, x == const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, bool, ne_scalar_u16, x != const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, bool, lt_scalar_u16, x < const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, bool, le_scalar_u16, x <= const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, bool, gt_scalar_u16, x > const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, bool, ge_scalar_u16, x >= const_val);

// unary - basic
UNARY_OP(uint16_t, sign_u16, (x > 0) ? 1 : 0);
UNARY_OP(uint16_t, square_u16, x *x);
UNARY_OP(uint16_t, sqrt_u16, (uint16_t)sqrt(float(x)));

// unary - activation
UNARY_OP(uint16_t, relu_u16, x > 0 ? x : 0);

// unary logical
UNARY_OP_OUTPUT(uint16_t, bool, logical_not_u16, x == 0u);

// unary with scalar - arithmetic
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, add_scalar_u16, x + const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, sub_scalar_u16, (x > const_val) ? x - const_val : 0);
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, mul_scalar_u16, x *const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, div_scalar_u16, x / const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, rem_scalar_u16, x % const_val);
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, pow_scalar_u16,
                       (uint16_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, maximum_scalar_u16, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(uint16_t, uint16_t, minimum_scalar_u16, minimum(x, const_val));

// ============================================================================
// UINT32 OPERATIONS
// ============================================================================

// cmp with scalar
UNARY_OP_WITH_CONSTANT(uint32_t, bool, eq_scalar_u32, x == const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, ne_scalar_u32, x != const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, lt_scalar_u32, x < const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, le_scalar_u32, x <= const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, gt_scalar_u32, x > const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, bool, ge_scalar_u32, x >= const_val);

// unary - basic
UNARY_OP(uint32_t, sign_u32, (x > 0) ? 1 : 0);
UNARY_OP(uint32_t, square_u32, x *x);
UNARY_OP(uint32_t, sqrt_u32, (uint32_t)sqrt(float(x)));

// unary - activation
UNARY_OP(uint32_t, relu_u32, x > 0 ? x : 0);

// unary logical
UNARY_OP_OUTPUT(uint32_t, bool, logical_not_u32, x == 0u);

// unary with scalar - arithmetic
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, add_scalar_u32, x + const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, sub_scalar_u32, (x > const_val) ? x - const_val : 0);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, mul_scalar_u32, x *const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, div_scalar_u32, x / const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, rem_scalar_u32, x % const_val);
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, pow_scalar_u32,
                       (uint32_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, maximum_scalar_u32, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(uint32_t, uint32_t, minimum_scalar_u32, minimum(x, const_val));

// ============================================================================
// UINT64 OPERATIONS
// ============================================================================

// cmp with scalar
UNARY_OP_WITH_CONSTANT(uint64_t, bool, eq_scalar_u64, x == const_val);
UNARY_OP_WITH_CONSTANT(uint64_t, bool, ne_scalar_u64, x != const_val);
UNARY_OP_WITH_CONSTANT(uint64_t, bool, lt_scalar_u64, x < const_val);
UNARY_OP_WITH_CONSTANT(uint64_t, bool, le_scalar_u64, x <= const_val);
UNARY_OP_WITH_CONSTANT(uint64_t, bool, gt_scalar_u64, x > const_val);
UNARY_OP_WITH_CONSTANT(uint64_t, bool, ge_scalar_u64, x >= const_val);

// unary - basic
UNARY_OP(uint64_t, sign_u64, (x > 0) ? 1 : 0);
UNARY_OP(uint64_t, square_u64, x *x);
UNARY_OP(uint64_t, sqrt_u64, (uint64_t)sqrt(float(x)));

// unary - activation
UNARY_OP(uint64_t, relu_u64, x > 0 ? x : 0);

// unary logical
UNARY_OP_OUTPUT(uint64_t, bool, logical_not_u64, x == 0u);

// unary with scalar - arithmetic
UNARY_OP_WITH_CONSTANT(uint64_t, uint64_t, add_scalar_u64, x + const_val);
UNARY_OP_WITH_CONSTANT(uint64_t, uint64_t, sub_scalar_u64, (x > const_val) ? x - const_val : 0);
UNARY_OP_WITH_CONSTANT(uint64_t, uint64_t, mul_scalar_u64, x *const_val);
UNARY_OP_WITH_CONSTANT(uint64_t, uint64_t, div_scalar_u64, x / const_val);
UNARY_OP_WITH_CONSTANT(uint64_t, uint64_t, rem_scalar_u64, x % const_val);
UNARY_OP_WITH_CONSTANT(uint64_t, uint64_t, pow_scalar_u64,
                       (uint64_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(uint64_t, uint64_t, maximum_scalar_u64, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(uint64_t, uint64_t, minimum_scalar_u64, minimum(x, const_val));

// ============================================================================
// INT8 OPERATIONS
// ============================================================================

// cmp with scalar
UNARY_OP_WITH_CONSTANT(int8_t, bool, eq_scalar_i8, x == const_val);
UNARY_OP_WITH_CONSTANT(int8_t, bool, ne_scalar_i8, x != const_val);
UNARY_OP_WITH_CONSTANT(int8_t, bool, lt_scalar_i8, x < const_val);
UNARY_OP_WITH_CONSTANT(int8_t, bool, le_scalar_i8, x <= const_val);
UNARY_OP_WITH_CONSTANT(int8_t, bool, gt_scalar_i8, x > const_val);
UNARY_OP_WITH_CONSTANT(int8_t, bool, ge_scalar_i8, x >= const_val);

// unary - basic
UNARY_OP(int8_t, neg_i8, -x);
UNARY_OP(int8_t, abs_i8, abs(x));
UNARY_OP(int8_t, sign_i8, (x > 0) ? 1 : ((x < 0) ? -1 : 0));
UNARY_OP(int8_t, square_i8, x *x);
UNARY_OP(int8_t, sqrt_i8, (int8_t)sqrt(float(abs(x))));

// unary - activation
UNARY_OP(int8_t, relu_i8, x > 0 ? x : 0);

// unary logical
UNARY_OP_OUTPUT(int8_t, bool, logical_not_i8, x == 0);

// unary with scalar - arithmetic
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, add_scalar_i8, x + const_val);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, sub_scalar_i8, x - const_val);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, mul_scalar_i8, x *const_val);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, div_scalar_i8, x / const_val);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, rem_scalar_i8, x % const_val);
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, pow_scalar_i8,
                       (int8_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, maximum_scalar_i8, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(int8_t, int8_t, minimum_scalar_i8, minimum(x, const_val));

// ============================================================================
// INT16 OPERATIONS
// ============================================================================

// cmp with scalar
UNARY_OP_WITH_CONSTANT(int16_t, bool, eq_scalar_i16, x == const_val);
UNARY_OP_WITH_CONSTANT(int16_t, bool, ne_scalar_i16, x != const_val);
UNARY_OP_WITH_CONSTANT(int16_t, bool, lt_scalar_i16, x < const_val);
UNARY_OP_WITH_CONSTANT(int16_t, bool, le_scalar_i16, x <= const_val);
UNARY_OP_WITH_CONSTANT(int16_t, bool, gt_scalar_i16, x > const_val);
UNARY_OP_WITH_CONSTANT(int16_t, bool, ge_scalar_i16, x >= const_val);

// unary - basic
UNARY_OP(int16_t, neg_i16, -x);
UNARY_OP(int16_t, abs_i16, abs(x));
UNARY_OP(int16_t, sign_i16, (x > 0) ? 1 : ((x < 0) ? -1 : 0));
UNARY_OP(int16_t, square_i16, x *x);
UNARY_OP(int16_t, sqrt_i16, (int16_t)sqrt(float(abs(x))));

// unary - activation
UNARY_OP(int16_t, relu_i16, x > 0 ? x : 0);

// unary logical
UNARY_OP_OUTPUT(int16_t, bool, logical_not_i16, x == 0);

// unary with scalar - arithmetic
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, add_scalar_i16, x + const_val);
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, sub_scalar_i16, x - const_val);
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, mul_scalar_i16, x *const_val);
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, div_scalar_i16, x / const_val);
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, rem_scalar_i16, x % const_val);
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, pow_scalar_i16,
                       (int16_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, maximum_scalar_i16, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(int16_t, int16_t, minimum_scalar_i16, minimum(x, const_val));

// ============================================================================
// INT32 OPERATIONS
// ============================================================================

// cmp with scalar
UNARY_OP_WITH_CONSTANT(int32_t, bool, eq_scalar_i32, x == const_val);
UNARY_OP_WITH_CONSTANT(int32_t, bool, ne_scalar_i32, x != const_val);
UNARY_OP_WITH_CONSTANT(int32_t, bool, lt_scalar_i32, x < const_val);
UNARY_OP_WITH_CONSTANT(int32_t, bool, le_scalar_i32, x <= const_val);
UNARY_OP_WITH_CONSTANT(int32_t, bool, gt_scalar_i32, x > const_val);
UNARY_OP_WITH_CONSTANT(int32_t, bool, ge_scalar_i32, x >= const_val);

// unary - basic
UNARY_OP(int32_t, neg_i32, -x);
UNARY_OP(int32_t, abs_i32, abs(x));
UNARY_OP(int32_t, sign_i32, (x > 0) ? 1 : ((x < 0) ? -1 : 0));
UNARY_OP(int32_t, square_i32, x *x);
UNARY_OP(int32_t, sqrt_i32, (int32_t)sqrt(float(abs(x))));

// unary - activation
UNARY_OP(int32_t, relu_i32, x > 0 ? x : 0);

// unary logical
UNARY_OP_OUTPUT(int32_t, bool, logical_not_i32, x == 0);

// unary with scalar - arithmetic
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, add_scalar_i32, x + const_val);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, sub_scalar_i32, x - const_val);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, mul_scalar_i32, x *const_val);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, div_scalar_i32, x / const_val);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, rem_scalar_i32, x % const_val);
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, pow_scalar_i32,
                       (int32_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, maximum_scalar_i32, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(int32_t, int32_t, minimum_scalar_i32, minimum(x, const_val));

// ============================================================================
// INT64 OPERATIONS
// ============================================================================

// cmp with scalar
UNARY_OP_WITH_CONSTANT(int64_t, bool, eq_scalar_i64, x == const_val);
UNARY_OP_WITH_CONSTANT(int64_t, bool, ne_scalar_i64, x != const_val);
UNARY_OP_WITH_CONSTANT(int64_t, bool, lt_scalar_i64, x < const_val);
UNARY_OP_WITH_CONSTANT(int64_t, bool, le_scalar_i64, x <= const_val);
UNARY_OP_WITH_CONSTANT(int64_t, bool, gt_scalar_i64, x > const_val);
UNARY_OP_WITH_CONSTANT(int64_t, bool, ge_scalar_i64, x >= const_val);

// unary - basic
UNARY_OP(int64_t, neg_i64, -x);
UNARY_OP(int64_t, abs_i64, abs(x));
UNARY_OP(int64_t, sign_i64, (x > 0) ? 1 : ((x < 0) ? -1 : 0));
UNARY_OP(int64_t, square_i64, x *x);
UNARY_OP(int64_t, sqrt_i64, (int64_t)sqrt(float(abs(x))));

// unary - activation
UNARY_OP(int64_t, relu_i64, x > 0 ? x : 0);

// unary logical
UNARY_OP_OUTPUT(int64_t, bool, logical_not_i64, x == 0);

// unary with scalar - arithmetic
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, add_scalar_i64, x + const_val);
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, sub_scalar_i64, x - const_val);
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, mul_scalar_i64, x *const_val);
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, div_scalar_i64, x / const_val);
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, rem_scalar_i64, x % const_val);
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, pow_scalar_i64,
                       (int64_t)m_pow_float((float)x, (float)const_val));
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, maximum_scalar_i64, maximum(x, const_val));
UNARY_OP_WITH_CONSTANT(int64_t, int64_t, minimum_scalar_i64, minimum(x, const_val));
