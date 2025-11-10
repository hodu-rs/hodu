#ifndef HODU_CPU_KERNELS_MATH_UTILS_H
#define HODU_CPU_KERNELS_MATH_UTILS_H

#include "constants.h"
#include "types.h"
#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// INFINITY CONSTANTS FOR EXOTIC FLOATING-POINT TYPES
// ============================================================================

// FP8 E4M3 infinity values (exponent=0xF)
#define F8E4M3_NEG_INF ((f8e4m3_t)0xFF) // 1 sign | 1111 exp | 111 mant
#define F8E4M3_POS_INF ((f8e4m3_t)0x7F) // 0 sign | 1111 exp | 111 mant

// FP8 E5M2 infinity values (exponent=0x1F)
#define F8E5M2_NEG_INF ((f8e5m2_t)0xFF) // 1 sign | 11111 exp | 11 mant
#define F8E5M2_POS_INF ((f8e5m2_t)0x7F) // 0 sign | 11111 exp | 11 mant

// BFloat16 infinity values (exponent=0xFF)
#define BF16_NEG_INF ((bf16_t)0xFF80) // 1 sign | 11111111 exp | 0000000 mant
#define BF16_POS_INF ((bf16_t)0x7F80) // 0 sign | 11111111 exp | 0000000 mant

// Float16 infinity values (exponent=0x1F)
#define F16_NEG_INF ((f16_t)0xFC00) // 1 sign | 11111 exp | 0000000000 mant
#define F16_POS_INF ((f16_t)0x7C00) // 0 sign | 11111 exp | 0000000000 mant

// ============================================================================
// GENERIC MATH HELPERS
// ============================================================================

// Integer power function for integer types
static inline int64_t ipow_i64(int64_t base, int64_t exp) {
    if (exp < 0)
        return 0;
    int64_t result = 1;
    while (exp > 0) {
        if (exp & 1)
            result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

static inline uint64_t ipow_u64(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1)
            result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

// Float power with optimization
static inline float powf_opt(float base, float exponent) {
    if (exponent == 0.0f)
        return 1.0f;
    if (base == 0.0f)
        return (exponent > 0.0f) ? 0.0f : INFINITY;
    if (base == 1.0f)
        return 1.0f;
    if (exponent == 1.0f)
        return base;

    // Check if exponent is an integer
    if (floorf(exponent) == exponent) {
        if (exponent >= 0.0f && exponent < 32.0f) {
            int iexp = (int)exponent;
            float result = 1.0f;
            while (iexp > 0) {
                if (iexp & 1)
                    result *= base;
                base *= base;
                iexp >>= 1;
            }
            return result;
        } else if (exponent < 0.0f && exponent > -32.0f) {
            return 1.0f / powf_opt(base, -exponent);
        }
    }

    if (base < 0.0f)
        return NAN;

    return powf(base, exponent);
}

static inline double pow_opt(double base, double exponent) {
    if (exponent == 0.0)
        return 1.0;
    if (base == 0.0)
        return (exponent > 0.0) ? 0.0 : INFINITY;
    if (base == 1.0)
        return 1.0;
    if (exponent == 1.0)
        return base;

    if (floor(exponent) == exponent) {
        if (exponent >= 0.0 && exponent < 32.0) {
            int iexp = (int)exponent;
            double result = 1.0;
            while (iexp > 0) {
                if (iexp & 1)
                    result *= base;
                base *= base;
                iexp >>= 1;
            }
            return result;
        } else if (exponent < 0.0 && exponent > -32.0) {
            return 1.0 / pow_opt(base, -exponent);
        }
    }

    if (base < 0.0)
        return NAN;

    return pow(base, exponent);
}

// Optimized tan
static inline float tanf_opt(float x) {
    x = fmodf(x, 2.0f * (float)M_PI);
    if (x > (float)M_PI)
        x -= 2.0f * (float)M_PI;
    else if (x < -(float)M_PI)
        x += 2.0f * (float)M_PI;

    float halfPi = (float)M_PI / 2.0f;
    float eps = 1e-6f;

    if (fabsf(fabsf(x) - halfPi) < eps) {
        return x > 0 ? 1e6f : -1e6f;
    }

    return sinf(x) / cosf(x);
}

static inline double tan_opt(double x) {
    x = fmod(x, 2.0 * M_PI);
    if (x > M_PI)
        x -= 2.0 * M_PI;
    else if (x < -M_PI)
        x += 2.0 * M_PI;

    double halfPi = M_PI / 2.0;
    double eps = 1e-14;

    if (fabs(fabs(x) - halfPi) < eps) {
        return x > 0 ? 1e14 : -1e14;
    }

    return sin(x) / cos(x);
}

// Base-10 exponential
static inline float exp10f_opt(float x) { return expf(x * (float)M_LN10); }

static inline double exp10_opt(double x) { return exp(x * M_LN10); }

// ============================================================================
// ACTIVATION FUNCTION HELPERS
// ============================================================================

static inline float gelu_helper_f32(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static inline double gelu_helper_f64(double x) {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

static inline float softplus_helper_f32(float x) { return logf(1.0f + expf(x)); }

static inline double softplus_helper_f64(double x) { return log(1.0 + exp(x)); }

static inline float silu_helper_f32(float x) { return x / (1.0f + expf(-x)); }

static inline double silu_helper_f64(double x) { return x / (1.0 + exp(-x)); }

static inline float mish_helper_f32(float x) { return x * tanhf(softplus_helper_f32(x)); }

static inline double mish_helper_f64(double x) { return x * tanh(softplus_helper_f64(x)); }

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_MATH_UTILS_H
