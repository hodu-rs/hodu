#ifndef HODU_CPU_KERNELS_T_F8E4M3_H
#define HODU_CPU_KERNELS_T_F8E4M3_H

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// FP8 E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
typedef uint8_t f8e4m3_t;

// ============================================================================
// CONSTANTS
// ============================================================================

#define F8E4M3_ZERO ((f8e4m3_t)0x00)
#define F8E4M3_ONE ((f8e4m3_t)0x38) // exp=7 (bias), mant=0
#define F8E4M3_NEG_INF ((f8e4m3_t)0xFF)
#define F8E4M3_POS_INF ((f8e4m3_t)0x7F)

// ============================================================================
// CONVERSION
// ============================================================================

static inline float f8e4m3_to_float(f8e4m3_t val) {
    uint32_t sign = (val >> 7) & 1;
    uint32_t exp = (val >> 3) & 0xF;
    uint32_t mant = val & 0x7;

    // E4M3 uses bias of 7
    // Special values: exp=15 is infinity/NaN
    if (exp == 15) {
        if (mant == 0)
            return sign ? -INFINITY : INFINITY;
        return NAN;
    }

    if (exp == 0) {
        if (mant == 0)
            return sign ? -0.0f : 0.0f;
        // Subnormal: 2^(-6) * (mant/8)
        return ldexpf((float)mant / 8.0f, -6) * (sign ? -1.0f : 1.0f);
    }

    // Normal: 2^(exp-7) * (1 + mant/8)
    float result = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    return sign ? -result : result;
}

static inline f8e4m3_t float_to_f8e4m3(float val) {
    if (val == 0.0f)
        return 0;
    if (isnan(val))
        return 0x7F; // NaN

    uint32_t sign = val < 0 ? 1 : 0;
    val = fabsf(val);

    // E4M3 max value is ~448
    if (isinf(val) || val > 448.0f)
        return (sign << 7) | 0x7F;

    // E4M3 min normal value is 2^(-6) ≈ 0.015625
    if (val < 0.001953125f) // 2^(-9)
        return sign << 7;

    int exp;
    float mant = frexpf(val, &exp);
    exp += 6; // bias = 7, but frexp returns [0.5, 1), so adjust

    if (exp <= 0) {
        // Subnormal
        mant = ldexpf(mant, exp);
        exp = 0;
    } else if (exp >= 15) {
        return (sign << 7) | 0x7F;
    }

    int mant_bits = (int)((mant * 2.0f - 1.0f) * 8.0f + 0.5f);
    mant_bits = mant_bits > 7 ? 7 : mant_bits;

    return (sign << 7) | (exp << 3) | mant_bits;
}

// ============================================================================
// ARITHMETIC OPERATIONS
// ============================================================================

static inline f8e4m3_t f8e4m3_add(f8e4m3_t a, f8e4m3_t b) {
    return float_to_f8e4m3(f8e4m3_to_float(a) + f8e4m3_to_float(b));
}

static inline f8e4m3_t f8e4m3_sub(f8e4m3_t a, f8e4m3_t b) {
    return float_to_f8e4m3(f8e4m3_to_float(a) - f8e4m3_to_float(b));
}

static inline f8e4m3_t f8e4m3_mul(f8e4m3_t a, f8e4m3_t b) {
    return float_to_f8e4m3(f8e4m3_to_float(a) * f8e4m3_to_float(b));
}

static inline f8e4m3_t f8e4m3_div(f8e4m3_t a, f8e4m3_t b) {
    return float_to_f8e4m3(f8e4m3_to_float(a) / f8e4m3_to_float(b));
}

static inline f8e4m3_t f8e4m3_neg(f8e4m3_t a) { return float_to_f8e4m3(-f8e4m3_to_float(a)); }

// ============================================================================
// COMPARISON
// ============================================================================

static inline int f8e4m3_lt(f8e4m3_t a, f8e4m3_t b) {
    return f8e4m3_to_float(a) < f8e4m3_to_float(b);
}

static inline int f8e4m3_le(f8e4m3_t a, f8e4m3_t b) {
    return f8e4m3_to_float(a) <= f8e4m3_to_float(b);
}

static inline int f8e4m3_gt(f8e4m3_t a, f8e4m3_t b) {
    return f8e4m3_to_float(a) > f8e4m3_to_float(b);
}

static inline int f8e4m3_ge(f8e4m3_t a, f8e4m3_t b) {
    return f8e4m3_to_float(a) >= f8e4m3_to_float(b);
}

static inline int f8e4m3_eq(f8e4m3_t a, f8e4m3_t b) {
    return f8e4m3_to_float(a) == f8e4m3_to_float(b);
}

// ============================================================================
// MATH FUNCTIONS
// ============================================================================

static inline f8e4m3_t f8e4m3_max(f8e4m3_t a, f8e4m3_t b) {
    float a_f = f8e4m3_to_float(a);
    float b_f = f8e4m3_to_float(b);
    return float_to_f8e4m3(a_f > b_f ? a_f : b_f);
}

static inline f8e4m3_t f8e4m3_min(f8e4m3_t a, f8e4m3_t b) {
    float a_f = f8e4m3_to_float(a);
    float b_f = f8e4m3_to_float(b);
    return float_to_f8e4m3(a_f < b_f ? a_f : b_f);
}

static inline f8e4m3_t f8e4m3_abs(f8e4m3_t a) { return float_to_f8e4m3(fabsf(f8e4m3_to_float(a))); }

static inline f8e4m3_t f8e4m3_sqrt(f8e4m3_t a) {
    return float_to_f8e4m3(sqrtf(f8e4m3_to_float(a)));
}

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_T_F8E4M3_H
