#ifndef HODU_CPU_KERNELS_T_F8E5M2_H
#define HODU_CPU_KERNELS_T_F8E5M2_H

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// FP8 E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits
typedef uint8_t f8e5m2_t;

// ============================================================================
// CONSTANTS
// ============================================================================

#define F8E5M2_ZERO ((f8e5m2_t)0x00)
#define F8E5M2_ONE ((f8e5m2_t)0x3C) // exp=15 (bias), mant=0
#define F8E5M2_NEG_INF ((f8e5m2_t)0xFF)
#define F8E5M2_POS_INF ((f8e5m2_t)0x7F)

// ============================================================================
// CONVERSION
// ============================================================================

static inline float f8e5m2_to_float(f8e5m2_t val) {
    uint32_t sign = (val >> 7) & 1;
    uint32_t exp = (val >> 2) & 0x1F;
    uint32_t mant = val & 0x3;

    // E5M2 uses bias of 15 (same as FP16)
    // Special values: exp=31 is infinity/NaN
    if (exp == 31) {
        if (mant == 0)
            return sign ? -INFINITY : INFINITY;
        return NAN;
    }

    if (exp == 0) {
        if (mant == 0)
            return sign ? -0.0f : 0.0f;
        // Subnormal: 2^(-14) * (mant/4)
        return ldexpf((float)mant / 4.0f, -14) * (sign ? -1.0f : 1.0f);
    }

    // Normal: 2^(exp-15) * (1 + mant/4)
    float result = ldexpf(1.0f + (float)mant / 4.0f, (int)exp - 15);
    return sign ? -result : result;
}

static inline f8e5m2_t float_to_f8e5m2(float val) {
    if (val == 0.0f)
        return 0;
    if (isnan(val))
        return 0x7F; // NaN

    uint32_t sign = val < 0 ? 1 : 0;
    val = fabsf(val);

    // E5M2 max value is ~57344
    if (isinf(val) || val > 57344.0f)
        return (sign << 7) | 0x7F;

    // E5M2 min normal value is 2^(-14) ≈ 0.000061
    if (val < 0.0000152587890625f) // 2^(-16)
        return sign << 7;

    int exp;
    float mant = frexpf(val, &exp);
    exp += 14; // bias = 15, but frexp returns [0.5, 1), so adjust

    if (exp <= 0) {
        // Subnormal
        mant = ldexpf(mant, exp);
        exp = 0;
    } else if (exp >= 31) {
        return (sign << 7) | 0x7F;
    }

    int mant_bits = (int)((mant * 2.0f - 1.0f) * 4.0f + 0.5f);
    mant_bits = mant_bits > 3 ? 3 : mant_bits;

    return (sign << 7) | (exp << 2) | mant_bits;
}

// ============================================================================
// ARITHMETIC OPERATIONS
// ============================================================================

static inline f8e5m2_t f8e5m2_add(f8e5m2_t a, f8e5m2_t b) {
    return float_to_f8e5m2(f8e5m2_to_float(a) + f8e5m2_to_float(b));
}

static inline f8e5m2_t f8e5m2_sub(f8e5m2_t a, f8e5m2_t b) {
    return float_to_f8e5m2(f8e5m2_to_float(a) - f8e5m2_to_float(b));
}

static inline f8e5m2_t f8e5m2_mul(f8e5m2_t a, f8e5m2_t b) {
    return float_to_f8e5m2(f8e5m2_to_float(a) * f8e5m2_to_float(b));
}

static inline f8e5m2_t f8e5m2_div(f8e5m2_t a, f8e5m2_t b) {
    return float_to_f8e5m2(f8e5m2_to_float(a) / f8e5m2_to_float(b));
}

static inline f8e5m2_t f8e5m2_neg(f8e5m2_t a) { return float_to_f8e5m2(-f8e5m2_to_float(a)); }

// ============================================================================
// COMPARISON
// ============================================================================

static inline int f8e5m2_lt(f8e5m2_t a, f8e5m2_t b) {
    return f8e5m2_to_float(a) < f8e5m2_to_float(b);
}

static inline int f8e5m2_le(f8e5m2_t a, f8e5m2_t b) {
    return f8e5m2_to_float(a) <= f8e5m2_to_float(b);
}

static inline int f8e5m2_gt(f8e5m2_t a, f8e5m2_t b) {
    return f8e5m2_to_float(a) > f8e5m2_to_float(b);
}

static inline int f8e5m2_ge(f8e5m2_t a, f8e5m2_t b) {
    return f8e5m2_to_float(a) >= f8e5m2_to_float(b);
}

static inline int f8e5m2_eq(f8e5m2_t a, f8e5m2_t b) {
    return f8e5m2_to_float(a) == f8e5m2_to_float(b);
}

// ============================================================================
// MATH FUNCTIONS
// ============================================================================

static inline f8e5m2_t f8e5m2_max(f8e5m2_t a, f8e5m2_t b) {
    float a_f = f8e5m2_to_float(a);
    float b_f = f8e5m2_to_float(b);
    return float_to_f8e5m2(a_f > b_f ? a_f : b_f);
}

static inline f8e5m2_t f8e5m2_min(f8e5m2_t a, f8e5m2_t b) {
    float a_f = f8e5m2_to_float(a);
    float b_f = f8e5m2_to_float(b);
    return float_to_f8e5m2(a_f < b_f ? a_f : b_f);
}

static inline f8e5m2_t f8e5m2_abs(f8e5m2_t a) { return float_to_f8e5m2(fabsf(f8e5m2_to_float(a))); }

static inline f8e5m2_t f8e5m2_sqrt(f8e5m2_t a) {
    return float_to_f8e5m2(sqrtf(f8e5m2_to_float(a)));
}

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_T_F8E5M2_H
