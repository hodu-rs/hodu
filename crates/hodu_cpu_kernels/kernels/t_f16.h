#ifndef HODU_CPU_KERNELS_T_F16_H
#define HODU_CPU_KERNELS_T_F16_H

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// IEEE 754 Float16: 1 sign bit, 5 exponent bits, 10 mantissa bits
typedef uint16_t f16_t;

// ============================================================================
// CONSTANTS
// ============================================================================

#define F16_ZERO ((f16_t)0x0000)
#define F16_ONE ((f16_t)0x3C00)
#define F16_NEG_INF ((f16_t)0xFC00)
#define F16_POS_INF ((f16_t)0x7C00)

// ============================================================================
// CONVERSION
// ============================================================================

static inline float f16_to_float(f16_t val) {
    uint32_t sign = (val >> 15) & 1;
    uint32_t exp = (val >> 10) & 0x1F;
    uint32_t mant = val & 0x3FF;

    if (exp == 0x1F) {
        if (mant == 0)
            return sign ? -INFINITY : INFINITY;
        return NAN;
    }

    if (exp == 0) {
        if (mant == 0)
            return sign ? -0.0f : 0.0f;
        return ldexpf((float)mant / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
    }

    float result = ldexpf(1.0f + (float)mant / 1024.0f, (int)exp - 15);
    return sign ? -result : result;
}

static inline f16_t float_to_f16(float val) {
    if (val == 0.0f)
        return 0;
    if (isnan(val))
        return 0x7E00;

    uint32_t sign = val < 0 ? 1 : 0;
    val = fabsf(val);

    if (isinf(val) || val > 65504.0f)
        return (sign << 15) | 0x7C00;
    if (val < 0.00006103515625f)
        return sign << 15;

    int exp;
    float mant = frexpf(val, &exp);
    exp += 14;

    if (exp <= 0) {
        mant = ldexpf(mant, exp);
        exp = 0;
    } else if (exp >= 31) {
        return (sign << 15) | 0x7C00;
    }

    int mant_bits = (int)((mant * 2.0f - 1.0f) * 1024.0f + 0.5f);
    mant_bits = mant_bits > 1023 ? 1023 : mant_bits;

    return (sign << 15) | (exp << 10) | mant_bits;
}

// ============================================================================
// ARITHMETIC OPERATIONS
// ============================================================================

static inline f16_t f16_add(f16_t a, f16_t b) {
    return float_to_f16(f16_to_float(a) + f16_to_float(b));
}

static inline f16_t f16_sub(f16_t a, f16_t b) {
    return float_to_f16(f16_to_float(a) - f16_to_float(b));
}

static inline f16_t f16_mul(f16_t a, f16_t b) {
    return float_to_f16(f16_to_float(a) * f16_to_float(b));
}

static inline f16_t f16_div(f16_t a, f16_t b) {
    return float_to_f16(f16_to_float(a) / f16_to_float(b));
}

static inline f16_t f16_neg(f16_t a) { return float_to_f16(-f16_to_float(a)); }

// ============================================================================
// COMPARISON
// ============================================================================

static inline int f16_lt(f16_t a, f16_t b) { return f16_to_float(a) < f16_to_float(b); }

static inline int f16_le(f16_t a, f16_t b) { return f16_to_float(a) <= f16_to_float(b); }

static inline int f16_gt(f16_t a, f16_t b) { return f16_to_float(a) > f16_to_float(b); }

static inline int f16_ge(f16_t a, f16_t b) { return f16_to_float(a) >= f16_to_float(b); }

static inline int f16_eq(f16_t a, f16_t b) { return f16_to_float(a) == f16_to_float(b); }

// ============================================================================
// MATH FUNCTIONS
// ============================================================================

static inline f16_t f16_max(f16_t a, f16_t b) {
    float a_f = f16_to_float(a);
    float b_f = f16_to_float(b);
    return float_to_f16(a_f > b_f ? a_f : b_f);
}

static inline f16_t f16_min(f16_t a, f16_t b) {
    float a_f = f16_to_float(a);
    float b_f = f16_to_float(b);
    return float_to_f16(a_f < b_f ? a_f : b_f);
}

static inline f16_t f16_abs(f16_t a) { return float_to_f16(fabsf(f16_to_float(a))); }

static inline f16_t f16_sqrt(f16_t a) { return float_to_f16(sqrtf(f16_to_float(a))); }

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_T_F16_H
