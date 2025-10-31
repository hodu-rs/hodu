#ifndef HODU_CPU_KERNELS_TYPES_H
#define HODU_CPU_KERNELS_TYPES_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

// Standard types
typedef float f32_t;
typedef double f64_t;
typedef uint8_t u8_t;
typedef uint16_t u16_t;
typedef uint32_t u32_t;
typedef uint64_t u64_t;
typedef int8_t i8_t;
typedef int16_t i16_t;
typedef int32_t i32_t;
typedef int64_t i64_t;

// Exotic floating-point types (stored as integer bit patterns)
typedef uint8_t f8e4m3_t; // FP8 E4M3: 1 sign, 4 exponent, 3 mantissa
typedef uint8_t f8e5m2_t; // FP8 E5M2: 1 sign, 5 exponent, 2 mantissa
typedef uint16_t bf16_t;  // BFloat16: 1 sign, 8 exponent, 7 mantissa
typedef uint16_t f16_t;   // IEEE 754 Float16: 1 sign, 5 exponent, 10 mantissa

// ============================================================================
// FP8 E4M3 CONVERSION (1 sign bit, 4 exponent, 3 mantissa)
// ============================================================================

static inline float fp8_e4m3_to_float(uint8_t val) {
    if (val == 0)
        return 0.0f;
    int sign = (val >> 7) & 1;
    int exp = (val >> 3) & 0xF;
    int mant = val & 0x7;

    if (exp == 0xF)
        return sign ? -INFINITY : INFINITY;
    if (exp == 0) {
        float result = ldexpf((float)mant / 8.0f, -6);
        return sign ? -result : result;
    }

    float result = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
    return sign ? -result : result;
}

static inline uint8_t float_to_fp8_e4m3(float val) {
    if (val == 0.0f)
        return 0;
    if (isnan(val))
        return 0x7F;

    uint32_t sign = val < 0 ? 1 : 0;
    val = fabsf(val);

    if (isinf(val) || val > 448.0f)
        return (sign << 7) | 0x7F;
    if (val < 0.001953125f)
        return sign << 7;

    int exp;
    float mant = frexpf(val, &exp);
    exp += 6;

    if (exp <= 0) {
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
// FP8 E5M2 CONVERSION (1 sign bit, 5 exponent, 2 mantissa)
// ============================================================================

static inline float fp8_e5m2_to_float(uint8_t val) {
    if (val == 0)
        return 0.0f;
    int sign = (val >> 7) & 1;
    int exp = (val >> 2) & 0x1F;
    int mant = val & 0x3;

    if (exp == 0x1F)
        return sign ? -INFINITY : INFINITY;
    if (exp == 0) {
        float result = ldexpf((float)mant / 4.0f, -14);
        return sign ? -result : result;
    }

    float result = ldexpf(1.0f + (float)mant / 4.0f, exp - 15);
    return sign ? -result : result;
}

static inline uint8_t float_to_fp8_e5m2(float val) {
    if (val == 0.0f)
        return 0;
    if (isnan(val))
        return 0x7F;

    uint32_t sign = val < 0 ? 1 : 0;
    val = fabsf(val);

    if (isinf(val) || val > 57344.0f)
        return (sign << 7) | 0x7F;
    if (val < 0.0000152587890625f)
        return sign << 7;

    int exp;
    float mant = frexpf(val, &exp);
    exp += 14;

    if (exp <= 0) {
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
// BFLOAT16 CONVERSION
// ============================================================================

static inline float bf16_to_float(uint16_t val) {
    uint32_t bits = ((uint32_t)val) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

static inline uint16_t float_to_bf16(float val) {
    uint32_t bits;
    memcpy(&bits, &val, sizeof(float));
    return (uint16_t)(bits >> 16);
}

// ============================================================================
// FLOAT16 CONVERSION (IEEE 754 half precision)
// ============================================================================

static inline float fp16_to_float(uint16_t val) {
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

static inline uint16_t float_to_fp16(float val) {
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

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_TYPES_H
