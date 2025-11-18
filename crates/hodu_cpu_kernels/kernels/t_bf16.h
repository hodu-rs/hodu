#ifndef HODU_CPU_KERNELS_T_BF16_H
#define HODU_CPU_KERNELS_T_BF16_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// BFloat16: 1 sign bit, 8 exponent bits, 7 mantissa bits
typedef uint16_t bf16_t;

// ============================================================================
// CONSTANTS
// ============================================================================

#define BF16_ZERO ((bf16_t)0x0000)
#define BF16_ONE ((bf16_t)0x3F80)
#define BF16_NEG_INF ((bf16_t)0xFF80)
#define BF16_POS_INF ((bf16_t)0x7F80)

// ============================================================================
// CONVERSION
// ============================================================================

static inline float bf16_to_float(bf16_t val) {
    uint32_t bits = ((uint32_t)val) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

static inline bf16_t float_to_bf16(float val) {
    uint32_t bits;
    memcpy(&bits, &val, sizeof(float));
    return (bf16_t)(bits >> 16);
}

// ============================================================================
// ARITHMETIC OPERATIONS
// ============================================================================

static inline bf16_t bf16_add(bf16_t a, bf16_t b) {
    return float_to_bf16(bf16_to_float(a) + bf16_to_float(b));
}

static inline bf16_t bf16_sub(bf16_t a, bf16_t b) {
    return float_to_bf16(bf16_to_float(a) - bf16_to_float(b));
}

static inline bf16_t bf16_mul(bf16_t a, bf16_t b) {
    return float_to_bf16(bf16_to_float(a) * bf16_to_float(b));
}

static inline bf16_t bf16_div(bf16_t a, bf16_t b) {
    return float_to_bf16(bf16_to_float(a) / bf16_to_float(b));
}

static inline bf16_t bf16_neg(bf16_t a) { return float_to_bf16(-bf16_to_float(a)); }

// ============================================================================
// COMPARISON
// ============================================================================

static inline int bf16_lt(bf16_t a, bf16_t b) { return bf16_to_float(a) < bf16_to_float(b); }

static inline int bf16_le(bf16_t a, bf16_t b) { return bf16_to_float(a) <= bf16_to_float(b); }

static inline int bf16_gt(bf16_t a, bf16_t b) { return bf16_to_float(a) > bf16_to_float(b); }

static inline int bf16_ge(bf16_t a, bf16_t b) { return bf16_to_float(a) >= bf16_to_float(b); }

static inline int bf16_eq(bf16_t a, bf16_t b) { return bf16_to_float(a) == bf16_to_float(b); }

// ============================================================================
// MATH FUNCTIONS
// ============================================================================

static inline bf16_t bf16_max(bf16_t a, bf16_t b) {
    float a_f = bf16_to_float(a);
    float b_f = bf16_to_float(b);
    return float_to_bf16(a_f > b_f ? a_f : b_f);
}

static inline bf16_t bf16_min(bf16_t a, bf16_t b) {
    float a_f = bf16_to_float(a);
    float b_f = bf16_to_float(b);
    return float_to_bf16(a_f < b_f ? a_f : b_f);
}

static inline bf16_t bf16_abs(bf16_t a) { return float_to_bf16(fabsf(bf16_to_float(a))); }

static inline bf16_t bf16_sqrt(bf16_t a) { return float_to_bf16(sqrtf(bf16_to_float(a))); }

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_T_BF16_H
