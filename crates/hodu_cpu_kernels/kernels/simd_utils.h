#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

#include "types.h"
#include <stddef.h>
#include <stdint.h>

// ============================================================================
// SIMD Platform Detection and Configuration
// ============================================================================
//
// This file provides portable SIMD abstractions that work across platforms:
// - x86/x86_64: SSE, AVX, AVX2, AVX-512
// - ARM: NEON
// - WASM: SIMD128
// - Fallback: Auto-vectorization hints for compiler
//
// Usage:
// - Enabled by default when ENABLE_SIMD_AUTO is defined
// - Disabled with DISABLE_SIMD
// - Automatically falls back to scalar code if SIMD unavailable

// Platform detection
#if defined(ENABLE_SIMD_AUTO) && !defined(DISABLE_SIMD)
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define SIMD_X86
#if defined(__AVX512F__)
#define SIMD_AVX512
#include <immintrin.h>
#elif defined(__AVX2__)
#define SIMD_AVX2
#include <immintrin.h>
#elif defined(__AVX__)
#define SIMD_AVX
#include <immintrin.h>
#elif defined(__SSE2__)
#define SIMD_SSE2
#include <emmintrin.h>
#endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
#define SIMD_ARM_NEON
#include <arm_neon.h>
#elif defined(__wasm_simd128__)
#define SIMD_WASM
#include <wasm_simd128.h>
#endif
#endif

// ============================================================================
// F32 SIMD Operations
// ============================================================================

#ifdef SIMD_AVX2
#define SIMD_F32_WIDTH 8
typedef __m256 simd_f32_t;

static inline simd_f32_t simd_f32_load(const float *ptr) { return _mm256_loadu_ps(ptr); }

static inline void simd_f32_store(float *ptr, simd_f32_t v) { _mm256_storeu_ps(ptr, v); }

static inline simd_f32_t simd_f32_add(simd_f32_t a, simd_f32_t b) { return _mm256_add_ps(a, b); }

static inline simd_f32_t simd_f32_sub(simd_f32_t a, simd_f32_t b) { return _mm256_sub_ps(a, b); }

static inline simd_f32_t simd_f32_mul(simd_f32_t a, simd_f32_t b) { return _mm256_mul_ps(a, b); }

static inline simd_f32_t simd_f32_div(simd_f32_t a, simd_f32_t b) { return _mm256_div_ps(a, b); }

static inline simd_f32_t simd_f32_set1(float a) { return _mm256_set1_ps(a); }

#if defined(__FMA__)
static inline simd_f32_t simd_f32_fmadd(simd_f32_t a, simd_f32_t b, simd_f32_t c) {
    return _mm256_fmadd_ps(a, b, c);
}
#else
static inline simd_f32_t simd_f32_fmadd(simd_f32_t a, simd_f32_t b, simd_f32_t c) {
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}
#endif

// Horizontal sum for matmul dot product
static inline float simd_f32_reduce_add(simd_f32_t v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    __m128 shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static inline simd_f32_t simd_f32_max(simd_f32_t a, simd_f32_t b) { return _mm256_max_ps(a, b); }

static inline simd_f32_t simd_f32_min(simd_f32_t a, simd_f32_t b) { return _mm256_min_ps(a, b); }

#elif defined(SIMD_SSE2)
#define SIMD_F32_WIDTH 4
typedef __m128 simd_f32_t;

static inline simd_f32_t simd_f32_load(const float *ptr) { return _mm_loadu_ps(ptr); }

static inline void simd_f32_store(float *ptr, simd_f32_t v) { _mm_storeu_ps(ptr, v); }

static inline simd_f32_t simd_f32_add(simd_f32_t a, simd_f32_t b) { return _mm_add_ps(a, b); }

static inline simd_f32_t simd_f32_sub(simd_f32_t a, simd_f32_t b) { return _mm_sub_ps(a, b); }

static inline simd_f32_t simd_f32_mul(simd_f32_t a, simd_f32_t b) { return _mm_mul_ps(a, b); }

static inline simd_f32_t simd_f32_div(simd_f32_t a, simd_f32_t b) { return _mm_div_ps(a, b); }

static inline simd_f32_t simd_f32_set1(float a) { return _mm_set1_ps(a); }

static inline simd_f32_t simd_f32_fmadd(simd_f32_t a, simd_f32_t b, simd_f32_t c) {
    return _mm_add_ps(_mm_mul_ps(a, b), c);
}

// Horizontal sum for matmul dot product
static inline float simd_f32_reduce_add(simd_f32_t v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static inline simd_f32_t simd_f32_max(simd_f32_t a, simd_f32_t b) { return _mm_max_ps(a, b); }

static inline simd_f32_t simd_f32_min(simd_f32_t a, simd_f32_t b) { return _mm_min_ps(a, b); }

#elif defined(SIMD_ARM_NEON)
#define SIMD_F32_WIDTH 4
typedef float32x4_t simd_f32_t;

static inline simd_f32_t simd_f32_load(const float *ptr) { return vld1q_f32(ptr); }

static inline void simd_f32_store(float *ptr, simd_f32_t v) { vst1q_f32(ptr, v); }

static inline simd_f32_t simd_f32_add(simd_f32_t a, simd_f32_t b) { return vaddq_f32(a, b); }

static inline simd_f32_t simd_f32_sub(simd_f32_t a, simd_f32_t b) { return vsubq_f32(a, b); }

static inline simd_f32_t simd_f32_mul(simd_f32_t a, simd_f32_t b) { return vmulq_f32(a, b); }

static inline simd_f32_t simd_f32_div(simd_f32_t a, simd_f32_t b) {
    // ARM NEON doesn't have direct div, use reciprocal estimate + Newton-Raphson
    float32x4_t recip = vrecpeq_f32(b);
    recip = vmulq_f32(vrecpsq_f32(b, recip), recip);
    return vmulq_f32(a, recip);
}

static inline simd_f32_t simd_f32_set1(float a) { return vdupq_n_f32(a); }

static inline simd_f32_t simd_f32_fmadd(simd_f32_t a, simd_f32_t b, simd_f32_t c) {
    return vfmaq_f32(c, a, b); // c + a * b
}

// Horizontal sum for matmul dot product
static inline float simd_f32_reduce_add(simd_f32_t v) {
    return vaddvq_f32(v); // ARM NEON horizontal add
}

static inline simd_f32_t simd_f32_max(simd_f32_t a, simd_f32_t b) { return vmaxq_f32(a, b); }

static inline simd_f32_t simd_f32_min(simd_f32_t a, simd_f32_t b) { return vminq_f32(a, b); }

#else
#define SIMD_F32_WIDTH 1
// No SIMD available, fall back to scalar (compiler may auto-vectorize)
#endif

// ============================================================================
// F64 SIMD Operations
// ============================================================================

#ifdef SIMD_AVX2
#define SIMD_F64_WIDTH 4
typedef __m256d simd_f64_t;

static inline simd_f64_t simd_f64_load(const double *ptr) { return _mm256_loadu_pd(ptr); }

static inline void simd_f64_store(double *ptr, simd_f64_t v) { _mm256_storeu_pd(ptr, v); }

static inline simd_f64_t simd_f64_add(simd_f64_t a, simd_f64_t b) { return _mm256_add_pd(a, b); }

static inline simd_f64_t simd_f64_sub(simd_f64_t a, simd_f64_t b) { return _mm256_sub_pd(a, b); }

static inline simd_f64_t simd_f64_mul(simd_f64_t a, simd_f64_t b) { return _mm256_mul_pd(a, b); }

static inline simd_f64_t simd_f64_div(simd_f64_t a, simd_f64_t b) { return _mm256_div_pd(a, b); }

static inline simd_f64_t simd_f64_set1(double a) { return _mm256_set1_pd(a); }

#if defined(__FMA__)
static inline simd_f64_t simd_f64_fmadd(simd_f64_t a, simd_f64_t b, simd_f64_t c) {
    return _mm256_fmadd_pd(a, b, c);
}
#else
static inline simd_f64_t simd_f64_fmadd(simd_f64_t a, simd_f64_t b, simd_f64_t c) {
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
}
#endif

// Horizontal sum for matmul dot product
static inline double simd_f64_reduce_add(simd_f64_t v) {
    __m128d low = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1);
    __m128d sum128 = _mm_add_pd(low, high);
    __m128d shuf = _mm_shuffle_pd(sum128, sum128, 1);
    sum128 = _mm_add_sd(sum128, shuf);
    return _mm_cvtsd_f64(sum128);
}

static inline simd_f64_t simd_f64_max(simd_f64_t a, simd_f64_t b) { return _mm256_max_pd(a, b); }

static inline simd_f64_t simd_f64_min(simd_f64_t a, simd_f64_t b) { return _mm256_min_pd(a, b); }

#elif defined(SIMD_SSE2)
#define SIMD_F64_WIDTH 2
typedef __m128d simd_f64_t;

static inline simd_f64_t simd_f64_load(const double *ptr) { return _mm_loadu_pd(ptr); }

static inline void simd_f64_store(double *ptr, simd_f64_t v) { _mm_storeu_pd(ptr, v); }

static inline simd_f64_t simd_f64_add(simd_f64_t a, simd_f64_t b) { return _mm_add_pd(a, b); }

static inline simd_f64_t simd_f64_sub(simd_f64_t a, simd_f64_t b) { return _mm_sub_pd(a, b); }

static inline simd_f64_t simd_f64_mul(simd_f64_t a, simd_f64_t b) { return _mm_mul_pd(a, b); }

static inline simd_f64_t simd_f64_div(simd_f64_t a, simd_f64_t b) { return _mm_div_pd(a, b); }

static inline simd_f64_t simd_f64_set1(double a) { return _mm_set1_pd(a); }

static inline simd_f64_t simd_f64_fmadd(simd_f64_t a, simd_f64_t b, simd_f64_t c) {
    return _mm_add_pd(_mm_mul_pd(a, b), c);
}

// Horizontal sum for matmul dot product
static inline double simd_f64_reduce_add(simd_f64_t v) {
    __m128d shuf = _mm_shuffle_pd(v, v, 1);
    __m128d sum = _mm_add_sd(v, shuf);
    return _mm_cvtsd_f64(sum);
}

static inline simd_f64_t simd_f64_max(simd_f64_t a, simd_f64_t b) { return _mm_max_pd(a, b); }

static inline simd_f64_t simd_f64_min(simd_f64_t a, simd_f64_t b) { return _mm_min_pd(a, b); }

#elif defined(SIMD_ARM_NEON)
#define SIMD_F64_WIDTH 2
typedef float64x2_t simd_f64_t;

static inline simd_f64_t simd_f64_load(const double *ptr) { return vld1q_f64(ptr); }

static inline void simd_f64_store(double *ptr, simd_f64_t v) { vst1q_f64(ptr, v); }

static inline simd_f64_t simd_f64_add(simd_f64_t a, simd_f64_t b) { return vaddq_f64(a, b); }

static inline simd_f64_t simd_f64_sub(simd_f64_t a, simd_f64_t b) { return vsubq_f64(a, b); }

static inline simd_f64_t simd_f64_mul(simd_f64_t a, simd_f64_t b) { return vmulq_f64(a, b); }

static inline simd_f64_t simd_f64_div(simd_f64_t a, simd_f64_t b) { return vdivq_f64(a, b); }

static inline simd_f64_t simd_f64_set1(double a) { return vdupq_n_f64(a); }

static inline simd_f64_t simd_f64_fmadd(simd_f64_t a, simd_f64_t b, simd_f64_t c) {
    return vfmaq_f64(c, a, b); // c + a * b
}

// Horizontal sum for matmul dot product
static inline double simd_f64_reduce_add(simd_f64_t v) { return vaddvq_f64(v); }

static inline simd_f64_t simd_f64_max(simd_f64_t a, simd_f64_t b) { return vmaxq_f64(a, b); }

static inline simd_f64_t simd_f64_min(simd_f64_t a, simd_f64_t b) { return vminq_f64(a, b); }

#else
#define SIMD_F64_WIDTH 1
// No SIMD available
#endif

#endif // SIMD_UTILS_H
