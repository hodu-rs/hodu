#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// Math Constants

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

#ifndef NAN
#define NAN __int_as_float(0x7fc00000)
#endif

// Type Conversion Utilities

template <typename T> __device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template <> __device__ __forceinline__ float to_float(__nv_fp8_e4m3 val) { return (float)val; }

template <> __device__ __forceinline__ float to_float(__nv_fp8_e5m2 val) { return (float)val; }

template <> __device__ __forceinline__ float to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template <> __device__ __forceinline__ float to_float(__half val) { return __half2float(val); }

template <typename T> __device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template <> __device__ __forceinline__ __nv_fp8_e4m3 from_float(float val) {
    return __nv_fp8_e4m3(val);
}

template <> __device__ __forceinline__ __nv_fp8_e5m2 from_float(float val) {
    return __nv_fp8_e5m2(val);
}

template <> __device__ __forceinline__ __nv_bfloat16 from_float(float val) {
    return __nv_bfloat16(val);
}

template <> __device__ __forceinline__ __half from_float(float val) { return __half(val); }

// Comparison Utilities

template <typename T> __device__ __forceinline__ T maximum(T x, T y) { return (x > y) ? x : y; }

template <> __device__ __forceinline__ __nv_fp8_e4m3 maximum(__nv_fp8_e4m3 x, __nv_fp8_e4m3 y) {
    return (to_float(x) > to_float(y)) ? x : y;
}

template <> __device__ __forceinline__ __nv_fp8_e5m2 maximum(__nv_fp8_e5m2 x, __nv_fp8_e5m2 y) {
    return (to_float(x) > to_float(y)) ? x : y;
}

template <> __device__ __forceinline__ __half maximum(__half x, __half y) {
    return (to_float(x) > to_float(y)) ? x : y;
}

template <> __device__ __forceinline__ __nv_bfloat16 maximum(__nv_bfloat16 x, __nv_bfloat16 y) {
    return (to_float(x) > to_float(y)) ? x : y;
}

template <typename T> __device__ __forceinline__ T minimum(T x, T y) { return (x < y) ? x : y; }

template <> __device__ __forceinline__ __nv_fp8_e4m3 minimum(__nv_fp8_e4m3 x, __nv_fp8_e4m3 y) {
    return (to_float(x) < to_float(y)) ? x : y;
}

template <> __device__ __forceinline__ __nv_fp8_e5m2 minimum(__nv_fp8_e5m2 x, __nv_fp8_e5m2 y) {
    return (to_float(x) < to_float(y)) ? x : y;
}

template <> __device__ __forceinline__ __half minimum(__half x, __half y) {
    return (to_float(x) < to_float(y)) ? x : y;
}

template <> __device__ __forceinline__ __nv_bfloat16 minimum(__nv_bfloat16 x, __nv_bfloat16 y) {
    return (to_float(x) < to_float(y)) ? x : y;
}

// Power and Exponential Functions

template <typename T, typename I> __device__ __forceinline__ T ipow(T base, I exp) {
    T result = 1;
    while (exp > 0) {
        if (exp & 1)
            result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

__device__ __forceinline__ float m_pow_float(float base, float exponent) {
    if (exponent == 0.0f)
        return 1.0f;
    if (base == 0.0f)
        return (exponent > 0.0f) ? 0.0f : INFINITY;
    if (base == 1.0f)
        return 1.0f;
    if (exponent == 1.0f)
        return base;

    if (floor(exponent) == exponent) {
        if (exponent >= 0.0f) {
            return ipow(base, (unsigned int)exponent);
        } else {
            return 1.0f / ipow(base, (unsigned int)(-exponent));
        }
    }

    if (base < 0.0f)
        return NAN;
    return powf(base, exponent);
}

// Trigonometric Functions

__device__ __forceinline__ float m_tan(float x) {
    x = fmodf(x, 2 * M_PI);
    if (x > M_PI)
        x -= 2 * M_PI;
    else if (x < -M_PI)
        x += 2 * M_PI;

    float halfPi = M_PI / 2;
    float eps = 1e-6f;

    if (fabsf(fabsf(x) - halfPi) < eps) {
        return x > 0 ? 1e6f : -1e6f;
    }

    return sinf(x) / cosf(x);
}

__device__ __forceinline__ float m_exp10(float x) { return exp10f(x); }

// Zero-checking Utilities

template <typename T> __device__ __forceinline__ bool is_nonzero(T val) { return val != T(0); }

template <> __device__ __forceinline__ bool is_nonzero(__nv_fp8_e4m3 val) {
    return to_float(val) != 0.0f;
}

template <> __device__ __forceinline__ bool is_nonzero(__nv_fp8_e5m2 val) {
    return to_float(val) != 0.0f;
}

template <> __device__ __forceinline__ bool is_nonzero(__nv_bfloat16 val) {
    return to_float(val) != 0.0f;
}

template <> __device__ __forceinline__ bool is_nonzero(__half val) { return to_float(val) != 0.0f; }

// Sign Function

template <typename T> __device__ __forceinline__ T sign(T x) {
    if (x > T(0))
        return T(1);
    if (x < T(0))
        return T(-1);
    return T(0);
}

template <> __device__ __forceinline__ __nv_fp8_e4m3 sign(__nv_fp8_e4m3 x) {
    float fx = to_float(x);
    return from_float<__nv_fp8_e4m3>(fx > 0.0f ? 1.0f : (fx < 0.0f ? -1.0f : 0.0f));
}

template <> __device__ __forceinline__ __nv_fp8_e5m2 sign(__nv_fp8_e5m2 x) {
    float fx = to_float(x);
    return from_float<__nv_fp8_e5m2>(fx > 0.0f ? 1.0f : (fx < 0.0f ? -1.0f : 0.0f));
}

template <> __device__ __forceinline__ __nv_bfloat16 sign(__nv_bfloat16 x) {
    float fx = to_float(x);
    return from_float<__nv_bfloat16>(fx > 0.0f ? 1.0f : (fx < 0.0f ? -1.0f : 0.0f));
}

template <> __device__ __forceinline__ __half sign(__half x) {
    float fx = to_float(x);
    return from_float<__half>(fx > 0.0f ? 1.0f : (fx < 0.0f ? -1.0f : 0.0f));
}

// Activation Functions

template <typename T> __device__ __forceinline__ T relu(T x) { return maximum(x, T(0)); }

// Specializations for types that don't have implicit int conversion
template <> __device__ __forceinline__ __nv_bfloat16 relu(__nv_bfloat16 x) {
    return maximum(x, __nv_bfloat16(0.0f));
}

template <> __device__ __forceinline__ __half relu(__half x) { return maximum(x, __half(0.0f)); }

template <typename T> __device__ __forceinline__ T sigmoid(T x) {
    float fx = to_float(x);
    return from_float<T>(1.0f / (1.0f + expf(-fx)));
}

template <typename T> __device__ __forceinline__ T hardsigmoid(T x) {
    float fx = to_float(x);
    return from_float<T>(fmaxf(0.0f, fminf(1.0f, (fx + 3.0f) / 6.0f)));
}

template <typename T> __device__ __forceinline__ T gelu(T x) {
    float fx = to_float(x);
    float result =
        0.5f * fx * (1.0f + tanhf(0.7978845608028654f * (fx + 0.044715f * fx * fx * fx)));
    return from_float<T>(result);
}

template <typename T> __device__ __forceinline__ T softplus(T x) {
    float fx = to_float(x);
    return from_float<T>(logf(1.0f + expf(fx)));
}

template <typename T> __device__ __forceinline__ T silu(T x) {
    float fx = to_float(x);
    return from_float<T>(fx / (1.0f + expf(-fx)));
}

template <typename T> __device__ __forceinline__ T hardsilu(T x) {
    float fx = to_float(x);
    return from_float<T>(fx * fmaxf(0.0f, fminf(1.0f, (fx + 3.0f) / 6.0f)));
}

template <typename T> __device__ __forceinline__ T mish(T x) {
    float fx = to_float(x);
    return from_float<T>(fx * tanhf(logf(1.0f + expf(fx))));
}
