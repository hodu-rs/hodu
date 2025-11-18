#ifndef HODU_CPU_KERNELS_TYPES_H
#define HODU_CPU_KERNELS_TYPES_H

#include <math.h>
#include <stdint.h>
#include <string.h>

// Include specialized type headers
#include "t_bf16.h"
#include "t_f16.h"
#include "t_f8e4m3.h"
#include "t_f8e5m2.h"

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

// Note: Exotic floating-point types are now defined in their respective headers:
// - bf16_t in t_bf16.h (with bf16_to_float, float_to_bf16, bf16_add, etc.)
// - f16_t in t_f16.h (with fp16_to_float, float_to_fp16, f16_add, etc.)
// - f8e4m3_t in t_f8e4m3.h (with f8e4m3_to_float, float_to_f8e4m3, f8e4m3_add, etc.)
// - f8e5m2_t in t_f8e5m2.h (with f8e5m2_to_float, float_to_f8e5m2, f8e5m2_add, etc.)

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_TYPES_H
