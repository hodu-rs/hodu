/**
 * @file ops_binary.h
 * @brief Binary tensor operations header
 *
 * Declares all element-wise binary operations for tensors including:
 * - Arithmetic operations (add, sub, mul, div, pow, maximum, minimum)
 * - Logical operations (logical_and, logical_or, logical_xor)
 * - Comparison operations (eq, ne, lt, le, gt, ge)
 *
 * All operations support multiple data types and handle both contiguous
 * and strided tensor layouts efficiently.
 */

#ifndef HODU_CPU_KERNELS_OPS_BINARY_H
#define HODU_CPU_KERNELS_OPS_BINARY_H

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// BINARY OPERATION FUNCTION SIGNATURES
// ============================================================================
//
// All binary operations follow a consistent signature:
//   void hodu_cpu_op_type(const void *lhs, const void *rhs, void *output, const size_t *metadata)
//
// Parameters:
//   lhs      - Pointer to left-hand side tensor data
//   rhs      - Pointer to right-hand side tensor data
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing tensor layout (see below)
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: lhs_shape
// - metadata[2+num_dims..2+2*num_dims]: rhs_shape
// - metadata[2+2*num_dims..2+3*num_dims]: lhs_strides
// - metadata[2+3*num_dims..2+4*num_dims]: rhs_strides
// - metadata[2+4*num_dims]: lhs_offset
// - metadata[2+4*num_dims+1]: rhs_offset
//
// Note: Broadcasting must be handled by the caller; these functions assume
// compatible shapes and perform element-wise operations only.

/// Macro to declare arithmetic binary operations for a given type
/// Declares: add, sub, mul, div, pow, maximum, minimum
#define DECLARE_BINARY_OP(TYPE_SUFFIX)                                                             \
    void hodu_cpu_add_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                \
                                    const size_t *metadata);                                       \
    void hodu_cpu_sub_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                \
                                    const size_t *metadata);                                       \
    void hodu_cpu_mul_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                \
                                    const size_t *metadata);                                       \
    void hodu_cpu_div_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                \
                                    const size_t *metadata);                                       \
    void hodu_cpu_pow_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                \
                                    const size_t *metadata);                                       \
    void hodu_cpu_maximum_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,            \
                                        const size_t *metadata);                                   \
    void hodu_cpu_minimum_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,            \
                                        const size_t *metadata);

/// Macro to declare logical binary operations for a given type
/// Declares: logical_and, logical_or, logical_xor
/// All return boolean results (uint8_t: 0 for false, 1 for true)
#define DECLARE_BINARY_LOGICAL(TYPE_SUFFIX)                                                        \
    void hodu_cpu_logical_and_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,        \
                                            const size_t *metadata);                               \
    void hodu_cpu_logical_or_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,         \
                                           const size_t *metadata);                                \
    void hodu_cpu_logical_xor_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,        \
                                            const size_t *metadata);

/// Macro to declare comparison binary operations for a given type
/// Declares: eq, ne, lt, le, gt, ge
/// All return boolean results (uint8_t: 0 for false, 1 for true)
#define DECLARE_BINARY_CMP(TYPE_SUFFIX)                                                            \
    void hodu_cpu_eq_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                 \
                                   const size_t *metadata);                                        \
    void hodu_cpu_ne_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                 \
                                   const size_t *metadata);                                        \
    void hodu_cpu_lt_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                 \
                                   const size_t *metadata);                                        \
    void hodu_cpu_le_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                 \
                                   const size_t *metadata);                                        \
    void hodu_cpu_gt_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                 \
                                   const size_t *metadata);                                        \
    void hodu_cpu_ge_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                 \
                                   const size_t *metadata);

// Bool type
DECLARE_BINARY_OP(bool)
DECLARE_BINARY_LOGICAL(bool)
DECLARE_BINARY_CMP(bool)

// Float types (f8e4m3, f8e5m2, bf16, f16, f32, f64)
DECLARE_BINARY_OP(f8e4m3)
DECLARE_BINARY_LOGICAL(f8e4m3)
DECLARE_BINARY_CMP(f8e4m3)

DECLARE_BINARY_OP(f8e5m2)
DECLARE_BINARY_LOGICAL(f8e5m2)
DECLARE_BINARY_CMP(f8e5m2)

DECLARE_BINARY_OP(bf16)
DECLARE_BINARY_LOGICAL(bf16)
DECLARE_BINARY_CMP(bf16)

DECLARE_BINARY_OP(f16)
DECLARE_BINARY_LOGICAL(f16)
DECLARE_BINARY_CMP(f16)

DECLARE_BINARY_OP(f32)
DECLARE_BINARY_LOGICAL(f32)
DECLARE_BINARY_CMP(f32)

DECLARE_BINARY_OP(f64)
DECLARE_BINARY_LOGICAL(f64)
DECLARE_BINARY_CMP(f64)

// Integer types (u8, u16, u32, u64, i8, i16, i32, i64)
DECLARE_BINARY_OP(u8)
DECLARE_BINARY_LOGICAL(u8)
DECLARE_BINARY_CMP(u8)

DECLARE_BINARY_OP(u16)
DECLARE_BINARY_LOGICAL(u16)
DECLARE_BINARY_CMP(u16)

DECLARE_BINARY_OP(u32)
DECLARE_BINARY_LOGICAL(u32)
DECLARE_BINARY_CMP(u32)

DECLARE_BINARY_OP(u64)
DECLARE_BINARY_LOGICAL(u64)
DECLARE_BINARY_CMP(u64)

DECLARE_BINARY_OP(i8)
DECLARE_BINARY_LOGICAL(i8)
DECLARE_BINARY_CMP(i8)

DECLARE_BINARY_OP(i16)
DECLARE_BINARY_LOGICAL(i16)
DECLARE_BINARY_CMP(i16)

DECLARE_BINARY_OP(i32)
DECLARE_BINARY_LOGICAL(i32)
DECLARE_BINARY_CMP(i32)

DECLARE_BINARY_OP(i64)
DECLARE_BINARY_LOGICAL(i64)
DECLARE_BINARY_CMP(i64)

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_OPS_BINARY_H
