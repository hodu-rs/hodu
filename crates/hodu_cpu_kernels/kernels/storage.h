/**
 * @file storage.h
 * @brief Storage operations header
 *
 * Provides storage-level operations for tensors:
 * - const_set: Fill tensor with a constant value
 *
 * All operations support strided tensor access and multiple data types.
 */

#ifndef HODU_CPU_KERNELS_STORAGE_H
#define HODU_CPU_KERNELS_STORAGE_H

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// STORAGE OPERATION FUNCTION SIGNATURES
// ============================================================================
//
// Storage operations handle low-level tensor buffer manipulation.
//
// Parameters:
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing tensor layout (see below)
//   value    - Pointer to constant value to fill with
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: strides
// - metadata[2+2*num_dims]: offset

/**
 * @brief Macro to declare const_set operation
 *
 * Fills tensor with a constant value. Supports strided access.
 */
#define DECLARE_CONST_SET_OP(TYPE_SUFFIX)                                                          \
    void hodu_cpu_const_set_##TYPE_SUFFIX(void *output, const size_t *metadata, const void *value);

// Declare const_set operations for all types
DECLARE_CONST_SET_OP(bool)
DECLARE_CONST_SET_OP(f8e4m3)
DECLARE_CONST_SET_OP(f8e5m2)
DECLARE_CONST_SET_OP(bf16)
DECLARE_CONST_SET_OP(f16)
DECLARE_CONST_SET_OP(f32)
DECLARE_CONST_SET_OP(f64)
DECLARE_CONST_SET_OP(u8)
DECLARE_CONST_SET_OP(u16)
DECLARE_CONST_SET_OP(u32)
DECLARE_CONST_SET_OP(u64)
DECLARE_CONST_SET_OP(i8)
DECLARE_CONST_SET_OP(i16)
DECLARE_CONST_SET_OP(i32)
DECLARE_CONST_SET_OP(i64)

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_STORAGE_H
