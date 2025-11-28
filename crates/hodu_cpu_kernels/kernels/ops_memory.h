/**
 * @file ops_memory.h
 * @brief Memory layout operations header
 *
 * Provides memory layout operations for tensors:
 * - contiguous: Copy tensor data to contiguous memory layout
 *
 * All operations support strided tensor access and multiple data types.
 */

#ifndef HODU_CPU_KERNELS_OPS_MEMORY_H
#define HODU_CPU_KERNELS_OPS_MEMORY_H

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MEMORY OPERATION FUNCTION SIGNATURES
// ============================================================================
//
// Memory operations handle tensor memory layout transformations.
//
// Parameters:
//   input    - Pointer to input tensor data (may be strided)
//   output   - Pointer to output buffer (pre-allocated, contiguous)
//   metadata - Array describing tensor layout (see below)
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: strides
// - metadata[2+2*num_dims]: offset

/**
 * @brief Macro to declare contiguous copy operation
 *
 * Copies tensor data from potentially strided layout to contiguous memory.
 */
#define DECLARE_CONTIGUOUS_OP(TYPE_SUFFIX)                                                         \
    void hodu_cpu_contiguous_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata);

// Declare contiguous operations for all types
DECLARE_CONTIGUOUS_OP(bool)
DECLARE_CONTIGUOUS_OP(f8e4m3)
DECLARE_CONTIGUOUS_OP(f8e5m2)
DECLARE_CONTIGUOUS_OP(bf16)
DECLARE_CONTIGUOUS_OP(f16)
DECLARE_CONTIGUOUS_OP(f32)
DECLARE_CONTIGUOUS_OP(f64)
DECLARE_CONTIGUOUS_OP(u8)
DECLARE_CONTIGUOUS_OP(u16)
DECLARE_CONTIGUOUS_OP(u32)
DECLARE_CONTIGUOUS_OP(u64)
DECLARE_CONTIGUOUS_OP(i8)
DECLARE_CONTIGUOUS_OP(i16)
DECLARE_CONTIGUOUS_OP(i32)
DECLARE_CONTIGUOUS_OP(i64)

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_OPS_MEMORY_H
