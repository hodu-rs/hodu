/**
 * @file ops_cast.h
 * @brief Type casting operations header
 *
 * Provides element-wise type conversion operations for tensors:
 * - to_dtype: Convert tensor elements from one data type to another
 *
 * All operations support strided tensor access and multiple data types.
 */

#ifndef HODU_CPU_KERNELS_OPS_CAST_H
#define HODU_CPU_KERNELS_OPS_CAST_H

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CAST OPERATION FUNCTION SIGNATURES
// ============================================================================
//
// Cast operations convert tensor elements from one type to another.
//
// Parameters:
//   input    - Pointer to input tensor data (may be strided)
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing tensor layout (see below)
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: strides
// - metadata[2+2*num_dims]: offset

/**
 * @brief Macro to declare cast operations from one type to all other types
 */
#define DECLARE_CAST_FROM(FROM_SUFFIX)                                                             \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_bool(const void *input, void *output,                    \
                                               const size_t *metadata);                            \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_f8e4m3(const void *input, void *output,                  \
                                                 const size_t *metadata);                          \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_f8e5m2(const void *input, void *output,                  \
                                                 const size_t *metadata);                          \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_bf16(const void *input, void *output,                    \
                                               const size_t *metadata);                            \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_f16(const void *input, void *output,                     \
                                              const size_t *metadata);                             \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_f32(const void *input, void *output,                     \
                                              const size_t *metadata);                             \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_f64(const void *input, void *output,                     \
                                              const size_t *metadata);                             \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_u8(const void *input, void *output,                      \
                                             const size_t *metadata);                              \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_u16(const void *input, void *output,                     \
                                              const size_t *metadata);                             \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_u32(const void *input, void *output,                     \
                                              const size_t *metadata);                             \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_u64(const void *input, void *output,                     \
                                              const size_t *metadata);                             \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_i8(const void *input, void *output,                      \
                                             const size_t *metadata);                              \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_i16(const void *input, void *output,                     \
                                              const size_t *metadata);                             \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_i32(const void *input, void *output,                     \
                                              const size_t *metadata);                             \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_i64(const void *input, void *output,                     \
                                              const size_t *metadata);

// Declare cast operations for all source types
DECLARE_CAST_FROM(bool)
DECLARE_CAST_FROM(f8e4m3)
DECLARE_CAST_FROM(f8e5m2)
DECLARE_CAST_FROM(bf16)
DECLARE_CAST_FROM(f16)
DECLARE_CAST_FROM(f32)
DECLARE_CAST_FROM(f64)
DECLARE_CAST_FROM(u8)
DECLARE_CAST_FROM(u16)
DECLARE_CAST_FROM(u32)
DECLARE_CAST_FROM(u64)
DECLARE_CAST_FROM(i8)
DECLARE_CAST_FROM(i16)
DECLARE_CAST_FROM(i32)
DECLARE_CAST_FROM(i64)

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_OPS_CAST_H
