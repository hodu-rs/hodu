/**
 * @file ops_indexing.h
 * @brief Tensor indexing operations header
 *
 * Provides advanced indexing operations for tensors:
 * - index_select: Select elements along a dimension
 * - index_put: Write values to specific positions
 * - gather: Gather elements using an indices tensor
 * - scatter: Scatter values to positions (with variants)
 *
 * All operations support negative indexing (Python-style, e.g., -1 = last element).
 */

#ifndef OPS_INDEXING_H
#define OPS_INDEXING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// INDEX SELECT OPERATIONS
// ============================================================================
//
// Select elements along a dimension using a 1D indices array.
//
// All index_select operations follow this signature:
//   void index_select_type(const void *input, const int32_t *indices,
//                          void *output, const size_t *metadata)
//
// Parameters:
//   input    - Pointer to input tensor data
//   indices  - Pointer to int32 indices array (1D)
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing operation (see below)
//
// Metadata layout:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: input_strides
// - metadata[2+2*num_dims]: input_offset
// - metadata[2+2*num_dims+1]: dim (dimension along which to select)
// - metadata[2+2*num_dims+2]: num_indices

void index_select_bool(const void *input, const int32_t *indices, void *output,
                       const size_t *metadata);
void index_select_f8e4m3(const void *input, const int32_t *indices, void *output,
                         const size_t *metadata);
void index_select_f8e5m2(const void *input, const int32_t *indices, void *output,
                         const size_t *metadata);
void index_select_bf16(const void *input, const int32_t *indices, void *output,
                       const size_t *metadata);
void index_select_f16(const void *input, const int32_t *indices, void *output,
                      const size_t *metadata);
void index_select_f32(const void *input, const int32_t *indices, void *output,
                      const size_t *metadata);
void index_select_f64(const void *input, const int32_t *indices, void *output,
                      const size_t *metadata);
void index_select_i8(const void *input, const int32_t *indices, void *output,
                     const size_t *metadata);
void index_select_i16(const void *input, const int32_t *indices, void *output,
                      const size_t *metadata);
void index_select_i32(const void *input, const int32_t *indices, void *output,
                      const size_t *metadata);
void index_select_i64(const void *input, const int32_t *indices, void *output,
                      const size_t *metadata);
void index_select_u8(const void *input, const int32_t *indices, void *output,
                     const size_t *metadata);
void index_select_u16(const void *input, const int32_t *indices, void *output,
                      const size_t *metadata);
void index_select_u32(const void *input, const int32_t *indices, void *output,
                      const size_t *metadata);
void index_select_u64(const void *input, const int32_t *indices, void *output,
                      const size_t *metadata);

// ============================================================================
// INDEX PUT OPERATIONS
// ============================================================================
//
// Write values to positions specified by a 1D indices array.
//
// All index_put operations follow this signature:
//   void index_put_type(const void *input, const int32_t *indices,
//                       const void *values, void *output, const size_t *metadata)
//
// Parameters:
//   input    - Pointer to input tensor data
//   indices  - Pointer to int32 indices array (1D)
//   values   - Pointer to values tensor to write
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing operation (see below)
//
// Metadata layout:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: input_strides
// - metadata[2+2*num_dims..2+3*num_dims]: values_strides
// - metadata[2+3*num_dims]: input_offset
// - metadata[2+3*num_dims+1]: values_offset
// - metadata[2+3*num_dims+2]: dim (dimension along which to write)
// - metadata[2+3*num_dims+3]: num_indices

void index_put_bool(const void *input, const int32_t *indices, const void *values, void *output,
                    const size_t *metadata);
void index_put_f8e4m3(const void *input, const int32_t *indices, const void *values, void *output,
                      const size_t *metadata);
void index_put_f8e5m2(const void *input, const int32_t *indices, const void *values, void *output,
                      const size_t *metadata);
void index_put_bf16(const void *input, const int32_t *indices, const void *values, void *output,
                    const size_t *metadata);
void index_put_f16(const void *input, const int32_t *indices, const void *values, void *output,
                   const size_t *metadata);
void index_put_f32(const void *input, const int32_t *indices, const void *values, void *output,
                   const size_t *metadata);
void index_put_f64(const void *input, const int32_t *indices, const void *values, void *output,
                   const size_t *metadata);
void index_put_i8(const void *input, const int32_t *indices, const void *values, void *output,
                  const size_t *metadata);
void index_put_i16(const void *input, const int32_t *indices, const void *values, void *output,
                   const size_t *metadata);
void index_put_i32(const void *input, const int32_t *indices, const void *values, void *output,
                   const size_t *metadata);
void index_put_i64(const void *input, const int32_t *indices, const void *values, void *output,
                   const size_t *metadata);
void index_put_u8(const void *input, const int32_t *indices, const void *values, void *output,
                  const size_t *metadata);
void index_put_u16(const void *input, const int32_t *indices, const void *values, void *output,
                   const size_t *metadata);
void index_put_u32(const void *input, const int32_t *indices, const void *values, void *output,
                   const size_t *metadata);
void index_put_u64(const void *input, const int32_t *indices, const void *values, void *output,
                   const size_t *metadata);

// ============================================================================
// GATHER OPERATIONS
// ============================================================================
//
// Gather elements using an indices tensor (can be multi-dimensional).
//
// All gather operations follow this signature:
//   void gather_type(const void *input, const int32_t *indices,
//                    void *output, const size_t *metadata)
//
// Parameters:
//   input    - Pointer to input tensor data
//   indices  - Pointer to int32 indices tensor
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing operation (see below)
//
// Metadata layout:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: input_strides
// - metadata[2+2*num_dims..2+3*num_dims]: indices_strides
// - metadata[2+3*num_dims]: input_offset
// - metadata[2+3*num_dims+1]: indices_offset
// - metadata[2+3*num_dims+2]: dim (dimension along which to gather)
// - metadata[2+3*num_dims+3]: num_indices

void gather_bool(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_f8e4m3(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_f8e5m2(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_bf16(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_f16(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_f32(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_f64(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_i8(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_i16(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_i32(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_i64(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_u8(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_u16(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_u32(const void *input, const int32_t *indices, void *output, const size_t *metadata);
void gather_u64(const void *input, const int32_t *indices, void *output, const size_t *metadata);

// ============================================================================
// SCATTER OPERATIONS
// ============================================================================
//
// Scatter src values to positions specified by indices (overwrite mode).
//
// All scatter operations follow this signature:
//   void scatter_type(const void *input, const int32_t *indices,
//                     const void *src, void *output, const size_t *metadata)
//
// Parameters:
//   input    - Pointer to input tensor data (copied to output first)
//   indices  - Pointer to int32 indices tensor
//   src      - Pointer to source values tensor
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing operation (see below)
//
// Metadata layout:
// - metadata[0]: num_els (number of elements in src to scatter)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: input_strides
// - metadata[2+2*num_dims..2+3*num_dims]: src_shape
// - metadata[2+3*num_dims..2+4*num_dims]: src_strides
// - metadata[2+4*num_dims..2+5*num_dims]: indices_strides
// - metadata[2+5*num_dims]: input_offset
// - metadata[2+5*num_dims+1]: src_offset
// - metadata[2+5*num_dims+2]: indices_offset
// - metadata[2+5*num_dims+3]: dim (dimension along which to scatter)

void scatter_bool(const void *input, const int32_t *indices, const void *src, void *output,
                  const size_t *metadata);
void scatter_f8e4m3(const void *input, const int32_t *indices, const void *src, void *output,
                    const size_t *metadata);
void scatter_f8e5m2(const void *input, const int32_t *indices, const void *src, void *output,
                    const size_t *metadata);
void scatter_bf16(const void *input, const int32_t *indices, const void *src, void *output,
                  const size_t *metadata);
void scatter_f16(const void *input, const int32_t *indices, const void *src, void *output,
                 const size_t *metadata);
void scatter_f32(const void *input, const int32_t *indices, const void *src, void *output,
                 const size_t *metadata);
void scatter_f64(const void *input, const int32_t *indices, const void *src, void *output,
                 const size_t *metadata);
void scatter_i8(const void *input, const int32_t *indices, const void *src, void *output,
                const size_t *metadata);
void scatter_i16(const void *input, const int32_t *indices, const void *src, void *output,
                 const size_t *metadata);
void scatter_i32(const void *input, const int32_t *indices, const void *src, void *output,
                 const size_t *metadata);
void scatter_i64(const void *input, const int32_t *indices, const void *src, void *output,
                 const size_t *metadata);
void scatter_u8(const void *input, const int32_t *indices, const void *src, void *output,
                const size_t *metadata);
void scatter_u16(const void *input, const int32_t *indices, const void *src, void *output,
                 const size_t *metadata);
void scatter_u32(const void *input, const int32_t *indices, const void *src, void *output,
                 const size_t *metadata);
void scatter_u64(const void *input, const int32_t *indices, const void *src, void *output,
                 const size_t *metadata);

// ============================================================================
// SCATTER ADD OPERATIONS
// ============================================================================
//
// Scatter src values by accumulating (adding) to existing values.
// Useful when multiple src elements map to the same position.
//
// Metadata layout: Same as scatter operations.
// Only available for numeric types (not bool).

void scatter_add_f8e4m3(const void *input, const int32_t *indices, const void *src, void *output,
                        const size_t *metadata);
void scatter_add_f8e5m2(const void *input, const int32_t *indices, const void *src, void *output,
                        const size_t *metadata);
void scatter_add_bf16(const void *input, const int32_t *indices, const void *src, void *output,
                      const size_t *metadata);
void scatter_add_f16(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_add_f32(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_add_i8(const void *input, const int32_t *indices, const void *src, void *output,
                    const size_t *metadata);
void scatter_add_i16(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_add_i32(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_add_i64(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_add_u8(const void *input, const int32_t *indices, const void *src, void *output,
                    const size_t *metadata);
void scatter_add_u16(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_add_u32(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_add_u64(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);

// ============================================================================
// SCATTER MAX OPERATIONS
// ============================================================================
//
// Scatter src values by taking the maximum with existing values.
//
// Metadata layout: Same as scatter operations.
// Only available for comparable numeric types (not bool).

void scatter_max_f8e4m3(const void *input, const int32_t *indices, const void *src, void *output,
                        const size_t *metadata);
void scatter_max_f8e5m2(const void *input, const int32_t *indices, const void *src, void *output,
                        const size_t *metadata);
void scatter_max_bf16(const void *input, const int32_t *indices, const void *src, void *output,
                      const size_t *metadata);
void scatter_max_f16(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_max_f32(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_max_i8(const void *input, const int32_t *indices, const void *src, void *output,
                    const size_t *metadata);
void scatter_max_i16(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_max_i32(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_max_i64(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_max_u8(const void *input, const int32_t *indices, const void *src, void *output,
                    const size_t *metadata);
void scatter_max_u16(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_max_u32(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_max_u64(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);

// ============================================================================
// SCATTER MIN OPERATIONS
// ============================================================================
//
// Scatter src values by taking the minimum with existing values.
//
// Metadata layout: Same as scatter operations.
// Only available for comparable numeric types (not bool).

void scatter_min_f8e4m3(const void *input, const int32_t *indices, const void *src, void *output,
                        const size_t *metadata);
void scatter_min_f8e5m2(const void *input, const int32_t *indices, const void *src, void *output,
                        const size_t *metadata);
void scatter_min_bf16(const void *input, const int32_t *indices, const void *src, void *output,
                      const size_t *metadata);
void scatter_min_f16(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_min_f32(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_min_i8(const void *input, const int32_t *indices, const void *src, void *output,
                    const size_t *metadata);
void scatter_min_i16(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_min_i32(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_min_i64(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_min_u8(const void *input, const int32_t *indices, const void *src, void *output,
                    const size_t *metadata);
void scatter_min_u16(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_min_u32(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);
void scatter_min_u64(const void *input, const int32_t *indices, const void *src, void *output,
                     const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // OPS_INDEXING_H
