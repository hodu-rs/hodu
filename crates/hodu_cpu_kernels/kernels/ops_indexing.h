#ifndef OPS_INDEXING_H
#define OPS_INDEXING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Indexing operations for tensors
// Supports index_select, index_put, gather, scatter, scatter_add, scatter_max, scatter_min

// ============================================================================
// INDEX SELECT OPERATIONS
// ============================================================================
// Select elements along a specified dimension using a 1D indices array
// Metadata layout: input_shape, input_strides, input_offset, dim, num_indices

void index_select_bool(const void *input, const int32_t *indices, void *output, size_t num_els,
                       size_t num_dims, const size_t *metadata);
void index_select_f8e4m3(const void *input, const int32_t *indices, void *output, size_t num_els,
                         size_t num_dims, const size_t *metadata);
void index_select_f8e5m2(const void *input, const int32_t *indices, void *output, size_t num_els,
                         size_t num_dims, const size_t *metadata);
void index_select_bf16(const void *input, const int32_t *indices, void *output, size_t num_els,
                       size_t num_dims, const size_t *metadata);
void index_select_f16(const void *input, const int32_t *indices, void *output, size_t num_els,
                      size_t num_dims, const size_t *metadata);
void index_select_f32(const void *input, const int32_t *indices, void *output, size_t num_els,
                      size_t num_dims, const size_t *metadata);
void index_select_f64(const void *input, const int32_t *indices, void *output, size_t num_els,
                      size_t num_dims, const size_t *metadata);
void index_select_i8(const void *input, const int32_t *indices, void *output, size_t num_els,
                     size_t num_dims, const size_t *metadata);
void index_select_i16(const void *input, const int32_t *indices, void *output, size_t num_els,
                      size_t num_dims, const size_t *metadata);
void index_select_i32(const void *input, const int32_t *indices, void *output, size_t num_els,
                      size_t num_dims, const size_t *metadata);
void index_select_i64(const void *input, const int32_t *indices, void *output, size_t num_els,
                      size_t num_dims, const size_t *metadata);
void index_select_u8(const void *input, const int32_t *indices, void *output, size_t num_els,
                     size_t num_dims, const size_t *metadata);
void index_select_u16(const void *input, const int32_t *indices, void *output, size_t num_els,
                      size_t num_dims, const size_t *metadata);
void index_select_u32(const void *input, const int32_t *indices, void *output, size_t num_els,
                      size_t num_dims, const size_t *metadata);
void index_select_u64(const void *input, const int32_t *indices, void *output, size_t num_els,
                      size_t num_dims, const size_t *metadata);

// ============================================================================
// INDEX PUT OPERATIONS
// ============================================================================
// Write values to positions specified by a 1D indices array
// Metadata layout: input_shape, input_strides, values_strides, input_offset, values_offset, dim,
// num_indices

void index_put_bool(const void *input, const int32_t *indices, const void *values, void *output,
                    size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_f8e4m3(const void *input, const int32_t *indices, const void *values, void *output,
                      size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_f8e5m2(const void *input, const int32_t *indices, const void *values, void *output,
                      size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_bf16(const void *input, const int32_t *indices, const void *values, void *output,
                    size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_f16(const void *input, const int32_t *indices, const void *values, void *output,
                   size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_f32(const void *input, const int32_t *indices, const void *values, void *output,
                   size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_f64(const void *input, const int32_t *indices, const void *values, void *output,
                   size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_i8(const void *input, const int32_t *indices, const void *values, void *output,
                  size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_i16(const void *input, const int32_t *indices, const void *values, void *output,
                   size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_i32(const void *input, const int32_t *indices, const void *values, void *output,
                   size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_i64(const void *input, const int32_t *indices, const void *values, void *output,
                   size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_u8(const void *input, const int32_t *indices, const void *values, void *output,
                  size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_u16(const void *input, const int32_t *indices, const void *values, void *output,
                   size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_u32(const void *input, const int32_t *indices, const void *values, void *output,
                   size_t num_els, size_t num_dims, const size_t *metadata);
void index_put_u64(const void *input, const int32_t *indices, const void *values, void *output,
                   size_t num_els, size_t num_dims, const size_t *metadata);

// ============================================================================
// GATHER OPERATIONS
// ============================================================================
// Gather elements using an indices tensor
// Metadata layout: input_shape, input_strides, indices_strides, input_offset, indices_offset, dim

void gather_bool(const void *input, const int32_t *indices, void *output, size_t num_els,
                 size_t num_dims, const size_t *metadata);
void gather_f8e4m3(const void *input, const int32_t *indices, void *output, size_t num_els,
                   size_t num_dims, const size_t *metadata);
void gather_f8e5m2(const void *input, const int32_t *indices, void *output, size_t num_els,
                   size_t num_dims, const size_t *metadata);
void gather_bf16(const void *input, const int32_t *indices, void *output, size_t num_els,
                 size_t num_dims, const size_t *metadata);
void gather_f16(const void *input, const int32_t *indices, void *output, size_t num_els,
                size_t num_dims, const size_t *metadata);
void gather_f32(const void *input, const int32_t *indices, void *output, size_t num_els,
                size_t num_dims, const size_t *metadata);
void gather_f64(const void *input, const int32_t *indices, void *output, size_t num_els,
                size_t num_dims, const size_t *metadata);
void gather_i8(const void *input, const int32_t *indices, void *output, size_t num_els,
               size_t num_dims, const size_t *metadata);
void gather_i16(const void *input, const int32_t *indices, void *output, size_t num_els,
                size_t num_dims, const size_t *metadata);
void gather_i32(const void *input, const int32_t *indices, void *output, size_t num_els,
                size_t num_dims, const size_t *metadata);
void gather_i64(const void *input, const int32_t *indices, void *output, size_t num_els,
                size_t num_dims, const size_t *metadata);
void gather_u8(const void *input, const int32_t *indices, void *output, size_t num_els,
               size_t num_dims, const size_t *metadata);
void gather_u16(const void *input, const int32_t *indices, void *output, size_t num_els,
                size_t num_dims, const size_t *metadata);
void gather_u32(const void *input, const int32_t *indices, void *output, size_t num_els,
                size_t num_dims, const size_t *metadata);
void gather_u64(const void *input, const int32_t *indices, void *output, size_t num_els,
                size_t num_dims, const size_t *metadata);

// ============================================================================
// SCATTER OPERATIONS
// ============================================================================
// Scatter src values to input at positions specified by indices
// Metadata layout: input_shape, input_strides, src_shape, src_strides, indices_strides,
//                  input_offset, src_offset, indices_offset, dim

void scatter_bool(const void *input, const int32_t *indices, const void *src, void *output,
                  size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_f8e4m3(const void *input, const int32_t *indices, const void *src, void *output,
                    size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_f8e5m2(const void *input, const int32_t *indices, const void *src, void *output,
                    size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_bf16(const void *input, const int32_t *indices, const void *src, void *output,
                  size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_f16(const void *input, const int32_t *indices, const void *src, void *output,
                 size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_f32(const void *input, const int32_t *indices, const void *src, void *output,
                 size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_f64(const void *input, const int32_t *indices, const void *src, void *output,
                 size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_i8(const void *input, const int32_t *indices, const void *src, void *output,
                size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_i16(const void *input, const int32_t *indices, const void *src, void *output,
                 size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_i32(const void *input, const int32_t *indices, const void *src, void *output,
                 size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_i64(const void *input, const int32_t *indices, const void *src, void *output,
                 size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_u8(const void *input, const int32_t *indices, const void *src, void *output,
                size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_u16(const void *input, const int32_t *indices, const void *src, void *output,
                 size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_u32(const void *input, const int32_t *indices, const void *src, void *output,
                 size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_u64(const void *input, const int32_t *indices, const void *src, void *output,
                 size_t num_els, size_t num_dims, const size_t *metadata);

// ============================================================================
// SCATTER ADD OPERATIONS
// ============================================================================
// Scatter add: accumulate src values (numeric types only)

void scatter_add_f8e4m3(const void *input, const int32_t *indices, const void *src, void *output,
                        size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_add_f8e5m2(const void *input, const int32_t *indices, const void *src, void *output,
                        size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_add_bf16(const void *input, const int32_t *indices, const void *src, void *output,
                      size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_add_f16(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_add_f32(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_add_i8(const void *input, const int32_t *indices, const void *src, void *output,
                    size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_add_i16(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_add_i32(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_add_i64(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_add_u8(const void *input, const int32_t *indices, const void *src, void *output,
                    size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_add_u16(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_add_u32(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_add_u64(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);

// ============================================================================
// SCATTER MAX OPERATIONS
// ============================================================================
// Scatter max: take maximum (comparable types only)

void scatter_max_f8e4m3(const void *input, const int32_t *indices, const void *src, void *output,
                        size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_max_f8e5m2(const void *input, const int32_t *indices, const void *src, void *output,
                        size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_max_bf16(const void *input, const int32_t *indices, const void *src, void *output,
                      size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_max_f16(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_max_f32(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_max_i8(const void *input, const int32_t *indices, const void *src, void *output,
                    size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_max_i16(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_max_i32(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_max_i64(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_max_u8(const void *input, const int32_t *indices, const void *src, void *output,
                    size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_max_u16(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_max_u32(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_max_u64(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);

// ============================================================================
// SCATTER MIN OPERATIONS
// ============================================================================
// Scatter min: take minimum (comparable types only)

void scatter_min_f8e4m3(const void *input, const int32_t *indices, const void *src, void *output,
                        size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_min_f8e5m2(const void *input, const int32_t *indices, const void *src, void *output,
                        size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_min_bf16(const void *input, const int32_t *indices, const void *src, void *output,
                      size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_min_f16(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_min_f32(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_min_i8(const void *input, const int32_t *indices, const void *src, void *output,
                    size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_min_i16(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_min_i32(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_min_i64(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_min_u8(const void *input, const int32_t *indices, const void *src, void *output,
                    size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_min_u16(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_min_u32(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);
void scatter_min_u64(const void *input, const int32_t *indices, const void *src, void *output,
                     size_t num_els, size_t num_dims, const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // OPS_INDEXING_H
