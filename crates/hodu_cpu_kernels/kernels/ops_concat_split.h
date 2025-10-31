/**
 * @file ops_concat_split.h
 * @brief Tensor concatenation and split operations header
 *
 * Provides operations to:
 * - Concatenate multiple tensors along a dimension
 * - Split a tensor into multiple outputs along a dimension
 *
 * These are inverse operations commonly used in neural networks for
 * combining or dividing feature maps, attention heads, etc.
 */

#ifndef OPS_CONCAT_SPLIT_H
#define OPS_CONCAT_SPLIT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONCATENATION OPERATIONS
// ============================================================================
//
// Concatenates multiple input tensors along a specified dimension.
//
// All concatenation operations follow this signature:
//   void concat_type(const void *input, void *output, const size_t *metadata)
//
// Parameters:
//   input    - Pointer to buffer containing all input tensors
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing concatenation (see below)
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements in output)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: output_shape (shape of concatenated output)
// - metadata[2+num_dims]: concat_dim (dimension along which to concatenate)
// - metadata[2+num_dims+1]: num_inputs (number of input tensors)
// - metadata[2+num_dims+2..2+num_dims+2+num_inputs*num_dims]: input_shapes (flattened)
// - metadata[...+num_inputs*num_dims]: input_strides (flattened)
// - metadata[...+num_inputs]: input_offsets (offset within each input tensor)
// - metadata[...+num_inputs]: input_buffer_offsets (element offset of each input in input buffer)

void concat_bool(const void *input, void *output, const size_t *metadata);
void concat_f8e4m3(const void *input, void *output, const size_t *metadata);
void concat_f8e5m2(const void *input, void *output, const size_t *metadata);
void concat_bf16(const void *input, void *output, const size_t *metadata);
void concat_f16(const void *input, void *output, const size_t *metadata);
void concat_f32(const void *input, void *output, const size_t *metadata);
void concat_f64(const void *input, void *output, const size_t *metadata);
void concat_i8(const void *input, void *output, const size_t *metadata);
void concat_i16(const void *input, void *output, const size_t *metadata);
void concat_i32(const void *input, void *output, const size_t *metadata);
void concat_i64(const void *input, void *output, const size_t *metadata);
void concat_u8(const void *input, void *output, const size_t *metadata);
void concat_u16(const void *input, void *output, const size_t *metadata);
void concat_u32(const void *input, void *output, const size_t *metadata);
void concat_u64(const void *input, void *output, const size_t *metadata);

// ============================================================================
// SPLIT OPERATIONS
// ============================================================================
//
// Extracts a slice from an input tensor along a specified dimension.
//
// All split operations follow this signature:
//   void split_type(const void *input, void *output, const size_t *metadata)
//
// Parameters:
//   input    - Pointer to input tensor data
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing split operation (see below)
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements in output)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape (shape of input tensor)
// - metadata[2+num_dims..2+2*num_dims]: input_strides (strides of input tensor)
// - metadata[2+2*num_dims]: input_offset (starting offset in input tensor)
// - metadata[2+2*num_dims+1]: split_dim (dimension along which to split)
// - metadata[2+2*num_dims+2]: output_size_on_dim (size of output along split dimension)
// - metadata[2+2*num_dims+3]: split_offset (offset along split dimension where output starts)

void split_bool(const void *input, void *output, const size_t *metadata);
void split_f8e4m3(const void *input, void *output, const size_t *metadata);
void split_f8e5m2(const void *input, void *output, const size_t *metadata);
void split_bf16(const void *input, void *output, const size_t *metadata);
void split_f16(const void *input, void *output, const size_t *metadata);
void split_f32(const void *input, void *output, const size_t *metadata);
void split_f64(const void *input, void *output, const size_t *metadata);
void split_i8(const void *input, void *output, const size_t *metadata);
void split_i16(const void *input, void *output, const size_t *metadata);
void split_i32(const void *input, void *output, const size_t *metadata);
void split_i64(const void *input, void *output, const size_t *metadata);
void split_u8(const void *input, void *output, const size_t *metadata);
void split_u16(const void *input, void *output, const size_t *metadata);
void split_u32(const void *input, void *output, const size_t *metadata);
void split_u64(const void *input, void *output, const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // OPS_CONCAT_SPLIT_H
