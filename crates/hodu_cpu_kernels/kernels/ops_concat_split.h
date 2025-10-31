#ifndef OPS_CONCAT_SPLIT_H
#define OPS_CONCAT_SPLIT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Concatenation and split operations for tensors
// These operations combine or split tensors along a specified dimension

// Concatenation operations
// Metadata layout:
// - output_shape: output_shape[0..num_dims]
// - concat_dim: the dimension along which to concatenate
// - num_inputs: number of input tensors
// - input_shapes: flattened array of input shapes (num_inputs * num_dims)
// - input_strides: flattened array of input strides (num_inputs * num_dims)
// - input_offsets: array of input offsets (num_inputs)
// - input_buffer_offsets: offset of each input in the input buffer (num_inputs)

void concat_bool(const void *input, void *output, size_t num_els, size_t num_dims,
                 const size_t *metadata);
void concat_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                   const size_t *metadata);
void concat_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                   const size_t *metadata);
void concat_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                 const size_t *metadata);
void concat_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void concat_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void concat_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void concat_i8(const void *input, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);
void concat_i16(const void *input, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void concat_i32(const void *input, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void concat_i64(const void *input, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void concat_u8(const void *input, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);
void concat_u16(const void *input, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void concat_u32(const void *input, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void concat_u64(const void *input, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);

// Split operations
// Metadata layout:
// - input_shape: input_shape[0..num_dims]
// - input_strides: input_strides[0..num_dims]
// - input_offset: scalar offset
// - split_dim: the dimension along which to split
// - output_size_on_dim: size of output along split dimension
// - split_offset: cumulative offset along split dimension for this output

void split_bool(const void *input, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void split_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                  const size_t *metadata);
void split_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                  const size_t *metadata);
void split_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void split_f16(const void *input, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);
void split_f32(const void *input, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);
void split_f64(const void *input, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);
void split_i8(const void *input, void *output, size_t num_els, size_t num_dims,
              const size_t *metadata);
void split_i16(const void *input, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);
void split_i32(const void *input, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);
void split_i64(const void *input, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);
void split_u8(const void *input, void *output, size_t num_els, size_t num_dims,
              const size_t *metadata);
void split_u16(const void *input, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);
void split_u32(const void *input, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);
void split_u64(const void *input, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // OPS_CONCAT_SPLIT_H
