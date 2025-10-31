#include "ops_concat_split.h"
#include "types.h"
#include <stdbool.h>

// ============================================================================
// CONCATENATION OPERATIONS
// ============================================================================

// Generic concatenation macro
// Concatenates multiple input tensors along a specified dimension
#define CONCAT_OP(TYPENAME, FN_NAME)                                                               \
    void FN_NAME(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,         \
                 const size_t *metadata) {                                                         \
        const TYPENAME *input = (const TYPENAME *)input_ptr;                                       \
        TYPENAME *output = (TYPENAME *)output_ptr;                                                 \
                                                                                                   \
        const size_t *output_shape = metadata;                                                     \
        const size_t concat_dim = metadata[num_dims];                                              \
        const size_t num_inputs = metadata[num_dims + 1];                                          \
        const size_t *input_shapes = metadata + num_dims + 2;                                      \
        const size_t *input_strides = input_shapes + num_inputs * num_dims;                        \
        const size_t *input_offsets = input_strides + num_inputs * num_dims;                       \
        const size_t *input_buffer_offsets = input_offsets + num_inputs;                           \
                                                                                                   \
        /* Process each output element */                                                          \
        for (size_t id = 0; id < num_els; id++) {                                                  \
            /* Calculate output indices from flat id */                                            \
            size_t output_indices[16];                                                             \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            /* Determine which input tensor to read from */                                        \
            size_t dim_index = output_indices[concat_dim];                                         \
            size_t tensor_idx = 0;                                                                 \
            size_t cumulative_size = 0;                                                            \
                                                                                                   \
            for (size_t i = 0; i < num_inputs; i++) {                                              \
                size_t dim_size = input_shapes[i * num_dims + concat_dim];                         \
                if (dim_index < cumulative_size + dim_size) {                                      \
                    tensor_idx = i;                                                                \
                    break;                                                                         \
                }                                                                                  \
                cumulative_size += dim_size;                                                       \
            }                                                                                      \
                                                                                                   \
            /* Calculate input index */                                                            \
            const size_t *input_stride = input_strides + tensor_idx * num_dims;                    \
            size_t input_offset = input_offsets[tensor_idx];                                       \
            size_t buffer_offset = input_buffer_offsets[tensor_idx];                               \
                                                                                                   \
            size_t flat_index = input_offset;                                                      \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                size_t idx =                                                                       \
                    (i == concat_dim) ? (dim_index - cumulative_size) : output_indices[i];         \
                flat_index += idx * input_stride[i];                                               \
            }                                                                                      \
                                                                                                   \
            output[id] = input[buffer_offset + flat_index];                                        \
        }                                                                                          \
    }

// Define concat operations for all types
CONCAT_OP(bool, concat_bool)
CONCAT_OP(f8e4m3_t, concat_f8e4m3)
CONCAT_OP(f8e5m2_t, concat_f8e5m2)
CONCAT_OP(bf16_t, concat_bf16)
CONCAT_OP(f16_t, concat_f16)
CONCAT_OP(float, concat_f32)
CONCAT_OP(double, concat_f64)
CONCAT_OP(int8_t, concat_i8)
CONCAT_OP(int16_t, concat_i16)
CONCAT_OP(int32_t, concat_i32)
CONCAT_OP(int64_t, concat_i64)
CONCAT_OP(uint8_t, concat_u8)
CONCAT_OP(uint16_t, concat_u16)
CONCAT_OP(uint32_t, concat_u32)
CONCAT_OP(uint64_t, concat_u64)

// ============================================================================
// SPLIT OPERATIONS
// ============================================================================

// Generic split macro
// Splits one tensor into multiple outputs along a specified dimension
#define SPLIT_OP(TYPENAME, FN_NAME)                                                                \
    void FN_NAME(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,         \
                 const size_t *metadata) {                                                         \
        const TYPENAME *input = (const TYPENAME *)input_ptr;                                       \
        TYPENAME *output = (TYPENAME *)output_ptr;                                                 \
                                                                                                   \
        const size_t *input_shape = metadata;                                                      \
        const size_t *input_strides = metadata + num_dims;                                         \
        const size_t input_offset = metadata[2 * num_dims];                                        \
        const size_t split_dim = metadata[2 * num_dims + 1];                                       \
        const size_t output_size_on_dim = metadata[2 * num_dims + 2];                              \
        const size_t split_offset = metadata[2 * num_dims + 3];                                    \
                                                                                                   \
        /* Calculate output shape */                                                               \
        size_t output_shape[16];                                                                   \
        for (size_t i = 0; i < num_dims; i++) {                                                    \
            output_shape[i] = (i == split_dim) ? output_size_on_dim : input_shape[i];              \
        }                                                                                          \
                                                                                                   \
        /* Process each output element */                                                          \
        for (size_t id = 0; id < num_els; id++) {                                                  \
            /* Calculate output indices from flat id */                                            \
            size_t output_indices[16];                                                             \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            /* Calculate input flat index */                                                       \
            size_t flat_index = input_offset;                                                      \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                size_t idx =                                                                       \
                    (i == split_dim) ? (output_indices[i] + split_offset) : output_indices[i];     \
                flat_index += idx * input_strides[i];                                              \
            }                                                                                      \
                                                                                                   \
            output[id] = input[flat_index];                                                        \
        }                                                                                          \
    }

// Define split operations for all types
SPLIT_OP(bool, split_bool)
SPLIT_OP(f8e4m3_t, split_f8e4m3)
SPLIT_OP(f8e5m2_t, split_f8e5m2)
SPLIT_OP(bf16_t, split_bf16)
SPLIT_OP(f16_t, split_f16)
SPLIT_OP(float, split_f32)
SPLIT_OP(double, split_f64)
SPLIT_OP(int8_t, split_i8)
SPLIT_OP(int16_t, split_i16)
SPLIT_OP(int32_t, split_i32)
SPLIT_OP(int64_t, split_i64)
SPLIT_OP(uint8_t, split_u8)
SPLIT_OP(uint16_t, split_u16)
SPLIT_OP(uint32_t, split_u32)
SPLIT_OP(uint64_t, split_u64)
