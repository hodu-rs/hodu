#include "./headers/utils.metal"
#include <metal_stdlib>

using namespace metal;

// Concatenation and Split Operations
// ====================================
// These operations combine or split tensors along a specified dimension.

// ============================================================================
// CONCATENATION OPERATIONS
// ============================================================================

// Concatenates multiple input tensors along a specified dimension.
// All input tensors must have the same shape except on the concatenation dimension.
//
// Metadata Layout (Total: 2 + num_dims + 2 + num_inputs * (2 * num_dims + 2)):
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: output_shape (shape of concatenated output)
// - metadata[2+num_dims]: concat_dim (dimension along which to concatenate)
// - metadata[2+num_dims+1]: num_inputs (number of input tensors)
// - metadata[2+num_dims+2..2+num_dims+2+num_inputs*num_dims]: input_shapes (flattened)
// - metadata[2+num_dims+2+num_inputs*num_dims..2+num_dims+2+2*num_inputs*num_dims]: input_strides
// (flattened)
// - metadata[2+num_dims+2+2*num_inputs*num_dims..2+num_dims+2+2*num_inputs*num_dims+num_inputs]:
// input_offsets
// - metadata[2+num_dims+2+2*num_inputs*num_dims+num_inputs..]: input_buffer_offsets
//
// Buffer Layout:
// - buffer(0): input tensor (const device T*) - combined buffer with all inputs
// - buffer(1): output tensor (device T*)
// - buffer(2): metadata (constant size_t*)

#define CONCAT_OP(TYPENAME, FN_NAME)                                                               \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *output_shape = metadata + 2;                                        \
        const size_t concat_dim = metadata[2 + num_dims];                                          \
        const size_t num_inputs = metadata[2 + num_dims + 1];                                      \
        const constant size_t *input_shapes = metadata + 2 + num_dims + 2;                         \
        const constant size_t *input_strides = input_shapes + num_inputs * num_dims;               \
        const constant size_t *input_offsets = input_strides + num_inputs * num_dims;              \
        const constant size_t *input_buffer_offsets = input_offsets + num_inputs;                  \
                                                                                                   \
        /* Grid-stride loop for better GPU utilization */                                          \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
                                                                                                   \
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
            const constant size_t *input_stride = input_strides + tensor_idx * num_dims;           \
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
CONCAT_OP(bfloat, concat_bf16)
CONCAT_OP(half, concat_f16)
CONCAT_OP(float, concat_f32)
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

// Extracts a portion of a tensor along a specified dimension.
// The output is a slice of the input tensor starting at split_offset with size output_size_on_dim.
//
// Metadata Layout (Total: 2 + num_dims * 2 + 4):
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape (shape of input tensor)
// - metadata[2+num_dims..2+2*num_dims]: strides (stride for each dimension)
// - metadata[2+2*num_dims]: offset (starting offset in input buffer)
// - metadata[2+2*num_dims+1]: split_dim (dimension along which to split)
// - metadata[2+2*num_dims+2]: output_size_on_dim (size of output on split dimension)
// - metadata[2+2*num_dims+3]: split_offset (offset on split dimension where extraction starts)
//
// Buffer Layout:
// - buffer(0): input tensor (const device T*)
// - buffer(1): output tensor (device T*)
// - buffer(2): metadata (constant size_t*)

#define SPLIT_OP(TYPENAME, FN_NAME)                                                                \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *input_shape = metadata + 2;                                         \
        const constant size_t *input_strides = metadata + 2 + num_dims;                            \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
        const size_t split_dim = metadata[2 + 2 * num_dims + 1];                                   \
        const size_t output_size_on_dim = metadata[2 + 2 * num_dims + 2];                          \
        const size_t split_offset = metadata[2 + 2 * num_dims + 3];                                \
                                                                                                   \
        /* Grid-stride loop for better GPU utilization */                                          \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
                                                                                                   \
            /* Calculate output shape */                                                           \
            size_t output_shape[16];                                                               \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                output_shape[i] = (i == split_dim) ? output_size_on_dim : input_shape[i];          \
            }                                                                                      \
                                                                                                   \
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
SPLIT_OP(bfloat, split_bf16)
SPLIT_OP(half, split_f16)
SPLIT_OP(float, split_f32)
SPLIT_OP(int8_t, split_i8)
SPLIT_OP(int16_t, split_i16)
SPLIT_OP(int32_t, split_i32)
SPLIT_OP(int64_t, split_i64)
SPLIT_OP(uint8_t, split_u8)
SPLIT_OP(uint16_t, split_u16)
SPLIT_OP(uint32_t, split_u32)
SPLIT_OP(uint64_t, split_u64)
