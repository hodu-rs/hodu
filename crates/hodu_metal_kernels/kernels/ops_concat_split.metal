#include "./headers/utils.metal"
#include <metal_stdlib>

using namespace metal;

// Concatenation and split operations for tensors
// These operations combine or split tensors along a specified dimension

// ============================================================================
// CONCATENATION OPERATIONS
// ============================================================================

// Generic concatenation kernel that works with multiple input tensors
// All inputs are concatenated into a single input buffer
// Metadata layout:
// - output_shape: output_shape[0..num_dims]
// - concat_dim: the dimension along which to concatenate
// - num_inputs: number of input tensors
// - input_shapes: flattened array of input shapes (num_inputs * num_dims)
// - input_strides: flattened array of input strides (num_inputs * num_dims)
// - input_offsets: array of input offsets (num_inputs)
// - input_buffer_offsets: offset of each input in the input buffer (num_inputs)

#define CONCAT_OP(TYPENAME, FN_NAME)                                                               \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t &num_els [[buffer(2)]], constant size_t &num_dims [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        /* Grid-stride loop for better GPU utilization */                                          \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
                                                                                                   \
            const constant size_t *output_shape = metadata;                                        \
            const size_t concat_dim = metadata[num_dims];                                          \
            const size_t num_inputs = metadata[num_dims + 1];                                      \
            const constant size_t *input_shapes = metadata + num_dims + 2;                         \
            const constant size_t *input_strides = input_shapes + num_inputs * num_dims;           \
            const constant size_t *input_offsets = input_strides + num_inputs * num_dims;          \
            const constant size_t *input_buffer_offsets = input_offsets + num_inputs;              \
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

// Generic split kernel that splits one tensor into multiple outputs
// Metadata layout:
// - input_shape: input_shape[0..num_dims]
// - input_strides: input_strides[0..num_dims]
// - input_offset: scalar offset
// - split_dim: the dimension along which to split
// - num_outputs: number of output tensors
// - output_sizes: sizes of each output along split dimension (num_outputs)
// - split_index: which split this kernel is computing
// - cumulative_offset: cumulative offset along split dimension for this output

#define SPLIT_OP(TYPENAME, FN_NAME)                                                                \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t &num_els [[buffer(2)]], constant size_t &num_dims [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        /* Grid-stride loop for better GPU utilization */                                          \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
                                                                                                   \
            const constant size_t *input_shape = metadata;                                         \
            const constant size_t *input_strides = metadata + num_dims;                            \
            const size_t input_offset = metadata[2 * num_dims];                                    \
            const size_t split_dim = metadata[2 * num_dims + 1];                                   \
            const size_t output_size_on_dim = metadata[2 * num_dims + 2];                          \
            const size_t split_offset = metadata[2 * num_dims + 3];                                \
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
