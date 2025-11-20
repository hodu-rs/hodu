#include "./headers/atomic.metal"
#include "./headers/utils.metal"
#include <metal_stdlib>

using namespace metal;

// Indexing Operations
// ===================
// Operations for selecting, gathering, and scattering tensor elements using index tensors.
// Supports: index_select, index_put, gather, scatter, scatter_add, scatter_max, scatter_min

// ============================================================================
// INDEX SELECT OPERATION
// ============================================================================

// Extracts elements along a specified dimension using a 1D indices array.
// The output tensor has the same shape as input except on the select dimension,
// which is replaced by the number of indices.
//
// Input shape:  [..., input_shape[dim], ...]
// Indices:      [num_indices] (1D array of indices along dim)
// Output shape: [..., num_indices, ...]
//
// Metadata Layout (Total: 2 + num_dims * 2 + 3):
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: input_strides
// - metadata[2+2*num_dims]: input_offset
// - metadata[2+2*num_dims+1]: dim (dimension along which to select)
// - metadata[2+2*num_dims+2]: num_indices (number of indices)
//
// Buffer Layout:
// - buffer(0): input tensor (const device T*)
// - buffer(1): indices (const device int32_t*)
// - buffer(2): output tensor (device T*)
// - buffer(3): metadata (constant size_t*)

#define INDEX_SELECT_OP(TYPENAME, FN_NAME)                                                         \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device int32_t *indices [[buffer(1)]],   \
        device TYPENAME *output [[buffer(2)]], constant size_t *metadata [[buffer(3)]],            \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
                                                                                                   \
        /* Grid-stride loop for better GPU utilization */                                          \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
                                                                                                   \
            const constant size_t *input_shape = metadata + 2;                                     \
            const constant size_t *input_strides = metadata + 2 + num_dims;                        \
            const size_t input_offset = metadata[2 + 2 * num_dims];                                \
            const size_t dim = metadata[2 + 2 * num_dims + 1];                                     \
            const size_t num_indices = metadata[2 + 2 * num_dims + 2];                             \
                                                                                                   \
            /* Calculate output shape (same as input but with dim replaced by num_indices) */      \
            size_t output_shape[16];                                                               \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                output_shape[i] = (i == dim) ? num_indices : input_shape[i];                       \
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
            /* Get the index to select from the indices array */                                   \
            int32_t selected_idx = indices[output_indices[dim]];                                   \
                                                                                                   \
            /* Handle negative indices */                                                          \
            if (selected_idx < 0) {                                                                \
                selected_idx += (int32_t)input_shape[dim];                                         \
            }                                                                                      \
                                                                                                   \
            /* Calculate input flat index */                                                       \
            size_t flat_index = input_offset;                                                      \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                size_t idx = (i == dim) ? (size_t)selected_idx : output_indices[i];                \
                flat_index += idx * input_strides[i];                                              \
            }                                                                                      \
                                                                                                   \
            output[id] = input[flat_index];                                                        \
        }                                                                                          \
    }

// Define index_select operations for all types
INDEX_SELECT_OP(bool, index_select_bool)
INDEX_SELECT_OP(bfloat, index_select_bf16)
INDEX_SELECT_OP(half, index_select_f16)
INDEX_SELECT_OP(float, index_select_f32)
INDEX_SELECT_OP(int8_t, index_select_i8)
INDEX_SELECT_OP(int16_t, index_select_i16)
INDEX_SELECT_OP(int32_t, index_select_i32)
INDEX_SELECT_OP(int64_t, index_select_i64)
INDEX_SELECT_OP(uint8_t, index_select_u8)
INDEX_SELECT_OP(uint16_t, index_select_u16)
INDEX_SELECT_OP(uint32_t, index_select_u32)
INDEX_SELECT_OP(uint64_t, index_select_u64)

// ============================================================================
// INDEX PUT OPERATION
// ============================================================================

// Write values to positions specified by a 1D indices array along a dimension
// Input: [..., input_shape[dim], ...]
// Indices: [num_indices] (1D array of indices along dim)
// Values: [..., num_indices, ...] (same shape as output)
// Output: same shape as input
//
// Metadata layout:
// - metadata[0]: num_els (total number of output elements, same as input)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: input_strides
// - metadata[2+2*num_dims..2+3*num_dims]: values_strides
// - metadata[2+3*num_dims]: input_offset
// - metadata[2+3*num_dims+1]: values_offset
// - metadata[2+3*num_dims+2]: dim (dimension along which to write)
// - metadata[2+3*num_dims+3]: num_indices

#define INDEX_PUT_OP(TYPENAME, FN_NAME)                                                            \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device int32_t *indices [[buffer(1)]],   \
        const device TYPENAME *values [[buffer(2)]], device TYPENAME *output [[buffer(3)]],        \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *input_shape = metadata + 2;                                         \
        const constant size_t *input_strides = metadata + 2 + num_dims;                            \
        const constant size_t *values_strides = metadata + 2 + 2 * num_dims;                       \
        const size_t input_offset = metadata[2 + 3 * num_dims];                                    \
        const size_t values_offset = metadata[2 + 3 * num_dims + 1];                               \
        const size_t dim = metadata[2 + 3 * num_dims + 2];                                         \
        const size_t num_indices = metadata[2 + 3 * num_dims + 3];                                 \
                                                                                                   \
        /* Single-pass: Each thread computes one output element */                                 \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
                                                                                                   \
            /* Calculate output indices from flat id */                                            \
            size_t output_indices[16];                                                             \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                output_indices[d] = temp % input_shape[d];                                         \
                temp /= input_shape[d];                                                            \
            }                                                                                      \
                                                                                                   \
            /* Check if this position should be replaced by looking through indices */             \
            bool found = false;                                                                    \
            size_t values_idx_in_dim = 0;                                                          \
            for (size_t i = 0; i < num_indices; i++) {                                             \
                int32_t target_idx = indices[i];                                                   \
                if (target_idx < 0) {                                                              \
                    target_idx += (int32_t)input_shape[dim];                                       \
                }                                                                                  \
                if ((size_t)target_idx == output_indices[dim]) {                                   \
                    found = true;                                                                  \
                    values_idx_in_dim = i;                                                         \
                    break;                                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            if (found) {                                                                           \
                /* Get value from values tensor */                                                 \
                size_t values_flat_idx = values_offset;                                            \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    size_t idx = (i == dim) ? values_idx_in_dim : output_indices[i];               \
                    values_flat_idx += idx * values_strides[i];                                    \
                }                                                                                  \
                output[id] = values[values_flat_idx];                                              \
            } else {                                                                               \
                /* Get value from input tensor */                                                  \
                size_t input_flat_idx = input_offset;                                              \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_flat_idx += output_indices[i] * input_strides[i];                        \
                }                                                                                  \
                output[id] = input[input_flat_idx];                                                \
            }                                                                                      \
        }                                                                                          \
    }

// Define index_put operations for all types
INDEX_PUT_OP(bool, index_put_bool)
INDEX_PUT_OP(bfloat, index_put_bf16)
INDEX_PUT_OP(half, index_put_f16)
INDEX_PUT_OP(float, index_put_f32)
INDEX_PUT_OP(int8_t, index_put_i8)
INDEX_PUT_OP(int16_t, index_put_i16)
INDEX_PUT_OP(int32_t, index_put_i32)
INDEX_PUT_OP(int64_t, index_put_i64)
INDEX_PUT_OP(uint8_t, index_put_u8)
INDEX_PUT_OP(uint16_t, index_put_u16)
INDEX_PUT_OP(uint32_t, index_put_u32)
INDEX_PUT_OP(uint64_t, index_put_u64)

// ============================================================================
// GATHER OPERATION
// ============================================================================

// Gather elements along a specified dimension using an indices tensor
// Input: [..., input_shape[dim], ...]
// Indices: same shape as input but with dim possibly different size
// Output: same shape as indices
//
// Metadata layout:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: output_shape
// - metadata[2+num_dims..2+2*num_dims]: input_shape
// - metadata[2+2*num_dims..2+3*num_dims]: input_strides
// - metadata[2+3*num_dims..2+4*num_dims]: indices_strides
// - metadata[2+4*num_dims]: input_offset
// - metadata[2+4*num_dims+1]: indices_offset
// - metadata[2+4*num_dims+2]: dim (dimension along which to gather)

#define GATHER_OP(TYPENAME, FN_NAME)                                                               \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device int32_t *indices [[buffer(1)]],   \
        device TYPENAME *output [[buffer(2)]], constant size_t *metadata [[buffer(3)]],            \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
                                                                                                   \
        /* Grid-stride loop for better GPU utilization */                                          \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
                                                                                                   \
            const constant size_t *output_shape = metadata + 2;                                    \
            const constant size_t *input_shape = metadata + 2 + num_dims;                          \
            const constant size_t *input_strides = metadata + 2 + 2 * num_dims;                    \
            const constant size_t *indices_strides = metadata + 2 + 3 * num_dims;                  \
            const size_t input_offset = metadata[2 + 4 * num_dims];                                \
            const size_t indices_offset = metadata[2 + 4 * num_dims + 1];                          \
            const size_t dim = metadata[2 + 4 * num_dims + 2];                                     \
                                                                                                   \
            /* Calculate output indices */                                                         \
            size_t temp = id;                                                                      \
            size_t output_indices[16];                                                             \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            /* Get the index from indices tensor */                                                \
            size_t indices_flat_idx = indices_offset;                                              \
            for (size_t d = 0; d < num_dims; d++) {                                                \
                indices_flat_idx += output_indices[d] * indices_strides[d];                        \
            }                                                                                      \
            int32_t selected_idx = indices[indices_flat_idx];                                      \
                                                                                                   \
            /* Handle negative indices */                                                          \
            if (selected_idx < 0) {                                                                \
                selected_idx += (int32_t)input_shape[dim];                                         \
            }                                                                                      \
                                                                                                   \
            /* Bounds check */                                                                     \
            if (selected_idx < 0 || (size_t)selected_idx >= input_shape[dim]) {                    \
                output[id] = (TYPENAME)0;                                                          \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            /* Calculate input flat index */                                                       \
            size_t input_flat_idx = input_offset;                                                  \
            for (size_t d = 0; d < num_dims; d++) {                                                \
                if (d == dim) {                                                                    \
                    input_flat_idx += ((size_t)selected_idx) * input_strides[d];                   \
                } else {                                                                           \
                    input_flat_idx += output_indices[d] * input_strides[d];                        \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[id] = input[input_flat_idx];                                                    \
        }                                                                                          \
    }

// Define gather operations for all types
GATHER_OP(bool, gather_bool)
GATHER_OP(bfloat, gather_bf16)
GATHER_OP(half, gather_f16)
GATHER_OP(float, gather_f32)
GATHER_OP(int8_t, gather_i8)
GATHER_OP(int16_t, gather_i16)
GATHER_OP(int32_t, gather_i32)
GATHER_OP(int64_t, gather_i64)
GATHER_OP(uint8_t, gather_u8)
GATHER_OP(uint16_t, gather_u16)
GATHER_OP(uint32_t, gather_u32)
GATHER_OP(uint64_t, gather_u64)

// ============================================================================
// SCATTER OPERATION
// ============================================================================

// Scatter src values to input at positions specified by indices along dim
// Input: [..., input_shape[dim], ...] (will be modified)
// Indices: same shape as src
// Src: [..., src_shape[dim], ...]
// Output: same shape as input
//
// Metadata layout:
// - metadata[0]: num_els (number of elements in src tensor to scatter)
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

#define SCATTER_OP(TYPENAME, FN_NAME)                                                              \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device int32_t *indices [[buffer(1)]],   \
        const device TYPENAME *src [[buffer(2)]], device TYPENAME *output [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *input_shape = metadata + 2;                                         \
        const constant size_t *input_strides = metadata + 2 + num_dims;                            \
        const constant size_t *src_shape = metadata + 2 + 2 * num_dims;                            \
        const constant size_t *src_strides = metadata + 2 + 3 * num_dims;                          \
        const constant size_t *indices_strides = metadata + 2 + 4 * num_dims;                      \
        const size_t input_offset = metadata[2 + 5 * num_dims];                                    \
        const size_t src_offset = metadata[2 + 5 * num_dims + 1];                                  \
        const size_t indices_offset = metadata[2 + 5 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 5 * num_dims + 3];                                         \
                                                                                                   \
        /* Single-pass: Scatter src values directly */                                             \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
                                                                                                   \
            /* Calculate src indices from flat id */                                               \
            size_t src_indices[16];                                                                \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                src_indices[d] = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
            }                                                                                      \
                                                                                                   \
            /* Calculate indices flat index */                                                     \
            size_t indices_flat_idx = indices_offset;                                              \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                indices_flat_idx += src_indices[i] * indices_strides[i];                           \
            }                                                                                      \
                                                                                                   \
            /* Get the index to scatter to */                                                      \
            int32_t target_idx = indices[indices_flat_idx];                                        \
                                                                                                   \
            /* Handle negative indices */                                                          \
            if (target_idx < 0) {                                                                  \
                target_idx += (int32_t)input_shape[dim];                                           \
            }                                                                                      \
                                                                                                   \
            /* Bounds check */                                                                     \
            if (target_idx < 0 || (size_t)target_idx >= input_shape[dim]) {                        \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            /* Calculate output flat index */                                                      \
            size_t output_flat_idx = input_offset;                                                 \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                size_t idx = (i == dim) ? (size_t)target_idx : src_indices[i];                     \
                output_flat_idx += idx * input_strides[i];                                         \
            }                                                                                      \
                                                                                                   \
            /* Calculate src flat index */                                                         \
            size_t src_flat_idx = src_offset;                                                      \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                src_flat_idx += src_indices[i] * src_strides[i];                                   \
            }                                                                                      \
                                                                                                   \
            output[output_flat_idx] = src[src_flat_idx];                                           \
        }                                                                                          \
    }

// Define scatter operations for all types
SCATTER_OP(bool, scatter_bool)
SCATTER_OP(bfloat, scatter_bf16)
SCATTER_OP(half, scatter_f16)
SCATTER_OP(float, scatter_f32)
SCATTER_OP(int8_t, scatter_i8)
SCATTER_OP(int16_t, scatter_i16)
SCATTER_OP(int32_t, scatter_i32)
SCATTER_OP(int64_t, scatter_i64)
SCATTER_OP(uint8_t, scatter_u8)
SCATTER_OP(uint16_t, scatter_u16)
SCATTER_OP(uint32_t, scatter_u32)
SCATTER_OP(uint64_t, scatter_u64)

// ============================================================================
// SCATTER ADD OPERATION
// ============================================================================

// Scatter add: accumulate src values to input at positions specified by indices
// Same metadata layout as SCATTER_OP

#define SCATTER_ADD_OP(TYPENAME, FN_NAME)                                                          \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device int32_t *indices [[buffer(1)]],   \
        const device TYPENAME *src [[buffer(2)]], device TYPENAME *output [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *input_shape = metadata + 2;                                         \
        const constant size_t *input_strides = metadata + 2 + num_dims;                            \
        const constant size_t *src_shape = metadata + 2 + 2 * num_dims;                            \
        const constant size_t *src_strides = metadata + 2 + 3 * num_dims;                          \
        const constant size_t *indices_strides = metadata + 2 + 4 * num_dims;                      \
        const size_t input_offset = metadata[2 + 5 * num_dims];                                    \
        const size_t src_offset = metadata[2 + 5 * num_dims + 1];                                  \
        const size_t indices_offset = metadata[2 + 5 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 5 * num_dims + 3];                                         \
                                                                                                   \
        /* Scatter add src values using atomic operations */                                       \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
                                                                                                   \
            /* Calculate src indices from flat id */                                               \
            size_t src_indices[16];                                                                \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                src_indices[d] = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
            }                                                                                      \
                                                                                                   \
            /* Calculate indices flat index */                                                     \
            size_t indices_flat_idx = indices_offset;                                              \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                indices_flat_idx += src_indices[i] * indices_strides[i];                           \
            }                                                                                      \
                                                                                                   \
            /* Get the index to scatter to */                                                      \
            int32_t target_idx = indices[indices_flat_idx];                                        \
                                                                                                   \
            /* Handle negative indices */                                                          \
            if (target_idx < 0) {                                                                  \
                target_idx += (int32_t)input_shape[dim];                                           \
            }                                                                                      \
                                                                                                   \
            /* Bounds check */                                                                     \
            if (target_idx < 0 || (size_t)target_idx >= input_shape[dim]) {                        \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            /* Calculate output flat index */                                                      \
            size_t output_flat_idx = input_offset;                                                 \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                size_t idx = (i == dim) ? (size_t)target_idx : src_indices[i];                     \
                output_flat_idx += idx * input_strides[i];                                         \
            }                                                                                      \
                                                                                                   \
            /* Calculate src flat index */                                                         \
            size_t src_flat_idx = src_offset;                                                      \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                src_flat_idx += src_indices[i] * src_strides[i];                                   \
            }                                                                                      \
                                                                                                   \
            /* Atomic add to handle concurrent writes */                                           \
            atomic_add_wrapper(&output[output_flat_idx], src[src_flat_idx]);                       \
        }                                                                                          \
    }

// Define scatter_add operations for numeric types only
SCATTER_ADD_OP(bfloat, scatter_add_bf16)
SCATTER_ADD_OP(half, scatter_add_f16)
SCATTER_ADD_OP(float, scatter_add_f32)
SCATTER_ADD_OP(int8_t, scatter_add_i8)
SCATTER_ADD_OP(int16_t, scatter_add_i16)
SCATTER_ADD_OP(int32_t, scatter_add_i32)
SCATTER_ADD_OP(int64_t, scatter_add_i64)
SCATTER_ADD_OP(uint8_t, scatter_add_u8)
SCATTER_ADD_OP(uint16_t, scatter_add_u16)
SCATTER_ADD_OP(uint32_t, scatter_add_u32)
SCATTER_ADD_OP(uint64_t, scatter_add_u64)

// ============================================================================
// SCATTER MAX OPERATION
// ============================================================================

// Scatter max: take maximum of input and src at positions specified by indices
// Same metadata layout as SCATTER_OP

#define SCATTER_MAX_OP(TYPENAME, FN_NAME)                                                          \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device int32_t *indices [[buffer(1)]],   \
        const device TYPENAME *src [[buffer(2)]], device TYPENAME *output [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *input_shape = metadata + 2;                                         \
        const constant size_t *input_strides = metadata + 2 + num_dims;                            \
        const constant size_t *src_shape = metadata + 2 + 2 * num_dims;                            \
        const constant size_t *src_strides = metadata + 2 + 3 * num_dims;                          \
        const constant size_t *indices_strides = metadata + 2 + 4 * num_dims;                      \
        const size_t input_offset = metadata[2 + 5 * num_dims];                                    \
        const size_t src_offset = metadata[2 + 5 * num_dims + 1];                                  \
        const size_t indices_offset = metadata[2 + 5 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 5 * num_dims + 3];                                         \
                                                                                                   \
        /* Scatter max values using atomic operations */                                           \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
                                                                                                   \
            /* Calculate src indices from flat id */                                               \
            size_t src_indices[16];                                                                \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                src_indices[d] = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
            }                                                                                      \
                                                                                                   \
            /* Calculate indices flat index */                                                     \
            size_t indices_flat_idx = indices_offset;                                              \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                indices_flat_idx += src_indices[i] * indices_strides[i];                           \
            }                                                                                      \
                                                                                                   \
            /* Get the index to scatter to */                                                      \
            int32_t target_idx = indices[indices_flat_idx];                                        \
                                                                                                   \
            /* Handle negative indices */                                                          \
            if (target_idx < 0) {                                                                  \
                target_idx += (int32_t)input_shape[dim];                                           \
            }                                                                                      \
                                                                                                   \
            /* Bounds check */                                                                     \
            if (target_idx < 0 || (size_t)target_idx >= input_shape[dim]) {                        \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            /* Calculate output flat index */                                                      \
            size_t output_flat_idx = input_offset;                                                 \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                size_t idx = (i == dim) ? (size_t)target_idx : src_indices[i];                     \
                output_flat_idx += idx * input_strides[i];                                         \
            }                                                                                      \
                                                                                                   \
            /* Calculate src flat index */                                                         \
            size_t src_flat_idx = src_offset;                                                      \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                src_flat_idx += src_indices[i] * src_strides[i];                                   \
            }                                                                                      \
                                                                                                   \
            /* Atomic max to handle concurrent writes */                                           \
            atomic_max_wrapper(&output[output_flat_idx], src[src_flat_idx]);                       \
        }                                                                                          \
    }

// Define scatter_max operations for comparable types
SCATTER_MAX_OP(bfloat, scatter_max_bf16)
SCATTER_MAX_OP(half, scatter_max_f16)
SCATTER_MAX_OP(float, scatter_max_f32)
SCATTER_MAX_OP(int8_t, scatter_max_i8)
SCATTER_MAX_OP(int16_t, scatter_max_i16)
SCATTER_MAX_OP(int32_t, scatter_max_i32)
SCATTER_MAX_OP(int64_t, scatter_max_i64)
SCATTER_MAX_OP(uint8_t, scatter_max_u8)
SCATTER_MAX_OP(uint16_t, scatter_max_u16)
SCATTER_MAX_OP(uint32_t, scatter_max_u32)
SCATTER_MAX_OP(uint64_t, scatter_max_u64)

// ============================================================================
// SCATTER MIN OPERATION
// ============================================================================

// Scatter min: take minimum of input and src at positions specified by indices
// Same metadata layout as SCATTER_OP

#define SCATTER_MIN_OP(TYPENAME, FN_NAME)                                                          \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device int32_t *indices [[buffer(1)]],   \
        const device TYPENAME *src [[buffer(2)]], device TYPENAME *output [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *input_shape = metadata + 2;                                         \
        const constant size_t *input_strides = metadata + 2 + num_dims;                            \
        const constant size_t *src_shape = metadata + 2 + 2 * num_dims;                            \
        const constant size_t *src_strides = metadata + 2 + 3 * num_dims;                          \
        const constant size_t *indices_strides = metadata + 2 + 4 * num_dims;                      \
        const size_t input_offset = metadata[2 + 5 * num_dims];                                    \
        const size_t src_offset = metadata[2 + 5 * num_dims + 1];                                  \
        const size_t indices_offset = metadata[2 + 5 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 5 * num_dims + 3];                                         \
                                                                                                   \
        /* Scatter min values using atomic operations */                                           \
        for (uint id = thread_index; id < num_els; id += threads_per_grid) {                       \
                                                                                                   \
            /* Calculate src indices from flat id */                                               \
            size_t src_indices[16];                                                                \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                src_indices[d] = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
            }                                                                                      \
                                                                                                   \
            /* Calculate indices flat index */                                                     \
            size_t indices_flat_idx = indices_offset;                                              \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                indices_flat_idx += src_indices[i] * indices_strides[i];                           \
            }                                                                                      \
                                                                                                   \
            /* Get the index to scatter to */                                                      \
            int32_t target_idx = indices[indices_flat_idx];                                        \
                                                                                                   \
            /* Handle negative indices */                                                          \
            if (target_idx < 0) {                                                                  \
                target_idx += (int32_t)input_shape[dim];                                           \
            }                                                                                      \
                                                                                                   \
            /* Bounds check */                                                                     \
            if (target_idx < 0 || (size_t)target_idx >= input_shape[dim]) {                        \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            /* Calculate output flat index */                                                      \
            size_t output_flat_idx = input_offset;                                                 \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                size_t idx = (i == dim) ? (size_t)target_idx : src_indices[i];                     \
                output_flat_idx += idx * input_strides[i];                                         \
            }                                                                                      \
                                                                                                   \
            /* Calculate src flat index */                                                         \
            size_t src_flat_idx = src_offset;                                                      \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                src_flat_idx += src_indices[i] * src_strides[i];                                   \
            }                                                                                      \
                                                                                                   \
            /* Atomic min to handle concurrent writes */                                           \
            atomic_min_wrapper(&output[output_flat_idx], src[src_flat_idx]);                       \
        }                                                                                          \
    }

// Define scatter_min operations for comparable types
SCATTER_MIN_OP(bfloat, scatter_min_bf16)
SCATTER_MIN_OP(half, scatter_min_f16)
SCATTER_MIN_OP(float, scatter_min_f32)
SCATTER_MIN_OP(int8_t, scatter_min_i8)
SCATTER_MIN_OP(int16_t, scatter_min_i16)
SCATTER_MIN_OP(int32_t, scatter_min_i32)
SCATTER_MIN_OP(int64_t, scatter_min_i64)
SCATTER_MIN_OP(uint8_t, scatter_min_u8)
SCATTER_MIN_OP(uint16_t, scatter_min_u16)
SCATTER_MIN_OP(uint32_t, scatter_min_u32)
SCATTER_MIN_OP(uint64_t, scatter_min_u64)
