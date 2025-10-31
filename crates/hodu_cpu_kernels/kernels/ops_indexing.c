#include "ops_indexing.h"
#include "types.h"
#include <stdbool.h>
#include <string.h>

// ============================================================================
// INDEX SELECT OPERATIONS
// ============================================================================
//
// Selects elements along a dimension using a 1D indices array.
//
// Metadata layout:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: input_strides
// - metadata[2+2*num_dims]: input_offset
// - metadata[2+2*num_dims+1]: dim (dimension along which to select)
// - metadata[2+2*num_dims+2]: num_indices (number of indices)
//
// Algorithm:
// For each output element, compute its multi-dimensional index, lookup the
// corresponding index value along the select dimension, and read from input.
// Supports negative indices (Python-style).

/// Macro to implement index_select operation
///
/// @param TYPENAME C type for the operation
/// @param FN_NAME Function name
#define INDEX_SELECT_OP(TYPENAME, FN_NAME)                                                         \
    void FN_NAME(const void *input_ptr, const int32_t *indices, void *output_ptr,                  \
                 const size_t *metadata) {                                                         \
        const TYPENAME *input = (const TYPENAME *)input_ptr;                                       \
        TYPENAME *output = (TYPENAME *)output_ptr;                                                 \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
        const size_t dim = metadata[2 + 2 * num_dims + 1];                                         \
        const size_t num_indices = metadata[2 + 2 * num_dims + 2];                                 \
                                                                                                   \
        for (size_t id = 0; id < num_els; id++) {                                                  \
            size_t temp = id;                                                                      \
            size_t idx_at_dim = 0;                                                                 \
                                                                                                   \
            size_t temp2 = id;                                                                     \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t output_shape_d = (d == (int)dim) ? num_indices : input_shape[d];            \
                if (d == (int)dim) {                                                               \
                    idx_at_dim = temp2 % output_shape_d;                                           \
                }                                                                                  \
                temp2 /= output_shape_d;                                                           \
            }                                                                                      \
                                                                                                   \
            int32_t selected_idx = indices[idx_at_dim];                                            \
                                                                                                   \
            if (selected_idx < 0) {                                                                \
                selected_idx += (int32_t)input_shape[dim];                                         \
            }                                                                                      \
                                                                                                   \
            if (selected_idx < 0 || (size_t)selected_idx >= input_shape[dim]) {                    \
                output[id] = (TYPENAME)0;                                                          \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            size_t flat_index = input_offset;                                                      \
            temp = id;                                                                             \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t output_shape_d = (d == (int)dim) ? num_indices : input_shape[d];            \
                size_t idx = temp % output_shape_d;                                                \
                temp /= output_shape_d;                                                            \
                if (d == (int)dim) {                                                               \
                    flat_index += ((size_t)selected_idx) * input_strides[d];                       \
                } else {                                                                           \
                    flat_index += idx * input_strides[d];                                          \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[id] = input[flat_index];                                                        \
        }                                                                                          \
    }

INDEX_SELECT_OP(bool, index_select_bool)
INDEX_SELECT_OP(f8e4m3_t, index_select_f8e4m3)
INDEX_SELECT_OP(f8e5m2_t, index_select_f8e5m2)
INDEX_SELECT_OP(bf16_t, index_select_bf16)
INDEX_SELECT_OP(f16_t, index_select_f16)
INDEX_SELECT_OP(float, index_select_f32)
INDEX_SELECT_OP(double, index_select_f64)
INDEX_SELECT_OP(int8_t, index_select_i8)
INDEX_SELECT_OP(int16_t, index_select_i16)
INDEX_SELECT_OP(int32_t, index_select_i32)
INDEX_SELECT_OP(int64_t, index_select_i64)
INDEX_SELECT_OP(uint8_t, index_select_u8)
INDEX_SELECT_OP(uint16_t, index_select_u16)
INDEX_SELECT_OP(uint32_t, index_select_u32)
INDEX_SELECT_OP(uint64_t, index_select_u64)

// ============================================================================
// INDEX PUT OPERATIONS
// ============================================================================
//
// Writes values to positions specified by a 1D indices array.
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
//
// Algorithm:
// For each output position, check if its index along the write dimension
// matches any index in the indices array. If so, write from values tensor;
// otherwise, copy from input tensor.

/// Macro to implement index_put operation
///
/// @param TYPENAME C type for the operation
/// @param FN_NAME Function name
#define INDEX_PUT_OP(TYPENAME, FN_NAME)                                                            \
    void FN_NAME(const void *input_ptr, const int32_t *indices, const void *values_ptr,            \
                 void *output_ptr, const size_t *metadata) {                                       \
        const TYPENAME *input = (const TYPENAME *)input_ptr;                                       \
        const TYPENAME *values = (const TYPENAME *)values_ptr;                                     \
        TYPENAME *output = (TYPENAME *)output_ptr;                                                 \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
                                                                                                   \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t *values_strides = metadata + 2 + 2 * num_dims;                                \
        const size_t input_offset = metadata[2 + 3 * num_dims];                                    \
        const size_t values_offset = metadata[2 + 3 * num_dims + 1];                               \
        const size_t dim = metadata[2 + 3 * num_dims + 2];                                         \
        const size_t num_indices = metadata[2 + 3 * num_dims + 3];                                 \
                                                                                                   \
        for (size_t id = 0; id < num_els; id++) {                                                  \
            size_t temp = id;                                                                      \
            size_t idx_at_dim = 0;                                                                 \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t idx = temp % input_shape[d];                                                \
                if (d == (int)dim) {                                                               \
                    idx_at_dim = idx;                                                              \
                }                                                                                  \
                temp /= input_shape[d];                                                            \
            }                                                                                      \
                                                                                                   \
            int found = 0;                                                                         \
            size_t values_idx_in_dim = 0;                                                          \
            for (size_t i = 0; i < num_indices; i++) {                                             \
                int32_t target_idx = indices[i];                                                   \
                if (target_idx < 0) {                                                              \
                    target_idx += (int32_t)input_shape[dim];                                       \
                }                                                                                  \
                if (target_idx >= 0 && (size_t)target_idx == idx_at_dim) {                         \
                    found = 1;                                                                     \
                    values_idx_in_dim = i;                                                         \
                    break;                                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            if (found) {                                                                           \
                size_t values_flat_idx = values_offset;                                            \
                temp = id;                                                                         \
                for (int d = (int)num_dims - 1; d >= 0; d--) {                                     \
                    size_t idx = temp % input_shape[d];                                            \
                    temp /= input_shape[d];                                                        \
                    if (d == (int)dim) {                                                           \
                        values_flat_idx += values_idx_in_dim * values_strides[d];                  \
                    } else {                                                                       \
                        values_flat_idx += idx * values_strides[d];                                \
                    }                                                                              \
                }                                                                                  \
                output[id] = values[values_flat_idx];                                              \
            } else {                                                                               \
                size_t input_flat_idx = input_offset;                                              \
                temp = id;                                                                         \
                for (int d = (int)num_dims - 1; d >= 0; d--) {                                     \
                    size_t idx = temp % input_shape[d];                                            \
                    temp /= input_shape[d];                                                        \
                    input_flat_idx += idx * input_strides[d];                                      \
                }                                                                                  \
                output[id] = input[input_flat_idx];                                                \
            }                                                                                      \
        }                                                                                          \
    }

INDEX_PUT_OP(bool, index_put_bool)
INDEX_PUT_OP(f8e4m3_t, index_put_f8e4m3)
INDEX_PUT_OP(f8e5m2_t, index_put_f8e5m2)
INDEX_PUT_OP(bf16_t, index_put_bf16)
INDEX_PUT_OP(f16_t, index_put_f16)
INDEX_PUT_OP(float, index_put_f32)
INDEX_PUT_OP(double, index_put_f64)
INDEX_PUT_OP(int8_t, index_put_i8)
INDEX_PUT_OP(int16_t, index_put_i16)
INDEX_PUT_OP(int32_t, index_put_i32)
INDEX_PUT_OP(int64_t, index_put_i64)
INDEX_PUT_OP(uint8_t, index_put_u8)
INDEX_PUT_OP(uint16_t, index_put_u16)
INDEX_PUT_OP(uint32_t, index_put_u32)
INDEX_PUT_OP(uint64_t, index_put_u64)

// ============================================================================
// GATHER OPERATIONS
// ============================================================================
//
// Gathers elements using an indices tensor.
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
// - metadata[2+3*num_dims+3]: num_indices (total number of indices)
//
// Algorithm:
// Similar to index_select but indices can be multi-dimensional. For each
// output element, lookup the index from the indices tensor and gather from input.

/// Macro to implement gather operation
///
/// @param TYPENAME C type for the operation
/// @param FN_NAME Function name
#define GATHER_OP(TYPENAME, FN_NAME)                                                               \
    void FN_NAME(const void *input_ptr, const int32_t *indices, void *output_ptr,                  \
                 const size_t *metadata) {                                                         \
        const TYPENAME *input = (const TYPENAME *)input_ptr;                                       \
        TYPENAME *output = (TYPENAME *)output_ptr;                                                 \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t *indices_strides = metadata + 2 + 2 * num_dims;                               \
        const size_t input_offset = metadata[2 + 2 * num_dims + 1];                                \
        const size_t indices_offset = metadata[2 + 2 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 2 * num_dims + 3];                                         \
        const size_t num_indices = metadata[2 + 2 * num_dims + 4];                                 \
                                                                                                   \
        for (size_t id = 0; id < num_els; id++) {                                                  \
            size_t temp = id;                                                                      \
            size_t idx_in_gather_dim = 0;                                                          \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t output_shape_d = (d == (int)dim) ? num_indices : input_shape[d];            \
                size_t idx_in_dim = temp % output_shape_d;                                         \
                temp /= output_shape_d;                                                            \
                if (d == (int)dim) {                                                               \
                    idx_in_gather_dim = idx_in_dim;                                                \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            size_t indices_flat_idx = indices_offset + idx_in_gather_dim * indices_strides[0];     \
            int32_t selected_idx = indices[indices_flat_idx];                                      \
                                                                                                   \
            if (selected_idx < 0) {                                                                \
                selected_idx += (int32_t)input_shape[dim];                                         \
            }                                                                                      \
                                                                                                   \
            if (selected_idx < 0 || (size_t)selected_idx >= input_shape[dim]) {                    \
                output[id] = (TYPENAME)0;                                                          \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            size_t input_flat_idx = input_offset;                                                  \
            temp = id;                                                                             \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t output_shape_d = (d == (int)dim) ? num_indices : input_shape[d];            \
                size_t idx_in_dim = temp % output_shape_d;                                         \
                temp /= output_shape_d;                                                            \
                if (d == (int)dim) {                                                               \
                    input_flat_idx += ((size_t)selected_idx) * input_strides[d];                   \
                } else {                                                                           \
                    input_flat_idx += idx_in_dim * input_strides[d];                               \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[id] = input[input_flat_idx];                                                    \
        }                                                                                          \
    }

GATHER_OP(bool, gather_bool)
GATHER_OP(f8e4m3_t, gather_f8e4m3)
GATHER_OP(f8e5m2_t, gather_f8e5m2)
GATHER_OP(bf16_t, gather_bf16)
GATHER_OP(f16_t, gather_f16)
GATHER_OP(float, gather_f32)
GATHER_OP(double, gather_f64)
GATHER_OP(int8_t, gather_i8)
GATHER_OP(int16_t, gather_i16)
GATHER_OP(int32_t, gather_i32)
GATHER_OP(int64_t, gather_i64)
GATHER_OP(uint8_t, gather_u8)
GATHER_OP(uint16_t, gather_u16)
GATHER_OP(uint32_t, gather_u32)
GATHER_OP(uint64_t, gather_u64)

// ============================================================================
// SCATTER OPERATIONS
// ============================================================================
//
// Scatters src values to input at positions specified by indices.
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
//
// Algorithm:
// First copy input to output. Then for each src element, lookup the target
// index from indices tensor and write src value to the computed output position.

/// Macro to implement scatter operation (overwrite mode)
///
/// @param TYPENAME C type for the operation
/// @param FN_NAME Function name
#define SCATTER_OP(TYPENAME, FN_NAME)                                                              \
    void FN_NAME(const void *input_ptr, const int32_t *indices, const void *src_ptr,               \
                 void *output_ptr, const size_t *metadata) {                                       \
        const TYPENAME *input = (const TYPENAME *)input_ptr;                                       \
        const TYPENAME *src = (const TYPENAME *)src_ptr;                                           \
        TYPENAME *output = (TYPENAME *)output_ptr;                                                 \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t *src_shape = metadata + 2 + 2 * num_dims;                                     \
        const size_t *src_strides = metadata + 2 + 3 * num_dims;                                   \
        const size_t *indices_strides = metadata + 2 + 4 * num_dims;                               \
        const size_t input_offset = metadata[2 + 5 * num_dims];                                    \
        const size_t src_offset = metadata[2 + 5 * num_dims + 1];                                  \
        const size_t indices_offset = metadata[2 + 5 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 5 * num_dims + 3];                                         \
                                                                                                   \
        size_t input_total_els = 1;                                                                \
        for (size_t i = 0; i < num_dims; i++) {                                                    \
            input_total_els *= input_shape[i];                                                     \
        }                                                                                          \
        for (size_t i = 0; i < input_total_els; i++) {                                             \
            size_t flat_idx = input_offset;                                                        \
            size_t temp = i;                                                                       \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t idx_in_dim = temp % input_shape[d];                                         \
                temp /= input_shape[d];                                                            \
                flat_idx += idx_in_dim * input_strides[d];                                         \
            }                                                                                      \
            output[i] = input[flat_idx];                                                           \
        }                                                                                          \
                                                                                                   \
        for (size_t id = 0; id < num_els; id++) {                                                  \
            size_t indices_flat_idx = indices_offset;                                              \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t src_idx = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
                indices_flat_idx += src_idx * indices_strides[d];                                  \
            }                                                                                      \
                                                                                                   \
            int32_t target_idx = indices[indices_flat_idx];                                        \
                                                                                                   \
            if (target_idx < 0) {                                                                  \
                target_idx += (int32_t)input_shape[dim];                                           \
            }                                                                                      \
                                                                                                   \
            if (target_idx < 0 || (size_t)target_idx >= input_shape[dim]) {                        \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            size_t multi_idx[32];                                                                  \
            temp = id;                                                                             \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                multi_idx[d] = temp % src_shape[d];                                                \
                temp /= src_shape[d];                                                              \
            }                                                                                      \
            multi_idx[dim] = (size_t)target_idx;                                                   \
                                                                                                   \
            size_t output_flat_idx = 0;                                                            \
            for (size_t d = 0; d < num_dims; d++) {                                                \
                size_t stride = 1;                                                                 \
                for (size_t dd = d + 1; dd < num_dims; dd++) {                                     \
                    stride *= input_shape[dd];                                                     \
                }                                                                                  \
                output_flat_idx += multi_idx[d] * stride;                                          \
            }                                                                                      \
                                                                                                   \
            if (output_flat_idx >= input_total_els) {                                              \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            size_t src_flat_idx = src_offset;                                                      \
            temp = id;                                                                             \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t src_idx = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
                src_flat_idx += src_idx * src_strides[d];                                          \
            }                                                                                      \
                                                                                                   \
            output[output_flat_idx] = src[src_flat_idx];                                           \
        }                                                                                          \
    }

SCATTER_OP(bool, scatter_bool)
SCATTER_OP(f8e4m3_t, scatter_f8e4m3)
SCATTER_OP(f8e5m2_t, scatter_f8e5m2)
SCATTER_OP(bf16_t, scatter_bf16)
SCATTER_OP(f16_t, scatter_f16)
SCATTER_OP(float, scatter_f32)
SCATTER_OP(double, scatter_f64)
SCATTER_OP(int8_t, scatter_i8)
SCATTER_OP(int16_t, scatter_i16)
SCATTER_OP(int32_t, scatter_i32)
SCATTER_OP(int64_t, scatter_i64)
SCATTER_OP(uint8_t, scatter_u8)
SCATTER_OP(uint16_t, scatter_u16)
SCATTER_OP(uint32_t, scatter_u32)
SCATTER_OP(uint64_t, scatter_u64)

// ============================================================================
// SCATTER ADD OPERATIONS
// ============================================================================
//
// Scatters src values by accumulating (adding) to existing values.
// Metadata layout: Same as scatter operations.
//
// Algorithm:
// Same as scatter but uses += instead of = to accumulate values.
// Useful when multiple src elements map to the same output position.

/// Macro to implement scatter_add operation (accumulation mode)
///
/// @param TYPENAME C type for the operation
/// @param FN_NAME Function name
#define SCATTER_ADD_OP(TYPENAME, FN_NAME)                                                          \
    void FN_NAME(const void *input_ptr, const int32_t *indices, const void *src_ptr,               \
                 void *output_ptr, const size_t *metadata) {                                       \
        const TYPENAME *input = (const TYPENAME *)input_ptr;                                       \
        const TYPENAME *src = (const TYPENAME *)src_ptr;                                           \
        TYPENAME *output = (TYPENAME *)output_ptr;                                                 \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t *src_shape = metadata + 2 + 2 * num_dims;                                     \
        const size_t *src_strides = metadata + 2 + 3 * num_dims;                                   \
        const size_t *indices_strides = metadata + 2 + 4 * num_dims;                               \
        const size_t input_offset = metadata[2 + 5 * num_dims];                                    \
        const size_t src_offset = metadata[2 + 5 * num_dims + 1];                                  \
        const size_t indices_offset = metadata[2 + 5 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 5 * num_dims + 3];                                         \
                                                                                                   \
        size_t input_total_els = 1;                                                                \
        for (size_t i = 0; i < num_dims; i++) {                                                    \
            input_total_els *= input_shape[i];                                                     \
        }                                                                                          \
        for (size_t i = 0; i < input_total_els; i++) {                                             \
            size_t flat_idx = input_offset;                                                        \
            size_t temp = i;                                                                       \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t idx_in_dim = temp % input_shape[d];                                         \
                temp /= input_shape[d];                                                            \
                flat_idx += idx_in_dim * input_strides[d];                                         \
            }                                                                                      \
            output[i] = input[flat_idx];                                                           \
        }                                                                                          \
                                                                                                   \
        for (size_t id = 0; id < num_els; id++) {                                                  \
            size_t indices_flat_idx = indices_offset;                                              \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t src_idx = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
                indices_flat_idx += src_idx * indices_strides[d];                                  \
            }                                                                                      \
                                                                                                   \
            int32_t target_idx = indices[indices_flat_idx];                                        \
                                                                                                   \
            if (target_idx < 0) {                                                                  \
                target_idx += (int32_t)input_shape[dim];                                           \
            }                                                                                      \
                                                                                                   \
            if (target_idx < 0 || (size_t)target_idx >= input_shape[dim]) {                        \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            size_t multi_idx[32];                                                                  \
            temp = id;                                                                             \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                multi_idx[d] = temp % src_shape[d];                                                \
                temp /= src_shape[d];                                                              \
            }                                                                                      \
            multi_idx[dim] = (size_t)target_idx;                                                   \
                                                                                                   \
            size_t output_flat_idx = 0;                                                            \
            for (size_t d = 0; d < num_dims; d++) {                                                \
                size_t stride = 1;                                                                 \
                for (size_t dd = d + 1; dd < num_dims; dd++) {                                     \
                    stride *= input_shape[dd];                                                     \
                }                                                                                  \
                output_flat_idx += multi_idx[d] * stride;                                          \
            }                                                                                      \
                                                                                                   \
            if (output_flat_idx >= input_total_els) {                                              \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            size_t src_flat_idx = src_offset;                                                      \
            temp = id;                                                                             \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t src_idx = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
                src_flat_idx += src_idx * src_strides[d];                                          \
            }                                                                                      \
                                                                                                   \
            output[output_flat_idx] += src[src_flat_idx];                                          \
        }                                                                                          \
    }

SCATTER_ADD_OP(f8e4m3_t, scatter_add_f8e4m3)
SCATTER_ADD_OP(f8e5m2_t, scatter_add_f8e5m2)
SCATTER_ADD_OP(bf16_t, scatter_add_bf16)
SCATTER_ADD_OP(f16_t, scatter_add_f16)
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
// SCATTER MAX OPERATIONS
// ============================================================================
//
// Scatters src values by taking the maximum with existing values.
// Metadata layout: Same as scatter operations.
//
// Algorithm:
// Same as scatter but compares and keeps the maximum value.

/// Macro to implement scatter_max operation (maximum mode)
///
/// @param TYPENAME C type for the operation
/// @param FN_NAME Function name
#define SCATTER_MAX_OP(TYPENAME, FN_NAME)                                                          \
    void FN_NAME(const void *input_ptr, const int32_t *indices, const void *src_ptr,               \
                 void *output_ptr, const size_t *metadata) {                                       \
        const TYPENAME *input = (const TYPENAME *)input_ptr;                                       \
        const TYPENAME *src = (const TYPENAME *)src_ptr;                                           \
        TYPENAME *output = (TYPENAME *)output_ptr;                                                 \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t *src_shape = metadata + 2 + 2 * num_dims;                                     \
        const size_t *src_strides = metadata + 2 + 3 * num_dims;                                   \
        const size_t *indices_strides = metadata + 2 + 4 * num_dims;                               \
        const size_t input_offset = metadata[2 + 5 * num_dims];                                    \
        const size_t src_offset = metadata[2 + 5 * num_dims + 1];                                  \
        const size_t indices_offset = metadata[2 + 5 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 5 * num_dims + 3];                                         \
                                                                                                   \
        size_t input_total_els = 1;                                                                \
        for (size_t i = 0; i < num_dims; i++) {                                                    \
            input_total_els *= input_shape[i];                                                     \
        }                                                                                          \
        for (size_t i = 0; i < input_total_els; i++) {                                             \
            size_t flat_idx = input_offset;                                                        \
            size_t temp = i;                                                                       \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t idx_in_dim = temp % input_shape[d];                                         \
                temp /= input_shape[d];                                                            \
                flat_idx += idx_in_dim * input_strides[d];                                         \
            }                                                                                      \
            output[i] = input[flat_idx];                                                           \
        }                                                                                          \
                                                                                                   \
        for (size_t id = 0; id < num_els; id++) {                                                  \
            size_t indices_flat_idx = indices_offset;                                              \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t src_idx = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
                indices_flat_idx += src_idx * indices_strides[d];                                  \
            }                                                                                      \
                                                                                                   \
            int32_t target_idx = indices[indices_flat_idx];                                        \
                                                                                                   \
            if (target_idx < 0) {                                                                  \
                target_idx += (int32_t)input_shape[dim];                                           \
            }                                                                                      \
                                                                                                   \
            if (target_idx < 0 || (size_t)target_idx >= input_shape[dim]) {                        \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            size_t multi_idx[32];                                                                  \
            temp = id;                                                                             \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                multi_idx[d] = temp % src_shape[d];                                                \
                temp /= src_shape[d];                                                              \
            }                                                                                      \
            multi_idx[dim] = (size_t)target_idx;                                                   \
                                                                                                   \
            size_t output_flat_idx = 0;                                                            \
            for (size_t d = 0; d < num_dims; d++) {                                                \
                size_t stride = 1;                                                                 \
                for (size_t dd = d + 1; dd < num_dims; dd++) {                                     \
                    stride *= input_shape[dd];                                                     \
                }                                                                                  \
                output_flat_idx += multi_idx[d] * stride;                                          \
            }                                                                                      \
                                                                                                   \
            if (output_flat_idx >= input_total_els) {                                              \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            size_t src_flat_idx = src_offset;                                                      \
            temp = id;                                                                             \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t src_idx = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
                src_flat_idx += src_idx * src_strides[d];                                          \
            }                                                                                      \
                                                                                                   \
            TYPENAME src_val = src[src_flat_idx];                                                  \
            if (src_val > output[output_flat_idx]) {                                               \
                output[output_flat_idx] = src_val;                                                 \
            }                                                                                      \
        }                                                                                          \
    }

SCATTER_MAX_OP(f8e4m3_t, scatter_max_f8e4m3)
SCATTER_MAX_OP(f8e5m2_t, scatter_max_f8e5m2)
SCATTER_MAX_OP(bf16_t, scatter_max_bf16)
SCATTER_MAX_OP(f16_t, scatter_max_f16)
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
// SCATTER MIN OPERATIONS
// ============================================================================
//
// Scatters src values by taking the minimum with existing values.
// Metadata layout: Same as scatter operations.
//
// Algorithm:
// Same as scatter but compares and keeps the minimum value.

/// Macro to implement scatter_min operation (minimum mode)
///
/// @param TYPENAME C type for the operation
/// @param FN_NAME Function name
#define SCATTER_MIN_OP(TYPENAME, FN_NAME)                                                          \
    void FN_NAME(const void *input_ptr, const int32_t *indices, const void *src_ptr,               \
                 void *output_ptr, const size_t *metadata) {                                       \
        const TYPENAME *input = (const TYPENAME *)input_ptr;                                       \
        const TYPENAME *src = (const TYPENAME *)src_ptr;                                           \
        TYPENAME *output = (TYPENAME *)output_ptr;                                                 \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t *src_shape = metadata + 2 + 2 * num_dims;                                     \
        const size_t *src_strides = metadata + 2 + 3 * num_dims;                                   \
        const size_t *indices_strides = metadata + 2 + 4 * num_dims;                               \
        const size_t input_offset = metadata[2 + 5 * num_dims];                                    \
        const size_t src_offset = metadata[2 + 5 * num_dims + 1];                                  \
        const size_t indices_offset = metadata[2 + 5 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 5 * num_dims + 3];                                         \
                                                                                                   \
        size_t input_total_els = 1;                                                                \
        for (size_t i = 0; i < num_dims; i++) {                                                    \
            input_total_els *= input_shape[i];                                                     \
        }                                                                                          \
        for (size_t i = 0; i < input_total_els; i++) {                                             \
            size_t flat_idx = input_offset;                                                        \
            size_t temp = i;                                                                       \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t idx_in_dim = temp % input_shape[d];                                         \
                temp /= input_shape[d];                                                            \
                flat_idx += idx_in_dim * input_strides[d];                                         \
            }                                                                                      \
            output[i] = input[flat_idx];                                                           \
        }                                                                                          \
                                                                                                   \
        for (size_t id = 0; id < num_els; id++) {                                                  \
            size_t indices_flat_idx = indices_offset;                                              \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t src_idx = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
                indices_flat_idx += src_idx * indices_strides[d];                                  \
            }                                                                                      \
                                                                                                   \
            int32_t target_idx = indices[indices_flat_idx];                                        \
                                                                                                   \
            if (target_idx < 0) {                                                                  \
                target_idx += (int32_t)input_shape[dim];                                           \
            }                                                                                      \
                                                                                                   \
            if (target_idx < 0 || (size_t)target_idx >= input_shape[dim]) {                        \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            size_t multi_idx[32];                                                                  \
            temp = id;                                                                             \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                multi_idx[d] = temp % src_shape[d];                                                \
                temp /= src_shape[d];                                                              \
            }                                                                                      \
            multi_idx[dim] = (size_t)target_idx;                                                   \
                                                                                                   \
            size_t output_flat_idx = 0;                                                            \
            for (size_t d = 0; d < num_dims; d++) {                                                \
                size_t stride = 1;                                                                 \
                for (size_t dd = d + 1; dd < num_dims; dd++) {                                     \
                    stride *= input_shape[dd];                                                     \
                }                                                                                  \
                output_flat_idx += multi_idx[d] * stride;                                          \
            }                                                                                      \
                                                                                                   \
            if (output_flat_idx >= input_total_els) {                                              \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            size_t src_flat_idx = src_offset;                                                      \
            temp = id;                                                                             \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t src_idx = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
                src_flat_idx += src_idx * src_strides[d];                                          \
            }                                                                                      \
                                                                                                   \
            TYPENAME src_val = src[src_flat_idx];                                                  \
            if (src_val < output[output_flat_idx]) {                                               \
                output[output_flat_idx] = src_val;                                                 \
            }                                                                                      \
        }                                                                                          \
    }

SCATTER_MIN_OP(f8e4m3_t, scatter_min_f8e4m3)
SCATTER_MIN_OP(f8e5m2_t, scatter_min_f8e5m2)
SCATTER_MIN_OP(bf16_t, scatter_min_bf16)
SCATTER_MIN_OP(f16_t, scatter_min_f16)
SCATTER_MIN_OP(float, scatter_min_f32)
SCATTER_MIN_OP(int8_t, scatter_min_i8)
SCATTER_MIN_OP(int16_t, scatter_min_i16)
SCATTER_MIN_OP(int32_t, scatter_min_i32)
SCATTER_MIN_OP(int64_t, scatter_min_i64)
SCATTER_MIN_OP(uint8_t, scatter_min_u8)
SCATTER_MIN_OP(uint16_t, scatter_min_u16)
SCATTER_MIN_OP(uint32_t, scatter_min_u32)
SCATTER_MIN_OP(uint64_t, scatter_min_u64)
