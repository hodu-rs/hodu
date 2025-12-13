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
    void hodu_cpu_##FN_NAME(const void *input_ptr, const int32_t *indices, void *output_ptr,       \
                            const size_t *metadata) {                                              \
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
    void hodu_cpu_##FN_NAME(const void *input_ptr, const int32_t *indices, const void *values_ptr, \
                            void *output_ptr, const size_t *metadata) {                            \
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
// - metadata[2..2+num_dims]: output_shape
// - metadata[2+num_dims..2+2*num_dims]: input_shape
// - metadata[2+2*num_dims..2+3*num_dims]: input_strides
// - metadata[2+3*num_dims..2+4*num_dims]: indices_strides
// - metadata[2+4*num_dims]: input_offset
// - metadata[2+4*num_dims+1]: indices_offset
// - metadata[2+4*num_dims+2]: dim (dimension along which to gather)
//
// Algorithm:
// Similar to index_select but indices can be multi-dimensional. For each
// output element, lookup the index from the indices tensor and gather from input.

/// Macro to implement gather operation
///
/// @param TYPENAME C type for the operation
/// @param FN_NAME Function name
#define GATHER_OP(TYPENAME, FN_NAME)                                                               \
    void hodu_cpu_##FN_NAME(const void *input_ptr, const int32_t *indices, void *output_ptr,       \
                            const size_t *metadata) {                                              \
        const TYPENAME *input = (const TYPENAME *)input_ptr;                                       \
        TYPENAME *output = (TYPENAME *)output_ptr;                                                 \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *output_shape = metadata + 2;                                                 \
        const size_t *input_shape = metadata + 2 + num_dims;                                       \
        const size_t *input_strides = metadata + 2 + 2 * num_dims;                                 \
        const size_t *indices_strides = metadata + 2 + 3 * num_dims;                               \
        const size_t input_offset = metadata[2 + 4 * num_dims];                                    \
        const size_t indices_offset = metadata[2 + 4 * num_dims + 1];                              \
        const size_t dim = metadata[2 + 4 * num_dims + 2];                                         \
                                                                                                   \
        for (size_t id = 0; id < num_els; id++) {                                                  \
            size_t temp = id;                                                                      \
            size_t output_indices[num_dims];                                                       \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            size_t indices_flat_idx = indices_offset;                                              \
            for (size_t d = 0; d < num_dims; d++) {                                                \
                indices_flat_idx += output_indices[d] * indices_strides[d];                        \
            }                                                                                      \
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
    void hodu_cpu_##FN_NAME(const void *input_ptr, const int32_t *indices, const void *src_ptr,    \
                            void *output_ptr, const size_t *metadata) {                            \
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
    void hodu_cpu_##FN_NAME(const void *input_ptr, const int32_t *indices, const void *src_ptr,    \
                            void *output_ptr, const size_t *metadata) {                            \
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

// Exotic floating-point scatter_add (uses proper float arithmetic)
#define SCATTER_ADD_OP_EXOTIC(TYPE, FN_NAME, ADD_FN)                                               \
    void hodu_cpu_##FN_NAME(const void *input_ptr, const int32_t *indices, const void *src_ptr,    \
                            void *output_ptr, const size_t *metadata) {                            \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *src = (const TYPE *)src_ptr;                                                   \
        TYPE *output = (TYPE *)output_ptr;                                                         \
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
            output[output_flat_idx] = ADD_FN(output[output_flat_idx], src[src_flat_idx]);          \
        }                                                                                          \
    }

SCATTER_ADD_OP_EXOTIC(f8e4m3_t, scatter_add_f8e4m3, f8e4m3_add)
SCATTER_ADD_OP_EXOTIC(f8e5m2_t, scatter_add_f8e5m2, f8e5m2_add)
SCATTER_ADD_OP_EXOTIC(bf16_t, scatter_add_bf16, bf16_add)
SCATTER_ADD_OP_EXOTIC(f16_t, scatter_add_f16, f16_add)
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
    void hodu_cpu_##FN_NAME(const void *input_ptr, const int32_t *indices, const void *src_ptr,    \
                            void *output_ptr, const size_t *metadata) {                            \
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
    void hodu_cpu_##FN_NAME(const void *input_ptr, const int32_t *indices, const void *src_ptr,    \
                            void *output_ptr, const size_t *metadata) {                            \
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

// ============================================================================
// ONEHOT OPERATIONS
// ============================================================================
//
// Converts integer indices to one-hot encoded vectors.
//
// Metadata layout:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_input_els (total number of input indices)
// - metadata[2]: num_classes (depth of one-hot dimension)
// - metadata[3]: axis (dimension for one-hot encoding, normalized to positive)
// - metadata[4]: num_dims_out (number of output dimensions)
// - metadata[5..5+num_dims_out]: output_shape
//
// Algorithm:
// For each input element, set output[..., index, ...] = 1.0 at the one-hot axis.
// All other positions are set to 0.0.

/// Macro to implement onehot operation
///
/// @param OUT_TYPENAME C type for the output (float, etc.)
/// @param FN_NAME Function name suffix
/// @param ONE_VALUE The value representing "1" for this type
/// @param ZERO_VALUE The value representing "0" for this type
#define ONEHOT_OP(OUT_TYPENAME, FN_NAME, ONE_VALUE, ZERO_VALUE)                                    \
    void hodu_cpu_##FN_NAME(const int32_t *indices, void *output_ptr, const size_t *metadata) {    \
        OUT_TYPENAME *output = (OUT_TYPENAME *)output_ptr;                                         \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_input_els = metadata[1];                                                  \
        const size_t num_classes = metadata[2];                                                    \
        const size_t axis = metadata[3];                                                           \
        const size_t num_dims_out = metadata[4];                                                   \
        const size_t *output_shape = metadata + 5;                                                 \
                                                                                                   \
        /* Initialize all output elements to zero */                                               \
        for (size_t i = 0; i < num_els; i++) {                                                     \
            output[i] = ZERO_VALUE;                                                                \
        }                                                                                          \
                                                                                                   \
        /* Compute strides for output tensor (row-major) */                                        \
        size_t output_strides[32];                                                                 \
        output_strides[num_dims_out - 1] = 1;                                                      \
        for (int d = (int)num_dims_out - 2; d >= 0; d--) {                                         \
            output_strides[d] = output_strides[d + 1] * output_shape[d + 1];                       \
        }                                                                                          \
                                                                                                   \
        /* Compute input shape (output shape without the axis dimension) */                        \
        size_t input_shape[32];                                                                    \
        size_t num_dims_in = num_dims_out - 1;                                                     \
        for (size_t d = 0, id = 0; d < num_dims_out; d++) {                                        \
            if (d != axis) {                                                                       \
                input_shape[id++] = output_shape[d];                                               \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        /* For each input element */                                                               \
        for (size_t i = 0; i < num_input_els; i++) {                                               \
            int32_t class_idx = indices[i];                                                        \
                                                                                                   \
            /* Handle negative indices */                                                          \
            if (class_idx < 0) {                                                                   \
                class_idx += (int32_t)num_classes;                                                 \
            }                                                                                      \
                                                                                                   \
            /* Skip out of bounds indices */                                                       \
            if (class_idx < 0 || (size_t)class_idx >= num_classes) {                               \
                continue;                                                                          \
            }                                                                                      \
                                                                                                   \
            /* Compute multi-dimensional input indices from flat input index */                    \
            size_t input_indices[32];                                                              \
            size_t temp = i;                                                                       \
            for (int d = (int)num_dims_in - 1; d >= 0; d--) {                                      \
                input_indices[d] = temp % input_shape[d];                                          \
                temp /= input_shape[d];                                                            \
            }                                                                                      \
                                                                                                   \
            /* Compute flat output index by inserting class_idx at axis */                         \
            size_t output_idx = 0;                                                                 \
            size_t input_dim = 0;                                                                  \
            for (size_t d = 0; d < num_dims_out; d++) {                                            \
                if (d == axis) {                                                                   \
                    output_idx += ((size_t)class_idx) * output_strides[d];                         \
                } else {                                                                           \
                    output_idx += input_indices[input_dim] * output_strides[d];                    \
                    input_dim++;                                                                   \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = ONE_VALUE;                                                        \
        }                                                                                          \
    }

// Define onehot operations for all output types
ONEHOT_OP(bool, onehot_bool, true, false)
ONEHOT_OP(f8e4m3_t, onehot_f8e4m3, float_to_f8e4m3(1.0f), float_to_f8e4m3(0.0f))
ONEHOT_OP(f8e5m2_t, onehot_f8e5m2, float_to_f8e5m2(1.0f), float_to_f8e5m2(0.0f))
ONEHOT_OP(bf16_t, onehot_bf16, float_to_bf16(1.0f), float_to_bf16(0.0f))
ONEHOT_OP(f16_t, onehot_f16, float_to_f16(1.0f), float_to_f16(0.0f))
ONEHOT_OP(float, onehot_f32, 1.0f, 0.0f)
ONEHOT_OP(double, onehot_f64, 1.0, 0.0)
ONEHOT_OP(int8_t, onehot_i8, 1, 0)
ONEHOT_OP(int16_t, onehot_i16, 1, 0)
ONEHOT_OP(int32_t, onehot_i32, 1, 0)
ONEHOT_OP(int64_t, onehot_i64, 1, 0)
ONEHOT_OP(uint8_t, onehot_u8, 1, 0)
ONEHOT_OP(uint16_t, onehot_u16, 1, 0)
ONEHOT_OP(uint32_t, onehot_u32, 1, 0)
ONEHOT_OP(uint64_t, onehot_u64, 1, 0)

// ============================================================================
// NONZERO OPERATIONS
// ============================================================================
//
// Returns indices of non-zero elements in the input tensor.
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements in input)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: input_strides
// - metadata[2+2*num_dims]: input_offset
//
// Output: [N, ndim] tensor where N is the count of non-zero elements.
// Each row contains the multi-dimensional indices of a non-zero element.

/// Macro to implement nonzero count operation (first pass)
/// Returns the count of non-zero elements
///
/// @param TYPENAME C type for the operation
/// @param FN_NAME Function name
/// @param IS_NONZERO Expression to check if value is non-zero
#define NONZERO_COUNT_OP(TYPENAME, FN_NAME, IS_NONZERO)                                            \
    size_t hodu_cpu_##FN_NAME(const void *input_ptr, const size_t *metadata) {                     \
        const TYPENAME *input = (const TYPENAME *)input_ptr;                                       \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
                                                                                                   \
        size_t count = 0;                                                                          \
        for (size_t id = 0; id < num_els; id++) {                                                  \
            /* Compute flat index from strided layout */                                           \
            size_t flat_idx = input_offset;                                                        \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t idx = temp % input_shape[d];                                                \
                temp /= input_shape[d];                                                            \
                flat_idx += idx * input_strides[d];                                                \
            }                                                                                      \
            TYPENAME val = input[flat_idx];                                                        \
            if (IS_NONZERO) {                                                                      \
                count++;                                                                           \
            }                                                                                      \
        }                                                                                          \
        return count;                                                                              \
    }

/// Macro to implement nonzero fill operation (second pass)
/// Fills the output buffer with indices of non-zero elements
///
/// @param TYPENAME C type for the operation
/// @param FN_NAME Function name
/// @param IS_NONZERO Expression to check if value is non-zero
#define NONZERO_FILL_OP(TYPENAME, FN_NAME, IS_NONZERO)                                             \
    void hodu_cpu_##FN_NAME(const void *input_ptr, int32_t *output, const size_t *metadata) {      \
        const TYPENAME *input = (const TYPENAME *)input_ptr;                                       \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
                                                                                                   \
        size_t out_idx = 0;                                                                        \
        for (size_t id = 0; id < num_els; id++) {                                                  \
            /* Compute flat index from strided layout */                                           \
            size_t flat_idx = input_offset;                                                        \
            size_t temp = id;                                                                      \
            size_t multi_idx[32];                                                                  \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                multi_idx[d] = temp % input_shape[d];                                              \
                temp /= input_shape[d];                                                            \
                flat_idx += multi_idx[d] * input_strides[d];                                       \
            }                                                                                      \
            TYPENAME val = input[flat_idx];                                                        \
            if (IS_NONZERO) {                                                                      \
                /* Write multi-dimensional indices to output */                                    \
                for (size_t d = 0; d < num_dims; d++) {                                            \
                    output[out_idx * num_dims + d] = (int32_t)multi_idx[d];                        \
                }                                                                                  \
                out_idx++;                                                                         \
            }                                                                                      \
        }                                                                                          \
    }

// Count operations
NONZERO_COUNT_OP(bool, nonzero_count_bool, val)
NONZERO_COUNT_OP(f8e4m3_t, nonzero_count_f8e4m3, f8e4m3_to_float(val) != 0.0f)
NONZERO_COUNT_OP(f8e5m2_t, nonzero_count_f8e5m2, f8e5m2_to_float(val) != 0.0f)
NONZERO_COUNT_OP(bf16_t, nonzero_count_bf16, bf16_to_float(val) != 0.0f)
NONZERO_COUNT_OP(f16_t, nonzero_count_f16, f16_to_float(val) != 0.0f)
NONZERO_COUNT_OP(float, nonzero_count_f32, val != 0.0f)
NONZERO_COUNT_OP(double, nonzero_count_f64, val != 0.0)
NONZERO_COUNT_OP(int8_t, nonzero_count_i8, val != 0)
NONZERO_COUNT_OP(int16_t, nonzero_count_i16, val != 0)
NONZERO_COUNT_OP(int32_t, nonzero_count_i32, val != 0)
NONZERO_COUNT_OP(int64_t, nonzero_count_i64, val != 0)
NONZERO_COUNT_OP(uint8_t, nonzero_count_u8, val != 0)
NONZERO_COUNT_OP(uint16_t, nonzero_count_u16, val != 0)
NONZERO_COUNT_OP(uint32_t, nonzero_count_u32, val != 0)
NONZERO_COUNT_OP(uint64_t, nonzero_count_u64, val != 0)

// Fill operations
NONZERO_FILL_OP(bool, nonzero_fill_bool, val)
NONZERO_FILL_OP(f8e4m3_t, nonzero_fill_f8e4m3, f8e4m3_to_float(val) != 0.0f)
NONZERO_FILL_OP(f8e5m2_t, nonzero_fill_f8e5m2, f8e5m2_to_float(val) != 0.0f)
NONZERO_FILL_OP(bf16_t, nonzero_fill_bf16, bf16_to_float(val) != 0.0f)
NONZERO_FILL_OP(f16_t, nonzero_fill_f16, f16_to_float(val) != 0.0f)
NONZERO_FILL_OP(float, nonzero_fill_f32, val != 0.0f)
NONZERO_FILL_OP(double, nonzero_fill_f64, val != 0.0)
NONZERO_FILL_OP(int8_t, nonzero_fill_i8, val != 0)
NONZERO_FILL_OP(int16_t, nonzero_fill_i16, val != 0)
NONZERO_FILL_OP(int32_t, nonzero_fill_i32, val != 0)
NONZERO_FILL_OP(int64_t, nonzero_fill_i64, val != 0)
NONZERO_FILL_OP(uint8_t, nonzero_fill_u8, val != 0)
NONZERO_FILL_OP(uint16_t, nonzero_fill_u16, val != 0)
NONZERO_FILL_OP(uint32_t, nonzero_fill_u32, val != 0)
NONZERO_FILL_OP(uint64_t, nonzero_fill_u64, val != 0)
