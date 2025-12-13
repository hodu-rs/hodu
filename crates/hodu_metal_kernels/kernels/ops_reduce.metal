#include "./headers/constants.metal"
#include "./headers/utils.metal"
#include <metal_stdlib>

using namespace metal;

// Reduce operations: sum, mean, max, min, prod
// These kernels reduce along specified dimensions
//
// Metadata layout:
// - metadata[0]: num_dims (number of dimensions in input)
// - metadata[1..1+num_dims]: dims (shape of input)
// - metadata[1+num_dims..1+2*num_dims]: strides (strides of input)
// - metadata[1+2*num_dims]: offset (starting offset in input)
// - metadata[2+2*num_dims]: output_shape_len (number of dimensions in output)
// - metadata[3+2*num_dims..3+2*num_dims+output_shape_len]: output_shape
// - metadata[3+2*num_dims+output_shape_len]: num_reduce_dims (number of dims to reduce)
// - metadata[4+2*num_dims+output_shape_len..]: reduce_dims (dimension indices to reduce)
// - metadata[...+num_reduce_dims]: keep_dim (1 to keep, 0 to squeeze)
// - metadata[...+1]: reduce_size (total elements to reduce per output)

// Helper macro for reduction operations
#define REDUCE_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME, INIT_VAL, ACCUMULATE)                        \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_dims = metadata[0];                                                       \
        const constant size_t *dims = metadata + 1;                                                \
        const constant size_t *strides = metadata + 1 + num_dims;                                  \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const constant size_t *output_shape = metadata + 3 + 2 * num_dims;                         \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const constant size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;       \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
                                                                                                   \
            OUT_TYPENAME acc = INIT_VAL;                                                           \
                                                                                                   \
            /* Generate output indices */                                                          \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            /* Map output indices to input indices */                                              \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                /* keep_dim=true: output_shape has same ndim as input, just with 1s */             \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                /* keep_dim=false: output_shape has reduced dimensions removed */                  \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0; /* Will iterate */                              \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            /* Iterate over reduced dimensions */                                                  \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                /* Compute indices for reduced dimensions */                                       \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                /* Calculate flat index */                                                         \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                IN_TYPENAME val = input[flat_index];                                               \
                ACCUMULATE;                                                                        \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = acc;                                                              \
        }                                                                                          \
    }

// ============================================================================
// SUM OPERATIONS
// ============================================================================

REDUCE_OP(bfloat, bfloat, sum_bf16, 0.0bf, acc += val)
REDUCE_OP(half, half, sum_f16, 0.0h, acc += val)
REDUCE_OP(float, float, sum_f32, 0.0f, acc += val)
REDUCE_OP(int8_t, int8_t, sum_i8, 0, acc += val)
REDUCE_OP(int16_t, int16_t, sum_i16, 0, acc += val)
REDUCE_OP(int32_t, int32_t, sum_i32, 0, acc += val)
REDUCE_OP(int64_t, int64_t, sum_i64, 0, acc += val)
REDUCE_OP(uint8_t, uint8_t, sum_u8, 0u, acc += val)
REDUCE_OP(uint16_t, uint16_t, sum_u16, 0u, acc += val)
REDUCE_OP(uint32_t, uint32_t, sum_u32, 0u, acc += val)
REDUCE_OP(uint64_t, uint64_t, sum_u64, 0u, acc += val)

// ============================================================================
// MAX OPERATIONS
// ============================================================================

REDUCE_OP(bfloat, bfloat, max_bf16, (bfloat)(-INFINITY), acc = maximum(acc, val))
REDUCE_OP(half, half, max_f16, (half)(-INFINITY), acc = maximum(acc, val))
REDUCE_OP(float, float, max_f32, -INFINITY, acc = maximum(acc, val))
REDUCE_OP(int8_t, int8_t, max_i8, INT8_MIN, acc = maximum(acc, val))
REDUCE_OP(int16_t, int16_t, max_i16, INT16_MIN, acc = maximum(acc, val))
REDUCE_OP(int32_t, int32_t, max_i32, INT32_MIN, acc = maximum(acc, val))
REDUCE_OP(int64_t, int64_t, max_i64, INT64_MIN, acc = maximum(acc, val))
REDUCE_OP(uint8_t, uint8_t, max_u8, 0u, acc = maximum(acc, val))
REDUCE_OP(uint16_t, uint16_t, max_u16, 0u, acc = maximum(acc, val))
REDUCE_OP(uint32_t, uint32_t, max_u32, 0u, acc = maximum(acc, val))
REDUCE_OP(uint64_t, uint64_t, max_u64, 0u, acc = maximum(acc, val))

// ============================================================================
// MIN OPERATIONS
// ============================================================================

REDUCE_OP(bfloat, bfloat, min_bf16, (bfloat)(INFINITY), acc = minimum(acc, val))
REDUCE_OP(half, half, min_f16, (half)(INFINITY), acc = minimum(acc, val))
REDUCE_OP(float, float, min_f32, INFINITY, acc = minimum(acc, val))
REDUCE_OP(int8_t, int8_t, min_i8, INT8_MAX, acc = minimum(acc, val))
REDUCE_OP(int16_t, int16_t, min_i16, INT16_MAX, acc = minimum(acc, val))
REDUCE_OP(int32_t, int32_t, min_i32, INT32_MAX, acc = minimum(acc, val))
REDUCE_OP(int64_t, int64_t, min_i64, INT64_MAX, acc = minimum(acc, val))
REDUCE_OP(uint8_t, uint8_t, min_u8, UINT8_MAX, acc = minimum(acc, val))
REDUCE_OP(uint16_t, uint16_t, min_u16, UINT16_MAX, acc = minimum(acc, val))
REDUCE_OP(uint32_t, uint32_t, min_u32, UINT32_MAX, acc = minimum(acc, val))
REDUCE_OP(uint64_t, uint64_t, min_u64, UINT64_MAX, acc = minimum(acc, val))

// ============================================================================
// PRODUCT OPERATIONS
// ============================================================================

REDUCE_OP(bfloat, bfloat, prod_bf16, 1.0bf, acc *= val)
REDUCE_OP(half, half, prod_f16, 1.0h, acc *= val)
REDUCE_OP(float, float, prod_f32, 1.0f, acc *= val)
REDUCE_OP(int8_t, int8_t, prod_i8, 1, acc *= val)
REDUCE_OP(int16_t, int16_t, prod_i16, 1, acc *= val)
REDUCE_OP(int32_t, int32_t, prod_i32, 1, acc *= val)
REDUCE_OP(int64_t, int64_t, prod_i64, 1, acc *= val)
REDUCE_OP(uint8_t, uint8_t, prod_u8, 1u, acc *= val)
REDUCE_OP(uint16_t, uint16_t, prod_u16, 1u, acc *= val)
REDUCE_OP(uint32_t, uint32_t, prod_u32, 1u, acc *= val)
REDUCE_OP(uint64_t, uint64_t, prod_u64, 1u, acc *= val)

// ============================================================================
// MEAN OPERATIONS (computed as sum / count)
// ============================================================================

#define REDUCE_MEAN_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME)                                         \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_dims = metadata[0];                                                       \
        const constant size_t *dims = metadata + 1;                                                \
        const constant size_t *strides = metadata + 1 + num_dims;                                  \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const constant size_t *output_shape = metadata + 3 + 2 * num_dims;                         \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const constant size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;       \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
                                                                                                   \
            OUT_TYPENAME sum = 0;                                                                  \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                OUT_TYPENAME val = (OUT_TYPENAME)input[flat_index];                                \
                sum += val;                                                                        \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = sum / (OUT_TYPENAME)reduce_size;                                  \
        }                                                                                          \
    }

REDUCE_MEAN_OP(bfloat, bfloat, mean_bf16)
REDUCE_MEAN_OP(half, half, mean_f16)
REDUCE_MEAN_OP(float, float, mean_f32)

// ============================================================================
// L2 NORM OPERATIONS (sqrt of sum of squares)
// ============================================================================

#define REDUCE_NORM_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME)                                         \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_dims = metadata[0];                                                       \
        const constant size_t *dims = metadata + 1;                                                \
        const constant size_t *strides = metadata + 1 + num_dims;                                  \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const constant size_t *output_shape = metadata + 3 + 2 * num_dims;                         \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const constant size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;       \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
                                                                                                   \
            OUT_TYPENAME sum_squares = 0;                                                          \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                OUT_TYPENAME val = (OUT_TYPENAME)input[flat_index];                                \
                sum_squares += val * val;                                                          \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = (OUT_TYPENAME)sqrt((float)sum_squares);                           \
        }                                                                                          \
    }

REDUCE_NORM_OP(bfloat, bfloat, norm_bf16)
REDUCE_NORM_OP(half, half, norm_f16)
REDUCE_NORM_OP(float, float, norm_f32)

// ============================================================================
// LOGSUM OPERATIONS (log of sum)
// ============================================================================

#define REDUCE_LOGSUM_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME)                                       \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_dims = metadata[0];                                                       \
        const constant size_t *dims = metadata + 1;                                                \
        const constant size_t *strides = metadata + 1 + num_dims;                                  \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const constant size_t *output_shape = metadata + 3 + 2 * num_dims;                         \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const constant size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;       \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
                                                                                                   \
            OUT_TYPENAME sum_val = 0;                                                              \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                IN_TYPENAME val = input[flat_index];                                               \
                sum_val += (OUT_TYPENAME)val;                                                      \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = (OUT_TYPENAME)log((float)sum_val);                                \
        }                                                                                          \
    }

REDUCE_LOGSUM_OP(bfloat, bfloat, logsum_bf16)
REDUCE_LOGSUM_OP(half, half, logsum_f16)
REDUCE_LOGSUM_OP(float, float, logsum_f32)

// ============================================================================
// LOGSUMEXP OPERATIONS (log of sum of exp, numerically stable)
// ============================================================================

#define REDUCE_LOGSUMEXP_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME)                                    \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_dims = metadata[0];                                                       \
        const constant size_t *dims = metadata + 1;                                                \
        const constant size_t *strides = metadata + 1 + num_dims;                                  \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const constant size_t *output_shape = metadata + 3 + 2 * num_dims;                         \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const constant size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;       \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            /* First pass: find max for numerical stability */                                     \
            float max_val = -FLT_MAX;                                                              \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                float val = (float)input[flat_index];                                              \
                if (val > max_val) {                                                               \
                    max_val = val;                                                                 \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            /* Second pass: sum exp(x - max) */                                                    \
            float sum_exp = 0.0f;                                                                  \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                float val = (float)input[flat_index];                                              \
                sum_exp += exp(val - max_val);                                                     \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = (OUT_TYPENAME)(max_val + log(sum_exp));                           \
        }                                                                                          \
    }

REDUCE_LOGSUMEXP_OP(bfloat, bfloat, logsumexp_bf16)
REDUCE_LOGSUMEXP_OP(half, half, logsumexp_f16)
REDUCE_LOGSUMEXP_OP(float, float, logsumexp_f32)

// ============================================================================
// ARGMAX / ARGMIN OPERATIONS (return index of max/min value)
// ============================================================================

#define REDUCE_ARGMAX_OP(IN_TYPENAME, FN_NAME)                                                     \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device int32_t *output [[buffer(1)]],       \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_dims = metadata[0];                                                       \
        const constant size_t *dims = metadata + 1;                                                \
        const constant size_t *strides = metadata + 1 + num_dims;                                  \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const constant size_t *output_shape = metadata + 3 + 2 * num_dims;                         \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const constant size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;       \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
                                                                                                   \
            IN_TYPENAME max_val;                                                                   \
            int32_t max_idx = 0;                                                                   \
            bool first = true;                                                                     \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t actual_dim_idx = input_indices[reduce_dims[0]];                             \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                IN_TYPENAME val = input[flat_index];                                               \
                if (first || val > max_val) {                                                      \
                    max_val = val;                                                                 \
                    max_idx = (int32_t)actual_dim_idx;                                             \
                    first = false;                                                                 \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = max_idx;                                                          \
        }                                                                                          \
    }

#define REDUCE_ARGMIN_OP(IN_TYPENAME, FN_NAME)                                                     \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device int32_t *output [[buffer(1)]],       \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_dims = metadata[0];                                                       \
        const constant size_t *dims = metadata + 1;                                                \
        const constant size_t *strides = metadata + 1 + num_dims;                                  \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const constant size_t *output_shape = metadata + 3 + 2 * num_dims;                         \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const constant size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;       \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
                                                                                                   \
            IN_TYPENAME min_val;                                                                   \
            int32_t min_idx = 0;                                                                   \
            bool first = true;                                                                     \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t actual_dim_idx = input_indices[reduce_dims[0]];                             \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                IN_TYPENAME val = input[flat_index];                                               \
                if (first || val < min_val) {                                                      \
                    min_val = val;                                                                 \
                    min_idx = (int32_t)actual_dim_idx;                                             \
                    first = false;                                                                 \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = min_idx;                                                          \
        }                                                                                          \
    }

REDUCE_ARGMAX_OP(bfloat, argmax_bf16)
REDUCE_ARGMAX_OP(half, argmax_f16)
REDUCE_ARGMAX_OP(float, argmax_f32)
REDUCE_ARGMAX_OP(int8_t, argmax_i8)
REDUCE_ARGMAX_OP(int16_t, argmax_i16)
REDUCE_ARGMAX_OP(int32_t, argmax_i32)
REDUCE_ARGMAX_OP(int64_t, argmax_i64)
REDUCE_ARGMAX_OP(uint8_t, argmax_u8)
REDUCE_ARGMAX_OP(uint16_t, argmax_u16)
REDUCE_ARGMAX_OP(uint32_t, argmax_u32)
REDUCE_ARGMAX_OP(uint64_t, argmax_u64)

REDUCE_ARGMIN_OP(bfloat, argmin_bf16)
REDUCE_ARGMIN_OP(half, argmin_f16)
REDUCE_ARGMIN_OP(float, argmin_f32)
REDUCE_ARGMIN_OP(int8_t, argmin_i8)
REDUCE_ARGMIN_OP(int16_t, argmin_i16)
REDUCE_ARGMIN_OP(int32_t, argmin_i32)
REDUCE_ARGMIN_OP(int64_t, argmin_i64)
REDUCE_ARGMIN_OP(uint8_t, argmin_u8)
REDUCE_ARGMIN_OP(uint16_t, argmin_u16)
REDUCE_ARGMIN_OP(uint32_t, argmin_u32)
REDUCE_ARGMIN_OP(uint64_t, argmin_u64)

// ============================================================================
// ANY / ALL OPERATIONS (boolean reductions)
// ============================================================================

// Helper function to check if value is non-zero (truthy)
template <typename T> inline bool is_nonzero(T val) { return val != T(0); }

#define REDUCE_ANY_OP(IN_TYPENAME, FN_NAME)                                                        \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device bool *output [[buffer(1)]],          \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_dims = metadata[0];                                                       \
        const constant size_t *dims = metadata + 1;                                                \
        const constant size_t *strides = metadata + 1 + num_dims;                                  \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const constant size_t *output_shape = metadata + 3 + 2 * num_dims;                         \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const constant size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;       \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
                                                                                                   \
            bool result = false;                                                                   \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                if (is_nonzero(input[flat_index])) {                                               \
                    result = true;                                                                 \
                    break;                                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = result;                                                           \
        }                                                                                          \
    }

#define REDUCE_ALL_OP(IN_TYPENAME, FN_NAME)                                                        \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device bool *output [[buffer(1)]],          \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_dims = metadata[0];                                                       \
        const constant size_t *dims = metadata + 1;                                                \
        const constant size_t *strides = metadata + 1 + num_dims;                                  \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const constant size_t *output_shape = metadata + 3 + 2 * num_dims;                         \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const constant size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;       \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
                                                                                                   \
            bool result = true;                                                                    \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                if (!is_nonzero(input[flat_index])) {                                              \
                    result = false;                                                                \
                    break;                                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = result;                                                           \
        }                                                                                          \
    }

REDUCE_ANY_OP(bool, any_bool)
REDUCE_ANY_OP(bfloat, any_bf16)
REDUCE_ANY_OP(half, any_f16)
REDUCE_ANY_OP(float, any_f32)
REDUCE_ANY_OP(int8_t, any_i8)
REDUCE_ANY_OP(int16_t, any_i16)
REDUCE_ANY_OP(int32_t, any_i32)
REDUCE_ANY_OP(int64_t, any_i64)
REDUCE_ANY_OP(uint8_t, any_u8)
REDUCE_ANY_OP(uint16_t, any_u16)
REDUCE_ANY_OP(uint32_t, any_u32)
REDUCE_ANY_OP(uint64_t, any_u64)

REDUCE_ALL_OP(bool, all_bool)
REDUCE_ALL_OP(bfloat, all_bf16)
REDUCE_ALL_OP(half, all_f16)
REDUCE_ALL_OP(float, all_f32)
REDUCE_ALL_OP(int8_t, all_i8)
REDUCE_ALL_OP(int16_t, all_i16)
REDUCE_ALL_OP(int32_t, all_i32)
REDUCE_ALL_OP(int64_t, all_i64)
REDUCE_ALL_OP(uint8_t, all_u8)
REDUCE_ALL_OP(uint16_t, all_u16)
REDUCE_ALL_OP(uint32_t, all_u32)
REDUCE_ALL_OP(uint64_t, all_u64)
