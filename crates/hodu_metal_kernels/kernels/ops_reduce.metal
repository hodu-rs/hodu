#include "./constants.metal"
#include "./utils.metal"
#include <metal_stdlib>

using namespace metal;

template <typename T> T maximum(T x, T y) { return (x > y) ? x : y; }
template <typename T> T minimum(T x, T y) { return (x < y) ? x : y; }

// Reduce operations: sum, mean, max, min, prod
// These kernels reduce along specified dimensions

// Helper macro for reduction operations
#define REDUCE_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME, INIT_VAL, ACCUMULATE)                        \
    kernel void FN_NAME(                                                                           \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t &num_els [[buffer(2)]], constant size_t &num_dims [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], constant size_t &reduce_size [[buffer(5)]],       \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
            const constant size_t *dims = metadata;                                                \
            const constant size_t *strides = metadata + num_dims;                                  \
            const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;                       \
            const constant size_t *output_shape = metadata + 2 * num_dims + 1;                     \
            const constant size_t *reduce_dims = metadata + 3 * num_dims + 1;                      \
            const size_t num_reduce_dims = metadata[4 * num_dims + 1];                             \
                                                                                                   \
            OUT_TYPENAME acc = INIT_VAL;                                                           \
                                                                                                   \
            /* Generate output indices */                                                          \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                if (output_shape[d] > 0) {                                                         \
                    output_indices[d] = temp % output_shape[d];                                    \
                    temp /= output_shape[d];                                                       \
                } else {                                                                           \
                    output_indices[d] = 0;                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            /* Iterate over reduced dimensions */                                                  \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t input_indices[16];                                                          \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
                                                                                                   \
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

REDUCE_OP(bfloat, bfloat, reduce_sum_bf16, 0.0bf, acc += val)
REDUCE_OP(half, half, reduce_sum_f16, 0.0h, acc += val)
REDUCE_OP(float, float, reduce_sum_f32, 0.0f, acc += val)
REDUCE_OP(int8_t, int32_t, reduce_sum_i8, 0, acc += (int32_t)val)
REDUCE_OP(int16_t, int32_t, reduce_sum_i16, 0, acc += (int32_t)val)
REDUCE_OP(int32_t, int32_t, reduce_sum_i32, 0, acc += val)
REDUCE_OP(int64_t, int64_t, reduce_sum_i64, 0, acc += val)
REDUCE_OP(uint8_t, uint32_t, reduce_sum_u8, 0u, acc += (uint32_t)val)
REDUCE_OP(uint16_t, uint32_t, reduce_sum_u16, 0u, acc += (uint32_t)val)
REDUCE_OP(uint32_t, uint32_t, reduce_sum_u32, 0u, acc += val)
REDUCE_OP(uint64_t, uint64_t, reduce_sum_u64, 0u, acc += val)

// ============================================================================
// MAX OPERATIONS
// ============================================================================

REDUCE_OP(bfloat, bfloat, reduce_max_bf16, (bfloat)(-INFINITY), acc = maximum(acc, val))
REDUCE_OP(half, half, reduce_max_f16, (half)(-INFINITY), acc = maximum(acc, val))
REDUCE_OP(float, float, reduce_max_f32, -INFINITY, acc = maximum(acc, val))
REDUCE_OP(int8_t, int8_t, reduce_max_i8, INT8_MIN, acc = maximum(acc, val))
REDUCE_OP(int16_t, int16_t, reduce_max_i16, INT16_MIN, acc = maximum(acc, val))
REDUCE_OP(int32_t, int32_t, reduce_max_i32, INT32_MIN, acc = maximum(acc, val))
REDUCE_OP(int64_t, int64_t, reduce_max_i64, INT64_MIN, acc = maximum(acc, val))
REDUCE_OP(uint8_t, uint8_t, reduce_max_u8, 0u, acc = maximum(acc, val))
REDUCE_OP(uint16_t, uint16_t, reduce_max_u16, 0u, acc = maximum(acc, val))
REDUCE_OP(uint32_t, uint32_t, reduce_max_u32, 0u, acc = maximum(acc, val))
REDUCE_OP(uint64_t, uint64_t, reduce_max_u64, 0u, acc = maximum(acc, val))

// ============================================================================
// MIN OPERATIONS
// ============================================================================

REDUCE_OP(bfloat, bfloat, reduce_min_bf16, (bfloat)(INFINITY), acc = minimum(acc, val))
REDUCE_OP(half, half, reduce_min_f16, (half)(INFINITY), acc = minimum(acc, val))
REDUCE_OP(float, float, reduce_min_f32, INFINITY, acc = minimum(acc, val))
REDUCE_OP(int8_t, int8_t, reduce_min_i8, INT8_MAX, acc = minimum(acc, val))
REDUCE_OP(int16_t, int16_t, reduce_min_i16, INT16_MAX, acc = minimum(acc, val))
REDUCE_OP(int32_t, int32_t, reduce_min_i32, INT32_MAX, acc = minimum(acc, val))
REDUCE_OP(int64_t, int64_t, reduce_min_i64, INT64_MAX, acc = minimum(acc, val))
REDUCE_OP(uint8_t, uint8_t, reduce_min_u8, UINT8_MAX, acc = minimum(acc, val))
REDUCE_OP(uint16_t, uint16_t, reduce_min_u16, UINT16_MAX, acc = minimum(acc, val))
REDUCE_OP(uint32_t, uint32_t, reduce_min_u32, UINT32_MAX, acc = minimum(acc, val))
REDUCE_OP(uint64_t, uint64_t, reduce_min_u64, UINT64_MAX, acc = minimum(acc, val))

// ============================================================================
// PRODUCT OPERATIONS
// ============================================================================

REDUCE_OP(bfloat, bfloat, reduce_prod_bf16, 1.0bf, acc *= val)
REDUCE_OP(float, float, reduce_prod_f32, 1.0f, acc *= val)
REDUCE_OP(half, half, reduce_prod_f16, 1.0h, acc *= val)
REDUCE_OP(int8_t, int32_t, reduce_prod_i8, 1, acc *= (int32_t)val)
REDUCE_OP(int16_t, int32_t, reduce_prod_i16, 1, acc *= (int32_t)val)
REDUCE_OP(int32_t, int32_t, reduce_prod_i32, 1, acc *= val)
REDUCE_OP(int64_t, int64_t, reduce_prod_i64, 1, acc *= val)
REDUCE_OP(uint8_t, uint32_t, reduce_prod_u8, 1u, acc *= (uint32_t)val)
REDUCE_OP(uint16_t, uint32_t, reduce_prod_u16, 1u, acc *= (uint32_t)val)
REDUCE_OP(uint32_t, uint32_t, reduce_prod_u32, 1u, acc *= val)
REDUCE_OP(uint64_t, uint64_t, reduce_prod_u64, 1u, acc *= val)

// ============================================================================
// MEAN OPERATIONS (computed as sum / count)
// ============================================================================

#define REDUCE_MEAN_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME)                                         \
    kernel void FN_NAME(                                                                           \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t &num_els [[buffer(2)]], constant size_t &num_dims [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], constant size_t &reduce_size [[buffer(5)]],       \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
            const constant size_t *dims = metadata;                                                \
            const constant size_t *strides = metadata + num_dims;                                  \
            const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;                       \
            const constant size_t *output_shape = metadata + 2 * num_dims + 1;                     \
            const constant size_t *reduce_dims = metadata + 3 * num_dims + 1;                      \
            const size_t num_reduce_dims = metadata[4 * num_dims + 1];                             \
                                                                                                   \
            OUT_TYPENAME sum = 0;                                                                  \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                if (output_shape[d] > 0) {                                                         \
                    output_indices[d] = temp % output_shape[d];                                    \
                    temp /= output_shape[d];                                                       \
                } else {                                                                           \
                    output_indices[d] = 0;                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t input_indices[16];                                                          \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
                                                                                                   \
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
                sum += (OUT_TYPENAME)input[flat_index];                                            \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = sum / (OUT_TYPENAME)reduce_size;                                  \
        }                                                                                          \
    }

REDUCE_MEAN_OP(bfloat, bfloat, reduce_mean_bf16)
REDUCE_MEAN_OP(half, half, reduce_mean_f16)
REDUCE_MEAN_OP(float, float, reduce_mean_f32)

// ============================================================================
// L2 NORM OPERATIONS (sqrt of sum of squares)
// ============================================================================

#define REDUCE_NORM_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME)                                         \
    kernel void FN_NAME(                                                                           \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t &num_els [[buffer(2)]], constant size_t &num_dims [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], constant size_t &reduce_size [[buffer(5)]],       \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
            const constant size_t *dims = metadata;                                                \
            const constant size_t *strides = metadata + num_dims;                                  \
            const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;                       \
            const constant size_t *output_shape = metadata + 2 * num_dims + 1;                     \
            const constant size_t *reduce_dims = metadata + 3 * num_dims + 1;                      \
            const size_t num_reduce_dims = metadata[4 * num_dims + 1];                             \
                                                                                                   \
            OUT_TYPENAME sum_squares = 0;                                                          \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                if (output_shape[d] > 0) {                                                         \
                    output_indices[d] = temp % output_shape[d];                                    \
                    temp /= output_shape[d];                                                       \
                } else {                                                                           \
                    output_indices[d] = 0;                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t input_indices[16];                                                          \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
                                                                                                   \
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

REDUCE_NORM_OP(bfloat, bfloat, reduce_norm_bf16)
REDUCE_NORM_OP(half, half, reduce_norm_f16)
REDUCE_NORM_OP(float, float, reduce_norm_f32)

// ============================================================================
// ARGMAX / ARGMIN OPERATIONS (return index of max/min value)
// ============================================================================

#define REDUCE_ARGMAX_OP(IN_TYPENAME, FN_NAME)                                                     \
    kernel void FN_NAME(                                                                           \
        const device IN_TYPENAME *input [[buffer(0)]], device int32_t *output [[buffer(1)]],       \
        constant size_t &num_els [[buffer(2)]], constant size_t &num_dims [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], constant size_t &reduce_size [[buffer(5)]],       \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
            const constant size_t *dims = metadata;                                                \
            const constant size_t *strides = metadata + num_dims;                                  \
            const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;                       \
            const constant size_t *output_shape = metadata + 2 * num_dims + 1;                     \
            const constant size_t *reduce_dims = metadata + 3 * num_dims + 1;                      \
            const size_t num_reduce_dims = metadata[4 * num_dims + 1];                             \
                                                                                                   \
            IN_TYPENAME max_val = (IN_TYPENAME)(-INFINITY);                                        \
            int32_t max_idx = 0;                                                                   \
            bool first = true;                                                                     \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                if (output_shape[d] > 0) {                                                         \
                    output_indices[d] = temp % output_shape[d];                                    \
                    temp /= output_shape[d];                                                       \
                } else {                                                                           \
                    output_indices[d] = 0;                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t input_indices[16];                                                          \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
                                                                                                   \
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
                if (first || val > max_val) {                                                      \
                    max_val = val;                                                                 \
                    max_idx = (int32_t)reduced_idx;                                                \
                    first = false;                                                                 \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = max_idx;                                                          \
        }                                                                                          \
    }

#define REDUCE_ARGMIN_OP(IN_TYPENAME, FN_NAME)                                                     \
    kernel void FN_NAME(                                                                           \
        const device IN_TYPENAME *input [[buffer(0)]], device int32_t *output [[buffer(1)]],       \
        constant size_t &num_els [[buffer(2)]], constant size_t &num_dims [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], constant size_t &reduce_size [[buffer(5)]],       \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
            const constant size_t *dims = metadata;                                                \
            const constant size_t *strides = metadata + num_dims;                                  \
            const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;                       \
            const constant size_t *output_shape = metadata + 2 * num_dims + 1;                     \
            const constant size_t *reduce_dims = metadata + 3 * num_dims + 1;                      \
            const size_t num_reduce_dims = metadata[4 * num_dims + 1];                             \
                                                                                                   \
            IN_TYPENAME min_val = (IN_TYPENAME)(INFINITY);                                         \
            int32_t min_idx = 0;                                                                   \
            bool first = true;                                                                     \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                if (output_shape[d] > 0) {                                                         \
                    output_indices[d] = temp % output_shape[d];                                    \
                    temp /= output_shape[d];                                                       \
                } else {                                                                           \
                    output_indices[d] = 0;                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t input_indices[16];                                                          \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
                                                                                                   \
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
                if (first || val < min_val) {                                                      \
                    min_val = val;                                                                 \
                    min_idx = (int32_t)reduced_idx;                                                \
                    first = false;                                                                 \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = min_idx;                                                          \
        }                                                                                          \
    }

REDUCE_ARGMAX_OP(bfloat, reduce_argmax_bf16)
REDUCE_ARGMAX_OP(half, reduce_argmax_f16)
REDUCE_ARGMAX_OP(float, reduce_argmax_f32)
REDUCE_ARGMAX_OP(int8_t, reduce_argmax_i8)
REDUCE_ARGMAX_OP(int16_t, reduce_argmax_i16)
REDUCE_ARGMAX_OP(int32_t, reduce_argmax_i32)
REDUCE_ARGMAX_OP(int64_t, reduce_argmax_i64)
REDUCE_ARGMAX_OP(uint8_t, reduce_argmax_u8)
REDUCE_ARGMAX_OP(uint16_t, reduce_argmax_u16)
REDUCE_ARGMAX_OP(uint32_t, reduce_argmax_u32)
REDUCE_ARGMAX_OP(uint64_t, reduce_argmax_u64)

REDUCE_ARGMIN_OP(bfloat, reduce_argmin_bf16)
REDUCE_ARGMIN_OP(half, reduce_argmin_f16)
REDUCE_ARGMIN_OP(float, reduce_argmin_f32)
REDUCE_ARGMIN_OP(int8_t, reduce_argmin_i8)
REDUCE_ARGMIN_OP(int16_t, reduce_argmin_i16)
REDUCE_ARGMIN_OP(int32_t, reduce_argmin_i32)
REDUCE_ARGMIN_OP(int64_t, reduce_argmin_i64)
REDUCE_ARGMIN_OP(uint8_t, reduce_argmin_u8)
REDUCE_ARGMIN_OP(uint16_t, reduce_argmin_u16)
REDUCE_ARGMIN_OP(uint32_t, reduce_argmin_u32)
REDUCE_ARGMIN_OP(uint64_t, reduce_argmin_u64)

// ============================================================================
// ANY / ALL OPERATIONS (boolean reductions)
// ============================================================================

// Helper function to check if value is non-zero (truthy)
template <typename T> inline bool is_nonzero(T val) { return val != T(0); }

#define REDUCE_ANY_OP(IN_TYPENAME, FN_NAME)                                                        \
    kernel void FN_NAME(                                                                           \
        const device IN_TYPENAME *input [[buffer(0)]], device bool *output [[buffer(1)]],          \
        constant size_t &num_els [[buffer(2)]], constant size_t &num_dims [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], constant size_t &reduce_size [[buffer(5)]],       \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
            const constant size_t *dims = metadata;                                                \
            const constant size_t *strides = metadata + num_dims;                                  \
            const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;                       \
            const constant size_t *output_shape = metadata + 2 * num_dims + 1;                     \
            const constant size_t *reduce_dims = metadata + 3 * num_dims + 1;                      \
            const size_t num_reduce_dims = metadata[4 * num_dims + 1];                             \
                                                                                                   \
            bool result = false;                                                                   \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                if (output_shape[d] > 0) {                                                         \
                    output_indices[d] = temp % output_shape[d];                                    \
                    temp /= output_shape[d];                                                       \
                } else {                                                                           \
                    output_indices[d] = 0;                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t input_indices[16];                                                          \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
                                                                                                   \
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
    kernel void FN_NAME(                                                                           \
        const device IN_TYPENAME *input [[buffer(0)]], device bool *output [[buffer(1)]],          \
        constant size_t &num_els [[buffer(2)]], constant size_t &num_dims [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], constant size_t &reduce_size [[buffer(5)]],       \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        for (uint output_idx = thread_index; output_idx < num_els;                                 \
             output_idx += threads_per_grid) {                                                     \
            const constant size_t *dims = metadata;                                                \
            const constant size_t *strides = metadata + num_dims;                                  \
            const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;                       \
            const constant size_t *output_shape = metadata + 2 * num_dims + 1;                     \
            const constant size_t *reduce_dims = metadata + 3 * num_dims + 1;                      \
            const size_t num_reduce_dims = metadata[4 * num_dims + 1];                             \
                                                                                                   \
            bool result = true;                                                                    \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                if (output_shape[d] > 0) {                                                         \
                    output_indices[d] = temp % output_shape[d];                                    \
                    temp /= output_shape[d];                                                       \
                } else {                                                                           \
                    output_indices[d] = 0;                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t input_indices[16];                                                          \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
                                                                                                   \
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

REDUCE_ANY_OP(bool, reduce_any_bool)
REDUCE_ANY_OP(bfloat, reduce_any_bf16)
REDUCE_ANY_OP(half, reduce_any_f16)
REDUCE_ANY_OP(float, reduce_any_f32)
REDUCE_ANY_OP(int8_t, reduce_any_i8)
REDUCE_ANY_OP(int16_t, reduce_any_i16)
REDUCE_ANY_OP(int32_t, reduce_any_i32)
REDUCE_ANY_OP(int64_t, reduce_any_i64)
REDUCE_ANY_OP(uint8_t, reduce_any_u8)
REDUCE_ANY_OP(uint16_t, reduce_any_u16)
REDUCE_ANY_OP(uint32_t, reduce_any_u32)
REDUCE_ANY_OP(uint64_t, reduce_any_u64)

REDUCE_ALL_OP(bool, reduce_all_bool)
REDUCE_ALL_OP(bfloat, reduce_all_bf16)
REDUCE_ALL_OP(half, reduce_all_f16)
REDUCE_ALL_OP(float, reduce_all_f32)
REDUCE_ALL_OP(int8_t, reduce_all_i8)
REDUCE_ALL_OP(int16_t, reduce_all_i16)
REDUCE_ALL_OP(int32_t, reduce_all_i32)
REDUCE_ALL_OP(int64_t, reduce_all_i64)
REDUCE_ALL_OP(uint8_t, reduce_all_u8)
REDUCE_ALL_OP(uint16_t, reduce_all_u16)
REDUCE_ALL_OP(uint32_t, reduce_all_u32)
REDUCE_ALL_OP(uint64_t, reduce_all_u64)
