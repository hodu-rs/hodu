#include "./headers/constants.metal"
#include "./headers/utils.metal"
#include <metal_stdlib>

using namespace metal;

// Reduce window operations: Max, Min, Sum
// These kernels perform reductions over sliding windows with configurable strides and padding
//
// Metadata layout:
// - metadata[0]: output_size (total number of elements in output)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: input_strides
// - metadata[2+2*num_dims]: input_offset (starting offset in input)
// - metadata[3+2*num_dims..3+3*num_dims]: window_shape (size of window in each dimension)
// - metadata[3+3*num_dims..3+4*num_dims]: strides (step size in each dimension)
// - metadata[3+4*num_dims..3+4*num_dims+2*num_dims]: padding (before and after for each dimension)
// - metadata[3+6*num_dims..]: output_shape

// Helper macro for reduce_window operations
#define REDUCE_WINDOW_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME, INIT_VAL, ACCUMULATE)                 \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *input_shape = metadata + 2;                                         \
        const constant size_t *input_strides = metadata + 2 + num_dims;                            \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
        const constant size_t *window_shape = metadata + 3 + 2 * num_dims;                         \
        const constant size_t *strides = metadata + 3 + 3 * num_dims;                              \
        const constant size_t *padding = metadata + 3 + 4 * num_dims;                              \
        const constant size_t *output_shape = metadata + 3 + 6 * num_dims;                         \
                                                                                                   \
        for (uint out_idx = thread_index; out_idx < num_els; out_idx += threads_per_grid) {        \
                                                                                                   \
            /* Calculate output coordinates from flat index */                                     \
            size_t out_coords[16];                                                                 \
            size_t tmp = out_idx;                                                                  \
            for (int i = (int)num_dims - 1; i >= 0; i--) {                                         \
                out_coords[i] = tmp % output_shape[i];                                             \
                tmp /= output_shape[i];                                                            \
            }                                                                                      \
                                                                                                   \
            /* Initialize accumulator */                                                           \
            OUT_TYPENAME acc = INIT_VAL;                                                           \
                                                                                                   \
            /* Calculate window size */                                                            \
            size_t window_size = 1;                                                                \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                window_size *= window_shape[i];                                                    \
            }                                                                                      \
                                                                                                   \
            /* Iterate over window */                                                              \
            for (size_t win_idx = 0; win_idx < window_size; win_idx++) {                           \
                /* Calculate window coordinates */                                                 \
                size_t window_coords[16];                                                          \
                size_t tmp_win = win_idx;                                                          \
                for (int i = (int)num_dims - 1; i >= 0; i--) {                                     \
                    window_coords[i] = tmp_win % window_shape[i];                                  \
                    tmp_win /= window_shape[i];                                                    \
                }                                                                                  \
                                                                                                   \
                /* Calculate input coordinates with padding */                                     \
                bool in_bounds = true;                                                             \
                size_t input_coords[16];                                                           \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    /* Position in padded space */                                                 \
                    size_t padded_pos = out_coords[i] * strides[i] + window_coords[i];             \
                    /* Check if within padding region (before) */                                  \
                    size_t pad_before = padding[i * 2];                                            \
                    if (padded_pos < pad_before) {                                                 \
                        in_bounds = false;                                                         \
                        break;                                                                     \
                    }                                                                              \
                    /* Convert to actual input coordinates */                                      \
                    size_t input_pos = padded_pos - pad_before;                                    \
                    if (input_pos >= input_shape[i]) {                                             \
                        in_bounds = false;                                                         \
                        break;                                                                     \
                    }                                                                              \
                    input_coords[i] = input_pos;                                                   \
                }                                                                                  \
                                                                                                   \
                /* Get value from input or use init value for padding */                           \
                IN_TYPENAME val;                                                                   \
                if (in_bounds) {                                                                   \
                    /* Calculate flat index using input strides */                                 \
                    size_t idx = input_offset;                                                     \
                    for (size_t i = 0; i < num_dims; i++) {                                        \
                        idx += input_coords[i] * input_strides[i];                                 \
                    }                                                                              \
                    val = input[idx];                                                              \
                } else {                                                                           \
                    val = (IN_TYPENAME)INIT_VAL;                                                   \
                }                                                                                  \
                                                                                                   \
                ACCUMULATE;                                                                        \
            }                                                                                      \
                                                                                                   \
            output[out_idx] = acc;                                                                 \
        }                                                                                          \
    }

// Helper macro for reduce_window mean operations
#define REDUCE_WINDOW_MEAN_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME)                                  \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t *metadata [[buffer(2)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *input_shape = metadata + 2;                                         \
        const constant size_t *input_strides = metadata + 2 + num_dims;                            \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
        const constant size_t *window_shape = metadata + 3 + 2 * num_dims;                         \
        const constant size_t *strides = metadata + 3 + 3 * num_dims;                              \
        const constant size_t *padding = metadata + 3 + 4 * num_dims;                              \
        const constant size_t *output_shape = metadata + 3 + 6 * num_dims;                         \
                                                                                                   \
        for (uint out_idx = thread_index; out_idx < num_els; out_idx += threads_per_grid) {        \
                                                                                                   \
            size_t out_coords[16];                                                                 \
            size_t tmp = out_idx;                                                                  \
            for (int i = (int)num_dims - 1; i >= 0; i--) {                                         \
                out_coords[i] = tmp % output_shape[i];                                             \
                tmp /= output_shape[i];                                                            \
            }                                                                                      \
                                                                                                   \
            OUT_TYPENAME sum = 0;                                                                  \
            size_t window_size = 1;                                                                \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                window_size *= window_shape[i];                                                    \
            }                                                                                      \
                                                                                                   \
            for (size_t win_idx = 0; win_idx < window_size; win_idx++) {                           \
                size_t window_coords[16];                                                          \
                size_t tmp_win = win_idx;                                                          \
                for (int i = (int)num_dims - 1; i >= 0; i--) {                                     \
                    window_coords[i] = tmp_win % window_shape[i];                                  \
                    tmp_win /= window_shape[i];                                                    \
                }                                                                                  \
                                                                                                   \
                bool in_bounds = true;                                                             \
                size_t input_coords[16];                                                           \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    size_t padded_pos = out_coords[i] * strides[i] + window_coords[i];             \
                    size_t pad_before = padding[i * 2];                                            \
                    if (padded_pos < pad_before) {                                                 \
                        in_bounds = false;                                                         \
                        break;                                                                     \
                    }                                                                              \
                    size_t input_pos = padded_pos - pad_before;                                    \
                    if (input_pos >= input_shape[i]) {                                             \
                        in_bounds = false;                                                         \
                        break;                                                                     \
                    }                                                                              \
                    input_coords[i] = input_pos;                                                   \
                }                                                                                  \
                                                                                                   \
                if (in_bounds) {                                                                   \
                    size_t idx = input_offset;                                                     \
                    for (size_t i = 0; i < num_dims; i++) {                                        \
                        idx += input_coords[i] * input_strides[i];                                 \
                    }                                                                              \
                    sum += (OUT_TYPENAME)input[idx];                                               \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[out_idx] = sum / (OUT_TYPENAME)window_size;                                     \
        }                                                                                          \
    }

// ============================================================================
// MAX REDUCE_WINDOW OPERATIONS
// ============================================================================

REDUCE_WINDOW_OP(bfloat, bfloat, reduce_window_max_bf16, (bfloat)(-INFINITY),
                 acc = maximum(acc, val))
REDUCE_WINDOW_OP(half, half, reduce_window_max_f16, (half)(-INFINITY), acc = maximum(acc, val))
REDUCE_WINDOW_OP(float, float, reduce_window_max_f32, -INFINITY, acc = maximum(acc, val))
REDUCE_WINDOW_OP(uint8_t, uint8_t, reduce_window_max_u8, 0u, acc = maximum(acc, val))
REDUCE_WINDOW_OP(uint16_t, uint16_t, reduce_window_max_u16, 0u, acc = maximum(acc, val))
REDUCE_WINDOW_OP(uint32_t, uint32_t, reduce_window_max_u32, 0u, acc = maximum(acc, val))
REDUCE_WINDOW_OP(uint64_t, uint64_t, reduce_window_max_u64, 0u, acc = maximum(acc, val))
REDUCE_WINDOW_OP(int8_t, int8_t, reduce_window_max_i8, INT8_MIN, acc = maximum(acc, val))
REDUCE_WINDOW_OP(int16_t, int16_t, reduce_window_max_i16, INT16_MIN, acc = maximum(acc, val))
REDUCE_WINDOW_OP(int32_t, int32_t, reduce_window_max_i32, INT32_MIN, acc = maximum(acc, val))
REDUCE_WINDOW_OP(int64_t, int64_t, reduce_window_max_i64, INT64_MIN, acc = maximum(acc, val))

// ============================================================================
// MIN REDUCE_WINDOW OPERATIONS
// ============================================================================

REDUCE_WINDOW_OP(bfloat, bfloat, reduce_window_min_bf16, (bfloat)(INFINITY),
                 acc = minimum(acc, val))
REDUCE_WINDOW_OP(half, half, reduce_window_min_f16, (half)(INFINITY), acc = minimum(acc, val))
REDUCE_WINDOW_OP(float, float, reduce_window_min_f32, INFINITY, acc = minimum(acc, val))
REDUCE_WINDOW_OP(uint8_t, uint8_t, reduce_window_min_u8, UINT8_MAX, acc = minimum(acc, val))
REDUCE_WINDOW_OP(uint16_t, uint16_t, reduce_window_min_u16, UINT16_MAX, acc = minimum(acc, val))
REDUCE_WINDOW_OP(uint32_t, uint32_t, reduce_window_min_u32, UINT32_MAX, acc = minimum(acc, val))
REDUCE_WINDOW_OP(uint64_t, uint64_t, reduce_window_min_u64, UINT64_MAX, acc = minimum(acc, val))
REDUCE_WINDOW_OP(int8_t, int8_t, reduce_window_min_i8, INT8_MAX, acc = minimum(acc, val))
REDUCE_WINDOW_OP(int16_t, int16_t, reduce_window_min_i16, INT16_MAX, acc = minimum(acc, val))
REDUCE_WINDOW_OP(int32_t, int32_t, reduce_window_min_i32, INT32_MAX, acc = minimum(acc, val))
REDUCE_WINDOW_OP(int64_t, int64_t, reduce_window_min_i64, INT64_MAX, acc = minimum(acc, val))

// ============================================================================
// SUM REDUCE_WINDOW OPERATIONS
// ============================================================================

REDUCE_WINDOW_OP(bfloat, bfloat, reduce_window_sum_bf16, 0.0bf, acc += val)
REDUCE_WINDOW_OP(half, half, reduce_window_sum_f16, 0.0h, acc += val)
REDUCE_WINDOW_OP(float, float, reduce_window_sum_f32, 0.0f, acc += val)
REDUCE_WINDOW_OP(uint8_t, uint8_t, reduce_window_sum_u8, 0u, acc += val)
REDUCE_WINDOW_OP(uint16_t, uint16_t, reduce_window_sum_u16, 0u, acc += val)
REDUCE_WINDOW_OP(uint32_t, uint32_t, reduce_window_sum_u32, 0u, acc += val)
REDUCE_WINDOW_OP(uint64_t, uint64_t, reduce_window_sum_u64, 0u, acc += val)
REDUCE_WINDOW_OP(int8_t, int8_t, reduce_window_sum_i8, 0, acc += val)
REDUCE_WINDOW_OP(int16_t, int16_t, reduce_window_sum_i16, 0, acc += val)
REDUCE_WINDOW_OP(int32_t, int32_t, reduce_window_sum_i32, 0, acc += val)
REDUCE_WINDOW_OP(int64_t, int64_t, reduce_window_sum_i64, 0, acc += val)

// ============================================================================
// MEAN REDUCE_WINDOW OPERATIONS
// ============================================================================

REDUCE_WINDOW_MEAN_OP(bfloat, bfloat, reduce_window_mean_bf16)
REDUCE_WINDOW_MEAN_OP(half, half, reduce_window_mean_f16)
REDUCE_WINDOW_MEAN_OP(float, float, reduce_window_mean_f32)
