#include "ops_windowing.h"
#include "types.h"
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ============================================================================
// WINDOWING OPERATION IMPLEMENTATION MACROS
// ============================================================================
//
// These macros generate sliding window reduction operations for various types.
//
// Metadata layout (same for all operations):
// - metadata[0]: output_size (total number of elements in output)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: input_strides
// - metadata[2+2*num_dims]: input_offset (starting offset in input)
// - metadata[3+2*num_dims..3+3*num_dims]: window_shape (size of window in each dimension)
// - metadata[3+3*num_dims..3+4*num_dims]: strides (step size in each dimension)
// - metadata[3+4*num_dims..3+4*num_dims+2*num_dims]: padding (before and after for each dimension)
// - metadata[3+6*num_dims..]: output_shape
//
// Algorithm:
// 1. For each output element, compute its coordinates
// 2. For each position in the window:
//    - Compute corresponding input coordinates with stride and padding
//    - Check if position is within bounds (considering padding)
//    - Apply reduction operation on valid values
// 3. Out-of-bounds values are treated according to operation:
//    - max: -infinity, min: +infinity, sum/mean: 0

/**
 * @brief Macro to implement windowing reduction operations (max, min, sum)
 *
 * @param IN_TYPE C type of input elements
 * @param OUT_TYPE C type of output elements
 * @param TYPE_SUFFIX Suffix for function name
 * @param INIT_VAL Initial value for accumulator
 * @param ACCUMULATE Statement to accumulate values (uses 'acc' and 'val')
 */
#define REDUCE_WINDOW_OP(IN_TYPE, OUT_TYPE, TYPE_SUFFIX, INIT_VAL, ACCUMULATE)                     \
    void reduce_window_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                      \
                                     const size_t *metadata) {                                     \
        const IN_TYPE *input = (const IN_TYPE *)input_ptr;                                         \
        OUT_TYPE *output = (OUT_TYPE *)output_ptr;                                                 \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
        const size_t *window_shape = metadata + 2 + 2 * num_dims + 1;                              \
        const size_t *strides = metadata + 2 + 3 * num_dims + 1;                                   \
        const size_t *padding = metadata + 2 + 4 * num_dims + 1;                                   \
        const size_t *output_shape = metadata + 2 + 6 * num_dims + 1;                              \
                                                                                                   \
        for (size_t out_idx = 0; out_idx < num_els; out_idx++) {                                   \
            size_t out_coords[16];                                                                 \
            size_t tmp = out_idx;                                                                  \
            for (int i = (int)num_dims - 1; i >= 0; i--) {                                         \
                out_coords[i] = tmp % output_shape[i];                                             \
                tmp /= output_shape[i];                                                            \
            }                                                                                      \
                                                                                                   \
            OUT_TYPE acc = INIT_VAL;                                                               \
                                                                                                   \
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
                IN_TYPE val;                                                                       \
                if (in_bounds) {                                                                   \
                    size_t idx = input_offset;                                                     \
                    for (size_t i = 0; i < num_dims; i++) {                                        \
                        idx += input_coords[i] * input_strides[i];                                 \
                    }                                                                              \
                    val = input[idx];                                                              \
                } else {                                                                           \
                    val = (IN_TYPE)INIT_VAL;                                                       \
                }                                                                                  \
                                                                                                   \
                ACCUMULATE;                                                                        \
            }                                                                                      \
                                                                                                   \
            output[out_idx] = acc;                                                                 \
        }                                                                                          \
    }

/**
 * @brief Macro to implement windowing mean operation
 *
 * Computes the average of values in each window. Padding areas are treated as 0.
 * Only available for floating-point types.
 *
 * @param IN_TYPE C type of input elements (float type)
 * @param OUT_TYPE C type of output elements (float type)
 * @param TYPE_SUFFIX Suffix for function name
 */
#define REDUCE_WINDOW_MEAN_OP(IN_TYPE, OUT_TYPE, TYPE_SUFFIX)                                      \
    void reduce_window_mean_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                 \
                                          const size_t *metadata) {                                \
        const IN_TYPE *input = (const IN_TYPE *)input_ptr;                                         \
        OUT_TYPE *output = (OUT_TYPE *)output_ptr;                                                 \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
                                                                                                   \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
        const size_t *window_shape = metadata + 2 + 2 * num_dims + 1;                              \
        const size_t *strides = metadata + 2 + 3 * num_dims + 1;                                   \
        const size_t *padding = metadata + 2 + 4 * num_dims + 1;                                   \
        const size_t *output_shape = metadata + 2 + 6 * num_dims + 1;                              \
                                                                                                   \
        for (size_t out_idx = 0; out_idx < num_els; out_idx++) {                                   \
            size_t out_coords[16];                                                                 \
            size_t tmp = out_idx;                                                                  \
            for (int i = (int)num_dims - 1; i >= 0; i--) {                                         \
                out_coords[i] = tmp % output_shape[i];                                             \
                tmp /= output_shape[i];                                                            \
            }                                                                                      \
                                                                                                   \
            OUT_TYPE sum = 0;                                                                      \
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
                    sum += (OUT_TYPE)input[idx];                                                   \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[out_idx] = sum / (OUT_TYPE)window_size;                                         \
        }                                                                                          \
    }

REDUCE_WINDOW_OP(f8e4m3_t, f8e4m3_t, max_f8e4m3, -INFINITY, acc = MAX(acc, val))
REDUCE_WINDOW_OP(f8e5m2_t, f8e5m2_t, max_f8e5m2, -INFINITY, acc = MAX(acc, val))
REDUCE_WINDOW_OP(bf16_t, bf16_t, max_bf16, -INFINITY, acc = MAX(acc, val))
REDUCE_WINDOW_OP(f16_t, f16_t, max_f16, -INFINITY, acc = MAX(acc, val))
REDUCE_WINDOW_OP(float, float, max_f32, -INFINITY, acc = MAX(acc, val))
REDUCE_WINDOW_OP(double, double, max_f64, -INFINITY, acc = MAX(acc, val))
REDUCE_WINDOW_OP(int8_t, int8_t, max_i8, INT8_MIN, acc = MAX(acc, val))
REDUCE_WINDOW_OP(int16_t, int16_t, max_i16, INT16_MIN, acc = MAX(acc, val))
REDUCE_WINDOW_OP(int32_t, int32_t, max_i32, INT32_MIN, acc = MAX(acc, val))
REDUCE_WINDOW_OP(int64_t, int64_t, max_i64, INT64_MIN, acc = MAX(acc, val))
REDUCE_WINDOW_OP(uint8_t, uint8_t, max_u8, 0u, acc = MAX(acc, val))
REDUCE_WINDOW_OP(uint16_t, uint16_t, max_u16, 0u, acc = MAX(acc, val))
REDUCE_WINDOW_OP(uint32_t, uint32_t, max_u32, 0u, acc = MAX(acc, val))
REDUCE_WINDOW_OP(uint64_t, uint64_t, max_u64, 0u, acc = MAX(acc, val))

REDUCE_WINDOW_MEAN_OP(f8e4m3_t, f8e4m3_t, f8e4m3)
REDUCE_WINDOW_MEAN_OP(f8e5m2_t, f8e5m2_t, f8e5m2)
REDUCE_WINDOW_MEAN_OP(bf16_t, bf16_t, bf16)
REDUCE_WINDOW_MEAN_OP(f16_t, f16_t, f16)
REDUCE_WINDOW_MEAN_OP(float, float, f32)
REDUCE_WINDOW_MEAN_OP(double, double, f64)

REDUCE_WINDOW_OP(f8e4m3_t, f8e4m3_t, sum_f8e4m3, 0, acc += val)
REDUCE_WINDOW_OP(f8e5m2_t, f8e5m2_t, sum_f8e5m2, 0, acc += val)
REDUCE_WINDOW_OP(bf16_t, bf16_t, sum_bf16, 0, acc += val)
REDUCE_WINDOW_OP(f16_t, f16_t, sum_f16, 0, acc += val)
REDUCE_WINDOW_OP(float, float, sum_f32, 0.0f, acc += val)
REDUCE_WINDOW_OP(double, double, sum_f64, 0.0, acc += val)
REDUCE_WINDOW_OP(int8_t, int8_t, sum_i8, 0, acc += val)
REDUCE_WINDOW_OP(int16_t, int16_t, sum_i16, 0, acc += val)
REDUCE_WINDOW_OP(int32_t, int32_t, sum_i32, 0, acc += val)
REDUCE_WINDOW_OP(int64_t, int64_t, sum_i64, 0, acc += val)
REDUCE_WINDOW_OP(uint8_t, uint8_t, sum_u8, 0u, acc += val)
REDUCE_WINDOW_OP(uint16_t, uint16_t, sum_u16, 0u, acc += val)
REDUCE_WINDOW_OP(uint32_t, uint32_t, sum_u32, 0u, acc += val)
REDUCE_WINDOW_OP(uint64_t, uint64_t, sum_u64, 0u, acc += val)

REDUCE_WINDOW_OP(f8e4m3_t, f8e4m3_t, min_f8e4m3, INFINITY, acc = MIN(acc, val))
REDUCE_WINDOW_OP(f8e5m2_t, f8e5m2_t, min_f8e5m2, INFINITY, acc = MIN(acc, val))
REDUCE_WINDOW_OP(bf16_t, bf16_t, min_bf16, INFINITY, acc = MIN(acc, val))
REDUCE_WINDOW_OP(f16_t, f16_t, min_f16, INFINITY, acc = MIN(acc, val))
REDUCE_WINDOW_OP(float, float, min_f32, INFINITY, acc = MIN(acc, val))
REDUCE_WINDOW_OP(double, double, min_f64, INFINITY, acc = MIN(acc, val))
REDUCE_WINDOW_OP(int8_t, int8_t, min_i8, INT8_MAX, acc = MIN(acc, val))
REDUCE_WINDOW_OP(int16_t, int16_t, min_i16, INT16_MAX, acc = MIN(acc, val))
REDUCE_WINDOW_OP(int32_t, int32_t, min_i32, INT32_MAX, acc = MIN(acc, val))
REDUCE_WINDOW_OP(int64_t, int64_t, min_i64, INT64_MAX, acc = MIN(acc, val))
REDUCE_WINDOW_OP(uint8_t, uint8_t, min_u8, UINT8_MAX, acc = MIN(acc, val))
REDUCE_WINDOW_OP(uint16_t, uint16_t, min_u16, UINT16_MAX, acc = MIN(acc, val))
REDUCE_WINDOW_OP(uint32_t, uint32_t, min_u32, UINT32_MAX, acc = MIN(acc, val))
REDUCE_WINDOW_OP(uint64_t, uint64_t, min_u64, UINT64_MAX, acc = MIN(acc, val))
