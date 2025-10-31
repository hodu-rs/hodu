/**
 * @file ops_windowing.h
 * @brief Windowing operations header
 *
 * Provides sliding window reduction operations:
 * - reduce_window_max: Maximum value in each window
 * - reduce_window_min: Minimum value in each window
 * - reduce_window_sum: Sum of values in each window
 * - reduce_window_mean: Mean (average) of values in each window
 *
 * These operations apply a reduction function over sliding windows with
 * configurable window size, stride, and padding.
 */

#ifndef OPS_WINDOWING_H
#define OPS_WINDOWING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// WINDOWING OPERATIONS
// ============================================================================
//
// All windowing operations follow consistent signatures:
//   void reduce_window_op_type(const void *input, void *output, const size_t *metadata)
//
// Parameters:
//   input    - Pointer to input tensor data
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing window operation (see below)
//
// Metadata layout (same for all operations):
// - metadata[0]: output_size (total number of elements in output)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: input_strides
// - metadata[2+2*num_dims]: offset (starting offset in input)
// - metadata[3+2*num_dims..3+3*num_dims]: window_shape (size of window in each dimension)
// - metadata[3+3*num_dims..3+4*num_dims]: strides (step size in each dimension)
// - metadata[3+4*num_dims..3+4*num_dims+2*num_dims]: padding (before and after for each dimension)
// - metadata[3+6*num_dims..]: output_shape
//
// Padding behavior:
// - For each dimension i, padding is [pad_before_i, pad_after_i]
// - reduce_window_max: padded areas treated as -infinity
// - reduce_window_min: padded areas treated as +infinity
// - reduce_window_sum/mean: padded areas treated as 0
//
// Type support:
// - reduce_window_max/min: all numeric types (f8e4m3, f8e5m2, bf16, f16, f32, f64, i8-i64, u8-u64)
// - reduce_window_sum: all numeric types
// - reduce_window_mean: float types only (f8e4m3, f8e5m2, bf16, f16, f32, f64)

/**
 * @brief Reduce window max operations (find maximum in each window)
 */
void reduce_window_max_f8e4m3(const void *input, void *output, const size_t *metadata);
void reduce_window_max_f8e5m2(const void *input, void *output, const size_t *metadata);
void reduce_window_max_bf16(const void *input, void *output, const size_t *metadata);
void reduce_window_max_f16(const void *input, void *output, const size_t *metadata);
void reduce_window_max_f32(const void *input, void *output, const size_t *metadata);
void reduce_window_max_f64(const void *input, void *output, const size_t *metadata);
void reduce_window_max_i8(const void *input, void *output, const size_t *metadata);
void reduce_window_max_i16(const void *input, void *output, const size_t *metadata);
void reduce_window_max_i32(const void *input, void *output, const size_t *metadata);
void reduce_window_max_i64(const void *input, void *output, const size_t *metadata);
void reduce_window_max_u8(const void *input, void *output, const size_t *metadata);
void reduce_window_max_u16(const void *input, void *output, const size_t *metadata);
void reduce_window_max_u32(const void *input, void *output, const size_t *metadata);
void reduce_window_max_u64(const void *input, void *output, const size_t *metadata);

/**
 * @brief Reduce window mean operations (compute average in each window)
 * Note: Only available for float types
 */
void reduce_window_mean_f8e4m3(const void *input, void *output, const size_t *metadata);
void reduce_window_mean_f8e5m2(const void *input, void *output, const size_t *metadata);
void reduce_window_mean_bf16(const void *input, void *output, const size_t *metadata);
void reduce_window_mean_f16(const void *input, void *output, const size_t *metadata);
void reduce_window_mean_f32(const void *input, void *output, const size_t *metadata);
void reduce_window_mean_f64(const void *input, void *output, const size_t *metadata);

/**
 * @brief Reduce window sum operations (sum values in each window)
 */
void reduce_window_sum_f8e4m3(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_f8e5m2(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_bf16(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_f16(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_f32(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_f64(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_i8(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_i16(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_i32(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_i64(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_u8(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_u16(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_u32(const void *input, void *output, const size_t *metadata);
void reduce_window_sum_u64(const void *input, void *output, const size_t *metadata);

/**
 * @brief Reduce window min operations (find minimum in each window)
 */
void reduce_window_min_f8e4m3(const void *input, void *output, const size_t *metadata);
void reduce_window_min_f8e5m2(const void *input, void *output, const size_t *metadata);
void reduce_window_min_bf16(const void *input, void *output, const size_t *metadata);
void reduce_window_min_f16(const void *input, void *output, const size_t *metadata);
void reduce_window_min_f32(const void *input, void *output, const size_t *metadata);
void reduce_window_min_f64(const void *input, void *output, const size_t *metadata);
void reduce_window_min_i8(const void *input, void *output, const size_t *metadata);
void reduce_window_min_i16(const void *input, void *output, const size_t *metadata);
void reduce_window_min_i32(const void *input, void *output, const size_t *metadata);
void reduce_window_min_i64(const void *input, void *output, const size_t *metadata);
void reduce_window_min_u8(const void *input, void *output, const size_t *metadata);
void reduce_window_min_u16(const void *input, void *output, const size_t *metadata);
void reduce_window_min_u32(const void *input, void *output, const size_t *metadata);
void reduce_window_min_u64(const void *input, void *output, const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // OPS_WINDOWING_H
