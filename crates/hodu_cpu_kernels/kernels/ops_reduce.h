/**
 * @file ops_reduce.h
 * @brief Tensor reduction operations header
 *
 * Provides various reduction operations to aggregate tensor values:
 * - Aggregations: sum, mean, prod
 * - Statistics: std, var, norm
 * - Extrema: max, min
 * - Indices: argmax, argmin
 * - Logical: any, all
 *
 * All operations reduce tensors along specified dimensions with optional dimension preservation.
 */

#ifndef OPS_REDUCE_H
#define OPS_REDUCE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// REDUCTION OPERATIONS
// ============================================================================
//
// All reduction operations follow consistent signatures:
//   void reduce_op_type(const void *input, void *output, const size_t *metadata)
//
// Parameters:
//   input    - Pointer to input tensor data
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing reduction (see below)
//
// Metadata layout (same for all operations):
// - metadata[0]: num_dims (number of dimensions in input)
// - metadata[1..1+num_dims]: dims (shape of input)
// - metadata[1+num_dims..1+2*num_dims]: strides (strides of input)
// - metadata[1+2*num_dims]: offset (starting offset in input)
// - metadata[2+2*num_dims]: output_shape_len (number of dimensions in output)
// - metadata[3+2*num_dims..3+2*num_dims+output_shape_len]: output_shape
// - metadata[3+2*num_dims+output_shape_len]: num_reduce_dims (number of dims to reduce)
// - metadata[4+2*num_dims+output_shape_len..]: reduce_dims (dimension indices)
// - metadata[...+num_reduce_dims]: keep_dim (1 to keep dims as size 1, 0 to squeeze)
// - metadata[...+1]: reduce_size (total elements to reduce per output element)
//
// keep_dim behavior:
// - If keep_dim=1: reduced dimensions have size 1 in output
// - If keep_dim=0: reduced dimensions are removed from output

// Sum operations (all numeric types)
void sum_f8e4m3(const void *input, void *output, const size_t *metadata);
void sum_f8e5m2(const void *input, void *output, const size_t *metadata);
void sum_bf16(const void *input, void *output, const size_t *metadata);
void sum_f16(const void *input, void *output, const size_t *metadata);
void sum_f32(const void *input, void *output, const size_t *metadata);
void sum_f64(const void *input, void *output, const size_t *metadata);
void sum_i8(const void *input, void *output, const size_t *metadata);
void sum_i16(const void *input, void *output, const size_t *metadata);
void sum_i32(const void *input, void *output, const size_t *metadata);
void sum_i64(const void *input, void *output, const size_t *metadata);
void sum_u8(const void *input, void *output, const size_t *metadata);
void sum_u16(const void *input, void *output, const size_t *metadata);
void sum_u32(const void *input, void *output, const size_t *metadata);
void sum_u64(const void *input, void *output, const size_t *metadata);

// Max operations (all numeric types)
void max_f8e4m3(const void *input, void *output, const size_t *metadata);
void max_f8e5m2(const void *input, void *output, const size_t *metadata);
void max_bf16(const void *input, void *output, const size_t *metadata);
void max_f16(const void *input, void *output, const size_t *metadata);
void max_f32(const void *input, void *output, const size_t *metadata);
void max_f64(const void *input, void *output, const size_t *metadata);
void max_i8(const void *input, void *output, const size_t *metadata);
void max_i16(const void *input, void *output, const size_t *metadata);
void max_i32(const void *input, void *output, const size_t *metadata);
void max_i64(const void *input, void *output, const size_t *metadata);
void max_u8(const void *input, void *output, const size_t *metadata);
void max_u16(const void *input, void *output, const size_t *metadata);
void max_u32(const void *input, void *output, const size_t *metadata);
void max_u64(const void *input, void *output, const size_t *metadata);

// Min operations (all numeric types)
void min_f8e4m3(const void *input, void *output, const size_t *metadata);
void min_f8e5m2(const void *input, void *output, const size_t *metadata);
void min_bf16(const void *input, void *output, const size_t *metadata);
void min_f16(const void *input, void *output, const size_t *metadata);
void min_f32(const void *input, void *output, const size_t *metadata);
void min_f64(const void *input, void *output, const size_t *metadata);
void min_i8(const void *input, void *output, const size_t *metadata);
void min_i16(const void *input, void *output, const size_t *metadata);
void min_i32(const void *input, void *output, const size_t *metadata);
void min_i64(const void *input, void *output, const size_t *metadata);
void min_u8(const void *input, void *output, const size_t *metadata);
void min_u16(const void *input, void *output, const size_t *metadata);
void min_u32(const void *input, void *output, const size_t *metadata);
void min_u64(const void *input, void *output, const size_t *metadata);

// Product operations (all numeric types)
void prod_f8e4m3(const void *input, void *output, const size_t *metadata);
void prod_f8e5m2(const void *input, void *output, const size_t *metadata);
void prod_bf16(const void *input, void *output, const size_t *metadata);
void prod_f16(const void *input, void *output, const size_t *metadata);
void prod_f32(const void *input, void *output, const size_t *metadata);
void prod_f64(const void *input, void *output, const size_t *metadata);
void prod_i8(const void *input, void *output, const size_t *metadata);
void prod_i16(const void *input, void *output, const size_t *metadata);
void prod_i32(const void *input, void *output, const size_t *metadata);
void prod_i64(const void *input, void *output, const size_t *metadata);
void prod_u8(const void *input, void *output, const size_t *metadata);
void prod_u16(const void *input, void *output, const size_t *metadata);
void prod_u32(const void *input, void *output, const size_t *metadata);
void prod_u64(const void *input, void *output, const size_t *metadata);

// Standard deviation operations (population std, float types only)
// Computes: sqrt(E[X²] - E[X]²)
void std_f8e4m3(const void *input, void *output, const size_t *metadata);
void std_f8e5m2(const void *input, void *output, const size_t *metadata);
void std_bf16(const void *input, void *output, const size_t *metadata);
void std_f16(const void *input, void *output, const size_t *metadata);
void std_f32(const void *input, void *output, const size_t *metadata);
void std_f64(const void *input, void *output, const size_t *metadata);

// Variance operations (population variance, float types only)
// Computes: E[X²] - E[X]²
void var_f8e4m3(const void *input, void *output, const size_t *metadata);
void var_f8e5m2(const void *input, void *output, const size_t *metadata);
void var_bf16(const void *input, void *output, const size_t *metadata);
void var_f16(const void *input, void *output, const size_t *metadata);
void var_f32(const void *input, void *output, const size_t *metadata);
void var_f64(const void *input, void *output, const size_t *metadata);

// Mean operations (arithmetic mean, float types only)
void mean_f8e4m3(const void *input, void *output, const size_t *metadata);
void mean_f8e5m2(const void *input, void *output, const size_t *metadata);
void mean_bf16(const void *input, void *output, const size_t *metadata);
void mean_f16(const void *input, void *output, const size_t *metadata);
void mean_f32(const void *input, void *output, const size_t *metadata);
void mean_f64(const void *input, void *output, const size_t *metadata);

// Norm operations (L2 norm, float types only)
// Computes: sqrt(sum(X²))
void norm_f8e4m3(const void *input, void *output, const size_t *metadata);
void norm_f8e5m2(const void *input, void *output, const size_t *metadata);
void norm_bf16(const void *input, void *output, const size_t *metadata);
void norm_f16(const void *input, void *output, const size_t *metadata);
void norm_f32(const void *input, void *output, const size_t *metadata);
void norm_f64(const void *input, void *output, const size_t *metadata);

// ArgMax operations (returns int32 indices, all types including bool)
void argmax_bool(const void *input, void *output, const size_t *metadata);
void argmax_f8e4m3(const void *input, void *output, const size_t *metadata);
void argmax_f8e5m2(const void *input, void *output, const size_t *metadata);
void argmax_bf16(const void *input, void *output, const size_t *metadata);
void argmax_f16(const void *input, void *output, const size_t *metadata);
void argmax_f32(const void *input, void *output, const size_t *metadata);
void argmax_f64(const void *input, void *output, const size_t *metadata);
void argmax_u8(const void *input, void *output, const size_t *metadata);
void argmax_u16(const void *input, void *output, const size_t *metadata);
void argmax_u32(const void *input, void *output, const size_t *metadata);
void argmax_u64(const void *input, void *output, const size_t *metadata);
void argmax_i8(const void *input, void *output, const size_t *metadata);
void argmax_i16(const void *input, void *output, const size_t *metadata);
void argmax_i32(const void *input, void *output, const size_t *metadata);
void argmax_i64(const void *input, void *output, const size_t *metadata);

// ArgMin operations (returns int32 indices, all types including bool)
void argmin_bool(const void *input, void *output, const size_t *metadata);
void argmin_f8e4m3(const void *input, void *output, const size_t *metadata);
void argmin_f8e5m2(const void *input, void *output, const size_t *metadata);
void argmin_bf16(const void *input, void *output, const size_t *metadata);
void argmin_f16(const void *input, void *output, const size_t *metadata);
void argmin_f32(const void *input, void *output, const size_t *metadata);
void argmin_f64(const void *input, void *output, const size_t *metadata);
void argmin_u8(const void *input, void *output, const size_t *metadata);
void argmin_u16(const void *input, void *output, const size_t *metadata);
void argmin_u32(const void *input, void *output, const size_t *metadata);
void argmin_u64(const void *input, void *output, const size_t *metadata);
void argmin_i8(const void *input, void *output, const size_t *metadata);
void argmin_i16(const void *input, void *output, const size_t *metadata);
void argmin_i32(const void *input, void *output, const size_t *metadata);
void argmin_i64(const void *input, void *output, const size_t *metadata);

// Any operations (logical OR, returns bool, all types including bool)
// Returns true if any element is non-zero
void any_bool(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_f8e4m3(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_f8e5m4(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_bf16(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_f16(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_f32(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_f64(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_u8(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_u16(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_u32(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_u64(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_i8(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_i16(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_i32(const void *input_ptr, void *output_ptr, const size_t *metadata);
void any_i64(const void *input_ptr, void *output_ptr, const size_t *metadata);

// All operations (logical AND, returns bool, all types including bool)
// Returns true if all elements are non-zero
void all_bool(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_f8e4m3(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_f8e5m4(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_bf16(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_f16(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_f32(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_f64(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_u8(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_u16(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_u32(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_u64(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_i8(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_i16(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_i32(const void *input_ptr, void *output_ptr, const size_t *metadata);
void all_i64(const void *input_ptr, void *output_ptr, const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // OPS_REDUCE_H
