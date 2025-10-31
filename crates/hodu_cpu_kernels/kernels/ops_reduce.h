#ifndef OPS_REDUCE_H
#define OPS_REDUCE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Reduce operations
// metadata layout: [dims..., strides..., offset, output_shape_len, output_shape...,
//                   num_reduce_dims, reduce_dims..., keep_dim]

// Sum operations
void reduce_sum_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_sum_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_sum_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_sum_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_sum_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_sum_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_sum_i8(const void *input, void *output, size_t num_els, size_t num_dims,
                   const size_t *metadata, size_t reduce_size);
void reduce_sum_i16(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_sum_i32(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_sum_i64(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_sum_u8(const void *input, void *output, size_t num_els, size_t num_dims,
                   const size_t *metadata, size_t reduce_size);
void reduce_sum_u16(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_sum_u32(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_sum_u64(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);

// Max operations
void reduce_max_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_max_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_max_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_max_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_max_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_max_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_max_i8(const void *input, void *output, size_t num_els, size_t num_dims,
                   const size_t *metadata, size_t reduce_size);
void reduce_max_i16(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_max_i32(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_max_i64(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_max_u8(const void *input, void *output, size_t num_els, size_t num_dims,
                   const size_t *metadata, size_t reduce_size);
void reduce_max_u16(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_max_u32(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_max_u64(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);

// Min operations
void reduce_min_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_min_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_min_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_min_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_min_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_min_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_min_i8(const void *input, void *output, size_t num_els, size_t num_dims,
                   const size_t *metadata, size_t reduce_size);
void reduce_min_i16(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_min_i32(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_min_i64(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_min_u8(const void *input, void *output, size_t num_els, size_t num_dims,
                   const size_t *metadata, size_t reduce_size);
void reduce_min_u16(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_min_u32(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_min_u64(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);

// Product operations
void reduce_prod_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                        const size_t *metadata, size_t reduce_size);
void reduce_prod_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                        const size_t *metadata, size_t reduce_size);
void reduce_prod_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                      const size_t *metadata, size_t reduce_size);
void reduce_prod_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_prod_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_prod_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_prod_i8(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_prod_i16(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_prod_i32(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_prod_i64(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_prod_u8(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_prod_u16(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_prod_u32(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_prod_u64(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);

// Std operations (standard deviation, float types only)
void reduce_std_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_std_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_std_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_std_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_std_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_std_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);

// Var operations (variance, float types only)
void reduce_var_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_var_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_var_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_var_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_var_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_var_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);

// Mean operations (float types only)
void reduce_mean_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                        const size_t *metadata, size_t reduce_size);
void reduce_mean_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                        const size_t *metadata, size_t reduce_size);
void reduce_mean_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                      const size_t *metadata, size_t reduce_size);
void reduce_mean_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_mean_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_mean_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);

// ArgMax operations
void reduce_argmax_bool(const void *input, void *output, size_t num_els, size_t num_dims,
                        const size_t *metadata, size_t reduce_size);
void reduce_argmax_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                          const size_t *metadata, size_t reduce_size);
void reduce_argmax_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                          const size_t *metadata, size_t reduce_size);
void reduce_argmax_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                        const size_t *metadata, size_t reduce_size);
void reduce_argmax_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmax_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmax_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmax_u8(const void *input, void *output, size_t num_els, size_t num_dims,
                      const size_t *metadata, size_t reduce_size);
void reduce_argmax_u16(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmax_u32(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmax_u64(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmax_i8(const void *input, void *output, size_t num_els, size_t num_dims,
                      const size_t *metadata, size_t reduce_size);
void reduce_argmax_i16(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmax_i32(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmax_i64(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);

// ArgMin operations
void reduce_argmin_bool(const void *input, void *output, size_t num_els, size_t num_dims,
                        const size_t *metadata, size_t reduce_size);
void reduce_argmin_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                          const size_t *metadata, size_t reduce_size);
void reduce_argmin_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                          const size_t *metadata, size_t reduce_size);
void reduce_argmin_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                        const size_t *metadata, size_t reduce_size);
void reduce_argmin_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmin_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmin_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmin_u8(const void *input, void *output, size_t num_els, size_t num_dims,
                      const size_t *metadata, size_t reduce_size);
void reduce_argmin_u16(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmin_u32(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmin_u64(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmin_i8(const void *input, void *output, size_t num_els, size_t num_dims,
                      const size_t *metadata, size_t reduce_size);
void reduce_argmin_i16(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmin_i32(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_argmin_i64(const void *input, void *output, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);

// Any operations
void reduce_any_bool(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_any_f8e4m3(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_any_f8e5m4(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_any_bf16(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_any_f16(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_any_f32(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_any_f64(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_any_u8(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                   const size_t *metadata, size_t reduce_size);
void reduce_any_u16(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_any_u32(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_any_u64(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_any_i8(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                   const size_t *metadata, size_t reduce_size);
void reduce_any_i16(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_any_i32(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_any_i64(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);

// All operations
void reduce_all_bool(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_all_f8e4m3(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_all_f8e5m4(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                       const size_t *metadata, size_t reduce_size);
void reduce_all_bf16(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                     const size_t *metadata, size_t reduce_size);
void reduce_all_f16(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_all_f32(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_all_f64(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_all_u8(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                   const size_t *metadata, size_t reduce_size);
void reduce_all_u16(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_all_u32(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_all_u64(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_all_i8(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                   const size_t *metadata, size_t reduce_size);
void reduce_all_i16(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_all_i32(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);
void reduce_all_i64(const void *input_ptr, void *output_ptr, size_t num_els, size_t num_dims,
                    const size_t *metadata, size_t reduce_size);

#ifdef __cplusplus
}
#endif

#endif // OPS_REDUCE_H
