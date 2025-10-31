#ifndef OPS_WINDOWING_H
#define OPS_WINDOWING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void reduce_window_max_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                              const size_t *metadata);
void reduce_window_max_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                              const size_t *metadata);
void reduce_window_max_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                            const size_t *metadata);
void reduce_window_max_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_max_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_max_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_max_i8(const void *input, void *output, size_t num_els, size_t num_dims,
                          const size_t *metadata);
void reduce_window_max_i16(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_max_i32(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_max_i64(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_max_u8(const void *input, void *output, size_t num_els, size_t num_dims,
                          const size_t *metadata);
void reduce_window_max_u16(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_max_u32(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_max_u64(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);

void reduce_window_mean_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                               const size_t *metadata);
void reduce_window_mean_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                               const size_t *metadata);
void reduce_window_mean_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                             const size_t *metadata);
void reduce_window_mean_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                            const size_t *metadata);
void reduce_window_mean_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                            const size_t *metadata);
void reduce_window_mean_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                            const size_t *metadata);

void reduce_window_sum_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                              const size_t *metadata);
void reduce_window_sum_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                              const size_t *metadata);
void reduce_window_sum_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                            const size_t *metadata);
void reduce_window_sum_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_sum_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_sum_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_sum_i8(const void *input, void *output, size_t num_els, size_t num_dims,
                          const size_t *metadata);
void reduce_window_sum_i16(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_sum_i32(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_sum_i64(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_sum_u8(const void *input, void *output, size_t num_els, size_t num_dims,
                          const size_t *metadata);
void reduce_window_sum_u16(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_sum_u32(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_sum_u64(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);

void reduce_window_min_f8e4m3(const void *input, void *output, size_t num_els, size_t num_dims,
                              const size_t *metadata);
void reduce_window_min_f8e5m2(const void *input, void *output, size_t num_els, size_t num_dims,
                              const size_t *metadata);
void reduce_window_min_bf16(const void *input, void *output, size_t num_els, size_t num_dims,
                            const size_t *metadata);
void reduce_window_min_f16(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_min_f32(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_min_f64(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_min_i8(const void *input, void *output, size_t num_els, size_t num_dims,
                          const size_t *metadata);
void reduce_window_min_i16(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_min_i32(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_min_i64(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_min_u8(const void *input, void *output, size_t num_els, size_t num_dims,
                          const size_t *metadata);
void reduce_window_min_u16(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_min_u32(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);
void reduce_window_min_u64(const void *input, void *output, size_t num_els, size_t num_dims,
                           const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // OPS_WINDOWING_H
