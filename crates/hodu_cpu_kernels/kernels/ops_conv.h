#ifndef OPS_CONV_H
#define OPS_CONV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void conv1d_f8e4m3(const void *input, const void *weight, void *output, size_t num_els,
                   const size_t *metadata);
void conv1d_f8e5m2(const void *input, const void *weight, void *output, size_t num_els,
                   const size_t *metadata);
void conv1d_bf16(const void *input, const void *weight, void *output, size_t num_els,
                 const size_t *metadata);
void conv1d_f16(const void *input, const void *weight, void *output, size_t num_els,
                const size_t *metadata);
void conv1d_f32(const void *input, const void *weight, void *output, size_t num_els,
                const size_t *metadata);
void conv1d_f64(const void *input, const void *weight, void *output, size_t num_els,
                const size_t *metadata);

void conv2d_f8e4m3(const void *input, const void *weight, void *output, size_t num_els,
                   const size_t *metadata);
void conv2d_f8e5m2(const void *input, const void *weight, void *output, size_t num_els,
                   const size_t *metadata);
void conv2d_bf16(const void *input, const void *weight, void *output, size_t num_els,
                 const size_t *metadata);
void conv2d_f16(const void *input, const void *weight, void *output, size_t num_els,
                const size_t *metadata);
void conv2d_f32(const void *input, const void *weight, void *output, size_t num_els,
                const size_t *metadata);
void conv2d_f64(const void *input, const void *weight, void *output, size_t num_els,
                const size_t *metadata);

void conv3d_f8e4m3(const void *input, const void *weight, void *output, size_t num_els,
                   const size_t *metadata);
void conv3d_f8e5m2(const void *input, const void *weight, void *output, size_t num_els,
                   const size_t *metadata);
void conv3d_bf16(const void *input, const void *weight, void *output, size_t num_els,
                 const size_t *metadata);
void conv3d_f16(const void *input, const void *weight, void *output, size_t num_els,
                const size_t *metadata);
void conv3d_f32(const void *input, const void *weight, void *output, size_t num_els,
                const size_t *metadata);
void conv3d_f64(const void *input, const void *weight, void *output, size_t num_els,
                const size_t *metadata);

void conv_transpose1d_f8e4m3(const void *input, const void *weight, void *output, size_t num_els,
                             const size_t *metadata);
void conv_transpose1d_f8e5m2(const void *input, const void *weight, void *output, size_t num_els,
                             const size_t *metadata);
void conv_transpose1d_bf16(const void *input, const void *weight, void *output, size_t num_els,
                           const size_t *metadata);
void conv_transpose1d_f16(const void *input, const void *weight, void *output, size_t num_els,
                          const size_t *metadata);
void conv_transpose1d_f32(const void *input, const void *weight, void *output, size_t num_els,
                          const size_t *metadata);
void conv_transpose1d_f64(const void *input, const void *weight, void *output, size_t num_els,
                          const size_t *metadata);

void conv_transpose2d_f8e4m3(const void *input, const void *weight, void *output, size_t num_els,
                             const size_t *metadata);
void conv_transpose2d_f8e5m2(const void *input, const void *weight, void *output, size_t num_els,
                             const size_t *metadata);
void conv_transpose2d_bf16(const void *input, const void *weight, void *output, size_t num_els,
                           const size_t *metadata);
void conv_transpose2d_f16(const void *input, const void *weight, void *output, size_t num_els,
                          const size_t *metadata);
void conv_transpose2d_f32(const void *input, const void *weight, void *output, size_t num_els,
                          const size_t *metadata);
void conv_transpose2d_f64(const void *input, const void *weight, void *output, size_t num_els,
                          const size_t *metadata);

void conv_transpose3d_f8e4m3(const void *input, const void *weight, void *output, size_t num_els,
                             const size_t *metadata);
void conv_transpose3d_f8e5m2(const void *input, const void *weight, void *output, size_t num_els,
                             const size_t *metadata);
void conv_transpose3d_bf16(const void *input, const void *weight, void *output, size_t num_els,
                           const size_t *metadata);
void conv_transpose3d_f16(const void *input, const void *weight, void *output, size_t num_els,
                          const size_t *metadata);
void conv_transpose3d_f32(const void *input, const void *weight, void *output, size_t num_els,
                          const size_t *metadata);
void conv_transpose3d_f64(const void *input, const void *weight, void *output, size_t num_els,
                          const size_t *metadata);

void conv1d_grad_weight_f8e4m3(const void *input, const void *grad_output, void *grad_weight,
                               size_t num_els, const size_t *metadata);
void conv1d_grad_weight_f8e5m2(const void *input, const void *grad_output, void *grad_weight,
                               size_t num_els, const size_t *metadata);
void conv1d_grad_weight_bf16(const void *input, const void *grad_output, void *grad_weight,
                             size_t num_els, const size_t *metadata);
void conv1d_grad_weight_f16(const void *input, const void *grad_output, void *grad_weight,
                            size_t num_els, const size_t *metadata);
void conv1d_grad_weight_f32(const void *input, const void *grad_output, void *grad_weight,
                            size_t num_els, const size_t *metadata);
void conv1d_grad_weight_f64(const void *input, const void *grad_output, void *grad_weight,
                            size_t num_els, const size_t *metadata);

void conv2d_grad_weight_f8e4m3(const void *input, const void *grad_output, void *grad_weight,
                               size_t num_els, const size_t *metadata);
void conv2d_grad_weight_f8e5m2(const void *input, const void *grad_output, void *grad_weight,
                               size_t num_els, const size_t *metadata);
void conv2d_grad_weight_bf16(const void *input, const void *grad_output, void *grad_weight,
                             size_t num_els, const size_t *metadata);
void conv2d_grad_weight_f16(const void *input, const void *grad_output, void *grad_weight,
                            size_t num_els, const size_t *metadata);
void conv2d_grad_weight_f32(const void *input, const void *grad_output, void *grad_weight,
                            size_t num_els, const size_t *metadata);
void conv2d_grad_weight_f64(const void *input, const void *grad_output, void *grad_weight,
                            size_t num_els, const size_t *metadata);

void conv3d_grad_weight_f8e4m3(const void *input, const void *grad_output, void *grad_weight,
                               size_t num_els, const size_t *metadata);
void conv3d_grad_weight_f8e5m2(const void *input, const void *grad_output, void *grad_weight,
                               size_t num_els, const size_t *metadata);
void conv3d_grad_weight_bf16(const void *input, const void *grad_output, void *grad_weight,
                             size_t num_els, const size_t *metadata);
void conv3d_grad_weight_f16(const void *input, const void *grad_output, void *grad_weight,
                            size_t num_els, const size_t *metadata);
void conv3d_grad_weight_f32(const void *input, const void *grad_output, void *grad_weight,
                            size_t num_els, const size_t *metadata);
void conv3d_grad_weight_f64(const void *input, const void *grad_output, void *grad_weight,
                            size_t num_els, const size_t *metadata);

void conv_transpose1d_grad_weight_f8e4m3(const void *input, const void *grad_output,
                                         void *grad_weight, size_t num_els, const size_t *metadata);
void conv_transpose1d_grad_weight_f8e5m2(const void *input, const void *grad_output,
                                         void *grad_weight, size_t num_els, const size_t *metadata);
void conv_transpose1d_grad_weight_bf16(const void *input, const void *grad_output,
                                       void *grad_weight, size_t num_els, const size_t *metadata);
void conv_transpose1d_grad_weight_f16(const void *input, const void *grad_output, void *grad_weight,
                                      size_t num_els, const size_t *metadata);
void conv_transpose1d_grad_weight_f32(const void *input, const void *grad_output, void *grad_weight,
                                      size_t num_els, const size_t *metadata);
void conv_transpose1d_grad_weight_f64(const void *input, const void *grad_output, void *grad_weight,
                                      size_t num_els, const size_t *metadata);

void conv_transpose2d_grad_weight_f8e4m3(const void *input, const void *grad_output,
                                         void *grad_weight, size_t num_els, const size_t *metadata);
void conv_transpose2d_grad_weight_f8e5m2(const void *input, const void *grad_output,
                                         void *grad_weight, size_t num_els, const size_t *metadata);
void conv_transpose2d_grad_weight_bf16(const void *input, const void *grad_output,
                                       void *grad_weight, size_t num_els, const size_t *metadata);
void conv_transpose2d_grad_weight_f16(const void *input, const void *grad_output, void *grad_weight,
                                      size_t num_els, const size_t *metadata);
void conv_transpose2d_grad_weight_f32(const void *input, const void *grad_output, void *grad_weight,
                                      size_t num_els, const size_t *metadata);
void conv_transpose2d_grad_weight_f64(const void *input, const void *grad_output, void *grad_weight,
                                      size_t num_els, const size_t *metadata);

void conv_transpose3d_grad_weight_f8e4m3(const void *input, const void *grad_output,
                                         void *grad_weight, size_t num_els, const size_t *metadata);
void conv_transpose3d_grad_weight_f8e5m2(const void *input, const void *grad_output,
                                         void *grad_weight, size_t num_els, const size_t *metadata);
void conv_transpose3d_grad_weight_bf16(const void *input, const void *grad_output,
                                       void *grad_weight, size_t num_els, const size_t *metadata);
void conv_transpose3d_grad_weight_f16(const void *input, const void *grad_output, void *grad_weight,
                                      size_t num_els, const size_t *metadata);
void conv_transpose3d_grad_weight_f32(const void *input, const void *grad_output, void *grad_weight,
                                      size_t num_els, const size_t *metadata);
void conv_transpose3d_grad_weight_f64(const void *input, const void *grad_output, void *grad_weight,
                                      size_t num_els, const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // OPS_CONV_H
