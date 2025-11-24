/**
 * @file ops_conv.h
 * @brief Convolution operations header
 *
 * Provides convolution operations for neural networks including:
 * - Standard convolutions (1D, 2D, 3D)
 * - Transposed convolutions (deconvolutions)
 * - Gradient weight computations for backpropagation
 *
 * All operations support:
 * - Padding: Zero-padding around input borders
 * - Stride: Step size for kernel movement
 * - Dilation: Spacing between kernel elements
 */

#ifndef OPS_CONV_H
#define OPS_CONV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONVOLUTION OPERATIONS
// ============================================================================
//
// All convolution operations follow consistent signatures:
//   void hodu_cpu_convND_type(const void *input, const void *weight, void *output, const size_t
//   *metadata)
//
// Parameters:
//   input    - Pointer to input tensor data
//   weight   - Pointer to convolution kernel/weight tensor
//   output   - Pointer to output buffer (pre-allocated)
//   metadata - Array describing convolution parameters (see layouts below)
//
// Metadata layout for conv1d / conv_transpose1d:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: in_channels
// - metadata[2]: out_channels
// - metadata[3]: in_width
// - metadata[4]: kernel_width
// - metadata[5]: out_width
// - metadata[6]: stride
// - metadata[7]: padding
// - metadata[8]: dilation
// - metadata[9]: input_offset
// - metadata[10]: weight_offset
//
// Metadata layout for conv2d / conv_transpose2d:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: in_channels
// - metadata[2]: out_channels
// - metadata[3]: in_height
// - metadata[4]: in_width
// - metadata[5]: kernel_height
// - metadata[6]: kernel_width
// - metadata[7]: out_height
// - metadata[8]: out_width
// - metadata[9]: stride_h
// - metadata[10]: stride_w
// - metadata[11]: padding_h
// - metadata[12]: padding_w
// - metadata[13]: dilation_h
// - metadata[14]: dilation_w
// - metadata[15]: input_offset
// - metadata[16]: weight_offset
//
// Metadata layout for conv3d / conv_transpose3d:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: in_channels
// - metadata[2]: out_channels
// - metadata[3]: in_depth
// - metadata[4]: in_height
// - metadata[5]: in_width
// - metadata[6]: kernel_depth
// - metadata[7]: kernel_height
// - metadata[8]: kernel_width
// - metadata[9]: out_depth
// - metadata[10]: out_height
// - metadata[11]: out_width
// - metadata[12]: stride_d
// - metadata[13]: stride_h
// - metadata[14]: stride_w
// - metadata[15]: padding_d
// - metadata[16]: padding_h
// - metadata[17]: padding_w
// - metadata[18]: dilation_d
// - metadata[19]: dilation_h
// - metadata[20]: dilation_w
// - metadata[21]: input_offset
// - metadata[22]: weight_offset

// 1D Convolution operations

void hodu_cpu_conv1d_f8e4m3(const void *input, const void *weight, void *output,
                            const size_t *metadata);
void hodu_cpu_conv1d_f8e5m2(const void *input, const void *weight, void *output,
                            const size_t *metadata);
void hodu_cpu_conv1d_bf16(const void *input, const void *weight, void *output,
                          const size_t *metadata);
void hodu_cpu_conv1d_f16(const void *input, const void *weight, void *output,
                         const size_t *metadata);
void hodu_cpu_conv1d_f32(const void *input, const void *weight, void *output,
                         const size_t *metadata);
void hodu_cpu_conv1d_f64(const void *input, const void *weight, void *output,
                         const size_t *metadata);

// 2D Convolution operations
void hodu_cpu_conv2d_f8e4m3(const void *input, const void *weight, void *output,
                            const size_t *metadata);
void hodu_cpu_conv2d_f8e5m2(const void *input, const void *weight, void *output,
                            const size_t *metadata);
void hodu_cpu_conv2d_bf16(const void *input, const void *weight, void *output,
                          const size_t *metadata);
void hodu_cpu_conv2d_f16(const void *input, const void *weight, void *output,
                         const size_t *metadata);
void hodu_cpu_conv2d_f32(const void *input, const void *weight, void *output,
                         const size_t *metadata);
void hodu_cpu_conv2d_f64(const void *input, const void *weight, void *output,
                         const size_t *metadata);

// 3D Convolution operations
void hodu_cpu_conv3d_f8e4m3(const void *input, const void *weight, void *output,
                            const size_t *metadata);
void hodu_cpu_conv3d_f8e5m2(const void *input, const void *weight, void *output,
                            const size_t *metadata);
void hodu_cpu_conv3d_bf16(const void *input, const void *weight, void *output,
                          const size_t *metadata);
void hodu_cpu_conv3d_f16(const void *input, const void *weight, void *output,
                         const size_t *metadata);
void hodu_cpu_conv3d_f32(const void *input, const void *weight, void *output,
                         const size_t *metadata);
void hodu_cpu_conv3d_f64(const void *input, const void *weight, void *output,
                         const size_t *metadata);

// ============================================================================
// TRANSPOSED CONVOLUTION OPERATIONS
// ============================================================================
//
// Transposed convolutions (deconvolutions) follow same metadata layouts
// as their forward counterparts.

// 1D Transposed Convolution operations
void hodu_cpu_conv_transpose1d_f8e4m3(const void *input, const void *weight, void *output,
                                      const size_t *metadata);
void hodu_cpu_conv_transpose1d_f8e5m2(const void *input, const void *weight, void *output,
                                      const size_t *metadata);
void hodu_cpu_conv_transpose1d_bf16(const void *input, const void *weight, void *output,
                                    const size_t *metadata);
void hodu_cpu_conv_transpose1d_f16(const void *input, const void *weight, void *output,
                                   const size_t *metadata);
void hodu_cpu_conv_transpose1d_f32(const void *input, const void *weight, void *output,
                                   const size_t *metadata);
void hodu_cpu_conv_transpose1d_f64(const void *input, const void *weight, void *output,
                                   const size_t *metadata);

// 2D Transposed Convolution operations
void hodu_cpu_conv_transpose2d_f8e4m3(const void *input, const void *weight, void *output,
                                      const size_t *metadata);
void hodu_cpu_conv_transpose2d_f8e5m2(const void *input, const void *weight, void *output,
                                      const size_t *metadata);
void hodu_cpu_conv_transpose2d_bf16(const void *input, const void *weight, void *output,
                                    const size_t *metadata);
void hodu_cpu_conv_transpose2d_f16(const void *input, const void *weight, void *output,
                                   const size_t *metadata);
void hodu_cpu_conv_transpose2d_f32(const void *input, const void *weight, void *output,
                                   const size_t *metadata);
void hodu_cpu_conv_transpose2d_f64(const void *input, const void *weight, void *output,
                                   const size_t *metadata);

// 3D Transposed Convolution operations
void hodu_cpu_conv_transpose3d_f8e4m3(const void *input, const void *weight, void *output,
                                      const size_t *metadata);
void hodu_cpu_conv_transpose3d_f8e5m2(const void *input, const void *weight, void *output,
                                      const size_t *metadata);
void hodu_cpu_conv_transpose3d_bf16(const void *input, const void *weight, void *output,
                                    const size_t *metadata);
void hodu_cpu_conv_transpose3d_f16(const void *input, const void *weight, void *output,
                                   const size_t *metadata);
void hodu_cpu_conv_transpose3d_f32(const void *input, const void *weight, void *output,
                                   const size_t *metadata);
void hodu_cpu_conv_transpose3d_f64(const void *input, const void *weight, void *output,
                                   const size_t *metadata);

// ============================================================================
// CONVOLUTION GRADIENT WEIGHT OPERATIONS
// ============================================================================
//
// Compute gradients with respect to convolution weights for backpropagation.
// Follow same metadata layouts as their forward counterparts.
//
// All gradient operations follow consistent signatures:
//   void hodu_cpu_convND_grad_weight_type(const void *input, const void *grad_output,
//                                void *grad_weight, const size_t *metadata)
//
// Parameters:
//   input       - Pointer to input tensor from forward pass
//   grad_output - Pointer to gradient tensor from next layer
//   grad_weight - Pointer to output gradient weight buffer
//   metadata    - Array describing convolution parameters (same as forward)

// 1D Convolution gradient weight operations
void hodu_cpu_conv1d_grad_weight_f8e4m3(const void *input, const void *grad_output,
                                        void *grad_weight, const size_t *metadata);
void hodu_cpu_conv1d_grad_weight_f8e5m2(const void *input, const void *grad_output,
                                        void *grad_weight, const size_t *metadata);
void hodu_cpu_conv1d_grad_weight_bf16(const void *input, const void *grad_output, void *grad_weight,
                                      const size_t *metadata);
void hodu_cpu_conv1d_grad_weight_f16(const void *input, const void *grad_output, void *grad_weight,
                                     const size_t *metadata);
void hodu_cpu_conv1d_grad_weight_f32(const void *input, const void *grad_output, void *grad_weight,
                                     const size_t *metadata);
void hodu_cpu_conv1d_grad_weight_f64(const void *input, const void *grad_output, void *grad_weight,
                                     const size_t *metadata);

// 2D Convolution gradient weight operations
void hodu_cpu_conv2d_grad_weight_f8e4m3(const void *input, const void *grad_output,
                                        void *grad_weight, const size_t *metadata);
void hodu_cpu_conv2d_grad_weight_f8e5m2(const void *input, const void *grad_output,
                                        void *grad_weight, const size_t *metadata);
void hodu_cpu_conv2d_grad_weight_bf16(const void *input, const void *grad_output, void *grad_weight,
                                      const size_t *metadata);
void hodu_cpu_conv2d_grad_weight_f16(const void *input, const void *grad_output, void *grad_weight,
                                     const size_t *metadata);
void hodu_cpu_conv2d_grad_weight_f32(const void *input, const void *grad_output, void *grad_weight,
                                     const size_t *metadata);
void hodu_cpu_conv2d_grad_weight_f64(const void *input, const void *grad_output, void *grad_weight,
                                     const size_t *metadata);

// 3D Convolution gradient weight operations
void hodu_cpu_conv3d_grad_weight_f8e4m3(const void *input, const void *grad_output,
                                        void *grad_weight, const size_t *metadata);
void hodu_cpu_conv3d_grad_weight_f8e5m2(const void *input, const void *grad_output,
                                        void *grad_weight, const size_t *metadata);
void hodu_cpu_conv3d_grad_weight_bf16(const void *input, const void *grad_output, void *grad_weight,
                                      const size_t *metadata);
void hodu_cpu_conv3d_grad_weight_f16(const void *input, const void *grad_output, void *grad_weight,
                                     const size_t *metadata);
void hodu_cpu_conv3d_grad_weight_f32(const void *input, const void *grad_output, void *grad_weight,
                                     const size_t *metadata);
void hodu_cpu_conv3d_grad_weight_f64(const void *input, const void *grad_output, void *grad_weight,
                                     const size_t *metadata);

// 1D Transposed Convolution gradient weight operations
void hodu_cpu_conv_transpose1d_grad_weight_f8e4m3(const void *input, const void *grad_output,
                                                  void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose1d_grad_weight_f8e5m2(const void *input, const void *grad_output,
                                                  void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose1d_grad_weight_bf16(const void *input, const void *grad_output,
                                                void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose1d_grad_weight_f16(const void *input, const void *grad_output,
                                               void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose1d_grad_weight_f32(const void *input, const void *grad_output,
                                               void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose1d_grad_weight_f64(const void *input, const void *grad_output,
                                               void *grad_weight, const size_t *metadata);

// 2D Transposed Convolution gradient weight operations
void hodu_cpu_conv_transpose2d_grad_weight_f8e4m3(const void *input, const void *grad_output,
                                                  void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose2d_grad_weight_f8e5m2(const void *input, const void *grad_output,
                                                  void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose2d_grad_weight_bf16(const void *input, const void *grad_output,
                                                void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose2d_grad_weight_f16(const void *input, const void *grad_output,
                                               void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose2d_grad_weight_f32(const void *input, const void *grad_output,
                                               void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose2d_grad_weight_f64(const void *input, const void *grad_output,
                                               void *grad_weight, const size_t *metadata);

// 3D Transposed Convolution gradient weight operations
void hodu_cpu_conv_transpose3d_grad_weight_f8e4m3(const void *input, const void *grad_output,
                                                  void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose3d_grad_weight_f8e5m2(const void *input, const void *grad_output,
                                                  void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose3d_grad_weight_bf16(const void *input, const void *grad_output,
                                                void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose3d_grad_weight_f16(const void *input, const void *grad_output,
                                               void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose3d_grad_weight_f32(const void *input, const void *grad_output,
                                               void *grad_weight, const size_t *metadata);
void hodu_cpu_conv_transpose3d_grad_weight_f64(const void *input, const void *grad_output,
                                               void *grad_weight, const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // OPS_CONV_H
