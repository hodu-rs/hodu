#include "atomic.h"
#include "ops_conv.h"
#include "types.h"
#include <cblas_new.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Forward declarations for fallback implementations
extern void hodu_cpu_conv2d_f32_fallback(const void *input_ptr, const void *weight_ptr,
                                         void *output_ptr, const size_t *metadata);
extern void hodu_cpu_conv2d_f64_fallback(const void *input_ptr, const void *weight_ptr,
                                         void *output_ptr, const size_t *metadata);
extern void hodu_cpu_conv2d_grad_weight_f32_fallback(const void *input_ptr,
                                                     const void *grad_output_ptr,
                                                     void *grad_weight_ptr, const size_t *metadata);
extern void hodu_cpu_conv2d_grad_weight_f64_fallback(const void *input_ptr,
                                                     const void *grad_output_ptr,
                                                     void *grad_weight_ptr, const size_t *metadata);

// Im2col for f32: Transform image patches to columns for GEMM-based convolution
static inline void im2col_f32(const float *input, float *col_buffer, size_t in_channels,
                              size_t in_height, size_t in_width, size_t kernel_height,
                              size_t kernel_width, size_t out_height, size_t out_width,
                              size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w,
                              size_t dilation_h, size_t dilation_w) {
    size_t col_idx = 0;
    for (size_t oh = 0; oh < out_height; oh++) {
        for (size_t ow = 0; ow < out_width; ow++) {
            for (size_t ic = 0; ic < in_channels; ic++) {
                for (size_t kh = 0; kh < kernel_height; kh++) {
                    for (size_t kw = 0; kw < kernel_width; kw++) {
                        const int ih =
                            (int)(oh * stride_h) - (int)padding_h + (int)(kh * dilation_h);
                        const int iw =
                            (int)(ow * stride_w) - (int)padding_w + (int)(kw * dilation_w);

                        if (ih >= 0 && ih < (int)in_height && iw >= 0 && iw < (int)in_width) {
                            const size_t input_idx = ic * in_height * in_width + ih * in_width + iw;
                            col_buffer[col_idx] = input[input_idx];
                        } else {
                            col_buffer[col_idx] = 0.0f; // Padding
                        }
                        col_idx++;
                    }
                }
            }
        }
    }
}

// Im2col for f64: Transform image patches to columns for GEMM-based convolution
static inline void im2col_f64(const double *input, double *col_buffer, size_t in_channels,
                              size_t in_height, size_t in_width, size_t kernel_height,
                              size_t kernel_width, size_t out_height, size_t out_width,
                              size_t stride_h, size_t stride_w, size_t padding_h, size_t padding_w,
                              size_t dilation_h, size_t dilation_w) {
    size_t col_idx = 0;
    for (size_t oh = 0; oh < out_height; oh++) {
        for (size_t ow = 0; ow < out_width; ow++) {
            for (size_t ic = 0; ic < in_channels; ic++) {
                for (size_t kh = 0; kh < kernel_height; kh++) {
                    for (size_t kw = 0; kw < kernel_width; kw++) {
                        const int ih =
                            (int)(oh * stride_h) - (int)padding_h + (int)(kh * dilation_h);
                        const int iw =
                            (int)(ow * stride_w) - (int)padding_w + (int)(kw * dilation_w);

                        if (ih >= 0 && ih < (int)in_height && iw >= 0 && iw < (int)in_width) {
                            const size_t input_idx = ic * in_height * in_width + ih * in_width + iw;
                            col_buffer[col_idx] = input[input_idx];
                        } else {
                            col_buffer[col_idx] = 0.0; // Padding
                        }
                        col_idx++;
                    }
                }
            }
        }
    }
}

// Accelerate BLAS-optimized conv2d for f32 using im2col + GEMM
void hodu_cpu_conv2d_f32(const void *input_ptr, const void *weight_ptr, void *output_ptr,
                         const size_t *metadata) {
    const float *input = (const float *)input_ptr;
    const float *weight = (const float *)weight_ptr;
    float *output = (float *)output_ptr;

    const size_t batch = metadata[1];
    const size_t in_channels = metadata[2];
    const size_t out_channels = metadata[3];
    const size_t in_height = metadata[4];
    const size_t in_width = metadata[5];
    const size_t kernel_height = metadata[6];
    const size_t kernel_width = metadata[7];
    const size_t out_height = metadata[8];
    const size_t out_width = metadata[9];
    const size_t stride_h = metadata[10];
    const size_t stride_w = metadata[11];
    const size_t padding_h = metadata[12];
    const size_t padding_w = metadata[13];
    const size_t dilation_h = metadata[14];
    const size_t dilation_w = metadata[15];
    const size_t input_offset = metadata[16];
    const size_t weight_offset = metadata[17];

    // Use im2col + GEMM for optimized convolution
    const size_t K = in_channels * kernel_height * kernel_width;
    const size_t N = out_height * out_width;
    const size_t M = out_channels;

    // Allocate column buffer: [K, N]
    float *col_buffer = (float *)malloc(K * N * sizeof(float));
    if (!col_buffer) {
        // Fallback to naive implementation if allocation fails
        hodu_cpu_conv2d_f32_fallback(input_ptr, weight_ptr, output_ptr, metadata);
        return;
    }

    // Process each batch element
    for (size_t b = 0; b < batch; b++) {
        // Im2col: Transform input patches to columns
        const float *batch_input = input + input_offset + b * in_channels * in_height * in_width;
        im2col_f32(batch_input, col_buffer, in_channels, in_height, in_width, kernel_height,
                   kernel_width, out_height, out_width, stride_h, stride_w, padding_h, padding_w,
                   dilation_h, dilation_w);

        // GEMM: output = weight × col_buffer
        float *batch_output = output + b * out_channels * out_height * out_width;
        const float *batch_weight = weight + weight_offset;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, batch_weight, K,
                    col_buffer, N, 0.0f, batch_output, N);
    }

    free(col_buffer);
}

// Accelerate BLAS-optimized conv2d for f64 using im2col + GEMM
void hodu_cpu_conv2d_f64(const void *input_ptr, const void *weight_ptr, void *output_ptr,
                         const size_t *metadata) {
    const double *input = (const double *)input_ptr;
    const double *weight = (const double *)weight_ptr;
    double *output = (double *)output_ptr;

    const size_t batch = metadata[1];
    const size_t in_channels = metadata[2];
    const size_t out_channels = metadata[3];
    const size_t in_height = metadata[4];
    const size_t in_width = metadata[5];
    const size_t kernel_height = metadata[6];
    const size_t kernel_width = metadata[7];
    const size_t out_height = metadata[8];
    const size_t out_width = metadata[9];
    const size_t stride_h = metadata[10];
    const size_t stride_w = metadata[11];
    const size_t padding_h = metadata[12];
    const size_t padding_w = metadata[13];
    const size_t dilation_h = metadata[14];
    const size_t dilation_w = metadata[15];
    const size_t input_offset = metadata[16];
    const size_t weight_offset = metadata[17];

    // Use im2col + GEMM for optimized convolution
    const size_t K = in_channels * kernel_height * kernel_width;
    const size_t N = out_height * out_width;
    const size_t M = out_channels;

    // Allocate column buffer: [K, N]
    double *col_buffer = (double *)malloc(K * N * sizeof(double));
    if (!col_buffer) {
        // Fallback to naive implementation if allocation fails
        hodu_cpu_conv2d_f64_fallback(input_ptr, weight_ptr, output_ptr, metadata);
        return;
    }

    // Process each batch element
    for (size_t b = 0; b < batch; b++) {
        // Im2col: Transform input patches to columns
        const double *batch_input = input + input_offset + b * in_channels * in_height * in_width;
        im2col_f64(batch_input, col_buffer, in_channels, in_height, in_width, kernel_height,
                   kernel_width, out_height, out_width, stride_h, stride_w, padding_h, padding_w,
                   dilation_h, dilation_w);

        // GEMM: output = weight × col_buffer
        double *batch_output = output + b * out_channels * out_height * out_width;
        const double *batch_weight = weight + weight_offset;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, batch_weight, K,
                    col_buffer, N, 0.0, batch_output, N);
    }

    free(col_buffer);
}

// Accelerate BLAS-optimized conv2d_grad_weight for f32 using im2col + GEMM
void hodu_cpu_conv2d_grad_weight_f32(const void *input_ptr, const void *grad_output_ptr,
                                     void *grad_weight_ptr, const size_t *metadata) {
    const float *input = (const float *)input_ptr;
    const float *grad_output = (const float *)grad_output_ptr;
    float *grad_weight = (float *)grad_weight_ptr;

    // Parse generic metadata: Conv2D has input_ndim=4, spatial_dims=2
    const size_t num_els = metadata[0];
    const size_t input_ndim = metadata[1];
    const size_t spatial_dims = metadata[2];
    const size_t batch = metadata[3];
    const size_t in_channels = metadata[4];
    const size_t in_height = metadata[5];
    const size_t in_width = metadata[6];
    const size_t grad_output_base = 3 + input_ndim;
    const size_t out_channels = metadata[grad_output_base + 1];
    const size_t out_height = metadata[grad_output_base + 2];
    const size_t out_width = metadata[grad_output_base + 3];
    const size_t weight_base = 3 + 2 * input_ndim;
    const size_t kernel_height = metadata[weight_base + 2];
    const size_t kernel_width = metadata[weight_base + 3];
    const size_t grad_output_stride_base = 3 + 4 * input_ndim;
    const size_t grad_output_stride_batch = metadata[grad_output_stride_base];
    const size_t grad_output_stride_channel = metadata[grad_output_stride_base + 1];
    const size_t grad_output_stride_h = metadata[grad_output_stride_base + 2];
    const size_t grad_output_stride_w = metadata[grad_output_stride_base + 3];
    const size_t offsets_base = 3 + 5 * input_ndim;
    const size_t input_offset = metadata[offsets_base];
    const size_t grad_output_offset = metadata[offsets_base + 1];
    const size_t conv_params_base = offsets_base + 2;
    const size_t stride_h = metadata[conv_params_base];
    const size_t stride_w = metadata[conv_params_base + 1];
    const size_t padding_h = metadata[conv_params_base + spatial_dims];
    const size_t padding_w = metadata[conv_params_base + spatial_dims + 1];
    const size_t dilation_h = metadata[conv_params_base + 2 * spatial_dims];
    const size_t dilation_w = metadata[conv_params_base + 2 * spatial_dims + 1];

    // Check if grad_output is contiguous (required for im2col + GEMM approach)
    const size_t expected_stride_w = 1;
    const size_t expected_stride_h = out_width;
    const size_t expected_stride_channel = out_height * out_width;
    const size_t expected_stride_batch = out_channels * out_height * out_width;

    if (grad_output_stride_w != expected_stride_w || grad_output_stride_h != expected_stride_h ||
        grad_output_stride_channel != expected_stride_channel ||
        grad_output_stride_batch != expected_stride_batch) {
        // grad_output is not contiguous (e.g., broadcasted), use fallback
        hodu_cpu_conv2d_grad_weight_f32_fallback(input_ptr, grad_output_ptr, grad_weight_ptr,
                                                 metadata);
        return;
    }

    memset(grad_weight, 0, num_els * sizeof(float));

    // Use im2col + GEMM for optimized gradient computation
    const size_t K = in_channels * kernel_height * kernel_width;
    const size_t N = out_height * out_width;
    const size_t M = out_channels;

    // Allocate column buffer: [K, N]
    float *col_buffer = (float *)malloc(K * N * sizeof(float));
    if (!col_buffer) {
        hodu_cpu_conv2d_grad_weight_f32_fallback(input_ptr, grad_output_ptr, grad_weight_ptr,
                                                 metadata);
        return;
    }

    // Process each batch element and accumulate gradients
    for (size_t b = 0; b < batch; b++) {
        // Im2col: Transform input patches to columns
        const float *batch_input = input + input_offset + b * in_channels * in_height * in_width;
        im2col_f32(batch_input, col_buffer, in_channels, in_height, in_width, kernel_height,
                   kernel_width, out_height, out_width, stride_h, stride_w, padding_h, padding_w,
                   dilation_h, dilation_w);

        // GEMM: grad_weight += grad_output × col_buffer^T
        const float *batch_grad_output =
            grad_output + grad_output_offset + b * out_channels * out_height * out_width;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, K, N, 1.0f, batch_grad_output, N,
                    col_buffer, N, 1.0f, grad_weight, K);
    }

    free(col_buffer);
}

// Accelerate BLAS-optimized conv2d_grad_weight for f64 using im2col + GEMM
void hodu_cpu_conv2d_grad_weight_f64(const void *input_ptr, const void *grad_output_ptr,
                                     void *grad_weight_ptr, const size_t *metadata) {
    const double *input = (const double *)input_ptr;
    const double *grad_output = (const double *)grad_output_ptr;
    double *grad_weight = (double *)grad_weight_ptr;

    // Parse generic metadata: Conv2D has input_ndim=4, spatial_dims=2
    const size_t num_els = metadata[0];
    const size_t input_ndim = metadata[1];
    const size_t spatial_dims = metadata[2];
    const size_t batch = metadata[3];
    const size_t in_channels = metadata[4];
    const size_t in_height = metadata[5];
    const size_t in_width = metadata[6];
    const size_t grad_output_base = 3 + input_ndim;
    const size_t out_channels = metadata[grad_output_base + 1];
    const size_t out_height = metadata[grad_output_base + 2];
    const size_t out_width = metadata[grad_output_base + 3];
    const size_t weight_base = 3 + 2 * input_ndim;
    const size_t kernel_height = metadata[weight_base + 2];
    const size_t kernel_width = metadata[weight_base + 3];
    const size_t grad_output_stride_base = 3 + 4 * input_ndim;
    const size_t grad_output_stride_batch = metadata[grad_output_stride_base];
    const size_t grad_output_stride_channel = metadata[grad_output_stride_base + 1];
    const size_t grad_output_stride_h = metadata[grad_output_stride_base + 2];
    const size_t grad_output_stride_w = metadata[grad_output_stride_base + 3];
    const size_t offsets_base = 3 + 5 * input_ndim;
    const size_t input_offset = metadata[offsets_base];
    const size_t grad_output_offset = metadata[offsets_base + 1];
    const size_t conv_params_base = offsets_base + 2;
    const size_t stride_h = metadata[conv_params_base];
    const size_t stride_w = metadata[conv_params_base + 1];
    const size_t padding_h = metadata[conv_params_base + spatial_dims];
    const size_t padding_w = metadata[conv_params_base + spatial_dims + 1];
    const size_t dilation_h = metadata[conv_params_base + 2 * spatial_dims];
    const size_t dilation_w = metadata[conv_params_base + 2 * spatial_dims + 1];

    // Check if grad_output is contiguous (required for im2col + GEMM approach)
    const size_t expected_stride_w = 1;
    const size_t expected_stride_h = out_width;
    const size_t expected_stride_channel = out_height * out_width;
    const size_t expected_stride_batch = out_channels * out_height * out_width;

    if (grad_output_stride_w != expected_stride_w || grad_output_stride_h != expected_stride_h ||
        grad_output_stride_channel != expected_stride_channel ||
        grad_output_stride_batch != expected_stride_batch) {
        // grad_output is not contiguous (e.g., broadcasted), use fallback
        hodu_cpu_conv2d_grad_weight_f64_fallback(input_ptr, grad_output_ptr, grad_weight_ptr,
                                                 metadata);
        return;
    }

    memset(grad_weight, 0, num_els * sizeof(double));

    // Use im2col + GEMM for optimized gradient computation
    const size_t K = in_channels * kernel_height * kernel_width;
    const size_t N = out_height * out_width;
    const size_t M = out_channels;

    // Allocate column buffer: [K, N]
    double *col_buffer = (double *)malloc(K * N * sizeof(double));
    if (!col_buffer) {
        hodu_cpu_conv2d_grad_weight_f64_fallback(input_ptr, grad_output_ptr, grad_weight_ptr,
                                                 metadata);
        return;
    }

    // Process each batch element and accumulate gradients
    for (size_t b = 0; b < batch; b++) {
        // Im2col: Transform input patches to columns
        const double *batch_input = input + input_offset + b * in_channels * in_height * in_width;
        im2col_f64(batch_input, col_buffer, in_channels, in_height, in_width, kernel_height,
                   kernel_width, out_height, out_width, stride_h, stride_w, padding_h, padding_w,
                   dilation_h, dilation_w);

        // GEMM: grad_weight += grad_output × col_buffer^T
        const double *batch_grad_output =
            grad_output + grad_output_offset + b * out_channels * out_height * out_width;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, K, N, 1.0, batch_grad_output, N,
                    col_buffer, N, 1.0, grad_weight, K);
    }

    free(col_buffer);
}
