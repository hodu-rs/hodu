#include "ops_conv.h"
#include "atomic.h"
#include "types.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_BLAS
#include <cblas.h>
#endif

// ============================================================================
// 1D CONVOLUTION OPERATIONS
// ============================================================================
//
// Metadata layout for conv1d:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: batch
// - metadata[2]: in_channels
// - metadata[3]: out_channels
// - metadata[4]: in_width
// - metadata[5]: kernel_width
// - metadata[6]: out_width
// - metadata[7]: stride
// - metadata[8]: padding
// - metadata[9]: dilation
// - metadata[10]: input_offset
// - metadata[11]: weight_offset

/// Macro to implement 1D convolution operation
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
#define CONV1D_OP(TYPE, TYPE_SUFFIX)                                                               \
    void conv1d_##TYPE_SUFFIX(const void *input_ptr, const void *weight_ptr, void *output_ptr,     \
                              const size_t *metadata) {                                            \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *weight = (const TYPE *)weight_ptr;                                             \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t batch = metadata[1];                                                          \
        const size_t in_channels = metadata[2];                                                    \
        const size_t out_channels = metadata[3];                                                   \
        const size_t in_width = metadata[4];                                                       \
        const size_t kernel_width = metadata[5];                                                   \
        const size_t out_width = metadata[6];                                                      \
        const size_t stride = metadata[7];                                                         \
        const size_t padding = metadata[8];                                                        \
        const size_t dilation = metadata[9];                                                       \
        const size_t input_offset = metadata[10];                                                  \
        const size_t weight_offset = metadata[11];                                                 \
                                                                                                   \
        (void)batch; /* Batch is computed from idx, kept for API consistency */                    \
                                                                                                   \
        for (size_t idx = 0; idx < num_els; idx++) {                                               \
            const size_t ow = idx % out_width;                                                     \
            const size_t oc = (idx / out_width) % out_channels;                                    \
            const size_t b = idx / (out_width * out_channels);                                     \
                                                                                                   \
            TYPE sum = 0;                                                                          \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t kw = 0; kw < kernel_width; kw++) {                                     \
                    const int iw = (int)(ow * stride) - (int)padding + (int)(kw * dilation);       \
                    if (iw >= 0 && iw < (int)in_width) {                                           \
                        const size_t input_idx =                                                   \
                            input_offset + b * in_channels * in_width + ic * in_width + iw;        \
                        const size_t weight_idx = weight_offset +                                  \
                                                  oc * in_channels * kernel_width +                \
                                                  ic * kernel_width + kw;                          \
                        sum += input[input_idx] * weight[weight_idx];                              \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            output[idx] = sum;                                                                     \
        }                                                                                          \
    }

CONV1D_OP(f8e4m3_t, f8e4m3)
CONV1D_OP(f8e5m2_t, f8e5m2)
CONV1D_OP(bf16_t, bf16)
CONV1D_OP(f16_t, f16)
CONV1D_OP(float, f32)
CONV1D_OP(double, f64)

// ============================================================================
// IM2COL HELPER FUNCTIONS FOR 2D CONVOLUTION
// ============================================================================

#ifdef USE_BLAS

// Im2col for f32: Transform image patches to columns for GEMM-based convolution
// col_buffer shape: [in_channels * kernel_height * kernel_width, out_height * out_width]
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

#endif

// ============================================================================
// 2D CONVOLUTION OPERATIONS
// ============================================================================
//
// Metadata layout for conv2d:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: batch
// - metadata[2]: in_channels
// - metadata[3]: out_channels
// - metadata[4]: in_height
// - metadata[5]: in_width
// - metadata[6]: kernel_height
// - metadata[7]: kernel_width
// - metadata[8]: out_height
// - metadata[9]: out_width
// - metadata[10]: stride_h
// - metadata[11]: stride_w
// - metadata[12]: padding_h
// - metadata[13]: padding_w
// - metadata[14]: dilation_h
// - metadata[15]: dilation_w
// - metadata[16]: input_offset
// - metadata[17]: weight_offset

/// Macro to implement 2D convolution operation
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
#define CONV2D_OP(TYPE, TYPE_SUFFIX)                                                               \
    void conv2d_##TYPE_SUFFIX(const void *input_ptr, const void *weight_ptr, void *output_ptr,     \
                              const size_t *metadata) {                                            \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *weight = (const TYPE *)weight_ptr;                                             \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t batch = metadata[1];                                                          \
        const size_t in_channels = metadata[2];                                                    \
        const size_t out_channels = metadata[3];                                                   \
        const size_t in_height = metadata[4];                                                      \
        const size_t in_width = metadata[5];                                                       \
        const size_t kernel_height = metadata[6];                                                  \
        const size_t kernel_width = metadata[7];                                                   \
        const size_t out_height = metadata[8];                                                     \
        const size_t out_width = metadata[9];                                                      \
        const size_t stride_h = metadata[10];                                                      \
        const size_t stride_w = metadata[11];                                                      \
        const size_t padding_h = metadata[12];                                                     \
        const size_t padding_w = metadata[13];                                                     \
        const size_t dilation_h = metadata[14];                                                    \
        const size_t dilation_w = metadata[15];                                                    \
        const size_t input_offset = metadata[16];                                                  \
        const size_t weight_offset = metadata[17];                                                 \
                                                                                                   \
        (void)batch; /* Batch is computed from idx, kept for API consistency */                    \
                                                                                                   \
        for (size_t idx = 0; idx < num_els; idx++) {                                               \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t oc = (idx / (out_width * out_height)) % out_channels;                     \
            const size_t b = idx / (out_width * out_height * out_channels);                        \
                                                                                                   \
            TYPE sum = 0;                                                                          \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t kh = 0; kh < kernel_height; kh++) {                                    \
                    for (size_t kw = 0; kw < kernel_width; kw++) {                                 \
                        const int ih =                                                             \
                            (int)(oh * stride_h) - (int)padding_h + (int)(kh * dilation_h);        \
                        const int iw =                                                             \
                            (int)(ow * stride_w) - (int)padding_w + (int)(kw * dilation_w);        \
                        if (ih >= 0 && ih < (int)in_height && iw >= 0 && iw < (int)in_width) {     \
                            const size_t input_idx =                                               \
                                input_offset + b * in_channels * in_height * in_width +            \
                                ic * in_height * in_width + ih * in_width + iw;                    \
                            const size_t weight_idx =                                              \
                                weight_offset + oc * in_channels * kernel_height * kernel_width +  \
                                ic * kernel_height * kernel_width + kh * kernel_width + kw;        \
                            sum += input[input_idx] * weight[weight_idx];                          \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            output[idx] = sum;                                                                     \
        }                                                                                          \
    }

CONV2D_OP(f8e4m3_t, f8e4m3)
CONV2D_OP(f8e5m2_t, f8e5m2)
CONV2D_OP(bf16_t, bf16)
CONV2D_OP(f16_t, f16)

// BLAS-optimized conv2d for f32 using im2col + GEMM
void conv2d_f32(const void *input_ptr, const void *weight_ptr, void *output_ptr,
                const size_t *metadata) {
    const float *input = (const float *)input_ptr;
    const float *weight = (const float *)weight_ptr;
    float *output = (float *)output_ptr;

    const size_t num_els = metadata[0];
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

#ifdef USE_BLAS
    // Use im2col + GEMM for optimized convolution
    const size_t K = in_channels * kernel_height * kernel_width;
    const size_t N = out_height * out_width;
    const size_t M = out_channels;

    // Allocate column buffer: [K, N]
    float *col_buffer = (float *)malloc(K * N * sizeof(float));
    if (!col_buffer) {
        // Fallback to naive implementation if allocation fails
        goto fallback_f32;
    }

    // Process each batch element
    for (size_t b = 0; b < batch; b++) {
        // Im2col: Transform input patches to columns
        const float *batch_input = input + input_offset + b * in_channels * in_height * in_width;
        im2col_f32(batch_input, col_buffer, in_channels, in_height, in_width, kernel_height,
                   kernel_width, out_height, out_width, stride_h, stride_w, padding_h, padding_w,
                   dilation_h, dilation_w);

        // GEMM: output = weight × col_buffer
        // weight shape: [M, K] (out_channels, in_channels * kh * kw)
        // col_buffer shape: [K, N] (in_channels * kh * kw, out_height * out_width)
        // output shape: [M, N] (out_channels, out_height * out_width)
        float *batch_output = output + b * out_channels * out_height * out_width;
        const float *batch_weight = weight + weight_offset;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, batch_weight, K,
                    col_buffer, N, 0.0f, batch_output, N);
    }

    free(col_buffer);
    return;

fallback_f32:
#endif
    // Fallback: Naive implementation (same as macro-generated code)
    (void)batch;
    for (size_t idx = 0; idx < num_els; idx++) {
        const size_t ow = idx % out_width;
        const size_t oh = (idx / out_width) % out_height;
        const size_t oc = (idx / (out_width * out_height)) % out_channels;
        const size_t b = idx / (out_width * out_height * out_channels);

        float sum = 0;
        for (size_t ic = 0; ic < in_channels; ic++) {
            for (size_t kh = 0; kh < kernel_height; kh++) {
                for (size_t kw = 0; kw < kernel_width; kw++) {
                    const int ih = (int)(oh * stride_h) - (int)padding_h + (int)(kh * dilation_h);
                    const int iw = (int)(ow * stride_w) - (int)padding_w + (int)(kw * dilation_w);
                    if (ih >= 0 && ih < (int)in_height && iw >= 0 && iw < (int)in_width) {
                        const size_t input_idx = input_offset +
                                                 b * in_channels * in_height * in_width +
                                                 ic * in_height * in_width + ih * in_width + iw;
                        const size_t weight_idx =
                            weight_offset + oc * in_channels * kernel_height * kernel_width +
                            ic * kernel_height * kernel_width + kh * kernel_width + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

// BLAS-optimized conv2d for f64 using im2col + GEMM
void conv2d_f64(const void *input_ptr, const void *weight_ptr, void *output_ptr,
                const size_t *metadata) {
    const double *input = (const double *)input_ptr;
    const double *weight = (const double *)weight_ptr;
    double *output = (double *)output_ptr;

    const size_t num_els = metadata[0];
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

#ifdef USE_BLAS
    // Use im2col + GEMM for optimized convolution
    const size_t K = in_channels * kernel_height * kernel_width;
    const size_t N = out_height * out_width;
    const size_t M = out_channels;

    // Allocate column buffer: [K, N]
    double *col_buffer = (double *)malloc(K * N * sizeof(double));
    if (!col_buffer) {
        // Fallback to naive implementation if allocation fails
        goto fallback_f64;
    }

    // Process each batch element
    for (size_t b = 0; b < batch; b++) {
        // Im2col: Transform input patches to columns
        const double *batch_input = input + input_offset + b * in_channels * in_height * in_width;
        im2col_f64(batch_input, col_buffer, in_channels, in_height, in_width, kernel_height,
                   kernel_width, out_height, out_width, stride_h, stride_w, padding_h, padding_w,
                   dilation_h, dilation_w);

        // GEMM: output = weight × col_buffer
        // weight shape: [M, K] (out_channels, in_channels * kh * kw)
        // col_buffer shape: [K, N] (in_channels * kh * kw, out_height * out_width)
        // output shape: [M, N] (out_channels, out_height * out_width)
        double *batch_output = output + b * out_channels * out_height * out_width;
        const double *batch_weight = weight + weight_offset;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, batch_weight, K,
                    col_buffer, N, 0.0, batch_output, N);
    }

    free(col_buffer);
    return;

fallback_f64:
#endif
    // Fallback: Naive implementation (same as macro-generated code)
    (void)batch;
    for (size_t idx = 0; idx < num_els; idx++) {
        const size_t ow = idx % out_width;
        const size_t oh = (idx / out_width) % out_height;
        const size_t oc = (idx / (out_width * out_height)) % out_channels;
        const size_t b = idx / (out_width * out_height * out_channels);

        double sum = 0;
        for (size_t ic = 0; ic < in_channels; ic++) {
            for (size_t kh = 0; kh < kernel_height; kh++) {
                for (size_t kw = 0; kw < kernel_width; kw++) {
                    const int ih = (int)(oh * stride_h) - (int)padding_h + (int)(kh * dilation_h);
                    const int iw = (int)(ow * stride_w) - (int)padding_w + (int)(kw * dilation_w);
                    if (ih >= 0 && ih < (int)in_height && iw >= 0 && iw < (int)in_width) {
                        const size_t input_idx = input_offset +
                                                 b * in_channels * in_height * in_width +
                                                 ic * in_height * in_width + ih * in_width + iw;
                        const size_t weight_idx =
                            weight_offset + oc * in_channels * kernel_height * kernel_width +
                            ic * kernel_height * kernel_width + kh * kernel_width + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

// ============================================================================
// 3D CONVOLUTION OPERATIONS
// ============================================================================
//
// Metadata layout for conv3d:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: batch
// - metadata[2]: in_channels
// - metadata[3]: out_channels
// - metadata[4]: in_depth
// - metadata[5]: in_height
// - metadata[6]: in_width
// - metadata[7]: kernel_depth
// - metadata[8]: kernel_height
// - metadata[9]: kernel_width
// - metadata[10]: out_depth
// - metadata[11]: out_height
// - metadata[12]: out_width
// - metadata[13]: stride_d
// - metadata[14]: stride_h
// - metadata[15]: stride_w
// - metadata[16]: padding_d
// - metadata[17]: padding_h
// - metadata[18]: padding_w
// - metadata[19]: dilation_d
// - metadata[20]: dilation_h
// - metadata[21]: dilation_w
// - metadata[22]: input_offset
// - metadata[23]: weight_offset

/// Macro to implement 3D convolution operation
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
#define CONV3D_OP(TYPE, TYPE_SUFFIX)                                                               \
    void conv3d_##TYPE_SUFFIX(const void *input_ptr, const void *weight_ptr, void *output_ptr,     \
                              const size_t *metadata) {                                            \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *weight = (const TYPE *)weight_ptr;                                             \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t batch = metadata[1];                                                          \
        const size_t in_channels = metadata[2];                                                    \
        const size_t out_channels = metadata[3];                                                   \
        const size_t in_depth = metadata[4];                                                       \
        const size_t in_height = metadata[5];                                                      \
        const size_t in_width = metadata[6];                                                       \
        const size_t kernel_depth = metadata[7];                                                   \
        const size_t kernel_height = metadata[8];                                                  \
        const size_t kernel_width = metadata[9];                                                   \
        const size_t out_depth = metadata[10];                                                     \
        const size_t out_height = metadata[11];                                                    \
        const size_t out_width = metadata[12];                                                     \
        const size_t stride_d = metadata[13];                                                      \
        const size_t stride_h = metadata[14];                                                      \
        const size_t stride_w = metadata[15];                                                      \
        const size_t padding_d = metadata[16];                                                     \
        const size_t padding_h = metadata[17];                                                     \
        const size_t padding_w = metadata[18];                                                     \
        const size_t dilation_d = metadata[19];                                                    \
        const size_t dilation_h = metadata[20];                                                    \
        const size_t dilation_w = metadata[21];                                                    \
        const size_t input_offset = metadata[22];                                                  \
        const size_t weight_offset = metadata[23];                                                 \
                                                                                                   \
        (void)batch; /* Batch is computed from idx, kept for API consistency */                    \
                                                                                                   \
        for (size_t idx = 0; idx < num_els; idx++) {                                               \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t od = (idx / (out_width * out_height)) % out_depth;                        \
            const size_t oc = (idx / (out_width * out_height * out_depth)) % out_channels;         \
            const size_t b = idx / (out_width * out_height * out_depth * out_channels);            \
                                                                                                   \
            TYPE sum = 0;                                                                          \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t kd = 0; kd < kernel_depth; kd++) {                                     \
                    for (size_t kh = 0; kh < kernel_height; kh++) {                                \
                        for (size_t kw = 0; kw < kernel_width; kw++) {                             \
                            const int id =                                                         \
                                (int)(od * stride_d) - (int)padding_d + (int)(kd * dilation_d);    \
                            const int ih =                                                         \
                                (int)(oh * stride_h) - (int)padding_h + (int)(kh * dilation_h);    \
                            const int iw =                                                         \
                                (int)(ow * stride_w) - (int)padding_w + (int)(kw * dilation_w);    \
                            if (id >= 0 && id < (int)in_depth && ih >= 0 && ih < (int)in_height && \
                                iw >= 0 && iw < (int)in_width) {                                   \
                                const size_t input_idx =                                           \
                                    input_offset +                                                 \
                                    b * in_channels * in_depth * in_height * in_width +            \
                                    ic * in_depth * in_height * in_width +                         \
                                    id * in_height * in_width + ih * in_width + iw;                \
                                const size_t weight_idx =                                          \
                                    weight_offset +                                                \
                                    oc * in_channels * kernel_depth * kernel_height *              \
                                        kernel_width +                                             \
                                    ic * kernel_depth * kernel_height * kernel_width +             \
                                    kd * kernel_height * kernel_width + kh * kernel_width + kw;    \
                                sum += input[input_idx] * weight[weight_idx];                      \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            output[idx] = sum;                                                                     \
        }                                                                                          \
    }

CONV3D_OP(f8e4m3_t, f8e4m3)
CONV3D_OP(f8e5m2_t, f8e5m2)
CONV3D_OP(bf16_t, bf16)
CONV3D_OP(f16_t, f16)
CONV3D_OP(float, f32)
CONV3D_OP(double, f64)

// ============================================================================
// 1D TRANSPOSED CONVOLUTION OPERATIONS
// ============================================================================
//
// Transposed convolution (deconvolution) is the inverse of convolution,
// commonly used in decoder networks and generative models.
//
// Metadata layout for conv_transpose1d: Same as conv1d
// output_padding is used only for calculating out_width at the Rust level,
// not needed in the kernel as it's already reflected in out_width

/// Macro to implement 1D transposed convolution operation
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
#define CONV_TRANSPOSE1D_OP(TYPE, TYPE_SUFFIX)                                                     \
    void conv_transpose1d_##TYPE_SUFFIX(const void *input_ptr, const void *weight_ptr,             \
                                        void *output_ptr, const size_t *metadata) {                \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *weight = (const TYPE *)weight_ptr;                                             \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
                                                                                                   \
        memset(output, 0, num_els * sizeof(TYPE));                                                 \
                                                                                                   \
        const size_t batch = metadata[1];                                                          \
        const size_t in_channels = metadata[2];                                                    \
        const size_t out_channels = metadata[3];                                                   \
        const size_t in_width = metadata[4];                                                       \
        const size_t kernel_width = metadata[5];                                                   \
        const size_t out_width = metadata[6];                                                      \
        const size_t stride = metadata[7];                                                         \
        const size_t padding = metadata[8];                                                        \
        const size_t dilation = metadata[9];                                                       \
        const size_t input_offset = metadata[10];                                                  \
        const size_t weight_offset = metadata[11];                                                 \
                                                                                                   \
        for (size_t b = 0; b < batch; b++) {                                                       \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t iw = 0; iw < in_width; iw++) {                                         \
                    for (size_t kw = 0; kw < kernel_width; kw++) {                                 \
                        const int ow = (int)(iw * stride) - (int)padding + (int)(kw * dilation);   \
                        if (ow >= 0 && ow < (int)out_width) {                                      \
                            for (size_t oc = 0; oc < out_channels; oc++) {                         \
                                const size_t input_idx = input_offset +                            \
                                                         b * in_channels * in_width +              \
                                                         ic * in_width + iw;                       \
                                const size_t weight_idx = weight_offset +                          \
                                                          ic * out_channels * kernel_width +       \
                                                          oc * kernel_width + kw;                  \
                                const size_t output_idx =                                          \
                                    b * out_channels * out_width + oc * out_width + ow;            \
                                output[output_idx] += input[input_idx] * weight[weight_idx];       \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV_TRANSPOSE1D_OP(f8e4m3_t, f8e4m3)
CONV_TRANSPOSE1D_OP(f8e5m2_t, f8e5m2)
CONV_TRANSPOSE1D_OP(bf16_t, bf16)
CONV_TRANSPOSE1D_OP(f16_t, f16)
CONV_TRANSPOSE1D_OP(float, f32)
CONV_TRANSPOSE1D_OP(double, f64)

// ============================================================================
// 2D TRANSPOSED CONVOLUTION OPERATIONS
// ============================================================================
//
// Metadata layout for conv_transpose2d: Same as conv2d
// output_padding is used only for calculating out_height/out_width at the Rust level,
// not needed in the kernel as it's already reflected in out_height/out_width

/// Macro to implement 2D transposed convolution operation
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
#define CONV_TRANSPOSE2D_OP(TYPE, TYPE_SUFFIX)                                                     \
    void conv_transpose2d_##TYPE_SUFFIX(const void *input_ptr, const void *weight_ptr,             \
                                        void *output_ptr, const size_t *metadata) {                \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *weight = (const TYPE *)weight_ptr;                                             \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
                                                                                                   \
        memset(output, 0, num_els * sizeof(TYPE));                                                 \
                                                                                                   \
        const size_t batch = metadata[1];                                                          \
        const size_t in_channels = metadata[2];                                                    \
        const size_t out_channels = metadata[3];                                                   \
        const size_t in_height = metadata[4];                                                      \
        const size_t in_width = metadata[5];                                                       \
        const size_t kernel_height = metadata[6];                                                  \
        const size_t kernel_width = metadata[7];                                                   \
        const size_t out_height = metadata[8];                                                     \
        const size_t out_width = metadata[9];                                                      \
        const size_t stride_h = metadata[10];                                                      \
        const size_t stride_w = metadata[11];                                                      \
        const size_t padding_h = metadata[12];                                                     \
        const size_t padding_w = metadata[13];                                                     \
        const size_t dilation_h = metadata[14];                                                    \
        const size_t dilation_w = metadata[15];                                                    \
        const size_t input_offset = metadata[16];                                                  \
        const size_t weight_offset = metadata[17];                                                 \
                                                                                                   \
        for (size_t b = 0; b < batch; b++) {                                                       \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t ih = 0; ih < in_height; ih++) {                                        \
                    for (size_t iw = 0; iw < in_width; iw++) {                                     \
                        for (size_t kh = 0; kh < kernel_height; kh++) {                            \
                            for (size_t kw = 0; kw < kernel_width; kw++) {                         \
                                const int oh = (int)(ih * stride_h) - (int)padding_h +             \
                                               (int)(kh * dilation_h);                             \
                                const int ow = (int)(iw * stride_w) - (int)padding_w +             \
                                               (int)(kw * dilation_w);                             \
                                if (oh >= 0 && oh < (int)out_height && ow >= 0 &&                  \
                                    ow < (int)out_width) {                                         \
                                    for (size_t oc = 0; oc < out_channels; oc++) {                 \
                                        const size_t input_idx =                                   \
                                            input_offset +                                         \
                                            b * in_channels * in_height * in_width +               \
                                            ic * in_height * in_width + ih * in_width + iw;        \
                                        const size_t weight_idx =                                  \
                                            weight_offset +                                        \
                                            ic * out_channels * kernel_height * kernel_width +     \
                                            oc * kernel_height * kernel_width +                    \
                                            kh * kernel_width + kw;                                \
                                        const size_t output_idx =                                  \
                                            b * out_channels * out_height * out_width +            \
                                            oc * out_height * out_width + oh * out_width + ow;     \
                                        output[output_idx] +=                                      \
                                            input[input_idx] * weight[weight_idx];                 \
                                    }                                                              \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV_TRANSPOSE2D_OP(f8e4m3_t, f8e4m3)
CONV_TRANSPOSE2D_OP(f8e5m2_t, f8e5m2)
CONV_TRANSPOSE2D_OP(bf16_t, bf16)
CONV_TRANSPOSE2D_OP(f16_t, f16)
CONV_TRANSPOSE2D_OP(float, f32)
CONV_TRANSPOSE2D_OP(double, f64)

// ============================================================================
// 3D TRANSPOSED CONVOLUTION OPERATIONS
// ============================================================================
//
// Metadata layout: Same as conv3d

/// Macro to implement 3D transposed convolution operation
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
#define CONV_TRANSPOSE3D_OP(TYPE, TYPE_SUFFIX)                                                     \
    void conv_transpose3d_##TYPE_SUFFIX(const void *input_ptr, const void *weight_ptr,             \
                                        void *output_ptr, const size_t *metadata) {                \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *weight = (const TYPE *)weight_ptr;                                             \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
                                                                                                   \
        memset(output, 0, num_els * sizeof(TYPE));                                                 \
                                                                                                   \
        const size_t batch = metadata[1];                                                          \
        const size_t in_channels = metadata[2];                                                    \
        const size_t out_channels = metadata[3];                                                   \
        const size_t in_depth = metadata[4];                                                       \
        const size_t in_height = metadata[5];                                                      \
        const size_t in_width = metadata[6];                                                       \
        const size_t kernel_depth = metadata[7];                                                   \
        const size_t kernel_height = metadata[8];                                                  \
        const size_t kernel_width = metadata[9];                                                   \
        const size_t out_depth = metadata[10];                                                     \
        const size_t out_height = metadata[11];                                                    \
        const size_t out_width = metadata[12];                                                     \
        const size_t stride_d = metadata[13];                                                      \
        const size_t stride_h = metadata[14];                                                      \
        const size_t stride_w = metadata[15];                                                      \
        const size_t padding_d = metadata[16];                                                     \
        const size_t padding_h = metadata[17];                                                     \
        const size_t padding_w = metadata[18];                                                     \
        const size_t dilation_d = metadata[19];                                                    \
        const size_t dilation_h = metadata[20];                                                    \
        const size_t dilation_w = metadata[21];                                                    \
        const size_t input_offset = metadata[22];                                                  \
        const size_t weight_offset = metadata[23];                                                 \
                                                                                                   \
        for (size_t b = 0; b < batch; b++) {                                                       \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t id = 0; id < in_depth; id++) {                                         \
                    for (size_t ih = 0; ih < in_height; ih++) {                                    \
                        for (size_t iw = 0; iw < in_width; iw++) {                                 \
                            for (size_t kd = 0; kd < kernel_depth; kd++) {                         \
                                for (size_t kh = 0; kh < kernel_height; kh++) {                    \
                                    for (size_t kw = 0; kw < kernel_width; kw++) {                 \
                                        const int od = (int)(id * stride_d) - (int)padding_d +     \
                                                       (int)(kd * dilation_d);                     \
                                        const int oh = (int)(ih * stride_h) - (int)padding_h +     \
                                                       (int)(kh * dilation_h);                     \
                                        const int ow = (int)(iw * stride_w) - (int)padding_w +     \
                                                       (int)(kw * dilation_w);                     \
                                        if (od >= 0 && od < (int)out_depth && oh >= 0 &&           \
                                            oh < (int)out_height && ow >= 0 &&                     \
                                            ow < (int)out_width) {                                 \
                                            for (size_t oc = 0; oc < out_channels; oc++) {         \
                                                const size_t input_idx =                           \
                                                    input_offset +                                 \
                                                    b * in_channels * in_depth * in_height *       \
                                                        in_width +                                 \
                                                    ic * in_depth * in_height * in_width +         \
                                                    id * in_height * in_width + ih * in_width +    \
                                                    iw;                                            \
                                                const size_t weight_idx =                          \
                                                    weight_offset +                                \
                                                    ic * out_channels * kernel_depth *             \
                                                        kernel_height * kernel_width +             \
                                                    oc * kernel_depth * kernel_height *            \
                                                        kernel_width +                             \
                                                    kd * kernel_height * kernel_width +            \
                                                    kh * kernel_width + kw;                        \
                                                const size_t output_idx =                          \
                                                    b * out_channels * out_depth * out_height *    \
                                                        out_width +                                \
                                                    oc * out_depth * out_height * out_width +      \
                                                    od * out_height * out_width + oh * out_width + \
                                                    ow;                                            \
                                                output[output_idx] +=                              \
                                                    input[input_idx] * weight[weight_idx];         \
                                            }                                                      \
                                        }                                                          \
                                    }                                                              \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV_TRANSPOSE3D_OP(f8e4m3_t, f8e4m3)
CONV_TRANSPOSE3D_OP(f8e5m2_t, f8e5m2)
CONV_TRANSPOSE3D_OP(bf16_t, bf16)
CONV_TRANSPOSE3D_OP(f16_t, f16)
CONV_TRANSPOSE3D_OP(float, f32)
CONV_TRANSPOSE3D_OP(double, f64)

// ============================================================================
// 1D CONVOLUTION GRADIENT WEIGHT OPERATIONS
// ============================================================================
//
// Computes gradients with respect to convolution weights for backpropagation.
// Uses atomic operations to handle concurrent updates from multiple threads.
//
// Metadata layout for conv1d_grad_weight:
// - metadata[0]: num_els (total number of grad_weight elements)
// - metadata[1]: batch
// - metadata[2]: in_channels
// - metadata[3]: out_channels
// - metadata[4]: in_width
// - metadata[5]: kernel_width
// - metadata[6]: out_width
// - metadata[7]: stride
// - metadata[8]: padding
// - metadata[9]: dilation
// - metadata[10]: input_offset
// - metadata[11]: grad_output_offset

/// Macro to implement 1D convolution gradient weight operation
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
/// @param ATOMIC_ADD_FN Atomic addition function for thread-safe updates
#define CONV1D_GRAD_WEIGHT_OP(TYPE, TYPE_SUFFIX, ATOMIC_ADD_FN)                                    \
    void conv1d_grad_weight_##TYPE_SUFFIX(const void *input_ptr, const void *grad_output_ptr,      \
                                          void *grad_weight_ptr, const size_t *metadata) {         \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *grad_output = (const TYPE *)grad_output_ptr;                                   \
        TYPE *grad_weight = (TYPE *)grad_weight_ptr;                                               \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t batch = metadata[1];                                                          \
        const size_t in_channels = metadata[2];                                                    \
                                                                                                   \
        memset(grad_weight, 0, num_els * sizeof(TYPE));                                            \
        const size_t out_channels = metadata[3];                                                   \
        const size_t in_width = metadata[4];                                                       \
        const size_t kernel_width = metadata[5];                                                   \
        const size_t out_width = metadata[6];                                                      \
        const size_t stride = metadata[7];                                                         \
        const size_t padding = metadata[8];                                                        \
        const size_t dilation = metadata[9];                                                       \
        const size_t input_offset = metadata[10];                                                  \
        const size_t grad_output_offset = metadata[11];                                            \
                                                                                                   \
        for (size_t b = 0; b < batch; b++) {                                                       \
            for (size_t oc = 0; oc < out_channels; oc++) {                                         \
                for (size_t ow = 0; ow < out_width; ow++) {                                        \
                    const size_t grad_output_idx =                                                 \
                        grad_output_offset + b * out_channels * out_width + oc * out_width + ow;   \
                    const TYPE grad_out_val = grad_output[grad_output_idx];                        \
                                                                                                   \
                    for (size_t ic = 0; ic < in_channels; ic++) {                                  \
                        for (size_t kw = 0; kw < kernel_width; kw++) {                             \
                            const int iw =                                                         \
                                (int)(ow * stride) - (int)padding + (int)(kw * dilation);          \
                            if (iw >= 0 && iw < (int)in_width) {                                   \
                                const size_t input_idx = input_offset +                            \
                                                         b * in_channels * in_width +              \
                                                         ic * in_width + iw;                       \
                                const size_t weight_idx =                                          \
                                    oc * in_channels * kernel_width + ic * kernel_width + kw;      \
                                const TYPE contribution = input[input_idx] * grad_out_val;         \
                                ATOMIC_ADD_FN(&grad_weight[weight_idx], contribution);             \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV1D_GRAD_WEIGHT_OP(f8e4m3_t, f8e4m3, atomic_add_f8e4m3)
CONV1D_GRAD_WEIGHT_OP(f8e5m2_t, f8e5m2, atomic_add_f8e5m2)
CONV1D_GRAD_WEIGHT_OP(bf16_t, bf16, atomic_add_bf16)
CONV1D_GRAD_WEIGHT_OP(f16_t, f16, atomic_add_f16)
CONV1D_GRAD_WEIGHT_OP(float, f32, atomic_add_f32)
CONV1D_GRAD_WEIGHT_OP(double, f64, atomic_add_f64)

// ============================================================================
// 2D CONVOLUTION GRADIENT WEIGHT OPERATIONS
// ============================================================================
//
// Metadata layout for conv2d_grad_weight:
// - metadata[0]: num_els
// - metadata[1]: batch
// - metadata[2]: in_channels
// - metadata[3]: out_channels
// - metadata[4]: in_height
// - metadata[5]: in_width
// - metadata[6]: kernel_height
// - metadata[7]: kernel_width
// - metadata[8]: out_height
// - metadata[9]: out_width
// - metadata[10]: stride_h
// - metadata[11]: stride_w
// - metadata[12]: padding_h
// - metadata[13]: padding_w
// - metadata[14]: dilation_h
// - metadata[15]: dilation_w
// - metadata[16]: input_offset
// - metadata[17]: grad_output_offset

/// Macro to implement 2D convolution gradient weight operation
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
/// @param ATOMIC_ADD_FN Atomic addition function for thread-safe updates
#define CONV2D_GRAD_WEIGHT_OP(TYPE, TYPE_SUFFIX, ATOMIC_ADD_FN)                                    \
    void conv2d_grad_weight_##TYPE_SUFFIX(const void *input_ptr, const void *grad_output_ptr,      \
                                          void *grad_weight_ptr, const size_t *metadata) {         \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *grad_output = (const TYPE *)grad_output_ptr;                                   \
        TYPE *grad_weight = (TYPE *)grad_weight_ptr;                                               \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t batch = metadata[1];                                                          \
        const size_t in_channels = metadata[2];                                                    \
                                                                                                   \
        memset(grad_weight, 0, num_els * sizeof(TYPE));                                            \
        const size_t out_channels = metadata[3];                                                   \
        const size_t in_height = metadata[4];                                                      \
        const size_t in_width = metadata[5];                                                       \
        const size_t kernel_height = metadata[6];                                                  \
        const size_t kernel_width = metadata[7];                                                   \
        const size_t out_height = metadata[8];                                                     \
        const size_t out_width = metadata[9];                                                      \
        const size_t stride_h = metadata[10];                                                      \
        const size_t stride_w = metadata[11];                                                      \
        const size_t padding_h = metadata[12];                                                     \
        const size_t padding_w = metadata[13];                                                     \
        const size_t dilation_h = metadata[14];                                                    \
        const size_t dilation_w = metadata[15];                                                    \
        const size_t input_offset = metadata[16];                                                  \
        const size_t grad_output_offset = metadata[17];                                            \
                                                                                                   \
        for (size_t b = 0; b < batch; b++) {                                                       \
            for (size_t oc = 0; oc < out_channels; oc++) {                                         \
                for (size_t oh = 0; oh < out_height; oh++) {                                       \
                    for (size_t ow = 0; ow < out_width; ow++) {                                    \
                        const size_t grad_output_idx =                                             \
                            grad_output_offset + b * out_channels * out_height * out_width +       \
                            oc * out_height * out_width + oh * out_width + ow;                     \
                        const TYPE grad_out_val = grad_output[grad_output_idx];                    \
                                                                                                   \
                        for (size_t ic = 0; ic < in_channels; ic++) {                              \
                            for (size_t kh = 0; kh < kernel_height; kh++) {                        \
                                for (size_t kw = 0; kw < kernel_width; kw++) {                     \
                                    const int ih = (int)(oh * stride_h) - (int)padding_h +         \
                                                   (int)(kh * dilation_h);                         \
                                    const int iw = (int)(ow * stride_w) - (int)padding_w +         \
                                                   (int)(kw * dilation_w);                         \
                                    if (ih >= 0 && ih < (int)in_height && iw >= 0 &&               \
                                        iw < (int)in_width) {                                      \
                                        const size_t input_idx =                                   \
                                            input_offset +                                         \
                                            b * in_channels * in_height * in_width +               \
                                            ic * in_height * in_width + ih * in_width + iw;        \
                                        const size_t weight_idx =                                  \
                                            oc * in_channels * kernel_height * kernel_width +      \
                                            ic * kernel_height * kernel_width +                    \
                                            kh * kernel_width + kw;                                \
                                        const TYPE contribution = input[input_idx] * grad_out_val; \
                                        ATOMIC_ADD_FN(&grad_weight[weight_idx], contribution);     \
                                    }                                                              \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV2D_GRAD_WEIGHT_OP(f8e4m3_t, f8e4m3, atomic_add_f8e4m3)
CONV2D_GRAD_WEIGHT_OP(f8e5m2_t, f8e5m2, atomic_add_f8e5m2)
CONV2D_GRAD_WEIGHT_OP(bf16_t, bf16, atomic_add_bf16)
CONV2D_GRAD_WEIGHT_OP(f16_t, f16, atomic_add_f16)

// BLAS-optimized conv2d_grad_weight for f32 using im2col + GEMM
void conv2d_grad_weight_f32(const void *input_ptr, const void *grad_output_ptr,
                            void *grad_weight_ptr, const size_t *metadata) {
    const float *input = (const float *)input_ptr;
    const float *grad_output = (const float *)grad_output_ptr;
    float *grad_weight = (float *)grad_weight_ptr;

    const size_t num_els = metadata[0];
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
    const size_t grad_output_offset = metadata[17];

    memset(grad_weight, 0, num_els * sizeof(float));

#ifdef USE_BLAS
    // Use im2col + GEMM for optimized gradient computation
    const size_t K = in_channels * kernel_height * kernel_width;
    const size_t N = out_height * out_width;
    const size_t M = out_channels;

    // Allocate column buffer: [K, N]
    float *col_buffer = (float *)malloc(K * N * sizeof(float));
    if (!col_buffer) {
        goto fallback_grad_weight_f32;
    }

    // Process each batch element and accumulate gradients
    for (size_t b = 0; b < batch; b++) {
        // Im2col: Transform input patches to columns
        const float *batch_input = input + input_offset + b * in_channels * in_height * in_width;
        im2col_f32(batch_input, col_buffer, in_channels, in_height, in_width, kernel_height,
                   kernel_width, out_height, out_width, stride_h, stride_w, padding_h, padding_w,
                   dilation_h, dilation_w);

        // GEMM: grad_weight += grad_output × col_buffer^T
        // grad_output shape: [M, N] (out_channels, out_height * out_width)
        // col_buffer^T shape: [N, K] (out_height * out_width, in_channels * kh * kw)
        // grad_weight shape: [M, K] (out_channels, in_channels * kh * kw)
        const float *batch_grad_output =
            grad_output + grad_output_offset + b * out_channels * out_height * out_width;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, K, N, 1.0f, batch_grad_output, N,
                    col_buffer, N, 1.0f, grad_weight,
                    K); // Note: beta=1.0 to accumulate across batches
    }

    free(col_buffer);
    return;

fallback_grad_weight_f32:
#endif
    // Fallback: Naive implementation
    for (size_t b = 0; b < batch; b++) {
        for (size_t oc = 0; oc < out_channels; oc++) {
            for (size_t oh = 0; oh < out_height; oh++) {
                for (size_t ow = 0; ow < out_width; ow++) {
                    const size_t grad_output_idx =
                        grad_output_offset + b * out_channels * out_height * out_width +
                        oc * out_height * out_width + oh * out_width + ow;
                    const float grad_out_val = grad_output[grad_output_idx];

                    for (size_t ic = 0; ic < in_channels; ic++) {
                        for (size_t kh = 0; kh < kernel_height; kh++) {
                            for (size_t kw = 0; kw < kernel_width; kw++) {
                                const int ih =
                                    (int)(oh * stride_h) - (int)padding_h + (int)(kh * dilation_h);
                                const int iw =
                                    (int)(ow * stride_w) - (int)padding_w + (int)(kw * dilation_w);
                                if (ih >= 0 && ih < (int)in_height && iw >= 0 &&
                                    iw < (int)in_width) {
                                    const size_t input_idx =
                                        input_offset + b * in_channels * in_height * in_width +
                                        ic * in_height * in_width + ih * in_width + iw;
                                    const size_t weight_idx =
                                        oc * in_channels * kernel_height * kernel_width +
                                        ic * kernel_height * kernel_width + kh * kernel_width + kw;
                                    const float contribution = input[input_idx] * grad_out_val;
                                    atomic_add_f32(&grad_weight[weight_idx], contribution);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// BLAS-optimized conv2d_grad_weight for f64 using im2col + GEMM
void conv2d_grad_weight_f64(const void *input_ptr, const void *grad_output_ptr,
                            void *grad_weight_ptr, const size_t *metadata) {
    const double *input = (const double *)input_ptr;
    const double *grad_output = (const double *)grad_output_ptr;
    double *grad_weight = (double *)grad_weight_ptr;

    const size_t num_els = metadata[0];
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
    const size_t grad_output_offset = metadata[17];

    memset(grad_weight, 0, num_els * sizeof(double));

#ifdef USE_BLAS
    // Use im2col + GEMM for optimized gradient computation
    const size_t K = in_channels * kernel_height * kernel_width;
    const size_t N = out_height * out_width;
    const size_t M = out_channels;

    // Allocate column buffer: [K, N]
    double *col_buffer = (double *)malloc(K * N * sizeof(double));
    if (!col_buffer) {
        goto fallback_grad_weight_f64;
    }

    // Process each batch element and accumulate gradients
    for (size_t b = 0; b < batch; b++) {
        // Im2col: Transform input patches to columns
        const double *batch_input = input + input_offset + b * in_channels * in_height * in_width;
        im2col_f64(batch_input, col_buffer, in_channels, in_height, in_width, kernel_height,
                   kernel_width, out_height, out_width, stride_h, stride_w, padding_h, padding_w,
                   dilation_h, dilation_w);

        // GEMM: grad_weight += grad_output × col_buffer^T
        // grad_output shape: [M, N] (out_channels, out_height * out_width)
        // col_buffer^T shape: [N, K] (out_height * out_width, in_channels * kh * kw)
        // grad_weight shape: [M, K] (out_channels, in_channels * kh * kw)
        const double *batch_grad_output =
            grad_output + grad_output_offset + b * out_channels * out_height * out_width;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, K, N, 1.0, batch_grad_output, N,
                    col_buffer, N, 1.0, grad_weight,
                    K); // Note: beta=1.0 to accumulate across batches
    }

    free(col_buffer);
    return;

fallback_grad_weight_f64:
#endif
    // Fallback: Naive implementation
    for (size_t b = 0; b < batch; b++) {
        for (size_t oc = 0; oc < out_channels; oc++) {
            for (size_t oh = 0; oh < out_height; oh++) {
                for (size_t ow = 0; ow < out_width; ow++) {
                    const size_t grad_output_idx =
                        grad_output_offset + b * out_channels * out_height * out_width +
                        oc * out_height * out_width + oh * out_width + ow;
                    const double grad_out_val = grad_output[grad_output_idx];

                    for (size_t ic = 0; ic < in_channels; ic++) {
                        for (size_t kh = 0; kh < kernel_height; kh++) {
                            for (size_t kw = 0; kw < kernel_width; kw++) {
                                const int ih =
                                    (int)(oh * stride_h) - (int)padding_h + (int)(kh * dilation_h);
                                const int iw =
                                    (int)(ow * stride_w) - (int)padding_w + (int)(kw * dilation_w);
                                if (ih >= 0 && ih < (int)in_height && iw >= 0 &&
                                    iw < (int)in_width) {
                                    const size_t input_idx =
                                        input_offset + b * in_channels * in_height * in_width +
                                        ic * in_height * in_width + ih * in_width + iw;
                                    const size_t weight_idx =
                                        oc * in_channels * kernel_height * kernel_width +
                                        ic * kernel_height * kernel_width + kh * kernel_width + kw;
                                    const double contribution = input[input_idx] * grad_out_val;
                                    atomic_add_f64(&grad_weight[weight_idx], contribution);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 3D CONVOLUTION GRADIENT WEIGHT OPERATIONS
// ============================================================================
//
// Metadata layout for conv3d_grad_weight:
// - metadata[0]: num_els
// - metadata[1]: batch
// - metadata[2]: in_channels
// - metadata[3]: out_channels
// - metadata[4]: in_depth
// - metadata[5]: in_height
// - metadata[6]: in_width
// - metadata[7]: kernel_depth
// - metadata[8]: kernel_height
// - metadata[9]: kernel_width
// - metadata[10]: out_depth
// - metadata[11]: out_height
// - metadata[12]: out_width
// - metadata[13]: stride_d
// - metadata[14]: stride_h
// - metadata[15]: stride_w
// - metadata[16]: padding_d
// - metadata[17]: padding_h
// - metadata[18]: padding_w
// - metadata[19]: dilation_d
// - metadata[20]: dilation_h
// - metadata[21]: dilation_w
// - metadata[22]: input_offset
// - metadata[23]: grad_output_offset

/// Macro to implement 3D convolution gradient weight operation
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
/// @param ATOMIC_ADD_FN Atomic addition function for thread-safe updates
#define CONV3D_GRAD_WEIGHT_OP(TYPE, TYPE_SUFFIX, ATOMIC_ADD_FN)                                    \
    void conv3d_grad_weight_##TYPE_SUFFIX(const void *input_ptr, const void *grad_output_ptr,      \
                                          void *grad_weight_ptr, const size_t *metadata) {         \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *grad_output = (const TYPE *)grad_output_ptr;                                   \
        TYPE *grad_weight = (TYPE *)grad_weight_ptr;                                               \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t batch = metadata[1];                                                          \
        const size_t in_channels = metadata[2];                                                    \
                                                                                                   \
        memset(grad_weight, 0, num_els * sizeof(TYPE));                                            \
        const size_t out_channels = metadata[3];                                                   \
        const size_t in_depth = metadata[4];                                                       \
        const size_t in_height = metadata[5];                                                      \
        const size_t in_width = metadata[6];                                                       \
        const size_t kernel_depth = metadata[7];                                                   \
        const size_t kernel_height = metadata[8];                                                  \
        const size_t kernel_width = metadata[9];                                                   \
        const size_t out_depth = metadata[10];                                                     \
        const size_t out_height = metadata[11];                                                    \
        const size_t out_width = metadata[12];                                                     \
        const size_t stride_d = metadata[13];                                                      \
        const size_t stride_h = metadata[14];                                                      \
        const size_t stride_w = metadata[15];                                                      \
        const size_t padding_d = metadata[16];                                                     \
        const size_t padding_h = metadata[17];                                                     \
        const size_t padding_w = metadata[18];                                                     \
        const size_t dilation_d = metadata[19];                                                    \
        const size_t dilation_h = metadata[20];                                                    \
        const size_t dilation_w = metadata[21];                                                    \
        const size_t input_offset = metadata[22];                                                  \
        const size_t grad_output_offset = metadata[23];                                            \
                                                                                                   \
        for (size_t b = 0; b < batch; b++) {                                                       \
            for (size_t oc = 0; oc < out_channels; oc++) {                                         \
                for (size_t od = 0; od < out_depth; od++) {                                        \
                    for (size_t oh = 0; oh < out_height; oh++) {                                   \
                        for (size_t ow = 0; ow < out_width; ow++) {                                \
                            const size_t grad_output_idx =                                         \
                                grad_output_offset +                                               \
                                b * out_channels * out_depth * out_height * out_width +            \
                                oc * out_depth * out_height * out_width +                          \
                                od * out_height * out_width + oh * out_width + ow;                 \
                            const TYPE grad_out_val = grad_output[grad_output_idx];                \
                                                                                                   \
                            for (size_t ic = 0; ic < in_channels; ic++) {                          \
                                for (size_t kd = 0; kd < kernel_depth; kd++) {                     \
                                    for (size_t kh = 0; kh < kernel_height; kh++) {                \
                                        for (size_t kw = 0; kw < kernel_width; kw++) {             \
                                            const int id = (int)(od * stride_d) - (int)padding_d + \
                                                           (int)(kd * dilation_d);                 \
                                            const int ih = (int)(oh * stride_h) - (int)padding_h + \
                                                           (int)(kh * dilation_h);                 \
                                            const int iw = (int)(ow * stride_w) - (int)padding_w + \
                                                           (int)(kw * dilation_w);                 \
                                            if (id >= 0 && id < (int)in_depth && ih >= 0 &&        \
                                                ih < (int)in_height && iw >= 0 &&                  \
                                                iw < (int)in_width) {                              \
                                                const size_t input_idx =                           \
                                                    input_offset +                                 \
                                                    b * in_channels * in_depth * in_height *       \
                                                        in_width +                                 \
                                                    ic * in_depth * in_height * in_width +         \
                                                    id * in_height * in_width + ih * in_width +    \
                                                    iw;                                            \
                                                const size_t weight_idx =                          \
                                                    oc * in_channels * kernel_depth *              \
                                                        kernel_height * kernel_width +             \
                                                    ic * kernel_depth * kernel_height *            \
                                                        kernel_width +                             \
                                                    kd * kernel_height * kernel_width +            \
                                                    kh * kernel_width + kw;                        \
                                                const TYPE contribution =                          \
                                                    input[input_idx] * grad_out_val;               \
                                                ATOMIC_ADD_FN(&grad_weight[weight_idx],            \
                                                              contribution);                       \
                                            }                                                      \
                                        }                                                          \
                                    }                                                              \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV3D_GRAD_WEIGHT_OP(f8e4m3_t, f8e4m3, atomic_add_f8e4m3)
CONV3D_GRAD_WEIGHT_OP(f8e5m2_t, f8e5m2, atomic_add_f8e5m2)
CONV3D_GRAD_WEIGHT_OP(bf16_t, bf16, atomic_add_bf16)
CONV3D_GRAD_WEIGHT_OP(f16_t, f16, atomic_add_f16)
CONV3D_GRAD_WEIGHT_OP(float, f32, atomic_add_f32)
CONV3D_GRAD_WEIGHT_OP(double, f64, atomic_add_f64)

// ============================================================================
// 1D TRANSPOSED CONVOLUTION GRADIENT WEIGHT OPERATIONS
// ============================================================================
//
// Metadata layout: Same as conv_transpose1d

/// Macro to implement 1D transposed convolution gradient weight operation
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
/// @param ATOMIC_ADD_FN Atomic addition function for thread-safe updates
#define CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(TYPE, TYPE_SUFFIX, ATOMIC_ADD_FN)                          \
    void conv_transpose1d_grad_weight_##TYPE_SUFFIX(                                               \
        const void *input_ptr, const void *grad_output_ptr, void *grad_weight_ptr,                 \
        const size_t *metadata) {                                                                  \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *grad_output = (const TYPE *)grad_output_ptr;                                   \
        TYPE *grad_weight = (TYPE *)grad_weight_ptr;                                               \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t batch = metadata[1];                                                          \
        const size_t in_channels = metadata[2];                                                    \
                                                                                                   \
        memset(grad_weight, 0, num_els * sizeof(TYPE));                                            \
        const size_t out_channels = metadata[3];                                                   \
        const size_t in_width = metadata[4];                                                       \
        const size_t kernel_width = metadata[5];                                                   \
        const size_t out_width = metadata[6];                                                      \
        const size_t stride = metadata[7];                                                         \
        const size_t padding = metadata[8];                                                        \
        const size_t dilation = metadata[9];                                                       \
        const size_t input_offset = metadata[10];                                                  \
        const size_t grad_output_offset = metadata[11];                                            \
                                                                                                   \
        for (size_t b = 0; b < batch; b++) {                                                       \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t iw = 0; iw < in_width; iw++) {                                         \
                    const size_t input_idx =                                                       \
                        input_offset + b * in_channels * in_width + ic * in_width + iw;            \
                    const TYPE input_val = input[input_idx];                                       \
                                                                                                   \
                    for (size_t oc = 0; oc < out_channels; oc++) {                                 \
                        for (size_t kw = 0; kw < kernel_width; kw++) {                             \
                            const int ow =                                                         \
                                (int)(iw * stride) - (int)padding + (int)(kw * dilation);          \
                            if (ow >= 0 && ow < (int)out_width) {                                  \
                                const size_t grad_output_idx = grad_output_offset +                \
                                                               b * out_channels * out_width +      \
                                                               oc * out_width + ow;                \
                                const size_t weight_idx =                                          \
                                    ic * out_channels * kernel_width + oc * kernel_width + kw;     \
                                const TYPE contribution =                                          \
                                    input_val * grad_output[grad_output_idx];                      \
                                ATOMIC_ADD_FN(&grad_weight[weight_idx], contribution);             \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(f8e4m3_t, f8e4m3, atomic_add_f8e4m3)
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(f8e5m2_t, f8e5m2, atomic_add_f8e5m2)
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(bf16_t, bf16, atomic_add_bf16)
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(f16_t, f16, atomic_add_f16)
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(float, f32, atomic_add_f32)
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(double, f64, atomic_add_f64)

// ============================================================================
// 2D TRANSPOSED CONVOLUTION GRADIENT WEIGHT OPERATIONS
// ============================================================================
//
// Metadata layout: Same as conv_transpose2d

/// Macro to implement 2D transposed convolution gradient weight operation
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
/// @param ATOMIC_ADD_FN Atomic addition function for thread-safe updates
#define CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(TYPE, TYPE_SUFFIX, ATOMIC_ADD_FN)                          \
    void conv_transpose2d_grad_weight_##TYPE_SUFFIX(                                               \
        const void *input_ptr, const void *grad_output_ptr, void *grad_weight_ptr,                 \
        const size_t *metadata) {                                                                  \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *grad_output = (const TYPE *)grad_output_ptr;                                   \
        TYPE *grad_weight = (TYPE *)grad_weight_ptr;                                               \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t batch = metadata[1];                                                          \
        const size_t in_channels = metadata[2];                                                    \
                                                                                                   \
        memset(grad_weight, 0, num_els * sizeof(TYPE));                                            \
        const size_t out_channels = metadata[3];                                                   \
        const size_t in_height = metadata[4];                                                      \
        const size_t in_width = metadata[5];                                                       \
        const size_t kernel_height = metadata[6];                                                  \
        const size_t kernel_width = metadata[7];                                                   \
        const size_t out_height = metadata[8];                                                     \
        const size_t out_width = metadata[9];                                                      \
        const size_t stride_h = metadata[10];                                                      \
        const size_t stride_w = metadata[11];                                                      \
        const size_t padding_h = metadata[12];                                                     \
        const size_t padding_w = metadata[13];                                                     \
        const size_t dilation_h = metadata[14];                                                    \
        const size_t dilation_w = metadata[15];                                                    \
        const size_t input_offset = metadata[16];                                                  \
        const size_t grad_output_offset = metadata[17];                                            \
                                                                                                   \
        for (size_t b = 0; b < batch; b++) {                                                       \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t ih = 0; ih < in_height; ih++) {                                        \
                    for (size_t iw = 0; iw < in_width; iw++) {                                     \
                        const size_t input_idx = input_offset +                                    \
                                                 b * in_channels * in_height * in_width +          \
                                                 ic * in_height * in_width + ih * in_width + iw;   \
                        const TYPE input_val = input[input_idx];                                   \
                                                                                                   \
                        for (size_t oc = 0; oc < out_channels; oc++) {                             \
                            for (size_t kh = 0; kh < kernel_height; kh++) {                        \
                                for (size_t kw = 0; kw < kernel_width; kw++) {                     \
                                    const int oh = (int)(ih * stride_h) - (int)padding_h +         \
                                                   (int)(kh * dilation_h);                         \
                                    const int ow = (int)(iw * stride_w) - (int)padding_w +         \
                                                   (int)(kw * dilation_w);                         \
                                    if (oh >= 0 && oh < (int)out_height && ow >= 0 &&              \
                                        ow < (int)out_width) {                                     \
                                        const size_t grad_output_idx =                             \
                                            grad_output_offset +                                   \
                                            b * out_channels * out_height * out_width +            \
                                            oc * out_height * out_width + oh * out_width + ow;     \
                                        const size_t weight_idx =                                  \
                                            ic * out_channels * kernel_height * kernel_width +     \
                                            oc * kernel_height * kernel_width +                    \
                                            kh * kernel_width + kw;                                \
                                        const TYPE contribution =                                  \
                                            input_val * grad_output[grad_output_idx];              \
                                        ATOMIC_ADD_FN(&grad_weight[weight_idx], contribution);     \
                                    }                                                              \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(f8e4m3_t, f8e4m3, atomic_add_f8e4m3)
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(f8e5m2_t, f8e5m2, atomic_add_f8e5m2)
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(bf16_t, bf16, atomic_add_bf16)
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(f16_t, f16, atomic_add_f16)
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(float, f32, atomic_add_f32)
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(double, f64, atomic_add_f64)

// ============================================================================
// 3D TRANSPOSED CONVOLUTION GRADIENT WEIGHT OPERATIONS
// ============================================================================
//
// Metadata layout: Same as conv_transpose3d

/// Macro to implement 3D transposed convolution gradient weight operation
///
/// @param TYPE C type for the operation
/// @param TYPE_SUFFIX Suffix for function naming
/// @param ATOMIC_ADD_FN Atomic addition function for thread-safe updates
#define CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(TYPE, TYPE_SUFFIX, ATOMIC_ADD_FN)                          \
    void conv_transpose3d_grad_weight_##TYPE_SUFFIX(                                               \
        const void *input_ptr, const void *grad_output_ptr, void *grad_weight_ptr,                 \
        const size_t *metadata) {                                                                  \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        const TYPE *grad_output = (const TYPE *)grad_output_ptr;                                   \
        TYPE *grad_weight = (TYPE *)grad_weight_ptr;                                               \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t batch = metadata[1];                                                          \
        const size_t in_channels = metadata[2];                                                    \
                                                                                                   \
        memset(grad_weight, 0, num_els * sizeof(TYPE));                                            \
        const size_t out_channels = metadata[3];                                                   \
        const size_t in_depth = metadata[4];                                                       \
        const size_t in_height = metadata[5];                                                      \
        const size_t in_width = metadata[6];                                                       \
        const size_t kernel_depth = metadata[7];                                                   \
        const size_t kernel_height = metadata[8];                                                  \
        const size_t kernel_width = metadata[9];                                                   \
        const size_t out_depth = metadata[10];                                                     \
        const size_t out_height = metadata[11];                                                    \
        const size_t out_width = metadata[12];                                                     \
        const size_t stride_d = metadata[13];                                                      \
        const size_t stride_h = metadata[14];                                                      \
        const size_t stride_w = metadata[15];                                                      \
        const size_t padding_d = metadata[16];                                                     \
        const size_t padding_h = metadata[17];                                                     \
        const size_t padding_w = metadata[18];                                                     \
        const size_t dilation_d = metadata[19];                                                    \
        const size_t dilation_h = metadata[20];                                                    \
        const size_t dilation_w = metadata[21];                                                    \
        const size_t input_offset = metadata[22];                                                  \
        const size_t grad_output_offset = metadata[23];                                            \
                                                                                                   \
        for (size_t b = 0; b < batch; b++) {                                                       \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t id = 0; id < in_depth; id++) {                                         \
                    for (size_t ih = 0; ih < in_height; ih++) {                                    \
                        for (size_t iw = 0; iw < in_width; iw++) {                                 \
                            const size_t input_idx =                                               \
                                input_offset + b * in_channels * in_depth * in_height * in_width + \
                                ic * in_depth * in_height * in_width + id * in_height * in_width + \
                                ih * in_width + iw;                                                \
                            const TYPE input_val = input[input_idx];                               \
                                                                                                   \
                            for (size_t oc = 0; oc < out_channels; oc++) {                         \
                                for (size_t kd = 0; kd < kernel_depth; kd++) {                     \
                                    for (size_t kh = 0; kh < kernel_height; kh++) {                \
                                        for (size_t kw = 0; kw < kernel_width; kw++) {             \
                                            const int od = (int)(id * stride_d) - (int)padding_d + \
                                                           (int)(kd * dilation_d);                 \
                                            const int oh = (int)(ih * stride_h) - (int)padding_h + \
                                                           (int)(kh * dilation_h);                 \
                                            const int ow = (int)(iw * stride_w) - (int)padding_w + \
                                                           (int)(kw * dilation_w);                 \
                                            if (od >= 0 && od < (int)out_depth && oh >= 0 &&       \
                                                oh < (int)out_height && ow >= 0 &&                 \
                                                ow < (int)out_width) {                             \
                                                const size_t grad_output_idx =                     \
                                                    grad_output_offset +                           \
                                                    b * out_channels * out_depth * out_height *    \
                                                        out_width +                                \
                                                    oc * out_depth * out_height * out_width +      \
                                                    od * out_height * out_width + oh * out_width + \
                                                    ow;                                            \
                                                const size_t weight_idx =                          \
                                                    ic * out_channels * kernel_depth *             \
                                                        kernel_height * kernel_width +             \
                                                    oc * kernel_depth * kernel_height *            \
                                                        kernel_width +                             \
                                                    kd * kernel_height * kernel_width +            \
                                                    kh * kernel_width + kw;                        \
                                                const TYPE contribution =                          \
                                                    input_val * grad_output[grad_output_idx];      \
                                                ATOMIC_ADD_FN(&grad_weight[weight_idx],            \
                                                              contribution);                       \
                                            }                                                      \
                                        }                                                          \
                                    }                                                              \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(f8e4m3_t, f8e4m3, atomic_add_f8e4m3)
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(f8e5m2_t, f8e5m2, atomic_add_f8e5m2)
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(bf16_t, bf16, atomic_add_bf16)
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(f16_t, f16, atomic_add_f16)
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(float, f32, atomic_add_f32)
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(double, f64, atomic_add_f64)
