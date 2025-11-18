#include "ops_conv.h"
#include "atomic.h"
#include "types.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// BLAS-specific implementations are in separate files:
// - ops_conv_openblas.c for OpenBLAS
// - ops_conv_blas_aarch64_apple_darwin.c for macOS Accelerate

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

// im2col helper functions moved to BLAS-specific files

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

CONV2D_OP(f32_t, f32_fallback)
CONV2D_OP(f64_t, f64_fallback)

CONV2D_OP(f8e4m3_t, f8e4m3)
CONV2D_OP(f8e5m2_t, f8e5m2)
CONV2D_OP(bf16_t, bf16)
CONV2D_OP(f16_t, f16)

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

CONV2D_GRAD_WEIGHT_OP(f32_t, f32_fallback, atomic_add_f32)
CONV2D_GRAD_WEIGHT_OP(f64_t, f64_fallback, atomic_add_f64)

CONV2D_GRAD_WEIGHT_OP(f8e4m3_t, f8e4m3, atomic_add_f8e4m3)
CONV2D_GRAD_WEIGHT_OP(f8e5m2_t, f8e5m2, atomic_add_f8e5m2)
CONV2D_GRAD_WEIGHT_OP(bf16_t, bf16, atomic_add_bf16)
CONV2D_GRAD_WEIGHT_OP(f16_t, f16, atomic_add_f16)

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
