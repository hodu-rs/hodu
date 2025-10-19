#include "./atomic.metal"
#include "./utils.metal"
#include <metal_stdlib>

using namespace metal;

// Convolution operations for tensors
// Supports 1D, 2D, 3D convolutions and their transposed versions
// Also includes weight gradient operations

// ============================================================================
// CONV1D OPERATIONS
// ============================================================================

// 1D Convolution
// Input: (batch, in_channels, width)
// Weight: (out_channels, in_channels, kernel_width)
// Output: (batch, out_channels, out_width)
//
// Metadata layout:
// - batch, in_channels, out_channels
// - in_width, kernel_width, out_width
// - stride, padding, dilation
// - input_offset, weight_offset

#define CONV1D_OP(TYPENAME, FN_NAME)                                                               \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device TYPENAME *weight [[buffer(1)]],   \
        device TYPENAME *output [[buffer(2)]], constant size_t &num_els [[buffer(3)]],             \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        (void)metadata[0]; /* batch - unused, computed from idx */                                 \
        const size_t in_channels = metadata[1];                                                    \
        const size_t out_channels = metadata[2];                                                   \
        const size_t in_width = metadata[3];                                                       \
        const size_t kernel_width = metadata[4];                                                   \
        const size_t out_width = metadata[5];                                                      \
        const size_t stride = metadata[6];                                                         \
        const size_t padding = metadata[7];                                                        \
        const size_t dilation = metadata[8];                                                       \
        const size_t input_offset = metadata[9];                                                   \
        const size_t weight_offset = metadata[10];                                                 \
                                                                                                   \
        for (uint idx = thread_index; idx < num_els; idx += threads_per_grid) {                    \
            const size_t ow = idx % out_width;                                                     \
            const size_t oc = (idx / out_width) % out_channels;                                    \
            const size_t b = idx / (out_width * out_channels);                                     \
                                                                                                   \
            TYPENAME sum = 0;                                                                      \
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

CONV1D_OP(bfloat, conv1d_bf16)
CONV1D_OP(half, conv1d_f16)
CONV1D_OP(float, conv1d_f32)

// ============================================================================
// CONV2D OPERATIONS
// ============================================================================

// 2D Convolution
// Input: (batch, in_channels, height, width)
// Weight: (out_channels, in_channels, kernel_height, kernel_width)
// Output: (batch, out_channels, out_height, out_width)
//
// Metadata layout:
// - batch, in_channels, out_channels
// - in_height, in_width, kernel_height, kernel_width
// - out_height, out_width
// - stride_h, stride_w, padding_h, padding_w
// - dilation_h, dilation_w
// - input_offset, weight_offset

#define CONV2D_OP(TYPENAME, FN_NAME)                                                               \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device TYPENAME *weight [[buffer(1)]],   \
        device TYPENAME *output [[buffer(2)]], constant size_t &num_els [[buffer(3)]],             \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        (void)metadata[0]; /* batch - unused, computed from idx */                                 \
        const size_t in_channels = metadata[1];                                                    \
        const size_t out_channels = metadata[2];                                                   \
        const size_t in_height = metadata[3];                                                      \
        const size_t in_width = metadata[4];                                                       \
        const size_t kernel_height = metadata[5];                                                  \
        const size_t kernel_width = metadata[6];                                                   \
        const size_t out_height = metadata[7];                                                     \
        const size_t out_width = metadata[8];                                                      \
        const size_t stride_h = metadata[9];                                                       \
        const size_t stride_w = metadata[10];                                                      \
        const size_t padding_h = metadata[11];                                                     \
        const size_t padding_w = metadata[12];                                                     \
        const size_t dilation_h = metadata[13];                                                    \
        const size_t dilation_w = metadata[14];                                                    \
        const size_t input_offset = metadata[15];                                                  \
        const size_t weight_offset = metadata[16];                                                 \
                                                                                                   \
        for (uint idx = thread_index; idx < num_els; idx += threads_per_grid) {                    \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t oc = (idx / (out_width * out_height)) % out_channels;                     \
            const size_t b = idx / (out_width * out_height * out_channels);                        \
                                                                                                   \
            TYPENAME sum = 0;                                                                      \
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

CONV2D_OP(bfloat, conv2d_bf16)
CONV2D_OP(half, conv2d_f16)
CONV2D_OP(float, conv2d_f32)

// ============================================================================
// CONV3D OPERATIONS
// ============================================================================

// 3D Convolution
// Input: (batch, in_channels, depth, height, width)
// Weight: (out_channels, in_channels, kernel_depth, kernel_height, kernel_width)
// Output: (batch, out_channels, out_depth, out_height, out_width)
//
// Metadata layout:
// - batch, in_channels, out_channels
// - in_depth, in_height, in_width
// - kernel_depth, kernel_height, kernel_width
// - out_depth, out_height, out_width
// - stride_d, stride_h, stride_w
// - padding_d, padding_h, padding_w
// - dilation_d, dilation_h, dilation_w
// - input_offset, weight_offset

#define CONV3D_OP(TYPENAME, FN_NAME)                                                               \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device TYPENAME *weight [[buffer(1)]],   \
        device TYPENAME *output [[buffer(2)]], constant size_t &num_els [[buffer(3)]],             \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        (void)metadata[0]; /* batch - unused, computed from idx */                                 \
        const size_t in_channels = metadata[1];                                                    \
        const size_t out_channels = metadata[2];                                                   \
        const size_t in_depth = metadata[3];                                                       \
        const size_t in_height = metadata[4];                                                      \
        const size_t in_width = metadata[5];                                                       \
        const size_t kernel_depth = metadata[6];                                                   \
        const size_t kernel_height = metadata[7];                                                  \
        const size_t kernel_width = metadata[8];                                                   \
        const size_t out_depth = metadata[9];                                                      \
        const size_t out_height = metadata[10];                                                    \
        const size_t out_width = metadata[11];                                                     \
        const size_t stride_d = metadata[12];                                                      \
        const size_t stride_h = metadata[13];                                                      \
        const size_t stride_w = metadata[14];                                                      \
        const size_t padding_d = metadata[15];                                                     \
        const size_t padding_h = metadata[16];                                                     \
        const size_t padding_w = metadata[17];                                                     \
        const size_t dilation_d = metadata[18];                                                    \
        const size_t dilation_h = metadata[19];                                                    \
        const size_t dilation_w = metadata[20];                                                    \
        const size_t input_offset = metadata[21];                                                  \
        const size_t weight_offset = metadata[22];                                                 \
                                                                                                   \
        for (uint idx = thread_index; idx < num_els; idx += threads_per_grid) {                    \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t od = (idx / (out_width * out_height)) % out_depth;                        \
            const size_t oc = (idx / (out_width * out_height * out_depth)) % out_channels;         \
            const size_t b = idx / (out_width * out_height * out_depth * out_channels);            \
                                                                                                   \
            TYPENAME sum = 0;                                                                      \
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

CONV3D_OP(bfloat, conv3d_bf16)
CONV3D_OP(half, conv3d_f16)
CONV3D_OP(float, conv3d_f32)

// ============================================================================
// CONV_TRANSPOSE1D OPERATIONS
// ============================================================================

// 1D Transposed Convolution (Deconvolution)
// Input: (batch, in_channels, width)
// Weight: (in_channels, out_channels, kernel_width)
// Output: (batch, out_channels, out_width)
//
// Metadata layout:
// - batch, in_channels, out_channels
// - in_width, kernel_width, out_width
// - stride, padding, dilation
// - input_offset, weight_offset

#define CONV_TRANSPOSE1D_OP(TYPENAME, FN_NAME)                                                     \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device TYPENAME *weight [[buffer(1)]],   \
        device TYPENAME *output [[buffer(2)]], constant size_t &num_els [[buffer(3)]],             \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        (void)metadata[0]; /* batch - unused, computed from idx */                                 \
        const size_t in_channels = metadata[1];                                                    \
        const size_t out_channels = metadata[2];                                                   \
        const size_t in_width = metadata[3];                                                       \
        const size_t kernel_width = metadata[4];                                                   \
        const size_t out_width = metadata[5];                                                      \
        const size_t stride = metadata[6];                                                         \
        const size_t padding = metadata[7];                                                        \
        const size_t dilation = metadata[8];                                                       \
        const size_t input_offset = metadata[9];                                                   \
        const size_t weight_offset = metadata[10];                                                 \
                                                                                                   \
        for (uint idx = thread_index; idx < num_els; idx += threads_per_grid) {                    \
            const size_t ow = idx % out_width;                                                     \
            const size_t oc = (idx / out_width) % out_channels;                                    \
            const size_t b = idx / (out_width * out_channels);                                     \
                                                                                                   \
            TYPENAME sum = 0;                                                                      \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t kw = 0; kw < kernel_width; kw++) {                                     \
                    const int tmp = (int)ow + (int)padding - (int)(kw * dilation);                 \
                    if (tmp % (int)stride == 0) {                                                  \
                        const int iw = tmp / (int)stride;                                          \
                        if (iw >= 0 && iw < (int)in_width) {                                       \
                            const size_t input_idx =                                               \
                                input_offset + b * in_channels * in_width + ic * in_width + iw;    \
                            const size_t weight_idx = weight_offset +                              \
                                                      ic * out_channels * kernel_width +           \
                                                      oc * kernel_width + kw;                      \
                            sum += input[input_idx] * weight[weight_idx];                          \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            output[idx] = sum;                                                                     \
        }                                                                                          \
    }

CONV_TRANSPOSE1D_OP(bfloat, conv_transpose1d_bf16)
CONV_TRANSPOSE1D_OP(half, conv_transpose1d_f16)
CONV_TRANSPOSE1D_OP(float, conv_transpose1d_f32)

// ============================================================================
// CONV_TRANSPOSE2D OPERATIONS
// ============================================================================

// 2D Transposed Convolution (Deconvolution)
// Input: (batch, in_channels, height, width)
// Weight: (in_channels, out_channels, kernel_height, kernel_width)
// Output: (batch, out_channels, out_height, out_width)
//
// Metadata layout:
// - batch, in_channels, out_channels
// - in_height, in_width, kernel_height, kernel_width
// - out_height, out_width
// - stride_h, stride_w, padding_h, padding_w
// - dilation_h, dilation_w
// - input_offset, weight_offset

#define CONV_TRANSPOSE2D_OP(TYPENAME, FN_NAME)                                                     \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device TYPENAME *weight [[buffer(1)]],   \
        device TYPENAME *output [[buffer(2)]], constant size_t &num_els [[buffer(3)]],             \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        (void)metadata[0]; /* batch - unused, computed from idx */                                 \
        const size_t in_channels = metadata[1];                                                    \
        const size_t out_channels = metadata[2];                                                   \
        const size_t in_height = metadata[3];                                                      \
        const size_t in_width = metadata[4];                                                       \
        const size_t kernel_height = metadata[5];                                                  \
        const size_t kernel_width = metadata[6];                                                   \
        const size_t out_height = metadata[7];                                                     \
        const size_t out_width = metadata[8];                                                      \
        const size_t stride_h = metadata[9];                                                       \
        const size_t stride_w = metadata[10];                                                      \
        const size_t padding_h = metadata[11];                                                     \
        const size_t padding_w = metadata[12];                                                     \
        const size_t dilation_h = metadata[13];                                                    \
        const size_t dilation_w = metadata[14];                                                    \
        const size_t input_offset = metadata[15];                                                  \
        const size_t weight_offset = metadata[16];                                                 \
                                                                                                   \
        for (uint idx = thread_index; idx < num_els; idx += threads_per_grid) {                    \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t oc = (idx / (out_width * out_height)) % out_channels;                     \
            const size_t b = idx / (out_width * out_height * out_channels);                        \
                                                                                                   \
            TYPENAME sum = 0;                                                                      \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t kh = 0; kh < kernel_height; kh++) {                                    \
                    for (size_t kw = 0; kw < kernel_width; kw++) {                                 \
                        const int tmp_h = (int)oh + (int)padding_h - (int)(kh * dilation_h);       \
                        const int tmp_w = (int)ow + (int)padding_w - (int)(kw * dilation_w);       \
                        if (tmp_h % (int)stride_h == 0 && tmp_w % (int)stride_w == 0) {            \
                            const int ih = tmp_h / (int)stride_h;                                  \
                            const int iw = tmp_w / (int)stride_w;                                  \
                            if (ih >= 0 && ih < (int)in_height && iw >= 0 && iw < (int)in_width) { \
                                const size_t input_idx =                                           \
                                    input_offset + b * in_channels * in_height * in_width +        \
                                    ic * in_height * in_width + ih * in_width + iw;                \
                                const size_t weight_idx =                                          \
                                    weight_offset +                                                \
                                    ic * out_channels * kernel_height * kernel_width +             \
                                    oc * kernel_height * kernel_width + kh * kernel_width + kw;    \
                                sum += input[input_idx] * weight[weight_idx];                      \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            output[idx] = sum;                                                                     \
        }                                                                                          \
    }

CONV_TRANSPOSE2D_OP(bfloat, conv_transpose2d_bf16)
CONV_TRANSPOSE2D_OP(half, conv_transpose2d_f16)
CONV_TRANSPOSE2D_OP(float, conv_transpose2d_f32)

// ============================================================================
// CONV_TRANSPOSE3D OPERATIONS
// ============================================================================

// 3D Transposed Convolution (Deconvolution)
// Input: (batch, in_channels, depth, height, width)
// Weight: (in_channels, out_channels, kernel_depth, kernel_height, kernel_width)
// Output: (batch, out_channels, out_depth, out_height, out_width)
//
// Metadata layout:
// - batch, in_channels, out_channels
// - in_depth, in_height, in_width
// - kernel_depth, kernel_height, kernel_width
// - out_depth, out_height, out_width
// - stride_d, stride_h, stride_w
// - padding_d, padding_h, padding_w
// - dilation_d, dilation_h, dilation_w
// - input_offset, weight_offset

#define CONV_TRANSPOSE3D_OP(TYPENAME, FN_NAME)                                                     \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device TYPENAME *weight [[buffer(1)]],   \
        device TYPENAME *output [[buffer(2)]], constant size_t &num_els [[buffer(3)]],             \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        (void)metadata[0]; /* batch - unused, computed from idx */                                 \
        const size_t in_channels = metadata[1];                                                    \
        const size_t out_channels = metadata[2];                                                   \
        const size_t in_depth = metadata[3];                                                       \
        const size_t in_height = metadata[4];                                                      \
        const size_t in_width = metadata[5];                                                       \
        const size_t kernel_depth = metadata[6];                                                   \
        const size_t kernel_height = metadata[7];                                                  \
        const size_t kernel_width = metadata[8];                                                   \
        const size_t out_depth = metadata[9];                                                      \
        const size_t out_height = metadata[10];                                                    \
        const size_t out_width = metadata[11];                                                     \
        const size_t stride_d = metadata[12];                                                      \
        const size_t stride_h = metadata[13];                                                      \
        const size_t stride_w = metadata[14];                                                      \
        const size_t padding_d = metadata[15];                                                     \
        const size_t padding_h = metadata[16];                                                     \
        const size_t padding_w = metadata[17];                                                     \
        const size_t dilation_d = metadata[18];                                                    \
        const size_t dilation_h = metadata[19];                                                    \
        const size_t dilation_w = metadata[20];                                                    \
        const size_t input_offset = metadata[21];                                                  \
        const size_t weight_offset = metadata[22];                                                 \
                                                                                                   \
        for (uint idx = thread_index; idx < num_els; idx += threads_per_grid) {                    \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t od = (idx / (out_width * out_height)) % out_depth;                        \
            const size_t oc = (idx / (out_width * out_height * out_depth)) % out_channels;         \
            const size_t b = idx / (out_width * out_height * out_depth * out_channels);            \
                                                                                                   \
            TYPENAME sum = 0;                                                                      \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t kd = 0; kd < kernel_depth; kd++) {                                     \
                    for (size_t kh = 0; kh < kernel_height; kh++) {                                \
                        for (size_t kw = 0; kw < kernel_width; kw++) {                             \
                            const int tmp_d = (int)od + (int)padding_d - (int)(kd * dilation_d);   \
                            const int tmp_h = (int)oh + (int)padding_h - (int)(kh * dilation_h);   \
                            const int tmp_w = (int)ow + (int)padding_w - (int)(kw * dilation_w);   \
                            if (tmp_d % (int)stride_d == 0 && tmp_h % (int)stride_h == 0 &&        \
                                tmp_w % (int)stride_w == 0) {                                      \
                                const int id = tmp_d / (int)stride_d;                              \
                                const int ih = tmp_h / (int)stride_h;                              \
                                const int iw = tmp_w / (int)stride_w;                              \
                                if (id >= 0 && id < (int)in_depth && ih >= 0 &&                    \
                                    ih < (int)in_height && iw >= 0 && iw < (int)in_width) {        \
                                    const size_t input_idx =                                       \
                                        input_offset +                                             \
                                        b * in_channels * in_depth * in_height * in_width +        \
                                        ic * in_depth * in_height * in_width +                     \
                                        id * in_height * in_width + ih * in_width + iw;            \
                                    const size_t weight_idx =                                      \
                                        weight_offset +                                            \
                                        ic * out_channels * kernel_depth * kernel_height *         \
                                            kernel_width +                                         \
                                        oc * kernel_depth * kernel_height * kernel_width +         \
                                        kd * kernel_height * kernel_width + kh * kernel_width +    \
                                        kw;                                                        \
                                    sum += input[input_idx] * weight[weight_idx];                  \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            output[idx] = sum;                                                                     \
        }                                                                                          \
    }

CONV_TRANSPOSE3D_OP(bfloat, conv_transpose3d_bf16)
CONV_TRANSPOSE3D_OP(half, conv_transpose3d_f16)
CONV_TRANSPOSE3D_OP(float, conv_transpose3d_f32)

// ============================================================================
// CONV1D_GRAD_WEIGHT OPERATIONS
// ============================================================================

// 1D Convolution Weight Gradient
// Input: (batch, in_channels, width)
// Grad_output: (batch, out_channels, out_width)
// Grad_weight: (out_channels, in_channels, kernel_width)
//
// Metadata layout:
// - batch, in_channels, out_channels
// - in_width, kernel_width, out_width
// - stride, padding, dilation
// - input_offset, grad_output_offset
//
// Note: Uses atomic operations for parallel reduction across batch and spatial dimensions

#define CONV1D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME)                                                   \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]],                                                \
        const device TYPENAME *grad_output [[buffer(1)]],                                          \
        device TYPENAME *grad_weight [[buffer(2)]], constant size_t &num_els [[buffer(3)]],        \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t batch = metadata[0];                                                          \
        const size_t in_channels = metadata[1];                                                    \
        const size_t out_channels = metadata[2];                                                   \
        const size_t in_width = metadata[3];                                                       \
        const size_t kernel_width = metadata[4];                                                   \
        const size_t out_width = metadata[5];                                                      \
        const size_t stride = metadata[6];                                                         \
        const size_t padding = metadata[7];                                                        \
        const size_t dilation = metadata[8];                                                       \
        const size_t input_offset = metadata[9];                                                   \
        const size_t grad_output_offset = metadata[10];                                            \
                                                                                                   \
        /* Parallelize over batch * out_channels * out_width */                                    \
        const size_t total_output_els = batch * out_channels * out_width;                          \
        for (uint idx = thread_index; idx < total_output_els; idx += threads_per_grid) {           \
            const size_t ow = idx % out_width;                                                     \
            const size_t oc = (idx / out_width) % out_channels;                                    \
            const size_t b = idx / (out_width * out_channels);                                     \
                                                                                                   \
            const size_t grad_output_idx =                                                         \
                grad_output_offset + b * out_channels * out_width + oc * out_width + ow;           \
            const TYPENAME grad_out_val = grad_output[grad_output_idx];                            \
                                                                                                   \
            /* Contribute to all relevant weight gradients */                                      \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t kw = 0; kw < kernel_width; kw++) {                                     \
                    const int iw = (int)(ow * stride) - (int)padding + (int)(kw * dilation);       \
                    if (iw >= 0 && iw < (int)in_width) {                                           \
                        const size_t input_idx =                                                   \
                            input_offset + b * in_channels * in_width + ic * in_width + iw;        \
                        const size_t weight_idx =                                                  \
                            oc * in_channels * kernel_width + ic * kernel_width + kw;              \
                        const TYPENAME contribution = input[input_idx] * grad_out_val;             \
                        atomic_add_wrapper(&grad_weight[weight_idx], contribution);                \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV1D_GRAD_WEIGHT_OP(bfloat, conv1d_grad_weight_bf16)
CONV1D_GRAD_WEIGHT_OP(half, conv1d_grad_weight_f16)
CONV1D_GRAD_WEIGHT_OP(float, conv1d_grad_weight_f32)

// ============================================================================
// CONV2D_GRAD_WEIGHT OPERATIONS
// ============================================================================

// 2D Convolution Weight Gradient
// Input: (batch, in_channels, height, width)
// Grad_output: (batch, out_channels, out_height, out_width)
// Grad_weight: (out_channels, in_channels, kernel_height, kernel_width)
//
// Metadata layout:
// - batch, in_channels, out_channels
// - in_height, in_width, kernel_height, kernel_width
// - out_height, out_width
// - stride_h, stride_w, padding_h, padding_w
// - dilation_h, dilation_w
// - input_offset, grad_output_offset
//
// Note: Uses atomic operations for parallel reduction across batch and spatial dimensions

#define CONV2D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME)                                                   \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]],                                                \
        const device TYPENAME *grad_output [[buffer(1)]],                                          \
        device TYPENAME *grad_weight [[buffer(2)]], constant size_t &num_els [[buffer(3)]],        \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t batch = metadata[0];                                                          \
        const size_t in_channels = metadata[1];                                                    \
        const size_t out_channels = metadata[2];                                                   \
        const size_t in_height = metadata[3];                                                      \
        const size_t in_width = metadata[4];                                                       \
        const size_t kernel_height = metadata[5];                                                  \
        const size_t kernel_width = metadata[6];                                                   \
        const size_t out_height = metadata[7];                                                     \
        const size_t out_width = metadata[8];                                                      \
        const size_t stride_h = metadata[9];                                                       \
        const size_t stride_w = metadata[10];                                                      \
        const size_t padding_h = metadata[11];                                                     \
        const size_t padding_w = metadata[12];                                                     \
        const size_t dilation_h = metadata[13];                                                    \
        const size_t dilation_w = metadata[14];                                                    \
        const size_t input_offset = metadata[15];                                                  \
        const size_t grad_output_offset = metadata[16];                                            \
                                                                                                   \
        /* Parallelize over batch * out_channels * out_height * out_width */                       \
        const size_t total_output_els = batch * out_channels * out_height * out_width;             \
        for (uint idx = thread_index; idx < total_output_els; idx += threads_per_grid) {           \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t oc = (idx / (out_width * out_height)) % out_channels;                     \
            const size_t b = idx / (out_width * out_height * out_channels);                        \
                                                                                                   \
            const size_t grad_output_idx = grad_output_offset +                                    \
                                           b * out_channels * out_height * out_width +             \
                                           oc * out_height * out_width + oh * out_width + ow;      \
            const TYPENAME grad_out_val = grad_output[grad_output_idx];                            \
                                                                                                   \
            /* Contribute to all relevant weight gradients */                                      \
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
                                oc * in_channels * kernel_height * kernel_width +                  \
                                ic * kernel_height * kernel_width + kh * kernel_width + kw;        \
                            const TYPENAME contribution = input[input_idx] * grad_out_val;         \
                            atomic_add_wrapper(&grad_weight[weight_idx], contribution);            \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV2D_GRAD_WEIGHT_OP(bfloat, conv2d_grad_weight_bf16)
CONV2D_GRAD_WEIGHT_OP(half, conv2d_grad_weight_f16)
CONV2D_GRAD_WEIGHT_OP(float, conv2d_grad_weight_f32)

// ============================================================================
// CONV3D_GRAD_WEIGHT OPERATIONS
// ============================================================================

// 3D Convolution Weight Gradient
// Input: (batch, in_channels, depth, height, width)
// Grad_output: (batch, out_channels, out_depth, out_height, out_width)
// Grad_weight: (out_channels, in_channels, kernel_depth, kernel_height, kernel_width)
//
// Metadata layout:
// - batch, in_channels, out_channels
// - in_depth, in_height, in_width
// - kernel_depth, kernel_height, kernel_width
// - out_depth, out_height, out_width
// - stride_d, stride_h, stride_w
// - padding_d, padding_h, padding_w
// - dilation_d, dilation_h, dilation_w
// - input_offset, grad_output_offset
//
// Note: Uses atomic operations for parallel reduction across batch and spatial dimensions

#define CONV3D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME)                                                   \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]],                                                \
        const device TYPENAME *grad_output [[buffer(1)]],                                          \
        device TYPENAME *grad_weight [[buffer(2)]], constant size_t &num_els [[buffer(3)]],        \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t batch = metadata[0];                                                          \
        const size_t in_channels = metadata[1];                                                    \
        const size_t out_channels = metadata[2];                                                   \
        const size_t in_depth = metadata[3];                                                       \
        const size_t in_height = metadata[4];                                                      \
        const size_t in_width = metadata[5];                                                       \
        const size_t kernel_depth = metadata[6];                                                   \
        const size_t kernel_height = metadata[7];                                                  \
        const size_t kernel_width = metadata[8];                                                   \
        const size_t out_depth = metadata[9];                                                      \
        const size_t out_height = metadata[10];                                                    \
        const size_t out_width = metadata[11];                                                     \
        const size_t stride_d = metadata[12];                                                      \
        const size_t stride_h = metadata[13];                                                      \
        const size_t stride_w = metadata[14];                                                      \
        const size_t padding_d = metadata[15];                                                     \
        const size_t padding_h = metadata[16];                                                     \
        const size_t padding_w = metadata[17];                                                     \
        const size_t dilation_d = metadata[18];                                                    \
        const size_t dilation_h = metadata[19];                                                    \
        const size_t dilation_w = metadata[20];                                                    \
        const size_t input_offset = metadata[21];                                                  \
        const size_t grad_output_offset = metadata[22];                                            \
                                                                                                   \
        /* Parallelize over batch * out_channels * out_depth * out_height * out_width */           \
        const size_t total_output_els = batch * out_channels * out_depth * out_height * out_width; \
        for (uint idx = thread_index; idx < total_output_els; idx += threads_per_grid) {           \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t od = (idx / (out_width * out_height)) % out_depth;                        \
            const size_t oc = (idx / (out_width * out_height * out_depth)) % out_channels;         \
            const size_t b = idx / (out_width * out_height * out_depth * out_channels);            \
                                                                                                   \
            const size_t grad_output_idx = grad_output_offset +                                    \
                                           b * out_channels * out_depth * out_height * out_width + \
                                           oc * out_depth * out_height * out_width +               \
                                           od * out_height * out_width + oh * out_width + ow;      \
            const TYPENAME grad_out_val = grad_output[grad_output_idx];                            \
                                                                                                   \
            /* Contribute to all relevant weight gradients */                                      \
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
                                    oc * in_channels * kernel_depth * kernel_height *              \
                                        kernel_width +                                             \
                                    ic * kernel_depth * kernel_height * kernel_width +             \
                                    kd * kernel_height * kernel_width + kh * kernel_width + kw;    \
                                const TYPENAME contribution = input[input_idx] * grad_out_val;     \
                                atomic_add_wrapper(&grad_weight[weight_idx], contribution);        \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV3D_GRAD_WEIGHT_OP(bfloat, conv3d_grad_weight_bf16)
CONV3D_GRAD_WEIGHT_OP(half, conv3d_grad_weight_f16)
CONV3D_GRAD_WEIGHT_OP(float, conv3d_grad_weight_f32)

// ============================================================================
// CONV_TRANSPOSE1D_GRAD_WEIGHT OPERATIONS
// ============================================================================

// 1D Transposed Convolution Weight Gradient
// Input: (batch, in_channels, width)
// Grad_output: (batch, out_channels, out_width)
// Grad_weight: (in_channels, out_channels, kernel_width)
//
// Metadata layout:
// - batch, in_channels, out_channels
// - in_width, kernel_width, out_width
// - stride, padding, dilation
// - input_offset, grad_output_offset
//
// Note: Uses atomic operations for parallel reduction across batch and spatial dimensions

#define CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME)                                         \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]],                                                \
        const device TYPENAME *grad_output [[buffer(1)]],                                          \
        device TYPENAME *grad_weight [[buffer(2)]], constant size_t &num_els [[buffer(3)]],        \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t batch = metadata[0];                                                          \
        const size_t in_channels = metadata[1];                                                    \
        const size_t out_channels = metadata[2];                                                   \
        const size_t in_width = metadata[3];                                                       \
        const size_t kernel_width = metadata[4];                                                   \
        const size_t out_width = metadata[5];                                                      \
        const size_t stride = metadata[6];                                                         \
        const size_t padding = metadata[7];                                                        \
        const size_t dilation = metadata[8];                                                       \
        const size_t input_offset = metadata[9];                                                   \
        const size_t grad_output_offset = metadata[10];                                            \
                                                                                                   \
        /* Parallelize over batch * out_channels * out_width */                                    \
        const size_t total_output_els = batch * out_channels * out_width;                          \
        for (uint idx = thread_index; idx < total_output_els; idx += threads_per_grid) {           \
            const size_t ow = idx % out_width;                                                     \
            const size_t oc = (idx / out_width) % out_channels;                                    \
            const size_t b = idx / (out_width * out_channels);                                     \
                                                                                                   \
            const size_t grad_output_idx =                                                         \
                grad_output_offset + b * out_channels * out_width + oc * out_width + ow;           \
            const TYPENAME grad_out_val = grad_output[grad_output_idx];                            \
                                                                                                   \
            /* Contribute to all relevant weight gradients */                                      \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t kw = 0; kw < kernel_width; kw++) {                                     \
                    const int tmp = (int)ow + (int)padding - (int)(kw * dilation);                 \
                    if (tmp % (int)stride == 0) {                                                  \
                        const int iw = tmp / (int)stride;                                          \
                        if (iw >= 0 && iw < (int)in_width) {                                       \
                            const size_t input_idx =                                               \
                                input_offset + b * in_channels * in_width + ic * in_width + iw;    \
                            const size_t weight_idx =                                              \
                                ic * out_channels * kernel_width + oc * kernel_width + kw;         \
                            const TYPENAME contribution = input[input_idx] * grad_out_val;         \
                            atomic_add_wrapper(&grad_weight[weight_idx], contribution);            \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(bfloat, conv_transpose1d_grad_weight_bf16)
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(half, conv_transpose1d_grad_weight_f16)
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(float, conv_transpose1d_grad_weight_f32)

// ============================================================================
// CONV_TRANSPOSE2D_GRAD_WEIGHT OPERATIONS
// ============================================================================

// 2D Transposed Convolution Weight Gradient
// Input: (batch, in_channels, height, width)
// Grad_output: (batch, out_channels, out_height, out_width)
// Grad_weight: (in_channels, out_channels, kernel_height, kernel_width)
//
// Metadata layout:
// - batch, in_channels, out_channels
// - in_height, in_width, kernel_height, kernel_width
// - out_height, out_width
// - stride_h, stride_w, padding_h, padding_w
// - dilation_h, dilation_w
// - input_offset, grad_output_offset
//
// Note: Uses atomic operations for parallel reduction across batch and spatial dimensions

#define CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME)                                         \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]],                                                \
        const device TYPENAME *grad_output [[buffer(1)]],                                          \
        device TYPENAME *grad_weight [[buffer(2)]], constant size_t &num_els [[buffer(3)]],        \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t batch = metadata[0];                                                          \
        const size_t in_channels = metadata[1];                                                    \
        const size_t out_channels = metadata[2];                                                   \
        const size_t in_height = metadata[3];                                                      \
        const size_t in_width = metadata[4];                                                       \
        const size_t kernel_height = metadata[5];                                                  \
        const size_t kernel_width = metadata[6];                                                   \
        const size_t out_height = metadata[7];                                                     \
        const size_t out_width = metadata[8];                                                      \
        const size_t stride_h = metadata[9];                                                       \
        const size_t stride_w = metadata[10];                                                      \
        const size_t padding_h = metadata[11];                                                     \
        const size_t padding_w = metadata[12];                                                     \
        const size_t dilation_h = metadata[13];                                                    \
        const size_t dilation_w = metadata[14];                                                    \
        const size_t input_offset = metadata[15];                                                  \
        const size_t grad_output_offset = metadata[16];                                            \
                                                                                                   \
        /* Parallelize over batch * out_channels * out_height * out_width */                       \
        const size_t total_output_els = batch * out_channels * out_height * out_width;             \
        for (uint idx = thread_index; idx < total_output_els; idx += threads_per_grid) {           \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t oc = (idx / (out_width * out_height)) % out_channels;                     \
            const size_t b = idx / (out_width * out_height * out_channels);                        \
                                                                                                   \
            const size_t grad_output_idx = grad_output_offset +                                    \
                                           b * out_channels * out_height * out_width +             \
                                           oc * out_height * out_width + oh * out_width + ow;      \
            const TYPENAME grad_out_val = grad_output[grad_output_idx];                            \
                                                                                                   \
            /* Contribute to all relevant weight gradients */                                      \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t kh = 0; kh < kernel_height; kh++) {                                    \
                    for (size_t kw = 0; kw < kernel_width; kw++) {                                 \
                        const int tmp_h = (int)oh + (int)padding_h - (int)(kh * dilation_h);       \
                        const int tmp_w = (int)ow + (int)padding_w - (int)(kw * dilation_w);       \
                        if (tmp_h % (int)stride_h == 0 && tmp_w % (int)stride_w == 0) {            \
                            const int ih = tmp_h / (int)stride_h;                                  \
                            const int iw = tmp_w / (int)stride_w;                                  \
                            if (ih >= 0 && ih < (int)in_height && iw >= 0 && iw < (int)in_width) { \
                                const size_t input_idx =                                           \
                                    input_offset + b * in_channels * in_height * in_width +        \
                                    ic * in_height * in_width + ih * in_width + iw;                \
                                const size_t weight_idx =                                          \
                                    ic * out_channels * kernel_height * kernel_width +             \
                                    oc * kernel_height * kernel_width + kh * kernel_width + kw;    \
                                const TYPENAME contribution = input[input_idx] * grad_out_val;     \
                                atomic_add_wrapper(&grad_weight[weight_idx], contribution);        \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(bfloat, conv_transpose2d_grad_weight_bf16)
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(half, conv_transpose2d_grad_weight_f16)
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(float, conv_transpose2d_grad_weight_f32)

// ============================================================================
// CONV_TRANSPOSE3D_GRAD_WEIGHT OPERATIONS
// ============================================================================

// 3D Transposed Convolution Weight Gradient
// Input: (batch, in_channels, depth, height, width)
// Grad_output: (batch, out_channels, out_depth, out_height, out_width)
// Grad_weight: (in_channels, out_channels, kernel_depth, kernel_height, kernel_width)
//
// Metadata layout:
// - batch, in_channels, out_channels
// - in_depth, in_height, in_width
// - kernel_depth, kernel_height, kernel_width
// - out_depth, out_height, out_width
// - stride_d, stride_h, stride_w
// - padding_d, padding_h, padding_w
// - dilation_d, dilation_h, dilation_w
// - input_offset, grad_output_offset
//
// Note: Uses atomic operations for parallel reduction across batch and spatial dimensions

#define CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME)                                         \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]],                                                \
        const device TYPENAME *grad_output [[buffer(1)]],                                          \
        device TYPENAME *grad_weight [[buffer(2)]], constant size_t &num_els [[buffer(3)]],        \
        constant size_t *metadata [[buffer(4)]], uint thread_index [[thread_position_in_grid]],    \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t batch = metadata[0];                                                          \
        const size_t in_channels = metadata[1];                                                    \
        const size_t out_channels = metadata[2];                                                   \
        const size_t in_depth = metadata[3];                                                       \
        const size_t in_height = metadata[4];                                                      \
        const size_t in_width = metadata[5];                                                       \
        const size_t kernel_depth = metadata[6];                                                   \
        const size_t kernel_height = metadata[7];                                                  \
        const size_t kernel_width = metadata[8];                                                   \
        const size_t out_depth = metadata[9];                                                      \
        const size_t out_height = metadata[10];                                                    \
        const size_t out_width = metadata[11];                                                     \
        const size_t stride_d = metadata[12];                                                      \
        const size_t stride_h = metadata[13];                                                      \
        const size_t stride_w = metadata[14];                                                      \
        const size_t padding_d = metadata[15];                                                     \
        const size_t padding_h = metadata[16];                                                     \
        const size_t padding_w = metadata[17];                                                     \
        const size_t dilation_d = metadata[18];                                                    \
        const size_t dilation_h = metadata[19];                                                    \
        const size_t dilation_w = metadata[20];                                                    \
        const size_t input_offset = metadata[21];                                                  \
        const size_t grad_output_offset = metadata[22];                                            \
                                                                                                   \
        /* Parallelize over batch * out_channels * out_depth * out_height * out_width */           \
        const size_t total_output_els = batch * out_channels * out_depth * out_height * out_width; \
        for (uint idx = thread_index; idx < total_output_els; idx += threads_per_grid) {           \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t od = (idx / (out_width * out_height)) % out_depth;                        \
            const size_t oc = (idx / (out_width * out_height * out_depth)) % out_channels;         \
            const size_t b = idx / (out_width * out_height * out_depth * out_channels);            \
                                                                                                   \
            const size_t grad_output_idx = grad_output_offset +                                    \
                                           b * out_channels * out_depth * out_height * out_width + \
                                           oc * out_depth * out_height * out_width +               \
                                           od * out_height * out_width + oh * out_width + ow;      \
            const TYPENAME grad_out_val = grad_output[grad_output_idx];                            \
                                                                                                   \
            /* Contribute to all relevant weight gradients */                                      \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t kd = 0; kd < kernel_depth; kd++) {                                     \
                    for (size_t kh = 0; kh < kernel_height; kh++) {                                \
                        for (size_t kw = 0; kw < kernel_width; kw++) {                             \
                            const int tmp_d = (int)od + (int)padding_d - (int)(kd * dilation_d);   \
                            const int tmp_h = (int)oh + (int)padding_h - (int)(kh * dilation_h);   \
                            const int tmp_w = (int)ow + (int)padding_w - (int)(kw * dilation_w);   \
                            if (tmp_d % (int)stride_d == 0 && tmp_h % (int)stride_h == 0 &&        \
                                tmp_w % (int)stride_w == 0) {                                      \
                                const int id = tmp_d / (int)stride_d;                              \
                                const int ih = tmp_h / (int)stride_h;                              \
                                const int iw = tmp_w / (int)stride_w;                              \
                                if (id >= 0 && id < (int)in_depth && ih >= 0 &&                    \
                                    ih < (int)in_height && iw >= 0 && iw < (int)in_width) {        \
                                    const size_t input_idx =                                       \
                                        input_offset +                                             \
                                        b * in_channels * in_depth * in_height * in_width +        \
                                        ic * in_depth * in_height * in_width +                     \
                                        id * in_height * in_width + ih * in_width + iw;            \
                                    const size_t weight_idx =                                      \
                                        ic * out_channels * kernel_depth * kernel_height *         \
                                            kernel_width +                                         \
                                        oc * kernel_depth * kernel_height * kernel_width +         \
                                        kd * kernel_height * kernel_width + kh * kernel_width +    \
                                        kw;                                                        \
                                    const TYPENAME contribution = input[input_idx] * grad_out_val; \
                                    atomic_add_wrapper(&grad_weight[weight_idx], contribution);    \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(bfloat, conv_transpose3d_grad_weight_bf16)
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(half, conv_transpose3d_grad_weight_f16)
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(float, conv_transpose3d_grad_weight_f32)
