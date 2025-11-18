#include "./headers/atomic.metal"
#include "./headers/utils.metal"
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

#define CONV1D_OP(TYPENAME, FN_NAME)                                                               \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device TYPENAME *weight [[buffer(1)]],   \
        device TYPENAME *output [[buffer(2)]], constant size_t *metadata [[buffer(3)]],            \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        (void)metadata[1]; /* batch - unused, computed from idx */                                 \
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
// - metadata[17]: weight_offset

#define CONV2D_OP(TYPENAME, FN_NAME)                                                               \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device TYPENAME *weight [[buffer(1)]],   \
        device TYPENAME *output [[buffer(2)]], constant size_t *metadata [[buffer(3)]],            \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        (void)metadata[1]; /* batch - unused, computed from idx */                                 \
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
// - metadata[23]: weight_offset

#define CONV3D_OP(TYPENAME, FN_NAME)                                                               \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], const device TYPENAME *weight [[buffer(1)]],   \
        device TYPENAME *output [[buffer(2)]], constant size_t *metadata [[buffer(3)]],            \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        (void)metadata[1]; /* batch - unused, computed from idx */                                 \
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
        device TYPENAME *output [[buffer(2)]], constant size_t *metadata [[buffer(3)]],            \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        (void)metadata[1]; /* batch - unused, computed from idx */                                 \
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
        device TYPENAME *output [[buffer(2)]], constant size_t *metadata [[buffer(3)]],            \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        (void)metadata[1]; /* batch - unused, computed from idx */                                 \
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
        device TYPENAME *output [[buffer(2)]], constant size_t *metadata [[buffer(3)]],            \
        uint thread_index [[thread_position_in_grid]],                                             \
        uint threads_per_grid [[threads_per_grid]]) {                                              \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        (void)metadata[1]; /* batch - unused, computed from idx */                                 \
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
// Generic metadata layout (input_ndim=3, spatial_dims=1):
// - metadata[0]: num_els
// - metadata[1]: input_ndim
// - metadata[2]: spatial_dims
// - metadata[3..6]: input_shape (batch, in_channels, in_width)
// - metadata[6..9]: grad_output_shape (batch, out_channels, out_width)
// - metadata[9..12]: weight_shape (out_channels, in_channels, kernel_width)
// - metadata[12..15]: input_strides
// - metadata[15..18]: grad_output_strides
// - metadata[18]: input_offset
// - metadata[19]: grad_output_offset
// - metadata[20]: stride
// - metadata[21]: padding
// - metadata[22]: dilation
//
// Note: Uses atomic operations for parallel reduction across batch and spatial dimensions

#define CONV1D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME)                                                   \
    kernel void FN_NAME(const device TYPENAME *input [[buffer(0)]],                                \
                        const device TYPENAME *grad_output [[buffer(1)]],                          \
                        device TYPENAME *grad_weight [[buffer(2)]],                                \
                        constant size_t *metadata [[buffer(3)]],                                   \
                        uint thread_index [[thread_position_in_grid]],                             \
                        uint threads_per_grid [[threads_per_grid]]) {                              \
                                                                                                   \
        /* Parse generic metadata: Conv1D has input_ndim=3, spatial_dims=1 */                      \
        const size_t input_ndim = metadata[1];                                                     \
        const size_t spatial_dims = metadata[2];                                                   \
        const size_t batch = metadata[3];                                                          \
        const size_t in_channels = metadata[4];                                                    \
        const size_t in_width = metadata[5];                                                       \
        const size_t grad_output_base = 3 + input_ndim;                                            \
        const size_t out_channels = metadata[grad_output_base + 1];                                \
        const size_t out_width = metadata[grad_output_base + 2];                                   \
        const size_t weight_base = 3 + 2 * input_ndim;                                             \
        const size_t kernel_width = metadata[weight_base + 2];                                     \
        const size_t grad_output_stride_base = 3 + 4 * input_ndim;                                 \
        const size_t grad_output_stride_batch = metadata[grad_output_stride_base];                 \
        const size_t grad_output_stride_channel = metadata[grad_output_stride_base + 1];           \
        const size_t grad_output_stride_w = metadata[grad_output_stride_base + 2];                 \
        const size_t offsets_base = 3 + 5 * input_ndim;                                            \
        const size_t input_offset = metadata[offsets_base];                                        \
        const size_t grad_output_offset = metadata[offsets_base + 1];                              \
        const size_t conv_params_base = offsets_base + 2;                                          \
        const size_t stride = metadata[conv_params_base];                                          \
        const size_t padding = metadata[conv_params_base + spatial_dims];                          \
        const size_t dilation = metadata[conv_params_base + 2 * spatial_dims];                     \
                                                                                                   \
        /* Parallelize over batch * out_channels * out_width */                                    \
        const size_t total_output_els = batch * out_channels * out_width;                          \
        for (uint idx = thread_index; idx < total_output_els; idx += threads_per_grid) {           \
            const size_t ow = idx % out_width;                                                     \
            const size_t oc = (idx / out_width) % out_channels;                                    \
            const size_t b = idx / (out_width * out_channels);                                     \
                                                                                                   \
            const size_t grad_output_idx = grad_output_offset + b * grad_output_stride_batch +     \
                                           oc * grad_output_stride_channel +                       \
                                           ow * grad_output_stride_w;                              \
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
// Generic metadata layout (input_ndim=4, spatial_dims=2):
// - metadata[0]: num_els
// - metadata[1]: input_ndim
// - metadata[2]: spatial_dims
// - metadata[3..7]: input_shape (batch, in_channels, in_height, in_width)
// - metadata[7..11]: grad_output_shape (batch, out_channels, out_height, out_width)
// - metadata[11..15]: weight_shape (out_channels, in_channels, kernel_height, kernel_width)
// - metadata[15..19]: input_strides
// - metadata[19..23]: grad_output_strides
// - metadata[23]: input_offset
// - metadata[24]: grad_output_offset
// - metadata[25..27]: stride (stride_h, stride_w)
// - metadata[27..29]: padding (padding_h, padding_w)
// - metadata[29..31]: dilation (dilation_h, dilation_w)
//
// Note: Uses atomic operations for parallel reduction across batch and spatial dimensions

#define CONV2D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME)                                                   \
    kernel void FN_NAME(const device TYPENAME *input [[buffer(0)]],                                \
                        const device TYPENAME *grad_output [[buffer(1)]],                          \
                        device TYPENAME *grad_weight [[buffer(2)]],                                \
                        constant size_t *metadata [[buffer(3)]],                                   \
                        uint thread_index [[thread_position_in_grid]],                             \
                        uint threads_per_grid [[threads_per_grid]]) {                              \
                                                                                                   \
        /* Parse generic metadata: Conv2D has input_ndim=4, spatial_dims=2 */                      \
        const size_t input_ndim = metadata[1];                                                     \
        const size_t spatial_dims = metadata[2];                                                   \
        const size_t batch = metadata[3];                                                          \
        const size_t in_channels = metadata[4];                                                    \
        const size_t in_height = metadata[5];                                                      \
        const size_t in_width = metadata[6];                                                       \
        const size_t grad_output_base = 3 + input_ndim;                                            \
        const size_t out_channels = metadata[grad_output_base + 1];                                \
        const size_t out_height = metadata[grad_output_base + 2];                                  \
        const size_t out_width = metadata[grad_output_base + 3];                                   \
        const size_t weight_base = 3 + 2 * input_ndim;                                             \
        const size_t kernel_height = metadata[weight_base + 2];                                    \
        const size_t kernel_width = metadata[weight_base + 3];                                     \
        const size_t grad_output_stride_base = 3 + 4 * input_ndim;                                 \
        const size_t grad_output_stride_batch = metadata[grad_output_stride_base];                 \
        const size_t grad_output_stride_channel = metadata[grad_output_stride_base + 1];           \
        const size_t grad_output_stride_h = metadata[grad_output_stride_base + 2];                 \
        const size_t grad_output_stride_w = metadata[grad_output_stride_base + 3];                 \
        const size_t offsets_base = 3 + 5 * input_ndim;                                            \
        const size_t input_offset = metadata[offsets_base];                                        \
        const size_t grad_output_offset = metadata[offsets_base + 1];                              \
        const size_t conv_params_base = offsets_base + 2;                                          \
        const size_t stride_h = metadata[conv_params_base];                                        \
        const size_t stride_w = metadata[conv_params_base + 1];                                    \
        const size_t padding_h = metadata[conv_params_base + spatial_dims];                        \
        const size_t padding_w = metadata[conv_params_base + spatial_dims + 1];                    \
        const size_t dilation_h = metadata[conv_params_base + 2 * spatial_dims];                   \
        const size_t dilation_w = metadata[conv_params_base + 2 * spatial_dims + 1];               \
                                                                                                   \
        /* Parallelize over batch * out_channels * out_height * out_width */                       \
        const size_t total_output_els = batch * out_channels * out_height * out_width;             \
        for (uint idx = thread_index; idx < total_output_els; idx += threads_per_grid) {           \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t oc = (idx / (out_width * out_height)) % out_channels;                     \
            const size_t b = idx / (out_width * out_height * out_channels);                        \
                                                                                                   \
            const size_t grad_output_idx = grad_output_offset + b * grad_output_stride_batch +     \
                                           oc * grad_output_stride_channel +                       \
                                           oh * grad_output_stride_h + ow * grad_output_stride_w;  \
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
// Generic metadata layout (input_ndim=5, spatial_dims=3):
// - metadata[0]: num_els
// - metadata[1]: input_ndim
// - metadata[2]: spatial_dims
// - metadata[3..8]: input_shape (batch, in_channels, in_depth, in_height, in_width)
// - metadata[8..13]: grad_output_shape (batch, out_channels, out_depth, out_height, out_width)
// - metadata[13..18]: weight_shape (out_channels, in_channels, kernel_depth, kernel_height,
// kernel_width)
// - metadata[18..23]: input_strides
// - metadata[23..28]: grad_output_strides
// - metadata[28]: input_offset
// - metadata[29]: grad_output_offset
// - metadata[30..33]: stride (stride_d, stride_h, stride_w)
// - metadata[33..36]: padding (padding_d, padding_h, padding_w)
// - metadata[36..39]: dilation (dilation_d, dilation_h, dilation_w)
//
// Note: Uses atomic operations for parallel reduction across batch and spatial dimensions

#define CONV3D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME)                                                   \
    kernel void FN_NAME(const device TYPENAME *input [[buffer(0)]],                                \
                        const device TYPENAME *grad_output [[buffer(1)]],                          \
                        device TYPENAME *grad_weight [[buffer(2)]],                                \
                        constant size_t *metadata [[buffer(3)]],                                   \
                        uint thread_index [[thread_position_in_grid]],                             \
                        uint threads_per_grid [[threads_per_grid]]) {                              \
                                                                                                   \
        /* Parse generic metadata: Conv3D has input_ndim=5, spatial_dims=3 */                      \
        const size_t input_ndim = metadata[1];                                                     \
        const size_t spatial_dims = metadata[2];                                                   \
        const size_t batch = metadata[3];                                                          \
        const size_t in_channels = metadata[4];                                                    \
        const size_t in_depth = metadata[5];                                                       \
        const size_t in_height = metadata[6];                                                      \
        const size_t in_width = metadata[7];                                                       \
        const size_t grad_output_base = 3 + input_ndim;                                            \
        const size_t out_channels = metadata[grad_output_base + 1];                                \
        const size_t out_depth = metadata[grad_output_base + 2];                                   \
        const size_t out_height = metadata[grad_output_base + 3];                                  \
        const size_t out_width = metadata[grad_output_base + 4];                                   \
        const size_t weight_base = 3 + 2 * input_ndim;                                             \
        const size_t kernel_depth = metadata[weight_base + 2];                                     \
        const size_t kernel_height = metadata[weight_base + 3];                                    \
        const size_t kernel_width = metadata[weight_base + 4];                                     \
        const size_t grad_output_stride_base = 3 + 4 * input_ndim;                                 \
        const size_t grad_output_stride_batch = metadata[grad_output_stride_base];                 \
        const size_t grad_output_stride_channel = metadata[grad_output_stride_base + 1];           \
        const size_t grad_output_stride_d = metadata[grad_output_stride_base + 2];                 \
        const size_t grad_output_stride_h = metadata[grad_output_stride_base + 3];                 \
        const size_t grad_output_stride_w = metadata[grad_output_stride_base + 4];                 \
        const size_t offsets_base = 3 + 5 * input_ndim;                                            \
        const size_t input_offset = metadata[offsets_base];                                        \
        const size_t grad_output_offset = metadata[offsets_base + 1];                              \
        const size_t conv_params_base = offsets_base + 2;                                          \
        const size_t stride_d = metadata[conv_params_base];                                        \
        const size_t stride_h = metadata[conv_params_base + 1];                                    \
        const size_t stride_w = metadata[conv_params_base + 2];                                    \
        const size_t padding_d = metadata[conv_params_base + spatial_dims];                        \
        const size_t padding_h = metadata[conv_params_base + spatial_dims + 1];                    \
        const size_t padding_w = metadata[conv_params_base + spatial_dims + 2];                    \
        const size_t dilation_d = metadata[conv_params_base + 2 * spatial_dims];                   \
        const size_t dilation_h = metadata[conv_params_base + 2 * spatial_dims + 1];               \
        const size_t dilation_w = metadata[conv_params_base + 2 * spatial_dims + 2];               \
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
            const size_t grad_output_idx = grad_output_offset + b * grad_output_stride_batch +     \
                                           oc * grad_output_stride_channel +                       \
                                           od * grad_output_stride_d + oh * grad_output_stride_h + \
                                           ow * grad_output_stride_w;                              \
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
// Generic metadata layout (input_ndim=3, spatial_dims=1):
// - metadata[0]: num_els
// - metadata[1]: input_ndim
// - metadata[2]: spatial_dims
// - metadata[3..6]: input_shape (batch, in_channels, in_width)
// - metadata[6..9]: grad_output_shape (batch, out_channels, out_width)
// - metadata[9..12]: weight_shape (in_channels, out_channels, kernel_width)
// - metadata[12..15]: input_strides
// - metadata[15..18]: grad_output_strides
// - metadata[18]: input_offset
// - metadata[19]: grad_output_offset
// - metadata[20]: stride
// - metadata[21]: padding
// - metadata[22]: dilation
//
// Note: Uses atomic operations for parallel reduction across batch and spatial dimensions

#define CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME)                                         \
    kernel void FN_NAME(const device TYPENAME *input [[buffer(0)]],                                \
                        const device TYPENAME *grad_output [[buffer(1)]],                          \
                        device TYPENAME *grad_weight [[buffer(2)]],                                \
                        constant size_t *metadata [[buffer(3)]],                                   \
                        uint thread_index [[thread_position_in_grid]],                             \
                        uint threads_per_grid [[threads_per_grid]]) {                              \
                                                                                                   \
        /* Parse generic metadata: ConvTranspose1D has input_ndim=3, spatial_dims=1 */             \
        const size_t input_ndim = metadata[1];                                                     \
        const size_t spatial_dims = metadata[2];                                                   \
        const size_t batch = metadata[3];                                                          \
        const size_t in_channels = metadata[4];                                                    \
        const size_t in_width = metadata[5];                                                       \
        const size_t grad_output_base = 3 + input_ndim;                                            \
        const size_t out_channels = metadata[grad_output_base + 1];                                \
        const size_t out_width = metadata[grad_output_base + 2];                                   \
        const size_t weight_base = 3 + 2 * input_ndim;                                             \
        const size_t kernel_width = metadata[weight_base + 2];                                     \
        const size_t grad_output_stride_base = 3 + 4 * input_ndim;                                 \
        const size_t grad_output_stride_batch = metadata[grad_output_stride_base];                 \
        const size_t grad_output_stride_channel = metadata[grad_output_stride_base + 1];           \
        const size_t grad_output_stride_w = metadata[grad_output_stride_base + 2];                 \
        const size_t offsets_base = 3 + 5 * input_ndim;                                            \
        const size_t input_offset = metadata[offsets_base];                                        \
        const size_t grad_output_offset = metadata[offsets_base + 1];                              \
        const size_t conv_params_base = offsets_base + 2;                                          \
        const size_t stride = metadata[conv_params_base];                                          \
        const size_t padding = metadata[conv_params_base + spatial_dims];                          \
        const size_t dilation = metadata[conv_params_base + 2 * spatial_dims];                     \
                                                                                                   \
        /* Parallelize over batch * out_channels * out_width */                                    \
        const size_t total_output_els = batch * out_channels * out_width;                          \
        for (uint idx = thread_index; idx < total_output_els; idx += threads_per_grid) {           \
            const size_t ow = idx % out_width;                                                     \
            const size_t oc = (idx / out_width) % out_channels;                                    \
            const size_t b = idx / (out_width * out_channels);                                     \
                                                                                                   \
            const size_t grad_output_idx = grad_output_offset + b * grad_output_stride_batch +     \
                                           oc * grad_output_stride_channel +                       \
                                           ow * grad_output_stride_w;                              \
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
// Generic metadata layout (input_ndim=4, spatial_dims=2):
// - metadata[0]: num_els
// - metadata[1]: input_ndim
// - metadata[2]: spatial_dims
// - metadata[3..7]: input_shape (batch, in_channels, in_height, in_width)
// - metadata[7..11]: grad_output_shape (batch, out_channels, out_height, out_width)
// - metadata[11..15]: weight_shape (in_channels, out_channels, kernel_height, kernel_width)
// - metadata[15..19]: input_strides
// - metadata[19..23]: grad_output_strides
// - metadata[23]: input_offset
// - metadata[24]: grad_output_offset
// - metadata[25..27]: stride (stride_h, stride_w)
// - metadata[27..29]: padding (padding_h, padding_w)
// - metadata[29..31]: dilation (dilation_h, dilation_w)
//
// Note: Uses atomic operations for parallel reduction across batch and spatial dimensions

#define CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME)                                         \
    kernel void FN_NAME(const device TYPENAME *input [[buffer(0)]],                                \
                        const device TYPENAME *grad_output [[buffer(1)]],                          \
                        device TYPENAME *grad_weight [[buffer(2)]],                                \
                        constant size_t *metadata [[buffer(3)]],                                   \
                        uint thread_index [[thread_position_in_grid]],                             \
                        uint threads_per_grid [[threads_per_grid]]) {                              \
                                                                                                   \
        /* Parse generic metadata: ConvTranspose2D has input_ndim=4, spatial_dims=2 */             \
        const size_t input_ndim = metadata[1];                                                     \
        const size_t spatial_dims = metadata[2];                                                   \
        const size_t batch = metadata[3];                                                          \
        const size_t in_channels = metadata[4];                                                    \
        const size_t in_height = metadata[5];                                                      \
        const size_t in_width = metadata[6];                                                       \
        const size_t grad_output_base = 3 + input_ndim;                                            \
        const size_t out_channels = metadata[grad_output_base + 1];                                \
        const size_t out_height = metadata[grad_output_base + 2];                                  \
        const size_t out_width = metadata[grad_output_base + 3];                                   \
        const size_t weight_base = 3 + 2 * input_ndim;                                             \
        const size_t kernel_height = metadata[weight_base + 2];                                    \
        const size_t kernel_width = metadata[weight_base + 3];                                     \
        const size_t grad_output_stride_base = 3 + 4 * input_ndim;                                 \
        const size_t grad_output_stride_batch = metadata[grad_output_stride_base];                 \
        const size_t grad_output_stride_channel = metadata[grad_output_stride_base + 1];           \
        const size_t grad_output_stride_h = metadata[grad_output_stride_base + 2];                 \
        const size_t grad_output_stride_w = metadata[grad_output_stride_base + 3];                 \
        const size_t offsets_base = 3 + 5 * input_ndim;                                            \
        const size_t input_offset = metadata[offsets_base];                                        \
        const size_t grad_output_offset = metadata[offsets_base + 1];                              \
        const size_t conv_params_base = offsets_base + 2;                                          \
        const size_t stride_h = metadata[conv_params_base];                                        \
        const size_t stride_w = metadata[conv_params_base + 1];                                    \
        const size_t padding_h = metadata[conv_params_base + spatial_dims];                        \
        const size_t padding_w = metadata[conv_params_base + spatial_dims + 1];                    \
        const size_t dilation_h = metadata[conv_params_base + 2 * spatial_dims];                   \
        const size_t dilation_w = metadata[conv_params_base + 2 * spatial_dims + 1];               \
                                                                                                   \
        /* Parallelize over batch * out_channels * out_height * out_width */                       \
        const size_t total_output_els = batch * out_channels * out_height * out_width;             \
        for (uint idx = thread_index; idx < total_output_els; idx += threads_per_grid) {           \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t oc = (idx / (out_width * out_height)) % out_channels;                     \
            const size_t b = idx / (out_width * out_height * out_channels);                        \
                                                                                                   \
            const size_t grad_output_idx = grad_output_offset + b * grad_output_stride_batch +     \
                                           oc * grad_output_stride_channel +                       \
                                           oh * grad_output_stride_h + ow * grad_output_stride_w;  \
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
// Generic metadata layout (input_ndim=5, spatial_dims=3):
// - metadata[0]: num_els
// - metadata[1]: input_ndim
// - metadata[2]: spatial_dims
// - metadata[3..8]: input_shape (batch, in_channels, in_depth, in_height, in_width)
// - metadata[8..13]: grad_output_shape (batch, out_channels, out_depth, out_height, out_width)
// - metadata[13..18]: weight_shape (in_channels, out_channels, kernel_depth, kernel_height,
// kernel_width)
// - metadata[18..23]: input_strides
// - metadata[23..28]: grad_output_strides
// - metadata[28]: input_offset
// - metadata[29]: grad_output_offset
// - metadata[30..33]: stride (stride_d, stride_h, stride_w)
// - metadata[33..36]: padding (padding_d, padding_h, padding_w)
// - metadata[36..39]: dilation (dilation_d, dilation_h, dilation_w)
//
// Note: Uses atomic operations for parallel reduction across batch and spatial dimensions

#define CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME)                                         \
    kernel void FN_NAME(const device TYPENAME *input [[buffer(0)]],                                \
                        const device TYPENAME *grad_output [[buffer(1)]],                          \
                        device TYPENAME *grad_weight [[buffer(2)]],                                \
                        constant size_t *metadata [[buffer(3)]],                                   \
                        uint thread_index [[thread_position_in_grid]],                             \
                        uint threads_per_grid [[threads_per_grid]]) {                              \
                                                                                                   \
        /* Parse generic metadata: ConvTranspose3D has input_ndim=5, spatial_dims=3 */             \
        const size_t input_ndim = metadata[1];                                                     \
        const size_t spatial_dims = metadata[2];                                                   \
        const size_t batch = metadata[3];                                                          \
        const size_t in_channels = metadata[4];                                                    \
        const size_t in_depth = metadata[5];                                                       \
        const size_t in_height = metadata[6];                                                      \
        const size_t in_width = metadata[7];                                                       \
        const size_t grad_output_base = 3 + input_ndim;                                            \
        const size_t out_channels = metadata[grad_output_base + 1];                                \
        const size_t out_depth = metadata[grad_output_base + 2];                                   \
        const size_t out_height = metadata[grad_output_base + 3];                                  \
        const size_t out_width = metadata[grad_output_base + 4];                                   \
        const size_t weight_base = 3 + 2 * input_ndim;                                             \
        const size_t kernel_depth = metadata[weight_base + 2];                                     \
        const size_t kernel_height = metadata[weight_base + 3];                                    \
        const size_t kernel_width = metadata[weight_base + 4];                                     \
        const size_t grad_output_stride_base = 3 + 4 * input_ndim;                                 \
        const size_t grad_output_stride_batch = metadata[grad_output_stride_base];                 \
        const size_t grad_output_stride_channel = metadata[grad_output_stride_base + 1];           \
        const size_t grad_output_stride_d = metadata[grad_output_stride_base + 2];                 \
        const size_t grad_output_stride_h = metadata[grad_output_stride_base + 3];                 \
        const size_t grad_output_stride_w = metadata[grad_output_stride_base + 4];                 \
        const size_t offsets_base = 3 + 5 * input_ndim;                                            \
        const size_t input_offset = metadata[offsets_base];                                        \
        const size_t grad_output_offset = metadata[offsets_base + 1];                              \
        const size_t conv_params_base = offsets_base + 2;                                          \
        const size_t stride_d = metadata[conv_params_base];                                        \
        const size_t stride_h = metadata[conv_params_base + 1];                                    \
        const size_t stride_w = metadata[conv_params_base + 2];                                    \
        const size_t padding_d = metadata[conv_params_base + spatial_dims];                        \
        const size_t padding_h = metadata[conv_params_base + spatial_dims + 1];                    \
        const size_t padding_w = metadata[conv_params_base + spatial_dims + 2];                    \
        const size_t dilation_d = metadata[conv_params_base + 2 * spatial_dims];                   \
        const size_t dilation_h = metadata[conv_params_base + 2 * spatial_dims + 1];               \
        const size_t dilation_w = metadata[conv_params_base + 2 * spatial_dims + 2];               \
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
            const size_t grad_output_idx = grad_output_offset + b * grad_output_stride_batch +     \
                                           oc * grad_output_stride_channel +                       \
                                           od * grad_output_stride_d + oh * grad_output_stride_h + \
                                           ow * grad_output_stride_w;                              \
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
