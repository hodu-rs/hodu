#include "atomic.cuh"
#include "math.cuh"
#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CONV1D_OP(TYPENAME, FN_NAME)                                                               \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, const TYPENAME *weight,              \
                                       TYPENAME *output, const size_t *metadata) {                 \
        const size_t num_els = metadata[0];                                                        \
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
        for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_els;                  \
             idx += blockDim.x * gridDim.x) {                                                      \
            const size_t ow = idx % out_width;                                                     \
            const size_t oc = (idx / out_width) % out_channels;                                    \
            const size_t b = idx / (out_width * out_channels);                                     \
            float sum = 0.0f;                                                                      \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t kw = 0; kw < kernel_width; kw++) {                                     \
                    const int iw = (int)(ow * stride) - (int)padding + (int)(kw * dilation);       \
                    if (iw >= 0 && iw < (int)in_width) {                                           \
                        const size_t input_idx =                                                   \
                            input_offset + b * in_channels * in_width + ic * in_width + iw;        \
                        const size_t weight_idx = weight_offset +                                  \
                                                  oc * in_channels * kernel_width +                \
                                                  ic * kernel_width + kw;                          \
                        sum += to_float(input[input_idx]) * to_float(weight[weight_idx]);          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            output[idx] = from_float<TYPENAME>(sum);                                               \
        }                                                                                          \
    }

#define CONV2D_OP(TYPENAME, FN_NAME)                                                               \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, const TYPENAME *weight,              \
                                       TYPENAME *output, const size_t *metadata) {                 \
        const size_t num_els = metadata[0];                                                        \
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
        for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_els;                  \
             idx += blockDim.x * gridDim.x) {                                                      \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t oc = (idx / (out_width * out_height)) % out_channels;                     \
            const size_t b = idx / (out_width * out_height * out_channels);                        \
            float sum = 0.0f;                                                                      \
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
                            sum += to_float(input[input_idx]) * to_float(weight[weight_idx]);      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            output[idx] = from_float<TYPENAME>(sum);                                               \
        }                                                                                          \
    }

#define CONV3D_OP(TYPENAME, FN_NAME)                                                               \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, const TYPENAME *weight,              \
                                       TYPENAME *output, const size_t *metadata) {                 \
        const size_t num_els = metadata[0];                                                        \
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
        for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_els;                  \
             idx += blockDim.x * gridDim.x) {                                                      \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t od = (idx / (out_width * out_height)) % out_depth;                        \
            const size_t oc = (idx / (out_width * out_height * out_depth)) % out_channels;         \
            const size_t b = idx / (out_width * out_height * out_depth * out_channels);            \
            float sum = 0.0f;                                                                      \
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
                                sum += to_float(input[input_idx]) * to_float(weight[weight_idx]);  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            output[idx] = from_float<TYPENAME>(sum);                                               \
        }                                                                                          \
    }

#define CONV_TRANSPOSE1D_OP(TYPENAME, FN_NAME)                                                     \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, const TYPENAME *weight,              \
                                       TYPENAME *output, const size_t *metadata) {                 \
        const size_t num_els = metadata[0];                                                        \
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
        for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_els;                  \
             idx += blockDim.x * gridDim.x) {                                                      \
            const size_t ow = idx % out_width;                                                     \
            const size_t oc = (idx / out_width) % out_channels;                                    \
            const size_t b = idx / (out_width * out_channels);                                     \
            float sum = 0.0f;                                                                      \
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
                            sum += to_float(input[input_idx]) * to_float(weight[weight_idx]);      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            output[idx] = from_float<TYPENAME>(sum);                                               \
        }                                                                                          \
    }

#define CONV_TRANSPOSE2D_OP(TYPENAME, FN_NAME)                                                     \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, const TYPENAME *weight,              \
                                       TYPENAME *output, const size_t *metadata) {                 \
        const size_t num_els = metadata[0];                                                        \
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
        for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_els;                  \
             idx += blockDim.x * gridDim.x) {                                                      \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t oc = (idx / (out_width * out_height)) % out_channels;                     \
            const size_t b = idx / (out_width * out_height * out_channels);                        \
            float sum = 0.0f;                                                                      \
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
                                sum += to_float(input[input_idx]) * to_float(weight[weight_idx]);  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            output[idx] = from_float<TYPENAME>(sum);                                               \
        }                                                                                          \
    }

#define CONV_TRANSPOSE3D_OP(TYPENAME, FN_NAME)                                                     \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, const TYPENAME *weight,              \
                                       TYPENAME *output, const size_t *metadata) {                 \
        const size_t num_els = metadata[0];                                                        \
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
        for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_els;                  \
             idx += blockDim.x * gridDim.x) {                                                      \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t od = (idx / (out_width * out_height)) % out_depth;                        \
            const size_t oc = (idx / (out_width * out_height * out_depth)) % out_channels;         \
            const size_t b = idx / (out_width * out_height * out_depth * out_channels);            \
            float sum = 0.0f;                                                                      \
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
                                    sum +=                                                         \
                                        to_float(input[input_idx]) * to_float(weight[weight_idx]); \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            output[idx] = from_float<TYPENAME>(sum);                                               \
        }                                                                                          \
    }

#define CONV1D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME, ATOMIC_ADD)                                       \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, const TYPENAME *grad_output,         \
                                       TYPENAME *grad_weight, const size_t *metadata) {            \
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
        const size_t total_output_els = batch * out_channels * out_width;                          \
        for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_output_els;         \
             idx += blockDim.x * gridDim.x) {                                                      \
            const size_t ow = idx % out_width;                                                     \
            const size_t oc = (idx / out_width) % out_channels;                                    \
            const size_t b = idx / (out_width * out_channels);                                     \
            /* Use stride-based indexing for grad_output */                                        \
            const size_t grad_output_idx = grad_output_offset + b * grad_output_stride_batch +     \
                                           oc * grad_output_stride_channel +                       \
                                           ow * grad_output_stride_w;                              \
            const float grad_out_val = to_float(grad_output[grad_output_idx]);                     \
            for (size_t ic = 0; ic < in_channels; ic++) {                                          \
                for (size_t kw = 0; kw < kernel_width; kw++) {                                     \
                    const int iw = (int)(ow * stride) - (int)padding + (int)(kw * dilation);       \
                    if (iw >= 0 && iw < (int)in_width) {                                           \
                        const size_t input_idx =                                                   \
                            input_offset + b * in_channels * in_width + ic * in_width + iw;        \
                        const size_t weight_idx =                                                  \
                            oc * in_channels * kernel_width + ic * kernel_width + kw;              \
                        const float contribution = to_float(input[input_idx]) * grad_out_val;      \
                        ATOMIC_ADD(&grad_weight[weight_idx], from_float<TYPENAME>(contribution));  \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

#define CONV2D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME, ATOMIC_ADD)                                       \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, const TYPENAME *grad_output,         \
                                       TYPENAME *grad_weight, const size_t *metadata) {            \
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
        const size_t total_output_els = batch * out_channels * out_height * out_width;             \
        for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_output_els;         \
             idx += blockDim.x * gridDim.x) {                                                      \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t oc = (idx / (out_width * out_height)) % out_channels;                     \
            const size_t b = idx / (out_width * out_height * out_channels);                        \
            /* Use stride-based indexing for grad_output */                                        \
            const size_t grad_output_idx = grad_output_offset + b * grad_output_stride_batch +     \
                                           oc * grad_output_stride_channel +                       \
                                           oh * grad_output_stride_h + ow * grad_output_stride_w;  \
            const float grad_out_val = to_float(grad_output[grad_output_idx]);                     \
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
                            const float contribution = to_float(input[input_idx]) * grad_out_val;  \
                            ATOMIC_ADD(&grad_weight[weight_idx],                                   \
                                       from_float<TYPENAME>(contribution));                        \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

#define CONV3D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME, ATOMIC_ADD)                                       \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, const TYPENAME *grad_output,         \
                                       TYPENAME *grad_weight, const size_t *metadata) {            \
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
        const size_t total_output_els = batch * out_channels * out_depth * out_height * out_width; \
        for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_output_els;         \
             idx += blockDim.x * gridDim.x) {                                                      \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t od = (idx / (out_width * out_height)) % out_depth;                        \
            const size_t oc = (idx / (out_width * out_height * out_depth)) % out_channels;         \
            const size_t b = idx / (out_width * out_height * out_depth * out_channels);            \
            /* Use stride-based indexing for grad_output */                                        \
            const size_t grad_output_idx = grad_output_offset + b * grad_output_stride_batch +     \
                                           oc * grad_output_stride_channel +                       \
                                           od * grad_output_stride_d + oh * grad_output_stride_h + \
                                           ow * grad_output_stride_w;                              \
            const float grad_out_val = to_float(grad_output[grad_output_idx]);                     \
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
                                const float contribution =                                         \
                                    to_float(input[input_idx]) * grad_out_val;                     \
                                ATOMIC_ADD(&grad_weight[weight_idx],                               \
                                           from_float<TYPENAME>(contribution));                    \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

#define CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME, ATOMIC_ADD)                             \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, const TYPENAME *grad_output,         \
                                       TYPENAME *grad_weight, const size_t *metadata) {            \
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
        const size_t total_output_els = batch * out_channels * out_width;                          \
        for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_output_els;         \
             idx += blockDim.x * gridDim.x) {                                                      \
            const size_t ow = idx % out_width;                                                     \
            const size_t oc = (idx / out_width) % out_channels;                                    \
            const size_t b = idx / (out_width * out_channels);                                     \
            /* Use stride-based indexing for grad_output */                                        \
            const size_t grad_output_idx = grad_output_offset + b * grad_output_stride_batch +     \
                                           oc * grad_output_stride_channel +                       \
                                           ow * grad_output_stride_w;                              \
            const float grad_out_val = to_float(grad_output[grad_output_idx]);                     \
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
                            const float contribution = to_float(input[input_idx]) * grad_out_val;  \
                            ATOMIC_ADD(&grad_weight[weight_idx],                                   \
                                       from_float<TYPENAME>(contribution));                        \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

#define CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME, ATOMIC_ADD)                             \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, const TYPENAME *grad_output,         \
                                       TYPENAME *grad_weight, const size_t *metadata) {            \
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
        const size_t total_output_els = batch * out_channels * out_height * out_width;             \
        for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_output_els;         \
             idx += blockDim.x * gridDim.x) {                                                      \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t oc = (idx / (out_width * out_height)) % out_channels;                     \
            const size_t b = idx / (out_width * out_height * out_channels);                        \
            /* Use stride-based indexing for grad_output */                                        \
            const size_t grad_output_idx = grad_output_offset + b * grad_output_stride_batch +     \
                                           oc * grad_output_stride_channel +                       \
                                           oh * grad_output_stride_h + ow * grad_output_stride_w;  \
            const float grad_out_val = to_float(grad_output[grad_output_idx]);                     \
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
                                const float contribution =                                         \
                                    to_float(input[input_idx]) * grad_out_val;                     \
                                ATOMIC_ADD(&grad_weight[weight_idx],                               \
                                           from_float<TYPENAME>(contribution));                    \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

#define CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(TYPENAME, FN_NAME, ATOMIC_ADD)                             \
    extern "C" __global__ void FN_NAME(const TYPENAME *input, const TYPENAME *grad_output,         \
                                       TYPENAME *grad_weight, const size_t *metadata) {            \
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
        const size_t total_output_els = batch * out_channels * out_depth * out_height * out_width; \
        for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_output_els;         \
             idx += blockDim.x * gridDim.x) {                                                      \
            const size_t ow = idx % out_width;                                                     \
            const size_t oh = (idx / out_width) % out_height;                                      \
            const size_t od = (idx / (out_width * out_height)) % out_depth;                        \
            const size_t oc = (idx / (out_width * out_height * out_depth)) % out_channels;         \
            const size_t b = idx / (out_width * out_height * out_depth * out_channels);            \
            /* Use stride-based indexing for grad_output */                                        \
            const size_t grad_output_idx = grad_output_offset + b * grad_output_stride_batch +     \
                                           oc * grad_output_stride_channel +                       \
                                           od * grad_output_stride_d + oh * grad_output_stride_h + \
                                           ow * grad_output_stride_w;                              \
            const float grad_out_val = to_float(grad_output[grad_output_idx]);                     \
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
                                    const float contribution =                                     \
                                        to_float(input[input_idx]) * grad_out_val;                 \
                                    ATOMIC_ADD(&grad_weight[weight_idx],                           \
                                               from_float<TYPENAME>(contribution));                \
                                }                                                                  \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// KERNEL INSTANTIATIONS
// ============================================================================

// Conv1D
CONV1D_OP(__nv_fp8_e4m3, conv1d_f8e4m3)
CONV1D_OP(__nv_fp8_e5m2, conv1d_f8e5m2)
CONV1D_OP(__nv_bfloat16, conv1d_bf16)
CONV1D_OP(__half, conv1d_f16)
CONV1D_OP(float, conv1d_f32)
CONV1D_OP(double, conv1d_f64)

// Conv2D
CONV2D_OP(__nv_fp8_e4m3, conv2d_f8e4m3)
CONV2D_OP(__nv_fp8_e5m2, conv2d_f8e5m2)
CONV2D_OP(__nv_bfloat16, conv2d_bf16)
CONV2D_OP(__half, conv2d_f16)
CONV2D_OP(float, conv2d_f32)
CONV2D_OP(double, conv2d_f64)

// Conv3D
CONV3D_OP(__nv_fp8_e4m3, conv3d_f8e4m3)
CONV3D_OP(__nv_fp8_e5m2, conv3d_f8e5m2)
CONV3D_OP(__nv_bfloat16, conv3d_bf16)
CONV3D_OP(__half, conv3d_f16)
CONV3D_OP(float, conv3d_f32)
CONV3D_OP(double, conv3d_f64)

// ConvTranspose1D
CONV_TRANSPOSE1D_OP(__nv_fp8_e4m3, conv_transpose1d_f8e4m3)
CONV_TRANSPOSE1D_OP(__nv_fp8_e5m2, conv_transpose1d_f8e5m2)
CONV_TRANSPOSE1D_OP(__nv_bfloat16, conv_transpose1d_bf16)
CONV_TRANSPOSE1D_OP(__half, conv_transpose1d_f16)
CONV_TRANSPOSE1D_OP(float, conv_transpose1d_f32)
CONV_TRANSPOSE1D_OP(double, conv_transpose1d_f64)

// ConvTranspose2D
CONV_TRANSPOSE2D_OP(__nv_fp8_e4m3, conv_transpose2d_f8e4m3)
CONV_TRANSPOSE2D_OP(__nv_fp8_e5m2, conv_transpose2d_f8e5m2)
CONV_TRANSPOSE2D_OP(__nv_bfloat16, conv_transpose2d_bf16)
CONV_TRANSPOSE2D_OP(__half, conv_transpose2d_f16)
CONV_TRANSPOSE2D_OP(float, conv_transpose2d_f32)
CONV_TRANSPOSE2D_OP(double, conv_transpose2d_f64)

// ConvTranspose3D
CONV_TRANSPOSE3D_OP(__nv_fp8_e4m3, conv_transpose3d_f8e4m3)
CONV_TRANSPOSE3D_OP(__nv_fp8_e5m2, conv_transpose3d_f8e5m2)
CONV_TRANSPOSE3D_OP(__nv_bfloat16, conv_transpose3d_bf16)
CONV_TRANSPOSE3D_OP(__half, conv_transpose3d_f16)
CONV_TRANSPOSE3D_OP(float, conv_transpose3d_f32)
CONV_TRANSPOSE3D_OP(double, conv_transpose3d_f64)

// Conv1D Grad Weight
CONV1D_GRAD_WEIGHT_OP(__nv_fp8_e4m3, conv1d_grad_weight_f8e4m3, atomic_add_f8e4m3)
CONV1D_GRAD_WEIGHT_OP(__nv_fp8_e5m2, conv1d_grad_weight_f8e5m2, atomic_add_f8e5m2)
CONV1D_GRAD_WEIGHT_OP(__nv_bfloat16, conv1d_grad_weight_bf16, atomic_add_bf16)
CONV1D_GRAD_WEIGHT_OP(__half, conv1d_grad_weight_f16, atomic_add_f16)
CONV1D_GRAD_WEIGHT_OP(float, conv1d_grad_weight_f32, atomic_add_f32)
CONV1D_GRAD_WEIGHT_OP(double, conv1d_grad_weight_f64, atomic_add_f64)

// Conv2D Grad Weight
CONV2D_GRAD_WEIGHT_OP(__nv_fp8_e4m3, conv2d_grad_weight_f8e4m3, atomic_add_f8e4m3)
CONV2D_GRAD_WEIGHT_OP(__nv_fp8_e5m2, conv2d_grad_weight_f8e5m2, atomic_add_f8e5m2)
CONV2D_GRAD_WEIGHT_OP(__nv_bfloat16, conv2d_grad_weight_bf16, atomic_add_bf16)
CONV2D_GRAD_WEIGHT_OP(__half, conv2d_grad_weight_f16, atomic_add_f16)
CONV2D_GRAD_WEIGHT_OP(float, conv2d_grad_weight_f32, atomic_add_f32)
CONV2D_GRAD_WEIGHT_OP(double, conv2d_grad_weight_f64, atomic_add_f64)

// Conv3D Grad Weight
CONV3D_GRAD_WEIGHT_OP(__nv_fp8_e4m3, conv3d_grad_weight_f8e4m3, atomic_add_f8e4m3)
CONV3D_GRAD_WEIGHT_OP(__nv_fp8_e5m2, conv3d_grad_weight_f8e5m2, atomic_add_f8e5m2)
CONV3D_GRAD_WEIGHT_OP(__nv_bfloat16, conv3d_grad_weight_bf16, atomic_add_bf16)
CONV3D_GRAD_WEIGHT_OP(__half, conv3d_grad_weight_f16, atomic_add_f16)
CONV3D_GRAD_WEIGHT_OP(float, conv3d_grad_weight_f32, atomic_add_f32)
CONV3D_GRAD_WEIGHT_OP(double, conv3d_grad_weight_f64, atomic_add_f64)

// ConvTranspose1D Grad Weight
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(__nv_fp8_e4m3, conv_transpose1d_grad_weight_f8e4m3,
                                atomic_add_f8e4m3)
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(__nv_fp8_e5m2, conv_transpose1d_grad_weight_f8e5m2,
                                atomic_add_f8e5m2)
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(__nv_bfloat16, conv_transpose1d_grad_weight_bf16, atomic_add_bf16)
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(__half, conv_transpose1d_grad_weight_f16, atomic_add_f16)
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(float, conv_transpose1d_grad_weight_f32, atomic_add_f32)
CONV_TRANSPOSE1D_GRAD_WEIGHT_OP(double, conv_transpose1d_grad_weight_f64, atomic_add_f64)

// ConvTranspose2D Grad Weight
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(__nv_fp8_e4m3, conv_transpose2d_grad_weight_f8e4m3,
                                atomic_add_f8e4m3)
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(__nv_fp8_e5m2, conv_transpose2d_grad_weight_f8e5m2,
                                atomic_add_f8e5m2)
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(__nv_bfloat16, conv_transpose2d_grad_weight_bf16, atomic_add_bf16)
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(__half, conv_transpose2d_grad_weight_f16, atomic_add_f16)
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(float, conv_transpose2d_grad_weight_f32, atomic_add_f32)
CONV_TRANSPOSE2D_GRAD_WEIGHT_OP(double, conv_transpose2d_grad_weight_f64, atomic_add_f64)

// ConvTranspose3D Grad Weight
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(__nv_fp8_e4m3, conv_transpose3d_grad_weight_f8e4m3,
                                atomic_add_f8e4m3)
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(__nv_fp8_e5m2, conv_transpose3d_grad_weight_f8e5m2,
                                atomic_add_f8e5m2)
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(__nv_bfloat16, conv_transpose3d_grad_weight_bf16, atomic_add_bf16)
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(__half, conv_transpose3d_grad_weight_f16, atomic_add_f16)
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(float, conv_transpose3d_grad_weight_f32, atomic_add_f32)
CONV_TRANSPOSE3D_GRAD_WEIGHT_OP(double, conv_transpose3d_grad_weight_f64, atomic_add_f64)
