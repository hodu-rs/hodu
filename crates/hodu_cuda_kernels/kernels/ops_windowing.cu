#include "math.cuh"
#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define REDUCE_WINDOW_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME, INIT_VAL, ACCUMULATE)                 \
    extern "C" __global__ void FN_NAME(const IN_TYPENAME *input, OUT_TYPENAME *output,             \
                                       const size_t *metadata) {                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
        const size_t *window_shape = metadata + 2 + 2 * num_dims + 1;                              \
        const size_t *strides = metadata + 2 + 3 * num_dims + 1;                                   \
        const size_t *padding = metadata + 2 + 4 * num_dims + 1;                                   \
        const size_t *output_shape = metadata + 2 + 6 * num_dims + 1;                              \
        for (uint32_t out_idx = blockIdx.x * blockDim.x + threadIdx.x; out_idx < num_els;          \
             out_idx += blockDim.x * gridDim.x) {                                                  \
            size_t out_coords[16];                                                                 \
            size_t tmp = out_idx;                                                                  \
            for (int i = (int)num_dims - 1; i >= 0; i--) {                                         \
                out_coords[i] = tmp % output_shape[i];                                             \
                tmp /= output_shape[i];                                                            \
            }                                                                                      \
            float acc = INIT_VAL;                                                                  \
            size_t window_size = 1;                                                                \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                window_size *= window_shape[i];                                                    \
            }                                                                                      \
            for (size_t win_idx = 0; win_idx < window_size; win_idx++) {                           \
                size_t window_coords[16];                                                          \
                size_t tmp_win = win_idx;                                                          \
                for (int i = (int)num_dims - 1; i >= 0; i--) {                                     \
                    window_coords[i] = tmp_win % window_shape[i];                                  \
                    tmp_win /= window_shape[i];                                                    \
                }                                                                                  \
                bool in_bounds = true;                                                             \
                size_t input_coords[16];                                                           \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    size_t padded_pos = out_coords[i] * strides[i] + window_coords[i];             \
                    size_t pad_before = padding[i * 2];                                            \
                    if (padded_pos < pad_before) {                                                 \
                        in_bounds = false;                                                         \
                        break;                                                                     \
                    }                                                                              \
                    size_t input_pos = padded_pos - pad_before;                                    \
                    if (input_pos >= input_shape[i]) {                                             \
                        in_bounds = false;                                                         \
                        break;                                                                     \
                    }                                                                              \
                    input_coords[i] = input_pos;                                                   \
                }                                                                                  \
                float val;                                                                         \
                if (in_bounds) {                                                                   \
                    size_t idx = input_offset;                                                     \
                    for (size_t i = 0; i < num_dims; i++) {                                        \
                        idx += input_coords[i] * input_strides[i];                                 \
                    }                                                                              \
                    val = to_float(input[idx]);                                                    \
                } else {                                                                           \
                    val = INIT_VAL;                                                                \
                }                                                                                  \
                ACCUMULATE;                                                                        \
            }                                                                                      \
            output[out_idx] = from_float<OUT_TYPENAME>(acc);                                       \
        }                                                                                          \
    }

#define REDUCE_WINDOW_MEAN_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME)                                  \
    extern "C" __global__ void FN_NAME(const IN_TYPENAME *input, OUT_TYPENAME *output,             \
                                       const size_t *metadata) {                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
        const size_t *window_shape = metadata + 2 + 2 * num_dims + 1;                              \
        const size_t *strides = metadata + 2 + 3 * num_dims + 1;                                   \
        const size_t *padding = metadata + 2 + 4 * num_dims + 1;                                   \
        const size_t *output_shape = metadata + 2 + 6 * num_dims + 1;                              \
        for (uint32_t out_idx = blockIdx.x * blockDim.x + threadIdx.x; out_idx < num_els;          \
             out_idx += blockDim.x * gridDim.x) {                                                  \
            size_t out_coords[16];                                                                 \
            size_t tmp = out_idx;                                                                  \
            for (int i = (int)num_dims - 1; i >= 0; i--) {                                         \
                out_coords[i] = tmp % output_shape[i];                                             \
                tmp /= output_shape[i];                                                            \
            }                                                                                      \
            float sum = 0.0f;                                                                      \
            size_t window_size = 1;                                                                \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                window_size *= window_shape[i];                                                    \
            }                                                                                      \
            for (size_t win_idx = 0; win_idx < window_size; win_idx++) {                           \
                size_t window_coords[16];                                                          \
                size_t tmp_win = win_idx;                                                          \
                for (int i = (int)num_dims - 1; i >= 0; i--) {                                     \
                    window_coords[i] = tmp_win % window_shape[i];                                  \
                    tmp_win /= window_shape[i];                                                    \
                }                                                                                  \
                bool in_bounds = true;                                                             \
                size_t input_coords[16];                                                           \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    size_t padded_pos = out_coords[i] * strides[i] + window_coords[i];             \
                    size_t pad_before = padding[i * 2];                                            \
                    if (padded_pos < pad_before) {                                                 \
                        in_bounds = false;                                                         \
                        break;                                                                     \
                    }                                                                              \
                    size_t input_pos = padded_pos - pad_before;                                    \
                    if (input_pos >= input_shape[i]) {                                             \
                        in_bounds = false;                                                         \
                        break;                                                                     \
                    }                                                                              \
                    input_coords[i] = input_pos;                                                   \
                }                                                                                  \
                if (in_bounds) {                                                                   \
                    size_t idx = input_offset;                                                     \
                    for (size_t i = 0; i < num_dims; i++) {                                        \
                        idx += input_coords[i] * input_strides[i];                                 \
                    }                                                                              \
                    sum += to_float(input[idx]);                                                   \
                }                                                                                  \
            }                                                                                      \
            output[out_idx] = from_float<OUT_TYPENAME>(sum / (float)window_size);                  \
        }                                                                                          \
    }

REDUCE_WINDOW_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, reduce_window_max_f8e4m3, -__builtin_huge_valf(),
                 acc = maximum(acc, val))
REDUCE_WINDOW_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, reduce_window_max_f8e5m2, -__builtin_huge_valf(),
                 acc = maximum(acc, val))
REDUCE_WINDOW_OP(__nv_bfloat16, __nv_bfloat16, reduce_window_max_bf16, -__builtin_huge_valf(),
                 acc = maximum(acc, val))
REDUCE_WINDOW_OP(__half, __half, reduce_window_max_f16, -__builtin_huge_valf(),
                 acc = maximum(acc, val))
REDUCE_WINDOW_OP(float, float, reduce_window_max_f32, -__builtin_huge_valf(),
                 acc = maximum(acc, val))
REDUCE_WINDOW_OP(double, double, reduce_window_max_f64, -__builtin_huge_val(),
                 acc = maximum(acc, val))
REDUCE_WINDOW_OP(uint8_t, uint8_t, reduce_window_max_u8, 0.0f, acc = maximum(acc, val))
REDUCE_WINDOW_OP(uint16_t, uint16_t, reduce_window_max_u16, 0.0f, acc = maximum(acc, val))
REDUCE_WINDOW_OP(uint32_t, uint32_t, reduce_window_max_u32, 0.0f, acc = maximum(acc, val))
REDUCE_WINDOW_OP(uint64_t, uint64_t, reduce_window_max_u64, 0.0f, acc = maximum(acc, val))
REDUCE_WINDOW_OP(int8_t, int8_t, reduce_window_max_i8, (float)INT8_MIN, acc = maximum(acc, val))
REDUCE_WINDOW_OP(int16_t, int16_t, reduce_window_max_i16, (float)INT16_MIN, acc = maximum(acc, val))
REDUCE_WINDOW_OP(int32_t, int32_t, reduce_window_max_i32, (float)INT32_MIN, acc = maximum(acc, val))
REDUCE_WINDOW_OP(int64_t, int64_t, reduce_window_max_i64, (float)INT64_MIN, acc = maximum(acc, val))

REDUCE_WINDOW_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, reduce_window_min_f8e4m3, __builtin_huge_valf(),
                 acc = minimum(acc, val))
REDUCE_WINDOW_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, reduce_window_min_f8e5m2, __builtin_huge_valf(),
                 acc = minimum(acc, val))
REDUCE_WINDOW_OP(__nv_bfloat16, __nv_bfloat16, reduce_window_min_bf16, __builtin_huge_valf(),
                 acc = minimum(acc, val))
REDUCE_WINDOW_OP(__half, __half, reduce_window_min_f16, __builtin_huge_valf(),
                 acc = minimum(acc, val))
REDUCE_WINDOW_OP(float, float, reduce_window_min_f32, __builtin_huge_valf(),
                 acc = minimum(acc, val))
REDUCE_WINDOW_OP(double, double, reduce_window_min_f64, __builtin_huge_val(),
                 acc = minimum(acc, val))
REDUCE_WINDOW_OP(uint8_t, uint8_t, reduce_window_min_u8, (float)UINT8_MAX, acc = minimum(acc, val))
REDUCE_WINDOW_OP(uint16_t, uint16_t, reduce_window_min_u16, (float)UINT16_MAX,
                 acc = minimum(acc, val))
REDUCE_WINDOW_OP(uint32_t, uint32_t, reduce_window_min_u32, (float)UINT32_MAX,
                 acc = minimum(acc, val))
REDUCE_WINDOW_OP(uint64_t, uint64_t, reduce_window_min_u64, (float)UINT64_MAX,
                 acc = minimum(acc, val))
REDUCE_WINDOW_OP(int8_t, int8_t, reduce_window_min_i8, (float)INT8_MAX, acc = minimum(acc, val))
REDUCE_WINDOW_OP(int16_t, int16_t, reduce_window_min_i16, (float)INT16_MAX, acc = minimum(acc, val))
REDUCE_WINDOW_OP(int32_t, int32_t, reduce_window_min_i32, (float)INT32_MAX, acc = minimum(acc, val))
REDUCE_WINDOW_OP(int64_t, int64_t, reduce_window_min_i64, (float)INT64_MAX, acc = minimum(acc, val))

REDUCE_WINDOW_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, reduce_window_sum_f8e4m3, 0.0f, acc += val)
REDUCE_WINDOW_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, reduce_window_sum_f8e5m2, 0.0f, acc += val)
REDUCE_WINDOW_OP(__nv_bfloat16, __nv_bfloat16, reduce_window_sum_bf16, 0.0f, acc += val)
REDUCE_WINDOW_OP(__half, __half, reduce_window_sum_f16, 0.0f, acc += val)
REDUCE_WINDOW_OP(float, float, reduce_window_sum_f32, 0.0f, acc += val)
REDUCE_WINDOW_OP(double, double, reduce_window_sum_f64, 0.0, acc += val)
REDUCE_WINDOW_OP(uint8_t, uint8_t, reduce_window_sum_u8, 0.0f, acc += val)
REDUCE_WINDOW_OP(uint16_t, uint16_t, reduce_window_sum_u16, 0.0f, acc += val)
REDUCE_WINDOW_OP(uint32_t, uint32_t, reduce_window_sum_u32, 0.0f, acc += val)
REDUCE_WINDOW_OP(uint64_t, uint64_t, reduce_window_sum_u64, 0.0f, acc += val)
REDUCE_WINDOW_OP(int8_t, int8_t, reduce_window_sum_i8, 0.0f, acc += val)
REDUCE_WINDOW_OP(int16_t, int16_t, reduce_window_sum_i16, 0.0f, acc += val)
REDUCE_WINDOW_OP(int32_t, int32_t, reduce_window_sum_i32, 0.0f, acc += val)
REDUCE_WINDOW_OP(int64_t, int64_t, reduce_window_sum_i64, 0.0f, acc += val)

REDUCE_WINDOW_MEAN_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, reduce_window_mean_f8e4m3)
REDUCE_WINDOW_MEAN_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, reduce_window_mean_f8e5m2)
REDUCE_WINDOW_MEAN_OP(__nv_bfloat16, __nv_bfloat16, reduce_window_mean_bf16)
REDUCE_WINDOW_MEAN_OP(__half, __half, reduce_window_mean_f16)
REDUCE_WINDOW_MEAN_OP(float, float, reduce_window_mean_f32)
REDUCE_WINDOW_MEAN_OP(double, double, reduce_window_mean_f64)
