#include "math.cuh"
#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Resize modes
#define RESIZE_MODE_NEAREST 0
#define RESIZE_MODE_LINEAR 1
#define RESIZE_MODE_CUBIC 2

// Coordinate transformation modes
#define RESIZE_COORD_HALF_PIXEL 0
#define RESIZE_COORD_ASYMMETRIC 1
#define RESIZE_COORD_ALIGN_CORNERS 2
#define RESIZE_COORD_PYTORCH_HALF_PIXEL 3

// Nearest rounding modes
#define RESIZE_NEAREST_FLOOR 0
#define RESIZE_NEAREST_CEIL 1
#define RESIZE_NEAREST_ROUND_PREFER_FLOOR 2
#define RESIZE_NEAREST_ROUND_PREFER_CEIL 3

__device__ __forceinline__ float transform_coord(float out_coord, size_t in_size, size_t out_size,
                                                 int coord_transform) {
    if (in_size == out_size)
        return out_coord;
    float scale = (float)in_size / (float)out_size;

    switch (coord_transform) {
    case RESIZE_COORD_HALF_PIXEL:
        return (out_coord + 0.5f) * scale - 0.5f;
    case RESIZE_COORD_ASYMMETRIC:
        return out_coord * scale;
    case RESIZE_COORD_ALIGN_CORNERS:
        if (out_size == 1)
            return 0.0f;
        return out_coord * (float)(in_size - 1) / (float)(out_size - 1);
    case RESIZE_COORD_PYTORCH_HALF_PIXEL:
        if (out_size == 1)
            return 0.0f;
        return (out_coord + 0.5f) * scale - 0.5f;
    default:
        return out_coord * scale;
    }
}

__device__ __forceinline__ size_t round_nearest(float coord, size_t max_idx, int nearest_mode) {
    float rounded;
    switch (nearest_mode) {
    case RESIZE_NEAREST_FLOOR:
        rounded = floorf(coord);
        break;
    case RESIZE_NEAREST_CEIL:
        rounded = ceilf(coord);
        break;
    case RESIZE_NEAREST_ROUND_PREFER_FLOOR:
        rounded = (coord == floorf(coord) + 0.5f) ? floorf(coord) : roundf(coord);
        break;
    case RESIZE_NEAREST_ROUND_PREFER_CEIL:
        rounded = roundf(coord);
        break;
    default:
        rounded = floorf(coord);
    }
    if (rounded < 0)
        return 0;
    if (rounded > (float)max_idx)
        return max_idx;
    return (size_t)rounded;
}

__device__ __forceinline__ size_t clamp_idx(int idx, size_t max_idx) {
    if (idx < 0)
        return 0;
    if ((size_t)idx > max_idx)
        return max_idx;
    return (size_t)idx;
}

__device__ __forceinline__ float cubic_interp(float p0, float p1, float p2, float p3, float t) {
    float a = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    float b = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
    float c = -0.5f * p0 + 0.5f * p2;
    float d = p1;
    return a * t * t * t + b * t * t + c * t + d;
}

// Metadata layout:
// [0]: output_size
// [1]: num_dims
// [2..2+num_dims]: input_shape
// [2+num_dims..2+2*num_dims]: input_strides
// [2+2*num_dims]: offset
// [3+2*num_dims..3+3*num_dims]: output_shape
// [3+3*num_dims]: mode
// [4+3*num_dims]: coord_transform
// [5+3*num_dims]: nearest_mode

#define RESIZE_KERNEL(TYPENAME, FN_SUFFIX)                                                         \
    extern "C" __global__ void hodu_cuda_resize_##FN_SUFFIX(                                       \
        const TYPENAME *input, TYPENAME *output, const size_t *metadata) {                         \
        const size_t output_size = metadata[0];                                                    \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *in_shape = metadata + 2;                                                     \
        const size_t *in_strides = metadata + 2 + num_dims;                                        \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        const size_t *out_shape = metadata + 3 + 2 * num_dims;                                     \
        const int mode = (int)metadata[3 + 3 * num_dims];                                          \
        const int coord_transform = (int)metadata[4 + 3 * num_dims];                               \
        const int nearest_mode = (int)metadata[5 + 3 * num_dims];                                  \
                                                                                                   \
        const size_t spatial_start = 2;                                                            \
        const size_t num_spatial = num_dims - spatial_start;                                       \
                                                                                                   \
        for (uint32_t out_idx = blockIdx.x * blockDim.x + threadIdx.x; out_idx < output_size;      \
             out_idx += blockDim.x * gridDim.x) {                                                  \
            size_t out_coords[8];                                                                  \
            size_t tmp = out_idx;                                                                  \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                out_coords[d] = tmp % out_shape[d];                                                \
                tmp /= out_shape[d];                                                               \
            }                                                                                      \
                                                                                                   \
            size_t base_in_idx =                                                                   \
                offset + out_coords[0] * in_strides[0] + out_coords[1] * in_strides[1];            \
                                                                                                   \
            if (mode == RESIZE_MODE_NEAREST) {                                                     \
                size_t in_idx = base_in_idx;                                                       \
                for (size_t s = 0; s < num_spatial; s++) {                                         \
                    size_t d = spatial_start + s;                                                  \
                    float coord = transform_coord((float)out_coords[d], in_shape[d], out_shape[d], \
                                                  coord_transform);                                \
                    size_t src_idx = round_nearest(coord, in_shape[d] - 1, nearest_mode);          \
                    in_idx += src_idx * in_strides[d];                                             \
                }                                                                                  \
                output[out_idx] = input[in_idx];                                                   \
            } else if (mode == RESIZE_MODE_LINEAR && num_spatial == 2) {                           \
                float y_coord = transform_coord((float)out_coords[2], in_shape[2], out_shape[2],   \
                                                coord_transform);                                  \
                float x_coord = transform_coord((float)out_coords[3], in_shape[3], out_shape[3],   \
                                                coord_transform);                                  \
                                                                                                   \
                int y0 = (int)floorf(y_coord);                                                     \
                int x0 = (int)floorf(x_coord);                                                     \
                int y1 = y0 + 1;                                                                   \
                int x1 = x0 + 1;                                                                   \
                                                                                                   \
                float fy = y_coord - (float)y0;                                                    \
                float fx = x_coord - (float)x0;                                                    \
                                                                                                   \
                size_t y0c = clamp_idx(y0, in_shape[2] - 1);                                       \
                size_t y1c = clamp_idx(y1, in_shape[2] - 1);                                       \
                size_t x0c = clamp_idx(x0, in_shape[3] - 1);                                       \
                size_t x1c = clamp_idx(x1, in_shape[3] - 1);                                       \
                                                                                                   \
                float v00 =                                                                        \
                    to_float(input[base_in_idx + y0c * in_strides[2] + x0c * in_strides[3]]);      \
                float v01 =                                                                        \
                    to_float(input[base_in_idx + y0c * in_strides[2] + x1c * in_strides[3]]);      \
                float v10 =                                                                        \
                    to_float(input[base_in_idx + y1c * in_strides[2] + x0c * in_strides[3]]);      \
                float v11 =                                                                        \
                    to_float(input[base_in_idx + y1c * in_strides[2] + x1c * in_strides[3]]);      \
                                                                                                   \
                float result = (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) +                      \
                               fy * ((1.0f - fx) * v10 + fx * v11);                                \
                output[out_idx] = from_float<TYPENAME>(result);                                    \
            } else if (mode == RESIZE_MODE_LINEAR && num_spatial == 3) {                           \
                float d_coord = transform_coord((float)out_coords[2], in_shape[2], out_shape[2],   \
                                                coord_transform);                                  \
                float h_coord = transform_coord((float)out_coords[3], in_shape[3], out_shape[3],   \
                                                coord_transform);                                  \
                float w_coord = transform_coord((float)out_coords[4], in_shape[4], out_shape[4],   \
                                                coord_transform);                                  \
                                                                                                   \
                int d0 = (int)floorf(d_coord);                                                     \
                int h0 = (int)floorf(h_coord);                                                     \
                int w0 = (int)floorf(w_coord);                                                     \
                int d1 = d0 + 1, h1 = h0 + 1, w1 = w0 + 1;                                         \
                                                                                                   \
                float fd = d_coord - (float)d0;                                                    \
                float fh = h_coord - (float)h0;                                                    \
                float fw = w_coord - (float)w0;                                                    \
                                                                                                   \
                size_t d0c = clamp_idx(d0, in_shape[2] - 1);                                       \
                size_t d1c = clamp_idx(d1, in_shape[2] - 1);                                       \
                size_t h0c = clamp_idx(h0, in_shape[3] - 1);                                       \
                size_t h1c = clamp_idx(h1, in_shape[3] - 1);                                       \
                size_t w0c = clamp_idx(w0, in_shape[4] - 1);                                       \
                size_t w1c = clamp_idx(w1, in_shape[4] - 1);                                       \
                                                                                                   \
                float v000 = to_float(input[base_in_idx + d0c * in_strides[2] +                    \
                                            h0c * in_strides[3] + w0c * in_strides[4]]);           \
                float v001 = to_float(input[base_in_idx + d0c * in_strides[2] +                    \
                                            h0c * in_strides[3] + w1c * in_strides[4]]);           \
                float v010 = to_float(input[base_in_idx + d0c * in_strides[2] +                    \
                                            h1c * in_strides[3] + w0c * in_strides[4]]);           \
                float v011 = to_float(input[base_in_idx + d0c * in_strides[2] +                    \
                                            h1c * in_strides[3] + w1c * in_strides[4]]);           \
                float v100 = to_float(input[base_in_idx + d1c * in_strides[2] +                    \
                                            h0c * in_strides[3] + w0c * in_strides[4]]);           \
                float v101 = to_float(input[base_in_idx + d1c * in_strides[2] +                    \
                                            h0c * in_strides[3] + w1c * in_strides[4]]);           \
                float v110 = to_float(input[base_in_idx + d1c * in_strides[2] +                    \
                                            h1c * in_strides[3] + w0c * in_strides[4]]);           \
                float v111 = to_float(input[base_in_idx + d1c * in_strides[2] +                    \
                                            h1c * in_strides[3] + w1c * in_strides[4]]);           \
                                                                                                   \
                float c00 = (1.0f - fw) * v000 + fw * v001;                                        \
                float c01 = (1.0f - fw) * v010 + fw * v011;                                        \
                float c10 = (1.0f - fw) * v100 + fw * v101;                                        \
                float c11 = (1.0f - fw) * v110 + fw * v111;                                        \
                float c0 = (1.0f - fh) * c00 + fh * c01;                                           \
                float c1 = (1.0f - fh) * c10 + fh * c11;                                           \
                output[out_idx] = from_float<TYPENAME>((1.0f - fd) * c0 + fd * c1);                \
            } else if (mode == RESIZE_MODE_CUBIC && num_spatial == 2) {                            \
                float y_coord = transform_coord((float)out_coords[2], in_shape[2], out_shape[2],   \
                                                coord_transform);                                  \
                float x_coord = transform_coord((float)out_coords[3], in_shape[3], out_shape[3],   \
                                                coord_transform);                                  \
                                                                                                   \
                int y0 = (int)floorf(y_coord);                                                     \
                int x0 = (int)floorf(x_coord);                                                     \
                float fy = y_coord - (float)y0;                                                    \
                float fx = x_coord - (float)x0;                                                    \
                                                                                                   \
                float cols[4];                                                                     \
                for (int j = -1; j <= 2; j++) {                                                    \
                    size_t yc = clamp_idx(y0 + j, in_shape[2] - 1);                                \
                    float p0 =                                                                     \
                        to_float(input[base_in_idx + yc * in_strides[2] +                          \
                                       clamp_idx(x0 - 1, in_shape[3] - 1) * in_strides[3]]);       \
                    float p1 = to_float(input[base_in_idx + yc * in_strides[2] +                   \
                                              clamp_idx(x0, in_shape[3] - 1) * in_strides[3]]);    \
                    float p2 =                                                                     \
                        to_float(input[base_in_idx + yc * in_strides[2] +                          \
                                       clamp_idx(x0 + 1, in_shape[3] - 1) * in_strides[3]]);       \
                    float p3 =                                                                     \
                        to_float(input[base_in_idx + yc * in_strides[2] +                          \
                                       clamp_idx(x0 + 2, in_shape[3] - 1) * in_strides[3]]);       \
                    cols[j + 1] = cubic_interp(p0, p1, p2, p3, fx);                                \
                }                                                                                  \
                output[out_idx] =                                                                  \
                    from_float<TYPENAME>(cubic_interp(cols[0], cols[1], cols[2], cols[3], fy));    \
            } else {                                                                               \
                size_t in_idx = base_in_idx;                                                       \
                for (size_t s = 0; s < num_spatial; s++) {                                         \
                    size_t d = spatial_start + s;                                                  \
                    float coord = transform_coord((float)out_coords[d], in_shape[d], out_shape[d], \
                                                  coord_transform);                                \
                    size_t src_idx = round_nearest(coord, in_shape[d] - 1, nearest_mode);          \
                    in_idx += src_idx * in_strides[d];                                             \
                }                                                                                  \
                output[out_idx] = input[in_idx];                                                   \
            }                                                                                      \
        }                                                                                          \
    }

RESIZE_KERNEL(__nv_fp8_e4m3, f8e4m3)
RESIZE_KERNEL(__nv_fp8_e5m2, f8e5m2)
RESIZE_KERNEL(__nv_bfloat16, bf16)
RESIZE_KERNEL(__half, f16)
RESIZE_KERNEL(float, f32)
RESIZE_KERNEL(double, f64)
