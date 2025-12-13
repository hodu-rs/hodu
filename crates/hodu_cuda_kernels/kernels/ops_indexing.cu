#include "atomic.cuh"
#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define INDEX_SELECT_OP(TYPENAME, FN_NAME)                                                         \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *input, const int32_t *indices,  \
                                                   TYPENAME *out, const size_t *metadata) {        \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
        const size_t dim = metadata[2 + 2 * num_dims + 1];                                         \
        const size_t num_indices = metadata[2 + 2 * num_dims + 2];                                 \
        for (uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_els;                    \
             id += blockDim.x * gridDim.x) {                                                       \
            size_t output_shape[16];                                                               \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                output_shape[i] = (i == dim) ? num_indices : input_shape[i];                       \
            }                                                                                      \
            size_t output_indices[16];                                                             \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
            int32_t selected_idx = indices[output_indices[dim]];                                   \
            if (selected_idx < 0) {                                                                \
                selected_idx += (int32_t)input_shape[dim];                                         \
            }                                                                                      \
            size_t flat_index = input_offset;                                                      \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                size_t idx = (i == dim) ? (size_t)selected_idx : output_indices[i];                \
                flat_index += idx * input_strides[i];                                              \
            }                                                                                      \
            out[id] = input[flat_index];                                                           \
        }                                                                                          \
    }

#define INDEX_PUT_OP(TYPENAME, FN_NAME)                                                            \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *input, const int32_t *indices,  \
                                                   const TYPENAME *values, TYPENAME *out,          \
                                                   const size_t *metadata) {                       \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t *values_strides = metadata + 2 + 2 * num_dims;                                \
        const size_t input_offset = metadata[2 + 3 * num_dims];                                    \
        const size_t values_offset = metadata[2 + 3 * num_dims + 1];                               \
        const size_t dim = metadata[2 + 3 * num_dims + 2];                                         \
        const size_t num_indices = metadata[2 + 3 * num_dims + 3];                                 \
        for (uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_els;                    \
             id += blockDim.x * gridDim.x) {                                                       \
            size_t output_indices[16];                                                             \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                output_indices[d] = temp % input_shape[d];                                         \
                temp /= input_shape[d];                                                            \
            }                                                                                      \
            bool found = false;                                                                    \
            size_t values_idx_in_dim = 0;                                                          \
            for (size_t i = 0; i < num_indices; i++) {                                             \
                int32_t target_idx = indices[i];                                                   \
                if (target_idx < 0) {                                                              \
                    target_idx += (int32_t)input_shape[dim];                                       \
                }                                                                                  \
                if ((size_t)target_idx == output_indices[dim]) {                                   \
                    found = true;                                                                  \
                    values_idx_in_dim = i;                                                         \
                    break;                                                                         \
                }                                                                                  \
            }                                                                                      \
            if (found) {                                                                           \
                size_t values_flat_idx = values_offset;                                            \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    size_t idx = (i == dim) ? values_idx_in_dim : output_indices[i];               \
                    values_flat_idx += idx * values_strides[i];                                    \
                }                                                                                  \
                out[id] = values[values_flat_idx];                                                 \
            } else {                                                                               \
                size_t input_flat_idx = input_offset;                                              \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_flat_idx += output_indices[i] * input_strides[i];                        \
                }                                                                                  \
                out[id] = input[input_flat_idx];                                                   \
            }                                                                                      \
        }                                                                                          \
    }

#define GATHER_OP(TYPENAME, FN_NAME)                                                               \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *input, const int32_t *indices,  \
                                                   TYPENAME *out, const size_t *metadata) {        \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *output_shape = metadata + 2;                                                 \
        const size_t *input_shape = metadata + 2 + num_dims;                                       \
        const size_t *input_strides = metadata + 2 + 2 * num_dims;                                 \
        const size_t *indices_strides = metadata + 2 + 3 * num_dims;                               \
        const size_t input_offset = metadata[2 + 4 * num_dims];                                    \
        const size_t indices_offset = metadata[2 + 4 * num_dims + 1];                              \
        const size_t dim = metadata[2 + 4 * num_dims + 2];                                         \
        for (uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_els;                    \
             id += blockDim.x * gridDim.x) {                                                       \
            size_t output_indices[16];                                                             \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
            size_t indices_flat_idx = indices_offset;                                              \
            for (size_t d = 0; d < num_dims; d++) {                                                \
                indices_flat_idx += output_indices[d] * indices_strides[d];                        \
            }                                                                                      \
            int32_t selected_idx = indices[indices_flat_idx];                                      \
            if (selected_idx < 0) {                                                                \
                selected_idx += (int32_t)input_shape[dim];                                         \
            }                                                                                      \
            if (selected_idx < 0 || (size_t)selected_idx >= input_shape[dim]) {                    \
                out[id] = TYPENAME{};                                                              \
                continue;                                                                          \
            }                                                                                      \
            size_t input_flat_idx = input_offset;                                                  \
            for (size_t d = 0; d < num_dims; d++) {                                                \
                if (d == dim) {                                                                    \
                    input_flat_idx += ((size_t)selected_idx) * input_strides[d];                   \
                } else {                                                                           \
                    input_flat_idx += output_indices[d] * input_strides[d];                        \
                }                                                                                  \
            }                                                                                      \
            out[id] = input[input_flat_idx];                                                       \
        }                                                                                          \
    }

#define SCATTER_OP(TYPENAME, FN_NAME)                                                              \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *input, const int32_t *indices,  \
                                                   const TYPENAME *src, TYPENAME *out,             \
                                                   const size_t *metadata) {                       \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t *src_shape = metadata + 2 + 2 * num_dims;                                     \
        const size_t *src_strides = metadata + 2 + 3 * num_dims;                                   \
        const size_t *indices_strides = metadata + 2 + 4 * num_dims;                               \
        const size_t input_offset = metadata[2 + 5 * num_dims];                                    \
        const size_t src_offset = metadata[2 + 5 * num_dims + 1];                                  \
        const size_t indices_offset = metadata[2 + 5 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 5 * num_dims + 3];                                         \
        for (uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_els;                    \
             id += blockDim.x * gridDim.x) {                                                       \
            size_t src_indices[16];                                                                \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                src_indices[d] = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
            }                                                                                      \
            size_t indices_flat_idx = indices_offset;                                              \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                indices_flat_idx += src_indices[i] * indices_strides[i];                           \
            }                                                                                      \
            int32_t target_idx = indices[indices_flat_idx];                                        \
            if (target_idx < 0) {                                                                  \
                target_idx += (int32_t)input_shape[dim];                                           \
            }                                                                                      \
            if (target_idx < 0 || (size_t)target_idx >= input_shape[dim]) {                        \
                continue;                                                                          \
            }                                                                                      \
            size_t output_flat_idx = input_offset;                                                 \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                size_t idx = (i == dim) ? (size_t)target_idx : src_indices[i];                     \
                output_flat_idx += idx * input_strides[i];                                         \
            }                                                                                      \
            size_t src_flat_idx = src_offset;                                                      \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                src_flat_idx += src_indices[i] * src_strides[i];                                   \
            }                                                                                      \
            out[output_flat_idx] = src[src_flat_idx];                                              \
        }                                                                                          \
    }

#define SCATTER_ADD_OP(TYPENAME, FN_NAME, ATOMIC_ADD)                                              \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *input, const int32_t *indices,  \
                                                   const TYPENAME *src, TYPENAME *out,             \
                                                   const size_t *metadata) {                       \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t *src_shape = metadata + 2 + 2 * num_dims;                                     \
        const size_t *src_strides = metadata + 2 + 3 * num_dims;                                   \
        const size_t *indices_strides = metadata + 2 + 4 * num_dims;                               \
        const size_t input_offset = metadata[2 + 5 * num_dims];                                    \
        const size_t src_offset = metadata[2 + 5 * num_dims + 1];                                  \
        const size_t indices_offset = metadata[2 + 5 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 5 * num_dims + 3];                                         \
        for (uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_els;                    \
             id += blockDim.x * gridDim.x) {                                                       \
            size_t src_indices[16];                                                                \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                src_indices[d] = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
            }                                                                                      \
            size_t indices_flat_idx = indices_offset;                                              \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                indices_flat_idx += src_indices[i] * indices_strides[i];                           \
            }                                                                                      \
            int32_t target_idx = indices[indices_flat_idx];                                        \
            if (target_idx < 0) {                                                                  \
                target_idx += (int32_t)input_shape[dim];                                           \
            }                                                                                      \
            if (target_idx < 0 || (size_t)target_idx >= input_shape[dim]) {                        \
                continue;                                                                          \
            }                                                                                      \
            size_t output_flat_idx = input_offset;                                                 \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                size_t idx = (i == dim) ? (size_t)target_idx : src_indices[i];                     \
                output_flat_idx += idx * input_strides[i];                                         \
            }                                                                                      \
            size_t src_flat_idx = src_offset;                                                      \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                src_flat_idx += src_indices[i] * src_strides[i];                                   \
            }                                                                                      \
            ATOMIC_ADD(&out[output_flat_idx], src[src_flat_idx]);                                  \
        }                                                                                          \
    }

#define SCATTER_MAX_OP(TYPENAME, FN_NAME, ATOMIC_MAX)                                              \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *input, const int32_t *indices,  \
                                                   const TYPENAME *src, TYPENAME *out,             \
                                                   const size_t *metadata) {                       \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t *src_shape = metadata + 2 + 2 * num_dims;                                     \
        const size_t *src_strides = metadata + 2 + 3 * num_dims;                                   \
        const size_t *indices_strides = metadata + 2 + 4 * num_dims;                               \
        const size_t input_offset = metadata[2 + 5 * num_dims];                                    \
        const size_t src_offset = metadata[2 + 5 * num_dims + 1];                                  \
        const size_t indices_offset = metadata[2 + 5 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 5 * num_dims + 3];                                         \
        for (uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_els;                    \
             id += blockDim.x * gridDim.x) {                                                       \
            size_t src_indices[16];                                                                \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                src_indices[d] = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
            }                                                                                      \
            size_t indices_flat_idx = indices_offset;                                              \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                indices_flat_idx += src_indices[i] * indices_strides[i];                           \
            }                                                                                      \
            int32_t target_idx = indices[indices_flat_idx];                                        \
            if (target_idx < 0) {                                                                  \
                target_idx += (int32_t)input_shape[dim];                                           \
            }                                                                                      \
            if (target_idx < 0 || (size_t)target_idx >= input_shape[dim]) {                        \
                continue;                                                                          \
            }                                                                                      \
            size_t output_flat_idx = input_offset;                                                 \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                size_t idx = (i == dim) ? (size_t)target_idx : src_indices[i];                     \
                output_flat_idx += idx * input_strides[i];                                         \
            }                                                                                      \
            size_t src_flat_idx = src_offset;                                                      \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                src_flat_idx += src_indices[i] * src_strides[i];                                   \
            }                                                                                      \
            ATOMIC_MAX(&out[output_flat_idx], src[src_flat_idx]);                                  \
        }                                                                                          \
    }

#define SCATTER_MIN_OP(TYPENAME, FN_NAME, ATOMIC_MIN)                                              \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *input, const int32_t *indices,  \
                                                   const TYPENAME *src, TYPENAME *out,             \
                                                   const size_t *metadata) {                       \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t *src_shape = metadata + 2 + 2 * num_dims;                                     \
        const size_t *src_strides = metadata + 2 + 3 * num_dims;                                   \
        const size_t *indices_strides = metadata + 2 + 4 * num_dims;                               \
        const size_t input_offset = metadata[2 + 5 * num_dims];                                    \
        const size_t src_offset = metadata[2 + 5 * num_dims + 1];                                  \
        const size_t indices_offset = metadata[2 + 5 * num_dims + 2];                              \
        const size_t dim = metadata[2 + 5 * num_dims + 3];                                         \
        for (uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_els;                    \
             id += blockDim.x * gridDim.x) {                                                       \
            size_t src_indices[16];                                                                \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                src_indices[d] = temp % src_shape[d];                                              \
                temp /= src_shape[d];                                                              \
            }                                                                                      \
            size_t indices_flat_idx = indices_offset;                                              \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                indices_flat_idx += src_indices[i] * indices_strides[i];                           \
            }                                                                                      \
            int32_t target_idx = indices[indices_flat_idx];                                        \
            if (target_idx < 0) {                                                                  \
                target_idx += (int32_t)input_shape[dim];                                           \
            }                                                                                      \
            if (target_idx < 0 || (size_t)target_idx >= input_shape[dim]) {                        \
                continue;                                                                          \
            }                                                                                      \
            size_t output_flat_idx = input_offset;                                                 \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                size_t idx = (i == dim) ? (size_t)target_idx : src_indices[i];                     \
                output_flat_idx += idx * input_strides[i];                                         \
            }                                                                                      \
            size_t src_flat_idx = src_offset;                                                      \
            for (size_t i = 0; i < num_dims; i++) {                                                \
                src_flat_idx += src_indices[i] * src_strides[i];                                   \
            }                                                                                      \
            ATOMIC_MIN(&out[output_flat_idx], src[src_flat_idx]);                                  \
        }                                                                                          \
    }

INDEX_SELECT_OP(bool, index_select_bool)
INDEX_SELECT_OP(__nv_fp8_e4m3, index_select_f8e4m3)
INDEX_SELECT_OP(__nv_fp8_e5m2, index_select_f8e5m2)
INDEX_SELECT_OP(__nv_bfloat16, index_select_bf16)
INDEX_SELECT_OP(__half, index_select_f16)
INDEX_SELECT_OP(float, index_select_f32)
INDEX_SELECT_OP(double, index_select_f64)
INDEX_SELECT_OP(uint8_t, index_select_u8)
INDEX_SELECT_OP(uint16_t, index_select_u16)
INDEX_SELECT_OP(uint32_t, index_select_u32)
INDEX_SELECT_OP(uint64_t, index_select_u64)
INDEX_SELECT_OP(int8_t, index_select_i8)
INDEX_SELECT_OP(int16_t, index_select_i16)
INDEX_SELECT_OP(int32_t, index_select_i32)
INDEX_SELECT_OP(int64_t, index_select_i64)

INDEX_PUT_OP(bool, index_put_bool)
INDEX_PUT_OP(__nv_fp8_e4m3, index_put_f8e4m3)
INDEX_PUT_OP(__nv_fp8_e5m2, index_put_f8e5m2)
INDEX_PUT_OP(__nv_bfloat16, index_put_bf16)
INDEX_PUT_OP(__half, index_put_f16)
INDEX_PUT_OP(float, index_put_f32)
INDEX_PUT_OP(double, index_put_f64)
INDEX_PUT_OP(uint8_t, index_put_u8)
INDEX_PUT_OP(uint16_t, index_put_u16)
INDEX_PUT_OP(uint32_t, index_put_u32)
INDEX_PUT_OP(uint64_t, index_put_u64)
INDEX_PUT_OP(int8_t, index_put_i8)
INDEX_PUT_OP(int16_t, index_put_i16)
INDEX_PUT_OP(int32_t, index_put_i32)
INDEX_PUT_OP(int64_t, index_put_i64)

GATHER_OP(bool, gather_bool)
GATHER_OP(__nv_fp8_e4m3, gather_f8e4m3)
GATHER_OP(__nv_fp8_e5m2, gather_f8e5m2)
GATHER_OP(__nv_bfloat16, gather_bf16)
GATHER_OP(__half, gather_f16)
GATHER_OP(float, gather_f32)
GATHER_OP(double, gather_f64)
GATHER_OP(uint8_t, gather_u8)
GATHER_OP(uint16_t, gather_u16)
GATHER_OP(uint32_t, gather_u32)
GATHER_OP(uint64_t, gather_u64)
GATHER_OP(int8_t, gather_i8)
GATHER_OP(int16_t, gather_i16)
GATHER_OP(int32_t, gather_i32)
GATHER_OP(int64_t, gather_i64)

SCATTER_OP(bool, scatter_bool)
SCATTER_OP(__nv_fp8_e4m3, scatter_f8e4m3)
SCATTER_OP(__nv_fp8_e5m2, scatter_f8e5m2)
SCATTER_OP(__nv_bfloat16, scatter_bf16)
SCATTER_OP(__half, scatter_f16)
SCATTER_OP(float, scatter_f32)
SCATTER_OP(double, scatter_f64)
SCATTER_OP(uint8_t, scatter_u8)
SCATTER_OP(uint16_t, scatter_u16)
SCATTER_OP(uint32_t, scatter_u32)
SCATTER_OP(uint64_t, scatter_u64)
SCATTER_OP(int8_t, scatter_i8)
SCATTER_OP(int16_t, scatter_i16)
SCATTER_OP(int32_t, scatter_i32)
SCATTER_OP(int64_t, scatter_i64)

SCATTER_ADD_OP(__nv_fp8_e4m3, scatter_add_f8e4m3, atomic_add_f8e4m3)
SCATTER_ADD_OP(__nv_fp8_e5m2, scatter_add_f8e5m2, atomic_add_f8e5m2)
SCATTER_ADD_OP(__nv_bfloat16, scatter_add_bf16, atomic_add_bf16)
SCATTER_ADD_OP(__half, scatter_add_f16, atomic_add_f16)
SCATTER_ADD_OP(float, scatter_add_f32, atomic_add_f32)
SCATTER_ADD_OP(double, scatter_add_f64, atomic_add_f64)
SCATTER_ADD_OP(int8_t, scatter_add_i8, atomic_add_i8)
SCATTER_ADD_OP(int16_t, scatter_add_i16, atomic_add_i16)
SCATTER_ADD_OP(int32_t, scatter_add_i32, atomic_add_i32)
SCATTER_ADD_OP(int64_t, scatter_add_i64, atomic_add_i64)
SCATTER_ADD_OP(uint8_t, scatter_add_u8, atomic_add_u8)
SCATTER_ADD_OP(uint16_t, scatter_add_u16, atomic_add_u16)
SCATTER_ADD_OP(uint32_t, scatter_add_u32, atomic_add_u32)
SCATTER_ADD_OP(uint64_t, scatter_add_u64, atomic_add_u64)

SCATTER_MAX_OP(__nv_fp8_e4m3, scatter_max_f8e4m3, atomic_max_f8e4m3)
SCATTER_MAX_OP(__nv_fp8_e5m2, scatter_max_f8e5m2, atomic_max_f8e5m2)
SCATTER_MAX_OP(__nv_bfloat16, scatter_max_bf16, atomic_max_bf16)
SCATTER_MAX_OP(__half, scatter_max_f16, atomic_max_f16)
SCATTER_MAX_OP(float, scatter_max_f32, atomic_max_f32)
SCATTER_MAX_OP(double, scatter_max_f64, atomic_max_f64)
SCATTER_MAX_OP(int8_t, scatter_max_i8, atomic_max_i8)
SCATTER_MAX_OP(int16_t, scatter_max_i16, atomic_max_i16)
SCATTER_MAX_OP(int32_t, scatter_max_i32, atomic_max_i32)
SCATTER_MAX_OP(int64_t, scatter_max_i64, atomic_max_i64)
SCATTER_MAX_OP(uint8_t, scatter_max_u8, atomic_max_u8)
SCATTER_MAX_OP(uint16_t, scatter_max_u16, atomic_max_u16)
SCATTER_MAX_OP(uint32_t, scatter_max_u32, atomic_max_u32)
SCATTER_MAX_OP(uint64_t, scatter_max_u64, atomic_max_u64)

SCATTER_MIN_OP(__nv_fp8_e4m3, scatter_min_f8e4m3, atomic_min_f8e4m3)
SCATTER_MIN_OP(__nv_fp8_e5m2, scatter_min_f8e5m2, atomic_min_f8e5m2)
SCATTER_MIN_OP(__nv_bfloat16, scatter_min_bf16, atomic_min_bf16)
SCATTER_MIN_OP(__half, scatter_min_f16, atomic_min_f16)
SCATTER_MIN_OP(float, scatter_min_f32, atomic_min_f32)
SCATTER_MIN_OP(double, scatter_min_f64, atomic_min_f64)
SCATTER_MIN_OP(int8_t, scatter_min_i8, atomic_min_i8)
SCATTER_MIN_OP(int16_t, scatter_min_i16, atomic_min_i16)
SCATTER_MIN_OP(int32_t, scatter_min_i32, atomic_min_i32)
SCATTER_MIN_OP(int64_t, scatter_min_i64, atomic_min_i64)
SCATTER_MIN_OP(uint8_t, scatter_min_u8, atomic_min_u8)
SCATTER_MIN_OP(uint16_t, scatter_min_u16, atomic_min_u16)
SCATTER_MIN_OP(uint32_t, scatter_min_u32, atomic_min_u32)
SCATTER_MIN_OP(uint64_t, scatter_min_u64, atomic_min_u64)

// ============================================================================
// ONEHOT OPERATION
// ============================================================================

// Convert integer indices to one-hot encoded vectors.
// Input: indices tensor (int32)
// Output: one-hot encoded tensor
//
// Metadata layout:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_input_els (total number of input indices)
// - metadata[2]: num_classes (depth of one-hot dimension)
// - metadata[3]: axis (dimension for one-hot encoding)
// - metadata[4]: num_dims_out (number of output dimensions)
// - metadata[5..5+num_dims_out]: output_shape

#define ONEHOT_OP(TYPENAME, FN_NAME, ONE_VALUE, ZERO_VALUE)                                        \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const int32_t *indices, TYPENAME *output,       \
                                                   const size_t *metadata) {                       \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_input_els = metadata[1];                                                  \
        const size_t num_classes = metadata[2];                                                    \
        const size_t axis = metadata[3];                                                           \
        const size_t num_dims_out = metadata[4];                                                   \
        const size_t *output_shape = metadata + 5;                                                 \
                                                                                                   \
        /* Grid-stride loop */                                                                     \
        for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_els;                      \
             id += blockDim.x * gridDim.x) {                                                       \
                                                                                                   \
            /* Calculate output strides (row-major) */                                             \
            size_t output_strides[16];                                                             \
            output_strides[num_dims_out - 1] = 1;                                                  \
            for (int d = (int)num_dims_out - 2; d >= 0; d--) {                                     \
                output_strides[d] = output_strides[d + 1] * output_shape[d + 1];                   \
            }                                                                                      \
                                                                                                   \
            /* Calculate multi-dimensional output indices from flat id */                          \
            size_t output_indices[16];                                                             \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims_out - 1; d >= 0; d--) {                                     \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
            /* Get the class index at this position */                                             \
            size_t class_idx = output_indices[axis];                                               \
                                                                                                   \
            /* Calculate input shape (output shape without axis dimension) */                      \
            size_t input_shape[16];                                                                \
            size_t num_dims_in = num_dims_out - 1;                                                 \
            for (size_t d = 0, id_in = 0; d < num_dims_out; d++) {                                 \
                if (d != axis) {                                                                   \
                    input_shape[id_in++] = output_shape[d];                                        \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            /* Calculate flat input index (skip axis dimension) */                                 \
            size_t input_flat_idx = 0;                                                             \
            size_t input_stride = 1;                                                               \
            for (int d = (int)num_dims_in - 1; d >= 0; d--) {                                      \
                /* Map output dimension to input dimension */                                      \
                size_t out_d = (size_t)d >= axis ? d + 1 : d;                                      \
                input_flat_idx += output_indices[out_d] * input_stride;                            \
                input_stride *= input_shape[d];                                                    \
            }                                                                                      \
                                                                                                   \
            /* Get the target class from indices */                                                \
            int32_t target_class = indices[input_flat_idx];                                        \
                                                                                                   \
            /* Handle negative indices */                                                          \
            if (target_class < 0) {                                                                \
                target_class += (int32_t)num_classes;                                              \
            }                                                                                      \
                                                                                                   \
            /* Set output value */                                                                 \
            if (target_class >= 0 && (size_t)target_class < num_classes &&                         \
                class_idx == (size_t)target_class) {                                               \
                output[id] = ONE_VALUE;                                                            \
            } else {                                                                               \
                output[id] = ZERO_VALUE;                                                           \
            }                                                                                      \
        }                                                                                          \
    }

// Define onehot operations for all output types
ONEHOT_OP(bool, onehot_bool, true, false)
ONEHOT_OP(__nv_fp8_e4m3, onehot_f8e4m3, __nv_fp8_e4m3(1.0f), __nv_fp8_e4m3(0.0f))
ONEHOT_OP(__nv_fp8_e5m2, onehot_f8e5m2, __nv_fp8_e5m2(1.0f), __nv_fp8_e5m2(0.0f))
ONEHOT_OP(__nv_bfloat16, onehot_bf16, __nv_bfloat16(1.0f), __nv_bfloat16(0.0f))
ONEHOT_OP(__half, onehot_f16, __half(1.0f), __half(0.0f))
ONEHOT_OP(float, onehot_f32, 1.0f, 0.0f)
ONEHOT_OP(double, onehot_f64, 1.0, 0.0)
ONEHOT_OP(int8_t, onehot_i8, 1, 0)
ONEHOT_OP(int16_t, onehot_i16, 1, 0)
ONEHOT_OP(int32_t, onehot_i32, 1, 0)
ONEHOT_OP(int64_t, onehot_i64, 1, 0)
ONEHOT_OP(uint8_t, onehot_u8, 1, 0)
ONEHOT_OP(uint16_t, onehot_u16, 1, 0)
ONEHOT_OP(uint32_t, onehot_u32, 1, 0)
ONEHOT_OP(uint64_t, onehot_u64, 1, 0)

// ============================================================================
// NONZERO OPERATIONS
// ============================================================================
//
// Returns indices of non-zero elements in the input tensor.
// This is a two-pass operation:
//   1. nonzero_count - counts non-zero elements using atomic add
//   2. nonzero_fill - fills output with indices using atomic counter
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements in input)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: input_shape
// - metadata[2+num_dims..2+2*num_dims]: input_strides
// - metadata[2+2*num_dims]: input_offset

// Count kernel - uses atomic to count nonzero elements
#define NONZERO_COUNT_OP(TYPE, FN_SUFFIX, IS_NONZERO)                                              \
    extern "C" __global__ void hodu_cuda_nonzero_count_##FN_SUFFIX(                                \
        const TYPE *input, unsigned int *count, const size_t *metadata) {                          \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
                                                                                                   \
        for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_els;                      \
             id += blockDim.x * gridDim.x) {                                                       \
                                                                                                   \
            /* Compute flat index from strided layout */                                           \
            size_t flat_idx = input_offset;                                                        \
            size_t temp = id;                                                                      \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                size_t idx = temp % input_shape[d];                                                \
                temp /= input_shape[d];                                                            \
                flat_idx += idx * input_strides[d];                                                \
            }                                                                                      \
                                                                                                   \
            TYPE val = input[flat_idx];                                                            \
            if (IS_NONZERO) {                                                                      \
                atomicAdd(count, 1);                                                               \
            }                                                                                      \
        }                                                                                          \
    }

// Fill kernel - uses atomic counter to fill indices
#define NONZERO_FILL_OP(TYPE, FN_SUFFIX, IS_NONZERO)                                               \
    extern "C" __global__ void hodu_cuda_nonzero_fill_##FN_SUFFIX(                                 \
        const TYPE *input, int32_t *output, unsigned int *counter, const size_t *metadata) {       \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *input_shape = metadata + 2;                                                  \
        const size_t *input_strides = metadata + 2 + num_dims;                                     \
        const size_t input_offset = metadata[2 + 2 * num_dims];                                    \
                                                                                                   \
        for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_els;                      \
             id += blockDim.x * gridDim.x) {                                                       \
                                                                                                   \
            /* Compute flat index and multi-dimensional indices */                                 \
            size_t flat_idx = input_offset;                                                        \
            size_t temp = id;                                                                      \
            size_t multi_idx[16];                                                                  \
            for (int d = (int)num_dims - 1; d >= 0; d--) {                                         \
                multi_idx[d] = temp % input_shape[d];                                              \
                temp /= input_shape[d];                                                            \
                flat_idx += multi_idx[d] * input_strides[d];                                       \
            }                                                                                      \
                                                                                                   \
            TYPE val = input[flat_idx];                                                            \
            if (IS_NONZERO) {                                                                      \
                unsigned int out_idx = atomicAdd(counter, 1);                                      \
                for (size_t d = 0; d < num_dims; d++) {                                            \
                    output[out_idx * num_dims + d] = (int32_t)multi_idx[d];                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

// Count operations
NONZERO_COUNT_OP(bool, bool, val)
NONZERO_COUNT_OP(__nv_fp8_e4m3, f8e4m3,
                 __half2float(__nv_cvt_fp8_to_halfraw(val, __NV_E4M3)) != 0.0f)
NONZERO_COUNT_OP(__nv_fp8_e5m2, f8e5m2,
                 __half2float(__nv_cvt_fp8_to_halfraw(val, __NV_E5M2)) != 0.0f)
NONZERO_COUNT_OP(__nv_bfloat16, bf16, __bfloat162float(val) != 0.0f)
NONZERO_COUNT_OP(__half, f16, __half2float(val) != 0.0f)
NONZERO_COUNT_OP(float, f32, val != 0.0f)
NONZERO_COUNT_OP(double, f64, val != 0.0)
NONZERO_COUNT_OP(int8_t, i8, val != 0)
NONZERO_COUNT_OP(int16_t, i16, val != 0)
NONZERO_COUNT_OP(int32_t, i32, val != 0)
NONZERO_COUNT_OP(int64_t, i64, val != 0)
NONZERO_COUNT_OP(uint8_t, u8, val != 0)
NONZERO_COUNT_OP(uint16_t, u16, val != 0)
NONZERO_COUNT_OP(uint32_t, u32, val != 0)
NONZERO_COUNT_OP(uint64_t, u64, val != 0)

// Fill operations
NONZERO_FILL_OP(bool, bool, val)
NONZERO_FILL_OP(__nv_fp8_e4m3, f8e4m3,
                __half2float(__nv_cvt_fp8_to_halfraw(val, __NV_E4M3)) != 0.0f)
NONZERO_FILL_OP(__nv_fp8_e5m2, f8e5m2,
                __half2float(__nv_cvt_fp8_to_halfraw(val, __NV_E5M2)) != 0.0f)
NONZERO_FILL_OP(__nv_bfloat16, bf16, __bfloat162float(val) != 0.0f)
NONZERO_FILL_OP(__half, f16, __half2float(val) != 0.0f)
NONZERO_FILL_OP(float, f32, val != 0.0f)
NONZERO_FILL_OP(double, f64, val != 0.0)
NONZERO_FILL_OP(int8_t, i8, val != 0)
NONZERO_FILL_OP(int16_t, i16, val != 0)
NONZERO_FILL_OP(int32_t, i32, val != 0)
NONZERO_FILL_OP(int64_t, i64, val != 0)
NONZERO_FILL_OP(uint8_t, u8, val != 0)
NONZERO_FILL_OP(uint16_t, u16, val != 0)
NONZERO_FILL_OP(uint32_t, u32, val != 0)
NONZERO_FILL_OP(uint64_t, u64, val != 0)
