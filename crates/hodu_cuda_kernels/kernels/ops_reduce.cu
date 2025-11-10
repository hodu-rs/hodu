#include "math.cuh"
#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define REDUCE_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME, INIT_VAL, ACCUMULATE)                        \
    extern "C" __global__ void FN_NAME(const IN_TYPENAME *input, OUT_TYPENAME *output,             \
                                       const size_t *metadata) {                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
        for (uint32_t output_idx = blockIdx.x * blockDim.x + threadIdx.x; output_idx < num_els;    \
             output_idx += blockDim.x * gridDim.x) {                                               \
            float acc = INIT_VAL;                                                                  \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                float val = to_float(input[flat_index]);                                           \
                ACCUMULATE;                                                                        \
            }                                                                                      \
            output[output_idx] = from_float<OUT_TYPENAME>(acc);                                    \
        }                                                                                          \
    }

#define REDUCE_MEAN_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME)                                         \
    extern "C" __global__ void FN_NAME(const IN_TYPENAME *input, OUT_TYPENAME *output,             \
                                       const size_t *metadata) {                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
        for (uint32_t output_idx = blockIdx.x * blockDim.x + threadIdx.x; output_idx < num_els;    \
             output_idx += blockDim.x * gridDim.x) {                                               \
            float sum = 0.0f;                                                                      \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                sum += to_float(input[flat_index]);                                                \
            }                                                                                      \
            output[output_idx] = from_float<OUT_TYPENAME>(sum / (float)reduce_size);               \
        }                                                                                          \
    }

#define REDUCE_NORM_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME)                                         \
    extern "C" __global__ void FN_NAME(const IN_TYPENAME *input, OUT_TYPENAME *output,             \
                                       const size_t *metadata) {                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
        for (uint32_t output_idx = blockIdx.x * blockDim.x + threadIdx.x; output_idx < num_els;    \
             output_idx += blockDim.x * gridDim.x) {                                               \
            float sum_squares = 0.0f;                                                              \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                float val = to_float(input[flat_index]);                                           \
                sum_squares += val * val;                                                          \
            }                                                                                      \
            output[output_idx] = from_float<OUT_TYPENAME>(sqrtf(sum_squares));                     \
        }                                                                                          \
    }

#define REDUCE_ARGMAX_OP(IN_TYPENAME, FN_NAME)                                                     \
    extern "C" __global__ void FN_NAME(const IN_TYPENAME *input, int32_t *output,                  \
                                       const size_t *metadata) {                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
        for (uint32_t output_idx = blockIdx.x * blockDim.x + threadIdx.x; output_idx < num_els;    \
             output_idx += blockDim.x * gridDim.x) {                                               \
            float max_val;                                                                         \
            int32_t max_idx = 0;                                                                   \
            bool first = true;                                                                     \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                size_t actual_dim_idx = input_indices[reduce_dims[0]];                             \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                float val = to_float(input[flat_index]);                                           \
                if (first || val > max_val) {                                                      \
                    max_val = val;                                                                 \
                    max_idx = (int32_t)actual_dim_idx;                                             \
                    first = false;                                                                 \
                }                                                                                  \
            }                                                                                      \
            output[output_idx] = max_idx;                                                          \
        }                                                                                          \
    }

#define REDUCE_ARGMIN_OP(IN_TYPENAME, FN_NAME)                                                     \
    extern "C" __global__ void FN_NAME(const IN_TYPENAME *input, int32_t *output,                  \
                                       const size_t *metadata) {                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
        for (uint32_t output_idx = blockIdx.x * blockDim.x + threadIdx.x; output_idx < num_els;    \
             output_idx += blockDim.x * gridDim.x) {                                               \
            float min_val;                                                                         \
            int32_t min_idx = 0;                                                                   \
            bool first = true;                                                                     \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                size_t actual_dim_idx = input_indices[reduce_dims[0]];                             \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                float val = to_float(input[flat_index]);                                           \
                if (first || val < min_val) {                                                      \
                    min_val = val;                                                                 \
                    min_idx = (int32_t)actual_dim_idx;                                             \
                    first = false;                                                                 \
                }                                                                                  \
            }                                                                                      \
            output[output_idx] = min_idx;                                                          \
        }                                                                                          \
    }

#define REDUCE_ANY_OP(IN_TYPENAME, FN_NAME)                                                        \
    extern "C" __global__ void FN_NAME(const IN_TYPENAME *input, bool *output,                     \
                                       const size_t *metadata) {                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
        for (uint32_t output_idx = blockIdx.x * blockDim.x + threadIdx.x; output_idx < num_els;    \
             output_idx += blockDim.x * gridDim.x) {                                               \
            bool result = false;                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                if (is_nonzero(input[flat_index])) {                                               \
                    result = true;                                                                 \
                    break;                                                                         \
                }                                                                                  \
            }                                                                                      \
            output[output_idx] = result;                                                           \
        }                                                                                          \
    }

#define REDUCE_ALL_OP(IN_TYPENAME, FN_NAME)                                                        \
    extern "C" __global__ void FN_NAME(const IN_TYPENAME *input, bool *output,                     \
                                       const size_t *metadata) {                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const bool keep_dim =                                                                      \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims] != 0;                  \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
        for (uint32_t output_idx = blockIdx.x * blockDim.x + threadIdx.x; output_idx < num_els;    \
             output_idx += blockDim.x * gridDim.x) {                                               \
            bool result = true;                                                                    \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
            size_t input_indices[16];                                                              \
            if (keep_dim) {                                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    input_indices[i] = output_indices[i];                                          \
                }                                                                                  \
            } else {                                                                               \
                size_t out_idx = 0;                                                                \
                for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {                             \
                    bool is_reduced = false;                                                       \
                    for (size_t r = 0; r < num_reduce_dims; r++) {                                 \
                        if (reduce_dims[r] == in_dim) {                                            \
                            is_reduced = true;                                                     \
                            break;                                                                 \
                        }                                                                          \
                    }                                                                              \
                    if (is_reduced) {                                                              \
                        input_indices[in_dim] = 0;                                                 \
                    } else {                                                                       \
                        input_indices[in_dim] =                                                    \
                            (out_idx < output_shape_len) ? output_indices[out_idx] : 0;            \
                        out_idx++;                                                                 \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                if (!is_nonzero(input[flat_index])) {                                              \
                    result = false;                                                                \
                    break;                                                                         \
                }                                                                                  \
            }                                                                                      \
            output[output_idx] = result;                                                           \
        }                                                                                          \
    }

REDUCE_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, sum_f8e4m3, 0.0f, acc += val)
REDUCE_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, sum_f8e5m2, 0.0f, acc += val)
REDUCE_OP(__nv_bfloat16, __nv_bfloat16, sum_bf16, 0.0f, acc += val)
REDUCE_OP(__half, __half, sum_f16, 0.0f, acc += val)
REDUCE_OP(float, float, sum_f32, 0.0f, acc += val)
REDUCE_OP(double, double, sum_f64, 0.0f, acc += val)
REDUCE_OP(int8_t, int8_t, sum_i8, 0.0f, acc += val)
REDUCE_OP(int16_t, int16_t, sum_i16, 0.0f, acc += val)
REDUCE_OP(int32_t, int32_t, sum_i32, 0.0f, acc += val)
REDUCE_OP(int64_t, int64_t, sum_i64, 0.0f, acc += val)
REDUCE_OP(uint8_t, uint8_t, sum_u8, 0.0f, acc += val)
REDUCE_OP(uint16_t, uint16_t, sum_u16, 0.0f, acc += val)
REDUCE_OP(uint32_t, uint32_t, sum_u32, 0.0f, acc += val)
REDUCE_OP(uint64_t, uint64_t, sum_u64, 0.0f, acc += val)

REDUCE_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, max_f8e4m3, -INFINITY, acc = maximum(acc, val))
REDUCE_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, max_f8e5m2, -INFINITY, acc = maximum(acc, val))
REDUCE_OP(__nv_bfloat16, __nv_bfloat16, max_bf16, -INFINITY, acc = maximum(acc, val))
REDUCE_OP(__half, __half, max_f16, -INFINITY, acc = maximum(acc, val))
REDUCE_OP(float, float, max_f32, -INFINITY, acc = maximum(acc, val))
REDUCE_OP(double, double, max_f64, -INFINITY, acc = maximum(acc, val))
REDUCE_OP(int8_t, int8_t, max_i8, (float)INT8_MIN, acc = maximum(acc, val))
REDUCE_OP(int16_t, int16_t, max_i16, (float)INT16_MIN, acc = maximum(acc, val))
REDUCE_OP(int32_t, int32_t, max_i32, (float)INT32_MIN, acc = maximum(acc, val))
REDUCE_OP(int64_t, int64_t, max_i64, (float)INT64_MIN, acc = maximum(acc, val))
REDUCE_OP(uint8_t, uint8_t, max_u8, 0.0f, acc = maximum(acc, val))
REDUCE_OP(uint16_t, uint16_t, max_u16, 0.0f, acc = maximum(acc, val))
REDUCE_OP(uint32_t, uint32_t, max_u32, 0.0f, acc = maximum(acc, val))
REDUCE_OP(uint64_t, uint64_t, max_u64, 0.0f, acc = maximum(acc, val))

REDUCE_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, min_f8e4m3, INFINITY, acc = minimum(acc, val))
REDUCE_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, min_f8e5m2, INFINITY, acc = minimum(acc, val))
REDUCE_OP(__nv_bfloat16, __nv_bfloat16, min_bf16, INFINITY, acc = minimum(acc, val))
REDUCE_OP(__half, __half, min_f16, INFINITY, acc = minimum(acc, val))
REDUCE_OP(float, float, min_f32, INFINITY, acc = minimum(acc, val))
REDUCE_OP(double, double, min_f64, INFINITY, acc = minimum(acc, val))
REDUCE_OP(int8_t, int8_t, min_i8, (float)INT8_MAX, acc = minimum(acc, val))
REDUCE_OP(int16_t, int16_t, min_i16, (float)INT16_MAX, acc = minimum(acc, val))
REDUCE_OP(int32_t, int32_t, min_i32, (float)INT32_MAX, acc = minimum(acc, val))
REDUCE_OP(int64_t, int64_t, min_i64, (float)INT64_MAX, acc = minimum(acc, val))
REDUCE_OP(uint8_t, uint8_t, min_u8, (float)UINT8_MAX, acc = minimum(acc, val))
REDUCE_OP(uint16_t, uint16_t, min_u16, (float)UINT16_MAX, acc = minimum(acc, val))
REDUCE_OP(uint32_t, uint32_t, min_u32, (float)UINT32_MAX, acc = minimum(acc, val))
REDUCE_OP(uint64_t, uint64_t, min_u64, (float)UINT64_MAX, acc = minimum(acc, val))

REDUCE_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, prod_f8e4m3, 1.0f, acc *= val)
REDUCE_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, prod_f8e5m2, 1.0f, acc *= val)
REDUCE_OP(__nv_bfloat16, __nv_bfloat16, prod_bf16, 1.0f, acc *= val)
REDUCE_OP(__half, __half, prod_f16, 1.0f, acc *= val)
REDUCE_OP(float, float, prod_f32, 1.0f, acc *= val)
REDUCE_OP(double, double, prod_f64, 1.0f, acc *= val)
REDUCE_OP(int8_t, int8_t, prod_i8, 1.0f, acc *= val)
REDUCE_OP(int16_t, int16_t, prod_i16, 1.0f, acc *= val)
REDUCE_OP(int32_t, int32_t, prod_i32, 1.0f, acc *= val)
REDUCE_OP(int64_t, int64_t, prod_i64, 1.0f, acc *= val)
REDUCE_OP(uint8_t, uint8_t, prod_u8, 1.0f, acc *= val)
REDUCE_OP(uint16_t, uint16_t, prod_u16, 1.0f, acc *= val)
REDUCE_OP(uint32_t, uint32_t, prod_u32, 1.0f, acc *= val)
REDUCE_OP(uint64_t, uint64_t, prod_u64, 1.0f, acc *= val)

REDUCE_MEAN_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, mean_f8e4m3)
REDUCE_MEAN_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, mean_f8e5m2)
REDUCE_MEAN_OP(__nv_bfloat16, __nv_bfloat16, mean_bf16)
REDUCE_MEAN_OP(__half, __half, mean_f16)
REDUCE_MEAN_OP(float, float, mean_f32)
REDUCE_MEAN_OP(double, double, mean_f64)

REDUCE_NORM_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, norm_f8e4m3)
REDUCE_NORM_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, norm_f8e5m2)
REDUCE_NORM_OP(__nv_bfloat16, __nv_bfloat16, norm_bf16)
REDUCE_NORM_OP(__half, __half, norm_f16)
REDUCE_NORM_OP(float, float, norm_f32)
REDUCE_NORM_OP(double, double, norm_f64)

REDUCE_ARGMAX_OP(__nv_fp8_e4m3, argmax_f8e4m3)
REDUCE_ARGMAX_OP(__nv_fp8_e5m2, argmax_f8e5m2)
REDUCE_ARGMAX_OP(__nv_bfloat16, argmax_bf16)
REDUCE_ARGMAX_OP(__half, argmax_f16)
REDUCE_ARGMAX_OP(float, argmax_f32)
REDUCE_ARGMAX_OP(double, argmax_f64)
REDUCE_ARGMAX_OP(int8_t, argmax_i8)
REDUCE_ARGMAX_OP(int16_t, argmax_i16)
REDUCE_ARGMAX_OP(int32_t, argmax_i32)
REDUCE_ARGMAX_OP(int64_t, argmax_i64)
REDUCE_ARGMAX_OP(uint8_t, argmax_u8)
REDUCE_ARGMAX_OP(uint16_t, argmax_u16)
REDUCE_ARGMAX_OP(uint32_t, argmax_u32)
REDUCE_ARGMAX_OP(uint64_t, argmax_u64)

REDUCE_ARGMIN_OP(__nv_fp8_e4m3, argmin_f8e4m3)
REDUCE_ARGMIN_OP(__nv_fp8_e5m2, argmin_f8e5m2)
REDUCE_ARGMIN_OP(__nv_bfloat16, argmin_bf16)
REDUCE_ARGMIN_OP(__half, argmin_f16)
REDUCE_ARGMIN_OP(float, argmin_f32)
REDUCE_ARGMIN_OP(double, argmin_f64)
REDUCE_ARGMIN_OP(int8_t, argmin_i8)
REDUCE_ARGMIN_OP(int16_t, argmin_i16)
REDUCE_ARGMIN_OP(int32_t, argmin_i32)
REDUCE_ARGMIN_OP(int64_t, argmin_i64)
REDUCE_ARGMIN_OP(uint8_t, argmin_u8)
REDUCE_ARGMIN_OP(uint16_t, argmin_u16)
REDUCE_ARGMIN_OP(uint32_t, argmin_u32)
REDUCE_ARGMIN_OP(uint64_t, argmin_u64)

REDUCE_ANY_OP(bool, any_bool)
REDUCE_ANY_OP(__nv_fp8_e4m3, any_f8e4m3)
REDUCE_ANY_OP(__nv_fp8_e5m2, any_f8e5m2)
REDUCE_ANY_OP(__nv_bfloat16, any_bf16)
REDUCE_ANY_OP(__half, any_f16)
REDUCE_ANY_OP(float, any_f32)
REDUCE_ANY_OP(double, any_f64)
REDUCE_ANY_OP(int8_t, any_i8)
REDUCE_ANY_OP(int16_t, any_i16)
REDUCE_ANY_OP(int32_t, any_i32)
REDUCE_ANY_OP(int64_t, any_i64)
REDUCE_ANY_OP(uint8_t, any_u8)
REDUCE_ANY_OP(uint16_t, any_u16)
REDUCE_ANY_OP(uint32_t, any_u32)
REDUCE_ANY_OP(uint64_t, any_u64)

REDUCE_ALL_OP(bool, all_bool)
REDUCE_ALL_OP(__nv_fp8_e4m3, all_f8e4m3)
REDUCE_ALL_OP(__nv_fp8_e5m2, all_f8e5m2)
REDUCE_ALL_OP(__nv_bfloat16, all_bf16)
REDUCE_ALL_OP(__half, all_f16)
REDUCE_ALL_OP(float, all_f32)
REDUCE_ALL_OP(double, all_f64)
REDUCE_ALL_OP(int8_t, all_i8)
REDUCE_ALL_OP(int16_t, all_i16)
REDUCE_ALL_OP(int32_t, all_i32)
REDUCE_ALL_OP(int64_t, all_i64)
REDUCE_ALL_OP(uint8_t, all_u8)
REDUCE_ALL_OP(uint16_t, all_u16)
REDUCE_ALL_OP(uint32_t, all_u32)
REDUCE_ALL_OP(uint64_t, all_u64)
