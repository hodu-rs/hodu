#include "ops_reduce.h"
#include "math_utils.h"
#include "simd_utils.h"
#include "types.h"
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ============================================================================
// GENERIC REDUCTION OPERATIONS
// ============================================================================
//
// All reduction operations follow the same metadata layout:
// - metadata[0]: num_dims (number of dimensions in input)
// - metadata[1..1+num_dims]: dims (shape of input)
// - metadata[1+num_dims..1+2*num_dims]: strides (strides of input)
// - metadata[1+2*num_dims]: offset (starting offset in input)
// - metadata[2+2*num_dims]: output_shape_len (number of dimensions in output)
// - metadata[3+2*num_dims..3+2*num_dims+output_shape_len]: output_shape
// - metadata[3+2*num_dims+output_shape_len]: num_reduce_dims (number of dims to reduce)
// - metadata[4+2*num_dims+output_shape_len..]: reduce_dims (dimension indices to reduce)
// - metadata[...+num_reduce_dims]: keep_dim (1 to keep, 0 to squeeze)
// - metadata[...+1]: reduce_size (total elements to reduce per output)
//
// Algorithm:
// For each output element:
// 1. Compute output multi-dimensional indices
// 2. Map to input indices (accounting for keep_dim and reduced dimensions)
// 3. Iterate over all reduced dimension combinations
// 4. Accumulate values according to the reduction operation
//
// keep_dim behavior:
// - If keep_dim=true: output shape matches input but reduced dims have size 1
// - If keep_dim=false: reduced dimensions are squeezed out of output

/// Macro to implement a generic reduction operation
///
/// @param IN_TYPE Input C type
/// @param OUT_TYPE Output C type (same as input for most ops)
/// @param TYPE_SUFFIX Suffix for function naming
/// @param INIT_VAL Initial accumulator value
/// @param ACCUMULATE Expression to accumulate values (e.g., acc += val)
#define REDUCE_OP(IN_TYPE, OUT_TYPE, TYPE_SUFFIX, INIT_VAL, ACCUMULATE)                            \
    void hodu_cpu_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr, const size_t *metadata) { \
        const IN_TYPE *input = (const IN_TYPE *)input_ptr;                                         \
        OUT_TYPE *output = (OUT_TYPE *)output_ptr;                                                 \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            OUT_TYPE acc = INIT_VAL;                                                               \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                IN_TYPE val = input[flat_index];                                                   \
                ACCUMULATE;                                                                        \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = acc;                                                              \
        }                                                                                          \
    }

REDUCE_OP(f8e4m3_t, f8e4m3_t, sum_f8e4m3, F8E4M3_ZERO, acc = f8e4m3_add(acc, val))
REDUCE_OP(f8e5m2_t, f8e5m2_t, sum_f8e5m2, F8E5M2_ZERO, acc = f8e5m2_add(acc, val))
REDUCE_OP(bf16_t, bf16_t, sum_bf16, BF16_ZERO, acc = bf16_add(acc, val))
REDUCE_OP(f16_t, f16_t, sum_f16, F16_ZERO, acc = f16_add(acc, val))

// SIMD-optimized sum_f32
void hodu_cpu_sum_f32(const void *input_ptr, void *output_ptr, const size_t *metadata) {
    const f32_t *input = (const f32_t *)input_ptr;
    f32_t *output = (f32_t *)output_ptr;

    const size_t num_dims = metadata[0];
    const size_t *dims = metadata + 1;
    const size_t *strides = metadata + 1 + num_dims;
    const size_t offset = metadata[1 + 2 * num_dims];
    const size_t output_shape_len = metadata[2 + 2 * num_dims];
    const size_t *output_shape = metadata + 3 + 2 * num_dims;
    const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];
    const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;
    const size_t keep_dim_val = metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];
    const bool keep_dim = (keep_dim_val != 0);
    const size_t reduce_size = metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];

    size_t num_els = 1;
    for (size_t i = 0; i < output_shape_len; i++) {
        num_els *= output_shape[i];
    }

    for (size_t output_idx = 0; output_idx < num_els; output_idx++) {
        f32_t acc = 0.0f;

        size_t output_indices[16];
        size_t temp = output_idx;
        for (int d = (int)output_shape_len - 1; d >= 0; d--) {
            output_indices[d] = temp % output_shape[d];
            temp /= output_shape[d];
        }

        size_t input_indices[16];
        if (keep_dim) {
            for (size_t i = 0; i < num_dims; i++) {
                input_indices[i] = output_indices[i];
            }
        } else {
            size_t out_idx = 0;
            for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {
                bool is_reduced = false;
                for (size_t r = 0; r < num_reduce_dims; r++) {
                    if (reduce_dims[r] == in_dim) {
                        is_reduced = true;
                        break;
                    }
                }
                if (is_reduced) {
                    input_indices[in_dim] = 0;
                } else {
                    input_indices[in_dim] =
                        (out_idx < output_shape_len) ? output_indices[out_idx] : 0;
                    out_idx++;
                }
            }
        }

#if SIMD_F32_WIDTH > 1
        // SIMD path: Check if we're reducing contiguous data (last dimension with stride 1)
        bool use_simd = (num_reduce_dims == 1) && (reduce_dims[0] == num_dims - 1) &&
                        (strides[num_dims - 1] == 1) && (reduce_size >= SIMD_F32_WIDTH);

        if (use_simd) {
            size_t flat_base = offset;
            for (size_t i = 0; i < num_dims - 1; i++) {
                flat_base += input_indices[i] * strides[i];
            }

            simd_f32_t vacc = simd_f32_set1(0.0f);
            const size_t simd_end = (reduce_size / SIMD_F32_WIDTH) * SIMD_F32_WIDTH;

            for (size_t i = 0; i < simd_end; i += SIMD_F32_WIDTH) {
                simd_f32_t v = simd_f32_load(&input[flat_base + i]);
                vacc = simd_f32_add(vacc, v);
            }
            acc = simd_f32_reduce_add(vacc);

            // Handle remainder
            for (size_t i = simd_end; i < reduce_size; i++) {
                acc += input[flat_base + i];
            }
        } else
#endif
        {
            // Fallback path
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {
                size_t temp_reduced = reduced_idx;
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {
                    size_t dim = reduce_dims[i];
                    input_indices[dim] = temp_reduced % dims[dim];
                    temp_reduced /= dims[dim];
                }

                size_t flat_index = offset;
                for (size_t i = 0; i < num_dims; i++) {
                    flat_index += input_indices[i] * strides[i];
                }

                f32_t val = input[flat_index];
                acc += val;
            }
        }

        output[output_idx] = acc;
    }
}

// SIMD-optimized sum_f64
void hodu_cpu_sum_f64(const void *input_ptr, void *output_ptr, const size_t *metadata) {
    const f64_t *input = (const f64_t *)input_ptr;
    f64_t *output = (f64_t *)output_ptr;

    const size_t num_dims = metadata[0];
    const size_t *dims = metadata + 1;
    const size_t *strides = metadata + 1 + num_dims;
    const size_t offset = metadata[1 + 2 * num_dims];
    const size_t output_shape_len = metadata[2 + 2 * num_dims];
    const size_t *output_shape = metadata + 3 + 2 * num_dims;
    const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];
    const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;
    const size_t keep_dim_val = metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];
    const bool keep_dim = (keep_dim_val != 0);
    const size_t reduce_size = metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];

    size_t num_els = 1;
    for (size_t i = 0; i < output_shape_len; i++) {
        num_els *= output_shape[i];
    }

    for (size_t output_idx = 0; output_idx < num_els; output_idx++) {
        f64_t acc = 0.0;

        size_t output_indices[16];
        size_t temp = output_idx;
        for (int d = (int)output_shape_len - 1; d >= 0; d--) {
            output_indices[d] = temp % output_shape[d];
            temp /= output_shape[d];
        }

        size_t input_indices[16];
        if (keep_dim) {
            for (size_t i = 0; i < num_dims; i++) {
                input_indices[i] = output_indices[i];
            }
        } else {
            size_t out_idx = 0;
            for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {
                bool is_reduced = false;
                for (size_t r = 0; r < num_reduce_dims; r++) {
                    if (reduce_dims[r] == in_dim) {
                        is_reduced = true;
                        break;
                    }
                }
                if (is_reduced) {
                    input_indices[in_dim] = 0;
                } else {
                    input_indices[in_dim] =
                        (out_idx < output_shape_len) ? output_indices[out_idx] : 0;
                    out_idx++;
                }
            }
        }

#if SIMD_F64_WIDTH > 1
        // SIMD path: Check if we're reducing contiguous data (last dimension with stride 1)
        bool use_simd = (num_reduce_dims == 1) && (reduce_dims[0] == num_dims - 1) &&
                        (strides[num_dims - 1] == 1) && (reduce_size >= SIMD_F64_WIDTH);

        if (use_simd) {
            size_t flat_base = offset;
            for (size_t i = 0; i < num_dims - 1; i++) {
                flat_base += input_indices[i] * strides[i];
            }

            simd_f64_t vacc = simd_f64_set1(0.0);
            const size_t simd_end = (reduce_size / SIMD_F64_WIDTH) * SIMD_F64_WIDTH;

            for (size_t i = 0; i < simd_end; i += SIMD_F64_WIDTH) {
                simd_f64_t v = simd_f64_load(&input[flat_base + i]);
                vacc = simd_f64_add(vacc, v);
            }
            acc = simd_f64_reduce_add(vacc);

            // Handle remainder
            for (size_t i = simd_end; i < reduce_size; i++) {
                acc += input[flat_base + i];
            }
        } else
#endif
        {
            // Fallback path
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {
                size_t temp_reduced = reduced_idx;
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {
                    size_t dim = reduce_dims[i];
                    input_indices[dim] = temp_reduced % dims[dim];
                    temp_reduced /= dims[dim];
                }

                size_t flat_index = offset;
                for (size_t i = 0; i < num_dims; i++) {
                    flat_index += input_indices[i] * strides[i];
                }

                f64_t val = input[flat_index];
                acc += val;
            }
        }

        output[output_idx] = acc;
    }
}
REDUCE_OP(int8_t, int8_t, sum_i8, 0, acc += val)
REDUCE_OP(int16_t, int16_t, sum_i16, 0, acc += val)
REDUCE_OP(int32_t, int32_t, sum_i32, 0, acc += val)
REDUCE_OP(int64_t, int64_t, sum_i64, 0, acc += val)
REDUCE_OP(uint8_t, uint8_t, sum_u8, 0u, acc += val)
REDUCE_OP(uint16_t, uint16_t, sum_u16, 0u, acc += val)
REDUCE_OP(uint32_t, uint32_t, sum_u32, 0u, acc += val)
REDUCE_OP(uint64_t, uint64_t, sum_u64, 0u, acc += val)

// Max reduction operations
REDUCE_OP(f8e4m3_t, f8e4m3_t, max_f8e4m3, F8E4M3_NEG_INF, acc = f8e4m3_max(acc, val))
REDUCE_OP(f8e5m2_t, f8e5m2_t, max_f8e5m2, F8E5M2_NEG_INF, acc = f8e5m2_max(acc, val))
REDUCE_OP(bf16_t, bf16_t, max_bf16, BF16_NEG_INF, acc = bf16_max(acc, val))
REDUCE_OP(f16_t, f16_t, max_f16, F16_NEG_INF, acc = f16_max(acc, val))

// SIMD-optimized max_f32
void hodu_cpu_max_f32(const void *input_ptr, void *output_ptr, const size_t *metadata) {
    const f32_t *input = (const f32_t *)input_ptr;
    f32_t *output = (f32_t *)output_ptr;

    const size_t num_dims = metadata[0];
    const size_t *dims = metadata + 1;
    const size_t *strides = metadata + 1 + num_dims;
    const size_t offset = metadata[1 + 2 * num_dims];
    const size_t output_shape_len = metadata[2 + 2 * num_dims];
    const size_t *output_shape = metadata + 3 + 2 * num_dims;
    const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];
    const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;
    const size_t keep_dim_val = metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];
    const bool keep_dim = (keep_dim_val != 0);
    const size_t reduce_size = metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];

    size_t num_els = 1;
    for (size_t i = 0; i < output_shape_len; i++) {
        num_els *= output_shape[i];
    }

    for (size_t output_idx = 0; output_idx < num_els; output_idx++) {
        f32_t acc = -INFINITY;

        size_t output_indices[16];
        size_t temp = output_idx;
        for (int d = (int)output_shape_len - 1; d >= 0; d--) {
            output_indices[d] = temp % output_shape[d];
            temp /= output_shape[d];
        }

        size_t input_indices[16];
        if (keep_dim) {
            for (size_t i = 0; i < num_dims; i++) {
                input_indices[i] = output_indices[i];
            }
        } else {
            size_t out_idx = 0;
            for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {
                bool is_reduced = false;
                for (size_t r = 0; r < num_reduce_dims; r++) {
                    if (reduce_dims[r] == in_dim) {
                        is_reduced = true;
                        break;
                    }
                }
                if (is_reduced) {
                    input_indices[in_dim] = 0;
                } else {
                    input_indices[in_dim] =
                        (out_idx < output_shape_len) ? output_indices[out_idx] : 0;
                    out_idx++;
                }
            }
        }

#if SIMD_F32_WIDTH > 1
        bool use_simd = (num_reduce_dims == 1) && (reduce_dims[0] == num_dims - 1) &&
                        (strides[num_dims - 1] == 1) && (reduce_size >= SIMD_F32_WIDTH);

        if (use_simd) {
            size_t flat_base = offset;
            for (size_t i = 0; i < num_dims - 1; i++) {
                flat_base += input_indices[i] * strides[i];
            }

            simd_f32_t vmax = simd_f32_set1(-INFINITY);
            const size_t simd_end = (reduce_size / SIMD_F32_WIDTH) * SIMD_F32_WIDTH;

            for (size_t i = 0; i < simd_end; i += SIMD_F32_WIDTH) {
                simd_f32_t v = simd_f32_load(&input[flat_base + i]);
                vmax = simd_f32_max(vmax, v);
            }
            acc = simd_f32_reduce_max(vmax);

            for (size_t i = simd_end; i < reduce_size; i++) {
                f32_t val = input[flat_base + i];
                acc = MAX(acc, val);
            }
        } else
#endif
        {
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {
                size_t temp_reduced = reduced_idx;
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {
                    size_t dim = reduce_dims[i];
                    input_indices[dim] = temp_reduced % dims[dim];
                    temp_reduced /= dims[dim];
                }

                size_t flat_index = offset;
                for (size_t i = 0; i < num_dims; i++) {
                    flat_index += input_indices[i] * strides[i];
                }

                f32_t val = input[flat_index];
                acc = MAX(acc, val);
            }
        }

        output[output_idx] = acc;
    }
}

// SIMD-optimized max_f64
void hodu_cpu_max_f64(const void *input_ptr, void *output_ptr, const size_t *metadata) {
    const f64_t *input = (const f64_t *)input_ptr;
    f64_t *output = (f64_t *)output_ptr;

    const size_t num_dims = metadata[0];
    const size_t *dims = metadata + 1;
    const size_t *strides = metadata + 1 + num_dims;
    const size_t offset = metadata[1 + 2 * num_dims];
    const size_t output_shape_len = metadata[2 + 2 * num_dims];
    const size_t *output_shape = metadata + 3 + 2 * num_dims;
    const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];
    const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;
    const size_t keep_dim_val = metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];
    const bool keep_dim = (keep_dim_val != 0);
    const size_t reduce_size = metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];

    size_t num_els = 1;
    for (size_t i = 0; i < output_shape_len; i++) {
        num_els *= output_shape[i];
    }

    for (size_t output_idx = 0; output_idx < num_els; output_idx++) {
        f64_t acc = -INFINITY;

        size_t output_indices[16];
        size_t temp = output_idx;
        for (int d = (int)output_shape_len - 1; d >= 0; d--) {
            output_indices[d] = temp % output_shape[d];
            temp /= output_shape[d];
        }

        size_t input_indices[16];
        if (keep_dim) {
            for (size_t i = 0; i < num_dims; i++) {
                input_indices[i] = output_indices[i];
            }
        } else {
            size_t out_idx = 0;
            for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {
                bool is_reduced = false;
                for (size_t r = 0; r < num_reduce_dims; r++) {
                    if (reduce_dims[r] == in_dim) {
                        is_reduced = true;
                        break;
                    }
                }
                if (is_reduced) {
                    input_indices[in_dim] = 0;
                } else {
                    input_indices[in_dim] =
                        (out_idx < output_shape_len) ? output_indices[out_idx] : 0;
                    out_idx++;
                }
            }
        }

#if SIMD_F64_WIDTH > 1
        bool use_simd = (num_reduce_dims == 1) && (reduce_dims[0] == num_dims - 1) &&
                        (strides[num_dims - 1] == 1) && (reduce_size >= SIMD_F64_WIDTH);

        if (use_simd) {
            size_t flat_base = offset;
            for (size_t i = 0; i < num_dims - 1; i++) {
                flat_base += input_indices[i] * strides[i];
            }

            simd_f64_t vmax = simd_f64_set1(-INFINITY);
            const size_t simd_end = (reduce_size / SIMD_F64_WIDTH) * SIMD_F64_WIDTH;

            for (size_t i = 0; i < simd_end; i += SIMD_F64_WIDTH) {
                simd_f64_t v = simd_f64_load(&input[flat_base + i]);
                vmax = simd_f64_max(vmax, v);
            }
            acc = simd_f64_reduce_max(vmax);

            for (size_t i = simd_end; i < reduce_size; i++) {
                f64_t val = input[flat_base + i];
                acc = MAX(acc, val);
            }
        } else
#endif
        {
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {
                size_t temp_reduced = reduced_idx;
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {
                    size_t dim = reduce_dims[i];
                    input_indices[dim] = temp_reduced % dims[dim];
                    temp_reduced /= dims[dim];
                }

                size_t flat_index = offset;
                for (size_t i = 0; i < num_dims; i++) {
                    flat_index += input_indices[i] * strides[i];
                }

                f64_t val = input[flat_index];
                acc = MAX(acc, val);
            }
        }

        output[output_idx] = acc;
    }
}
REDUCE_OP(int8_t, int8_t, max_i8, INT8_MIN, acc = MAX(acc, val))
REDUCE_OP(int16_t, int16_t, max_i16, INT16_MIN, acc = MAX(acc, val))
REDUCE_OP(int32_t, int32_t, max_i32, INT32_MIN, acc = MAX(acc, val))
REDUCE_OP(int64_t, int64_t, max_i64, INT64_MIN, acc = MAX(acc, val))
REDUCE_OP(uint8_t, uint8_t, max_u8, 0u, acc = MAX(acc, val))
REDUCE_OP(uint16_t, uint16_t, max_u16, 0u, acc = MAX(acc, val))
REDUCE_OP(uint32_t, uint32_t, max_u32, 0u, acc = MAX(acc, val))
REDUCE_OP(uint64_t, uint64_t, max_u64, 0u, acc = MAX(acc, val))

// Min reduction operations
REDUCE_OP(f8e4m3_t, f8e4m3_t, min_f8e4m3, F8E4M3_POS_INF, acc = f8e4m3_min(acc, val))
REDUCE_OP(f8e5m2_t, f8e5m2_t, min_f8e5m2, F8E5M2_POS_INF, acc = f8e5m2_min(acc, val))
REDUCE_OP(bf16_t, bf16_t, min_bf16, BF16_POS_INF, acc = bf16_min(acc, val))
REDUCE_OP(f16_t, f16_t, min_f16, F16_POS_INF, acc = f16_min(acc, val))

// SIMD-optimized min_f32
void hodu_cpu_min_f32(const void *input_ptr, void *output_ptr, const size_t *metadata) {
    const f32_t *input = (const f32_t *)input_ptr;
    f32_t *output = (f32_t *)output_ptr;

    const size_t num_dims = metadata[0];
    const size_t *dims = metadata + 1;
    const size_t *strides = metadata + 1 + num_dims;
    const size_t offset = metadata[1 + 2 * num_dims];
    const size_t output_shape_len = metadata[2 + 2 * num_dims];
    const size_t *output_shape = metadata + 3 + 2 * num_dims;
    const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];
    const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;
    const size_t keep_dim_val = metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];
    const bool keep_dim = (keep_dim_val != 0);
    const size_t reduce_size = metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];

    size_t num_els = 1;
    for (size_t i = 0; i < output_shape_len; i++) {
        num_els *= output_shape[i];
    }

    for (size_t output_idx = 0; output_idx < num_els; output_idx++) {
        f32_t acc = INFINITY;

        size_t output_indices[16];
        size_t temp = output_idx;
        for (int d = (int)output_shape_len - 1; d >= 0; d--) {
            output_indices[d] = temp % output_shape[d];
            temp /= output_shape[d];
        }

        size_t input_indices[16];
        if (keep_dim) {
            for (size_t i = 0; i < num_dims; i++) {
                input_indices[i] = output_indices[i];
            }
        } else {
            size_t out_idx = 0;
            for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {
                bool is_reduced = false;
                for (size_t r = 0; r < num_reduce_dims; r++) {
                    if (reduce_dims[r] == in_dim) {
                        is_reduced = true;
                        break;
                    }
                }
                if (is_reduced) {
                    input_indices[in_dim] = 0;
                } else {
                    input_indices[in_dim] =
                        (out_idx < output_shape_len) ? output_indices[out_idx] : 0;
                    out_idx++;
                }
            }
        }

#if SIMD_F32_WIDTH > 1
        bool use_simd = (num_reduce_dims == 1) && (reduce_dims[0] == num_dims - 1) &&
                        (strides[num_dims - 1] == 1) && (reduce_size >= SIMD_F32_WIDTH);

        if (use_simd) {
            size_t flat_base = offset;
            for (size_t i = 0; i < num_dims - 1; i++) {
                flat_base += input_indices[i] * strides[i];
            }

            simd_f32_t vmin = simd_f32_set1(INFINITY);
            const size_t simd_end = (reduce_size / SIMD_F32_WIDTH) * SIMD_F32_WIDTH;

            for (size_t i = 0; i < simd_end; i += SIMD_F32_WIDTH) {
                simd_f32_t v = simd_f32_load(&input[flat_base + i]);
                vmin = simd_f32_min(vmin, v);
            }
            acc = simd_f32_reduce_min(vmin);

            for (size_t i = simd_end; i < reduce_size; i++) {
                f32_t val = input[flat_base + i];
                acc = MIN(acc, val);
            }
        } else
#endif
        {
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {
                size_t temp_reduced = reduced_idx;
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {
                    size_t dim = reduce_dims[i];
                    input_indices[dim] = temp_reduced % dims[dim];
                    temp_reduced /= dims[dim];
                }

                size_t flat_index = offset;
                for (size_t i = 0; i < num_dims; i++) {
                    flat_index += input_indices[i] * strides[i];
                }

                f32_t val = input[flat_index];
                acc = MIN(acc, val);
            }
        }

        output[output_idx] = acc;
    }
}

// SIMD-optimized min_f64
void hodu_cpu_min_f64(const void *input_ptr, void *output_ptr, const size_t *metadata) {
    const f64_t *input = (const f64_t *)input_ptr;
    f64_t *output = (f64_t *)output_ptr;

    const size_t num_dims = metadata[0];
    const size_t *dims = metadata + 1;
    const size_t *strides = metadata + 1 + num_dims;
    const size_t offset = metadata[1 + 2 * num_dims];
    const size_t output_shape_len = metadata[2 + 2 * num_dims];
    const size_t *output_shape = metadata + 3 + 2 * num_dims;
    const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];
    const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;
    const size_t keep_dim_val = metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];
    const bool keep_dim = (keep_dim_val != 0);
    const size_t reduce_size = metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];

    size_t num_els = 1;
    for (size_t i = 0; i < output_shape_len; i++) {
        num_els *= output_shape[i];
    }

    for (size_t output_idx = 0; output_idx < num_els; output_idx++) {
        f64_t acc = INFINITY;

        size_t output_indices[16];
        size_t temp = output_idx;
        for (int d = (int)output_shape_len - 1; d >= 0; d--) {
            output_indices[d] = temp % output_shape[d];
            temp /= output_shape[d];
        }

        size_t input_indices[16];
        if (keep_dim) {
            for (size_t i = 0; i < num_dims; i++) {
                input_indices[i] = output_indices[i];
            }
        } else {
            size_t out_idx = 0;
            for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {
                bool is_reduced = false;
                for (size_t r = 0; r < num_reduce_dims; r++) {
                    if (reduce_dims[r] == in_dim) {
                        is_reduced = true;
                        break;
                    }
                }
                if (is_reduced) {
                    input_indices[in_dim] = 0;
                } else {
                    input_indices[in_dim] =
                        (out_idx < output_shape_len) ? output_indices[out_idx] : 0;
                    out_idx++;
                }
            }
        }

#if SIMD_F64_WIDTH > 1
        bool use_simd = (num_reduce_dims == 1) && (reduce_dims[0] == num_dims - 1) &&
                        (strides[num_dims - 1] == 1) && (reduce_size >= SIMD_F64_WIDTH);

        if (use_simd) {
            size_t flat_base = offset;
            for (size_t i = 0; i < num_dims - 1; i++) {
                flat_base += input_indices[i] * strides[i];
            }

            simd_f64_t vmin = simd_f64_set1(INFINITY);
            const size_t simd_end = (reduce_size / SIMD_F64_WIDTH) * SIMD_F64_WIDTH;

            for (size_t i = 0; i < simd_end; i += SIMD_F64_WIDTH) {
                simd_f64_t v = simd_f64_load(&input[flat_base + i]);
                vmin = simd_f64_min(vmin, v);
            }
            acc = simd_f64_reduce_min(vmin);

            for (size_t i = simd_end; i < reduce_size; i++) {
                f64_t val = input[flat_base + i];
                acc = MIN(acc, val);
            }
        } else
#endif
        {
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {
                size_t temp_reduced = reduced_idx;
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {
                    size_t dim = reduce_dims[i];
                    input_indices[dim] = temp_reduced % dims[dim];
                    temp_reduced /= dims[dim];
                }

                size_t flat_index = offset;
                for (size_t i = 0; i < num_dims; i++) {
                    flat_index += input_indices[i] * strides[i];
                }

                f64_t val = input[flat_index];
                acc = MIN(acc, val);
            }
        }

        output[output_idx] = acc;
    }
}
REDUCE_OP(int8_t, int8_t, min_i8, INT8_MAX, acc = MIN(acc, val))
REDUCE_OP(int16_t, int16_t, min_i16, INT16_MAX, acc = MIN(acc, val))
REDUCE_OP(int32_t, int32_t, min_i32, INT32_MAX, acc = MIN(acc, val))
REDUCE_OP(int64_t, int64_t, min_i64, INT64_MAX, acc = MIN(acc, val))
REDUCE_OP(uint8_t, uint8_t, min_u8, UINT8_MAX, acc = MIN(acc, val))
REDUCE_OP(uint16_t, uint16_t, min_u16, UINT16_MAX, acc = MIN(acc, val))
REDUCE_OP(uint32_t, uint32_t, min_u32, UINT32_MAX, acc = MIN(acc, val))
REDUCE_OP(uint64_t, uint64_t, min_u64, UINT64_MAX, acc = MIN(acc, val))

// Product reduction operations
REDUCE_OP(f8e4m3_t, f8e4m3_t, prod_f8e4m3, F8E4M3_ONE, acc = f8e4m3_mul(acc, val))
REDUCE_OP(f8e5m2_t, f8e5m2_t, prod_f8e5m2, F8E5M2_ONE, acc = f8e5m2_mul(acc, val))
REDUCE_OP(bf16_t, bf16_t, prod_bf16, BF16_ONE, acc = bf16_mul(acc, val))
REDUCE_OP(f16_t, f16_t, prod_f16, F16_ONE, acc = f16_mul(acc, val))
REDUCE_OP(f32_t, f32_t, prod_f32, 1.0f, acc *= val)
REDUCE_OP(f64_t, f64_t, prod_f64, 1.0, acc *= val)
REDUCE_OP(int8_t, int8_t, prod_i8, 1, acc *= val)
REDUCE_OP(int16_t, int16_t, prod_i16, 1, acc *= val)
REDUCE_OP(int32_t, int32_t, prod_i32, 1, acc *= val)
REDUCE_OP(int64_t, int64_t, prod_i64, 1, acc *= val)
REDUCE_OP(uint8_t, uint8_t, prod_u8, 1u, acc *= val)
REDUCE_OP(uint16_t, uint16_t, prod_u16, 1u, acc *= val)
REDUCE_OP(uint32_t, uint32_t, prod_u32, 1u, acc *= val)
REDUCE_OP(uint64_t, uint64_t, prod_u64, 1u, acc *= val)

// ============================================================================
// STANDARD DEVIATION REDUCTION
// ============================================================================
//
// Computes population standard deviation: sqrt(E[X²] - E[X]²)
// Two-pass algorithm: accumulate sum and sum of squares, then compute std.
//
// Metadata layout: Same as generic reduction operations

/// Macro to implement standard deviation reduction (float types only)
///
/// @param TYPE C float type
/// @param TYPE_SUFFIX Suffix for function naming
#define REDUCE_STD_OP(TYPE, TYPE_SUFFIX)                                                           \
    void hodu_cpu_std_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                       \
                                    const size_t *metadata) {                                      \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            TYPE sum = 0;                                                                          \
            TYPE sum_squares = 0;                                                                  \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                TYPE val = input[flat_index];                                                      \
                sum += val;                                                                        \
                sum_squares += val * val;                                                          \
            }                                                                                      \
                                                                                                   \
            TYPE mean = sum / (TYPE)reduce_size;                                                   \
            TYPE variance = (sum_squares / (TYPE)reduce_size) - (mean * mean);                     \
            output[output_idx] = sqrt(variance);                                                   \
        }                                                                                          \
    }

/// Macro to implement standard deviation reduction for exotic types
///
/// @param TYPE C float type
/// @param TYPE_SUFFIX Suffix for function naming
/// @param ZERO Zero constant for the type
/// @param ADD_FN Addition function
/// @param MUL_FN Multiplication function
/// @param DIV_FN Division function
/// @param SUB_FN Subtraction function
/// @param SQRT_FN Square root function
#define REDUCE_STD_OP_EXOTIC(TYPE, TYPE_SUFFIX, ZERO, ADD_FN, MUL_FN, DIV_FN, SUB_FN, SQRT_FN)     \
    void hodu_cpu_std_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                       \
                                    const size_t *metadata) {                                      \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            TYPE sum = ZERO;                                                                       \
            TYPE sum_squares = ZERO;                                                               \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                TYPE val = input[flat_index];                                                      \
                sum = ADD_FN(sum, val);                                                            \
                sum_squares = ADD_FN(sum_squares, MUL_FN(val, val));                               \
            }                                                                                      \
                                                                                                   \
            float reduce_size_f = (float)reduce_size;                                              \
            TYPE reduce_size_typed = float_to_##TYPE_SUFFIX(reduce_size_f);                        \
            TYPE mean = DIV_FN(sum, reduce_size_typed);                                            \
            TYPE variance = SUB_FN(DIV_FN(sum_squares, reduce_size_typed), MUL_FN(mean, mean));    \
            output[output_idx] = SQRT_FN(variance);                                                \
        }                                                                                          \
    }

REDUCE_STD_OP_EXOTIC(f8e4m3_t, f8e4m3, F8E4M3_ZERO, f8e4m3_add, f8e4m3_mul, f8e4m3_div, f8e4m3_sub,
                     f8e4m3_sqrt)
REDUCE_STD_OP_EXOTIC(f8e5m2_t, f8e5m2, F8E5M2_ZERO, f8e5m2_add, f8e5m2_mul, f8e5m2_div, f8e5m2_sub,
                     f8e5m2_sqrt)
REDUCE_STD_OP_EXOTIC(bf16_t, bf16, BF16_ZERO, bf16_add, bf16_mul, bf16_div, bf16_sub, bf16_sqrt)
REDUCE_STD_OP_EXOTIC(f16_t, f16, F16_ZERO, f16_add, f16_mul, f16_div, f16_sub, f16_sqrt)
REDUCE_STD_OP(f32_t, f32)
REDUCE_STD_OP(f64_t, f64)

// ============================================================================
// VARIANCE REDUCTION
// ============================================================================
//
// Computes population variance: E[X²] - E[X]²
// Same as std but without the sqrt.
//
// Metadata layout: Same as generic reduction operations

/// Macro to implement variance reduction (float types only)
///
/// @param TYPE C float type
/// @param TYPE_SUFFIX Suffix for function naming
#define REDUCE_VAR_OP(TYPE, TYPE_SUFFIX)                                                           \
    void hodu_cpu_var_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                       \
                                    const size_t *metadata) {                                      \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            TYPE sum = 0;                                                                          \
            TYPE sum_squares = 0;                                                                  \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                TYPE val = input[flat_index];                                                      \
                sum += val;                                                                        \
                sum_squares += val * val;                                                          \
            }                                                                                      \
                                                                                                   \
            TYPE mean = sum / (TYPE)reduce_size;                                                   \
            output[output_idx] = (sum_squares / (TYPE)reduce_size) - (mean * mean);                \
        }                                                                                          \
    }

/// Macro to implement variance reduction for exotic types
///
/// @param TYPE C float type
/// @param TYPE_SUFFIX Suffix for function naming
/// @param ZERO Zero constant for the type
/// @param ADD_FN Addition function
/// @param MUL_FN Multiplication function
/// @param DIV_FN Division function
/// @param SUB_FN Subtraction function
#define REDUCE_VAR_OP_EXOTIC(TYPE, TYPE_SUFFIX, ZERO, ADD_FN, MUL_FN, DIV_FN, SUB_FN)              \
    void hodu_cpu_var_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                       \
                                    const size_t *metadata) {                                      \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            TYPE sum = ZERO;                                                                       \
            TYPE sum_squares = ZERO;                                                               \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                TYPE val = input[flat_index];                                                      \
                sum = ADD_FN(sum, val);                                                            \
                sum_squares = ADD_FN(sum_squares, MUL_FN(val, val));                               \
            }                                                                                      \
                                                                                                   \
            float reduce_size_f = (float)reduce_size;                                              \
            TYPE reduce_size_typed = float_to_##TYPE_SUFFIX(reduce_size_f);                        \
            TYPE mean = DIV_FN(sum, reduce_size_typed);                                            \
            output[output_idx] =                                                                   \
                SUB_FN(DIV_FN(sum_squares, reduce_size_typed), MUL_FN(mean, mean));                \
        }                                                                                          \
    }

REDUCE_VAR_OP_EXOTIC(f8e4m3_t, f8e4m3, F8E4M3_ZERO, f8e4m3_add, f8e4m3_mul, f8e4m3_div, f8e4m3_sub)
REDUCE_VAR_OP_EXOTIC(f8e5m2_t, f8e5m2, F8E5M2_ZERO, f8e5m2_add, f8e5m2_mul, f8e5m2_div, f8e5m2_sub)
REDUCE_VAR_OP_EXOTIC(bf16_t, bf16, BF16_ZERO, bf16_add, bf16_mul, bf16_div, bf16_sub)
REDUCE_VAR_OP_EXOTIC(f16_t, f16, F16_ZERO, f16_add, f16_mul, f16_div, f16_sub)

// SIMD-optimized var_f32 (variance: E[X^2] - E[X]^2)
void hodu_cpu_var_f32(const void *input_ptr, void *output_ptr, const size_t *metadata) {
    const f32_t *input = (const f32_t *)input_ptr;
    f32_t *output = (f32_t *)output_ptr;

    const size_t num_dims = metadata[0];
    const size_t *dims = metadata + 1;
    const size_t *strides = metadata + 1 + num_dims;
    const size_t offset = metadata[1 + 2 * num_dims];
    const size_t output_shape_len = metadata[2 + 2 * num_dims];
    const size_t *output_shape = metadata + 3 + 2 * num_dims;
    const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];
    const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;
    const size_t keep_dim_val = metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];
    const bool keep_dim = (keep_dim_val != 0);
    const size_t reduce_size = metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];

    size_t num_els = 1;
    for (size_t i = 0; i < output_shape_len; i++) {
        num_els *= output_shape[i];
    }

    for (size_t output_idx = 0; output_idx < num_els; output_idx++) {
        f32_t sum = 0.0f;
        f32_t sum_squares = 0.0f;

        size_t output_indices[16];
        size_t temp = output_idx;
        for (int d = (int)output_shape_len - 1; d >= 0; d--) {
            output_indices[d] = temp % output_shape[d];
            temp /= output_shape[d];
        }

        size_t input_indices[16];
        if (keep_dim) {
            for (size_t i = 0; i < num_dims; i++) {
                input_indices[i] = output_indices[i];
            }
        } else {
            size_t out_idx = 0;
            for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {
                bool is_reduced = false;
                for (size_t r = 0; r < num_reduce_dims; r++) {
                    if (reduce_dims[r] == in_dim) {
                        is_reduced = true;
                        break;
                    }
                }
                if (is_reduced) {
                    input_indices[in_dim] = 0;
                } else {
                    input_indices[in_dim] =
                        (out_idx < output_shape_len) ? output_indices[out_idx] : 0;
                    out_idx++;
                }
            }
        }

#if SIMD_F32_WIDTH > 1
        bool use_simd = (num_reduce_dims == 1) && (reduce_dims[0] == num_dims - 1) &&
                        (strides[num_dims - 1] == 1) && (reduce_size >= SIMD_F32_WIDTH);

        if (use_simd) {
            size_t flat_base = offset;
            for (size_t i = 0; i < num_dims - 1; i++) {
                flat_base += input_indices[i] * strides[i];
            }

            simd_f32_t vsum = simd_f32_set1(0.0f);
            simd_f32_t vsum_sq = simd_f32_set1(0.0f);
            const size_t simd_end = (reduce_size / SIMD_F32_WIDTH) * SIMD_F32_WIDTH;

            for (size_t i = 0; i < simd_end; i += SIMD_F32_WIDTH) {
                simd_f32_t v = simd_f32_load(&input[flat_base + i]);
                vsum = simd_f32_add(vsum, v);
                vsum_sq = simd_f32_fmadd(v, v, vsum_sq); // v*v + vsum_sq
            }
            sum = simd_f32_reduce_add(vsum);
            sum_squares = simd_f32_reduce_add(vsum_sq);

            for (size_t i = simd_end; i < reduce_size; i++) {
                f32_t val = input[flat_base + i];
                sum += val;
                sum_squares += val * val;
            }
        } else
#endif
        {
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {
                size_t temp_reduced = reduced_idx;
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {
                    size_t dim = reduce_dims[i];
                    input_indices[dim] = temp_reduced % dims[dim];
                    temp_reduced /= dims[dim];
                }

                size_t flat_index = offset;
                for (size_t i = 0; i < num_dims; i++) {
                    flat_index += input_indices[i] * strides[i];
                }

                f32_t val = input[flat_index];
                sum += val;
                sum_squares += val * val;
            }
        }

        f32_t mean = sum / (f32_t)reduce_size;
        output[output_idx] = (sum_squares / (f32_t)reduce_size) - (mean * mean);
    }
}

// SIMD-optimized var_f64 (variance: E[X^2] - E[X]^2)
void hodu_cpu_var_f64(const void *input_ptr, void *output_ptr, const size_t *metadata) {
    const f64_t *input = (const f64_t *)input_ptr;
    f64_t *output = (f64_t *)output_ptr;

    const size_t num_dims = metadata[0];
    const size_t *dims = metadata + 1;
    const size_t *strides = metadata + 1 + num_dims;
    const size_t offset = metadata[1 + 2 * num_dims];
    const size_t output_shape_len = metadata[2 + 2 * num_dims];
    const size_t *output_shape = metadata + 3 + 2 * num_dims;
    const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];
    const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;
    const size_t keep_dim_val = metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];
    const bool keep_dim = (keep_dim_val != 0);
    const size_t reduce_size = metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];

    size_t num_els = 1;
    for (size_t i = 0; i < output_shape_len; i++) {
        num_els *= output_shape[i];
    }

    for (size_t output_idx = 0; output_idx < num_els; output_idx++) {
        f64_t sum = 0.0;
        f64_t sum_squares = 0.0;

        size_t output_indices[16];
        size_t temp = output_idx;
        for (int d = (int)output_shape_len - 1; d >= 0; d--) {
            output_indices[d] = temp % output_shape[d];
            temp /= output_shape[d];
        }

        size_t input_indices[16];
        if (keep_dim) {
            for (size_t i = 0; i < num_dims; i++) {
                input_indices[i] = output_indices[i];
            }
        } else {
            size_t out_idx = 0;
            for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {
                bool is_reduced = false;
                for (size_t r = 0; r < num_reduce_dims; r++) {
                    if (reduce_dims[r] == in_dim) {
                        is_reduced = true;
                        break;
                    }
                }
                if (is_reduced) {
                    input_indices[in_dim] = 0;
                } else {
                    input_indices[in_dim] =
                        (out_idx < output_shape_len) ? output_indices[out_idx] : 0;
                    out_idx++;
                }
            }
        }

#if SIMD_F64_WIDTH > 1
        bool use_simd = (num_reduce_dims == 1) && (reduce_dims[0] == num_dims - 1) &&
                        (strides[num_dims - 1] == 1) && (reduce_size >= SIMD_F64_WIDTH);

        if (use_simd) {
            size_t flat_base = offset;
            for (size_t i = 0; i < num_dims - 1; i++) {
                flat_base += input_indices[i] * strides[i];
            }

            simd_f64_t vsum = simd_f64_set1(0.0);
            simd_f64_t vsum_sq = simd_f64_set1(0.0);
            const size_t simd_end = (reduce_size / SIMD_F64_WIDTH) * SIMD_F64_WIDTH;

            for (size_t i = 0; i < simd_end; i += SIMD_F64_WIDTH) {
                simd_f64_t v = simd_f64_load(&input[flat_base + i]);
                vsum = simd_f64_add(vsum, v);
                vsum_sq = simd_f64_fmadd(v, v, vsum_sq); // v*v + vsum_sq
            }
            sum = simd_f64_reduce_add(vsum);
            sum_squares = simd_f64_reduce_add(vsum_sq);

            for (size_t i = simd_end; i < reduce_size; i++) {
                f64_t val = input[flat_base + i];
                sum += val;
                sum_squares += val * val;
            }
        } else
#endif
        {
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {
                size_t temp_reduced = reduced_idx;
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {
                    size_t dim = reduce_dims[i];
                    input_indices[dim] = temp_reduced % dims[dim];
                    temp_reduced /= dims[dim];
                }

                size_t flat_index = offset;
                for (size_t i = 0; i < num_dims; i++) {
                    flat_index += input_indices[i] * strides[i];
                }

                f64_t val = input[flat_index];
                sum += val;
                sum_squares += val * val;
            }
        }

        f64_t mean = sum / (f64_t)reduce_size;
        output[output_idx] = (sum_squares / (f64_t)reduce_size) - (mean * mean);
    }
}

// ============================================================================
// MEAN REDUCTION
// ============================================================================
//
// Computes arithmetic mean by calling sum and dividing by reduce_size.
//
// Metadata layout: Same as generic reduction operations

/// Macro to implement mean reduction (float types only)
///
/// @param TYPE C float type
/// @param TYPE_SUFFIX Suffix for function naming
#define REDUCE_MEAN_OP(TYPE, TYPE_SUFFIX)                                                          \
    void hodu_cpu_mean_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                      \
                                     const size_t *metadata) {                                     \
        const size_t num_dims = metadata[0];                                                       \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
        hodu_cpu_sum_##TYPE_SUFFIX(input_ptr, output_ptr, metadata);                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
        for (size_t i = 0; i < num_els; i++) {                                                     \
            output[i] /= (TYPE)reduce_size;                                                        \
        }                                                                                          \
    }

/// Macro to implement mean reduction for exotic types
///
/// @param TYPE C float type
/// @param TYPE_SUFFIX Suffix for function naming
/// @param DIV_FN Division function
#define REDUCE_MEAN_OP_EXOTIC(TYPE, TYPE_SUFFIX, DIV_FN)                                           \
    void hodu_cpu_mean_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                      \
                                     const size_t *metadata) {                                     \
        const size_t num_dims = metadata[0];                                                       \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
        hodu_cpu_sum_##TYPE_SUFFIX(input_ptr, output_ptr, metadata);                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
        float reduce_size_f = (float)reduce_size;                                                  \
        TYPE reduce_size_typed = float_to_##TYPE_SUFFIX(reduce_size_f);                            \
        for (size_t i = 0; i < num_els; i++) {                                                     \
            output[i] = DIV_FN(output[i], reduce_size_typed);                                      \
        }                                                                                          \
    }

REDUCE_MEAN_OP_EXOTIC(f8e4m3_t, f8e4m3, f8e4m3_div)
REDUCE_MEAN_OP_EXOTIC(f8e5m2_t, f8e5m2, f8e5m2_div)
REDUCE_MEAN_OP_EXOTIC(bf16_t, bf16, bf16_div)
REDUCE_MEAN_OP_EXOTIC(f16_t, f16, f16_div)
REDUCE_MEAN_OP(f32_t, f32)
REDUCE_MEAN_OP(f64_t, f64)

// ============================================================================
// NORM REDUCTION (L2 NORM)
// ============================================================================
//
// Computes L2 norm: sqrt(sum(X²))
//
// Metadata layout: Same as generic reduction operations

/// Macro to implement L2 norm reduction (float types only)
///
/// @param TYPE C float type
/// @param TYPE_SUFFIX Suffix for function naming
#define REDUCE_NORM_OP(TYPE, TYPE_SUFFIX)                                                          \
    void hodu_cpu_norm_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                      \
                                     const size_t *metadata) {                                     \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            TYPE sum_squares = 0;                                                                  \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                TYPE val = input[flat_index];                                                      \
                sum_squares += val * val;                                                          \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = (TYPE)sqrt((float)sum_squares);                                   \
        }                                                                                          \
    }

/// Macro to implement L2 norm reduction for exotic types
///
/// @param TYPE C float type
/// @param TYPE_SUFFIX Suffix for function naming
/// @param ZERO Zero constant for the type
/// @param ADD_FN Addition function
/// @param MUL_FN Multiplication function
/// @param SQRT_FN Square root function
#define REDUCE_NORM_OP_EXOTIC(TYPE, TYPE_SUFFIX, ZERO, ADD_FN, MUL_FN, SQRT_FN)                    \
    void hodu_cpu_norm_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                      \
                                     const size_t *metadata) {                                     \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            TYPE sum_squares = ZERO;                                                               \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                TYPE val = input[flat_index];                                                      \
                sum_squares = ADD_FN(sum_squares, MUL_FN(val, val));                               \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = SQRT_FN(sum_squares);                                             \
        }                                                                                          \
    }

REDUCE_NORM_OP_EXOTIC(f8e4m3_t, f8e4m3, F8E4M3_ZERO, f8e4m3_add, f8e4m3_mul, f8e4m3_sqrt)
REDUCE_NORM_OP_EXOTIC(f8e5m2_t, f8e5m2, F8E5M2_ZERO, f8e5m2_add, f8e5m2_mul, f8e5m2_sqrt)
REDUCE_NORM_OP_EXOTIC(bf16_t, bf16, BF16_ZERO, bf16_add, bf16_mul, bf16_sqrt)
REDUCE_NORM_OP_EXOTIC(f16_t, f16, F16_ZERO, f16_add, f16_mul, f16_sqrt)

// SIMD-optimized norm_f32 (L2 norm: sqrt(sum(x^2)))
void hodu_cpu_norm_f32(const void *input_ptr, void *output_ptr, const size_t *metadata) {
    const f32_t *input = (const f32_t *)input_ptr;
    f32_t *output = (f32_t *)output_ptr;

    const size_t num_dims = metadata[0];
    const size_t *dims = metadata + 1;
    const size_t *strides = metadata + 1 + num_dims;
    const size_t offset = metadata[1 + 2 * num_dims];
    const size_t output_shape_len = metadata[2 + 2 * num_dims];
    const size_t *output_shape = metadata + 3 + 2 * num_dims;
    const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];
    const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;
    const size_t keep_dim_val = metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];
    const bool keep_dim = (keep_dim_val != 0);
    const size_t reduce_size = metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];

    size_t num_els = 1;
    for (size_t i = 0; i < output_shape_len; i++) {
        num_els *= output_shape[i];
    }

    for (size_t output_idx = 0; output_idx < num_els; output_idx++) {
        f32_t sum_squares = 0.0f;

        size_t output_indices[16];
        size_t temp = output_idx;
        for (int d = (int)output_shape_len - 1; d >= 0; d--) {
            output_indices[d] = temp % output_shape[d];
            temp /= output_shape[d];
        }

        size_t input_indices[16];
        if (keep_dim) {
            for (size_t i = 0; i < num_dims; i++) {
                input_indices[i] = output_indices[i];
            }
        } else {
            size_t out_idx = 0;
            for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {
                bool is_reduced = false;
                for (size_t r = 0; r < num_reduce_dims; r++) {
                    if (reduce_dims[r] == in_dim) {
                        is_reduced = true;
                        break;
                    }
                }
                if (is_reduced) {
                    input_indices[in_dim] = 0;
                } else {
                    input_indices[in_dim] =
                        (out_idx < output_shape_len) ? output_indices[out_idx] : 0;
                    out_idx++;
                }
            }
        }

#if SIMD_F32_WIDTH > 1
        bool use_simd = (num_reduce_dims == 1) && (reduce_dims[0] == num_dims - 1) &&
                        (strides[num_dims - 1] == 1) && (reduce_size >= SIMD_F32_WIDTH);

        if (use_simd) {
            size_t flat_base = offset;
            for (size_t i = 0; i < num_dims - 1; i++) {
                flat_base += input_indices[i] * strides[i];
            }

            simd_f32_t vsum_sq = simd_f32_set1(0.0f);
            const size_t simd_end = (reduce_size / SIMD_F32_WIDTH) * SIMD_F32_WIDTH;

            for (size_t i = 0; i < simd_end; i += SIMD_F32_WIDTH) {
                simd_f32_t v = simd_f32_load(&input[flat_base + i]);
                vsum_sq = simd_f32_fmadd(v, v, vsum_sq); // v*v + vsum_sq
            }
            sum_squares = simd_f32_reduce_add(vsum_sq);

            for (size_t i = simd_end; i < reduce_size; i++) {
                f32_t val = input[flat_base + i];
                sum_squares += val * val;
            }
        } else
#endif
        {
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {
                size_t temp_reduced = reduced_idx;
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {
                    size_t dim = reduce_dims[i];
                    input_indices[dim] = temp_reduced % dims[dim];
                    temp_reduced /= dims[dim];
                }

                size_t flat_index = offset;
                for (size_t i = 0; i < num_dims; i++) {
                    flat_index += input_indices[i] * strides[i];
                }

                f32_t val = input[flat_index];
                sum_squares += val * val;
            }
        }

        output[output_idx] = sqrtf(sum_squares);
    }
}

// SIMD-optimized norm_f64 (L2 norm: sqrt(sum(x^2)))
void hodu_cpu_norm_f64(const void *input_ptr, void *output_ptr, const size_t *metadata) {
    const f64_t *input = (const f64_t *)input_ptr;
    f64_t *output = (f64_t *)output_ptr;

    const size_t num_dims = metadata[0];
    const size_t *dims = metadata + 1;
    const size_t *strides = metadata + 1 + num_dims;
    const size_t offset = metadata[1 + 2 * num_dims];
    const size_t output_shape_len = metadata[2 + 2 * num_dims];
    const size_t *output_shape = metadata + 3 + 2 * num_dims;
    const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];
    const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;
    const size_t keep_dim_val = metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];
    const bool keep_dim = (keep_dim_val != 0);
    const size_t reduce_size = metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];

    size_t num_els = 1;
    for (size_t i = 0; i < output_shape_len; i++) {
        num_els *= output_shape[i];
    }

    for (size_t output_idx = 0; output_idx < num_els; output_idx++) {
        f64_t sum_squares = 0.0;

        size_t output_indices[16];
        size_t temp = output_idx;
        for (int d = (int)output_shape_len - 1; d >= 0; d--) {
            output_indices[d] = temp % output_shape[d];
            temp /= output_shape[d];
        }

        size_t input_indices[16];
        if (keep_dim) {
            for (size_t i = 0; i < num_dims; i++) {
                input_indices[i] = output_indices[i];
            }
        } else {
            size_t out_idx = 0;
            for (size_t in_dim = 0; in_dim < num_dims; in_dim++) {
                bool is_reduced = false;
                for (size_t r = 0; r < num_reduce_dims; r++) {
                    if (reduce_dims[r] == in_dim) {
                        is_reduced = true;
                        break;
                    }
                }
                if (is_reduced) {
                    input_indices[in_dim] = 0;
                } else {
                    input_indices[in_dim] =
                        (out_idx < output_shape_len) ? output_indices[out_idx] : 0;
                    out_idx++;
                }
            }
        }

#if SIMD_F64_WIDTH > 1
        bool use_simd = (num_reduce_dims == 1) && (reduce_dims[0] == num_dims - 1) &&
                        (strides[num_dims - 1] == 1) && (reduce_size >= SIMD_F64_WIDTH);

        if (use_simd) {
            size_t flat_base = offset;
            for (size_t i = 0; i < num_dims - 1; i++) {
                flat_base += input_indices[i] * strides[i];
            }

            simd_f64_t vsum_sq = simd_f64_set1(0.0);
            const size_t simd_end = (reduce_size / SIMD_F64_WIDTH) * SIMD_F64_WIDTH;

            for (size_t i = 0; i < simd_end; i += SIMD_F64_WIDTH) {
                simd_f64_t v = simd_f64_load(&input[flat_base + i]);
                vsum_sq = simd_f64_fmadd(v, v, vsum_sq); // v*v + vsum_sq
            }
            sum_squares = simd_f64_reduce_add(vsum_sq);

            for (size_t i = simd_end; i < reduce_size; i++) {
                f64_t val = input[flat_base + i];
                sum_squares += val * val;
            }
        } else
#endif
        {
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {
                size_t temp_reduced = reduced_idx;
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {
                    size_t dim = reduce_dims[i];
                    input_indices[dim] = temp_reduced % dims[dim];
                    temp_reduced /= dims[dim];
                }

                size_t flat_index = offset;
                for (size_t i = 0; i < num_dims; i++) {
                    flat_index += input_indices[i] * strides[i];
                }

                f64_t val = input[flat_index];
                sum_squares += val * val;
            }
        }

        output[output_idx] = sqrt(sum_squares);
    }
}

// ============================================================================
// LOGSUM REDUCTION
// ============================================================================
//
// Computes log(sum(X)) along specified dimensions.
// Numerically: first sum all elements, then take log.
//
// Metadata layout: Same as generic reduction operations

/// Macro to implement logsum reduction for exotic types (via float conversion)
///
/// @param TYPE C float type
/// @param TYPE_SUFFIX Suffix for function naming
/// @param TO_FLOAT_FN Conversion to float function
/// @param FROM_FLOAT_FN Conversion from float function
#define REDUCE_LOGSUM_OP_EXOTIC(TYPE, TYPE_SUFFIX, TO_FLOAT_FN, FROM_FLOAT_FN)                     \
    void hodu_cpu_logsum_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                    \
                                       const size_t *metadata) {                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
        hodu_cpu_sum_##TYPE_SUFFIX(input_ptr, output_ptr, metadata);                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
        for (size_t i = 0; i < num_els; i++) {                                                     \
            float val = TO_FLOAT_FN(output[i]);                                                    \
            output[i] = FROM_FLOAT_FN(logf(val));                                                  \
        }                                                                                          \
    }

/// Macro to implement logsum reduction for standard types
///
/// @param TYPE C float type
/// @param TYPE_SUFFIX Suffix for function naming
#define REDUCE_LOGSUM_OP(TYPE, TYPE_SUFFIX)                                                        \
    void hodu_cpu_logsum_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                    \
                                       const size_t *metadata) {                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
        hodu_cpu_sum_##TYPE_SUFFIX(input_ptr, output_ptr, metadata);                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
        for (size_t i = 0; i < num_els; i++) {                                                     \
            output[i] = log(output[i]);                                                            \
        }                                                                                          \
    }

REDUCE_LOGSUM_OP_EXOTIC(f8e4m3_t, f8e4m3, f8e4m3_to_float, float_to_f8e4m3)
REDUCE_LOGSUM_OP_EXOTIC(f8e5m2_t, f8e5m2, f8e5m2_to_float, float_to_f8e5m2)
REDUCE_LOGSUM_OP_EXOTIC(bf16_t, bf16, bf16_to_float, float_to_bf16)
REDUCE_LOGSUM_OP_EXOTIC(f16_t, f16, f16_to_float, float_to_f16)
REDUCE_LOGSUM_OP(f32_t, f32)
REDUCE_LOGSUM_OP(f64_t, f64)

// ============================================================================
// LOGSUMEXP REDUCTION
// ============================================================================
//
// Computes log(sum(exp(X))) along specified dimensions.
// Uses the numerically stable formula: max(X) + log(sum(exp(X - max(X))))
//
// Metadata layout: Same as generic reduction operations

/// Macro to implement logsumexp reduction for exotic types (via float conversion)
///
/// @param TYPE C float type
/// @param TYPE_SUFFIX Suffix for function naming
/// @param TO_FLOAT_FN Conversion to float function
/// @param FROM_FLOAT_FN Conversion from float function
#define REDUCE_LOGSUMEXP_OP_EXOTIC(TYPE, TYPE_SUFFIX, TO_FLOAT_FN, FROM_FLOAT_FN)                  \
    void hodu_cpu_logsumexp_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                 \
                                          const size_t *metadata) {                                \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
                                                                                                   \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            /* First pass: find max */                                                             \
            float max_val = -FLT_MAX;                                                              \
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
                float val = TO_FLOAT_FN(input[flat_index]);                                        \
                if (val > max_val) {                                                               \
                    max_val = val;                                                                 \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            /* Second pass: sum exp(x - max) */                                                    \
            float sum_exp = 0.0f;                                                                  \
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
                float val = TO_FLOAT_FN(input[flat_index]);                                        \
                sum_exp += expf(val - max_val);                                                    \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = FROM_FLOAT_FN(max_val + logf(sum_exp));                           \
        }                                                                                          \
    }

/// Macro to implement logsumexp reduction for standard types
///
/// @param TYPE C float type
/// @param TYPE_SUFFIX Suffix for function naming
/// @param NEG_INF Negative infinity value
#define REDUCE_LOGSUMEXP_OP(TYPE, TYPE_SUFFIX, NEG_INF)                                            \
    void hodu_cpu_logsumexp_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                 \
                                          const size_t *metadata) {                                \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
                                                                                                   \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            /* First pass: find max */                                                             \
            TYPE max_val = NEG_INF;                                                                \
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
                TYPE val = input[flat_index];                                                      \
                if (val > max_val) {                                                               \
                    max_val = val;                                                                 \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            /* Second pass: sum exp(x - max) */                                                    \
            TYPE sum_exp = 0;                                                                      \
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
                TYPE val = input[flat_index];                                                      \
                sum_exp += exp(val - max_val);                                                     \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = max_val + log(sum_exp);                                           \
        }                                                                                          \
    }

REDUCE_LOGSUMEXP_OP_EXOTIC(f8e4m3_t, f8e4m3, f8e4m3_to_float, float_to_f8e4m3)
REDUCE_LOGSUMEXP_OP_EXOTIC(f8e5m2_t, f8e5m2, f8e5m2_to_float, float_to_f8e5m2)
REDUCE_LOGSUMEXP_OP_EXOTIC(bf16_t, bf16, bf16_to_float, float_to_bf16)
REDUCE_LOGSUMEXP_OP_EXOTIC(f16_t, f16, f16_to_float, float_to_f16)
REDUCE_LOGSUMEXP_OP(f32_t, f32, -FLT_MAX)
REDUCE_LOGSUMEXP_OP(f64_t, f64, -DBL_MAX)

// ============================================================================
// ARGMAX REDUCTION
// ============================================================================
//
// Returns the index of the maximum value along the first reduce dimension.
// Output type is always int32.
//
// Metadata layout: Same as generic reduction operations

/// Macro to implement argmax reduction (returns int32 indices)
///
/// @param IN_TYPE Input C type
/// @param TYPE_SUFFIX Suffix for function naming
#define REDUCE_ARGMAX_OP(IN_TYPE, TYPE_SUFFIX)                                                     \
    void hodu_cpu_argmax_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                    \
                                       const size_t *metadata) {                                   \
        const IN_TYPE *input = (const IN_TYPE *)input_ptr;                                         \
        int32_t *output = (int32_t *)output_ptr;                                                   \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            IN_TYPE max_val = 0;                                                                   \
            int32_t max_idx = 0;                                                                   \
            bool first = true;                                                                     \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t actual_dim_idx = input_indices[reduce_dims[0]];                             \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                IN_TYPE val = input[flat_index];                                                   \
                if (first || val > max_val) {                                                      \
                    max_val = val;                                                                 \
                    max_idx = (int32_t)actual_dim_idx;                                             \
                    first = false;                                                                 \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = max_idx;                                                          \
        }                                                                                          \
    }

/// Macro to implement argmax reduction for exotic types (returns int32 indices)
///
/// @param IN_TYPE Input C type
/// @param TYPE_SUFFIX Suffix for function naming
/// @param GT_FN Greater-than comparison function
#define REDUCE_ARGMAX_OP_EXOTIC(IN_TYPE, TYPE_SUFFIX, GT_FN)                                       \
    void hodu_cpu_argmax_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                    \
                                       const size_t *metadata) {                                   \
        const IN_TYPE *input = (const IN_TYPE *)input_ptr;                                         \
        int32_t *output = (int32_t *)output_ptr;                                                   \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            IN_TYPE max_val;                                                                       \
            int32_t max_idx = 0;                                                                   \
            bool first = true;                                                                     \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t actual_dim_idx = input_indices[reduce_dims[0]];                             \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                IN_TYPE val = input[flat_index];                                                   \
                if (first || GT_FN(val, max_val)) {                                                \
                    max_val = val;                                                                 \
                    max_idx = (int32_t)actual_dim_idx;                                             \
                    first = false;                                                                 \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = max_idx;                                                          \
        }                                                                                          \
    }

// ============================================================================
// ARGMIN REDUCTION
// ============================================================================
//
// Returns the index of the minimum value along the first reduce dimension.
// Output type is always int32.
//
// Metadata layout: Same as generic reduction operations

/// Macro to implement argmin reduction (returns int32 indices)
///
/// @param IN_TYPE Input C type
/// @param TYPE_SUFFIX Suffix for function naming
#define REDUCE_ARGMIN_OP(IN_TYPE, TYPE_SUFFIX)                                                     \
    void hodu_cpu_argmin_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                    \
                                       const size_t *metadata) {                                   \
        const IN_TYPE *input = (const IN_TYPE *)input_ptr;                                         \
        int32_t *output = (int32_t *)output_ptr;                                                   \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            IN_TYPE min_val = 0;                                                                   \
            int32_t min_idx = 0;                                                                   \
            bool first = true;                                                                     \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t actual_dim_idx = input_indices[reduce_dims[0]];                             \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                IN_TYPE val = input[flat_index];                                                   \
                if (first || val < min_val) {                                                      \
                    min_val = val;                                                                 \
                    min_idx = (int32_t)actual_dim_idx;                                             \
                    first = false;                                                                 \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = min_idx;                                                          \
        }                                                                                          \
    }

/// Macro to implement argmin reduction for exotic types (returns int32 indices)
///
/// @param IN_TYPE Input C type
/// @param TYPE_SUFFIX Suffix for function naming
/// @param LT_FN Less-than comparison function
#define REDUCE_ARGMIN_OP_EXOTIC(IN_TYPE, TYPE_SUFFIX, LT_FN)                                       \
    void hodu_cpu_argmin_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                    \
                                       const size_t *metadata) {                                   \
        const IN_TYPE *input = (const IN_TYPE *)input_ptr;                                         \
        int32_t *output = (int32_t *)output_ptr;                                                   \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            IN_TYPE min_val;                                                                       \
            int32_t min_idx = 0;                                                                   \
            bool first = true;                                                                     \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t actual_dim_idx = input_indices[reduce_dims[0]];                             \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                IN_TYPE val = input[flat_index];                                                   \
                if (first || LT_FN(val, min_val)) {                                                \
                    min_val = val;                                                                 \
                    min_idx = (int32_t)actual_dim_idx;                                             \
                    first = false;                                                                 \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = min_idx;                                                          \
        }                                                                                          \
    }

REDUCE_ARGMAX_OP(bool, bool)
REDUCE_ARGMAX_OP_EXOTIC(f8e4m3_t, f8e4m3, f8e4m3_gt)
REDUCE_ARGMAX_OP_EXOTIC(f8e5m2_t, f8e5m2, f8e5m2_gt)
REDUCE_ARGMAX_OP_EXOTIC(bf16_t, bf16, bf16_gt)
REDUCE_ARGMAX_OP_EXOTIC(f16_t, f16, f16_gt)
REDUCE_ARGMAX_OP(f32_t, f32)
REDUCE_ARGMAX_OP(f64_t, f64)
REDUCE_ARGMAX_OP(int8_t, i8)
REDUCE_ARGMAX_OP(int16_t, i16)
REDUCE_ARGMAX_OP(int32_t, i32)
REDUCE_ARGMAX_OP(int64_t, i64)
REDUCE_ARGMAX_OP(uint8_t, u8)
REDUCE_ARGMAX_OP(uint16_t, u16)
REDUCE_ARGMAX_OP(uint32_t, u32)
REDUCE_ARGMAX_OP(uint64_t, u64)

REDUCE_ARGMIN_OP(bool, bool)
REDUCE_ARGMIN_OP_EXOTIC(f8e4m3_t, f8e4m3, f8e4m3_lt)
REDUCE_ARGMIN_OP_EXOTIC(f8e5m2_t, f8e5m2, f8e5m2_lt)
REDUCE_ARGMIN_OP_EXOTIC(bf16_t, bf16, bf16_lt)
REDUCE_ARGMIN_OP_EXOTIC(f16_t, f16, f16_lt)
REDUCE_ARGMIN_OP(f32_t, f32)
REDUCE_ARGMIN_OP(f64_t, f64)
REDUCE_ARGMIN_OP(int8_t, i8)
REDUCE_ARGMIN_OP(int16_t, i16)
REDUCE_ARGMIN_OP(int32_t, i32)
REDUCE_ARGMIN_OP(int64_t, i64)
REDUCE_ARGMIN_OP(uint8_t, u8)
REDUCE_ARGMIN_OP(uint16_t, u16)
REDUCE_ARGMIN_OP(uint32_t, u32)
REDUCE_ARGMIN_OP(uint64_t, u64)

// ============================================================================
// ANY REDUCTION (LOGICAL OR)
// ============================================================================
//
// Returns true if any element is non-zero along reduce dimensions.
// Output type is always bool. Short-circuits on first true value.
//
// Metadata layout: Same as generic reduction operations

#define IS_NONZERO(val) ((val) != 0)

/// Macro to implement any reduction (logical OR, returns bool)
///
/// @param IN_TYPE Input C type
/// @param TYPE_SUFFIX Suffix for function naming
#define REDUCE_ANY_OP(IN_TYPE, TYPE_SUFFIX)                                                        \
    void hodu_cpu_any_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                       \
                                    const size_t *metadata) {                                      \
        const IN_TYPE *input = (const IN_TYPE *)input_ptr;                                         \
        bool *output = (bool *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            bool result = false;                                                                   \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                if (IS_NONZERO(input[flat_index])) {                                               \
                    result = true;                                                                 \
                    break;                                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = result;                                                           \
        }                                                                                          \
    }

// ============================================================================
// ALL REDUCTION (LOGICAL AND)
// ============================================================================
//
// Returns true if all elements are non-zero along reduce dimensions.
// Output type is always bool. Short-circuits on first false value.
//
// Metadata layout: Same as generic reduction operations

/// Macro to implement all reduction (logical AND, returns bool)
///
/// @param IN_TYPE Input C type
/// @param TYPE_SUFFIX Suffix for function naming
#define REDUCE_ALL_OP(IN_TYPE, TYPE_SUFFIX)                                                        \
    void hodu_cpu_all_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                       \
                                    const size_t *metadata) {                                      \
        const IN_TYPE *input = (const IN_TYPE *)input_ptr;                                         \
        bool *output = (bool *)output_ptr;                                                         \
                                                                                                   \
        const size_t num_dims = metadata[0];                                                       \
        const size_t *dims = metadata + 1;                                                         \
        const size_t *strides = metadata + 1 + num_dims;                                           \
        const size_t offset = metadata[1 + 2 * num_dims];                                          \
        const size_t output_shape_len = metadata[2 + 2 * num_dims];                                \
        const size_t *output_shape = metadata + 3 + 2 * num_dims;                                  \
        const size_t num_reduce_dims = metadata[3 + 2 * num_dims + output_shape_len];              \
        const size_t *reduce_dims = metadata + 4 + 2 * num_dims + output_shape_len;                \
        const size_t keep_dim_val =                                                                \
            metadata[4 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        const bool keep_dim = (keep_dim_val != 0);                                                 \
        const size_t reduce_size =                                                                 \
            metadata[5 + 2 * num_dims + output_shape_len + num_reduce_dims];                       \
        size_t num_els = 1;                                                                        \
        for (size_t i = 0; i < output_shape_len; i++) {                                            \
            num_els *= output_shape[i];                                                            \
        }                                                                                          \
                                                                                                   \
        for (size_t output_idx = 0; output_idx < num_els; output_idx++) {                          \
            bool result = true;                                                                    \
                                                                                                   \
            size_t output_indices[16];                                                             \
            size_t temp = output_idx;                                                              \
            for (int d = (int)output_shape_len - 1; d >= 0; d--) {                                 \
                output_indices[d] = temp % output_shape[d];                                        \
                temp /= output_shape[d];                                                           \
            }                                                                                      \
                                                                                                   \
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
                                                                                                   \
            for (size_t reduced_idx = 0; reduced_idx < reduce_size; reduced_idx++) {               \
                size_t temp_reduced = reduced_idx;                                                 \
                for (int i = (int)num_reduce_dims - 1; i >= 0; i--) {                              \
                    size_t dim = reduce_dims[i];                                                   \
                    input_indices[dim] = temp_reduced % dims[dim];                                 \
                    temp_reduced /= dims[dim];                                                     \
                }                                                                                  \
                                                                                                   \
                size_t flat_index = offset;                                                        \
                for (size_t i = 0; i < num_dims; i++) {                                            \
                    flat_index += input_indices[i] * strides[i];                                   \
                }                                                                                  \
                                                                                                   \
                if (!IS_NONZERO(input[flat_index])) {                                              \
                    result = false;                                                                \
                    break;                                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            output[output_idx] = result;                                                           \
        }                                                                                          \
    }

REDUCE_ANY_OP(bool, bool)
REDUCE_ANY_OP(f8e4m3_t, f8e4m3)
REDUCE_ANY_OP(f8e5m2_t, f8e5m2)
REDUCE_ANY_OP(bf16_t, bf16)
REDUCE_ANY_OP(f16_t, f16)
REDUCE_ANY_OP(f32_t, f32)
REDUCE_ANY_OP(f64_t, f64)
REDUCE_ANY_OP(int8_t, i8)
REDUCE_ANY_OP(int16_t, i16)
REDUCE_ANY_OP(int32_t, i32)
REDUCE_ANY_OP(int64_t, i64)
REDUCE_ANY_OP(uint8_t, u8)
REDUCE_ANY_OP(uint16_t, u16)
REDUCE_ANY_OP(uint32_t, u32)
REDUCE_ANY_OP(uint64_t, u64)

REDUCE_ALL_OP(bool, bool)
REDUCE_ALL_OP(f8e4m3_t, f8e4m3)
REDUCE_ALL_OP(f8e5m2_t, f8e5m2)
REDUCE_ALL_OP(bf16_t, bf16)
REDUCE_ALL_OP(f16_t, f16)
REDUCE_ALL_OP(f32_t, f32)
REDUCE_ALL_OP(f64_t, f64)
REDUCE_ALL_OP(int8_t, i8)
REDUCE_ALL_OP(int16_t, i16)
REDUCE_ALL_OP(int32_t, i32)
REDUCE_ALL_OP(int64_t, i64)
REDUCE_ALL_OP(uint8_t, u8)
REDUCE_ALL_OP(uint16_t, u16)
REDUCE_ALL_OP(uint32_t, u32)
REDUCE_ALL_OP(uint64_t, u64)
