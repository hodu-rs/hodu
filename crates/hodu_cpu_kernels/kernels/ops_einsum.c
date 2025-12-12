#include "ops_einsum.h"
#include "types.h"

#define MAX_INPUTS 4
#define MAX_INDICES 16
#define MAX_DIMS 16

typedef struct {
    size_t num_output_els;
    size_t num_inputs;
    size_t num_total_indices;
    size_t num_contraction_indices;
    size_t output_ndim;
    size_t output_shape[MAX_DIMS];
    struct {
        size_t ndim;
        size_t shape[MAX_DIMS];
        size_t strides[MAX_DIMS];
        size_t offset;
        size_t dim_to_index[MAX_DIMS]; // dim d -> which index id
    } inputs[MAX_INPUTS];
    size_t contraction_index_ids[MAX_INDICES];
    size_t index_sizes[MAX_INDICES];
    size_t output_index_ids[MAX_DIMS];
} EinsumMeta;

static void parse_metadata(const size_t *metadata, EinsumMeta *meta) {
    size_t pos = 0;

    meta->num_output_els = metadata[pos++];
    meta->num_inputs = metadata[pos++];
    meta->num_total_indices = metadata[pos++];
    meta->num_contraction_indices = metadata[pos++];
    meta->output_ndim = metadata[pos++];

    for (size_t i = 0; i < meta->output_ndim; i++) {
        meta->output_shape[i] = metadata[pos++];
    }

    for (size_t inp = 0; inp < meta->num_inputs; inp++) {
        meta->inputs[inp].ndim = metadata[pos++];

        for (size_t d = 0; d < meta->inputs[inp].ndim; d++) {
            meta->inputs[inp].shape[d] = metadata[pos++];
        }

        for (size_t d = 0; d < meta->inputs[inp].ndim; d++) {
            meta->inputs[inp].strides[d] = metadata[pos++];
        }

        meta->inputs[inp].offset = metadata[pos++];

        // dim_to_index: each dim maps to which index id
        for (size_t d = 0; d < meta->inputs[inp].ndim; d++) {
            meta->inputs[inp].dim_to_index[d] = metadata[pos++];
        }
    }

    for (size_t i = 0; i < meta->num_contraction_indices; i++) {
        meta->contraction_index_ids[i] = metadata[pos++];
    }

    for (size_t i = 0; i < meta->num_total_indices; i++) {
        meta->index_sizes[i] = metadata[pos++];
    }

    for (size_t i = 0; i < meta->output_ndim; i++) {
        meta->output_index_ids[i] = metadata[pos++];
    }
}

#define IMPL_EINSUM(TYPE, TYPE_SUFFIX)                                                             \
    void hodu_cpu_einsum_##TYPE_SUFFIX(const void **inputs, void *output,                          \
                                       const size_t *metadata) {                                   \
        EinsumMeta meta;                                                                           \
        parse_metadata(metadata, &meta);                                                           \
                                                                                                   \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        size_t num_contraction_els = 1;                                                            \
        for (size_t i = 0; i < meta.num_contraction_indices; i++) {                                \
            num_contraction_els *= meta.index_sizes[meta.contraction_index_ids[i]];                \
        }                                                                                          \
                                                                                                   \
        for (size_t out_idx = 0; out_idx < meta.num_output_els; out_idx++) {                       \
            size_t index_values[MAX_INDICES] = {0};                                                \
                                                                                                   \
            size_t tmp = out_idx;                                                                  \
            for (int d = (int)meta.output_ndim - 1; d >= 0; d--) {                                 \
                size_t coord = tmp % meta.output_shape[d];                                         \
                tmp /= meta.output_shape[d];                                                       \
                index_values[meta.output_index_ids[d]] = coord;                                    \
            }                                                                                      \
                                                                                                   \
            TYPE sum = (TYPE)0;                                                                    \
                                                                                                   \
            for (size_t c_idx = 0; c_idx < num_contraction_els; c_idx++) {                         \
                size_t tmp_c = c_idx;                                                              \
                for (int i = (int)meta.num_contraction_indices - 1; i >= 0; i--) {                 \
                    size_t idx_id = meta.contraction_index_ids[i];                                 \
                    size_t sz = meta.index_sizes[idx_id];                                          \
                    index_values[idx_id] = tmp_c % sz;                                             \
                    tmp_c /= sz;                                                                   \
                }                                                                                  \
                                                                                                   \
                TYPE product = (TYPE)1;                                                            \
                for (size_t inp = 0; inp < meta.num_inputs; inp++) {                               \
                    const TYPE *in = (const TYPE *)inputs[inp] + meta.inputs[inp].offset;          \
                    size_t in_idx = 0;                                                             \
                    for (size_t d = 0; d < meta.inputs[inp].ndim; d++) {                           \
                        size_t idx_id = meta.inputs[inp].dim_to_index[d];                          \
                        in_idx += index_values[idx_id] * meta.inputs[inp].strides[d];              \
                    }                                                                              \
                    product *= in[in_idx];                                                         \
                }                                                                                  \
                sum += product;                                                                    \
            }                                                                                      \
                                                                                                   \
            out[out_idx] = sum;                                                                    \
        }                                                                                          \
    }

IMPL_EINSUM(f32_t, f32)
IMPL_EINSUM(f64_t, f64)
IMPL_EINSUM(u8_t, u8)
IMPL_EINSUM(u16_t, u16)
IMPL_EINSUM(u32_t, u32)
IMPL_EINSUM(u64_t, u64)
IMPL_EINSUM(i8_t, i8)
IMPL_EINSUM(i16_t, i16)
IMPL_EINSUM(i32_t, i32)
IMPL_EINSUM(i64_t, i64)

#define IMPL_EINSUM_CONVERT(TYPE, TYPE_SUFFIX, TO_FLOAT, FROM_FLOAT)                               \
    void hodu_cpu_einsum_##TYPE_SUFFIX(const void **inputs, void *output,                          \
                                       const size_t *metadata) {                                   \
        EinsumMeta meta;                                                                           \
        parse_metadata(metadata, &meta);                                                           \
                                                                                                   \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        size_t num_contraction_els = 1;                                                            \
        for (size_t i = 0; i < meta.num_contraction_indices; i++) {                                \
            num_contraction_els *= meta.index_sizes[meta.contraction_index_ids[i]];                \
        }                                                                                          \
                                                                                                   \
        for (size_t out_idx = 0; out_idx < meta.num_output_els; out_idx++) {                       \
            size_t index_values[MAX_INDICES] = {0};                                                \
                                                                                                   \
            size_t tmp = out_idx;                                                                  \
            for (int d = (int)meta.output_ndim - 1; d >= 0; d--) {                                 \
                size_t coord = tmp % meta.output_shape[d];                                         \
                tmp /= meta.output_shape[d];                                                       \
                index_values[meta.output_index_ids[d]] = coord;                                    \
            }                                                                                      \
                                                                                                   \
            float sum = 0.0f;                                                                      \
                                                                                                   \
            for (size_t c_idx = 0; c_idx < num_contraction_els; c_idx++) {                         \
                size_t tmp_c = c_idx;                                                              \
                for (int i = (int)meta.num_contraction_indices - 1; i >= 0; i--) {                 \
                    size_t idx_id = meta.contraction_index_ids[i];                                 \
                    size_t sz = meta.index_sizes[idx_id];                                          \
                    index_values[idx_id] = tmp_c % sz;                                             \
                    tmp_c /= sz;                                                                   \
                }                                                                                  \
                                                                                                   \
                float product = 1.0f;                                                              \
                for (size_t inp = 0; inp < meta.num_inputs; inp++) {                               \
                    const TYPE *in = (const TYPE *)inputs[inp] + meta.inputs[inp].offset;          \
                    size_t in_idx = 0;                                                             \
                    for (size_t d = 0; d < meta.inputs[inp].ndim; d++) {                           \
                        size_t idx_id = meta.inputs[inp].dim_to_index[d];                          \
                        in_idx += index_values[idx_id] * meta.inputs[inp].strides[d];              \
                    }                                                                              \
                    product *= TO_FLOAT(in[in_idx]);                                               \
                }                                                                                  \
                sum += product;                                                                    \
            }                                                                                      \
                                                                                                   \
            out[out_idx] = FROM_FLOAT(sum);                                                        \
        }                                                                                          \
    }

IMPL_EINSUM_CONVERT(bf16_t, bf16, bf16_to_float, float_to_bf16)
IMPL_EINSUM_CONVERT(f16_t, f16, f16_to_float, float_to_f16)
IMPL_EINSUM_CONVERT(f8e4m3_t, f8e4m3, f8e4m3_to_float, float_to_f8e4m3)
IMPL_EINSUM_CONVERT(f8e5m2_t, f8e5m2, f8e5m2_to_float, float_to_f8e5m2)
