#include <metal_stdlib>
using namespace metal;

#define MAX_INPUTS 4
#define MAX_INDICES 16
#define MAX_DIMS 16

struct InputInfo {
    size_t ndim;
    size_t shape[MAX_DIMS];
    size_t strides[MAX_DIMS];
    size_t offset;
    size_t dim_to_index[MAX_DIMS]; // dim d -> which index id
};

struct EinsumMeta {
    size_t num_output_els;
    size_t num_inputs;
    size_t num_total_indices;
    size_t num_contraction_indices;
    size_t output_ndim;
    size_t output_shape[MAX_DIMS];
    InputInfo inputs[MAX_INPUTS];
    size_t contraction_index_ids[MAX_INDICES];
    size_t index_sizes[MAX_INDICES];
    size_t output_index_ids[MAX_DIMS];
};

void parse_einsum_metadata(constant size_t *metadata, thread EinsumMeta &meta) {
    size_t pos = 0;

    meta.num_output_els = metadata[pos++];
    meta.num_inputs = metadata[pos++];
    meta.num_total_indices = metadata[pos++];
    meta.num_contraction_indices = metadata[pos++];
    meta.output_ndim = metadata[pos++];

    for (size_t i = 0; i < meta.output_ndim; i++) {
        meta.output_shape[i] = metadata[pos++];
    }

    for (size_t inp = 0; inp < meta.num_inputs; inp++) {
        meta.inputs[inp].ndim = metadata[pos++];

        for (size_t d = 0; d < meta.inputs[inp].ndim; d++) {
            meta.inputs[inp].shape[d] = metadata[pos++];
        }

        for (size_t d = 0; d < meta.inputs[inp].ndim; d++) {
            meta.inputs[inp].strides[d] = metadata[pos++];
        }

        meta.inputs[inp].offset = metadata[pos++];

        // dim_to_index: each dim maps to which index id
        for (size_t d = 0; d < meta.inputs[inp].ndim; d++) {
            meta.inputs[inp].dim_to_index[d] = metadata[pos++];
        }
    }

    for (size_t i = 0; i < meta.num_contraction_indices; i++) {
        meta.contraction_index_ids[i] = metadata[pos++];
    }

    for (size_t i = 0; i < meta.num_total_indices; i++) {
        meta.index_sizes[i] = metadata[pos++];
    }

    for (size_t i = 0; i < meta.output_ndim; i++) {
        meta.output_index_ids[i] = metadata[pos++];
    }
}

#define EINSUM_OP(TYPE, TYPE_SUFFIX)                                                               \
    kernel void hodu_metal_einsum_##TYPE_SUFFIX(                                                   \
        device const TYPE *input0 [[buffer(0)]], device const TYPE *input1 [[buffer(1)]],          \
        device const TYPE *input2 [[buffer(2)]], device const TYPE *input3 [[buffer(3)]],          \
        device TYPE *output [[buffer(4)]], constant size_t *metadata [[buffer(5)]],                \
        uint tid [[thread_position_in_grid]]) {                                                    \
                                                                                                   \
        EinsumMeta meta;                                                                           \
        parse_einsum_metadata(metadata, meta);                                                     \
                                                                                                   \
        if (tid >= meta.num_output_els)                                                            \
            return;                                                                                \
                                                                                                   \
        device const TYPE *inputs[MAX_INPUTS] = {input0, input1, input2, input3};                  \
                                                                                                   \
        size_t num_contraction_els = 1;                                                            \
        for (size_t i = 0; i < meta.num_contraction_indices; i++) {                                \
            num_contraction_els *= meta.index_sizes[meta.contraction_index_ids[i]];                \
        }                                                                                          \
                                                                                                   \
        size_t index_values[MAX_INDICES] = {0};                                                    \
                                                                                                   \
        size_t tmp = tid;                                                                          \
        for (int d = (int)meta.output_ndim - 1; d >= 0; d--) {                                     \
            size_t coord = tmp % meta.output_shape[d];                                             \
            tmp /= meta.output_shape[d];                                                           \
            index_values[meta.output_index_ids[d]] = coord;                                        \
        }                                                                                          \
                                                                                                   \
        TYPE sum = static_cast<TYPE>(0);                                                           \
                                                                                                   \
        for (size_t c_idx = 0; c_idx < num_contraction_els; c_idx++) {                             \
            size_t tmp_c = c_idx;                                                                  \
            for (int i = (int)meta.num_contraction_indices - 1; i >= 0; i--) {                     \
                size_t idx_id = meta.contraction_index_ids[i];                                     \
                size_t sz = meta.index_sizes[idx_id];                                              \
                index_values[idx_id] = tmp_c % sz;                                                 \
                tmp_c /= sz;                                                                       \
            }                                                                                      \
                                                                                                   \
            TYPE product = static_cast<TYPE>(1);                                                   \
            for (size_t inp = 0; inp < meta.num_inputs; inp++) {                                   \
                size_t in_idx = meta.inputs[inp].offset;                                           \
                for (size_t d = 0; d < meta.inputs[inp].ndim; d++) {                               \
                    size_t idx_id = meta.inputs[inp].dim_to_index[d];                              \
                    in_idx += index_values[idx_id] * meta.inputs[inp].strides[d];                  \
                }                                                                                  \
                product *= inputs[inp][in_idx];                                                    \
            }                                                                                      \
            sum += product;                                                                        \
        }                                                                                          \
                                                                                                   \
        output[tid] = sum;                                                                         \
    }

EINSUM_OP(bfloat, bf16)
EINSUM_OP(half, f16)
EINSUM_OP(float, f32)
EINSUM_OP(uint8_t, u8)
EINSUM_OP(uint16_t, u16)
EINSUM_OP(uint32_t, u32)
EINSUM_OP(uint64_t, u64)
EINSUM_OP(int8_t, i8)
EINSUM_OP(int16_t, i16)
EINSUM_OP(int32_t, i32)
EINSUM_OP(int64_t, i64)
