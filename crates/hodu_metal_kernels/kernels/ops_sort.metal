#include <metal_stdlib>
using namespace metal;

// TopK operation: returns the k largest or smallest elements along the last dimension
//
// Metadata layout:
// - metadata[0]: output_size (k * outer_size)
// - metadata[1]: k (number of top elements to return)
// - metadata[2]: last_dim_size (size of the dimension to search along)
// - metadata[3]: outer_size (product of all dimensions except last)
// - metadata[4]: largest (1 = largest, 0 = smallest)
// - metadata[5]: sorted (1 = sorted, 0 = unsorted)
// - metadata[6]: offset

#define TOPK_OP(TYPE, TYPE_SUFFIX)                                                                 \
    kernel void hodu_metal_topk_##TYPE_SUFFIX(                                                     \
        device const TYPE *input [[buffer(0)]], device TYPE *values [[buffer(1)]],                 \
        device int32_t *indices [[buffer(2)]], constant size_t *metadata [[buffer(3)]],            \
        uint tid [[thread_position_in_grid]]) {                                                    \
        const size_t k = metadata[1];                                                              \
        const size_t last_dim_size = metadata[2];                                                  \
        const size_t outer_size = metadata[3];                                                     \
        const int largest = (int)metadata[4];                                                      \
        const int sorted = (int)metadata[5];                                                       \
        const size_t offset = metadata[6];                                                         \
                                                                                                   \
        if (tid >= outer_size)                                                                     \
            return;                                                                                \
                                                                                                   \
        device const TYPE *row = input + offset + tid * last_dim_size;                             \
        device TYPE *val_out = values + tid * k;                                                   \
        device int32_t *idx_out = indices + tid * k;                                               \
                                                                                                   \
        float temp_vals[64];                                                                       \
        int32_t temp_idxs[64];                                                                     \
        size_t actual_k = min(k, (size_t)64);                                                      \
        size_t actual_dim = min(last_dim_size, (size_t)64);                                        \
                                                                                                   \
        /* Load all elements into temp array */                                                    \
        for (size_t i = 0; i < actual_dim; i++) {                                                  \
            temp_vals[i] = (float)row[i];                                                          \
            temp_idxs[i] = (int32_t)i;                                                             \
        }                                                                                          \
                                                                                                   \
        /* Simple selection sort to find top-k */                                                  \
        for (size_t i = 0; i < actual_k && i < actual_dim; i++) {                                  \
            size_t best_idx = i;                                                                   \
            for (size_t j = i + 1; j < actual_dim; j++) {                                          \
                bool is_better = largest ? (temp_vals[j] > temp_vals[best_idx])                    \
                                         : (temp_vals[j] < temp_vals[best_idx]);                   \
                if (is_better) {                                                                   \
                    best_idx = j;                                                                  \
                }                                                                                  \
            }                                                                                      \
            if (best_idx != i) {                                                                   \
                float tv = temp_vals[i];                                                           \
                temp_vals[i] = temp_vals[best_idx];                                                \
                temp_vals[best_idx] = tv;                                                          \
                int32_t ti = temp_idxs[i];                                                         \
                temp_idxs[i] = temp_idxs[best_idx];                                                \
                temp_idxs[best_idx] = ti;                                                          \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        /* Write output */                                                                         \
        for (size_t i = 0; i < actual_k; i++) {                                                    \
            val_out[i] = (TYPE)temp_vals[i];                                                       \
            idx_out[i] = temp_idxs[i];                                                             \
        }                                                                                          \
        (void)sorted;                                                                              \
    }

TOPK_OP(bfloat, bf16)
TOPK_OP(half, f16)
TOPK_OP(float, f32)
TOPK_OP(uint8_t, u8)
TOPK_OP(uint16_t, u16)
TOPK_OP(uint32_t, u32)
TOPK_OP(uint64_t, u64)
TOPK_OP(int8_t, i8)
TOPK_OP(int16_t, i16)
TOPK_OP(int32_t, i32)
TOPK_OP(int64_t, i64)
