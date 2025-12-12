#include "cuda_fp8.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

#define TOPK_OP(TYPE, TYPE_SUFFIX, TO_FLOAT)                                                       \
    extern "C" __global__ void hodu_cuda_topk_##TYPE_SUFFIX(                                       \
        const TYPE *input, TYPE *values, int32_t *indices, const size_t *metadata) {               \
        const size_t k = metadata[1];                                                              \
        const size_t last_dim_size = metadata[2];                                                  \
        const size_t outer_size = metadata[3];                                                     \
        const int largest = (int)metadata[4];                                                      \
        const int sorted = (int)metadata[5];                                                       \
        const size_t offset = metadata[6];                                                         \
                                                                                                   \
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;                                  \
        if (tid >= outer_size)                                                                     \
            return;                                                                                \
                                                                                                   \
        const TYPE *row = input + offset + tid * last_dim_size;                                    \
        TYPE *val_out = values + tid * k;                                                          \
        int32_t *idx_out = indices + tid * k;                                                      \
                                                                                                   \
        /* Local arrays for sorting - limited to 64 elements */                                    \
        float temp_vals[64];                                                                       \
        int32_t temp_idxs[64];                                                                     \
        size_t actual_k = k < 64 ? k : 64;                                                         \
        size_t actual_dim = last_dim_size < 64 ? last_dim_size : 64;                               \
                                                                                                   \
        /* Load all elements into temp array */                                                    \
        for (size_t i = 0; i < actual_dim; i++) {                                                  \
            temp_vals[i] = TO_FLOAT(row[i]);                                                       \
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

#define IDENTITY(x) (x)
#define FP8_E4M3_TO_FLOAT(x) __half2float(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3))
#define FP8_E5M2_TO_FLOAT(x) __half2float(__nv_cvt_fp8_to_halfraw(x, __NV_E5M2))

TOPK_OP(__nv_fp8_e4m3, f8e4m3, FP8_E4M3_TO_FLOAT)
TOPK_OP(__nv_fp8_e5m2, f8e5m2, FP8_E5M2_TO_FLOAT)
TOPK_OP(__nv_bfloat16, bf16, __bfloat162float)
TOPK_OP(__half, f16, __half2float)
TOPK_OP(float, f32, IDENTITY)
TOPK_OP(double, f64, IDENTITY)
TOPK_OP(uint8_t, u8, IDENTITY)
TOPK_OP(uint16_t, u16, IDENTITY)
TOPK_OP(uint32_t, u32, IDENTITY)
TOPK_OP(uint64_t, u64, IDENTITY)
TOPK_OP(int8_t, i8, IDENTITY)
TOPK_OP(int16_t, i16, IDENTITY)
TOPK_OP(int32_t, i32, IDENTITY)
TOPK_OP(int64_t, i64, IDENTITY)
