#include "math.cuh"
#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ============================================================================
// MATRIX DETERMINANT (DET)
// ============================================================================
//
// Computes the determinant of square matrices with batch support.
// Uses direct formulas for small matrices (1x1, 2x2, 3x3) and
// LU decomposition with partial pivoting for larger matrices.
//
// Metadata layout:
// - metadata[0]: batch_size (product of batch dimensions)
// - metadata[1]: n (matrix size, N×N)
// - metadata[2]: ndim (total number of dimensions)
// - metadata[3..3+ndim]: shape
// - metadata[3+ndim..3+2*ndim]: strides
// - metadata[3+2*ndim]: offset

#define MAX_DET_SIZE 16

#define DET_GET(input, batch_offset, row, col, row_stride, col_stride)                             \
    ((input)[(batch_offset) + (row) * (row_stride) + (col) * (col_stride)])

#define DET_OP(TYPENAME, FN_NAME)                                                                  \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *input, TYPENAME *out,           \
                                                   const size_t *metadata) {                       \
        const size_t batch_size = metadata[0];                                                     \
        const size_t n = metadata[1];                                                              \
        const size_t ndim = metadata[2];                                                           \
        const size_t *shape = metadata + 3;                                                        \
        const size_t *strides = metadata + 3 + ndim;                                               \
        const size_t offset = metadata[3 + 2 * ndim];                                              \
                                                                                                   \
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;                                        \
        if (tid >= batch_size)                                                                     \
            return;                                                                                \
                                                                                                   \
        const size_t row_stride = (ndim >= 2) ? strides[ndim - 2] : n;                             \
        const size_t col_stride = (ndim >= 1) ? strides[ndim - 1] : 1;                             \
                                                                                                   \
        /* Calculate batch offset */                                                               \
        size_t batch_offset = offset;                                                              \
        if (ndim > 2) {                                                                            \
            size_t temp = tid;                                                                     \
            for (int d = (int)ndim - 3; d >= 0; d--) {                                             \
                size_t dim_size = shape[d];                                                        \
                size_t idx = temp % dim_size;                                                      \
                temp /= dim_size;                                                                  \
                batch_offset += idx * strides[d];                                                  \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        float det;                                                                                 \
                                                                                                   \
        if (n == 1) {                                                                              \
            det = to_float(DET_GET(input, batch_offset, 0, 0, row_stride, col_stride));            \
        } else if (n == 2) {                                                                       \
            float a = to_float(DET_GET(input, batch_offset, 0, 0, row_stride, col_stride));        \
            float b = to_float(DET_GET(input, batch_offset, 0, 1, row_stride, col_stride));        \
            float c = to_float(DET_GET(input, batch_offset, 1, 0, row_stride, col_stride));        \
            float d_val = to_float(DET_GET(input, batch_offset, 1, 1, row_stride, col_stride));    \
            det = a * d_val - b * c;                                                               \
        } else if (n == 3) {                                                                       \
            float a = to_float(DET_GET(input, batch_offset, 0, 0, row_stride, col_stride));        \
            float b = to_float(DET_GET(input, batch_offset, 0, 1, row_stride, col_stride));        \
            float c = to_float(DET_GET(input, batch_offset, 0, 2, row_stride, col_stride));        \
            float d_val = to_float(DET_GET(input, batch_offset, 1, 0, row_stride, col_stride));    \
            float e = to_float(DET_GET(input, batch_offset, 1, 1, row_stride, col_stride));        \
            float f = to_float(DET_GET(input, batch_offset, 1, 2, row_stride, col_stride));        \
            float g = to_float(DET_GET(input, batch_offset, 2, 0, row_stride, col_stride));        \
            float h = to_float(DET_GET(input, batch_offset, 2, 1, row_stride, col_stride));        \
            float i = to_float(DET_GET(input, batch_offset, 2, 2, row_stride, col_stride));        \
            det = a * (e * i - f * h) - b * (d_val * i - f * g) + c * (d_val * h - e * g);         \
        } else {                                                                                   \
            /* LU decomposition with partial pivoting (thread-local storage) */                    \
            float lu[MAX_DET_SIZE * MAX_DET_SIZE];                                                 \
                                                                                                   \
            /* Copy input to local LU matrix */                                                    \
            for (size_t i = 0; i < n; i++) {                                                       \
                for (size_t j = 0; j < n; j++) {                                                   \
                    lu[i * n + j] =                                                                \
                        to_float(DET_GET(input, batch_offset, i, j, row_stride, col_stride));      \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            int swaps = 0;                                                                         \
            det = 1.0f;                                                                            \
                                                                                                   \
            for (size_t k = 0; k < n; k++) {                                                       \
                /* Find pivot */                                                                   \
                size_t pivot_row = k;                                                              \
                float max_val = fabsf(lu[k * n + k]);                                              \
                                                                                                   \
                for (size_t i = k + 1; i < n; i++) {                                               \
                    float val = fabsf(lu[i * n + k]);                                              \
                    if (val > max_val) {                                                           \
                        max_val = val;                                                             \
                        pivot_row = i;                                                             \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                /* Swap rows if needed */                                                          \
                if (pivot_row != k) {                                                              \
                    for (size_t j = 0; j < n; j++) {                                               \
                        float tmp = lu[k * n + j];                                                 \
                        lu[k * n + j] = lu[pivot_row * n + j];                                     \
                        lu[pivot_row * n + j] = tmp;                                               \
                    }                                                                              \
                    swaps++;                                                                       \
                }                                                                                  \
                                                                                                   \
                float pivot = lu[k * n + k];                                                       \
                                                                                                   \
                /* Check for singular matrix */                                                    \
                if (fabsf(pivot) < 1e-15f) {                                                       \
                    det = 0.0f;                                                                    \
                    break;                                                                         \
                }                                                                                  \
                                                                                                   \
                det *= pivot;                                                                      \
                                                                                                   \
                /* Eliminate below */                                                              \
                for (size_t i = k + 1; i < n; i++) {                                               \
                    float factor = lu[i * n + k] / pivot;                                          \
                    for (size_t j = k; j < n; j++) {                                               \
                        lu[i * n + j] -= factor * lu[k * n + j];                                   \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            /* Apply sign from row swaps */                                                        \
            if (swaps % 2 != 0) {                                                                  \
                det = -det;                                                                        \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        out[tid] = from_float<TYPENAME>(det);                                                      \
    }

DET_OP(__nv_fp8_e4m3, det_f8e4m3)
DET_OP(__nv_fp8_e5m2, det_f8e5m2)
DET_OP(__nv_bfloat16, det_bf16)
DET_OP(__half, det_f16)
DET_OP(float, det_f32)
DET_OP(double, det_f64)
DET_OP(int8_t, det_i8)
DET_OP(int16_t, det_i16)
DET_OP(int32_t, det_i32)
DET_OP(int64_t, det_i64)
DET_OP(uint8_t, det_u8)
DET_OP(uint16_t, det_u16)
DET_OP(uint32_t, det_u32)
DET_OP(uint64_t, det_u64)
