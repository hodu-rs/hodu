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

// ============================================================================
// MATRIX INVERSE (INV)
// ============================================================================
//
// Computes the inverse of square matrices with batch support.
// Uses direct formulas for small matrices (1x1, 2x2, 3x3) and
// Gauss-Jordan elimination with partial pivoting for larger matrices.
//
// Input: [..., N, N]
// Output: [..., N, N]
//
// Metadata layout (same as det):
// - metadata[0]: batch_size (product of batch dimensions)
// - metadata[1]: n (matrix size, N×N)
// - metadata[2]: ndim (total number of dimensions)
// - metadata[3..3+ndim]: shape
// - metadata[3+ndim..3+2*ndim]: strides
// - metadata[3+2*ndim]: offset

#define MAX_INV_SIZE 16

#define INV_SET(output, batch_offset, row, col, n, value)                                          \
    ((output)[(batch_offset) + (row) * (n) + (col)] = (value))

#define INV_OP(TYPENAME, FN_NAME)                                                                  \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *input, TYPENAME *output,        \
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
        /* Calculate batch offset for input */                                                     \
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
        size_t out_batch_offset = tid * n * n;                                                     \
                                                                                                   \
        if (n == 1) {                                                                              \
            float a = to_float(DET_GET(input, batch_offset, 0, 0, row_stride, col_stride));        \
            INV_SET(output, out_batch_offset, 0, 0, n, from_float<TYPENAME>(1.0f / a));            \
        } else if (n == 2) {                                                                       \
            float a = to_float(DET_GET(input, batch_offset, 0, 0, row_stride, col_stride));        \
            float b = to_float(DET_GET(input, batch_offset, 0, 1, row_stride, col_stride));        \
            float c = to_float(DET_GET(input, batch_offset, 1, 0, row_stride, col_stride));        \
            float d_val = to_float(DET_GET(input, batch_offset, 1, 1, row_stride, col_stride));    \
            float det = a * d_val - b * c;                                                         \
            float inv_det = 1.0f / det;                                                            \
            INV_SET(output, out_batch_offset, 0, 0, n, from_float<TYPENAME>(d_val * inv_det));     \
            INV_SET(output, out_batch_offset, 0, 1, n, from_float<TYPENAME>(-b * inv_det));        \
            INV_SET(output, out_batch_offset, 1, 0, n, from_float<TYPENAME>(-c * inv_det));        \
            INV_SET(output, out_batch_offset, 1, 1, n, from_float<TYPENAME>(a * inv_det));         \
        } else if (n == 3) {                                                                       \
            float a = to_float(DET_GET(input, batch_offset, 0, 0, row_stride, col_stride));        \
            float b = to_float(DET_GET(input, batch_offset, 0, 1, row_stride, col_stride));        \
            float cv = to_float(DET_GET(input, batch_offset, 0, 2, row_stride, col_stride));       \
            float d_val = to_float(DET_GET(input, batch_offset, 1, 0, row_stride, col_stride));    \
            float e = to_float(DET_GET(input, batch_offset, 1, 1, row_stride, col_stride));        \
            float f = to_float(DET_GET(input, batch_offset, 1, 2, row_stride, col_stride));        \
            float g = to_float(DET_GET(input, batch_offset, 2, 0, row_stride, col_stride));        \
            float h = to_float(DET_GET(input, batch_offset, 2, 1, row_stride, col_stride));        \
            float iv = to_float(DET_GET(input, batch_offset, 2, 2, row_stride, col_stride));       \
                                                                                                   \
            float det =                                                                            \
                a * (e * iv - f * h) - b * (d_val * iv - f * g) + cv * (d_val * h - e * g);        \
            float inv_det = 1.0f / det;                                                            \
                                                                                                   \
            INV_SET(output, out_batch_offset, 0, 0, n,                                             \
                    from_float<TYPENAME>((e * iv - f * h) * inv_det));                             \
            INV_SET(output, out_batch_offset, 0, 1, n,                                             \
                    from_float<TYPENAME>((cv * h - b * iv) * inv_det));                            \
            INV_SET(output, out_batch_offset, 0, 2, n,                                             \
                    from_float<TYPENAME>((b * f - cv * e) * inv_det));                             \
            INV_SET(output, out_batch_offset, 1, 0, n,                                             \
                    from_float<TYPENAME>((f * g - d_val * iv) * inv_det));                         \
            INV_SET(output, out_batch_offset, 1, 1, n,                                             \
                    from_float<TYPENAME>((a * iv - cv * g) * inv_det));                            \
            INV_SET(output, out_batch_offset, 1, 2, n,                                             \
                    from_float<TYPENAME>((cv * d_val - a * f) * inv_det));                         \
            INV_SET(output, out_batch_offset, 2, 0, n,                                             \
                    from_float<TYPENAME>((d_val * h - e * g) * inv_det));                          \
            INV_SET(output, out_batch_offset, 2, 1, n,                                             \
                    from_float<TYPENAME>((b * g - a * h) * inv_det));                              \
            INV_SET(output, out_batch_offset, 2, 2, n,                                             \
                    from_float<TYPENAME>((a * e - b * d_val) * inv_det));                          \
        } else {                                                                                   \
            /* Gauss-Jordan elimination (thread-local storage) */                                  \
            float aug[MAX_INV_SIZE * MAX_INV_SIZE * 2];                                            \
                                                                                                   \
            /* Initialize augmented matrix [A | I] */                                              \
            for (size_t ii = 0; ii < n; ii++) {                                                    \
                for (size_t jj = 0; jj < n; jj++) {                                                \
                    aug[ii * 2 * n + jj] =                                                         \
                        to_float(DET_GET(input, batch_offset, ii, jj, row_stride, col_stride));    \
                    aug[ii * 2 * n + n + jj] = (ii == jj) ? 1.0f : 0.0f;                           \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t k = 0; k < n; k++) {                                                       \
                /* Find pivot */                                                                   \
                size_t pivot_row = k;                                                              \
                float max_val = fabsf(aug[k * 2 * n + k]);                                         \
                                                                                                   \
                for (size_t ii = k + 1; ii < n; ii++) {                                            \
                    float val = fabsf(aug[ii * 2 * n + k]);                                        \
                    if (val > max_val) {                                                           \
                        max_val = val;                                                             \
                        pivot_row = ii;                                                            \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                /* Swap rows if needed */                                                          \
                if (pivot_row != k) {                                                              \
                    for (size_t jj = 0; jj < 2 * n; jj++) {                                        \
                        float tmp = aug[k * 2 * n + jj];                                           \
                        aug[k * 2 * n + jj] = aug[pivot_row * 2 * n + jj];                         \
                        aug[pivot_row * 2 * n + jj] = tmp;                                         \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                /* Scale pivot row */                                                              \
                float pivot = aug[k * 2 * n + k];                                                  \
                for (size_t jj = 0; jj < 2 * n; jj++) {                                            \
                    aug[k * 2 * n + jj] /= pivot;                                                  \
                }                                                                                  \
                                                                                                   \
                /* Eliminate column k in other rows */                                             \
                for (size_t ii = 0; ii < n; ii++) {                                                \
                    if (ii != k) {                                                                 \
                        float factor = aug[ii * 2 * n + k];                                        \
                        for (size_t jj = 0; jj < 2 * n; jj++) {                                    \
                            aug[ii * 2 * n + jj] -= factor * aug[k * 2 * n + jj];                  \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            /* Copy result from right half of augmented matrix */                                  \
            for (size_t ii = 0; ii < n; ii++) {                                                    \
                for (size_t jj = 0; jj < n; jj++) {                                                \
                    INV_SET(output, out_batch_offset, ii, jj, n,                                   \
                            from_float<TYPENAME>(aug[ii * 2 * n + n + jj]));                       \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

INV_OP(__nv_fp8_e4m3, inv_f8e4m3)
INV_OP(__nv_fp8_e5m2, inv_f8e5m2)
INV_OP(__nv_bfloat16, inv_bf16)
INV_OP(__half, inv_f16)
INV_OP(float, inv_f32)
INV_OP(double, inv_f64)
INV_OP(int8_t, inv_i8)
INV_OP(int16_t, inv_i16)
INV_OP(int32_t, inv_i32)
INV_OP(int64_t, inv_i64)
INV_OP(uint8_t, inv_u8)
INV_OP(uint16_t, inv_u16)
INV_OP(uint32_t, inv_u32)
INV_OP(uint64_t, inv_u64)
