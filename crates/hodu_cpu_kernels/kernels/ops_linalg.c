#include "ops_linalg.h"
#include "types.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// ============================================================================
// MATRIX DETERMINANT (DET)
// ============================================================================
//
// Computes the determinant of square matrices with batch support.
// Input: [..., N, N] -> Output: [...]
//
// Uses LU decomposition with partial pivoting for NxN matrices.
// For small matrices (1x1, 2x2, 3x3), uses direct formulas.
//
// Metadata layout:
// - metadata[0]: batch_size
// - metadata[1]: n (matrix size, N×N)
// - metadata[2]: ndim
// - metadata[3..3+ndim]: shape
// - metadata[3+ndim..3+2*ndim]: strides
// - metadata[3+2*ndim]: offset
//
// Algorithm:
// - 1x1: det = a[0,0]
// - 2x2: det = a*d - b*c
// - 3x3: Sarrus' rule
// - NxN: LU decomposition with partial pivoting
//        det = (-1)^swaps * product(diagonal of U)

/// Helper to get element from strided matrix for det
#define DET_GET(input, batch_offset, row, col, row_stride, col_stride)                             \
    ((input)[(batch_offset) + (row) * (row_stride) + (col) * (col_stride)])

/// Macro for determinant operation
#define DET_OP(TYPE, TYPE_SUFFIX)                                                                  \
    void hodu_cpu_det_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                       \
                                    const size_t *metadata) {                                      \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t batch_size = metadata[0];                                                     \
        const size_t n = metadata[1];                                                              \
        const size_t ndim = metadata[2];                                                           \
        const size_t *strides = metadata + 3 + ndim;                                               \
        const size_t offset = metadata[3 + 2 * ndim];                                              \
                                                                                                   \
        /* Get row and column strides (last two dimensions) */                                     \
        const size_t row_stride = (ndim >= 2) ? strides[ndim - 2] : n;                             \
        const size_t col_stride = (ndim >= 1) ? strides[ndim - 1] : 1;                             \
                                                                                                   \
        for (size_t batch = 0; batch < batch_size; batch++) {                                      \
            /* Calculate batch offset using strides */                                             \
            size_t batch_offset = offset;                                                          \
            if (ndim > 2) {                                                                        \
                size_t temp = batch;                                                               \
                const size_t *shape = metadata + 3;                                                \
                for (int d = (int)ndim - 3; d >= 0; d--) {                                         \
                    size_t dim_size = shape[d];                                                    \
                    size_t idx = temp % dim_size;                                                  \
                    temp /= dim_size;                                                              \
                    batch_offset += idx * strides[d];                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            TYPE det;                                                                              \
                                                                                                   \
            if (n == 1) {                                                                          \
                det = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);                  \
            } else if (n == 2) {                                                                   \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                TYPE b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
                TYPE c = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);               \
                TYPE d_val = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);           \
                det = a * d_val - b * c;                                                           \
            } else if (n == 3) {                                                                   \
                /* Sarrus' rule for 3x3 */                                                         \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                TYPE b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
                TYPE c = DET_GET(input, batch_offset, 0, 2, row_stride, col_stride);               \
                TYPE d_val = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);           \
                TYPE e = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);               \
                TYPE f = DET_GET(input, batch_offset, 1, 2, row_stride, col_stride);               \
                TYPE g = DET_GET(input, batch_offset, 2, 0, row_stride, col_stride);               \
                TYPE h = DET_GET(input, batch_offset, 2, 1, row_stride, col_stride);               \
                TYPE i = DET_GET(input, batch_offset, 2, 2, row_stride, col_stride);               \
                det = a * (e * i - f * h) - b * (d_val * i - f * g) + c * (d_val * h - e * g);     \
            } else {                                                                               \
                /* LU decomposition with partial pivoting for NxN */                               \
                /* Allocate temporary matrix */                                                    \
                TYPE *lu = (TYPE *)malloc(n * n * sizeof(TYPE));                                   \
                                                                                                   \
                /* Copy input to LU matrix */                                                      \
                for (size_t i = 0; i < n; i++) {                                                   \
                    for (size_t j = 0; j < n; j++) {                                               \
                        lu[i * n + j] =                                                            \
                            DET_GET(input, batch_offset, i, j, row_stride, col_stride);            \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                int swaps = 0;                                                                     \
                det = 1;                                                                           \
                                                                                                   \
                /* LU decomposition with partial pivoting */                                       \
                for (size_t k = 0; k < n; k++) {                                                   \
                    /* Find pivot */                                                               \
                    size_t pivot_row = k;                                                          \
                    TYPE max_val = lu[k * n + k];                                                  \
                    if (max_val < 0)                                                               \
                        max_val = -max_val;                                                        \
                                                                                                   \
                    for (size_t i = k + 1; i < n; i++) {                                           \
                        TYPE val = lu[i * n + k];                                                  \
                        if (val < 0)                                                               \
                            val = -val;                                                            \
                        if (val > max_val) {                                                       \
                            max_val = val;                                                         \
                            pivot_row = i;                                                         \
                        }                                                                          \
                    }                                                                              \
                                                                                                   \
                    /* Swap rows if needed */                                                      \
                    if (pivot_row != k) {                                                          \
                        for (size_t j = 0; j < n; j++) {                                           \
                            TYPE tmp = lu[k * n + j];                                              \
                            lu[k * n + j] = lu[pivot_row * n + j];                                 \
                            lu[pivot_row * n + j] = tmp;                                           \
                        }                                                                          \
                        swaps++;                                                                   \
                    }                                                                              \
                                                                                                   \
                    TYPE pivot = lu[k * n + k];                                                    \
                                                                                                   \
                    /* Check for singular matrix */                                                \
                    TYPE abs_pivot = pivot;                                                        \
                    if (abs_pivot < 0)                                                             \
                        abs_pivot = -abs_pivot;                                                    \
                    if (abs_pivot < (TYPE)1e-15) {                                                 \
                        det = 0;                                                                   \
                        free(lu);                                                                  \
                        goto store_result_##TYPE_SUFFIX;                                           \
                    }                                                                              \
                                                                                                   \
                    det *= pivot;                                                                  \
                                                                                                   \
                    /* Eliminate below */                                                          \
                    for (size_t i = k + 1; i < n; i++) {                                           \
                        TYPE factor = lu[i * n + k] / pivot;                                       \
                        for (size_t j = k; j < n; j++) {                                           \
                            lu[i * n + j] -= factor * lu[k * n + j];                               \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                /* Apply sign from row swaps */                                                    \
                if (swaps % 2 != 0) {                                                              \
                    det = -det;                                                                    \
                }                                                                                  \
                                                                                                   \
                free(lu);                                                                          \
            }                                                                                      \
                                                                                                   \
            store_result_##TYPE_SUFFIX : output[batch] = det;                                      \
        }                                                                                          \
    }

/// Macro for determinant operation with exotic types
#define DET_OP_EXOTIC(TYPE, TYPE_SUFFIX, ZERO, ONE, ADD_FN, SUB_FN, MUL_FN, DIV_FN, NEG_FN,        \
                      ABS_FN, LT_FN)                                                               \
    void hodu_cpu_det_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                       \
                                    const size_t *metadata) {                                      \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t batch_size = metadata[0];                                                     \
        const size_t n = metadata[1];                                                              \
        const size_t ndim = metadata[2];                                                           \
        const size_t *strides = metadata + 3 + ndim;                                               \
        const size_t offset = metadata[3 + 2 * ndim];                                              \
                                                                                                   \
        const size_t row_stride = (ndim >= 2) ? strides[ndim - 2] : n;                             \
        const size_t col_stride = (ndim >= 1) ? strides[ndim - 1] : 1;                             \
                                                                                                   \
        for (size_t batch = 0; batch < batch_size; batch++) {                                      \
            size_t batch_offset = offset;                                                          \
            if (ndim > 2) {                                                                        \
                size_t temp = batch;                                                               \
                const size_t *shape = metadata + 3;                                                \
                for (int d = (int)ndim - 3; d >= 0; d--) {                                         \
                    size_t dim_size = shape[d];                                                    \
                    size_t idx = temp % dim_size;                                                  \
                    temp /= dim_size;                                                              \
                    batch_offset += idx * strides[d];                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            TYPE det;                                                                              \
                                                                                                   \
            if (n == 1) {                                                                          \
                det = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);                  \
            } else if (n == 2) {                                                                   \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                TYPE b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
                TYPE c = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);               \
                TYPE d_val = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);           \
                det = SUB_FN(MUL_FN(a, d_val), MUL_FN(b, c));                                      \
            } else if (n == 3) {                                                                   \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                TYPE b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
                TYPE c = DET_GET(input, batch_offset, 0, 2, row_stride, col_stride);               \
                TYPE d_val = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);           \
                TYPE e = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);               \
                TYPE f = DET_GET(input, batch_offset, 1, 2, row_stride, col_stride);               \
                TYPE g = DET_GET(input, batch_offset, 2, 0, row_stride, col_stride);               \
                TYPE h = DET_GET(input, batch_offset, 2, 1, row_stride, col_stride);               \
                TYPE i = DET_GET(input, batch_offset, 2, 2, row_stride, col_stride);               \
                TYPE t1 = MUL_FN(a, SUB_FN(MUL_FN(e, i), MUL_FN(f, h)));                           \
                TYPE t2 = MUL_FN(b, SUB_FN(MUL_FN(d_val, i), MUL_FN(f, g)));                       \
                TYPE t3 = MUL_FN(c, SUB_FN(MUL_FN(d_val, h), MUL_FN(e, g)));                       \
                det = ADD_FN(SUB_FN(t1, t2), t3);                                                  \
            } else {                                                                               \
                /* LU decomposition for exotic types */                                            \
                TYPE *lu = (TYPE *)malloc(n * n * sizeof(TYPE));                                   \
                for (size_t i = 0; i < n; i++) {                                                   \
                    for (size_t j = 0; j < n; j++) {                                               \
                        lu[i * n + j] =                                                            \
                            DET_GET(input, batch_offset, i, j, row_stride, col_stride);            \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                int swaps = 0;                                                                     \
                det = ONE;                                                                         \
                                                                                                   \
                for (size_t k = 0; k < n; k++) {                                                   \
                    size_t pivot_row = k;                                                          \
                    TYPE max_val = ABS_FN(lu[k * n + k]);                                          \
                                                                                                   \
                    for (size_t i = k + 1; i < n; i++) {                                           \
                        TYPE val = ABS_FN(lu[i * n + k]);                                          \
                        if (LT_FN(max_val, val)) {                                                 \
                            max_val = val;                                                         \
                            pivot_row = i;                                                         \
                        }                                                                          \
                    }                                                                              \
                                                                                                   \
                    if (pivot_row != k) {                                                          \
                        for (size_t j = 0; j < n; j++) {                                           \
                            TYPE tmp = lu[k * n + j];                                              \
                            lu[k * n + j] = lu[pivot_row * n + j];                                 \
                            lu[pivot_row * n + j] = tmp;                                           \
                        }                                                                          \
                        swaps++;                                                                   \
                    }                                                                              \
                                                                                                   \
                    TYPE pivot = lu[k * n + k];                                                    \
                    det = MUL_FN(det, pivot);                                                      \
                                                                                                   \
                    for (size_t i = k + 1; i < n; i++) {                                           \
                        TYPE factor = DIV_FN(lu[i * n + k], pivot);                                \
                        for (size_t j = k; j < n; j++) {                                           \
                            lu[i * n + j] = SUB_FN(lu[i * n + j], MUL_FN(factor, lu[k * n + j]));  \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                if (swaps % 2 != 0) {                                                              \
                    det = NEG_FN(det);                                                             \
                }                                                                                  \
                                                                                                   \
                free(lu);                                                                          \
            }                                                                                      \
                                                                                                   \
            output[batch] = det;                                                                   \
        }                                                                                          \
    }

// Generate det implementations
DET_OP(f32_t, f32)
DET_OP(f64_t, f64)
DET_OP_EXOTIC(f8e4m3_t, f8e4m3, F8E4M3_ZERO, F8E4M3_ONE, f8e4m3_add, f8e4m3_sub, f8e4m3_mul,
              f8e4m3_div, f8e4m3_neg, f8e4m3_abs, f8e4m3_lt)
DET_OP_EXOTIC(f8e5m2_t, f8e5m2, F8E5M2_ZERO, F8E5M2_ONE, f8e5m2_add, f8e5m2_sub, f8e5m2_mul,
              f8e5m2_div, f8e5m2_neg, f8e5m2_abs, f8e5m2_lt)
DET_OP_EXOTIC(bf16_t, bf16, BF16_ZERO, BF16_ONE, bf16_add, bf16_sub, bf16_mul, bf16_div, bf16_neg,
              bf16_abs, bf16_lt)
DET_OP_EXOTIC(f16_t, f16, F16_ZERO, F16_ONE, f16_add, f16_sub, f16_mul, f16_div, f16_neg, f16_abs,
              f16_lt)
DET_OP(int8_t, i8)
DET_OP(int16_t, i16)
DET_OP(int32_t, i32)
DET_OP(int64_t, i64)
DET_OP(uint8_t, u8)
DET_OP(uint16_t, u16)
DET_OP(uint32_t, u32)
DET_OP(uint64_t, u64)

// ============================================================================
// MATRIX INVERSE (INV)
// ============================================================================
//
// Computes the inverse of square matrices with batch support.
// Input: [..., N, N] -> Output: [..., N, N]
//
// Uses Gauss-Jordan elimination with partial pivoting.
// For small matrices (1x1, 2x2, 3x3), uses direct formulas.
//
// Metadata layout (same as det):
// - metadata[0]: batch_size
// - metadata[1]: n (matrix size, N×N)
// - metadata[2]: ndim
// - metadata[3..3+ndim]: shape
// - metadata[3+ndim..3+2*ndim]: strides
// - metadata[3+2*ndim]: offset

/// Helper to set element in output matrix for inv
#define INV_SET(output, batch_offset, row, col, n, value)                                          \
    ((output)[(batch_offset) + (row) * (n) + (col)] = (value))

/// Macro for matrix inverse operation
#define INV_OP(TYPE, TYPE_SUFFIX)                                                                  \
    void hodu_cpu_inv_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                       \
                                    const size_t *metadata) {                                      \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t batch_size = metadata[0];                                                     \
        const size_t n = metadata[1];                                                              \
        const size_t ndim = metadata[2];                                                           \
        const size_t *strides = metadata + 3 + ndim;                                               \
        const size_t offset = metadata[3 + 2 * ndim];                                              \
                                                                                                   \
        const size_t row_stride = (ndim >= 2) ? strides[ndim - 2] : n;                             \
        const size_t col_stride = (ndim >= 1) ? strides[ndim - 1] : 1;                             \
                                                                                                   \
        for (size_t batch = 0; batch < batch_size; batch++) {                                      \
            size_t batch_offset = offset;                                                          \
            if (ndim > 2) {                                                                        \
                size_t temp = batch;                                                               \
                const size_t *shape = metadata + 3;                                                \
                for (int d = (int)ndim - 3; d >= 0; d--) {                                         \
                    size_t dim_size = shape[d];                                                    \
                    size_t idx = temp % dim_size;                                                  \
                    temp /= dim_size;                                                              \
                    batch_offset += idx * strides[d];                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            size_t out_batch_offset = batch * n * n;                                               \
                                                                                                   \
            if (n == 1) {                                                                          \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                INV_SET(output, out_batch_offset, 0, 0, n, (TYPE)1 / a);                           \
            } else if (n == 2) {                                                                   \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                TYPE b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
                TYPE c = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);               \
                TYPE d_val = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);           \
                TYPE det = a * d_val - b * c;                                                      \
                TYPE inv_det = (TYPE)1 / det;                                                      \
                INV_SET(output, out_batch_offset, 0, 0, n, d_val * inv_det);                       \
                INV_SET(output, out_batch_offset, 0, 1, n, -b * inv_det);                          \
                INV_SET(output, out_batch_offset, 1, 0, n, -c * inv_det);                          \
                INV_SET(output, out_batch_offset, 1, 1, n, a * inv_det);                           \
            } else if (n == 3) {                                                                   \
                /* 3x3 inverse using adjugate/determinant method */                                \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                TYPE b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
                TYPE c = DET_GET(input, batch_offset, 0, 2, row_stride, col_stride);               \
                TYPE d_val = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);           \
                TYPE e = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);               \
                TYPE f = DET_GET(input, batch_offset, 1, 2, row_stride, col_stride);               \
                TYPE g = DET_GET(input, batch_offset, 2, 0, row_stride, col_stride);               \
                TYPE h = DET_GET(input, batch_offset, 2, 1, row_stride, col_stride);               \
                TYPE i = DET_GET(input, batch_offset, 2, 2, row_stride, col_stride);               \
                                                                                                   \
                TYPE det =                                                                         \
                    a * (e * i - f * h) - b * (d_val * i - f * g) + c * (d_val * h - e * g);       \
                TYPE inv_det = (TYPE)1 / det;                                                      \
                                                                                                   \
                /* Adjugate matrix / det */                                                        \
                INV_SET(output, out_batch_offset, 0, 0, n, (e * i - f * h) * inv_det);             \
                INV_SET(output, out_batch_offset, 0, 1, n, (c * h - b * i) * inv_det);             \
                INV_SET(output, out_batch_offset, 0, 2, n, (b * f - c * e) * inv_det);             \
                INV_SET(output, out_batch_offset, 1, 0, n, (f * g - d_val * i) * inv_det);         \
                INV_SET(output, out_batch_offset, 1, 1, n, (a * i - c * g) * inv_det);             \
                INV_SET(output, out_batch_offset, 1, 2, n, (c * d_val - a * f) * inv_det);         \
                INV_SET(output, out_batch_offset, 2, 0, n, (d_val * h - e * g) * inv_det);         \
                INV_SET(output, out_batch_offset, 2, 1, n, (b * g - a * h) * inv_det);             \
                INV_SET(output, out_batch_offset, 2, 2, n, (a * e - b * d_val) * inv_det);         \
            } else {                                                                               \
                /* Gauss-Jordan elimination for NxN */                                             \
                TYPE *aug = (TYPE *)malloc(n * 2 * n * sizeof(TYPE));                              \
                                                                                                   \
                /* Initialize augmented matrix [A | I] */                                          \
                for (size_t ii = 0; ii < n; ii++) {                                                \
                    for (size_t jj = 0; jj < n; jj++) {                                            \
                        aug[ii * 2 * n + jj] =                                                     \
                            DET_GET(input, batch_offset, ii, jj, row_stride, col_stride);          \
                        aug[ii * 2 * n + n + jj] = (ii == jj) ? (TYPE)1 : (TYPE)0;                 \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                /* Gauss-Jordan elimination with partial pivoting */                               \
                for (size_t k = 0; k < n; k++) {                                                   \
                    /* Find pivot */                                                               \
                    size_t pivot_row = k;                                                          \
                    TYPE max_val = aug[k * 2 * n + k];                                             \
                    if (max_val < 0)                                                               \
                        max_val = -max_val;                                                        \
                                                                                                   \
                    for (size_t ii = k + 1; ii < n; ii++) {                                        \
                        TYPE val = aug[ii * 2 * n + k];                                            \
                        if (val < 0)                                                               \
                            val = -val;                                                            \
                        if (val > max_val) {                                                       \
                            max_val = val;                                                         \
                            pivot_row = ii;                                                        \
                        }                                                                          \
                    }                                                                              \
                                                                                                   \
                    /* Swap rows if needed */                                                      \
                    if (pivot_row != k) {                                                          \
                        for (size_t jj = 0; jj < 2 * n; jj++) {                                    \
                            TYPE tmp = aug[k * 2 * n + jj];                                        \
                            aug[k * 2 * n + jj] = aug[pivot_row * 2 * n + jj];                     \
                            aug[pivot_row * 2 * n + jj] = tmp;                                     \
                        }                                                                          \
                    }                                                                              \
                                                                                                   \
                    /* Scale pivot row */                                                          \
                    TYPE pivot = aug[k * 2 * n + k];                                               \
                    for (size_t jj = 0; jj < 2 * n; jj++) {                                        \
                        aug[k * 2 * n + jj] /= pivot;                                              \
                    }                                                                              \
                                                                                                   \
                    /* Eliminate column k in other rows */                                         \
                    for (size_t ii = 0; ii < n; ii++) {                                            \
                        if (ii != k) {                                                             \
                            TYPE factor = aug[ii * 2 * n + k];                                     \
                            for (size_t jj = 0; jj < 2 * n; jj++) {                                \
                                aug[ii * 2 * n + jj] -= factor * aug[k * 2 * n + jj];              \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                /* Copy result from right half of augmented matrix */                              \
                for (size_t ii = 0; ii < n; ii++) {                                                \
                    for (size_t jj = 0; jj < n; jj++) {                                            \
                        INV_SET(output, out_batch_offset, ii, jj, n, aug[ii * 2 * n + n + jj]);    \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                free(aug);                                                                         \
            }                                                                                      \
        }                                                                                          \
    }

/// Macro for matrix inverse with exotic types
#define INV_OP_EXOTIC(TYPE, TYPE_SUFFIX, ZERO, ONE, ADD_FN, SUB_FN, MUL_FN, DIV_FN, NEG_FN,        \
                      ABS_FN, LT_FN)                                                               \
    void hodu_cpu_inv_##TYPE_SUFFIX(const void *input_ptr, void *output_ptr,                       \
                                    const size_t *metadata) {                                      \
        const TYPE *input = (const TYPE *)input_ptr;                                               \
        TYPE *output = (TYPE *)output_ptr;                                                         \
                                                                                                   \
        const size_t batch_size = metadata[0];                                                     \
        const size_t n = metadata[1];                                                              \
        const size_t ndim = metadata[2];                                                           \
        const size_t *strides = metadata + 3 + ndim;                                               \
        const size_t offset = metadata[3 + 2 * ndim];                                              \
                                                                                                   \
        const size_t row_stride = (ndim >= 2) ? strides[ndim - 2] : n;                             \
        const size_t col_stride = (ndim >= 1) ? strides[ndim - 1] : 1;                             \
                                                                                                   \
        for (size_t batch = 0; batch < batch_size; batch++) {                                      \
            size_t batch_offset = offset;                                                          \
            if (ndim > 2) {                                                                        \
                size_t temp = batch;                                                               \
                const size_t *shape = metadata + 3;                                                \
                for (int d = (int)ndim - 3; d >= 0; d--) {                                         \
                    size_t dim_size = shape[d];                                                    \
                    size_t idx = temp % dim_size;                                                  \
                    temp /= dim_size;                                                              \
                    batch_offset += idx * strides[d];                                              \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            size_t out_batch_offset = batch * n * n;                                               \
                                                                                                   \
            if (n == 1) {                                                                          \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                INV_SET(output, out_batch_offset, 0, 0, n, DIV_FN(ONE, a));                        \
            } else if (n == 2) {                                                                   \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                TYPE b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
                TYPE c = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);               \
                TYPE d_val = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);           \
                TYPE det = SUB_FN(MUL_FN(a, d_val), MUL_FN(b, c));                                 \
                TYPE inv_det = DIV_FN(ONE, det);                                                   \
                INV_SET(output, out_batch_offset, 0, 0, n, MUL_FN(d_val, inv_det));                \
                INV_SET(output, out_batch_offset, 0, 1, n, MUL_FN(NEG_FN(b), inv_det));            \
                INV_SET(output, out_batch_offset, 1, 0, n, MUL_FN(NEG_FN(c), inv_det));            \
                INV_SET(output, out_batch_offset, 1, 1, n, MUL_FN(a, inv_det));                    \
            } else if (n == 3) {                                                                   \
                TYPE a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
                TYPE b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
                TYPE cv = DET_GET(input, batch_offset, 0, 2, row_stride, col_stride);              \
                TYPE d_val = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);           \
                TYPE e = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);               \
                TYPE f = DET_GET(input, batch_offset, 1, 2, row_stride, col_stride);               \
                TYPE g = DET_GET(input, batch_offset, 2, 0, row_stride, col_stride);               \
                TYPE h = DET_GET(input, batch_offset, 2, 1, row_stride, col_stride);               \
                TYPE iv = DET_GET(input, batch_offset, 2, 2, row_stride, col_stride);              \
                                                                                                   \
                TYPE t1 = MUL_FN(a, SUB_FN(MUL_FN(e, iv), MUL_FN(f, h)));                          \
                TYPE t2 = MUL_FN(b, SUB_FN(MUL_FN(d_val, iv), MUL_FN(f, g)));                      \
                TYPE t3 = MUL_FN(cv, SUB_FN(MUL_FN(d_val, h), MUL_FN(e, g)));                      \
                TYPE det = ADD_FN(SUB_FN(t1, t2), t3);                                             \
                TYPE inv_det = DIV_FN(ONE, det);                                                   \
                                                                                                   \
                INV_SET(output, out_batch_offset, 0, 0, n,                                         \
                        MUL_FN(SUB_FN(MUL_FN(e, iv), MUL_FN(f, h)), inv_det));                     \
                INV_SET(output, out_batch_offset, 0, 1, n,                                         \
                        MUL_FN(SUB_FN(MUL_FN(cv, h), MUL_FN(b, iv)), inv_det));                    \
                INV_SET(output, out_batch_offset, 0, 2, n,                                         \
                        MUL_FN(SUB_FN(MUL_FN(b, f), MUL_FN(cv, e)), inv_det));                     \
                INV_SET(output, out_batch_offset, 1, 0, n,                                         \
                        MUL_FN(SUB_FN(MUL_FN(f, g), MUL_FN(d_val, iv)), inv_det));                 \
                INV_SET(output, out_batch_offset, 1, 1, n,                                         \
                        MUL_FN(SUB_FN(MUL_FN(a, iv), MUL_FN(cv, g)), inv_det));                    \
                INV_SET(output, out_batch_offset, 1, 2, n,                                         \
                        MUL_FN(SUB_FN(MUL_FN(cv, d_val), MUL_FN(a, f)), inv_det));                 \
                INV_SET(output, out_batch_offset, 2, 0, n,                                         \
                        MUL_FN(SUB_FN(MUL_FN(d_val, h), MUL_FN(e, g)), inv_det));                  \
                INV_SET(output, out_batch_offset, 2, 1, n,                                         \
                        MUL_FN(SUB_FN(MUL_FN(b, g), MUL_FN(a, h)), inv_det));                      \
                INV_SET(output, out_batch_offset, 2, 2, n,                                         \
                        MUL_FN(SUB_FN(MUL_FN(a, e), MUL_FN(b, d_val)), inv_det));                  \
            } else {                                                                               \
                /* Gauss-Jordan for exotic types - work in native format */                        \
                TYPE *aug = (TYPE *)malloc(n * 2 * n * sizeof(TYPE));                              \
                                                                                                   \
                for (size_t ii = 0; ii < n; ii++) {                                                \
                    for (size_t jj = 0; jj < n; jj++) {                                            \
                        aug[ii * 2 * n + jj] =                                                     \
                            DET_GET(input, batch_offset, ii, jj, row_stride, col_stride);          \
                        aug[ii * 2 * n + n + jj] = (ii == jj) ? ONE : ZERO;                        \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                for (size_t k = 0; k < n; k++) {                                                   \
                    size_t pivot_row = k;                                                          \
                    TYPE max_val = ABS_FN(aug[k * 2 * n + k]);                                     \
                                                                                                   \
                    for (size_t ii = k + 1; ii < n; ii++) {                                        \
                        TYPE val = ABS_FN(aug[ii * 2 * n + k]);                                    \
                        if (LT_FN(max_val, val)) {                                                 \
                            max_val = val;                                                         \
                            pivot_row = ii;                                                        \
                        }                                                                          \
                    }                                                                              \
                                                                                                   \
                    if (pivot_row != k) {                                                          \
                        for (size_t jj = 0; jj < 2 * n; jj++) {                                    \
                            TYPE tmp = aug[k * 2 * n + jj];                                        \
                            aug[k * 2 * n + jj] = aug[pivot_row * 2 * n + jj];                     \
                            aug[pivot_row * 2 * n + jj] = tmp;                                     \
                        }                                                                          \
                    }                                                                              \
                                                                                                   \
                    TYPE pivot = aug[k * 2 * n + k];                                               \
                    for (size_t jj = 0; jj < 2 * n; jj++) {                                        \
                        aug[k * 2 * n + jj] = DIV_FN(aug[k * 2 * n + jj], pivot);                  \
                    }                                                                              \
                                                                                                   \
                    for (size_t ii = 0; ii < n; ii++) {                                            \
                        if (ii != k) {                                                             \
                            TYPE factor = aug[ii * 2 * n + k];                                     \
                            for (size_t jj = 0; jj < 2 * n; jj++) {                                \
                                aug[ii * 2 * n + jj] = SUB_FN(                                     \
                                    aug[ii * 2 * n + jj], MUL_FN(factor, aug[k * 2 * n + jj]));    \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                for (size_t ii = 0; ii < n; ii++) {                                                \
                    for (size_t jj = 0; jj < n; jj++) {                                            \
                        INV_SET(output, out_batch_offset, ii, jj, n, aug[ii * 2 * n + n + jj]);    \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                free(aug);                                                                         \
            }                                                                                      \
        }                                                                                          \
    }

// Generate inv implementations
INV_OP(f32_t, f32)
INV_OP(f64_t, f64)
INV_OP_EXOTIC(f8e4m3_t, f8e4m3, F8E4M3_ZERO, F8E4M3_ONE, f8e4m3_add, f8e4m3_sub, f8e4m3_mul,
              f8e4m3_div, f8e4m3_neg, f8e4m3_abs, f8e4m3_lt)
INV_OP_EXOTIC(f8e5m2_t, f8e5m2, F8E5M2_ZERO, F8E5M2_ONE, f8e5m2_add, f8e5m2_sub, f8e5m2_mul,
              f8e5m2_div, f8e5m2_neg, f8e5m2_abs, f8e5m2_lt)
INV_OP_EXOTIC(bf16_t, bf16, BF16_ZERO, BF16_ONE, bf16_add, bf16_sub, bf16_mul, bf16_div, bf16_neg,
              bf16_abs, bf16_lt)
INV_OP_EXOTIC(f16_t, f16, F16_ZERO, F16_ONE, f16_add, f16_sub, f16_mul, f16_div, f16_neg, f16_abs,
              f16_lt)
INV_OP(int8_t, i8)
INV_OP(int16_t, i16)
INV_OP(int32_t, i32)
INV_OP(int64_t, i64)
INV_OP(uint8_t, u8)
INV_OP(uint16_t, u16)
INV_OP(uint32_t, u32)
INV_OP(uint64_t, u64)
