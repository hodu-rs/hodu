#include "./headers/utils.metal"
#include <metal_stdlib>

using namespace metal;

// Linear algebra operations for tensors
// Currently provides:
// - det: Matrix determinant computation

// ============================================================================
// MATRIX DETERMINANT (DET)
// ============================================================================
//
// Computes the determinant of square matrices with batch support.
// Uses direct formulas for small matrices (1x1, 2x2, 3x3) and
// LU decomposition with partial pivoting for larger matrices.
//
// Input: [..., N, N]
// Output: [...]
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

// Float types use fabs() with float cast to avoid ambiguity
#define DET_OP_FLOAT(TYPENAME, FN_NAME)                                                            \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t *metadata [[buffer(2)]], uint tid [[thread_position_in_grid]]) {           \
                                                                                                   \
        const size_t batch_size = metadata[0];                                                     \
        const size_t n = metadata[1];                                                              \
        const size_t ndim = metadata[2];                                                           \
        constant size_t *shape = metadata + 3;                                                     \
        constant size_t *strides = metadata + 3 + ndim;                                            \
        const size_t offset = metadata[3 + 2 * ndim];                                              \
                                                                                                   \
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
        TYPENAME det;                                                                              \
                                                                                                   \
        if (n == 1) {                                                                              \
            det = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);                      \
        } else if (n == 2) {                                                                       \
            TYPENAME a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
            TYPENAME b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
            TYPENAME c = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);               \
            TYPENAME d_val = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);           \
            det = a * d_val - b * c;                                                               \
        } else if (n == 3) {                                                                       \
            TYPENAME a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
            TYPENAME b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
            TYPENAME c = DET_GET(input, batch_offset, 0, 2, row_stride, col_stride);               \
            TYPENAME d_val = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);           \
            TYPENAME e = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);               \
            TYPENAME f = DET_GET(input, batch_offset, 1, 2, row_stride, col_stride);               \
            TYPENAME g = DET_GET(input, batch_offset, 2, 0, row_stride, col_stride);               \
            TYPENAME h = DET_GET(input, batch_offset, 2, 1, row_stride, col_stride);               \
            TYPENAME i = DET_GET(input, batch_offset, 2, 2, row_stride, col_stride);               \
            det = a * (e * i - f * h) - b * (d_val * i - f * g) + c * (d_val * h - e * g);         \
        } else {                                                                                   \
            /* LU decomposition with partial pivoting (thread-local storage) */                    \
            TYPENAME lu[MAX_DET_SIZE * MAX_DET_SIZE];                                              \
                                                                                                   \
            /* Copy input to local LU matrix */                                                    \
            for (size_t i = 0; i < n; i++) {                                                       \
                for (size_t j = 0; j < n; j++) {                                                   \
                    lu[i * n + j] = DET_GET(input, batch_offset, i, j, row_stride, col_stride);    \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            int swaps = 0;                                                                         \
            det = 1;                                                                               \
                                                                                                   \
            for (size_t k = 0; k < n; k++) {                                                       \
                /* Find pivot (use float for abs to avoid ambiguity) */                            \
                size_t pivot_row = k;                                                              \
                float max_val = fabs((float)lu[k * n + k]);                                        \
                                                                                                   \
                for (size_t i = k + 1; i < n; i++) {                                               \
                    float val = fabs((float)lu[i * n + k]);                                        \
                    if (val > max_val) {                                                           \
                        max_val = val;                                                             \
                        pivot_row = i;                                                             \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                /* Swap rows if needed */                                                          \
                if (pivot_row != k) {                                                              \
                    for (size_t j = 0; j < n; j++) {                                               \
                        TYPENAME tmp = lu[k * n + j];                                              \
                        lu[k * n + j] = lu[pivot_row * n + j];                                     \
                        lu[pivot_row * n + j] = tmp;                                               \
                    }                                                                              \
                    swaps++;                                                                       \
                }                                                                                  \
                                                                                                   \
                TYPENAME pivot = lu[k * n + k];                                                    \
                                                                                                   \
                /* Check for singular matrix */                                                    \
                if (fabs((float)pivot) < 1e-15f) {                                                 \
                    det = 0;                                                                       \
                    break;                                                                         \
                }                                                                                  \
                                                                                                   \
                det *= pivot;                                                                      \
                                                                                                   \
                /* Eliminate below */                                                              \
                for (size_t i = k + 1; i < n; i++) {                                               \
                    TYPENAME factor = lu[i * n + k] / pivot;                                       \
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
        output[tid] = det;                                                                         \
    }

// Integer types use abs() directly (unambiguous for integers)
#define DET_OP_INT(TYPENAME, FN_NAME)                                                              \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t *metadata [[buffer(2)]], uint tid [[thread_position_in_grid]]) {           \
                                                                                                   \
        const size_t batch_size = metadata[0];                                                     \
        const size_t n = metadata[1];                                                              \
        const size_t ndim = metadata[2];                                                           \
        constant size_t *shape = metadata + 3;                                                     \
        constant size_t *strides = metadata + 3 + ndim;                                            \
        const size_t offset = metadata[3 + 2 * ndim];                                              \
                                                                                                   \
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
        TYPENAME det;                                                                              \
                                                                                                   \
        if (n == 1) {                                                                              \
            det = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);                      \
        } else if (n == 2) {                                                                       \
            TYPENAME a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
            TYPENAME b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
            TYPENAME c = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);               \
            TYPENAME d_val = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);           \
            det = a * d_val - b * c;                                                               \
        } else if (n == 3) {                                                                       \
            TYPENAME a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
            TYPENAME b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
            TYPENAME c = DET_GET(input, batch_offset, 0, 2, row_stride, col_stride);               \
            TYPENAME d_val = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);           \
            TYPENAME e = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);               \
            TYPENAME f = DET_GET(input, batch_offset, 1, 2, row_stride, col_stride);               \
            TYPENAME g = DET_GET(input, batch_offset, 2, 0, row_stride, col_stride);               \
            TYPENAME h = DET_GET(input, batch_offset, 2, 1, row_stride, col_stride);               \
            TYPENAME i = DET_GET(input, batch_offset, 2, 2, row_stride, col_stride);               \
            det = a * (e * i - f * h) - b * (d_val * i - f * g) + c * (d_val * h - e * g);         \
        } else {                                                                                   \
            /* LU decomposition with partial pivoting (thread-local storage) */                    \
            TYPENAME lu[MAX_DET_SIZE * MAX_DET_SIZE];                                              \
                                                                                                   \
            /* Copy input to local LU matrix */                                                    \
            for (size_t i = 0; i < n; i++) {                                                       \
                for (size_t j = 0; j < n; j++) {                                                   \
                    lu[i * n + j] = DET_GET(input, batch_offset, i, j, row_stride, col_stride);    \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            int swaps = 0;                                                                         \
            det = 1;                                                                               \
                                                                                                   \
            for (size_t k = 0; k < n; k++) {                                                       \
                /* Find pivot */                                                                   \
                size_t pivot_row = k;                                                              \
                TYPENAME max_val = lu[k * n + k] < 0 ? -lu[k * n + k] : lu[k * n + k];             \
                                                                                                   \
                for (size_t i = k + 1; i < n; i++) {                                               \
                    TYPENAME val = lu[i * n + k] < 0 ? -lu[i * n + k] : lu[i * n + k];             \
                    if (val > max_val) {                                                           \
                        max_val = val;                                                             \
                        pivot_row = i;                                                             \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                /* Swap rows if needed */                                                          \
                if (pivot_row != k) {                                                              \
                    for (size_t j = 0; j < n; j++) {                                               \
                        TYPENAME tmp = lu[k * n + j];                                              \
                        lu[k * n + j] = lu[pivot_row * n + j];                                     \
                        lu[pivot_row * n + j] = tmp;                                               \
                    }                                                                              \
                    swaps++;                                                                       \
                }                                                                                  \
                                                                                                   \
                TYPENAME pivot = lu[k * n + k];                                                    \
                                                                                                   \
                /* Check for singular matrix (pivot == 0 for integers) */                          \
                if (pivot == 0) {                                                                  \
                    det = 0;                                                                       \
                    break;                                                                         \
                }                                                                                  \
                                                                                                   \
                det *= pivot;                                                                      \
                                                                                                   \
                /* Eliminate below (integer division) */                                           \
                for (size_t i = k + 1; i < n; i++) {                                               \
                    TYPENAME factor = lu[i * n + k] / pivot;                                       \
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
        output[tid] = det;                                                                         \
    }

// Define det operations for float types (use fabs with float cast)
DET_OP_FLOAT(bfloat, det_bf16)
DET_OP_FLOAT(half, det_f16)
DET_OP_FLOAT(float, det_f32)

// Define det operations for integer types (use ternary for abs)
DET_OP_INT(int8_t, det_i8)
DET_OP_INT(int16_t, det_i16)
DET_OP_INT(int32_t, det_i32)
DET_OP_INT(int64_t, det_i64)
DET_OP_INT(uint8_t, det_u8)
DET_OP_INT(uint16_t, det_u16)
DET_OP_INT(uint32_t, det_u32)
DET_OP_INT(uint64_t, det_u64)

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

// Float types use fabs() with float cast to avoid ambiguity
#define INV_OP_FLOAT(TYPENAME, FN_NAME)                                                            \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t *metadata [[buffer(2)]], uint tid [[thread_position_in_grid]]) {           \
                                                                                                   \
        const size_t batch_size = metadata[0];                                                     \
        const size_t n = metadata[1];                                                              \
        const size_t ndim = metadata[2];                                                           \
        constant size_t *shape = metadata + 3;                                                     \
        constant size_t *strides = metadata + 3 + ndim;                                            \
        const size_t offset = metadata[3 + 2 * ndim];                                              \
                                                                                                   \
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
            TYPENAME a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
            INV_SET(output, out_batch_offset, 0, 0, n, (TYPENAME)1 / a);                           \
        } else if (n == 2) {                                                                       \
            TYPENAME a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
            TYPENAME b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
            TYPENAME c = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);               \
            TYPENAME d_val = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);           \
            TYPENAME det = a * d_val - b * c;                                                      \
            TYPENAME inv_det = (TYPENAME)1 / det;                                                  \
            INV_SET(output, out_batch_offset, 0, 0, n, d_val * inv_det);                           \
            INV_SET(output, out_batch_offset, 0, 1, n, -b * inv_det);                              \
            INV_SET(output, out_batch_offset, 1, 0, n, -c * inv_det);                              \
            INV_SET(output, out_batch_offset, 1, 1, n, a * inv_det);                               \
        } else if (n == 3) {                                                                       \
            TYPENAME a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
            TYPENAME b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
            TYPENAME cv = DET_GET(input, batch_offset, 0, 2, row_stride, col_stride);              \
            TYPENAME d_val = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);           \
            TYPENAME e = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);               \
            TYPENAME f = DET_GET(input, batch_offset, 1, 2, row_stride, col_stride);               \
            TYPENAME g = DET_GET(input, batch_offset, 2, 0, row_stride, col_stride);               \
            TYPENAME h = DET_GET(input, batch_offset, 2, 1, row_stride, col_stride);               \
            TYPENAME iv = DET_GET(input, batch_offset, 2, 2, row_stride, col_stride);              \
                                                                                                   \
            TYPENAME det =                                                                         \
                a * (e * iv - f * h) - b * (d_val * iv - f * g) + cv * (d_val * h - e * g);        \
            TYPENAME inv_det = (TYPENAME)1 / det;                                                  \
                                                                                                   \
            INV_SET(output, out_batch_offset, 0, 0, n, (e * iv - f * h) * inv_det);                \
            INV_SET(output, out_batch_offset, 0, 1, n, (cv * h - b * iv) * inv_det);               \
            INV_SET(output, out_batch_offset, 0, 2, n, (b * f - cv * e) * inv_det);                \
            INV_SET(output, out_batch_offset, 1, 0, n, (f * g - d_val * iv) * inv_det);            \
            INV_SET(output, out_batch_offset, 1, 1, n, (a * iv - cv * g) * inv_det);               \
            INV_SET(output, out_batch_offset, 1, 2, n, (cv * d_val - a * f) * inv_det);            \
            INV_SET(output, out_batch_offset, 2, 0, n, (d_val * h - e * g) * inv_det);             \
            INV_SET(output, out_batch_offset, 2, 1, n, (b * g - a * h) * inv_det);                 \
            INV_SET(output, out_batch_offset, 2, 2, n, (a * e - b * d_val) * inv_det);             \
        } else {                                                                                   \
            /* Gauss-Jordan elimination (thread-local storage) */                                  \
            TYPENAME aug[MAX_INV_SIZE * MAX_INV_SIZE * 2];                                         \
                                                                                                   \
            /* Initialize augmented matrix [A | I] */                                              \
            for (size_t ii = 0; ii < n; ii++) {                                                    \
                for (size_t jj = 0; jj < n; jj++) {                                                \
                    aug[ii * 2 * n + jj] =                                                         \
                        DET_GET(input, batch_offset, ii, jj, row_stride, col_stride);              \
                    aug[ii * 2 * n + n + jj] = (ii == jj) ? (TYPENAME)1 : (TYPENAME)0;             \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t k = 0; k < n; k++) {                                                       \
                /* Find pivot */                                                                   \
                size_t pivot_row = k;                                                              \
                float max_val = fabs((float)aug[k * 2 * n + k]);                                   \
                                                                                                   \
                for (size_t ii = k + 1; ii < n; ii++) {                                            \
                    float val = fabs((float)aug[ii * 2 * n + k]);                                  \
                    if (val > max_val) {                                                           \
                        max_val = val;                                                             \
                        pivot_row = ii;                                                            \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                /* Swap rows if needed */                                                          \
                if (pivot_row != k) {                                                              \
                    for (size_t jj = 0; jj < 2 * n; jj++) {                                        \
                        TYPENAME tmp = aug[k * 2 * n + jj];                                        \
                        aug[k * 2 * n + jj] = aug[pivot_row * 2 * n + jj];                         \
                        aug[pivot_row * 2 * n + jj] = tmp;                                         \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                /* Scale pivot row */                                                              \
                TYPENAME pivot = aug[k * 2 * n + k];                                               \
                for (size_t jj = 0; jj < 2 * n; jj++) {                                            \
                    aug[k * 2 * n + jj] /= pivot;                                                  \
                }                                                                                  \
                                                                                                   \
                /* Eliminate column k in other rows */                                             \
                for (size_t ii = 0; ii < n; ii++) {                                                \
                    if (ii != k) {                                                                 \
                        TYPENAME factor = aug[ii * 2 * n + k];                                     \
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
                    INV_SET(output, out_batch_offset, ii, jj, n, aug[ii * 2 * n + n + jj]);        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

// Integer types - inverse with integer division (limited precision)
#define INV_OP_INT(TYPENAME, FN_NAME)                                                              \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t *metadata [[buffer(2)]], uint tid [[thread_position_in_grid]]) {           \
                                                                                                   \
        const size_t batch_size = metadata[0];                                                     \
        const size_t n = metadata[1];                                                              \
        const size_t ndim = metadata[2];                                                           \
        constant size_t *shape = metadata + 3;                                                     \
        constant size_t *strides = metadata + 3 + ndim;                                            \
        const size_t offset = metadata[3 + 2 * ndim];                                              \
                                                                                                   \
        if (tid >= batch_size)                                                                     \
            return;                                                                                \
                                                                                                   \
        const size_t row_stride = (ndim >= 2) ? strides[ndim - 2] : n;                             \
        const size_t col_stride = (ndim >= 1) ? strides[ndim - 1] : 1;                             \
                                                                                                   \
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
            TYPENAME a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
            INV_SET(output, out_batch_offset, 0, 0, n, (TYPENAME)1 / a);                           \
        } else if (n == 2) {                                                                       \
            TYPENAME a = DET_GET(input, batch_offset, 0, 0, row_stride, col_stride);               \
            TYPENAME b = DET_GET(input, batch_offset, 0, 1, row_stride, col_stride);               \
            TYPENAME c = DET_GET(input, batch_offset, 1, 0, row_stride, col_stride);               \
            TYPENAME d_val = DET_GET(input, batch_offset, 1, 1, row_stride, col_stride);           \
            TYPENAME det = a * d_val - b * c;                                                      \
            INV_SET(output, out_batch_offset, 0, 0, n, d_val / det);                               \
            INV_SET(output, out_batch_offset, 0, 1, n, -b / det);                                  \
            INV_SET(output, out_batch_offset, 1, 0, n, -c / det);                                  \
            INV_SET(output, out_batch_offset, 1, 1, n, a / det);                                   \
        } else {                                                                                   \
            /* For larger integer matrices, use float and convert back */                          \
            float aug[MAX_INV_SIZE * MAX_INV_SIZE * 2];                                            \
                                                                                                   \
            for (size_t ii = 0; ii < n; ii++) {                                                    \
                for (size_t jj = 0; jj < n; jj++) {                                                \
                    aug[ii * 2 * n + jj] =                                                         \
                        (float)DET_GET(input, batch_offset, ii, jj, row_stride, col_stride);       \
                    aug[ii * 2 * n + n + jj] = (ii == jj) ? 1.0f : 0.0f;                           \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            for (size_t k = 0; k < n; k++) {                                                       \
                size_t pivot_row = k;                                                              \
                float max_val = fabs(aug[k * 2 * n + k]);                                          \
                                                                                                   \
                for (size_t ii = k + 1; ii < n; ii++) {                                            \
                    float val = fabs(aug[ii * 2 * n + k]);                                         \
                    if (val > max_val) {                                                           \
                        max_val = val;                                                             \
                        pivot_row = ii;                                                            \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                if (pivot_row != k) {                                                              \
                    for (size_t jj = 0; jj < 2 * n; jj++) {                                        \
                        float tmp = aug[k * 2 * n + jj];                                           \
                        aug[k * 2 * n + jj] = aug[pivot_row * 2 * n + jj];                         \
                        aug[pivot_row * 2 * n + jj] = tmp;                                         \
                    }                                                                              \
                }                                                                                  \
                                                                                                   \
                float pivot = aug[k * 2 * n + k];                                                  \
                for (size_t jj = 0; jj < 2 * n; jj++) {                                            \
                    aug[k * 2 * n + jj] /= pivot;                                                  \
                }                                                                                  \
                                                                                                   \
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
            for (size_t ii = 0; ii < n; ii++) {                                                    \
                for (size_t jj = 0; jj < n; jj++) {                                                \
                    INV_SET(output, out_batch_offset, ii, jj, n,                                   \
                            (TYPENAME)aug[ii * 2 * n + n + jj]);                                   \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

// Define inv operations for float types
INV_OP_FLOAT(bfloat, inv_bf16)
INV_OP_FLOAT(half, inv_f16)
INV_OP_FLOAT(float, inv_f32)

// Define inv operations for integer types
INV_OP_INT(int8_t, inv_i8)
INV_OP_INT(int16_t, inv_i16)
INV_OP_INT(int32_t, inv_i32)
INV_OP_INT(int64_t, inv_i64)
INV_OP_INT(uint8_t, inv_u8)
INV_OP_INT(uint16_t, inv_u16)
INV_OP_INT(uint32_t, inv_u32)
INV_OP_INT(uint64_t, inv_u64)

// ============================================================================
// MATRIX TRACE
// ============================================================================
//
// Computes the trace (sum of diagonal elements) of square matrices with batch support.
// Input: [..., N, N] -> Output: [...]
//
// Metadata layout (same as det/inv):
// - metadata[0]: batch_size (product of batch dimensions)
// - metadata[1]: n (matrix size, N×N)
// - metadata[2]: ndim (total number of dimensions)
// - metadata[3..3+ndim]: shape
// - metadata[3+ndim..3+2*ndim]: strides
// - metadata[3+2*ndim]: offset

#define TRACE_OP(TYPENAME, FN_NAME)                                                                \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t *metadata [[buffer(2)]], uint tid [[thread_position_in_grid]]) {           \
                                                                                                   \
        const size_t batch_size = metadata[0];                                                     \
        const size_t n = metadata[1];                                                              \
        const size_t ndim = metadata[2];                                                           \
        constant size_t *shape = metadata + 3;                                                     \
        constant size_t *strides = metadata + 3 + ndim;                                            \
        const size_t offset = metadata[3 + 2 * ndim];                                              \
                                                                                                   \
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
        /* Sum diagonal elements */                                                                \
        TYPENAME trace = 0;                                                                        \
        for (size_t i = 0; i < n; i++) {                                                           \
            trace += DET_GET(input, batch_offset, i, i, row_stride, col_stride);                   \
        }                                                                                          \
                                                                                                   \
        output[tid] = trace;                                                                       \
    }

// Define trace operations for float types
TRACE_OP(bfloat, trace_bf16)
TRACE_OP(half, trace_f16)
TRACE_OP(float, trace_f32)

// Define trace operations for integer types
TRACE_OP(int8_t, trace_i8)
TRACE_OP(int16_t, trace_i16)
TRACE_OP(int32_t, trace_i32)
TRACE_OP(int64_t, trace_i64)
TRACE_OP(uint8_t, trace_u8)
TRACE_OP(uint16_t, trace_u16)
TRACE_OP(uint32_t, trace_u32)
TRACE_OP(uint64_t, trace_u64)
