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
