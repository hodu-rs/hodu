#include "ops_matrix.h"
#include "types.h"
#include <stdbool.h>
#include <stdint.h>

#include <cblas_new.h>

// Forward declarations for fallback implementations
extern void matmul_f32_fallback(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,
                                const size_t *metadata);
extern void matmul_f64_fallback(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,
                                const size_t *metadata);
extern void dot_f32_fallback(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,
                             const size_t *metadata);
extern void dot_f64_fallback(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,
                             const size_t *metadata);

/// F32 matmul using Accelerate cblas_sgemm
void matmul_f32(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,
                const size_t *metadata) {
    const f32_t *lhs = (const f32_t *)lhs_ptr;
    const f32_t *rhs = (const f32_t *)rhs_ptr;
    f32_t *output = (f32_t *)output_ptr;

    const size_t lhs_ndim = metadata[1];
    const size_t rhs_ndim = metadata[2];
    const size_t batch_ndim = metadata[3];

    const size_t *lhs_shape = metadata + 4;
    const size_t *rhs_shape = lhs_shape + lhs_ndim;
    const size_t *batch_shape = rhs_shape + rhs_ndim;
    const size_t *lhs_strides = batch_shape + batch_ndim;
    const size_t *rhs_strides = lhs_strides + lhs_ndim;
    const size_t lhs_offset = *(rhs_strides + rhs_ndim);
    const size_t rhs_offset = *(rhs_strides + rhs_ndim + 1);
    const size_t M = *(rhs_strides + rhs_ndim + 2);
    const size_t K = *(rhs_strides + rhs_ndim + 3);
    const size_t N = *(rhs_strides + rhs_ndim + 4);

    // Apply offsets
    lhs += lhs_offset;
    rhs += rhs_offset;

    // Determine matrix layouts and transpose flags
    // Row-major: innermost (last) stride is 1
    // Col-major: outermost (second-to-last) stride is 1
    bool lhs_row_major = (lhs_strides[lhs_ndim - 1] == 1);
    bool lhs_col_major = (lhs_strides[lhs_ndim - 2] == 1);
    bool rhs_row_major = (rhs_strides[rhs_ndim - 1] == 1);
    bool rhs_col_major = (rhs_strides[rhs_ndim - 2] == 1);

    // Can use BLAS if either row-major or col-major (transposed)
    if ((lhs_row_major || lhs_col_major) && (rhs_row_major || rhs_col_major)) {
        enum CBLAS_TRANSPOSE trans_lhs = lhs_row_major ? CblasNoTrans : CblasTrans;
        enum CBLAS_TRANSPOSE trans_rhs = rhs_row_major ? CblasNoTrans : CblasTrans;

        // Leading dimensions
        size_t lda = lhs_row_major ? lhs_strides[lhs_ndim - 2] : lhs_strides[lhs_ndim - 1];
        size_t ldb = rhs_row_major ? rhs_strides[rhs_ndim - 2] : rhs_strides[rhs_ndim - 1];
        size_t ldc = N;

        // Fix leading dimensions when they're too small
        // For row-major: lda >= K, ldb >= N
        // For col-major: lda >= M, ldb >= K
        if (trans_lhs == CblasNoTrans) {
            if (lda < K)
                lda = K;
        } else {
            if (lda < M)
                lda = M;
        }
        if (trans_rhs == CblasNoTrans) {
            if (ldb < N)
                ldb = N;
        } else {
            if (ldb < K)
                ldb = K;
        }

        if (batch_ndim == 0) {
            // No batching - single BLAS call
            cblas_sgemm(CblasRowMajor, trans_lhs, trans_rhs, M, N, K, 1.0f, lhs, lda, rhs, ldb,
                        0.0f, output, ldc);
        } else {
            // Batched case - iterate over all batch indices
            size_t total_batches = 1;
            for (size_t i = 0; i < batch_ndim; i++) {
                total_batches *= batch_shape[i];
            }

            size_t out_batch_stride = M * N;

            for (size_t batch_idx = 0; batch_idx < total_batches; batch_idx++) {
                // Compute multi-dimensional batch index
                size_t idx = batch_idx;
                size_t lhs_batch_offset = 0;
                size_t rhs_batch_offset = 0;

                for (size_t i = batch_ndim; i > 0; i--) {
                    size_t dim_idx = idx % batch_shape[i - 1];
                    idx /= batch_shape[i - 1];

                    // Apply broadcasting: if shape is 1, use index 0
                    size_t lhs_dim_idx = (lhs_shape[lhs_ndim - 2 - i] == 1) ? 0 : dim_idx;
                    size_t rhs_dim_idx = (rhs_shape[rhs_ndim - 2 - i] == 1) ? 0 : dim_idx;

                    lhs_batch_offset += lhs_dim_idx * lhs_strides[lhs_ndim - 2 - i];
                    rhs_batch_offset += rhs_dim_idx * rhs_strides[rhs_ndim - 2 - i];
                }

                const f32_t *lhs_batch = lhs + lhs_batch_offset;
                const f32_t *rhs_batch = rhs + rhs_batch_offset;
                f32_t *out_batch = output + batch_idx * out_batch_stride;

                cblas_sgemm(CblasRowMajor, trans_lhs, trans_rhs, M, N, K, 1.0f, lhs_batch, lda,
                            rhs_batch, ldb, 0.0f, out_batch, ldc);
            }
        }
    } else {
        // Fallback for complex stride patterns
        matmul_f32_fallback(lhs_ptr, rhs_ptr, output_ptr, metadata);
    }
}

/// F64 matmul using Accelerate cblas_dgemm
void matmul_f64(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr,
                const size_t *metadata) {
    const f64_t *lhs = (const f64_t *)lhs_ptr;
    const f64_t *rhs = (const f64_t *)rhs_ptr;
    f64_t *output = (f64_t *)output_ptr;

    const size_t lhs_ndim = metadata[1];
    const size_t rhs_ndim = metadata[2];
    const size_t batch_ndim = metadata[3];

    const size_t *lhs_shape = metadata + 4;
    const size_t *rhs_shape = lhs_shape + lhs_ndim;
    const size_t *batch_shape = rhs_shape + rhs_ndim;
    const size_t *lhs_strides = batch_shape + batch_ndim;
    const size_t *rhs_strides = lhs_strides + lhs_ndim;
    const size_t lhs_offset = *(rhs_strides + rhs_ndim);
    const size_t rhs_offset = *(rhs_strides + rhs_ndim + 1);
    const size_t M = *(rhs_strides + rhs_ndim + 2);
    const size_t K = *(rhs_strides + rhs_ndim + 3);
    const size_t N = *(rhs_strides + rhs_ndim + 4);

    // Apply offsets
    lhs += lhs_offset;
    rhs += rhs_offset;

    // Determine matrix layouts and transpose flags
    bool lhs_row_major = (lhs_strides[lhs_ndim - 1] == 1);
    bool lhs_col_major = (lhs_strides[lhs_ndim - 2] == 1);
    bool rhs_row_major = (rhs_strides[rhs_ndim - 1] == 1);
    bool rhs_col_major = (rhs_strides[rhs_ndim - 2] == 1);

    if ((lhs_row_major || lhs_col_major) && (rhs_row_major || rhs_col_major)) {
        enum CBLAS_TRANSPOSE trans_lhs = lhs_row_major ? CblasNoTrans : CblasTrans;
        enum CBLAS_TRANSPOSE trans_rhs = rhs_row_major ? CblasNoTrans : CblasTrans;

        size_t lda = lhs_row_major ? lhs_strides[lhs_ndim - 2] : lhs_strides[lhs_ndim - 1];
        size_t ldb = rhs_row_major ? rhs_strides[rhs_ndim - 2] : rhs_strides[rhs_ndim - 1];
        size_t ldc = N;

        // Fix leading dimensions when they're too small
        if (trans_lhs == CblasNoTrans) {
            if (lda < K)
                lda = K;
        } else {
            if (lda < M)
                lda = M;
        }
        if (trans_rhs == CblasNoTrans) {
            if (ldb < N)
                ldb = N;
        } else {
            if (ldb < K)
                ldb = K;
        }

        if (batch_ndim == 0) {
            // No batching - single BLAS call
            cblas_dgemm(CblasRowMajor, trans_lhs, trans_rhs, M, N, K, 1.0, lhs, lda, rhs, ldb, 0.0,
                        output, ldc);
        } else {
            // Batched case - iterate over all batch indices
            size_t total_batches = 1;
            for (size_t i = 0; i < batch_ndim; i++) {
                total_batches *= batch_shape[i];
            }

            size_t out_batch_stride = M * N;

            for (size_t batch_idx = 0; batch_idx < total_batches; batch_idx++) {
                // Compute multi-dimensional batch index
                size_t idx = batch_idx;
                size_t lhs_batch_offset = 0;
                size_t rhs_batch_offset = 0;

                for (size_t i = batch_ndim; i > 0; i--) {
                    size_t dim_idx = idx % batch_shape[i - 1];
                    idx /= batch_shape[i - 1];

                    // Apply broadcasting: if shape is 1, use index 0
                    size_t lhs_dim_idx = (lhs_shape[lhs_ndim - 2 - i] == 1) ? 0 : dim_idx;
                    size_t rhs_dim_idx = (rhs_shape[rhs_ndim - 2 - i] == 1) ? 0 : dim_idx;

                    lhs_batch_offset += lhs_dim_idx * lhs_strides[lhs_ndim - 2 - i];
                    rhs_batch_offset += rhs_dim_idx * rhs_strides[rhs_ndim - 2 - i];
                }

                const f64_t *lhs_batch = lhs + lhs_batch_offset;
                const f64_t *rhs_batch = rhs + rhs_batch_offset;
                f64_t *out_batch = output + batch_idx * out_batch_stride;

                cblas_dgemm(CblasRowMajor, trans_lhs, trans_rhs, M, N, K, 1.0, lhs_batch, lda,
                            rhs_batch, ldb, 0.0, out_batch, ldc);
            }
        }
    } else {
        // Fallback for complex stride patterns
        matmul_f64_fallback(lhs_ptr, rhs_ptr, output_ptr, metadata);
    }
}

/// F32 dot product using Accelerate cblas_sgemm
void dot_f32(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr, const size_t *metadata) {
    const f32_t *lhs = (const f32_t *)lhs_ptr;
    const f32_t *rhs = (const f32_t *)rhs_ptr;
    f32_t *output = (f32_t *)output_ptr;

    const size_t M = metadata[0];
    const size_t K = metadata[1];
    const size_t N = metadata[2];
    const size_t lhs_stride_m = metadata[3];
    const size_t lhs_stride_k = metadata[4];
    const size_t rhs_stride_k = metadata[5];
    const size_t rhs_stride_n = metadata[6];
    const size_t lhs_offset = metadata[7];
    const size_t rhs_offset = metadata[8];

    // Apply offsets
    lhs += lhs_offset;
    rhs += rhs_offset;

    // Determine matrix layouts and transpose flags
    bool lhs_row_major = (lhs_stride_k == 1);
    bool lhs_col_major = (lhs_stride_m == 1);
    bool rhs_row_major = (rhs_stride_n == 1);
    bool rhs_col_major = (rhs_stride_k == 1);

    if ((lhs_row_major || lhs_col_major) && (rhs_row_major || rhs_col_major)) {
        enum CBLAS_TRANSPOSE trans_lhs = lhs_row_major ? CblasNoTrans : CblasTrans;
        enum CBLAS_TRANSPOSE trans_rhs = rhs_row_major ? CblasNoTrans : CblasTrans;

        size_t lda = lhs_row_major ? lhs_stride_m : lhs_stride_k;
        size_t ldb = rhs_row_major ? rhs_stride_k : rhs_stride_n;

        if (trans_lhs == CblasNoTrans) {
            if (lda < K)
                lda = K;
        } else {
            if (lda < M)
                lda = M;
        }
        if (trans_rhs == CblasNoTrans) {
            if (ldb < N)
                ldb = N;
        } else {
            if (ldb < K)
                ldb = K;
        }

        cblas_sgemm(CblasRowMajor, trans_lhs, trans_rhs, M, N, K, 1.0f, lhs, lda, rhs, ldb, 0.0f,
                    output, N);
    } else {
        // Fallback for complex stride patterns
        dot_f32_fallback(lhs_ptr, rhs_ptr, output_ptr, metadata);
    }
}

/// F64 dot product using Accelerate cblas_dgemm
void dot_f64(const void *lhs_ptr, const void *rhs_ptr, void *output_ptr, const size_t *metadata) {
    const f64_t *lhs = (const f64_t *)lhs_ptr;
    const f64_t *rhs = (const f64_t *)rhs_ptr;
    f64_t *output = (f64_t *)output_ptr;

    const size_t M = metadata[0];
    const size_t K = metadata[1];
    const size_t N = metadata[2];
    const size_t lhs_stride_m = metadata[3];
    const size_t lhs_stride_k = metadata[4];
    const size_t rhs_stride_k = metadata[5];
    const size_t rhs_stride_n = metadata[6];
    const size_t lhs_offset = metadata[7];
    const size_t rhs_offset = metadata[8];

    // Apply offsets
    lhs += lhs_offset;
    rhs += rhs_offset;

    // Determine matrix layouts and transpose flags
    bool lhs_row_major = (lhs_stride_k == 1);
    bool lhs_col_major = (lhs_stride_m == 1);
    bool rhs_row_major = (rhs_stride_n == 1);
    bool rhs_col_major = (rhs_stride_k == 1);

    if ((lhs_row_major || lhs_col_major) && (rhs_row_major || rhs_col_major)) {
        enum CBLAS_TRANSPOSE trans_lhs = lhs_row_major ? CblasNoTrans : CblasTrans;
        enum CBLAS_TRANSPOSE trans_rhs = rhs_row_major ? CblasNoTrans : CblasTrans;

        size_t lda = lhs_row_major ? lhs_stride_m : lhs_stride_k;
        size_t ldb = rhs_row_major ? rhs_stride_k : rhs_stride_n;

        // Fix leading dimensions when they're too small
        if (trans_lhs == CblasNoTrans) {
            if (lda < K)
                lda = K;
        } else {
            if (lda < M)
                lda = M;
        }
        if (trans_rhs == CblasNoTrans) {
            if (ldb < N)
                ldb = N;
        } else {
            if (ldb < K)
                ldb = K;
        }

        cblas_dgemm(CblasRowMajor, trans_lhs, trans_rhs, M, N, K, 1.0, lhs, lda, rhs, ldb, 0.0,
                    output, N);
    } else {
        // Fallback for complex stride patterns
        dot_f64_fallback(lhs_ptr, rhs_ptr, output_ptr, metadata);
    }
}
