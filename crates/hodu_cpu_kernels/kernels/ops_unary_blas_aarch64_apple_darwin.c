#include "ops_unary.h"
#include "types.h"
#include "utils.h"
#include <cblas_new.h>

// Forward declarations for fallback
extern void mul_scalar_f32_fallback(const void *input, void *output, const size_t *metadata,
                                    const void *scalar);
extern void mul_scalar_f64_fallback(const void *input, void *output, const size_t *metadata,
                                    const void *scalar);

// mul_scalar_f32: Accelerate BLAS-optimized version
void mul_scalar_f32(const void *input, void *output, const size_t *metadata, const void *scalar) {
    const size_t num_els = metadata[0];
    const size_t num_dims = metadata[1];
    const f32_t *in = (const f32_t *)input;
    f32_t *out = (f32_t *)output;
    f32_t const_val = *(const f32_t *)scalar;

    const size_t *dims = metadata + 2;
    const size_t *strides = metadata ? metadata + 2 + num_dims : NULL;
    const size_t offset = (metadata && num_dims > 0) ? metadata[2 + 2 * num_dims] : 0;

    bool contiguous = (metadata == NULL) || is_contiguous(num_dims, dims, strides);

    // BLAS path: Use cblas_sscal for in-place scalar multiplication
    if (contiguous && in == NULL) {
        // In-place operation
        cblas_sscal(num_els, const_val, out, 1);
        return;
    } else if (contiguous && in != NULL && offset == 0) {
        // Copy input to output, then scale in-place
        if (in != out) {
            cblas_scopy(num_els, in, 1, out, 1);
        }
        cblas_sscal(num_els, const_val, out, 1);
        return;
    }

    // Fallback for non-contiguous or offset cases
    mul_scalar_f32_fallback(input, output, metadata, scalar);
}

// mul_scalar_f64: Accelerate BLAS-optimized version
void mul_scalar_f64(const void *input, void *output, const size_t *metadata, const void *scalar) {
    const size_t num_els = metadata[0];
    const size_t num_dims = metadata[1];
    const f64_t *in = (const f64_t *)input;
    f64_t *out = (f64_t *)output;
    f64_t const_val = *(const f64_t *)scalar;

    const size_t *dims = metadata + 2;
    const size_t *strides = metadata ? metadata + 2 + num_dims : NULL;
    const size_t offset = (metadata && num_dims > 0) ? metadata[2 + 2 * num_dims] : 0;

    bool contiguous = (metadata == NULL) || is_contiguous(num_dims, dims, strides);

    // BLAS path: Use cblas_dscal for in-place scalar multiplication
    if (contiguous && in == NULL) {
        // In-place operation
        cblas_dscal(num_els, const_val, out, 1);
        return;
    } else if (contiguous && in != NULL && offset == 0) {
        // Copy input to output, then scale in-place
        if (in != out) {
            cblas_dcopy(num_els, in, 1, out, 1);
        }
        cblas_dscal(num_els, const_val, out, 1);
        return;
    }

    // Fallback for non-contiguous or offset cases
    mul_scalar_f64_fallback(input, output, metadata, scalar);
}
