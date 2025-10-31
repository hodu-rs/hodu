#ifndef OPS_MATRIX_H
#define OPS_MATRIX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Batched matrix multiplication with broadcasting
// metadata layout: [lhs_ndim, rhs_ndim, batch_ndim, lhs_shape..., rhs_shape..., batch_shape...,
//                   lhs_strides..., rhs_strides..., lhs_offset, rhs_offset, M, K, N]
void matmul_f8e4m3(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                   const size_t *metadata);
void matmul_f8e5m2(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                   const size_t *metadata);
void matmul_bf16(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                 const size_t *metadata);
void matmul_f16(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void matmul_f32(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void matmul_f64(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void matmul_i8(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);
void matmul_i16(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void matmul_i32(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void matmul_i64(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void matmul_u8(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
               const size_t *metadata);
void matmul_u16(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void matmul_u32(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void matmul_u64(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);

// Tiled 2D dot product (simple version without threading)
// metadata layout: [M, K, N, lhs_stride_m, lhs_stride_k, rhs_stride_k, rhs_stride_n, lhs_offset,
// rhs_offset]
void dot_f8e4m3(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void dot_f8e5m2(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
                const size_t *metadata);
void dot_bf16(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
              const size_t *metadata);
void dot_f16(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
             const size_t *metadata);
void dot_f32(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
             const size_t *metadata);
void dot_f64(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
             const size_t *metadata);
void dot_i8(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
            const size_t *metadata);
void dot_i16(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
             const size_t *metadata);
void dot_i32(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
             const size_t *metadata);
void dot_i64(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
             const size_t *metadata);
void dot_u8(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
            const size_t *metadata);
void dot_u16(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
             const size_t *metadata);
void dot_u32(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
             const size_t *metadata);
void dot_u64(const void *lhs, const void *rhs, void *output, size_t num_els, size_t num_dims,
             const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // OPS_MATRIX_H
