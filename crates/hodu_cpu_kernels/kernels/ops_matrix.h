/**
 * @file ops_matrix.h
 * @brief Matrix multiplication operations header
 *
 * Provides matrix multiplication operations:
 * - matmul: Batched matrix multiplication with broadcasting
 * - dot: Simple 2D matrix multiplication
 *
 * Both operations implement C = A @ B for compatible matrix dimensions.
 */

#ifndef OPS_MATRIX_H
#define OPS_MATRIX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// BATCHED MATRIX MULTIPLICATION (MATMUL)
// ============================================================================
//
// Performs batched matrix multiplication with broadcasting support.
//
// All matmul operations follow this signature:
//   void hodu_cpu_matmul_type(const void *lhs, const void *rhs, void *output, const size_t
//   *metadata)
//
// Parameters:
//   lhs      - Pointer to left-hand side tensor data
//   rhs      - Pointer to right-hand side tensor data
//   output   - Pointer to output tensor buffer (pre-allocated)
//   metadata - Array describing operation (see below)
//
// Metadata layout:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: lhs_ndim (number of dimensions in lhs)
// - metadata[2]: rhs_ndim (number of dimensions in rhs)
// - metadata[3]: batch_ndim (number of batch dimensions in output)
// - metadata[4..4+lhs_ndim]: lhs_shape
// - metadata[4+lhs_ndim..4+lhs_ndim+rhs_ndim]: rhs_shape
// - metadata[4+lhs_ndim+rhs_ndim..4+lhs_ndim+rhs_ndim+batch_ndim]: batch_shape
// - metadata[4+lhs_ndim+rhs_ndim+batch_ndim..4+2*lhs_ndim+rhs_ndim+batch_ndim]: lhs_strides
// - metadata[4+2*lhs_ndim+rhs_ndim+batch_ndim..4+2*lhs_ndim+2*rhs_ndim+batch_ndim]: rhs_strides
// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim]: lhs_offset
// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+1]: rhs_offset
// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+2]: M (rows of lhs matrix)
// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+3]: K (cols of lhs / rows of rhs)
// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+4]: N (cols of rhs matrix)
//
// Broadcasting:
// Batch dimensions of size 1 are automatically broadcast to match the output shape.

void hodu_cpu_matmul_f8e4m3(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_f8e5m2(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_bf16(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_f16(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_f32(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_f64(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_i8(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_i16(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_i32(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_i64(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_u8(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_u16(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_u32(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_matmul_u64(const void *lhs, const void *rhs, void *output, const size_t *metadata);

// ============================================================================
// 2D MATRIX MULTIPLICATION (DOT)
// ============================================================================
//
// Performs simple 2D matrix multiplication without batching.
//
// All dot operations follow this signature:
//   void hodu_cpu_dot_type(const void *lhs, const void *rhs, void *output, const size_t *metadata)
//
// Parameters:
//   lhs      - Pointer to left-hand side matrix data (2D)
//   rhs      - Pointer to right-hand side matrix data (2D)
//   output   - Pointer to output matrix buffer (pre-allocated, 2D)
//   metadata - Array describing operation (see below)
//
// Metadata layout:
// - metadata[0]: M (number of rows in lhs)
// - metadata[1]: K (number of cols in lhs / rows in rhs)
// - metadata[2]: N (number of cols in rhs)
// - metadata[3]: lhs_stride_m (stride for lhs rows)
// - metadata[4]: lhs_stride_k (stride for lhs cols)
// - metadata[5]: rhs_stride_k (stride for rhs rows)
// - metadata[6]: rhs_stride_n (stride for rhs cols)
// - metadata[7]: lhs_offset (starting offset in lhs)
// - metadata[8]: rhs_offset (starting offset in rhs)

void hodu_cpu_dot_f8e4m3(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_f8e5m2(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_bf16(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_f16(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_f32(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_f64(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_i8(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_i16(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_i32(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_i64(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_u8(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_u16(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_u32(const void *lhs, const void *rhs, void *output, const size_t *metadata);
void hodu_cpu_dot_u64(const void *lhs, const void *rhs, void *output, const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // OPS_MATRIX_H
