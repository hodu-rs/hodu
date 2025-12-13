/**
 * @file ops_linalg.h
 * @brief Linear algebra operations header
 *
 * Provides linear algebra operations:
 * - det: Matrix determinant computation
 * - inv: Matrix inverse computation
 * - trace: Matrix trace computation (sum of diagonal)
 */

#ifndef OPS_LINALG_H
#define OPS_LINALG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MATRIX DETERMINANT (DET)
// ============================================================================
//
// Computes the determinant of square matrices with optional batch dimensions.
//
// All det operations follow this signature:
//   void hodu_cpu_det_type(const void *input, void *output, const size_t *metadata)
//
// Parameters:
//   input    - Pointer to input tensor data (square matrix)
//   output   - Pointer to output tensor buffer (scalar per batch)
//   metadata - Array describing operation (see below)
//
// Metadata layout:
// - metadata[0]: batch_size (product of batch dimensions)
// - metadata[1]: n (matrix size, N×N)
// - metadata[2]: ndim (total number of dimensions)
// - metadata[3..3+ndim]: shape
// - metadata[3+ndim..3+2*ndim]: strides
// - metadata[3+2*ndim]: offset

void hodu_cpu_det_f8e4m3(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_f8e5m2(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_bf16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_f16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_f32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_f64(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_i8(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_i16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_i32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_i64(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_u8(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_u16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_u32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_det_u64(const void *input, void *output, const size_t *metadata);

// ============================================================================
// MATRIX INVERSE (INV)
// ============================================================================
//
// Computes the inverse of square matrices with optional batch dimensions.
// Uses Gauss-Jordan elimination with partial pivoting for numerical stability.
//
// All inv operations follow this signature:
//   void hodu_cpu_inv_type(const void *input, void *output, const size_t *metadata)
//
// Parameters:
//   input    - Pointer to input tensor data (square matrix)
//   output   - Pointer to output tensor buffer (same shape as input)
//   metadata - Array describing operation (see below)
//
// Metadata layout (same as det):
// - metadata[0]: batch_size (product of batch dimensions)
// - metadata[1]: n (matrix size, N×N)
// - metadata[2]: ndim (total number of dimensions)
// - metadata[3..3+ndim]: shape
// - metadata[3+ndim..3+2*ndim]: strides
// - metadata[3+2*ndim]: offset
//
// Note: For singular matrices, the output will contain inf/nan values.

void hodu_cpu_inv_f8e4m3(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_f8e5m2(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_bf16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_f16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_f32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_f64(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_i8(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_i16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_i32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_i64(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_u8(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_u16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_u32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_inv_u64(const void *input, void *output, const size_t *metadata);

// ============================================================================
// MATRIX TRACE
// ============================================================================
//
// Computes the trace (sum of diagonal elements) of square matrices.
//
// All trace operations follow this signature:
//   void hodu_cpu_trace_type(const void *input, void *output, const size_t *metadata)
//
// Parameters:
//   input    - Pointer to input tensor data (square matrix)
//   output   - Pointer to output tensor buffer (scalar per batch)
//   metadata - Array describing operation (same as det/inv)
//
// Metadata layout (same as det/inv):
// - metadata[0]: batch_size (product of batch dimensions)
// - metadata[1]: n (matrix size, N×N)
// - metadata[2]: ndim (total number of dimensions)
// - metadata[3..3+ndim]: shape
// - metadata[3+ndim..3+2*ndim]: strides
// - metadata[3+2*ndim]: offset

void hodu_cpu_trace_f8e4m3(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_f8e5m2(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_bf16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_f16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_f32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_f64(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_i8(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_i16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_i32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_i64(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_u8(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_u16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_u32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_trace_u64(const void *input, void *output, const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // OPS_LINALG_H
