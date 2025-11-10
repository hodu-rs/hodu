# hodu_cuda_kernels

High-performance CUDA kernels for tensor operations on NVIDIA GPUs.

## cuBLAS Integration

### Supported Operations
- **matmul**: Batched matrix multiplication with GEMM
- **dot**: 2D matrix multiplication with GEMM

### Supported Data Types
- **bf16**: BFloat16 (compute in FP32, I/O in BF16)
- **f16**: Float16/Half (compute in FP32, I/O in FP16)
- **f32**: Float32 (native precision)
- **f64**: Float64 (native precision)

### Features
- Automatic fallback to custom CUDA kernels for unsupported types or non-contiguous matrices
- Handles non-contiguous matrices via leading dimension parameters
- Transparent row-major to column-major layout conversion
