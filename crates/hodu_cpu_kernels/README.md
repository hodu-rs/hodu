# Hodu CPU Kernels

High-performance CPU kernels for tensor operations in pure C.

## Features

- **Data Types**: f8, bf16, f16, f32, f64, i8-i64, u8-u64, bool
- **Operations**: Unary, binary, reduce, matrix, convolution, pooling, indexing, concat/split
- **SIMD**: Auto-detected AVX2/SSE2 (x86_64), NEON (ARM) for f32/f64
- **BLAS**: Accelerate (macOS), OpenBLAS (opt-in via feature)
- **Multi-threading**: pthread parallelization for large operations

## Cargo Features

- `std` - Standard library support (enables multi-threading)
- `openblas` - Use OpenBLAS instead of OS-provided BLAS

## Environment Variables

- `HODU_DISABLE_NATIVE` - Disable `-march=native`
- `HODU_DISABLE_SIMD` - Disable SIMD vectorization
- `HODU_DISABLE_THREADS` - Disable multi-threading
- `OPENBLAS_DIR` / `OPENBLAS_INCLUDE_DIR` / `OPENBLAS_LIB_DIR` - Custom OpenBLAS path (for `openblas` feature)

## Examples

```bash
# macOS: Uses Accelerate by default
cargo build --release

# Linux: Use OpenBLAS
cargo build --release --features openblas

# Single-threaded build
HODU_DISABLE_THREADS=1 cargo build --release
```

## License

BSD-3-Clause
