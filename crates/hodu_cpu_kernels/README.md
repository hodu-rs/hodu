# Hodu CPU Kernels

High-performance CPU kernels for tensor operations in pure C, with optional SIMD acceleration, OpenBLAS integration, and multi-threading support.

## Features

### Data Type Support
- **Floating Point**: f8e4m3, f8e5m2, bf16, f16, f32, f64
- **Integer**: i8, i16, i32, i64, u8, u16, u32, u64
- **Boolean**: bool

### Operation Categories
- **Unary**: neg, abs, sqrt, exp, ln, sin, cos, relu, gelu, tanh, sigmoid, etc.
- **Binary**: add, sub, mul, div, pow, maximum, minimum, comparison ops
- **Reduce**: sum, max, min, mean along arbitrary dimensions
- **Matrix**: matmul (batched with broadcasting), dot product
- **Convolution**: conv1d, conv2d, conv3d with padding/stride/dilation
- **Windowing**: max/min/sum/mean pooling operations
- **Indexing**: index_select, index_put, gather, scatter operations
- **Concat/Split**: tensor concatenation and splitting

### Performance Optimizations

#### SIMD Vectorization (f32/f64 only)
- **x86_64**: AVX2 (256-bit), SSE2 (128-bit)
- **ARM**: NEON (128-bit)
- Automatic architecture detection at compile time
- Vectorized hot paths with scalar fallback for strided layouts

#### OpenBLAS Integration (f32/f64 only)
- Matrix multiplication via GEMM
- Convolution via im2col + GEMM
- Automatic detection with fallback to pure C implementation

#### Multi-threading (std feature only)
- **pthread** support on Linux/macOS/MinGW
- Automatic parallelization for large matrix operations
- Work-stealing with optimal thread count based on workload size
- Low overhead for small matrices (automatic single-thread fallback)

## Environment Variables

Control build-time optimizations and dependencies:

### Performance Tuning
- `HODU_DISABLE_NATIVE` - Disable `-march=native` optimization
- `HODU_DISABLE_SIMD` - Disable SIMD auto-detection and vectorization
- `HODU_DISABLE_THREADS` - Disable pthread multi-threading

### Dependencies
- `HODU_DISABLE_BLAS` - Force disable OpenBLAS integration
- `OPENBLAS_DIR` - OpenBLAS installation directory
- `OPENBLAS_INCLUDE_DIR` - OpenBLAS headers directory
- `OPENBLAS_LIB_DIR` - OpenBLAS library directory

### Examples

```bash
# Default build with all optimizations
cargo build --release

# Disable multi-threading for reproducible single-threaded performance
HODU_DISABLE_THREADS=1 cargo build --release

# Build without OpenBLAS for minimal dependencies
HODU_DISABLE_BLAS=1 cargo build --release

# Build for older CPUs without native optimizations
HODU_DISABLE_NATIVE=1 cargo build --release

# Cross-compile with custom OpenBLAS
OPENBLAS_DIR=/opt/openblas cargo build --target aarch64-unknown-linux-gnu --release
```

## File Structure

```
kernels/
├── atomic.h             # Thread-safe atomic operations
├── constants.h          # Math constants
├── math_utils.h         # Math helper functions
├── simd_utils.h         # SIMD abstractions (AVX2/SSE2/NEON)
├── thread_utils.h       # Multi-threading utilities (pthread)
├── types.h              # Data type definitions
├── utils.h              # Tensor utilities
├── ops_binary.h/c       # Binary operations
└── ops_concat_split.h/c # Concat/split operations
├── ops_conv.h/c         # Convolution operations
├── ops_indexing.h/c     # Indexing operations
├── ops_matrix.h/c       # Matrix operations
├── ops_reduce.h/c       # Reduction operations
├── ops_unary.h/c        # Unary operations
├── ops_windowing.h/c    # Windowing/pooling operations
```

## Architecture

### Portability
- Pure C implementation for maximum compatibility
- Works on embedded systems (`no_std`) and general-purpose platforms
- Supports both contiguous and strided tensor layouts
- No mandatory external dependencies

### Optimization Strategy
1. **SIMD fast path**: Vectorized operations for contiguous f32/f64 tensors
2. **BLAS fast path**: OpenBLAS GEMM for matrix operations when available
3. **Multi-threaded fast path**: pthread parallelization for large matrices
4. **Scalar fallback**: Universal C implementation for all types and layouts

## Building

```bash
# Standard build
cargo build --release

# With OpenBLAS (auto-detected on Linux/macOS)
brew install openblas gfortran  # macOS
# or
sudo apt install libopenblas-dev pkg-config gfortran  # Linux

# Cross-compilation example (ARM with OpenBLAS)
OPENBLAS_DIR=/opt/arm-openblas cargo build --target aarch64-unknown-linux-gnu
```

## Testing

```bash
# Run all tests
cargo test

# Test specific operation categories
cargo test --test ops_matrix
cargo test --test ops_unary
cargo test --test ops_binary
```

## Performance Notes

- **SIMD**: 2-8x speedup for f32/f64 operations on contiguous data
- **OpenBLAS**: 10-100x speedup for large matrix multiplications
- **Multi-threading**: Near-linear scaling up to CPU core count for large matrices
- **Exotic types** (f8, bf16, f16): Computed via f32 conversion (no SIMD/BLAS acceleration)
- **Integer operations**: Pure C implementation (no SIMD/BLAS)

## Usage from Rust

```rust
use hodu_cpu_kernels::*;

// Direct FFI call to C kernel
unsafe {
    let input = vec![1.0f32; 100];
    let mut output = vec![0.0f32; 100];

    ops_unary::unary_relu_f32(
        input.as_ptr() as *const _,
        output.as_mut_ptr() as *mut _,
        100,
        0,
        std::ptr::null(),
    );
}
```

## License

BSD-3-Clause
