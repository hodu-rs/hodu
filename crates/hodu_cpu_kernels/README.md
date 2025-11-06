# Hodu CPU Kernels

High-performance CPU kernels for tensor operations, supporting a wide range of data types including exotic floating-point formats.

## Supported Data Types

### Floating Point Types
- **f8e4m3**: 8-bit float (1 sign, 4 exponent, 3 mantissa bits)
- **f8e5m2**: 8-bit float (1 sign, 5 exponent, 2 mantissa bits)
- **bf16**: BFloat16 (1 sign, 8 exponent, 7 mantissa bits)
- **f16**: Float16/Half (1 sign, 5 exponent, 10 mantissa bits)
- **f32**: Float32 (standard single precision)
- **f64**: Float64 (standard double precision)

### Integer Types
- **u8, u16, u32, u64**: Unsigned integers (8, 16, 32, 64 bits)
- **i8, i16, i32, i64**: Signed integers (8, 16, 32, 64 bits)

### Boolean Type
- **bool**: Boolean type (1 byte)

## Supported Operations

### Unary Operations
- **Basic**: neg, abs, sign, square, sqrt, recip
- **Activation Functions**: relu, sigmoid, tanh, gelu, softplus, silu, mish
- **Trigonometric**: sin, cos, tan
- **Exponential/Logarithmic**: exp, exp2, exp10, ln, log2, log10
- **Logical**: logical_not
- **Scalar Operations**: add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar, maximum_scalar, minimum_scalar
- **Comparison**: eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar

### Binary Operations
- **Arithmetic**: add, sub, mul, div, pow
- **Min/Max**: maximum, minimum
- **Logical**: logical_and, logical_or, logical_xor
- **Comparison**: eq, ne, lt, le, gt, ge

### Reduce Operations
- **Reduction Functions**: sum, max, min, mean
- **Features**:
  - Reduce along arbitrary dimensions
  - Keep dimension option
  - Strided tensor support

### Matrix Operations
- **matmul**: Batched matrix multiplication with broadcasting
- **Features**:
  - Batch dimension support
  - Broadcasting for batch dimensions
  - Strided input support

### Convolution Operations
- **conv1d, conv2d, conv3d**: 1D, 2D, and 3D convolution
- **Features**:
  - Padding support
  - Stride support
  - Dilation support
  - Groups support

### Windowing Operations
- **reduce_window_max**: Sliding window maximum (pooling)
- **reduce_window_min**: Sliding window minimum
- **reduce_window_sum**: Sliding window sum
- **reduce_window_mean**: Sliding window mean (average pooling)
- **Features**:
  - Arbitrary window shapes
  - Configurable strides
  - Padding support

### Indexing Operations
- **index_select**: Select elements along a dimension
- **index_put**: Put values at specified indices
- **gather**: Gather elements from tensor
- **scatter**: Scatter values to indices
- **scatter_add**: Scatter-add values to indices
- **scatter_max**: Scatter-max values to indices
- **scatter_min**: Scatter-min values to indices

### Concat/Split Operations
- **concat**: Concatenate tensors along a dimension
- **split**: Split tensor into multiple tensors

## SIMD Supported Data Types

### Floating Point Types
- **f32**: Float32 (standard single precision)
- **f64**: Float64 (standard double precision)

## SIMD Supported Operations

- **binary**: add, sub, mul, div
- **matrix**: matmul, dot

## Architecture

### Portability
- **Pure C implementation** for maximum portability
- Works on embedded systems and general-purpose platforms
- No external dependencies (only standard C library)
- Supports both contiguous and strided tensor layouts

### Performance Features
- Optimized for contiguous memory layouts
- Efficient strided access support
- Type conversion handled transparently
- Math operations optimized for common cases

## File Structure

```
kernels/
├── constants.h           # Math and type constants
├── types.h              # Data type definitions and conversions
├── math_utils.h         # Math helper functions
├── utils.h              # Tensor utilities (striding, contiguity checks)
├── atomic.h             # Atomic operations for thread safety
├── ops_unary.h/c        # Unary operations
├── ops_binary.h/c       # Binary operations
├── ops_reduce.h/c       # Reduction operations
├── ops_matrix.h/c       # Matrix operations (matmul)
├── ops_conv.h/c         # Convolution operations
├── ops_windowing.h/c    # Windowing operations (pooling)
├── ops_indexing.h/c     # Indexing operations
└── ops_concat_split.h/c # Concat and split operations
```

## Usage Example

```c
#include "ops_unary.h"
#include "ops_binary.h"
#include "ops_matrix.h"

// Unary operation: relu on f32 array
float input[100];
float output[100];
unary_relu_f32(input, output, 100, 0, NULL);

// Binary operation: add two f32 arrays
float lhs[100], rhs[100], result[100];
binary_add_f32(lhs, rhs, result, 100, 0, NULL);

// Matrix multiplication
size_t metadata[] = {/* lhs_ndim, rhs_ndim, batch_ndim, shapes, strides, offsets, M, K, N */};
float A[100], B[100], C[100];
matmul_f32(A, B, C, 100, 2, metadata);
```

## Building

This crate uses Rust's build system to compile the C kernels. The C code is compiled as a static library and linked with the Rust code.

```bash
cargo build --release
```

### Testing

Run all tests including unit tests for each operation category:

```bash
cargo test
```

Run specific test suites:

```bash
cargo test --test unary
cargo test --test binary
cargo test --test reduce
cargo test --test matrix
cargo test --test conv
cargo test --test windowing
cargo test --test indexing
cargo test --test concat_split
```

## Notes

- The C kernels are designed to be callable from Rust via FFI
- All exotic float types (f8, bf16, f16) are converted to/from f32 for computation
- Integer overflow behavior follows C semantics
- Division by zero returns 0 (not NaN) for integer types
- Windowing operations use padding values based on the reduction type (e.g., -∞ for max, +∞ for min, 0 for sum)
