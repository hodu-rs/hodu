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
├── constants.h      # Math and type constants
├── types.h          # Data type definitions and conversions
├── math_utils.h     # Math helper functions
├── utils.h          # Tensor utilities (striding, contiguity checks)
├── ops_unary.h      # Unary operation declarations
├── ops_unary.c      # Unary operation implementations
├── ops_binary.h     # Binary operation declarations
└── ops_binary.c     # Binary operation implementations
```

## Usage Example

```c
#include "ops_unary.h"
#include "ops_binary.h"

// Unary operation: relu on f32 array
float input[100];
float output[100];
unary_relu_f32(input, output, 100, 0, NULL);

// Binary operation: add two f32 arrays
float lhs[100], rhs[100], result[100];
binary_add_f32(lhs, rhs, result, 100, 0, NULL);
```

## Building

This crate uses Rust's build system to compile the C kernels. The C code is compiled as a static library and linked with the Rust code.

```bash
cargo build --release
```

## Implementation Status

- ✅ Core type system with f8, bf16, f16 support
- ✅ Unary operations for float types (f32, f64)
- ✅ Binary operations for float types (f32, f64)
- ✅ Basic integer operations (u8, i32 examples)
- 🚧 Complete coverage for all integer types
- 🚧 Reduce operations
- 🚧 Matrix operations (matmul, dot)
- 🚧 Convolution operations
- 🚧 Memory/indexing operations

## Notes

- The C kernels are designed to be callable from Rust via FFI
- All exotic float types (f8, bf16, f16) are converted to/from f32 for computation
- Integer overflow behavior follows C semantics
- Division by zero returns 0 (not NaN) for integer types
