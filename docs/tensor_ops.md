# Tensor Operations

This document lists all supported tensor operations in Hodu.

## Binary Operations

Element-wise operations between two tensors.

| Operation | Description |
|-----------|-------------|
| `add` | Element-wise addition: `a + b` |
| `sub` | Element-wise subtraction: `a - b` |
| `mul` | Element-wise multiplication: `a * b` |
| `div` | Element-wise division: `a / b` |
| `pow` | Element-wise power: `a^b` |
| `maximum` | Element-wise maximum: `max(a, b)` |
| `minimum` | Element-wise minimum: `min(a, b)` |

## Binary Logical Operations

Logical operations between two tensors.

| Operation | Description |
|-----------|-------------|
| `logical_and` | Logical AND: `a && b` |
| `logical_or` | Logical OR: `a \|\| b` |
| `logical_xor` | Logical XOR: `a ^ b` |

## Comparison Operations

Element-wise comparison between two tensors, returns boolean tensor.

| Operation | Description |
|-----------|-------------|
| `eq` | Equal: `a == b` |
| `ne` | Not equal: `a != b` |
| `lt` | Less than: `a < b` |
| `le` | Less than or equal: `a <= b` |
| `gt` | Greater than: `a > b` |
| `ge` | Greater than or equal: `a >= b` |

## Comparison with Scalar

Element-wise comparison between tensor and scalar.

| Operation | Description |
|-----------|-------------|
| `eq_scalar` | Equal to scalar: `a == c` |
| `ne_scalar` | Not equal to scalar: `a != c` |
| `lt_scalar` | Less than scalar: `a < c` |
| `le_scalar` | Less than or equal to scalar: `a <= c` |
| `gt_scalar` | Greater than scalar: `a > c` |
| `ge_scalar` | Greater than or equal to scalar: `a >= c` |

## Unary Operations

Element-wise operations on a single tensor.

### Basic Operations

| Operation | Description |
|-----------|-------------|
| `neg` | Negation: `-a` |
| `abs` | Absolute value: `\|a\|` |
| `sign` | Sign function: `sign(a)` |
| `square` | Square: `a²` |
| `sqrt` | Square root: `√a` |
| `recip` | Reciprocal: `1/a` |

### Activation Functions

| Operation | Description |
|-----------|-------------|
| `relu` | ReLU: `max(0, a)` |
| `sigmoid` | Sigmoid: `1 / (1 + exp(-a))` |
| `tanh` | Hyperbolic tangent |
| `gelu` | Gaussian Error Linear Unit |
| `softplus` | Softplus: `ln(1 + exp(a))` |

### Trigonometric Functions

| Operation | Description |
|-----------|-------------|
| `sin` | Sine function |
| `cos` | Cosine function |
| `tan` | Tangent function |

### Exponential and Logarithmic Functions

| Operation | Description |
|-----------|-------------|
| `exp` | Natural exponential: `e^a` |
| `exp2` | Base-2 exponential: `2^a` |
| `exp10` | Base-10 exponential: `10^a` |
| `ln` | Natural logarithm: `log_e(a)` |
| `log2` | Base-2 logarithm: `log_2(a)` |
| `log10` | Base-10 logarithm: `log_10(a)` |

## Unary Logical Operations

| Operation | Description |
|-----------|-------------|
| `logical_not` | Logical NOT: `!a` |

## Unary Operations with Scalar

Element-wise operations between tensor and scalar.

### Arithmetic Operations

| Operation | Description |
|-----------|-------------|
| `add_scalar` | Add scalar: `a + c` |
| `sub_scalar` | Subtract scalar: `a - c` |
| `mul_scalar` | Multiply by scalar: `a * c` |
| `div_scalar` | Divide by scalar: `a / c` |
| `pow_scalar` | Power with scalar exponent: `a^c` |
| `maximum_scalar` | Maximum with scalar: `max(a, c)` |
| `minimum_scalar` | Minimum with scalar: `min(a, c)` |

### Activation Functions with Parameters

| Operation | Description |
|-----------|-------------|
| `leaky_relu` | Leaky ReLU with negative slope parameter |
| `elu` | ELU with alpha parameter |

## Matrix Operations

Hodu provides two matrix multiplication operations, following XLA's design:

### matmul - Batched Matrix Multiplication

Supports 1D, 2D, and ND tensors with broadcasting.

| Input Shapes | Operation Type | Output Shape | Example |
|--------------|----------------|--------------|---------|
| `[N]` x `[N]` | Vector dot product | scalar `[]` | `[3] x [3] → []` |
| `[M, K]` x `[K]` | Matrix-vector product | `[M]` | `[2, 3] x [3] → [2]` |
| `[K]` x `[K, N]` | Vector-matrix product | `[N]` | `[3] x [3, 4] → [4]` |
| `[M, K]` x `[K, N]` | Matrix multiplication | `[M, N]` | `[2, 3] x [3, 4] → [2, 4]` |
| `[B..., M, K]` x `[B..., K, N]` | Batched matmul | `[B..., M, N]` | `[2, 3, 4] x [2, 4, 5] → [2, 3, 5]` |
| `[B1..., M, K]` x `[B2..., K, N]` | Broadcast batched matmul | `[broadcast(B1, B2)..., M, N]` | `[1, 2, 3] x [4, 3, 2] → [4, 2, 2]` |

### dot - Simple Dot Product

Supports only 1D and 2D combinations (for simplicity).

| Input Shapes | Operation Type | Output Shape | Example |
|--------------|----------------|--------------|---------|
| `[N]` x `[N]` | Vector dot product | scalar `[]` | `[3] x [3] → []` |
| `[M, K]` x `[K]` | Matrix-vector product | `[M]` | `[2, 3] x [3] → [2]` |
| `[K]` x `[K, N]` | Vector-matrix product | `[N]` | `[3] x [3, 4] → [4]` |
| `[M, K]` x `[K, N]` | Matrix multiplication | `[M, N]` | `[2, 3] x [3, 4] → [2, 4]` |

**Notes:**
- For batched/ND operations (3D+), use `matmul` instead of `dot`
- `matmul` handles broadcasting automatically for batch dimensions
- Both operations follow XLA's semantics for consistency

## Reduction Operations

Operations that reduce tensor dimensions.

| Operation | Description | Variants |
|-----------|-------------|----------|
| `sum` | Sum along specified dimensions | `sum_all`, `sum_to_shape` |
| `mean` | Mean along specified dimensions | `mean_all` |
| `max` | Maximum along specified dimensions | - |
| `min` | Minimum along specified dimensions | - |
| `prod` | Product along specified dimensions | - |
| `std` | Standard deviation along specified dimensions | `std_all` |
| `var` | Variance along specified dimensions | `var_all` |
| `norm` | Norm along specified dimensions | `l2_norm`, `l1_norm` |

**Note**: `l1_norm` is implemented as a combination of `abs` and `sum` operations

## Concat Operations

Operations that combine tensors.

| Operation | Description | Aliases |
|-----------|-------------|---------|
| `concat` | Concatenate along existing dimension | `cat` |
| `stack` | Stack along new dimension | - |

**Note**: `stack` is implemented as a combination of `unsqueeze` and `concat` operations

## Split Operations

Operations that split tensor.

| Operation | Description | Aliases |
|-----------|-------------|---------|
| `split` | Split into specified sizes | - |
| `chunk` | Split into N equal chunks | - |

**Note**: `chunk` is implemented as a wrapper around `split` with automatically calculated equal sizes

## Normalization Operations

| Operation | Description | Aliases |
|-----------|-------------|---------|
| `softmax` | Softmax normalization along dimension | - |
| `log_softmax` | Log-softmax normalization along dimension | - |

**Note**: `softmax` and `log_softmax` are implemented using `exp`, `sum`, and `ln` with numerically stable computation by subtracting the maximum value

## Shape Operations

Operations that manipulate tensor shape and layout.

| Operation | Description | Aliases |
|-----------|-------------|---------|
| `reshape` | Change tensor shape | `view` |
| `flatten` | Flatten tensor to 1D | - |
| `squeeze` | Remove dimensions of size 1 | - |
| `unsqueeze` | Add dimension of size 1 | - |
| `broadcast` | Broadcast to larger shape | `broadcast_like`, `broadcast_left` |
| `transpose` | Swap two dimensions | `t` (for last two dims) |
| `permute` | Reorder dimensions | - |

## Type Operations

| Operation | Description |
|-----------|-------------|
| `to_dtype` | Cast tensor to different data type |

## Memory Operations

| Operation | Description |
|-----------|-------------|
| `contiguous` | Ensure tensor has contiguous memory layout |
