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
| `argmax` | Indices of maximum values along dimension | - |
| `argmin` | Indices of minimum values along dimension | - |

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

| Operation | Description |
|-----------|-------------|
| `split` | Split into specified sizes |
| `chunk` | Split into N equal chunks |

**Note**: `chunk` is implemented as a wrapper around `split` with automatically calculated equal sizes

## Normalization Operations

| Operation | Description |
|-----------|-------------|
| `softmax` | Softmax normalization along dimension |
| `log_softmax` | Log-softmax normalization along dimension |

**Note**: `softmax` and `log_softmax` are implemented using `exp`, `sum`, and `ln` with numerically stable computation by subtracting the maximum value

## Indexing Operations

Operations for selecting and gathering tensor elements.

| Operation | Description |
|-----------|-------------|
| `index_select` | Select elements along dimension |
| `gather` | Gather elements along dimension |
| `scatter` | Scatter elements along dimension |
| `scatter_add` | Scatter elements along dimension and add |
| `scatter_max` | Scatter elements along dimension and take max |
| `scatter_min` | Scatter elements along dimension and take min |

## Selection Operations

Operations for conditional selection and masking.

| Operation | Description |
|-----------|-------------|
| `where3` | Select between self and other based on condition |
| `masked_fill` | Fill elements where mask is true |
| `clamp` | Clamp values to range `[min, max]` |
| `clamp_min` | Clamp values to minimum |
| `clamp_max` | Clamp values to maximum |

**Notes**:
- `where3` is implemented using element-wise operations with automatic broadcasting
- `masked_fill` is implemented using `where3`
- `clamp`, `clamp_min`, and `clamp_max` are implemented using `where3` and comparison operations

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

## Differentiability Summary

Overview of which operations support automatic differentiation (gradient computation).

### Differentiable Operations ✓

| Category | Operations |
|----------|-----------|
| **Binary Ops** | `add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum` |
| **Unary Ops** | `neg`, `abs`, `square`, `sqrt`, `recip`, `relu`, `sigmoid`, `tanh`, `gelu`, `softplus`, `sin`, `cos`, `tan`, `exp`, `exp2`, `exp10`, `ln`, `log2`, `log10`, `leaky_relu`, `elu` |
| **Unary Scalar** | `add_scalar`, `sub_scalar`, `mul_scalar`, `div_scalar`, `pow_scalar`, `maximum_scalar`, `minimum_scalar` |
| **Matrix Ops** | `matmul`, `dot` |
| **Reduction Ops** | `sum`, `mean`, `max`, `min`, `prod`, `std`, `var`, `norm` |
| **Concat Ops** | `concat`, `stack` |
| **Split Ops** | `split`, `chunk` |
| **Normalization** | `softmax`, `log_softmax` |
| **Indexing Ops** | `index_select`, `gather`, `scatter`, `scatter_add`, `scatter_max`, `scatter_min` |
| **Selection Ops** | `where3`, `masked_fill`, `clamp`, `clamp_min`, `clamp_max` |
| **Shape Ops** | `reshape`, `flatten`, `squeeze`, `unsqueeze`, `broadcast`, `transpose`, `permute` |
| **Type Ops** | `to_dtype` |
| **Memory Ops** | `contiguous` |

### Non-Differentiable Operations ✗

| Category | Operations | Reason |
|----------|-----------|--------|
| **Logical Ops** | `logical_and`, `logical_or`, `logical_xor`, `logical_not` | Boolean operations have no meaningful gradients |
| **Comparison Ops** | `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `eq_scalar`, `ne_scalar`, `lt_scalar`, `le_scalar`, `gt_scalar`, `ge_scalar` | Comparison operations return discrete boolean values |
| **Index Ops** | `argmax`, `argmin` | Return discrete indices, not differentiable |
| **Sign Op** | `sign` | Discontinuous function with undefined gradient at zero |
