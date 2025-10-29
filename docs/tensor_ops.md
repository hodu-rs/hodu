# Tensor Operations

This document lists all supported tensor operations in Hodu.

## Device and Backend DType Support

Some data types are not supported on specific devices or backends:

**Device Restrictions:**
- **Metal**: `f8e4m3`, `f8e5m2`, `f64` are not supported

**Backend Restrictions:**
- **XLA**: `f8e4m3`, `f8e5m2` are not supported

## Binary Operations

Element-wise operations between two tensors.

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `add` | Element-wise addition: `a + b` | `bool` |
| `sub` | Element-wise subtraction: `a - b` | `bool` |
| `mul` | Element-wise multiplication: `a * b` | `bool` |
| `div` | Element-wise division: `a / b` | `bool` |
| `pow` | Element-wise power: `a^b` | `bool` |
| `maximum` | Element-wise maximum: `max(a, b)` | `bool` |
| `minimum` | Element-wise minimum: `min(a, b)` | `bool` |

## Binary Logical Operations

Logical operations between two tensors.

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `logical_and` | Logical AND: `a && b` | - |
| `logical_or` | Logical OR: `a \|\| b` | - |
| `logical_xor` | Logical XOR: `a ^ b` | - |

## Comparison Operations

Element-wise comparison between two tensors, returns boolean tensor.

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `eq` | Equal: `a == b` | - |
| `ne` | Not equal: `a != b` | - |
| `lt` | Less than: `a < b` | `bool` |
| `le` | Less than or equal: `a <= b` | `bool` |
| `gt` | Greater than: `a > b` | `bool` |
| `ge` | Greater than or equal: `a >= b` | `bool` |

## Comparison with Scalar

Element-wise comparison between tensor and scalar.

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `eq_scalar` | Equal to scalar: `a == c` | - |
| `ne_scalar` | Not equal to scalar: `a != c` | - |
| `lt_scalar` | Less than scalar: `a < c` | `bool` |
| `le_scalar` | Less than or equal to scalar: `a <= c` | `bool` |
| `gt_scalar` | Greater than scalar: `a > c` | `bool` |
| `ge_scalar` | Greater than or equal to scalar: `a >= c` | `bool` |

## Unary Operations

Element-wise operations on a single tensor.

### Basic Operations

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `neg` | Negation: `-a` | `bool`, `u8`-`u64` |
| `abs` | Absolute value: `\|a\|` | `bool`, `u8`-`u64` |
| `sign` | Sign function: `sign(a)` | `bool`, `u8`-`u64` |
| `square` | Square: `a²` | `bool` |
| `sqrt` | Square root: `√a` | `bool`, `u8`-`u64`, `i8`-`i64` |
| `recip` | Reciprocal: `1/a` | `bool`, `u8`-`u64`, `i8`-`i64` |

### Activation Functions

| Operation | Description | Aliases | Unsupported Types |
|-----------|-------------|---------|-------------------|
| `relu` | ReLU: `max(0, a)` | - | `bool`, `u8u64` |
| `sigmoid` | Sigmoid: `1 / (1 + exp(-a))` | - | `bool`, `u8u64`, `i8i64` |
| `tanh` | Hyperbolic tangent | - | `bool`, `u8u64`, `i8i64` |
| `gelu` | Gaussian Error Linear Unit | - | `bool`, `u8u64`, `i8i64` |
| `softplus` | Softplus: `ln(1 + exp(a))` | - | `bool`, `u8u64`, `i8i64` |
| `silu` | SiLU (Swish): `a * sigmoid(a)` | `swish` | `bool`, `u8u64`, `i8i64` |
| `mish` | Mish: `a * tanh(softplus(a))` | - | `bool`, `u8u64`, `i8i64` |

### Trigonometric Functions

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `sin` | Sine function | `bool`, `u8`-`u64`, `i8`-`i64` |
| `cos` | Cosine function | `bool`, `u8`-`u64`, `i8`-`i64` |
| `tan` | Tangent function | `bool`, `u8`-`u64`, `i8`-`i64` |

### Exponential and Logarithmic Functions

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `exp` | Natural exponential: `e^a` | `bool`, `u8`-`u64`, `i8`-`i64` |
| `exp2` | Base-2 exponential: `2^a` | `bool`, `u8`-`u64`, `i8`-`i64` |
| `exp10` | Base-10 exponential: `10^a` | `bool`, `u8`-`u64`, `i8`-`i64` |
| `ln` | Natural logarithm: `log_e(a)` | `bool`, `u8`-`u64`, `i8`-`i64` |
| `log2` | Base-2 logarithm: `log_2(a)` | `bool`, `u8`-`u64`, `i8`-`i64` |
| `log10` | Base-10 logarithm: `log_10(a)` | `bool`, `u8`-`u64`, `i8`-`i64` |

## Unary Logical Operations

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `logical_not` | Logical NOT: `!a` | - |

## Unary Operations with Scalar

Element-wise operations between tensor and scalar.

### Arithmetic Operations

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `add_scalar` | Add scalar: `a + c` | `bool` |
| `sub_scalar` | Subtract scalar: `a - c` | `bool` |
| `mul_scalar` | Multiply by scalar: `a * c` | `bool` |
| `div_scalar` | Divide by scalar: `a / c` | `bool` |
| `pow_scalar` | Power with scalar exponent: `a^c` | `bool` |
| `maximum_scalar` | Maximum with scalar: `max(a, c)` | `bool` |
| `minimum_scalar` | Minimum with scalar: `min(a, c)` | `bool` |

### Activation Functions with Parameters

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `leaky_relu` | Leaky ReLU with negative slope parameter | `bool`, `u8`-`u64`, `i8`-`i64` |
| `elu` | ELU with alpha parameter | `bool`, `u8`-`u64`, `i8`-`i64` |
| `prelu` | PReLU (Parametric ReLU) with learnable slope | `bool`, `u8`-`u64`, `i8`-`i64` |

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

**Unsupported Types:** `bool`, `u8`-`u64`, `i8`-`i64`

## Reduction Operations

Operations that reduce tensor dimensions.

| Operation | Description | Variants | Unsupported Types |
|-----------|-------------|----------|-------------------|
| `sum` | Sum along specified dimensions | `sum_all`, `sum_to_shape` | `bool` |
| `mean` | Mean along specified dimensions | `mean_all` | `bool`, `u8`-`u64`, `i8`-`i64` |
| `max` | Maximum along specified dimensions | - | `bool` |
| `min` | Minimum along specified dimensions | - | `bool` |
| `prod` | Product along specified dimensions | - | `bool` |
| `std` | Standard deviation along specified dimensions | `std_all` | `bool`, `u8`-`u64`, `i8`-`i64` |
| `var` | Variance along specified dimensions | `var_all` | `bool`, `u8`-`u64`, `i8`-`i64` |
| `norm` | Lp norm with parameter `p` (p=1: L1, p=2: L2, other: Lp) | `l2_norm`, `l1_norm` | `bool`, `u8`-`u64`, `i8`-`i64` |
| `argmax` | Indices of maximum values along dimension | - | - |
| `argmin` | Indices of minimum values along dimension | - | - |
| `any` | Check if any element is true (logical reduction) | - | - |
| `all` | Check if all elements are true (logical reduction) | - | - |

**Notes**:
- `norm(p, dims, keep_dim)` computes the Lp norm: `(sum(|x|^p))^(1/p)`
  - When `p=1`: delegates to `l1_norm` (combination of `abs` + `sum`)
  - When `p=2`: delegates to `l2_norm` (native reduction operation)
  - Other `p` values: combination of `abs`, `pow_scalar`, `sum`, and `pow_scalar`

## Concat Operations

Operations that combine tensors.

| Operation | Description | Aliases | Unsupported Types |
|-----------|-------------|---------|-------------------|
| `concat` | Concatenate along existing dimension | `cat` | - |
| `stack` | Stack along new dimension | - | - |

**Note**: `stack` is implemented as a combination of `unsqueeze` and `concat` operations

## Split Operations

Operations that split tensor.

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `split` | Split into specified sizes | - |
| `chunk` | Split into N equal chunks | - |

**Note**: `chunk` is implemented as a wrapper around `split` with automatically calculated equal sizes

## Normalization Operations

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `softmax` | Softmax normalization along dimension | (depends on underlying ops) |
| `log_softmax` | Log-softmax normalization along dimension | (depends on underlying ops) |

**Note**: `softmax` and `log_softmax` are implemented using `exp`, `sum`, and `ln` with numerically stable computation by subtracting the maximum value. They inherit dtype restrictions from these operations.

## Indexing Operations

Operations for selecting and gathering tensor elements.

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `index_select` | Select elements along dimension using indices | - |
| `index_put` | Put values at specified indices along dimension | - |
| `gather` | Gather elements along dimension | - |
| `scatter` | Scatter elements along dimension | - |
| `scatter_add` | Scatter elements along dimension and add | `bool` |
| `scatter_max` | Scatter elements along dimension and take max | `bool` |
| `scatter_min` | Scatter elements along dimension and take min | `bool` |

**Examples:**
- `index_select(dim, indices)`: Select rows/columns at given indices
  - Input: `[3, 4, 5]`, indices: `[0, 2]`, dim: 0 → Output: `[2, 4, 5]` (selects 1st and 3rd elements)
- `index_put(dim, indices, values)`: Put values at given indices
  - Input: `[3, 4, 5]`, indices: `[1]`, values: `[4]`, dim: 0 → Output: `[3, 4, 5]` with 2nd element set to values

## Selection Operations

Operations for conditional selection and masking.

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `where3` | Select between self and other based on condition | (depends on underlying ops) |
| `masked_fill` | Fill elements where mask is true | (depends on underlying ops) |
| `clamp` | Clamp values to range `[min, max]` | (depends on underlying ops) |
| `clamp_min` | Clamp values to minimum | (depends on underlying ops) |
| `clamp_max` | Clamp values to maximum | (depends on underlying ops) |

**Notes**:
- `where3` is implemented using element-wise operations with automatic broadcasting
- `masked_fill` is implemented using `where3`
- `clamp`, `clamp_min`, and `clamp_max` are implemented using `where3` and comparison operations
- These operations inherit dtype restrictions from their underlying operations

## Convolution Operations

Neural network convolution operations with support for stride, padding, and dilation.

### Standard Convolutions

| Operation | Input Shape | Weight Shape | Output Shape | Description |
|-----------|-------------|--------------|--------------|-------------|
| `conv1d` | `[B, C_in, L]` | `[C_out, C_in, K]` | `[B, C_out, L_out]` | 1D convolution |
| `conv2d` | `[B, C_in, H, W]` | `[C_out, C_in, K_h, K_w]` | `[B, C_out, H_out, W_out]` | 2D convolution |
| `conv3d` | `[B, C_in, D, H, W]` | `[C_out, C_in, K_d, K_h, K_w]` | `[B, C_out, D_out, H_out, W_out]` | 3D convolution |

### Transposed Convolutions (Deconvolutions)

| Operation | Input Shape | Weight Shape | Output Shape | Description |
|-----------|-------------|--------------|--------------|-------------|
| `conv_transpose1d` | `[B, C_in, L]` | `[C_in, C_out, K]` | `[B, C_out, L_out]` | 1D transposed convolution |
| `conv_transpose2d` | `[B, C_in, H, W]` | `[C_in, C_out, K_h, K_w]` | `[B, C_out, H_out, W_out]` | 2D transposed convolution |
| `conv_transpose3d` | `[B, C_in, D, H, W]` | `[C_in, C_out, K_d, K_h, K_w]` | `[B, C_out, D_out, H_out, W_out]` | 3D transposed convolution |

**Unsupported Types:** `bool`, `u8`-`u64`, `i8`-`i64`

## Windowing Operations

Sliding window reduction operations, commonly used for pooling.

| Operation | Description |
|-----------|-------------|
| `reduce_window` | Apply reduction function over sliding windows |

**Parameters:**
- `window_shape`: Size of the sliding window for each dimension
- `strides`: Step size for sliding the window (default: 1 for each dim)
- `padding`: Padding applied before windowing
- `reduction`: Reduction type - `'max'`, `'mean'`, `'sum'`, `'min'`

**Unsupported Types by Reduction:**
- `max`: `bool`
- `mean`: `bool`, `u8`-`u64`, `i8`-`i64`
- `sum`: `bool`
- `min`: `bool`

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

## Shape Operations with Scalars


| Operation | Description | Aliases |
|-----------|-------------|---------|
| `slice` | Extract sub-tensor along dimension with optional stride | |


## Type Operations

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `to_dtype` | Cast tensor to different data type | - |

## Memory Operations

| Operation | Description | Unsupported Types |
|-----------|-------------|-------------------|
| `contiguous` | Ensure tensor has contiguous memory layout | - |
