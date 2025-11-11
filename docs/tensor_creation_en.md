# Tensor Creation Guide

## Overview

There are three main ways to create tensors in Hodu:

1. **Dynamic Creation**: Immediately create actual tensors with data
2. **Operational Creation**: Create tensors from other tensors with gradient propagation
3. **Static Creation**: Define input placeholders in Script mode

## Dynamic Tensor Creation

Methods to create tensors with actual data in dynamic mode.

### Runtime Device Configuration

Set the default device where tensors will be created:

```rust
use hodu::prelude::*;

// Create on CPU (default)
set_runtime_device(Device::CPU);

// Create on CUDA GPU
#[cfg(feature = "cuda")]
set_runtime_device(Device::CUDA(0));

// Create on Metal GPU (macOS)
#[cfg(feature = "metal")]
set_runtime_device(Device::METAL(0));
```

**Important**: During script building (`builder.start()` ~ `builder.end()`), runtime device settings are ignored and tensors are always created on CPU.

### Creating from Data

#### Tensor::new()

Create tensors from arrays or vectors:

```rust
use hodu::prelude::*;

// 1D tensor
let t1 = Tensor::new(vec![1.0, 2.0, 3.0])?;
println!("{}", t1);  // [1, 2, 3]

// 2D tensor
let t2 = Tensor::new(vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
])?;
println!("{}", t2);
// [[1, 2, 3],
//  [4, 5, 6]]

// 3D tensor
let t3 = Tensor::new(vec![
    vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ],
    vec![
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ],
])?;
```

**Features:**
- Data type is automatically inferred
- Shape is determined from input array structure
- Works with any type implementing `IntoFlattened` trait

#### Tensor::from_slice()

Create a tensor from data with an explicitly specified shape:

```rust
use hodu::prelude::*;

// 1D data reshaped to 2D tensor
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let t = Tensor::from_slice(data, &[2, 3])?;
println!("{}", t);
// [[1, 2, 3],
//  [4, 5, 6]]

// 2D data reshaped to 3D tensor
let data = vec![
    vec![1.0, 2.0],
    vec![3.0, 4.0],
];
let t = Tensor::from_slice(data, &[2, 2, 1])?;
// Shape: [2, 2, 1]

// Error: data size doesn't match shape size
let data = vec![1.0, 2.0, 3.0];
let result = Tensor::from_slice(data, &[2, 2])?;
// Error: SizeMismatch { expected: 4, got: 3 }
```

**Features:**
- Data and shape can be specified separately
- Validates that data size matches shape size
- Raises `HoduError::SizeMismatch` if sizes don't match
- Useful for reshaping flat data into multidimensional tensors
- Works with any type implementing `IntoFlattened` trait

**Difference from new():**
- `new()`: Shape is automatically inferred from data structure
- `from_slice()`: Shape is explicitly specified, validated against data size

### Initialization with Specific Values

#### Tensor::zeros()

Create a tensor filled with zeros:

```rust
use hodu::prelude::*;

// Shape [2, 3], type F32
let zeros = Tensor::zeros(&[2, 3], DType::F32)?;
println!("{}", zeros);
// [[0, 0, 0],
//  [0, 0, 0]]

// Shape [3], type I32
let zeros_i32 = Tensor::zeros(&[3], DType::I32)?;
println!("{}", zeros_i32);  // [0, 0, 0]
```

#### Tensor::zeros_like()

Create a zero tensor with same shape and dtype as existing tensor:

```rust
let original = Tensor::new(vec![1.0, 2.0, 3.0])?;
let zeros = Tensor::zeros_like(&original)?;
println!("{}", zeros);  // [0, 0, 0]
```

#### Tensor::ones()

Create a tensor filled with ones:

```rust
use hodu::prelude::*;

let ones = Tensor::ones(&[2, 3], DType::F32)?;
println!("{}", ones);
// [[1, 1, 1],
//  [1, 1, 1]]
```

#### Tensor::ones_like()

Create a ones tensor with same shape and dtype as existing tensor:

```rust
let original = Tensor::new(vec![1.0, 2.0, 3.0])?;
let ones = Tensor::ones_like(&original)?;
println!("{}", ones);  // [1, 1, 1]
```

#### Tensor::full()

Create a tensor filled with a specified value:

```rust
use hodu::prelude::*;

// All elements are 3.14
let pi = Tensor::full(&[2, 2], 3.14)?;
println!("{}", pi);
// [[3.14, 3.14],
//  [3.14, 3.14]]

// Scalar (shape [])
let scalar = Tensor::full(&[], 42.0)?;
println!("{}", scalar);  // 42
```

**Features:**
- DType is automatically determined from value type
- Works with any type implementing `Into<Scalar>` (f32, f64, i32, etc.)
- Primitive types like f32, i32 are automatically converted to Scalar

#### Tensor::full_like()

Create a tensor with same shape and dtype as existing tensor, filled with specified value:

```rust
let original = Tensor::new(vec![1.0, 2.0, 3.0])?;
let filled = Tensor::full_like(&original, 9.0)?;  // f32 auto-converts to Scalar::F32
println!("{}", filled);  // [9, 9, 9]
```

### Random Initialization

#### Tensor::randn()

Create a random tensor sampled from normal distribution:

```rust
use hodu::prelude::*;

// Mean 0, standard deviation 1 (standard normal)
let randn = Tensor::randn(&[2, 3], 0.0, 1.0)?;
println!("{}", randn);
// [[0.5432, -1.234, 0.891],
//  [-0.234, 0.123, 1.456]]

// Mean 10, standard deviation 2
let custom_randn = Tensor::randn(&[3], 10.0, 2.0)?;
println!("{}", custom_randn);  // [11.234, 8.567, 12.345]
```

**Features:**
- Normal distribution sampling using Box-Muller transform
- Result is float type if either mean or std is float
- Useful for neural network initialization (Xavier, He initialization, etc.)

#### Tensor::randn_like()

Create a random tensor with same shape as existing tensor:

```rust
let original = Tensor::new(vec![1.0, 2.0, 3.0])?;
let randn = Tensor::randn_like(&original, 0.0, 1.0)?;
println!("{}", randn);  // [-0.234, 1.567, 0.891]
```

#### Tensor::rand_uniform()

Create a random tensor sampled from uniform distribution:

```rust
use hodu::prelude::*;

// Uniform distribution in range [0, 1)
let uniform = Tensor::rand_uniform(&[2, 3], 0.0, 1.0)?;
println!("{}", uniform);
// [[0.234, 0.891, 0.456],
//  [0.123, 0.789, 0.345]]

// Uniform distribution in range [-1, 1)
let uniform_centered = Tensor::rand_uniform(&[3], -1.0, 1.0)?;
println!("{}", uniform_centered);  // [-0.234, 0.567, 0.891]

// Uniform distribution in range [0, 10)
let uniform_scaled = Tensor::rand_uniform(&[2, 2], 0.0, 10.0)?;
println!("{}", uniform_scaled);
// [[3.456, 7.891],
//  [1.234, 9.012]]
```

**Features:**
- Uniformly distributed within the specified range [low, high)
- Result is float type if either low or high is float
- Useful for Dropout, data augmentation, probabilistic sampling

**Use Cases:**
- **Dropout**: Generate random masks
- **Data augmentation**: Random noise, transformation parameters
- **Reinforcement learning**: Action sampling, exploration noise
- **Monte Carlo simulation**: Uniform sampling

#### Tensor::rand_uniform_like()

Create a uniform random tensor with same shape as existing tensor:

```rust
let original = Tensor::new(vec![1.0, 2.0, 3.0])?;
let uniform = Tensor::rand_uniform_like(&original, 0.0, 1.0)?;
println!("{}", uniform);  // [0.234, 0.567, 0.891]
```

## Operational Tensor Creation

Create new tensors from existing tensors through operations. These methods support gradient propagation for automatic differentiation.

### Tensor::where3_select()

Select elements from two tensors based on a condition:

```rust
use hodu::prelude::*;

// Create condition, x, and y tensors
let condition = Tensor::new(vec![true, false, true, false])?;
let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?;
let y = Tensor::new(vec![10.0, 20.0, 30.0, 40.0])?;

// Select x where condition is true, y where condition is false
let result = Tensor::where3_select(&condition, &x, &y)?;
println!("{}", result);  // [1, 20, 3, 40]
```

**Features:**
- Supports automatic broadcasting of condition, x, and y tensors
- Condition tensor is automatically converted to match x's dtype
- Result shape is determined by broadcasting rules
- Gradient flows through both x and y tensors

**Broadcasting Example:**

```rust
use hodu::prelude::*;

// Condition: [2, 1], x: [2, 3], y: scalar
let condition = Tensor::new(vec![vec![true], vec![false]])?;
let x = Tensor::new(vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
])?;
let y = Tensor::full(&[1], 100.0)?;

let result = Tensor::where3_select(&condition, &x, &y)?;
println!("{}", result);
// [[1, 2, 3],      // condition[0] is true -> x[0]
//  [100, 100, 100]] // condition[1] is false -> y
```

**Use Cases:**
- Conditional value selection in neural networks
- Masking operations
- Implementing custom activation functions
- Gradient clipping implementations

## Static Tensor Creation (Script Mode)

Used to define input placeholders in Script mode.

### Tensor::input()

Create an input placeholder for Script:

```rust
use hodu::prelude::*;

let builder = Builder::new("my_script".to_string());
builder.start()?;

// Define input placeholders
let x = Tensor::input("x", &[2, 3])?;
let y = Tensor::input("y", &[3, 4])?;

// Define operations
let result = x.matmul(&y)?;

builder.add_output("result", result)?;
builder.end()?;

let mut script = builder.build()?;
```

**Features:**
- **Builder context required**: Only usable after calling `builder.start()`
- Has no actual data (storage = None)
- Only shape is defined; actual data must be provided at execution time
- Automatically registered as input to the builder

**Error:**

```rust
// Incorrect usage: calling without Builder
let x = Tensor::input("x", &[2, 3])?;
// Error: StaticTensorCreationRequiresBuilderContext
```

**Correct usage:**

```rust
let builder = Builder::new("script".to_string());
builder.start()?;  // Activate builder context

let x = Tensor::input("x", &[10, 20])?;  // OK
let y = Tensor::input("y", &[20, 30])?;  // OK

// ... define operations ...

builder.end()?;
```

## Creation Functions Comparison

### Dynamic Creation

| Function | Purpose | DType | Initial Value | Example |
|----------|---------|-------|---------------|---------|
| `new()` | Create from data | Auto-inferred | Input data | `Tensor::new(vec![1, 2, 3])` |
| `zeros()` | Create zero tensor | Explicit | 0 | `Tensor::zeros(&[2, 3], DType::F32)` |
| `zeros_like()` | Zero tensor based on existing | Copied | 0 | `Tensor::zeros_like(&tensor)` |
| `ones()` | Create ones tensor | Explicit | 1 | `Tensor::ones(&[2, 3], DType::F32)` |
| `ones_like()` | Ones tensor based on existing | Copied | 1 | `Tensor::ones_like(&tensor)` |
| `full()` | Fill with specific value | Auto-inferred | User-specified | `Tensor::full(&[2, 3], 3.14)` |
| `full_like()` | Filled tensor based on existing | Copied | User-specified | `Tensor::full_like(&tensor, val)` |
| `randn()` | Normal distribution random | Auto-inferred | N(μ, σ²) | `Tensor::randn(&[2, 3], 0., 1.)` |
| `randn_like()` | Normal random based on existing | Copied | N(μ, σ²) | `Tensor::randn_like(&tensor, 0., 1.)` |
| `rand_uniform()` | Uniform distribution random | Auto-inferred | U(low, high) | `Tensor::rand_uniform(&[2, 3], 0., 1.)` |
| `rand_uniform_like()` | Uniform random based on existing | Copied | U(low, high) | `Tensor::rand_uniform_like(&tensor, 0., 1.)` |

### Operational Creation

| Function | Purpose | Features |
|----------|---------|----------|
| `where3_select()` | Conditional selection | Broadcasting support, gradient propagation |

### Static Creation

| Function | Purpose | Requirements | Features |
|----------|---------|--------------|----------|
| `input()` | Script input placeholder | Builder context | No data, only shape defined |

## Practical Examples

### Example 1: Neural Network Weight Initialization

```rust
use hodu::prelude::*;

// Xavier/Glorot initialization
fn xavier_init(in_features: usize, out_features: usize) -> HoduResult<Tensor> {
    let std = (2.0 / (in_features + out_features) as f64).sqrt();
    Tensor::randn(&[out_features, in_features], 0.0, std)
}

// He initialization (for ReLU)
fn he_init(in_features: usize, out_features: usize) -> HoduResult<Tensor> {
    let std = (2.0 / in_features as f64).sqrt();
    Tensor::randn(&[out_features, in_features], 0.0, std)
}

// Usage
let weight = xavier_init(784, 128)?;
let bias = Tensor::zeros(&[128], DType::F32)?;
```

### Example 2: Batch Data Preparation

```rust
use hodu::prelude::*;

// Batch size 32, input dimension 784
let batch_size = 32;
let input_dim = 784;

// Generate input data (would load from dataset in practice)
let inputs = Tensor::randn(&[batch_size, input_dim], 0.0, 1.0)?;

// Generate labels (one-hot encoding, 10 classes)
let labels = Tensor::zeros(&[batch_size, 10], DType::F32)?;

// Generate mask (indicate padding positions)
let mask = Tensor::ones(&[batch_size, input_dim], DType::BOOL)?;
```

### Example 3: Using in Script Mode

```rust
use hodu::prelude::*;

// Build script
let builder = Builder::new("linear_model".to_string());
builder.start()?;

// Define input placeholder
let input = Tensor::input("input", &[100, 784])?;

// Weight and bias are dynamically created (contain actual data)
let weight = Tensor::randn(&[784, 10], 0.0, 0.1)?;
let bias = Tensor::zeros(&[10], DType::F32)?;

// Define operations
let logits = input.matmul(&weight)?.add(&bias)?;

builder.add_output("logits", logits)?;
builder.end()?;

let mut script = builder.build()?;

// Provide actual data
let input_data = Tensor::randn(&[100, 784], 0.0, 1.0)?;
script.add_input("input", input_data);

// Execute
let outputs = script.run()?;
```

### Example 4: Device Specification

```rust
use hodu::prelude::*;

// Create on CPU
set_runtime_device(Device::CPU);
let cpu_tensor = Tensor::randn(&[1000, 1000], 0.0, 1.0)?;

// Switch to CUDA GPU
#[cfg(feature = "cuda")]
{
    set_runtime_device(Device::CUDA(0));
    let gpu_tensor = Tensor::randn(&[1000, 1000], 0.0, 1.0)?;
    // gpu_tensor is created in GPU memory
}

// Runtime device is ignored in script mode
let builder = Builder::new("test".to_string());
builder.start()?;

set_runtime_device(Device::CUDA(0));  // Ignored!
let tensor = Tensor::zeros(&[10, 10], DType::F32)?;  // Created on CPU

builder.end()?;
```

## Important Notes

### 1. Runtime Device Ignored in Builder Mode

```rust
let builder = Builder::new("script".to_string());
builder.start()?;

set_runtime_device(Device::CUDA(0));  // Ignored!

// Always created on CPU while builder is active
let tensor = Tensor::randn(&[10, 10], 0.0, 1.0)?;
// -> Created on CPU

builder.end()?;

// Runtime device applied after builder ends
let tensor2 = Tensor::randn(&[10, 10], 0.0, 1.0)?;
// -> Created on CUDA(0)
```

### 2. input() Requires Builder

```rust
// Error!
let x = Tensor::input("x", &[10])?;
// Error: StaticTensorCreationRequiresBuilderContext

// Correct usage
let builder = Builder::new("script".to_string());
builder.start()?;
let x = Tensor::input("x", &[10])?;  // OK
```

### 3. Automatic DType Inference

```rust
// Inferred as f32
let t1 = Tensor::new(vec![1.0, 2.0, 3.0])?;  // DType::F32

// Inferred as i32
let t2 = Tensor::new(vec![1, 2, 3])?;  // DType::I32

// For full(), inferred from input type
let t3 = Tensor::full(&[3], 1.0)?;   // DType::F32
let t4 = Tensor::full(&[3], 1)?;     // DType::I32
```

### 4. _like Functions Copy Both Shape and DType

```rust
let original = Tensor::new(vec![1.0_f32, 2.0, 3.0])?;

let zeros = Tensor::zeros_like(&original)?;
// Shape: [3], DType: F32

let ones = Tensor::ones_like(&original)?;
// Shape: [3], DType: F32

let filled = Tensor::full_like(&original, 7.0)?;  // f32 auto-converts to Scalar::F32
// Shape: [3], DType: F32
```

## Summary

| Category | Functions | Use Case |
|----------|-----------|----------|
| **Data-based** | `new()` | When you have actual data |
| **Zero initialization** | `zeros()`, `zeros_like()` | Weight, buffer initialization |
| **Ones initialization** | `ones()`, `ones_like()` | Bias, masks |
| **Specific value** | `full()`, `full_like()` | When constant tensor is needed |
| **Normal random** | `randn()`, `randn_like()` | Neural network weight initialization |
| **Uniform random** | `rand_uniform()`, `rand_uniform_like()` | Dropout, data augmentation |
| **Conditional selection** | `where3_select()` | Select from tensors based on condition |
| **Placeholder** | `input()` | Define script inputs |

**Core Principles:**
- Dynamic execution: All creation functions available, runtime device applied
- During script building: Dynamic creation functions use CPU, use `input()` for placeholders
- Operational creation: Creates tensors from other tensors with gradient support
- Use dynamic creation when actual data is needed, use `input()` when only placeholder is needed
