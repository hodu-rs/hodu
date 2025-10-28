# Neural Network Modules Guide

This document provides a comprehensive overview of neural network modules available in `hodu::nn`.

## Overview

`hodu::nn` provides PyTorch-style neural network building blocks organized into three main categories:

1. **Modules**: Layers and transformations (Linear, Activation functions)
2. **Loss Functions**: Training objectives (MSE, CrossEntropy, etc.)
3. **Optimizers**: Parameter update algorithms (SGD, Adam)

## Training/Evaluation Mode

Macros to switch the behavior mode of neural network modules.

### train!()

Switch to training mode. Regularization layers like Dropout are activated.

```rust
use hodu::prelude::*;

train!();  // Activate training mode
```

### eval!()

Switch to evaluation mode. Regularization layers like Dropout are deactivated.

```rust
use hodu::prelude::*;

eval!();  // Activate evaluation mode
```

**Usage Example:**

```rust
use hodu::prelude::*;

let dropout = Dropout::new(0.5);
let input = Tensor::randn(&[32, 128], 0.0, 1.0)?;

// During training
train!();
let output = dropout.forward(&input)?;  // Dropout applied

// During evaluation
eval!();
let output = dropout.forward(&input)?;  // Dropout not applied
```

## Modules

### Linear Layers

#### Linear

Fully connected (dense) layer with optional bias.

```rust
use hodu::prelude::*;
use hodu::nn::modules::Linear;

// Create linear layer: 784 -> 128
let layer = Linear::new(784, 128, true, DType::F32)?;

// Forward pass
let input = Tensor::randn(&[32, 784], 0.0, 1.0)?;  // batch_size=32
let output = layer.forward(&input)?;  // [32, 128]
```

**Parameters:**
- `in_features`: Input dimension
- `out_features`: Output dimension
- `with_bias`: Whether to include bias term
- `dtype`: Data type (F32, F64, etc.)

**Initialization:**
- Weight: Xavier/Glorot initialization scaled by `k = 1/√in_features`
- Bias: Same initialization if enabled

**Computation:**
```
output = input @ weight.T + bias
```

### Convolutional Layers

Convolutional layers are used for processing data with spatial structure such as images, time series, and 3D data.

#### Conv1D

1D convolutional layer for time series data or text sequences.

```rust
use hodu::nn::modules::Conv1D;

// Create Conv1D layer: 16 channels -> 32 channels, kernel_size=3
let conv = Conv1D::new(
    16,    // in_channels
    32,    // out_channels
    3,     // kernel_size
    1,     // stride
    1,     // padding
    1,     // dilation
    true,  // with_bias
    DType::F32
)?;

// Forward pass: [batch, channels, length]
let input = Tensor::randn(&[8, 16, 100], 0.0, 1.0)?;  // batch=8, in_channels=16, length=100
let output = conv.forward(&input)?;  // [8, 32, 100]
```

**Parameters:**
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels (number of filters)
- `kernel_size`: Size of the convolutional kernel
- `stride`: Stride of the convolution
- `padding`: Zero-padding added to input
- `dilation`: Spacing between kernel elements
- `with_bias`: Whether to include bias term
- `dtype`: Data type

**Initialization:**
- Weight: Kaiming initialization, `k = √(2/(in_channels * kernel_size))`
- Bias: Same initialization

**Output Size:**
```
L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

#### Conv2D

2D convolutional layer, most widely used for image processing.

```rust
use hodu::nn::modules::Conv2D;

// Create Conv2D layer: 3 channels (RGB) -> 64 channels, 3x3 kernel
let conv = Conv2D::new(
    3,     // in_channels
    64,    // out_channels
    3,     // kernel_size
    1,     // stride
    1,     // padding
    1,     // dilation
    true,  // with_bias
    DType::F32
)?;

// Forward pass: [batch, channels, height, width]
let input = Tensor::randn(&[16, 3, 224, 224], 0.0, 1.0)?;  // batch=16, RGB image 224x224
let output = conv.forward(&input)?;  // [16, 64, 224, 224]
```

**Parameters:** Same as Conv1D but applied in 2D space

**Initialization:**
- Weight: Kaiming initialization, `k = √(2/(in_channels * kernel_size²))`

**Output Size:**
```
H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
W_out = floor((W_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

#### Conv3D

3D convolutional layer for video or 3D medical imaging.

```rust
use hodu::nn::modules::Conv3D;

// Create Conv3D layer
let conv = Conv3D::new(
    1,     // in_channels (grayscale)
    32,    // out_channels
    3,     // kernel_size
    1,     // stride
    1,     // padding
    1,     // dilation
    true,  // with_bias
    DType::F32
)?;

// Forward pass: [batch, channels, depth, height, width]
let input = Tensor::randn(&[4, 1, 64, 64, 64], 0.0, 1.0)?;  // batch=4, 64x64x64 volume
let output = conv.forward(&input)?;  // [4, 32, 64, 64, 64]
```

**Initialization:**
- Weight: Kaiming initialization, `k = √(2/(in_channels * kernel_size³))`

**Output Size:**
```
D_out = floor((D_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
W_out = floor((W_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

#### ConvTranspose1D

Transposed convolution (upsampling) layer for increasing resolution of 1D data.

```rust
use hodu::nn::modules::ConvTranspose1D;

// Create ConvTranspose1D layer
let conv_t = ConvTranspose1D::new(
    32,    // in_channels
    16,    // out_channels
    3,     // kernel_size
    2,     // stride (upsampling factor)
    1,     // padding
    0,     // output_padding
    1,     // dilation
    true,  // with_bias
    DType::F32
)?;

// Forward pass
let input = Tensor::randn(&[8, 32, 50], 0.0, 1.0)?;  // [batch, channels, length]
let output = conv_t.forward(&input)?;  // [8, 16, 99] (upsampled)
```

**Output Size:**
```
L_out = (L_in - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
```

#### ConvTranspose2D

Transposed convolution layer for image upsampling, decoders, GAN generators, etc.

```rust
use hodu::nn::modules::ConvTranspose2D;

// Create ConvTranspose2D layer
let conv_t = ConvTranspose2D::new(
    64,    // in_channels
    3,     // out_channels (RGB)
    4,     // kernel_size
    2,     // stride (2x upsampling)
    1,     // padding
    0,     // output_padding
    1,     // dilation
    true,  // with_bias
    DType::F32
)?;

// Forward pass
let input = Tensor::randn(&[16, 64, 56, 56], 0.0, 1.0)?;
let output = conv_t.forward(&input)?;  // [16, 3, 112, 112] (2x upsampled)
```

**Output Size:**
```
H_out = (H_in - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
W_out = (W_in - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
```

#### ConvTranspose3D

3D transposed convolution layer for upsampling 3D data.

```rust
use hodu::nn::modules::ConvTranspose3D;

// Create ConvTranspose3D layer
let conv_t = ConvTranspose3D::new(
    32,    // in_channels
    1,     // out_channels
    4,     // kernel_size
    2,     // stride
    1,     // padding
    0,     // output_padding
    1,     // dilation
    true,  // with_bias
    DType::F32
)?;

// Forward pass
let input = Tensor::randn(&[4, 32, 32, 32, 32], 0.0, 1.0)?;
let output = conv_t.forward(&input)?;  // [4, 1, 64, 64, 64] (2x upsampled)
```

### Convolutional Layer Comparison

| Layer | Input Shape | Use Case | Resolution Change |
|-------|------------|----------|------------------|
| Conv1D | `[N, C, L]` | Time series, text | Decrease/maintain |
| Conv2D | `[N, C, H, W]` | Images | Decrease/maintain |
| Conv3D | `[N, C, D, H, W]` | Video, 3D scans | Decrease/maintain |
| ConvTranspose1D | `[N, C, L]` | 1D upsampling | Increase |
| ConvTranspose2D | `[N, C, H, W]` | Image generation | Increase |
| ConvTranspose3D | `[N, C, D, H, W]` | 3D reconstruction | Increase |

**Key Concepts:**
- **stride**: Larger values decrease (Conv) or increase (ConvTranspose) output size
- **padding**: Add zeros around input to control output size
- **dilation**: Spacing between kernel elements, expands receptive field
- **output_padding**: Fine-tune output size in ConvTranspose

### Pooling Layers

Pooling layers downsample spatial dimensions and provide translation invariance.

#### MaxPool1D, MaxPool2D, MaxPool3D

Max pooling selects the maximum value in each window.

```rust
use hodu::nn::modules::{MaxPool1d, MaxPool2d, MaxPool3d};

// MaxPool1D: Time series data
let pool1d = MaxPool1d::new(
    2,  // kernel_size
    2,  // stride
    0,  // padding
);
let input = Tensor::randn(&[8, 16, 100], 0.0, 1.0)?;  // [batch, channels, length]
let output = pool1d.forward(&input)?;  // [8, 16, 50]

// MaxPool2D: Images
let pool2d = MaxPool2d::new(
    2,  // kernel_size
    2,  // stride
    0,  // padding
);
let input = Tensor::randn(&[16, 64, 224, 224], 0.0, 1.0)?;  // [batch, channels, H, W]
let output = pool2d.forward(&input)?;  // [16, 64, 112, 112]

// MaxPool3D: 3D data
let pool3d = MaxPool3d::new(
    2,  // kernel_size
    2,  // stride
    0,  // padding
);
let input = Tensor::randn(&[4, 32, 64, 64, 64], 0.0, 1.0)?;  // [batch, channels, D, H, W]
let output = pool3d.forward(&input)?;  // [4, 32, 32, 32, 32]
```

**Parameters:**
- `kernel_size`: Size of the pooling window
- `stride`: Stride of the pooling operation
- `padding`: Padding size

**Properties:**
- Preserves only maximum values (useful for feature detection)
- Limited backpropagation (only max positions recorded)
- Reduces spatial dimensions

#### AvgPool1D, AvgPool2D, AvgPool3D

Average pooling computes the average of each window.

```rust
use hodu::nn::modules::{AvgPool1d, AvgPool2d, AvgPool3d};

// AvgPool1D
let pool = AvgPool1d::new(2, 2, 0);
let output = pool.forward(&input)?;

// AvgPool2D
let pool = AvgPool2d::new(2, 2, 0);
let output = pool.forward(&input)?;

// AvgPool3D
let pool = AvgPool3d::new(2, 2, 0);
let output = pool.forward(&input)?;
```

**Properties:**
- Averages all values in the window
- Smooth downsampling
- Less information loss than MaxPool

#### AdaptiveMaxPool1D, AdaptiveMaxPool2D, AdaptiveMaxPool3D

Adaptive max pooling produces fixed output size regardless of input size.

```rust
use hodu::nn::modules::{AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d};

// AdaptiveMaxPool1D
let pool = AdaptiveMaxPool1d::new(50);  // output_size
let input = Tensor::randn(&[8, 16, 100], 0.0, 1.0)?;
let output = pool.forward(&input)?;  // [8, 16, 50]

// AdaptiveMaxPool2D
let pool = AdaptiveMaxPool2d::new((7, 7));  // output_size (H, W)
let input = Tensor::randn(&[16, 512, 14, 14], 0.0, 1.0)?;
let output = pool.forward(&input)?;  // [16, 512, 7, 7]

// AdaptiveMaxPool3D
let pool = AdaptiveMaxPool3d::new((4, 4, 4));  // output_size (D, H, W)
let input = Tensor::randn(&[4, 64, 16, 16, 16], 0.0, 1.0)?;
let output = pool.forward(&input)?;  // [4, 64, 4, 4, 4]
```

**Parameters:**
- `output_size`: Desired output size (usize for 1D, tuple for 2D/3D)

**Properties:**
- Fixed output size regardless of input size
- Automatically calculates kernel size and stride
- Useful for networks that handle variable-sized inputs

#### AdaptiveAvgPool1D, AdaptiveAvgPool2D, AdaptiveAvgPool3D

Adaptive average pooling is the same as adaptive max pooling but uses averaging.

```rust
use hodu::nn::modules::{AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d};

// AdaptiveAvgPool1D
let pool = AdaptiveAvgPool1d::new(50);
let output = pool.forward(&input)?;

// AdaptiveAvgPool2D (e.g., Global Average Pooling in ResNet)
let pool = AdaptiveAvgPool2d::new((1, 1));  // Global pooling
let input = Tensor::randn(&[16, 2048, 7, 7], 0.0, 1.0)?;
let output = pool.forward(&input)?;  // [16, 2048, 1, 1]

// AdaptiveAvgPool3D
let pool = AdaptiveAvgPool3d::new((1, 1, 1));  // Global pooling
let output = pool.forward(&input)?;
```

**Use Cases:**
- Global Average Pooling (output size = 1x1)
- Final layer in classification networks
- Variable-sized input handling

### Pooling Layer Comparison

| Layer | Operation | Output Size | Backprop | Use Case |
|-------|-----------|-------------|----------|----------|
| MaxPool | Max | Computed | Limited | Feature detection, CNN |
| AvgPool | Mean | Computed | Full | Smooth downsampling |
| AdaptiveMaxPool | Max | Fixed | Limited | Variable-sized input |
| AdaptiveAvgPool | Mean | Fixed | Full | Global pooling, classification |

**Selection Guide:**
- **CNN intermediate layers**: MaxPool2d (feature detection)
- **Smooth downsampling**: AvgPool
- **Classification head**: AdaptiveAvgPool2d((1, 1)) (Global Average Pooling)
- **Variable-sized inputs**: Adaptive variants

### Embedding Layers

#### Embedding

Embedding layer converts integer indices to dense vectors. Primarily used in natural language processing to represent words as vectors.

```rust
use hodu::nn::modules::Embedding;

// Create Embedding layer: vocabulary size 10000, embedding dimension 256
let embedding = Embedding::new(
    10000,      // num_embeddings (vocabulary size)
    256,        // embedding_dim
    None,       // padding_idx (optional)
    None,       // max_norm (optional)
    2.0,        // norm_type (L2 norm)
    DType::F32
)?;

// Forward pass
let indices = Tensor::new(vec![5, 142, 8, 99])?;  // [batch_size]
let embedded = embedding.forward(&indices)?;  // [4, 256]

// For sequences
let indices = Tensor::new(vec![/* ... */])?.reshape(&[32, 50])?;  // [batch, seq_len]
let embedded = embedding.forward(&indices)?;  // [32, 50, 256]
```

**Parameters:**
- `num_embeddings`: Size of the embedding table (vocabulary size)
- `embedding_dim`: Dimension of each embedding vector
- `padding_idx`: Embeddings at this index are initialized to zeros and gradients are not updated (optional)
- `max_norm`: If specified, embedding vectors with norm exceeding this value are normalized (optional)
- `norm_type`: Norm type to use for max_norm (typically 2.0 for L2)
- `dtype`: Data type

**Initialization:**
- Weight: Xavier/Glorot initialization scaled by `k = 1/√embedding_dim`
- If padding_idx is specified, that embedding is initialized to zeros

**Shape:**
```
Input:  [batch_size] or [batch_size, seq_len] or any shape
Output: [...input_shape..., embedding_dim]
Weight: [num_embeddings, embedding_dim]
```

**Loading Pretrained Embeddings:**

```rust
// Create Embedding from pretrained weights
let pretrained_weight = Tensor::randn(&[10000, 256], 0.0, 1.0)?;
let embedding = Embedding::from_pretrained(
    pretrained_weight,
    false  // freeze: if true, weights are not updated
)?;
```

**Padding Handling:**

```rust
// Using padding_idx to handle padding tokens
let embedding = Embedding::new(
    10000,
    256,
    Some(0),    // Use index 0 as padding
    None,
    2.0,
    DType::F32
)?;

// Sequence with padding
let indices = Tensor::new(vec![5, 142, 0, 0, 8, 99])?;  // 0 is padding
let embedded = embedding.forward(&indices)?;  // Padding positions are zero vectors
```

**Max Norm Constraint:**

```rust
// Limit L2 norm of embedding vectors
let embedding = Embedding::new(
    10000,
    256,
    None,
    Some(1.0),  // max_norm: normalize if vector norm exceeds 1.0
    2.0,        // Use L2 norm
    DType::F32
)?;
```

**Features:**
- Efficient lookup of embedding vectors from indices (gather operation)
- Gradients for padding_idx are automatically set to zero
- max_norm prevents embeddings from growing too large (regularization effect)
- Can load pretrained embeddings (e.g., Word2Vec, GloVe)

**Use Cases:**
- Natural Language Processing: word, character, subword embeddings
- Recommendation systems: user/item embeddings
- Categorical feature encoding
- Knowledge Graph embeddings

**Example: Simple Text Classifier:**

```rust
use hodu::prelude::*;
use hodu::nn::modules::{Embedding, Linear, ReLU};

// Model setup
let embedding = Embedding::new(10000, 128, Some(0), None, 2.0, DType::F32)?;
let linear = Linear::new(128, 10, true, DType::F32)?;

// Forward pass
let input_ids = Tensor::new(vec![45, 123, 8, 0, 0])?.reshape(&[1, 5])?;  // [1, 5]
let embedded = embedding.forward(&input_ids)?;  // [1, 5, 128]

// Average pooling
let pooled = embedded.mean(&[1], false)?;  // [1, 128]
let logits = linear.forward(&pooled)?;  // [1, 10]
```

### Regularization Layers

#### Dropout

Randomly deactivates neurons to prevent overfitting.

```rust
use hodu::nn::modules::Dropout;

let dropout = Dropout::new(0.5);  // 50% drop probability
let output = dropout.forward(&input)?;
```

**Parameters:**
- `p`: Drop probability (0.0 ~ 1.0)

**Behavior:**
- Training: `output = input * mask * (1/(1-p))` (uniform distribution mask)
- Inference: `output = input` (no dropout)

**Recommended rates:** Hidden layers 0.3~0.5, Input layer 0.1~0.2

### Normalization Layers

Normalization layers stabilize training by normalizing layer inputs, enabling higher learning rates and faster convergence.

#### BatchNorm1D, BatchNorm2D, BatchNorm3D

Batch Normalization normalizes inputs using mini-batch statistics, reducing internal covariate shift.

```rust
use hodu::nn::modules::{BatchNorm1D, BatchNorm2D, BatchNorm3D};

// BatchNorm1D: For 2D inputs [N, C] or 3D inputs [N, C, L]
let bn1d = BatchNorm1D::new(
    128,        // num_features (number of channels)
    1e-5,       // eps (numerical stability)
    0.1,        // momentum (for running stats update)
    true,       // affine (learnable gamma/beta)
    DType::F32
)?;

// BatchNorm2D: For 4D inputs [N, C, H, W] (images)
let bn2d = BatchNorm2D::new(
    64,         // num_features
    1e-5,       // eps
    0.1,        // momentum
    true,       // affine
    DType::F32
)?;

// BatchNorm3D: For 5D inputs [N, C, D, H, W] (videos)
let bn3d = BatchNorm3D::new(
    32,         // num_features
    1e-5,       // eps
    0.1,        // momentum
    true,       // affine
    DType::F32
)?;

// Usage example
let input = Tensor::randn(&[16, 64, 32, 32], 0.0, 1.0)?;  // [N, C, H, W]
let output = bn2d.forward(&input)?;
```

**Parameters:**
- `num_features`: Number of channels (C dimension)
- `eps`: Small constant for numerical stability (default: 1e-5)
- `momentum`: Momentum for running statistics update (default: 0.1)
- `affine`: If true, learnable affine parameters (gamma, beta) are applied
- `dtype`: Data type

**Behavior:**

- **Training mode**: Normalizes using batch statistics and updates running statistics
  ```
  output = (input - batch_mean) / sqrt(batch_var + eps)
  output = gamma * output + beta  // if affine=true

  // Running statistics update (exponential moving average)
  running_mean = momentum * running_mean + (1 - momentum) * batch_mean
  running_var = momentum * running_var + (1 - momentum) * batch_var
  ```

- **Evaluation mode**: Normalizes using accumulated running statistics
  ```
  output = (input - running_mean) / sqrt(running_var + eps)
  output = gamma * output + beta  // if affine=true
  ```

**Input Shapes:**
- BatchNorm1D: `[N, C]` or `[N, C, L]`
  - Normalizes over N dimension (or [N, L] for 3D input)
- BatchNorm2D: `[N, C, H, W]`
  - Normalizes over [N, H, W] for each channel
- BatchNorm3D: `[N, C, D, H, W]`
  - Normalizes over [N, D, H, W] for each channel

**Use Cases:**
- CNN intermediate layers
- Stabilizing training in deep networks
- Enabling higher learning rates
- Reducing sensitivity to initialization

#### LayerNorm

Layer Normalization normalizes over feature dimensions instead of the batch dimension.

```rust
use hodu::nn::modules::LayerNorm;

// Single dimension
let ln = LayerNorm::new(
    vec![768],  // normalized_shape
    1e-5,       // eps
    true,       // elementwise_affine
    DType::F32
)?;

// Multiple dimensions (e.g., for images)
let ln_2d = LayerNorm::new(
    vec![64, 32, 32],  // normalized_shape: [C, H, W]
    1e-5,              // eps
    true,              // elementwise_affine
    DType::F32
)?;

// Usage example: Transformer
let input = Tensor::randn(&[32, 128, 768], 0.0, 1.0)?;  // [batch, seq_len, hidden_dim]
let output = ln.forward(&input)?;  // Normalizes over the last dimension (768)
```

**Parameters:**
- `normalized_shape`: Shape of dimensions to normalize over (Vec<usize>)
- `eps`: Small constant for numerical stability (default: 1e-5)
- `elementwise_affine`: If true, learnable affine parameters are applied
- `dtype`: Data type

**Behavior:**

LayerNorm always computes statistics from the current input (no running statistics).

```
mean = mean(input, dims=normalized_shape)
var = var(input, dims=normalized_shape)
output = (input - mean) / sqrt(var + eps)
output = gamma * output + beta  // if elementwise_affine=true
```

**Normalization axes:**
If `normalized_shape = [d1, d2, ..., dk]` and input shape is `[N, ..., d1, d2, ..., dk]`, normalization is performed over the last k dimensions.

**Example:**
```rust
// Input: [batch=32, seq_len=100, hidden=512]
// LayerNorm with normalized_shape=[512]
// -> Normalizes over the last dimension for each (batch, seq_len) position
let ln = LayerNorm::new(vec![512], 1e-5, true, DType::F32)?;
```

**Characteristics:**
- Batch-size independent
- No distinction between training/evaluation modes
- Preferred for RNNs and Transformers
- Suitable for small batch sizes or online learning

**Use Cases:**
- Transformer models (BERT, GPT, etc.)
- Recurrent Neural Networks (RNNs, LSTMs)
- Situations with variable batch sizes
- Small batch training

### Normalization Comparison

| Feature | BatchNorm | LayerNorm |
|---------|-----------|-----------|
| Normalization Axis | Batch dimension | Feature dimensions |
| Batch Dependency | Yes | No |
| Train/Eval Difference | Yes | No |
| Running Statistics | Yes | No |
| Best For | CNNs, fixed batch sizes | RNNs, Transformers, variable batch |
| Computational Cost | Lower | Higher |

**Selection Guide:**
- **CNN image processing**: BatchNorm2D
- **Transformers/NLP**: LayerNorm
- **RNNs/sequence modeling**: LayerNorm
- **Small or variable batch sizes**: LayerNorm
- **Large fixed batch sizes**: BatchNorm
- **Training stability issues**: Try LayerNorm

**Complete Example:**

```rust
use hodu::prelude::*;

// CNN with BatchNorm
let conv = Conv2D::new(3, 64, 3, 1, 1, 1, true, DType::F32)?;
let bn = BatchNorm2D::new(64, 1e-5, 0.1, true, DType::F32)?;
let relu = ReLU::new();

train!();  // Training mode
let x = Tensor::randn(&[16, 3, 224, 224], 0.0, 1.0)?;
let x = conv.forward(&x)?;
let x = bn.forward(&x)?;       // Uses batch statistics
let x = relu.forward(&x)?;

eval!();   // Evaluation mode
let x = bn.forward(&x)?;       // Uses running statistics

// Transformer with LayerNorm
let ln = LayerNorm::new(vec![512], 1e-5, true, DType::F32)?;
let x = Tensor::randn(&[32, 100, 512], 0.0, 1.0)?;
let x = ln.forward(&x)?;       // Same behavior in train/eval
```

### Activation Functions

All activation functions are stateless and have no parameters.

#### ReLU

Rectified Linear Unit: `max(0, x)`

```rust
use hodu::nn::modules::ReLU;

let relu = ReLU::new();
let output = relu.forward(&input)?;
```

**Properties:**
- Range: `[0, ∞)`
- Gradient: 1 if x > 0, else 0
- Most common activation in deep learning

#### Sigmoid

Logistic sigmoid: `σ(x) = 1 / (1 + e^(-x))`

```rust
use hodu::nn::modules::Sigmoid;

let sigmoid = Sigmoid::new();
let output = sigmoid.forward(&input)?;
```

**Properties:**
- Range: `(0, 1)`
- Used for binary classification
- Gradient: `σ(x) * (1 - σ(x))`

#### Tanh

Hyperbolic tangent: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

```rust
use hodu::nn::modules::Tanh;

let tanh = Tanh::new();
let output = tanh.forward(&input)?;
```

**Properties:**
- Range: `(-1, 1)`
- Zero-centered (better than sigmoid)
- Gradient: `1 - tanh²(x)`

#### GELU

Gaussian Error Linear Unit: `x * Φ(x)` where Φ is standard normal CDF

```rust
use hodu::nn::modules::Gelu;

let gelu = Gelu::new();
let output = gelu.forward(&input)?;
```

**Properties:**
- Smooth approximation of ReLU
- Used in Transformers (BERT, GPT)
- Non-monotonic

#### Softplus

Smooth approximation of ReLU: `log(1 + e^x)`

```rust
use hodu::nn::modules::Softplus;

let softplus = Softplus::new();
let output = softplus.forward(&input)?;
```

**Properties:**
- Range: `(0, ∞)`
- Smooth everywhere
- Gradient: sigmoid function

#### SiLU (Swish)

Sigmoid Linear Unit (also known as Swish): `x * σ(x)`

```rust
use hodu::nn::modules::SiLU;
// or
use hodu::nn::modules::Swish;

let silu = SiLU::new();
let output = silu.forward(&input)?;
```

**Formula:**
```
f(x) = x * sigmoid(x) = x / (1 + e^(-x))
```

**Properties:**
- Smooth and non-monotonic
- Self-gating activation
- Used in modern architectures (EfficientNet, etc.)
- Better than ReLU in many cases

#### Mish

Self-regularized non-monotonic activation: `x * tanh(softplus(x))`

```rust
use hodu::nn::modules::Mish;

let mish = Mish::new();
let output = mish.forward(&input)?;
```

**Formula:**
```
f(x) = x * tanh(ln(1 + e^x))
```

**Properties:**
- Smooth and non-monotonic
- Unbounded above, bounded below
- Improved accuracy over ReLU and Swish in some tasks
- Self-regularizing

#### LeakyReLU

Leaky ReLU: `max(αx, x)`

```rust
use hodu::nn::modules::LeakyReLU;

let leaky = LeakyReLU::new(0.01);  // α = 0.01
let output = leaky.forward(&input)?;
```

**Parameters:**
- `exponent`: Slope for negative values (typically 0.01)

**Properties:**
- Prevents "dying ReLU" problem
- Non-zero gradient for x < 0

#### ELU

Exponential Linear Unit

```rust
use hodu::nn::modules::ELU;

let elu = ELU::new(1.0);  // α = 1.0
let output = elu.forward(&input)?;
```

**Formula:**
```
f(x) = x           if x > 0
     = α(e^x - 1)  if x ≤ 0
```

**Properties:**
- Smooth everywhere
- Negative values push mean towards zero
- Self-normalizing in some networks

#### PReLU

Parametric ReLU with learnable slope

```rust
use hodu::nn::modules::PReLU;

let prelu = PReLU::new(0.25);  // Initial weight/slope
let output = prelu.forward(&input)?;
```

**Formula:**
```
f(x) = x           if x > 0
     = weight * x  if x ≤ 0
```

**Properties:**
- Learnable negative slope parameter
- Can adapt to data
- More flexible than LeakyReLU

#### RReLU

Randomized Leaky ReLU

```rust
use hodu::nn::modules::RReLU;

let rrelu = RReLU::new(0.125, 0.333);  // lower, upper bounds
let output = rrelu.forward(&input)?;
```

**Formula:**
```
f(x) = x           if x > 0
     = α * x       if x ≤ 0
where α is randomly sampled from uniform(lower, upper) during training
      α = (lower + upper) / 2 during inference
```

**Properties:**
- Randomized negative slope during training
- Acts as regularization
- Reduces overfitting

### Activation Function Comparison

| Activation | Range | Smooth | Zero-centered | Use Case |
|-----------|-------|--------|---------------|----------|
| ReLU | `[0, ∞)` | No | No | General purpose, fast |
| Sigmoid | `(0, 1)` | Yes | No | Binary classification output |
| Tanh | `(-1, 1)` | Yes | Yes | RNNs, hidden layers |
| GELU | `(-∞, ∞)` | Yes | Yes | Transformers, modern architectures |
| Softplus | `(0, ∞)` | Yes | No | Smooth ReLU alternative |
| SiLU/Swish | `(-∞, ∞)` | Yes | No | Modern CNNs (EfficientNet) |
| Mish | `(-∞, ∞)` | Yes | No | High accuracy tasks |
| LeakyReLU | `(-∞, ∞)` | No | Yes | Prevent dying ReLU |
| ELU | `(-α, ∞)` | Yes | No | Self-normalizing networks |
| PReLU | `(-∞, ∞)` | No | Yes | Learnable negative slope |
| RReLU | `(-∞, ∞)` | No | Yes | Regularization, reduce overfitting |

## Loss Functions

### Regression Losses

#### MSELoss

Mean Squared Error: Average of squared differences

```rust
use hodu::nn::losses::MSELoss;

let criterion = MSELoss::new();
let loss = criterion.forward((&predictions, &targets))?;
```

**Formula:**
```
MSE = mean((predictions - targets)²)
```

**Use Cases:**
- Regression tasks
- Sensitive to outliers

#### MAELoss

Mean Absolute Error: Average of absolute differences

```rust
use hodu::nn::losses::MAELoss;

let criterion = MAELoss::new();
let loss = criterion.forward((&predictions, &targets))?;
```

**Formula:**
```
MAE = mean(|predictions - targets|)
```

**Use Cases:**
- Regression tasks
- More robust to outliers than MSE

#### HuberLoss

Combination of MSE and MAE with configurable threshold

```rust
use hodu::nn::losses::HuberLoss;

let criterion = HuberLoss::new(1.0);  // delta = 1.0
let loss = criterion.forward((&predictions, &targets))?;
```

**Formula:**
```
L(x) = 0.5 * x²           if |x| ≤ δ
     = δ * (|x| - 0.5*δ)  if |x| > δ
where x = predictions - targets
```

**Use Cases:**
- Regression with outliers
- Balances MSE and MAE benefits

### Classification Losses

#### BCELoss

Binary Cross Entropy: For binary classification with probabilities

```rust
use hodu::nn::losses::BCELoss;

let criterion = BCELoss::new();
// Or with custom epsilon for numerical stability
let criterion = BCELoss::with_epsilon(1e-7);

let predictions = predictions.sigmoid()?;  // Must be in (0, 1)
let loss = criterion.forward((&predictions, &targets))?;
```

**Formula:**
```
BCE = -mean[target * log(pred) + (1 - target) * log(1 - pred)]
```

**Features:**
- Automatic clamping: `pred ∈ [ε, 1-ε]` to avoid `log(0)`
- Configurable epsilon (default: 1e-7)

**Requirements:**
- Predictions must be probabilities (0 to 1)
- Apply sigmoid before this loss

#### BCEWithLogitsLoss

Binary Cross Entropy with logits: More numerically stable

```rust
use hodu::nn::losses::BCEWithLogitsLoss;

let criterion = BCEWithLogitsLoss::new();
let loss = criterion.forward((&logits, &targets))?;  // No sigmoid needed
```

**Formula:**
```
loss = max(x, 0) - x * target + log(1 + exp(-|x|))
where x = logits
```

**Advantages:**
- More numerically stable than BCELoss
- Combines sigmoid + BCE in one operation
- No need to apply sigmoid separately

#### NLLLoss

Negative Log Likelihood: For multi-class classification with log probabilities

```rust
use hodu::nn::losses::NLLLoss;

let criterion = NLLLoss::new();  // Default: dim=-1
// Or specify class dimension
let criterion = NLLLoss::with_dim(1);

let log_probs = logits.log_softmax(-1)?;
let loss = criterion.forward((&log_probs, &targets))?;
```

**Parameters:**
- `dim`: Dimension along which classes are arranged (default: -1)

**Formula:**
```
NLL = -mean(log_probs[batch_idx, target[batch_idx]])
```

**Requirements:**
- Input must be log probabilities
- Apply `log_softmax` before this loss
- Targets are class indices (not one-hot)

**Shape Examples:**
```rust
// Example 1: Standard classification
// log_probs: [batch=32, classes=10]
// targets: [batch=32] with values in [0, 9]

// Example 2: Sequence modeling
// log_probs: [batch=8, seq_len=50, vocab=1000]
// targets: [batch=8, seq_len=50]
// Use NLLLoss::with_dim(-1) or with_dim(2)
```

#### CrossEntropyLoss

Cross Entropy: Combines log_softmax + NLLLoss

```rust
use hodu::nn::losses::CrossEntropyLoss;

let criterion = CrossEntropyLoss::new();  // Default: dim=-1
// Or specify class dimension
let criterion = CrossEntropyLoss::with_dim(1);

let loss = criterion.forward((&logits, &targets))?;  // No softmax needed
```

**Parameters:**
- `dim`: Dimension along which classes are arranged (default: -1)

**Formula:**
```
CE = -mean(log(softmax(logits))[batch_idx, target[batch_idx]])
```

**Advantages:**
- Most commonly used for classification
- More numerically stable than softmax + log + NLLLoss
- Combines everything in one operation

**Shape Examples:**
```rust
// Example 1: Image classification
// logits: [batch=32, classes=10]
// targets: [batch=32]

// Example 2: Language modeling
// logits: [batch=4, seq_len=100, vocab=50000]
// targets: [batch=4, seq_len=100]
```

### Loss Function Comparison

| Loss | Input Type | Use Case | Stability |
|------|-----------|----------|-----------|
| MSELoss | Any | Regression | Good |
| MAELoss | Any | Regression (robust) | Good |
| HuberLoss | Any | Regression (outliers) | Excellent |
| BCELoss | Probabilities (0-1) | Binary classification | Good with clamping |
| BCEWithLogitsLoss | Logits | Binary classification | Excellent |
| NLLLoss | Log probabilities | Multi-class | Good |
| CrossEntropyLoss | Logits | Multi-class | Excellent |

### Loss Selection Guide

**For Regression:**
- Default: `MSELoss`
- With outliers: `HuberLoss` or `MAELoss`

**For Binary Classification:**
- Default: `BCEWithLogitsLoss` (most stable)
- If you need probabilities: `sigmoid()` + `BCELoss`

**For Multi-class Classification:**
- Default: `CrossEntropyLoss` (most convenient and stable)
- If you need probabilities: `log_softmax()` + `NLLLoss`

## Optimizers

### SGD

Stochastic Gradient Descent

```rust
use hodu::nn::optimizers::SGD;

let mut optimizer = SGD::new(0.01);  // learning_rate = 0.01

// Training loop
for epoch in 0..epochs {
    let loss = model.forward(&input)?;
    loss.backward()?;

    optimizer.step(&mut model.parameters())?;
    model.zero_grad()?;
}

// Adjust learning rate
optimizer.set_learning_rate(0.001);
```

**Update Rule:**
```
θ_new = θ_old - lr * ∇θ
```

**Parameters:**
- `learning_rate`: Step size (typically 0.001 - 0.1)

**Characteristics:**
- Simple and fast
- Works well for convex problems
- May need learning rate scheduling

### Adam

Adaptive Moment Estimation

```rust
use hodu::nn::optimizers::Adam;

let mut optimizer = Adam::new(
    0.001,  // learning_rate
    0.9,    // beta1 (first moment decay)
    0.999,  // beta2 (second moment decay)
    1e-8,   // epsilon (numerical stability)
);

// Training loop
for epoch in 0..epochs {
    let loss = model.forward(&input)?;
    loss.backward()?;

    optimizer.step(&mut model.parameters())?;
    model.zero_grad()?;
}
```

**Update Rules:**
```
m_t = β₁ * m_(t-1) + (1 - β₁) * g_t
v_t = β₂ * v_(t-1) + (1 - β₂) * g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ_new = θ_old - lr * m̂_t / (√v̂_t + ε)
```

**Parameters:**
- `learning_rate`: Base learning rate (typically 0.001)
- `beta1`: First moment decay (typically 0.9)
- `beta2`: Second moment decay (typically 0.999)
- `epsilon`: Numerical stability (typically 1e-8)

**Characteristics:**
- Adaptive learning rates per parameter
- Works well in practice
- Default choice for most deep learning tasks
- Includes bias correction for initial steps

### AdamW

Adam with Decoupled Weight Decay

```rust
use hodu::nn::optimizers::AdamW;

let mut optimizer = AdamW::new(
    0.001,  // learning_rate
    0.9,    // beta1 (first moment decay)
    0.999,  // beta2 (second moment decay)
    1e-8,   // epsilon (numerical stability)
    0.01,   // weight_decay
);

// Training loop
for epoch in 0..epochs {
    let loss = model.forward(&input)?;
    loss.backward()?;

    optimizer.step(&mut model.parameters())?;
    model.zero_grad()?;
}

// Adjust hyperparameters
optimizer.set_learning_rate(0.0001);
optimizer.set_weight_decay(0.001);
```

**Update Rules:**
```
θ = θ * (1 - lr * λ)                    // Weight decay (decoupled)
m_t = β₁ * m_(t-1) + (1 - β₁) * g_t
v_t = β₂ * v_(t-1) + (1 - β₂) * g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ = θ - lr * m̂_t / (√v̂_t + ε)         // Gradient update
```

**Parameters:**
- `learning_rate`: Base learning rate (typically 0.001)
- `beta1`: First moment decay (typically 0.9)
- `beta2`: Second moment decay (typically 0.999)
- `epsilon`: Numerical stability (typically 1e-8)
- `weight_decay`: L2 regularization strength (typically 0.01)

**Characteristics:**
- Decouples weight decay from gradient-based optimization
- Better generalization than Adam in many cases
- Fixes issues with L2 regularization in adaptive optimizers
- Recommended for transformer models and fine-tuning

**Key Difference from Adam:**
- Adam: Weight decay is coupled with adaptive learning rate
- AdamW: Weight decay is applied directly to parameters before gradient update
- This makes weight decay behave more consistently across different learning rates

### Optimizer Comparison

| Optimizer | Speed | Memory | Hyperparameters | Convergence | Weight Decay |
|-----------|-------|--------|-----------------|-------------|--------------|
| SGD | Fast | Low | 1 (lr) | Requires tuning | Not built-in |
| Adam | Medium | High | 4 (lr, β₁, β₂, ε) | Robust, fast | Coupled (incorrect) |
| AdamW | Medium | High | 5 (lr, β₁, β₂, ε, λ) | Robust, fast | Decoupled (correct) |

**Selection Guide:**
- **Default choice**: AdamW with default parameters (best for most cases)
- **Without regularization**: Adam with default parameters
- **Limited memory**: SGD
- **Fine-tuning pretrained models**: AdamW with small weight decay
- **Fast prototyping**: AdamW or Adam

## Complete Example

```rust
use hodu::prelude::*;
use hodu::nn::{modules::*, losses::*, optimizers::*};

fn main() -> HoduResult<()> {
    // Create model layers
    let layer1 = Linear::new(784, 128, true, DType::F32)?;
    let relu = ReLU::new();
    let layer2 = Linear::new(128, 10, true, DType::F32)?;

    // Create loss and optimizer
    let criterion = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    // Training loop
    for epoch in 0..10 {
        // Forward pass
        let input = Tensor::randn(&[32, 784], 0.0, 1.0)?;
        let targets = Tensor::new(vec![0i64, 1, 2, 3, 4, 5, 6, 7, 8, 9])?;

        let hidden = layer1.forward(&input)?;
        let activated = relu.forward(&hidden)?;
        let logits = layer2.forward(&activated)?;

        // Compute loss
        let loss = criterion.forward((&logits, &targets))?;

        // Backward pass
        loss.backward()?;

        // Update parameters
        let mut params = layer1.parameters();
        params.extend(layer2.parameters());
        optimizer.step(&mut params)?;

        // Zero gradients
        for param in params {
            param.zero_grad()?;
        }

        println!("Epoch {}: Loss = {}", epoch, loss);
    }

    Ok(())
}
```

## Notes

### Module Pattern

All modules follow a consistent pattern:

```rust
impl Module {
    pub fn new(...) -> Self { ... }
    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> { ... }
    pub fn parameters(&mut self) -> Vec<&mut Tensor> { ... }
}
```

### Loss Pattern

All losses accept `(&Tensor, &Tensor)` tuple:

```rust
impl Loss {
    pub fn new() -> Self { ... }
    pub fn forward(&self, (prediction, target): (&Tensor, &Tensor)) -> HoduResult<Tensor> { ... }
}
```

### Optimizer Pattern

All optimizers operate on mutable parameter slices:

```rust
impl Optimizer {
    pub fn new(...) -> Self { ... }
    pub fn step(&mut self, parameters: &mut [&mut Tensor]) -> HoduResult<()> { ... }
    pub fn set_learning_rate(&mut self, lr: impl Into<Scalar>) { ... }
}
```

### Gradient Management

Remember to:
1. Call `backward()` after computing loss
2. Call `optimizer.step()` to update parameters
3. Call `zero_grad()` on all parameters before next iteration

### Numerical Stability

For classification:
- Use `BCEWithLogitsLoss` instead of `sigmoid() + BCELoss`
- Use `CrossEntropyLoss` instead of `softmax() + log() + NLLLoss`
- These combined operations are more numerically stable
