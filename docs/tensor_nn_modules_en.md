# Neural Network Modules Guide

This document provides a comprehensive overview of neural network modules available in `hodu::nn`.

## Overview

`hodu::nn` provides PyTorch-style neural network building blocks organized into three main categories:

1. **Modules**: Layers and transformations (Linear, Activation functions)
2. **Loss Functions**: Training objectives (MSE, CrossEntropy, etc.)
3. **Optimizers**: Parameter update algorithms (SGD, Adam)

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

### Activation Function Comparison

| Activation | Range | Smooth | Zero-centered | Use Case |
|-----------|-------|--------|---------------|----------|
| ReLU | `[0, ∞)` | No | No | General purpose, fast |
| Sigmoid | `(0, 1)` | Yes | No | Binary classification output |
| Tanh | `(-1, 1)` | Yes | Yes | RNNs, hidden layers |
| GELU | `(-∞, ∞)` | Yes | Yes | Transformers, modern architectures |
| Softplus | `(0, ∞)` | Yes | No | Smooth ReLU alternative |
| LeakyReLU | `(-∞, ∞)` | No | Yes | Prevent dying ReLU |
| ELU | `(-α, ∞)` | Yes | No | Self-normalizing networks |

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
