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

### Optimizer Comparison

| Optimizer | Speed | Memory | Hyperparameters | Convergence |
|-----------|-------|--------|-----------------|-------------|
| SGD | Fast | Low | 1 (lr) | Requires tuning |
| Adam | Medium | High | 4 (lr, β₁, β₂, ε) | Robust, fast |

**Selection Guide:**
- **Default choice**: Adam with default parameters
- **Limited memory**: SGD
- **Fine-tuning**: SGD with momentum (not yet implemented)
- **Fast prototyping**: Adam

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
