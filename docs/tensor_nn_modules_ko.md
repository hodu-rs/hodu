# 신경망 모듈 가이드

이 문서는 `hodu::nn`에서 제공하는 신경망 모듈에 대한 포괄적인 개요를 제공합니다.

## 개요

`hodu::nn`은 PyTorch 스타일의 신경망 구성 요소를 세 가지 주요 카테고리로 제공합니다:

1. **모듈 (Modules)**: 레이어와 변환 (Linear, 활성화 함수)
2. **손실 함수 (Loss Functions)**: 학습 목표 (MSE, CrossEntropy 등)
3. **최적화기 (Optimizers)**: 파라미터 업데이트 알고리즘 (SGD, Adam)

## 모듈

### 선형 레이어

#### Linear

선택적 바이어스를 가진 완전 연결(dense) 레이어입니다.

```rust
use hodu::prelude::*;
use hodu::nn::modules::Linear;

// Linear 레이어 생성: 784 -> 128
let layer = Linear::new(784, 128, true, DType::F32)?;

// Forward pass
let input = Tensor::randn(&[32, 784], 0.0, 1.0)?;  // batch_size=32
let output = layer.forward(&input)?;  // [32, 128]
```

**파라미터:**
- `in_features`: 입력 차원
- `out_features`: 출력 차원
- `with_bias`: 바이어스 항 포함 여부
- `dtype`: 데이터 타입 (F32, F64 등)

**초기화:**
- Weight: Xavier/Glorot 초기화, `k = 1/√in_features`로 스케일링
- Bias: 활성화된 경우 동일한 초기화

**계산:**
```
output = input @ weight.T + bias
```

### 활성화 함수

모든 활성화 함수는 상태가 없으며 파라미터를 가지지 않습니다.

#### ReLU

정류 선형 유닛(Rectified Linear Unit): `max(0, x)`

```rust
use hodu::nn::modules::ReLU;

let relu = ReLU::new();
let output = relu.forward(&input)?;
```

**특성:**
- 범위: `[0, ∞)`
- Gradient: x > 0이면 1, 아니면 0
- 딥러닝에서 가장 일반적인 활성화 함수

#### Sigmoid

로지스틱 시그모이드: `σ(x) = 1 / (1 + e^(-x))`

```rust
use hodu::nn::modules::Sigmoid;

let sigmoid = Sigmoid::new();
let output = sigmoid.forward(&input)?;
```

**특성:**
- 범위: `(0, 1)`
- 이진 분류에 사용
- Gradient: `σ(x) * (1 - σ(x))`

#### Tanh

쌍곡 탄젠트: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

```rust
use hodu::nn::modules::Tanh;

let tanh = Tanh::new();
let output = tanh.forward(&input)?;
```

**특성:**
- 범위: `(-1, 1)`
- 제로 중심(sigmoid보다 우수)
- Gradient: `1 - tanh²(x)`

#### GELU

가우시안 오차 선형 유닛(Gaussian Error Linear Unit): `x * Φ(x)` (Φ는 표준 정규 CDF)

```rust
use hodu::nn::modules::Gelu;

let gelu = Gelu::new();
let output = gelu.forward(&input)?;
```

**특성:**
- ReLU의 부드러운 근사
- Transformer에서 사용 (BERT, GPT)
- 비단조(non-monotonic)

#### Softplus

ReLU의 부드러운 근사: `log(1 + e^x)`

```rust
use hodu::nn::modules::Softplus;

let softplus = Softplus::new();
let output = softplus.forward(&input)?;
```

**특성:**
- 범위: `(0, ∞)`
- 모든 곳에서 부드러움
- Gradient: sigmoid 함수

#### LeakyReLU

Leaky ReLU: `max(αx, x)`

```rust
use hodu::nn::modules::LeakyReLU;

let leaky = LeakyReLU::new(0.01);  // α = 0.01
let output = leaky.forward(&input)?;
```

**파라미터:**
- `exponent`: 음수 값에 대한 기울기 (일반적으로 0.01)

**특성:**
- "dying ReLU" 문제 방지
- x < 0에서 0이 아닌 gradient

#### ELU

지수 선형 유닛(Exponential Linear Unit)

```rust
use hodu::nn::modules::ELU;

let elu = ELU::new(1.0);  // α = 1.0
let output = elu.forward(&input)?;
```

**공식:**
```
f(x) = x           if x > 0
     = α(e^x - 1)  if x ≤ 0
```

**특성:**
- 모든 곳에서 부드러움
- 음수 값이 평균을 0으로 밀어냄
- 일부 네트워크에서 자기 정규화(self-normalizing)

### 활성화 함수 비교

| 활성화 함수 | 범위 | 부드러움 | 제로 중심 | 사용 사례 |
|-----------|-------|--------|-----------|----------|
| ReLU | `[0, ∞)` | 아니오 | 아니오 | 범용, 빠름 |
| Sigmoid | `(0, 1)` | 예 | 아니오 | 이진 분류 출력 |
| Tanh | `(-1, 1)` | 예 | 예 | RNN, 은닉층 |
| GELU | `(-∞, ∞)` | 예 | 예 | Transformer, 현대적 아키텍처 |
| Softplus | `(0, ∞)` | 예 | 아니오 | 부드러운 ReLU 대안 |
| LeakyReLU | `(-∞, ∞)` | 아니오 | 예 | Dying ReLU 방지 |
| ELU | `(-α, ∞)` | 예 | 아니오 | 자기 정규화 네트워크 |

## 손실 함수

### 회귀 손실

#### MSELoss

평균 제곱 오차(Mean Squared Error): 제곱 차이의 평균

```rust
use hodu::nn::losses::MSELoss;

let criterion = MSELoss::new();
let loss = criterion.forward((&predictions, &targets))?;
```

**공식:**
```
MSE = mean((predictions - targets)²)
```

**사용 사례:**
- 회귀 작업
- 이상치에 민감

#### MAELoss

평균 절대 오차(Mean Absolute Error): 절대 차이의 평균

```rust
use hodu::nn::losses::MAELoss;

let criterion = MAELoss::new();
let loss = criterion.forward((&predictions, &targets))?;
```

**공식:**
```
MAE = mean(|predictions - targets|)
```

**사용 사례:**
- 회귀 작업
- MSE보다 이상치에 강건

#### HuberLoss

설정 가능한 임계값을 가진 MSE와 MAE의 조합

```rust
use hodu::nn::losses::HuberLoss;

let criterion = HuberLoss::new(1.0);  // delta = 1.0
let loss = criterion.forward((&predictions, &targets))?;
```

**공식:**
```
L(x) = 0.5 * x²           if |x| ≤ δ
     = δ * (|x| - 0.5*δ)  if |x| > δ
여기서 x = predictions - targets
```

**사용 사례:**
- 이상치가 있는 회귀
- MSE와 MAE의 장점 균형

### 분류 손실

#### BCELoss

이진 교차 엔트로피(Binary Cross Entropy): 확률을 사용한 이진 분류

```rust
use hodu::nn::losses::BCELoss;

let criterion = BCELoss::new();
// 또는 수치 안정성을 위한 커스텀 epsilon
let criterion = BCELoss::with_epsilon(1e-7);

let predictions = predictions.sigmoid()?;  // 반드시 (0, 1) 범위
let loss = criterion.forward((&predictions, &targets))?;
```

**공식:**
```
BCE = -mean[target * log(pred) + (1 - target) * log(1 - pred)]
```

**특징:**
- 자동 clamping: `log(0)` 방지를 위해 `pred ∈ [ε, 1-ε]`
- 설정 가능한 epsilon (기본값: 1e-7)

**요구사항:**
- 예측값은 확률이어야 함 (0에서 1)
- 이 손실 전에 sigmoid 적용 필요

#### BCEWithLogitsLoss

로짓을 사용한 이진 교차 엔트로피: 더 수치적으로 안정적

```rust
use hodu::nn::losses::BCEWithLogitsLoss;

let criterion = BCEWithLogitsLoss::new();
let loss = criterion.forward((&logits, &targets))?;  // sigmoid 불필요
```

**공식:**
```
loss = max(x, 0) - x * target + log(1 + exp(-|x|))
여기서 x = logits
```

**장점:**
- BCELoss보다 수치적으로 더 안정적
- sigmoid + BCE를 하나의 연산으로 결합
- sigmoid를 별도로 적용할 필요 없음

#### NLLLoss

음의 로그 우도(Negative Log Likelihood): 로그 확률을 사용한 다중 클래스 분류

```rust
use hodu::nn::losses::NLLLoss;

let criterion = NLLLoss::new();  // 기본값: dim=-1
// 또는 클래스 차원 지정
let criterion = NLLLoss::with_dim(1);

let log_probs = logits.log_softmax(-1)?;
let loss = criterion.forward((&log_probs, &targets))?;
```

**파라미터:**
- `dim`: 클래스가 배열된 차원 (기본값: -1)

**공식:**
```
NLL = -mean(log_probs[batch_idx, target[batch_idx]])
```

**요구사항:**
- 입력은 로그 확률이어야 함
- 이 손실 전에 `log_softmax` 적용
- 타겟은 클래스 인덱스 (원-핫이 아님)

**Shape 예제:**
```rust
// 예제 1: 표준 분류
// log_probs: [batch=32, classes=10]
// targets: [batch=32], 값은 [0, 9] 범위

// 예제 2: 시퀀스 모델링
// log_probs: [batch=8, seq_len=50, vocab=1000]
// targets: [batch=8, seq_len=50]
// NLLLoss::with_dim(-1) 또는 with_dim(2) 사용
```

#### CrossEntropyLoss

교차 엔트로피: log_softmax + NLLLoss 결합

```rust
use hodu::nn::losses::CrossEntropyLoss;

let criterion = CrossEntropyLoss::new();  // 기본값: dim=-1
// 또는 클래스 차원 지정
let criterion = CrossEntropyLoss::with_dim(1);

let loss = criterion.forward((&logits, &targets))?;  // softmax 불필요
```

**파라미터:**
- `dim`: 클래스가 배열된 차원 (기본값: -1)

**공식:**
```
CE = -mean(log(softmax(logits))[batch_idx, target[batch_idx]])
```

**장점:**
- 분류에서 가장 일반적으로 사용
- softmax + log + NLLLoss보다 수치적으로 더 안정적
- 모든 것을 하나의 연산으로 결합

**Shape 예제:**
```rust
// 예제 1: 이미지 분류
// logits: [batch=32, classes=10]
// targets: [batch=32]

// 예제 2: 언어 모델링
// logits: [batch=4, seq_len=100, vocab=50000]
// targets: [batch=4, seq_len=100]
```

### 손실 함수 비교

| 손실 | 입력 타입 | 사용 사례 | 안정성 |
|------|-----------|----------|--------|
| MSELoss | 임의 | 회귀 | 좋음 |
| MAELoss | 임의 | 회귀 (강건) | 좋음 |
| HuberLoss | 임의 | 회귀 (이상치) | 우수 |
| BCELoss | 확률 (0-1) | 이진 분류 | clamping으로 좋음 |
| BCEWithLogitsLoss | 로짓 | 이진 분류 | 우수 |
| NLLLoss | 로그 확률 | 다중 클래스 | 좋음 |
| CrossEntropyLoss | 로짓 | 다중 클래스 | 우수 |

### 손실 선택 가이드

**회귀:**
- 기본값: `MSELoss`
- 이상치가 있는 경우: `HuberLoss` 또는 `MAELoss`

**이진 분류:**
- 기본값: `BCEWithLogitsLoss` (가장 안정적)
- 확률이 필요한 경우: `sigmoid()` + `BCELoss`

**다중 클래스 분류:**
- 기본값: `CrossEntropyLoss` (가장 편리하고 안정적)
- 확률이 필요한 경우: `log_softmax()` + `NLLLoss`

## 최적화기

### SGD

확률적 경사 하강법(Stochastic Gradient Descent)

```rust
use hodu::nn::optimizers::SGD;

let mut optimizer = SGD::new(0.01);  // learning_rate = 0.01

// 학습 루프
for epoch in 0..epochs {
    let loss = model.forward(&input)?;
    loss.backward()?;

    optimizer.step(&mut model.parameters())?;
    model.zero_grad()?;
}

// 학습률 조정
optimizer.set_learning_rate(0.001);
```

**업데이트 규칙:**
```
θ_new = θ_old - lr * ∇θ
```

**파라미터:**
- `learning_rate`: 스텝 크기 (일반적으로 0.001 - 0.1)

**특성:**
- 간단하고 빠름
- 볼록 문제에 잘 작동
- 학습률 스케줄링이 필요할 수 있음

### Adam

적응적 모멘트 추정(Adaptive Moment Estimation)

```rust
use hodu::nn::optimizers::Adam;

let mut optimizer = Adam::new(
    0.001,  // learning_rate
    0.9,    // beta1 (1차 모멘트 감쇠)
    0.999,  // beta2 (2차 모멘트 감쇠)
    1e-8,   // epsilon (수치 안정성)
);

// 학습 루프
for epoch in 0..epochs {
    let loss = model.forward(&input)?;
    loss.backward()?;

    optimizer.step(&mut model.parameters())?;
    model.zero_grad()?;
}
```

**업데이트 규칙:**
```
m_t = β₁ * m_(t-1) + (1 - β₁) * g_t
v_t = β₂ * v_(t-1) + (1 - β₂) * g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ_new = θ_old - lr * m̂_t / (√v̂_t + ε)
```

**파라미터:**
- `learning_rate`: 기본 학습률 (일반적으로 0.001)
- `beta1`: 1차 모멘트 감쇠 (일반적으로 0.9)
- `beta2`: 2차 모멘트 감쇠 (일반적으로 0.999)
- `epsilon`: 수치 안정성 (일반적으로 1e-8)

**특성:**
- 파라미터별 적응 학습률
- 실제로 잘 작동
- 대부분의 딥러닝 작업에서 기본 선택
- 초기 스텝에 대한 바이어스 보정 포함

### AdamW

분리된 가중치 감쇠를 가진 Adam

```rust
use hodu::nn::optimizers::AdamW;

let mut optimizer = AdamW::new(
    0.001,  // learning_rate
    0.9,    // beta1 (1차 모멘트 감쇠)
    0.999,  // beta2 (2차 모멘트 감쇠)
    1e-8,   // epsilon (수치 안정성)
    0.01,   // weight_decay
);

// 학습 루프
for epoch in 0..epochs {
    let loss = model.forward(&input)?;
    loss.backward()?;

    optimizer.step(&mut model.parameters())?;
    model.zero_grad()?;
}

// 하이퍼파라미터 조정
optimizer.set_learning_rate(0.0001);
optimizer.set_weight_decay(0.001);
```

**업데이트 규칙:**
```
θ = θ * (1 - lr * λ)                    // 가중치 감쇠 (분리됨)
m_t = β₁ * m_(t-1) + (1 - β₁) * g_t
v_t = β₂ * v_(t-1) + (1 - β₂) * g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ = θ - lr * m̂_t / (√v̂_t + ε)         // Gradient 업데이트
```

**파라미터:**
- `learning_rate`: 기본 학습률 (일반적으로 0.001)
- `beta1`: 1차 모멘트 감쇠 (일반적으로 0.9)
- `beta2`: 2차 모멘트 감쇠 (일반적으로 0.999)
- `epsilon`: 수치 안정성 (일반적으로 1e-8)
- `weight_decay`: L2 정규화 강도 (일반적으로 0.01)

**특성:**
- 가중치 감쇠를 gradient 기반 최적화와 분리
- 많은 경우 Adam보다 더 나은 일반화 성능
- 적응적 최적화기에서 L2 정규화 문제 해결
- Transformer 모델과 파인튜닝에 권장됨

**Adam과의 주요 차이:**
- Adam: 가중치 감쇠가 적응 학습률과 결합됨
- AdamW: 가중치 감쇠가 gradient 업데이트 전에 파라미터에 직접 적용됨
- 이를 통해 가중치 감쇠가 다양한 학습률에서 더 일관되게 동작함

### 최적화기 비교

| 최적화기 | 속도 | 메모리 | 하이퍼파라미터 | 수렴 | 가중치 감쇠 |
|----------|------|--------|----------------|------|------------|
| SGD | 빠름 | 낮음 | 1 (lr) | 튜닝 필요 | 내장 안됨 |
| Adam | 보통 | 높음 | 4 (lr, β₁, β₂, ε) | 강건, 빠름 | 결합됨 (부정확) |
| AdamW | 보통 | 높음 | 5 (lr, β₁, β₂, ε, λ) | 강건, 빠름 | 분리됨 (정확) |

**선택 가이드:**
- **기본 선택**: 기본 파라미터로 AdamW (대부분의 경우 최선)
- **정규화 없이**: 기본 파라미터로 Adam
- **메모리 제한**: SGD
- **사전 학습된 모델 파인튜닝**: 작은 가중치 감쇠로 AdamW
- **빠른 프로토타이핑**: AdamW 또는 Adam

## 완전한 예제

```rust
use hodu::prelude::*;
use hodu::nn::{modules::*, losses::*, optimizers::*};

fn main() -> HoduResult<()> {
    // 모델 레이어 생성
    let layer1 = Linear::new(784, 128, true, DType::F32)?;
    let relu = ReLU::new();
    let layer2 = Linear::new(128, 10, true, DType::F32)?;

    // 손실과 최적화기 생성
    let criterion = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    // 학습 루프
    for epoch in 0..10 {
        // Forward pass
        let input = Tensor::randn(&[32, 784], 0.0, 1.0)?;
        let targets = Tensor::new(vec![0i64, 1, 2, 3, 4, 5, 6, 7, 8, 9])?;

        let hidden = layer1.forward(&input)?;
        let activated = relu.forward(&hidden)?;
        let logits = layer2.forward(&activated)?;

        // 손실 계산
        let loss = criterion.forward((&logits, &targets))?;

        // Backward pass
        loss.backward()?;

        // 파라미터 업데이트
        let mut params = layer1.parameters();
        params.extend(layer2.parameters());
        optimizer.step(&mut params)?;

        // Gradient 초기화
        for param in params {
            param.zero_grad()?;
        }

        println!("Epoch {}: Loss = {}", epoch, loss);
    }

    Ok(())
}
```

## 참고사항

### 모듈 패턴

모든 모듈은 일관된 패턴을 따릅니다:

```rust
impl Module {
    pub fn new(...) -> Self { ... }
    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> { ... }
    pub fn parameters(&mut self) -> Vec<&mut Tensor> { ... }
}
```

### 손실 패턴

모든 손실은 `(&Tensor, &Tensor)` 튜플을 받습니다:

```rust
impl Loss {
    pub fn new() -> Self { ... }
    pub fn forward(&self, (prediction, target): (&Tensor, &Tensor)) -> HoduResult<Tensor> { ... }
}
```

### 최적화기 패턴

모든 최적화기는 가변 파라미터 슬라이스에서 작동합니다:

```rust
impl Optimizer {
    pub fn new(...) -> Self { ... }
    pub fn step(&mut self, parameters: &mut [&mut Tensor]) -> HoduResult<()> { ... }
    pub fn set_learning_rate(&mut self, lr: impl Into<Scalar>) { ... }
}
```

### Gradient 관리

다음을 기억하세요:
1. 손실 계산 후 `backward()` 호출
2. `optimizer.step()`을 호출하여 파라미터 업데이트
3. 다음 반복 전에 모든 파라미터에 대해 `zero_grad()` 호출

### 수치 안정성

분류의 경우:
- `sigmoid() + BCELoss` 대신 `BCEWithLogitsLoss` 사용
- `softmax() + log() + NLLLoss` 대신 `CrossEntropyLoss` 사용
- 이러한 결합된 연산이 수치적으로 더 안정적
