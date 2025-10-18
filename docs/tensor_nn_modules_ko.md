# 신경망 모듈 가이드

이 문서는 `hodu::nn`에서 제공하는 신경망 모듈에 대한 포괄적인 개요를 제공합니다.

## 개요

`hodu::nn`은 PyTorch 스타일의 신경망 구성 요소를 세 가지 주요 카테고리로 제공합니다:

1. **모듈 (Modules)**: 레이어와 변환 (Linear, 활성화 함수)
2. **손실 함수 (Loss Functions)**: 학습 목표 (MSE, CrossEntropy 등)
3. **최적화기 (Optimizers)**: 파라미터 업데이트 알고리즘 (SGD, Adam)

## 학습/평가 모드 전환

신경망 모듈의 동작 모드를 전환하는 매크로입니다.

### train!()

학습 모드로 전환합니다. Dropout 등의 정규화 레이어가 활성화됩니다.

```rust
use hodu::prelude::*;

train!();  // 학습 모드 활성화
```

### eval!()

평가 모드로 전환합니다. Dropout 등의 정규화 레이어가 비활성화됩니다.

```rust
use hodu::prelude::*;

eval!();  // 평가 모드 활성화
```

**사용 예시:**

```rust
use hodu::prelude::*;

let dropout = Dropout::new(0.5);
let input = Tensor::randn(&[32, 128], 0.0, 1.0)?;

// 학습 시
train!();
let output = dropout.forward(&input)?;  // Dropout 적용됨

// 평가 시
eval!();
let output = dropout.forward(&input)?;  // Dropout 적용 안됨
```

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

### 합성곱 레이어 (Convolutional Layers)

합성곱 레이어는 이미지, 시계열, 3D 데이터 등의 공간적 구조를 가진 데이터 처리에 사용됩니다.

#### Conv1D

1차원 합성곱 레이어로 시계열 데이터나 텍스트 시퀀스 처리에 사용됩니다.

```rust
use hodu::nn::modules::Conv1D;

// Conv1D 레이어 생성: 16 channels -> 32 channels, kernel_size=3
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

**파라미터:**
- `in_channels`: 입력 채널 수
- `out_channels`: 출력 채널 수 (필터 개수)
- `kernel_size`: 커널 크기
- `stride`: 이동 보폭
- `padding`: 패딩 크기
- `dilation`: 확장(dilation) 비율
- `with_bias`: 바이어스 항 포함 여부
- `dtype`: 데이터 타입

**초기화:**
- Weight: Kaiming 초기화, `k = √(2/(in_channels * kernel_size))`
- Bias: 동일한 초기화

**출력 크기:**
```
L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

#### Conv2D

2차원 합성곱 레이어로 이미지 처리에 가장 널리 사용됩니다.

```rust
use hodu::nn::modules::Conv2D;

// Conv2D 레이어 생성: 3 channels (RGB) -> 64 channels, 3x3 kernel
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

**파라미터:** Conv1D와 동일하지만 2D 공간에 적용

**초기화:**
- Weight: Kaiming 초기화, `k = √(2/(in_channels * kernel_size²))`

**출력 크기:**
```
H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
W_out = floor((W_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

#### Conv3D

3차원 합성곱 레이어로 비디오나 3D 의료 이미지 처리에 사용됩니다.

```rust
use hodu::nn::modules::Conv3D;

// Conv3D 레이어 생성
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

**초기화:**
- Weight: Kaiming 초기화, `k = √(2/(in_channels * kernel_size³))`

**출력 크기:**
```
D_out = floor((D_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
W_out = floor((W_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

#### ConvTranspose1D

전치 합성곱(업샘플링) 레이어로 1D 데이터의 해상도를 높입니다.

```rust
use hodu::nn::modules::ConvTranspose1D;

// ConvTranspose1D 레이어 생성
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

**출력 크기:**
```
L_out = (L_in - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
```

#### ConvTranspose2D

전치 합성곱 레이어로 이미지 업샘플링, 디코더, GAN 생성기 등에 사용됩니다.

```rust
use hodu::nn::modules::ConvTranspose2D;

// ConvTranspose2D 레이어 생성
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
let output = conv_t.forward(&input)?;  // [16, 3, 112, 112] (2배 업샘플링)
```

**출력 크기:**
```
H_out = (H_in - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
W_out = (W_in - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
```

#### ConvTranspose3D

3D 전치 합성곱 레이어로 3D 데이터의 업샘플링에 사용됩니다.

```rust
use hodu::nn::modules::ConvTranspose3D;

// ConvTranspose3D 레이어 생성
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
let output = conv_t.forward(&input)?;  // [4, 1, 64, 64, 64] (2배 업샘플링)
```

### 합성곱 레이어 비교

| 레이어 | 입력 형태 | 사용 사례 | 해상도 변화 |
|--------|----------|----------|------------|
| Conv1D | `[N, C, L]` | 시계열, 텍스트 | 감소/유지 |
| Conv2D | `[N, C, H, W]` | 이미지 | 감소/유지 |
| Conv3D | `[N, C, D, H, W]` | 비디오, 3D 스캔 | 감소/유지 |
| ConvTranspose1D | `[N, C, L]` | 1D 업샘플링 | 증가 |
| ConvTranspose2D | `[N, C, H, W]` | 이미지 생성 | 증가 |
| ConvTranspose3D | `[N, C, D, H, W]` | 3D 재구성 | 증가 |

**주요 개념:**
- **stride**: 큰 값일수록 출력 크기가 작아짐 (Conv) 또는 커짐 (ConvTranspose)
- **padding**: 입력 주변에 0을 추가하여 출력 크기 조절
- **dilation**: 커널 요소 사이의 간격, 수용 영역(receptive field)을 확장
- **output_padding**: ConvTranspose에서 출력 크기를 미세 조정

### 풀링 레이어 (Pooling Layers)

풀링 레이어는 공간 차원을 다운샘플링하고 평행 이동 불변성(translation invariance)을 제공합니다.

#### MaxPool1D, MaxPool2D, MaxPool3D

최대 풀링은 각 윈도우에서 최대값을 선택합니다.

```rust
use hodu::nn::modules::{MaxPool1d, MaxPool2d, MaxPool3d};

// MaxPool1D: 시계열 데이터
let pool1d = MaxPool1d::new(
    2,  // kernel_size
    2,  // stride
    0,  // padding
);
let input = Tensor::randn(&[8, 16, 100], 0.0, 1.0)?;  // [batch, channels, length]
let output = pool1d.forward(&input)?;  // [8, 16, 50]

// MaxPool2D: 이미지
let pool2d = MaxPool2d::new(
    2,  // kernel_size
    2,  // stride
    0,  // padding
);
let input = Tensor::randn(&[16, 64, 224, 224], 0.0, 1.0)?;  // [batch, channels, H, W]
let output = pool2d.forward(&input)?;  // [16, 64, 112, 112]

// MaxPool3D: 3D 데이터
let pool3d = MaxPool3d::new(
    2,  // kernel_size
    2,  // stride
    0,  // padding
);
let input = Tensor::randn(&[4, 32, 64, 64, 64], 0.0, 1.0)?;  // [batch, channels, D, H, W]
let output = pool3d.forward(&input)?;  // [4, 32, 32, 32, 32]
```

**파라미터:**
- `kernel_size`: 풀링 윈도우 크기
- `stride`: 이동 보폭
- `padding`: 패딩 크기

**특성:**
- 최대값만 보존 (특징 검출에 유용)
- 역전파 불가능 (최대값 위치만 기록)
- 공간 차원 축소

#### AvgPool1D, AvgPool2D, AvgPool3D

평균 풀링은 각 윈도우의 평균을 계산합니다.

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

**특성:**
- 윈도우의 모든 값을 평균화
- 부드러운 다운샘플링
- MaxPool보다 정보 손실이 적음

#### AdaptiveMaxPool1D, AdaptiveMaxPool2D, AdaptiveMaxPool3D

적응형 최대 풀링은 입력 크기와 관계없이 고정된 출력 크기를 생성합니다.

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

**파라미터:**
- `output_size`: 원하는 출력 크기 (1D의 경우 usize, 2D/3D의 경우 튜플)

**특성:**
- 입력 크기와 무관하게 고정된 출력 크기
- 커널 크기와 stride를 자동으로 계산
- 가변 크기 입력을 처리하는 네트워크에 유용

#### AdaptiveAvgPool1D, AdaptiveAvgPool2D, AdaptiveAvgPool3D

적응형 평균 풀링은 적응형 최대 풀링과 동일하지만 평균을 사용합니다.

```rust
use hodu::nn::modules::{AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d};

// AdaptiveAvgPool1D
let pool = AdaptiveAvgPool1d::new(50);
let output = pool.forward(&input)?;

// AdaptiveAvgPool2D (ResNet의 Global Average Pooling 등)
let pool = AdaptiveAvgPool2d::new((1, 1));  // Global pooling
let input = Tensor::randn(&[16, 2048, 7, 7], 0.0, 1.0)?;
let output = pool.forward(&input)?;  // [16, 2048, 1, 1]

// AdaptiveAvgPool3D
let pool = AdaptiveAvgPool3d::new((1, 1, 1));  // Global pooling
let output = pool.forward(&input)?;
```

**사용 사례:**
- Global Average Pooling (출력 크기 = 1x1)
- 분류 네트워크의 최종 레이어
- 가변 크기 입력 처리

### 풀링 레이어 비교

| 레이어 | 연산 | 출력 크기 | 역전파 | 사용 사례 |
|--------|------|----------|--------|----------|
| MaxPool | Max | 계산됨 | 제한적 | 특징 검출, CNN |
| AvgPool | Mean | 계산됨 | 가능 | 부드러운 다운샘플링 |
| AdaptiveMaxPool | Max | 고정됨 | 제한적 | 가변 크기 입력 |
| AdaptiveAvgPool | Mean | 고정됨 | 가능 | Global pooling, 분류 |

**선택 가이드:**
- **CNN 중간층**: MaxPool2d (특징 검출)
- **부드러운 다운샘플링**: AvgPool
- **분류 헤드**: AdaptiveAvgPool2d((1, 1)) (Global Average Pooling)
- **가변 크기 입력**: Adaptive 변형

### 임베딩 레이어

#### Embedding

임베딩 레이어는 정수 인덱스를 밀집 벡터로 변환합니다. 자연어 처리에서 단어를 벡터로 표현하는 데 주로 사용됩니다.

```rust
use hodu::nn::modules::Embedding;

// Embedding 레이어 생성: 어휘 크기 10000, 임베딩 차원 256
let embedding = Embedding::new(
    10000,      // num_embeddings (어휘 크기)
    256,        // embedding_dim
    None,       // padding_idx (선택적)
    None,       // max_norm (선택적)
    2.0,        // norm_type (L2 norm)
    DType::F32
)?;

// Forward pass
let indices = Tensor::new(vec![5, 142, 8, 99])?;  // [batch_size]
let embedded = embedding.forward(&indices)?;  // [4, 256]

// 시퀀스의 경우
let indices = Tensor::new(vec![/* ... */])?.reshape(&[32, 50])?;  // [batch, seq_len]
let embedded = embedding.forward(&indices)?;  // [32, 50, 256]
```

**파라미터:**
- `num_embeddings`: 임베딩 테이블의 크기 (어휘 크기)
- `embedding_dim`: 각 임베딩 벡터의 차원
- `padding_idx`: 이 인덱스의 임베딩은 0으로 초기화되고 gradient가 업데이트되지 않음 (선택적)
- `max_norm`: 지정된 경우, 임베딩 벡터의 norm이 이 값을 초과하면 정규화됨 (선택적)
- `norm_type`: max_norm 적용 시 사용할 norm 타입 (일반적으로 2.0 for L2)
- `dtype`: 데이터 타입

**초기화:**
- Weight: Xavier/Glorot 초기화, `k = 1/√embedding_dim`로 스케일링
- padding_idx가 지정된 경우 해당 임베딩은 0으로 초기화

**Shape:**
```
입력:  [batch_size] 또는 [batch_size, seq_len] 또는 임의 shape
출력:  [...input_shape..., embedding_dim]
Weight: [num_embeddings, embedding_dim]
```

**사전 학습된 임베딩 로드:**

```rust
// 사전 학습된 가중치에서 Embedding 생성
let pretrained_weight = Tensor::randn(&[10000, 256], 0.0, 1.0)?;
let embedding = Embedding::from_pretrained(
    pretrained_weight,
    false  // freeze: true면 가중치 업데이트 안됨
)?;
```

**Padding 처리:**

```rust
// padding_idx를 사용하여 패딩 토큰 처리
let embedding = Embedding::new(
    10000,
    256,
    Some(0),    // 인덱스 0을 패딩으로 사용
    None,
    2.0,
    DType::F32
)?;

// 패딩이 포함된 시퀀스
let indices = Tensor::new(vec![5, 142, 0, 0, 8, 99])?;  // 0은 패딩
let embedded = embedding.forward(&indices)?;  // 패딩 위치는 0 벡터
```

**Max Norm 제약:**

```rust
// 임베딩 벡터의 L2 norm을 제한
let embedding = Embedding::new(
    10000,
    256,
    None,
    Some(1.0),  // max_norm: 벡터 norm이 1.0을 초과하면 정규화
    2.0,        // L2 norm 사용
    DType::F32
)?;
```

**특징:**
- 인덱스에서 임베딩 벡터를 효율적으로 조회 (gather 연산)
- padding_idx의 gradient는 자동으로 0으로 설정됨
- max_norm은 임베딩이 너무 커지는 것을 방지 (정규화 효과)
- 사전 학습된 임베딩(예: Word2Vec, GloVe)을 로드할 수 있음

**사용 사례:**
- 자연어 처리: 단어, 문자, 서브워드 임베딩
- 추천 시스템: 사용자/아이템 임베딩
- 범주형 특징 인코딩
- Knowledge Graph 임베딩

**예제: 간단한 텍스트 분류기:**

```rust
use hodu::prelude::*;
use hodu::nn::modules::{Embedding, Linear, ReLU};

// 모델 구성
let embedding = Embedding::new(10000, 128, Some(0), None, 2.0, DType::F32)?;
let linear = Linear::new(128, 10, true, DType::F32)?;

// Forward pass
let input_ids = Tensor::new(vec![45, 123, 8, 0, 0])?.reshape(&[1, 5])?;  // [1, 5]
let embedded = embedding.forward(&input_ids)?;  // [1, 5, 128]

// 평균 풀링
let pooled = embedded.mean(&[1], false)?;  // [1, 128]
let logits = linear.forward(&pooled)?;  // [1, 10]
```

### 정규화 레이어

#### Dropout

무작위로 뉴런을 비활성화하여 과적합을 방지합니다.

```rust
use hodu::nn::modules::Dropout;

let dropout = Dropout::new(0.5);  // 50% 확률로 드롭
let output = dropout.forward(&input)?;
```

**파라미터:**
- `p`: 드롭 확률 (0.0 ~ 1.0)

**동작:**
- 학습 시: `output = input * mask * (1/(1-p))` (균등분포 마스크)
- 추론 시: `output = input` (드롭 없음)

**권장 비율:** 은닉층 0.3~0.5, 입력층 0.1~0.2

#### BatchNorm1D, BatchNorm2D, BatchNorm3D

배치 정규화는 미니배치의 통계를 사용하여 입력을 정규화합니다.

```rust
use hodu::nn::modules::{BatchNorm1D, BatchNorm2D, BatchNorm3D};

// BatchNorm1D: [N, C] 또는 [N, C, L] 입력
let bn1d = BatchNorm1D::new(
    128,    // num_features (채널 수)
    1e-5,   // eps (수치 안정성)
    0.1,    // momentum (running stats 업데이트)
    true,   // affine (학습 가능한 gamma, beta)
    DType::F32
)?;

// BatchNorm2D: [N, C, H, W] 입력 (이미지)
let bn2d = BatchNorm2D::new(64, 1e-5, 0.1, true, DType::F32)?;
let input = Tensor::randn(&[16, 64, 32, 32], 0.0, 1.0)?;
let output = bn2d.forward(&input)?;

// BatchNorm3D: [N, C, D, H, W] 입력 (3D 데이터)
let bn3d = BatchNorm3D::new(32, 1e-5, 0.1, true, DType::F32)?;
```

**파라미터:**
- `num_features`: 정규화할 특징(채널) 수
- `eps`: 분모에 더해지는 작은 값 (0 나눗셈 방지)
- `momentum`: running statistics의 EMA 계수
- `affine`: 학습 가능한 affine 변환 사용 여부
- `dtype`: 데이터 타입

**동작:**
- **학습 모드**: 배치 통계로 정규화하고 running statistics 업데이트
  ```
  output = (input - batch_mean) / sqrt(batch_var + eps)
  output = gamma * output + beta  // affine=true인 경우
  running_mean = momentum * running_mean + (1 - momentum) * batch_mean
  running_var = momentum * running_var + (1 - momentum) * batch_var
  ```
- **평가 모드**: 누적된 running statistics로 정규화

**특징:**
- 학습 안정화 및 가속화
- Internal Covariate Shift 감소
- 더 높은 학습률 사용 가능
- 일부 정규화 효과 (Dropout 대체 가능)

**사용 위치:**
- 일반적으로 활성화 함수 전에 배치
- `Conv -> BatchNorm -> ReLU` 패턴이 일반적

#### LayerNorm

레이어 정규화는 각 샘플의 특징 차원에 대해 정규화합니다.

```rust
use hodu::nn::modules::LayerNorm;

// LayerNorm: 마지막 차원들을 정규화
let ln = LayerNorm::new(
    vec![512],  // normalized_shape (정규화할 차원들)
    1e-5,       // eps
    true,       // elementwise_affine
    DType::F32
)?;

// Transformer의 경우
let input = Tensor::randn(&[32, 128, 512], 0.0, 1.0)?;  // [batch, seq_len, hidden]
let output = ln.forward(&input)?;

// 여러 차원 정규화
let ln2d = LayerNorm::new(vec![64, 64], 1e-5, true, DType::F32)?;
let input = Tensor::randn(&[16, 64, 64], 0.0, 1.0)?;  // [batch, H, W]
let output = ln2d.forward(&input)?;
```

**파라미터:**
- `normalized_shape`: 정규화할 마지막 차원들의 크기
- `eps`: 수치 안정성을 위한 작은 값
- `elementwise_affine`: 학습 가능한 affine 변환 사용 여부
- `dtype`: 데이터 타입

**동작:**
```
mean = mean(input, dim=normalized_dims)
var = var(input, dim=normalized_dims)
output = (input - mean) / sqrt(var + eps)
output = gamma * output + beta  // elementwise_affine=true인 경우
```

**특징:**
- 배치 크기와 무관
- Training/Evaluation 모드 구분 없음 (항상 동일하게 동작)
- RNN과 Transformer에서 선호됨
- 각 샘플이 독립적으로 정규화됨

**BatchNorm vs LayerNorm:**

| 특성 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 정규화 축 | Batch 차원 | Feature 차원 |
| 배치 의존성 | 있음 | 없음 |
| Train/Eval 차이 | 있음 | 없음 |
| Running stats | 있음 | 없음 |
| 주 사용처 | CNN | RNN, Transformer |
| 작은 배치 | 불안정 | 안정적 |

**선택 가이드:**
- **CNN (이미지)**: BatchNorm2D
- **RNN/Transformer**: LayerNorm
- **작은 배치**: LayerNorm
- **배치 크기가 일정하지 않음**: LayerNorm

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
