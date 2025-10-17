# Tensor 생성 가이드

## 개요

Hodu에서 텐서를 생성하는 방법은 크게 세 가지로 나뉩니다:

1. **동적 생성 (Dynamic Creation)**: 데이터를 포함한 실제 텐서를 즉시 생성
2. **연산적 생성 (Operational Creation)**: 다른 텐서로부터 연산을 통해 새로운 텐서 생성 (gradient 전파 지원)
3. **정적 생성 (Static Creation)**: Script 모드에서 입력 placeholder를 정의

## 동적 텐서 생성

동적 모드에서 실제 데이터를 가진 텐서를 생성하는 방법들입니다.

### Runtime Device 설정

텐서가 생성될 기본 장치를 설정합니다:

```rust
use hodu::prelude::*;

// CPU에서 생성 (기본값)
set_runtime_device(Device::CPU);

// CUDA GPU에서 생성
#[cfg(feature = "cuda")]
set_runtime_device(Device::CUDA(0));

// Metal GPU에서 생성 (macOS)
#[cfg(feature = "metal")]
set_runtime_device(Device::METAL(0));
```

**중요**: Script 빌드 중(`builder.start()` ~ `builder.end()`)에는 runtime device 설정이 무시되고 항상 CPU에서 생성됩니다.

### 데이터로부터 생성

#### Tensor::new()

배열이나 벡터로부터 텐서를 생성합니다:

```rust
use hodu::prelude::*;

// 1D 텐서
let t1 = Tensor::new(vec![1.0, 2.0, 3.0])?;
println!("{}", t1);  // [1, 2, 3]

// 2D 텐서
let t2 = Tensor::new(vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
])?;
println!("{}", t2);
// [[1, 2, 3],
//  [4, 5, 6]]

// 3D 텐서
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

**특징:**
- 데이터 타입은 자동으로 추론됨
- Shape은 입력 배열의 구조로부터 결정됨
- `IntoFlattened` trait을 구현한 모든 타입 사용 가능

### 특정 값으로 초기화

#### Tensor::zeros()

모든 요소가 0인 텐서를 생성합니다:

```rust
use hodu::prelude::*;

// Shape [2, 3], 타입 F32
let zeros = Tensor::zeros(&[2, 3], DType::F32)?;
println!("{}", zeros);
// [[0, 0, 0],
//  [0, 0, 0]]

// Shape [3], 타입 I32
let zeros_i32 = Tensor::zeros(&[3], DType::I32)?;
println!("{}", zeros_i32);  // [0, 0, 0]
```

#### Tensor::zeros_like()

기존 텐서와 같은 shape, dtype의 영(zero) 텐서를 생성합니다:

```rust
let original = Tensor::new(vec![1.0, 2.0, 3.0])?;
let zeros = Tensor::zeros_like(&original)?;
println!("{}", zeros);  // [0, 0, 0]
```

#### Tensor::ones()

모든 요소가 1인 텐서를 생성합니다:

```rust
use hodu::prelude::*;

let ones = Tensor::ones(&[2, 3], DType::F32)?;
println!("{}", ones);
// [[1, 1, 1],
//  [1, 1, 1]]
```

#### Tensor::ones_like()

기존 텐서와 같은 shape, dtype의 일(one) 텐서를 생성합니다:

```rust
let original = Tensor::new(vec![1.0, 2.0, 3.0])?;
let ones = Tensor::ones_like(&original)?;
println!("{}", ones);  // [1, 1, 1]
```

#### Tensor::full()

모든 요소가 지정된 값인 텐서를 생성합니다:

```rust
use hodu::prelude::*;

// 모든 요소가 3.14
let pi = Tensor::full(&[2, 2], 3.14)?;
println!("{}", pi);
// [[3.14, 3.14],
//  [3.14, 3.14]]

// 스칼라 (shape [])
let scalar = Tensor::full(&[], 42.0)?;
println!("{}", scalar);  // 42
```

**특징:**
- 값의 타입으로부터 DType이 자동 결정됨
- `Into<Scalar>`를 구현한 모든 타입 사용 가능 (f32, f64, i32, etc.)

#### Tensor::full_like()

기존 텐서와 같은 shape, dtype의 텐서를 지정된 값으로 채워서 생성합니다:

```rust
let original = Tensor::new(vec![1.0, 2.0, 3.0])?;
let filled = Tensor::full_like(&original, 9.0)?;  // 자동으로 Scalar::F32(9.0)으로 변환
println!("{}", filled);  // [9, 9, 9]
```

### 랜덤 초기화

#### Tensor::randn()

정규분포(Normal distribution)로부터 샘플링한 랜덤 텐서를 생성합니다:

```rust
use hodu::prelude::*;

// 평균 0, 표준편차 1 (표준 정규분포)
let randn = Tensor::randn(&[2, 3], 0.0, 1.0)?;
println!("{}", randn);
// [[0.5432, -1.234, 0.891],
//  [-0.234, 0.123, 1.456]]

// 평균 10, 표준편차 2
let custom_randn = Tensor::randn(&[3], 10.0, 2.0)?;
println!("{}", custom_randn);  // [11.234, 8.567, 12.345]
```

**특징:**
- Box-Muller 변환을 사용한 정규분포 샘플링
- mean과 std 중 하나라도 float이면 결과는 float 타입
- 신경망 초기화에 유용 (Xavier, He initialization 등)

#### Tensor::randn_like()

기존 텐서와 같은 shape의 랜덤 텐서를 생성합니다:

```rust
let original = Tensor::new(vec![1.0, 2.0, 3.0])?;
let randn = Tensor::randn_like(&original, 0.0, 1.0)?;
println!("{}", randn);  // [-0.234, 1.567, 0.891]
```

#### Tensor::rand_uniform()

균등분포(Uniform distribution)로부터 샘플링한 랜덤 텐서를 생성합니다:

```rust
use hodu::prelude::*;

// 범위 [0, 1)의 균등분포
let uniform = Tensor::rand_uniform(&[2, 3], 0.0, 1.0)?;
println!("{}", uniform);
// [[0.234, 0.891, 0.456],
//  [0.123, 0.789, 0.345]]

// 범위 [-1, 1)의 균등분포
let uniform_centered = Tensor::rand_uniform(&[3], -1.0, 1.0)?;
println!("{}", uniform_centered);  // [-0.234, 0.567, 0.891]

// 범위 [0, 10)의 균등분포
let uniform_scaled = Tensor::rand_uniform(&[2, 2], 0.0, 10.0)?;
println!("{}", uniform_scaled);
// [[3.456, 7.891],
//  [1.234, 9.012]]
```

**특징:**
- 지정된 범위 [low, high) 내에서 균등하게 분포
- low와 high 중 하나라도 float이면 결과는 float 타입
- Dropout, 데이터 증강, 확률적 샘플링에 유용

**사용 사례:**
- **Dropout**: 랜덤 마스크 생성
- **데이터 증강**: 랜덤 노이즈, 변환 파라미터
- **강화 학습**: 액션 샘플링, 탐색 노이즈
- **Monte Carlo 시뮬레이션**: 균등 샘플링

#### Tensor::rand_uniform_like()

기존 텐서와 같은 shape의 균등분포 랜덤 텐서를 생성합니다:

```rust
let original = Tensor::new(vec![1.0, 2.0, 3.0])?;
let uniform = Tensor::rand_uniform_like(&original, 0.0, 1.0)?;
println!("{}", uniform);  // [0.234, 0.567, 0.891]
```

## 연산적 텐서 생성

기존 텐서들로부터 연산을 통해 새로운 텐서를 생성합니다. 이 메서드들은 자동 미분을 위한 gradient 전파를 지원합니다.

### Tensor::where3_select()

조건에 따라 두 텐서 중 하나를 선택합니다:

```rust
use hodu::prelude::*;

// 조건, x, y 텐서 생성
let condition = Tensor::new(vec![true, false, true, false])?;
let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?;
let y = Tensor::new(vec![10.0, 20.0, 30.0, 40.0])?;

// 조건이 true인 위치는 x, false인 위치는 y 선택
let result = Tensor::where3_select(&condition, &x, &y)?;
println!("{}", result);  // [1, 20, 3, 40]
```

**특징:**
- 조건, x, y 텐서의 자동 브로드캐스팅 지원
- 조건 텐서는 자동으로 x의 dtype에 맞춰 변환됨
- 결과 shape는 브로드캐스팅 규칙에 따라 결정됨
- x와 y 텐서 모두로 gradient가 전파됨

**브로드캐스팅 예제:**

```rust
use hodu::prelude::*;

// 조건: [2, 1], x: [2, 3], y: 스칼라
let condition = Tensor::new(vec![vec![true], vec![false]])?;
let x = Tensor::new(vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
])?;
let y = Tensor::full(&[1], 100.0)?;

let result = Tensor::where3_select(&condition, &x, &y)?;
println!("{}", result);
// [[1, 2, 3],      // condition[0]이 true -> x[0]
//  [100, 100, 100]] // condition[1]이 false -> y
```

**사용 사례:**
- 신경망에서 조건부 값 선택
- 마스킹 연산
- 커스텀 활성화 함수 구현
- Gradient clipping 구현

## 정적 텐서 생성 (Script Mode)

Script 모드에서 입력 placeholder를 정의할 때 사용합니다.

### Tensor::input()

Script의 입력 placeholder를 생성합니다:

```rust
use hodu::prelude::*;

let builder = Builder::new("my_script".to_string());
builder.start()?;

// 입력 placeholder 정의
let x = Tensor::input("x", &[2, 3])?;
let y = Tensor::input("y", &[3, 4])?;

// 연산 정의
let result = x.matmul(&y)?;

builder.add_output("result", result)?;
builder.end()?;

let mut script = builder.build()?;
```

**특징:**
- **Builder 컨텍스트 필수**: `builder.start()` 호출 후에만 사용 가능
- 실제 데이터를 가지지 않음 (storage = None)
- Shape만 정의되고, 실제 실행 시 데이터를 제공해야 함
- 자동으로 builder에 입력으로 등록됨

**에러:**

```rust
// 잘못된 사용: Builder 없이 호출
let x = Tensor::input("x", &[2, 3])?;
// Error: StaticTensorCreationRequiresBuilderContext
```

**올바른 사용:**

```rust
let builder = Builder::new("script".to_string());
builder.start()?;  // Builder 컨텍스트 활성화

let x = Tensor::input("x", &[10, 20])?;  // OK
let y = Tensor::input("y", &[20, 30])?;  // OK

// ... 연산 정의 ...

builder.end()?;
```

## 생성 함수 비교표

### 동적 생성

| 함수 | 용도 | DType | 초기값 | 예시 |
|------|------|-------|--------|------|
| `new()` | 데이터로부터 생성 | 자동 추론 | 입력 데이터 | `Tensor::new(vec![1, 2, 3])` |
| `zeros()` | 영 텐서 생성 | 명시적 지정 | 0 | `Tensor::zeros(&[2, 3], DType::F32)` |
| `zeros_like()` | 기존 텐서 기반 영 텐서 | 복사 | 0 | `Tensor::zeros_like(&tensor)` |
| `ones()` | 일 텐서 생성 | 명시적 지정 | 1 | `Tensor::ones(&[2, 3], DType::F32)` |
| `ones_like()` | 기존 텐서 기반 일 텐서 | 복사 | 1 | `Tensor::ones_like(&tensor)` |
| `full()` | 특정 값으로 채움 | 자동 추론 | 사용자 지정 | `Tensor::full(&[2, 3], 3.14)` |
| `full_like()` | 기존 텐서 기반 특정 값 | 복사 | 사용자 지정 | `Tensor::full_like(&tensor, val)` |
| `randn()` | 정규분포 랜덤 | 자동 추론 | N(μ, σ²) | `Tensor::randn(&[2, 3], 0., 1.)` |
| `randn_like()` | 기존 텐서 기반 정규분포 | 복사 | N(μ, σ²) | `Tensor::randn_like(&tensor, 0., 1.)` |
| `rand_uniform()` | 균등분포 랜덤 | 자동 추론 | U(low, high) | `Tensor::rand_uniform(&[2, 3], 0., 1.)` |
| `rand_uniform_like()` | 기존 텐서 기반 균등분포 | 복사 | U(low, high) | `Tensor::rand_uniform_like(&tensor, 0., 1.)` |

### 연산적 생성

| 함수 | 용도 | 특징 |
|------|------|------|
| `where3_select()` | 조건부 선택 | 브로드캐스팅 지원, gradient 전파 |

### 정적 생성

| 함수 | 용도 | 요구사항 | 특징 |
|------|------|----------|------|
| `input()` | Script 입력 placeholder | Builder 컨텍스트 | 데이터 없음, shape만 정의 |

## 실전 예제

### 예제 1: 신경망 가중치 초기화

```rust
use hodu::prelude::*;

// Xavier/Glorot 초기화
fn xavier_init(in_features: usize, out_features: usize) -> HoduResult<Tensor> {
    let std = (2.0 / (in_features + out_features) as f64).sqrt();
    Tensor::randn(&[out_features, in_features], 0.0, std)
}

// He 초기화 (ReLU용)
fn he_init(in_features: usize, out_features: usize) -> HoduResult<Tensor> {
    let std = (2.0 / in_features as f64).sqrt();
    Tensor::randn(&[out_features, in_features], 0.0, std)
}

// 사용
let weight = xavier_init(784, 128)?;
let bias = Tensor::zeros(&[128], DType::F32)?;
```

### 예제 2: 배치 데이터 준비

```rust
use hodu::prelude::*;

// 배치 크기 32, 입력 차원 784
let batch_size = 32;
let input_dim = 784;

// 입력 데이터 생성 (실제로는 데이터셋에서 로드)
let inputs = Tensor::randn(&[batch_size, input_dim], 0.0, 1.0)?;

// 레이블 생성 (원-핫 인코딩, 10개 클래스)
let labels = Tensor::zeros(&[batch_size, 10], DType::F32)?;

// 마스크 생성 (패딩 위치 표시)
let mask = Tensor::ones(&[batch_size, input_dim], DType::BOOL)?;
```

### 예제 3: Script 모드에서 사용

```rust
use hodu::prelude::*;

// Script 빌드
let builder = Builder::new("linear_model".to_string());
builder.start()?;

// 입력 placeholder 정의
let input = Tensor::input("input", &[100, 784])?;

// 가중치와 바이어스는 동적 생성 (실제 데이터 포함)
let weight = Tensor::randn(&[784, 10], 0.0, 0.1)?;
let bias = Tensor::zeros(&[10], DType::F32)?;

// 연산 정의
let logits = input.matmul(&weight)?.add(&bias)?;

builder.add_output("logits", logits)?;
builder.end()?;

let mut script = builder.build()?;

// 실제 데이터 제공
let input_data = Tensor::randn(&[100, 784], 0.0, 1.0)?;
script.add_input("input", input_data);

// 실행
let outputs = script.run()?;
```

### 예제 4: Device 지정 사용

```rust
use hodu::prelude::*;

// CPU에서 생성
set_runtime_device(Device::CPU);
let cpu_tensor = Tensor::randn(&[1000, 1000], 0.0, 1.0)?;

// CUDA GPU로 전환
#[cfg(feature = "cuda")]
{
    set_runtime_device(Device::CUDA(0));
    let gpu_tensor = Tensor::randn(&[1000, 1000], 0.0, 1.0)?;
    // gpu_tensor는 GPU 메모리에 생성됨
}

// Script 모드에서는 runtime device 무시됨
let builder = Builder::new("test".to_string());
builder.start()?;

set_runtime_device(Device::CUDA(0));  // 무시됨!
let tensor = Tensor::zeros(&[10, 10], DType::F32)?;  // CPU에 생성됨

builder.end()?;
```

## 주의사항

### 1. Builder 모드에서 Runtime Device 무시

```rust
let builder = Builder::new("script".to_string());
builder.start()?;

set_runtime_device(Device::CUDA(0));  // 무시됨!

// Builder 활성화 중에는 항상 CPU에서 생성
let tensor = Tensor::randn(&[10, 10], 0.0, 1.0)?;
// -> CPU에 생성됨

builder.end()?;

// Builder 종료 후에는 runtime device 적용됨
let tensor2 = Tensor::randn(&[10, 10], 0.0, 1.0)?;
// -> CUDA(0)에 생성됨
```

### 2. input()은 Builder 필수

```rust
// 에러!
let x = Tensor::input("x", &[10])?;
// Error: StaticTensorCreationRequiresBuilderContext

// 올바른 사용
let builder = Builder::new("script".to_string());
builder.start()?;
let x = Tensor::input("x", &[10])?;  // OK
```

### 3. DType 자동 추론

```rust
// f32로 추론
let t1 = Tensor::new(vec![1.0, 2.0, 3.0])?;  // DType::F32

// i32로 추론
let t2 = Tensor::new(vec![1, 2, 3])?;  // DType::I32

// full()의 경우 입력 타입으로 추론
let t3 = Tensor::full(&[3], 1.0)?;   // DType::F32
let t4 = Tensor::full(&[3], 1)?;     // DType::I32
```

### 4. _like 함수는 Shape과 DType 모두 복사

```rust
let original = Tensor::new(vec![1.0_f32, 2.0, 3.0])?;

let zeros = Tensor::zeros_like(&original)?;
// Shape: [3], DType: F32

let ones = Tensor::ones_like(&original)?;
// Shape: [3], DType: F32

let filled = Tensor::full_like(&original, 7.0)?;  // f32 -> Scalar::F32 자동 변환
// Shape: [3], DType: F32
```

## 요약

| 카테고리 | 함수들 | 사용 시점 |
|---------|--------|----------|
| **데이터 기반** | `new()` | 실제 데이터가 있을 때 |
| **영 초기화** | `zeros()`, `zeros_like()` | 가중치, 버퍼 초기화 |
| **일 초기화** | `ones()`, `ones_like()` | 바이어스, 마스크 |
| **특정 값** | `full()`, `full_like()` | 상수 텐서 필요 시 |
| **정규분포 랜덤** | `randn()`, `randn_like()` | 신경망 가중치 초기화 |
| **균등분포 랜덤** | `rand_uniform()`, `rand_uniform_like()` | Dropout, 데이터 증강 |
| **조건부 선택** | `where3_select()` | 조건에 따라 텐서 선택 |
| **Placeholder** | `input()` | Script 입력 정의 |

**핵심 원칙:**
- 동적 실행: 모든 생성 함수 사용 가능, runtime device 적용
- Script 빌드 중: 동적 생성 함수는 CPU 고정, `input()`으로 placeholder 정의
- 연산적 생성: 다른 텐서로부터 연산을 통해 텐서를 생성하며 gradient 지원
- 실제 데이터가 필요하면 동적 생성, placeholder만 필요하면 `input()` 사용
