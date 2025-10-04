# Builder와 Script 가이드

## 개요

Hodu는 두 가지 실행 모드를 제공합니다:

1. **동적 실행 (Dynamic Execution)**: 연산을 즉시 실행
2. **정적 실행 (Static Execution)**: Builder로 그래프를 구성한 후 Script로 컴파일하여 실행

**Builder**는 연산 그래프를 정의하는 도구이고, **Script**는 그 그래프를 컴파일하고 실행하는 객체입니다.

## Builder란?

Builder는 연산 그래프를 단계적으로 구성하는 컨텍스트를 제공합니다.

### Builder 생명주기

```rust
use hodu::prelude::*;

// 1. Builder 생성
let builder = Builder::new("my_script".to_string());

// 2. Builder 시작 (컨텍스트 활성화)
builder.start()?;

// 3. 이 구간에서 모든 텐서 연산이 그래프에 기록됨
let x = Tensor::input("x", &[2, 3])?;
let y = Tensor::input("y", &[3, 4])?;
let result = x.matmul(&y)?;

// 4. 출력 등록
builder.add_output("result", result)?;

// 5. Builder 종료 (컨텍스트 비활성화)
builder.end()?;

// 6. Script 빌드
let mut script = builder.build()?;
```

### Builder 주요 메서드

#### Builder::new()

새로운 Builder를 생성합니다:

```rust
let builder = Builder::new("computation_name".to_string());
```

#### builder.start()

Builder 컨텍스트를 활성화합니다. 이후 모든 텐서 연산이 그래프에 기록됩니다:

```rust
builder.start()?;

// 이제 연산들이 기록됨
let a = Tensor::input("a", &[10])?;
let b = a.mul_scalar(2.0)?;
```

**중요**: `start()` 호출 후에만 `Tensor::input()` 사용 가능합니다.

#### builder.add_input()

입력 placeholder를 명시적으로 등록합니다 (보통 `Tensor::input()`이 자동으로 호출):

```rust
let x = Tensor::input("x", &[10, 20])?;  // 자동으로 add_input 호출됨
```

#### builder.add_output()

출력 텐서를 등록합니다:

```rust
builder.add_output("result", result_tensor)?;
builder.add_output("loss", loss_tensor)?;
```

여러 출력을 등록할 수 있습니다.

#### builder.end()

Builder 컨텍스트를 비활성화합니다:

```rust
builder.end()?;

// 이제 연산들이 더 이상 그래프에 기록되지 않음
let c = Tensor::randn(&[5], 0.0, 1.0)?;  // 동적 실행
```

#### builder.build()

기록된 그래프로부터 Script를 생성합니다:

```rust
let mut script = builder.build()?;
```

## Script란?

Script는 Builder로 정의된 연산 그래프를 컴파일하여 최적화된 실행을 가능하게 하는 객체입니다. 동적 실행 모드와 달리 연산들을 먼저 IR(Intermediate Representation)로 변환한 후, 백엔드(HODU, XLA)에서 최적화하여 실행합니다.

## 기본 사용법

### 1. Script 빌드

```rust
use hodu::prelude::*;

// Builder 생성 및 시작
let builder = Builder::new("my_script".to_string());
builder.start()?;

// Script 모드에서 연산 정의
let x = Tensor::input("x", &[2, 3])?;
let y = Tensor::input("y", &[3, 4])?;
let result = x.matmul(&y)?;

// 출력 등록
builder.add_output("result", result)?;
builder.end()?;

// Script 빌드
let mut script = builder.build()?;
```

### 2. Script 실행

```rust
// 입력 데이터 준비
let x_data = Tensor::randn(&[2, 3], DType::F32)?;
let y_data = Tensor::randn(&[3, 4], DType::F32)?;

// 입력 추가
script.add_input("x", x_data);
script.add_input("y", y_data);

// 실행
let outputs = script.run()?;
let result = &outputs["result"];
```

## 컴파일 캐싱

Script는 컴파일 결과를 캐싱하여 반복 실행 시 성능을 향상시킵니다.

### 자동 컴파일

```rust
let mut script = builder.build()?;

// 첫 run() 호출 시 자동으로 컴파일
script.add_input("x", x_data);
let output1 = script.run()?;  // 컴파일 + 실행

// 두 번째 run()부터는 캐시된 컴파일 결과 사용
script.add_input("x", x_data2);
let output2 = script.run()?;  // 실행만
```

### 명시적 컴파일

```rust
let mut script = builder.build()?;

// 명시적으로 미리 컴파일 (warm-up)
println!("Compiling...");
script.compile()?;
println!("Compilation done!");

// 이후 run()은 실행만 수행
script.add_input("x", x_data);
let output = script.run()?;  // 실행만
```

### 컴파일 시간 측정

```rust
use std::time::Instant;

let mut script = builder.build()?;
script.add_input("x", x_data);

// 컴파일 시간 측정
let compile_start = Instant::now();
script.compile()?;
let compile_time = compile_start.elapsed();
println!("Compilation: {:?}", compile_time);

// 실행 시간 측정
let run_start = Instant::now();
let output = script.run()?;
let run_time = run_start.elapsed();
println!("Execution: {:?}", run_time);
```

## 백엔드 선택

### HODU 백엔드 (기본)

```rust
let mut script = builder.build()?;
script.set_backend(Backend::HODU);  // 기본값
```

### XLA 백엔드

```rust
#[cfg(feature = "xla")]
{
    let mut script = builder.build()?;
    script.set_backend(Backend::XLA);
}
```

**주의**: 백엔드를 변경하면 캐시된 컴파일 결과가 무효화됩니다.

```rust
let mut script = builder.build()?;
script.compile()?;  // HODU로 컴파일

script.set_backend(Backend::XLA);  // 캐시 무효화!
// 다음 run()에서 XLA로 재컴파일됨
```

## Script 모드에서 Training

Script 모드에서 training loop를 작성할 때는 주의가 필요합니다.

### Training Loop 예제

```rust
let builder = Builder::new("training".to_string());
builder.start()?;

let mut linear = Linear::new(3, 1, true, DType::F32)?;
let mse_loss = MSE::new();
let mut optimizer = SGD::new(0.01);

let input = Tensor::input("input", &[100, 3])?;
let target = Tensor::input("target", &[100, 1])?;

let epochs = 1000;
let mut final_loss = Tensor::full(&[], 0.0)?;

for _ in 0..epochs {
    let pred = linear.forward(&input)?;
    let loss = mse_loss.forward((&pred, &target))?;

    // Backward - gradient 계산
    loss.backward()?;

    // Optimizer update
    optimizer.step(&mut linear.parameters())?;
    optimizer.zero_grad(&mut linear.parameters())?;

    final_loss = loss;
}

builder.add_output("loss", final_loss)?;
builder.end()?;

let mut script = builder.build()?;
```

### Training Script의 특징

1. **Loop Unrolling**: Script 빌드 시 loop가 완전히 펼쳐집니다.
   - 1000 epoch = 1000개의 forward + backward + update 연산이 그래프에 기록
   - 그래프 크기가 매우 커질 수 있음

2. **Gradient Tape**: Builder 모드에서도 gradient tape가 작동합니다.
   - `backward()`는 실제로 gradient를 계산하고 tape에 기록
   - `requires_grad=true`인 연산은 반드시 tape에 기록되어야 함

3. **컴파일 시간**: 큰 그래프는 컴파일에 오랜 시간이 걸립니다.
   - XLA: 최적화 과정이 복잡하여 시간이 더 오래 걸림
   - HODU: 상대적으로 빠른 컴파일

## 주의사항

### 1. Transpose와 Gradient

Builder 모드에서 `transpose()` 같은 shape 연산은 **반드시 gradient tape에도 기록**되어야 합니다:

```rust
// 올바른 구현 (ops.rs)
if builder::is_builder_active() {
    let (tensor_id, result) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);
    let op = Op::Shape(op::ShapeOp::Transpose, self.id());
    register_operation_in_builder(op.clone(), tensor_id, ...);

    if self.is_requires_grad() {
        gradient::record_operation(tensor_id, op, vec![self.id()])?;  // 필수!
    }

    Ok(result)
}
```

이를 빼먹으면 backward 시 "Gradient not computed" 에러 발생.

### 2. Script는 Inference 최적화에 적합

Training보다는 **추론(inference)**에 더 적합합니다:

```rust
// Inference용 Script
let builder = Builder::new("inference".to_string());
builder.start()?;

let input = Tensor::input("input", &[1, 784])?;
let output = model.forward(&input)?;  // requires_grad=false

builder.add_output("output", output)?;
builder.end()?;

let mut script = builder.build()?;
script.compile()?;  // 빠른 컴파일

// 여러 번 재사용
for batch in batches {
    script.add_input("input", batch);
    let result = script.run()?;  // 빠른 실행
}
```

### 3. 캐시 무효화 조건

다음의 경우 컴파일 캐시가 무효화됩니다:

- `set_backend()` 호출
- `set_device()` 호출

```rust
let mut script = builder.build()?;
script.compile()?;  // 컴파일 완료

script.set_device(Device::CUDA(0));  // 캐시 무효화!
// 다음 run()에서 재컴파일
```

### 4. Script 재사용

같은 Script를 다른 입력으로 여러 번 실행 가능:

```rust
let mut script = builder.build()?;
script.compile()?;  // 한 번만 컴파일

for i in 0..100 {
    let input_data = generate_input(i);
    script.clear_inputs();  // 이전 입력 제거
    script.add_input("x", input_data);
    let output = script.run()?;  // 빠른 실행
}
```

## 성능 팁

### 1. 명시적 컴파일로 Warm-up

```rust
// 프로그램 시작 시 미리 컴파일
let mut script = builder.build()?;
script.add_input("x", dummy_input);
script.compile()?;  // Warm-up

// 실제 데이터 처리는 빠르게
for batch in data {
    script.add_input("x", batch);
    let output = script.run()?;
}
```

### 2. 백엔드 선택

- **HODU**: 빠른 컴파일, 범용적 성능
- **XLA**: 느린 컴파일, 최적화된 실행 (특히 GPU)

```rust
// CPU 추론: HODU 추천
script.set_backend(Backend::HODU);

// GPU 추론: XLA 추천 (컴파일 한 번, 여러 번 실행)
#[cfg(feature = "xla")]
{
    script.set_backend(Backend::XLA);
    script.set_device(Device::CUDA(0));
    script.compile()?;  // 시간 걸림
    // 이후 실행은 매우 빠름
}
```

### 3. Training은 동적 모드 사용

큰 training loop는 Script보다 동적 실행이 더 효율적:

```rust
// Script 없이 직접 실행
let mut optimizer = SGD::new(0.01);

for epoch in 0..1000 {
    let pred = model.forward(&input)?;
    let loss = compute_loss(&pred, &target)?;
    loss.backward()?;
    optimizer.step(&mut model.parameters())?;
    optimizer.zero_grad(&mut model.parameters())?;
}
```

## Script 저장 및 로드

```rust
#[cfg(all(feature = "serde", feature = "std"))]
{
    // Script 저장
    script.save("model.hoduscript")?;

    // Script 로드
    let mut loaded_script = Script::load("model.hoduscript")?;
    loaded_script.set_backend(Backend::XLA);
    loaded_script.compile()?;

    // 사용
    loaded_script.add_input("x", input_data);
    let output = loaded_script.run()?;
}
```

## 요약

| 항목 | 설명 |
|------|------|
| **용도** | Inference 최적화, 반복 실행되는 고정 그래프 |
| **장점** | 컴파일 후 빠른 실행, 백엔드 최적화 |
| **단점** | 컴파일 시간, 큰 training loop 비효율적 |
| **캐싱** | `compile()` 결과 자동 캐싱, 반복 실행 시 재사용 |
| **무효화** | `set_backend()`, `set_device()` 호출 시 |
| **Gradient** | Training 가능하지만 동적 모드 권장 |
