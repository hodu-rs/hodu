# Gradient Tape 관리 가이드

## Optimizer에서의 Tape 관리

### zero_grad() 메서드의 동작

모든 Optimizer(SGD, Adam)는 `#[derive(Optimizer)]` 매크로를 통해 동일한 방식으로 테이프를 관리합니다:

```rust
// Optimizer 매크로가 자동으로 생성하는 zero_grad() 구현
#[derive(Optimizer)]
pub struct SGD { /* ... */ }

// 매크로가 생성하는 코드:
impl Optimizer for SGD {
    fn zero_grad(&mut self, parameters: &mut [&mut Tensor]) -> HoduResult<()> {
        // 1. 각 파라미터의 그래디언트를 0으로 초기화
        for param in parameters.iter_mut() {
            param.zero_grad()?;
        }

        // 2. 기본 컨텍스트(context 0)의 테이프만 정리
        hodu_core::tensor::clear_default_context_tape();

        Ok(())
    }
}
```

**중요**:
- `#[derive(Optimizer)]` 매크로가 `zero_grad()` 구현을 자동으로 생성합니다
- `clear_default_context_tape()`는 **기본 컨텍스트(ID: 0)만** 정리합니다
- 커스텀 컨텍스트에는 영향을 주지 않습니다

### 기본 컨텍스트에서 Optimizer 사용

```rust
use hodu_nn::optimizers::SGD;
use hodu_core::tensor::compute_gradients;

let mut optimizer = SGD::new(0.01);
let mut weight = Tensor::randn(&[10, 5], DType::F32)?.set_requires_grad(true);

for epoch in 0..100 {
    // Forward - 연산이 기본 컨텍스트(0) 테이프에 기록
    let output = input.matmul(&weight)?;
    let loss = output.mean(&[], false)?;

    // Backward - 기본 컨텍스트(0) 테이프를 역순으로 읽음
    compute_gradients(loss.id())?;

    // Update
    let mut params = vec![&mut weight];
    optimizer.step(&mut params)?;

    // Gradient 초기화 + 기본 컨텍스트(0) 테이프 정리
    optimizer.zero_grad(&mut params)?;
}
```

## 커스텀 Gradient Context 사용

### GradientContext란?

독립적인 그래디언트 계산을 위한 별도의 테이프 공간입니다.

```rust
use hodu_core::tensor::GradientContext;

{
    let _ctx = GradientContext::new();  // 새로운 컨텍스트 생성 (예: ID 1)

    // 이 스코프 내의 연산은 컨텍스트 1의 테이프에 기록됨
    let x = Tensor::randn(&[2, 3], DType::F32)?.set_requires_grad(true);
    let y = x.mul_scalar(2.0)?;
    compute_gradients(y.id())?;

    // _ctx가 drop되면 컨텍스트 1과 테이프가 자동 삭제됨
}

// 다시 기본 컨텍스트(0)로 돌아옴
```

### 커스텀 컨텍스트에서 Optimizer 사용 시 주의사항

커스텀 컨텍스트를 사용할 때는 **수동으로 테이프를 정리**해야 합니다:

```rust
use hodu_core::tensor::{GradientContext, compute_gradients, clear_tape};
use hodu_nn::optimizers::SGD;

let _ctx = GradientContext::new();  // 커스텀 컨텍스트 생성 (예: ID 1)

let mut optimizer = SGD::new(0.01);
let mut weight = Tensor::randn(&[10, 5], DType::F32)?.set_requires_grad(true);

for epoch in 0..100 {
    // Forward - 연산이 컨텍스트 1 테이프에 기록
    let output = input.matmul(&weight)?;
    let loss = output.mean(&[], false)?;

    // Backward - 컨텍스트 1 테이프를 역순으로 읽음
    compute_gradients(loss.id())?;

    // Update
    let mut params = vec![&mut weight];
    optimizer.step(&mut params)?;

    // Gradient 초기화
    optimizer.zero_grad(&mut params)?;
    // 주의: zero_grad()는 기본 컨텍스트(0)만 정리하므로
    // 커스텀 컨텍스트(1)의 테이프는 그대로 남아있음!

    // 커스텀 컨텍스트의 테이프를 수동으로 정리
    clear_tape();  // 현재 활성 컨텍스트(1)의 테이프 정리
}

// 스코프를 벗어나면 _ctx가 drop되어 컨텍스트 1 자체가 삭제됨
```

## Tape 관리 함수 정리

| 함수 | 대상 | 사용처 |
|------|------|--------|
| `clear_default_context_tape()` | 기본 컨텍스트(0)만 | Optimizer의 `zero_grad()` |
| `clear_tape()` | 현재 활성 컨텍스트 | 커스텀 컨텍스트에서 수동 정리 |
| `GradientContext::drop()` | 해당 컨텍스트 전체 | 스코프 종료 시 자동 호출 |

## 예제 1: 기본 컨텍스트에서 학습

```rust
use hodu_nn::optimizers::Adam;
use hodu_core::tensor::{Tensor, compute_gradients};

// 기본 컨텍스트(0) 사용 - 별도의 GradientContext 생성 없음
let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
let mut weight = Tensor::randn(&[784, 10], DType::F32)?.set_requires_grad(true);
let mut bias = Tensor::zeros(&[10], DType::F32)?.set_requires_grad(true);

for epoch in 0..100 {
    // Forward
    let logits = input.matmul(&weight)?.add(&bias)?;
    let loss = logits.mean(&[], false)?;

    // Backward (기본 컨텍스트 0의 테이프 사용)
    compute_gradients(loss.id())?;

    // Update
    let mut params = vec![&mut weight, &mut bias];
    optimizer.step(&mut params)?;

    // zero_grad()가 자동으로 기본 컨텍스트 0의 테이프 정리
    optimizer.zero_grad(&mut params)?;
}
```

## 예제 2: 커스텀 컨텍스트로 독립적인 학습

메인 모델과 보조 모델을 별도의 컨텍스트에서 학습하는 예제:

```rust
use hodu_core::tensor::{GradientContext, compute_gradients, clear_tape};
use hodu_nn::optimizers::{SGD, Adam};

// === 메인 모델 (기본 컨텍스트 0) ===
let mut main_optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
let mut main_weight = Tensor::randn(&[100, 50], DType::F32)?.set_requires_grad(true);

// === 보조 모델 (커스텀 컨텍스트 1) ===
let _aux_ctx = GradientContext::new();  // 컨텍스트 1 생성
let mut aux_optimizer = SGD::new(0.01);
let mut aux_weight = Tensor::randn(&[50, 10], DType::F32)?.set_requires_grad(true);

for epoch in 0..100 {
    // === 메인 모델 학습 (컨텍스트 0) ===
    {
        // 메인 모델은 기본 컨텍스트에서 실행되므로
        // 보조 모델 연산과 완전히 독립적

        let main_output = input.matmul(&main_weight)?;
        let main_loss = main_output.mean(&[], false)?;

        compute_gradients(main_loss.id())?;  // 컨텍스트 0의 테이프 사용

        let mut params = vec![&mut main_weight];
        main_optimizer.step(&mut params)?;
        main_optimizer.zero_grad(&mut params)?;  // 컨텍스트 0 테이프 정리
    }

    // === 보조 모델 학습 (컨텍스트 1) ===
    {
        // _aux_ctx가 스코프에 있으므로 컨텍스트 1이 활성 상태

        let aux_input = main_weight.detach()?;  // 메인 모델 출력 사용
        let aux_output = aux_input.matmul(&aux_weight)?;
        let aux_loss = aux_output.mean(&[], false)?;

        compute_gradients(aux_loss.id())?;  // 컨텍스트 1의 테이프 사용

        let mut params = vec![&mut aux_weight];
        aux_optimizer.step(&mut params)?;
        aux_optimizer.zero_grad(&mut params)?;  // 주의: 이것만으로는 부족!

        // 커스텀 컨텍스트 1의 테이프를 수동으로 정리해야 함
        clear_tape();
    }
}

// 루프 종료 후 _aux_ctx가 drop되어 컨텍스트 1이 자동 삭제됨
```

## 예제 3: 중첩된 컨텍스트

여러 개의 독립적인 계산을 동시에 수행:

```rust
use hodu_core::tensor::{GradientContext, compute_gradients, clear_tape};

// 기본 컨텍스트(0)에서 메인 작업
let main_x = Tensor::randn(&[2, 3], DType::F32)?.set_requires_grad(true);
let main_y = main_x.mul_scalar(2.0)?;

{
    // 컨텍스트 1: 첫 번째 보조 계산
    let _ctx1 = GradientContext::new();

    let aux1_x = Tensor::randn(&[3, 4], DType::F32)?.set_requires_grad(true);
    let aux1_y = aux1_x.mul_scalar(3.0)?;
    compute_gradients(aux1_y.id())?;
    println!("Aux1 gradient: {:?}", aux1_x.grad()?);

    clear_tape();  // 컨텍스트 1 테이프 정리

    {
        // 컨텍스트 2: 두 번째 보조 계산 (중첩)
        let _ctx2 = GradientContext::new();

        let aux2_x = Tensor::randn(&[4, 5], DType::F32)?.set_requires_grad(true);
        let aux2_y = aux2_x.mul_scalar(4.0)?;
        compute_gradients(aux2_y.id())?;
        println!("Aux2 gradient: {:?}", aux2_x.grad()?);

        clear_tape();  // 컨텍스트 2 테이프 정리

    } // _ctx2 drop -> 컨텍스트 2 삭제, 컨텍스트 1로 복귀

} // _ctx1 drop -> 컨텍스트 1 삭제, 기본 컨텍스트(0)로 복귀

// 메인 작업 계속 (기본 컨텍스트 0)
compute_gradients(main_y.id())?;
println!("Main gradient: {:?}", main_x.grad()?);
```

## 주의사항

### 1. Optimizer의 zero_grad()는 기본 컨텍스트만 정리

```rust
// 잘못된 사용: 커스텀 컨텍스트에서 zero_grad()만 호출
let _ctx = GradientContext::new();
// ... 학습 루프 ...
optimizer.zero_grad(&mut params)?;  // 기본 컨텍스트(0)만 정리됨!
// 커스텀 컨텍스트의 테이프는 계속 누적되어 메모리 누수 발생

// 올바른 사용: clear_tape()를 추가로 호출
let _ctx = GradientContext::new();
// ... 학습 루프 ...
optimizer.zero_grad(&mut params)?;
clear_tape();  // 현재 활성 컨텍스트 테이프 정리
```

### 2. GradientContext는 RAII 패턴

```rust
{
    let _ctx = GradientContext::new();
    // 연산 수행
} // _ctx drop -> 컨텍스트와 테이프가 자동 삭제

// 스코프를 벗어나면 해당 컨텍스트의 모든 정보가 사라짐
```

### 3. 기본 컨텍스트(0)는 절대 삭제되지 않음

```rust
// 기본 컨텍스트는 프로그램 시작 시 자동 생성되고
// 프로그램 종료까지 유지됨
// 테이프만 clear_default_context_tape()로 정리 가능
```

## 테이프 정리를 빼먹으면 어떻게 되나?

```rust
let _ctx = GradientContext::new();

for epoch in 0..10000 {
    let output = model.forward(&input)?;
    let loss = output.mean(&[], false)?;
    compute_gradients(loss.id())?;
    optimizer.step(&mut params)?;
    optimizer.zero_grad(&mut params)?;
    // 잘못: clear_tape() 호출 안함!
}

// 결과: 테이프에 10000번의 연산이 모두 기록되어
// - 메모리 사용량 급증
// - 역전파 속도 저하
// - 결국 메모리 부족으로 크래시
```

**해결책**: 매 iteration마다 `clear_tape()` 호출
