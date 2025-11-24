# Hodu 0.3.0-1 TODO

## 컴파일 파이프라인 설계

### 데이터 구조

- **Snapshot**: Hodu Script IR의 직렬화 가능한 표현
  - `SnapshotInput`: 입력 텐서 정보 (name, tensor_id, shape, dtype)
  - `SnapshotTarget`: 출력 텐서 정보 (name, tensor_id)
  - `SnapshotNode`: 연산 노드 (op, params, input_ids, output_id, layouts)

- **Script**: Snapshot을 담는 컨테이너, compile/run/save 인터페이스 제공
  - 내부에 `Snapshot` 보유
  - 모든 소스(CaptureBoard, ONNX, hdss)는 최종적으로 Script 객체로 통일

### Phase 1: Capture (no_std / std)

- `CaptureBoard`로 텐서 연산 기록
  - `new()` / `with_name()`: Board 생성
  - `with_target()`: 추적할 출력 텐서 추가 (체이닝 가능)
  - `open()`: 기록 시작
  - `close()`: 기록 종료
  - `capture()`: CapturedOps → Snapshot 변환 → Script 반환

```rust
// Basic usage
let board = CaptureBoard::new();
board.open();
// ... tensor operations ...
board.close();
let script = board.with_target("output", output_tensor).capture();

// With name and multiple targets
let board = CaptureBoard::with_name("my_model");
board.open();
let y1 = x.add(w1)?;
let y2 = y1.mul(w2)?;
board.close();
let script = board
    .with_target("output1", y1)
    .with_target("output2", y2)
    .capture();
```

### Phase 2: Save/Load (std)

**파일 I/O가 필요하므로 std 전용**

**Script::save()** - Snapshot을 파일로 직렬화
- `.hdss` → postcard를 사용해 바이너리 직렬화
- `.onnx` → ONNX 형식으로 변환 후 저장 (미래 작업)

**Script::load()** - 파일을 Snapshot으로 역직렬화 → Script 생성
- `.hdss` → postcard로 역직렬화 → Snapshot → Script
- `.onnx` → ONNX 파싱 → Snapshot 변환 → Script (미래 작업)

```rust
// Save
script.save("model.hdss")?;

// Load
let script = Script::load("model.hdss")?;
```

### Phase 3: Compile (no_std / std)

**메모리 내 LLVM IR 생성, 검증, 최적화**

**Script::compile()** - Runtime/Device/Compiler에 맞게 실행 가능한 형태로 컴파일
- Runtime이 Device를 지원하는지 검증 (`runtime.is_supported(device)`)
- Runtime별 분기:
  - `Runtime::HODU` → Compiler별 분기 (LLVM / MLIR / Cranelift)
  - `Runtime::XLA` → XLA HLO로 컴파일
- Snapshot IR 검증 (dtype/device 호환성, shape 정합성)
- IR 최적화 (상수 폴딩, 연산 융합 등)
- 컴파일된 상태로 전환 (실행 가능)

```rust
// Load or create script
let mut script = Script::load("model.hdss")?;

// Set device, runtime, and compiler (required before compile)
script.set_device(Device::CPU);
script.set_runtime(Runtime::HODU);
script.set_compiler(HoduRuntimeCompiler::LLVM);  // Optional for HODU runtime: defaults to LLVM

// Compile
script.compile()?;

// Or with different configuration
script.set_device(Device::CUDA(0));
script.set_runtime(Runtime::HODU);
// Compiler automatically defaults to LLVM if not set
script.compile()?;  // Previous compilation is cleared

// Future: Use different compiler backend for HODU runtime
#[cfg(feature = "mlir")]
{
    script.set_compiler(HoduRuntimeCompiler::MLIR);
    script.compile()?;
}

// Future: Use XLA runtime
#[cfg(feature = "xla")]
{
    script.set_device(Device::CUDA(0));
    script.set_runtime(Runtime::XLA);  // No compiler needed for XLA
    script.compile()?;
}
```

### Phase 4: Execute or Build

**Phase 4-1: run() - JIT 실행** (no_std / std)

**메모리 내 컴파일 및 즉시 실행**

- `compile()`된 Script를 즉시 실행
- 입력 텐서를 받아 출력 텐서 반환
- compile() 호출 없이 run()을 호출하면 자동으로 compile 수행

```rust
// Complete workflow: Capture → Compile → Execute
let board = CaptureBoard::with_name("simple_add");
board.open();
let a = Tensor::input("a", [2, 2], DType::F32)?;
let b = Tensor::input("b", [2, 2], DType::F32)?;
let c = a.add(&b)?;
board.close();

let mut script = board.with_target("output", c).capture();

// Configure runtime environment
script.set_device(Device::CPU);
script.set_runtime(Runtime::HODU);
script.set_compiler(HoduRuntimeCompiler::LLVM);  // Optional for HODU runtime: defaults to LLVM

// Prepare inputs: &[(&str, &Tensor)]
let input_a = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], [2, 2]);
let input_b = Tensor::from_slice(&[50.0f32, 60.0, 70.0, 80.0], [2, 2]);
let inputs = [
    ("a", &input_a),
    ("b", &input_b),
];

// Execute and get outputs: HashMap<String, Tensor>
// (auto-compiles if not already compiled)
let outputs = script.run(&inputs)?;

// Access outputs by target name
let result = &outputs["output"];
// let result = outputs["output"]; <- move tensor
println!("Result: {:?}", result);

// Load from file and execute
let mut script = Script::load("model.hdss")?;
script.set_device(Device::CPU);
script.set_runtime(Runtime::HODU);

// Can explicitly compile first
script.compile()?;
let outputs = script.run(&inputs)?;

// Or run directly (auto-compiles)
let outputs = script.run(&inputs)?;
```

**Phase 4-2: Builder + build() - AOT 컴파일하여 네이티브 바이너리/라이브러리 생성** (std)
- Script를 특정 타겟 환경용으로 AOT 컴파일
- LLVM IR을 생성하여 네이티브 코드로 컴파일
- `.so`, `.dylib`, `.dll`, 또는 실행 파일 생성
- `compile()`과 독립적으로 동작

**Builder 구조**:
```rust
pub struct Builder<'a> {
    script: &'a Script,
    target: TargetConfig,      // arch, vendor, os, env
    build_type: BuildType,      // Binary or Library
    output_path: Option<String>,
}
```

**사용 예시**:
```rust
let builder = Builder::new(&script)
    .target(TargetConfig::new().arch(TargetArch::X86_64))
    .build_type(BuildType::Library)
    .output_path("output.so")
    .build()?;
```

**CLI 사용 예시**:

**JIT 실행 (즉시 실행)**:
```bash
# 기본 JIT 실행 (CPU, HODU runtime, LLVM compiler)
hodu run model.hdss --input input.json

# Device 지정
hodu run model.hdss --input input.json --device cpu
hodu run model.hdss --input input.json --device cuda:0
hodu run model.hdss --input input.json --device metal

# Runtime 지정
hodu run model.hdss --input input.json --runtime hodu
hodu run model.hdss --input input.json --runtime xla --device cuda:0

# HODU runtime의 컴파일러 지정
hodu run model.hdss --input input.json --runtime hodu --compiler llvm
hodu run model.hdss --input input.json --runtime hodu --compiler mlir
hodu run model.hdss --input input.json --runtime hodu --compiler cranelift

# 출력 형식 지정
hodu run model.hdss --input input.json -o output.json
hodu run model.hdss --input input.json --output-format numpy  # .npy 파일 저장

# ONNX 모델 직접 실행
hodu run model.onnx --input input.json
```

**AOT 컴파일 (네이티브 바이너리/라이브러리 생성)**:
```bash
# 공유 라이브러리 생성 (.so, .dylib, .dll)
hodu build model.hdss -o libmodel.so

# Device/Runtime 지정하여 빌드
hodu build model.hdss -o libmodel.so --device cpu --runtime hodu
hodu build model.hdss -o libmodel_cuda.so --device cuda:0 --runtime hodu

# HODU runtime의 컴파일러 지정
hodu build model.hdss -o libmodel.so --runtime hodu --compiler llvm
hodu build model.hdss -o libmodel.so --runtime hodu --compiler mlir

# ONNX 모델을 공유 라이브러리로 컴파일
hodu build model.onnx -o libmodel.so

# 실행 파일 생성
hodu build model.hdss --binary -o model_exec

# 특정 타겟 아키텍처 지정 (크로스 컴파일)
hodu build model.hdss --target x86_64-unknown-linux-gnu -o libmodel.so
hodu build model.hdss --target aarch64-apple-darwin -o libmodel.dylib
hodu build model.hdss --target aarch64-unknown-linux-gnu -o libmodel_arm64.so

# 최적화 레벨 지정
hodu build model.hdss -O0 -o libmodel.so
hodu build model.hdss -O1 -o libmodel.so
hodu build model.hdss -O2 -o libmodel.so
hodu build model.hdss -O3 -o libmodel.so
```

**모델 정보 확인**:
```bash
# 모델 정보 출력
hodu info model.hdss

# 예상 출력:
# Model: simple_add
# Inputs: 2
#   - a: [2, 2] f32
#   - b: [2, 2] f32
# Outputs: 1
#   - output: [2, 2] f32
# Operations: 6
# Constants: 2
```

**모델 변환**:
```bash
# ONNX → hdss
hodu convert model.onnx -o model.hdss

# hdss → ONNX (미래)
hodu convert model.hdss -o model.onnx
```

**LLVM 기반 컴파일 파이프라인**:

1. **Snapshot → LLVM IR 변환**
   - Snapshot의 각 노드를 LLVM IR로 변환
   - 입력: Function parameters
   - 출력: Function returns
   - 연산: Op별 LLVM instruction 생성

2. **LLVM 최적화**
   - Function inlining
   - Constant folding
   - Vectorization (SIMD)
   - Dead code elimination

3. **타겟별 코드 생성**
   - TargetConfig의 triple 사용 (e.g., "x86_64-apple-darwin")
   - 타겟 머신 코드 생성
   - Object file 생성

4. **링킹**
   - BuildType::Library → 공유 라이브러리 (.so, .dylib, .dll)
   - BuildType::Binary → 실행 파일

**구현 계획**:

**[no_std + std 모두 지원]**
- [x] `inkwell` crate 추가 (LLVM wrapper for Rust)
- [x] Snapshot → LLVM IR 변환 로직 구현
  - [x] CodeGenerator 구조체 설계
  - [x] Target triple 기반 pointer width 자동 감지
  - [x] Function signature 생성 (inputs + outputs)
  - [x] Constant 텐서 로딩 (모든 DType 지원)
  - [x] Kernel 함수 선언 (`{runtime}_{device}_{op}_{dtype}` 형식)
  - [x] Metadata constant 생성
  - [x] no_std 환경 지원 (core::mem::size_of 사용)
- [x] LLVM module 검증 및 최적화 패스 적용
- [x] JIT ExecutionEngine wrapper 구현
  - [x] `CodeGenerator::create_jit_engine()` 메서드
- [x] Compiled state 구조 설계
  - [x] `compiled/` 폴더 생성
  - [x] `compiled.rs`: CompiledState enum (Runtime별 분기)
  - [x] `compiled/hodu.rs`: HoduCompiledState enum (Compiler별 분기)
  - [x] `compiled/hodu/llvm.rs`: LLVMJitState (Context + ExecutionEngine 관리)
  - [x] `compiled/xla.rs`: XLAExecutable (placeholder)
  - [x] Context lifetime 관리 (Box::leak 패턴)
- [x] Script 구조 재설계
  - [x] Device/Runtime/Compiler를 Option으로 변경
  - [x] Compiled state: Option<CompiledState>
  - [x] `set_device()`/`set_runtime()`/`set_compiler()` 시 compiled state 초기화
  - [x] `compiler()` getter 추가
- [x] Script::compile() 구현
  - [x] Runtime별 분기 (HODU / XLA)
  - [x] HODU runtime: Compiler별 분기 (LLVM / MLIR / Cranelift)
  - [x] LLVM: CodeGenerator로 LLVM IR 생성 → JIT engine 생성
  - [x] CompiledState 저장
- [x] Script::run() 구현
  - [x] 입력: `&[(&str, &Tensor)]`
  - [x] 출력: `HashMap<String, Tensor>`
  - [x] 자동 compile 호출
  - [x] CompiledState::execute() 호출
- [ ] JIT 실제 실행 로직 구현 (LLVMJitState::execute)
  - [ ] LLVM ExecutionEngine에서 함수 포인터 가져오기
  - [ ] 입력 텐서 검증 (snapshot.inputs와 매칭)
  - [ ] 입력 버퍼 준비 (Tensor → raw pointer)
  - [ ] 출력 버퍼 할당
  - [ ] JIT 함수 호출
  - [ ] 출력 버퍼 → Tensor 변환
  - [ ] HashMap<target_name, Tensor> 반환
- [x] jit_test.rs 예제 업데이트
  - [x] Script::compile() 테스트
  - [x] Device/Runtime/Compiler 설정 확인

**[확장 계획 - 다양한 컴파일러 백엔드]**
- [x] HoduRuntimeCompiler enum 추가 (types/runtime.rs)
  - [x] LLVM (기본)
  - [ ] MLIR (미래)
  - [ ] Cranelift (미래)
- [ ] MLIR 컴파일러 백엔드
  - [ ] compiled/hodu/mlir.rs
  - [ ] Snapshot → MLIR dialect 변환
- [ ] Cranelift 컴파일러 백엔드
  - [ ] compiled/hodu/cranelift.rs
  - [ ] Snapshot → Cranelift IR 변환

**[확장 계획 - 다양한 런타임]**
- [x] Runtime enum 확장 (types/runtime.rs)
  - [x] HODU (LLVM 컴파일)
  - [x] XLA (placeholder)
  - [ ] ONNX (미래)
  - [ ] TVM (미래)
- [ ] XLA 런타임 구현
  - [ ] compiled/xla.rs: XLAExecutable 구현
  - [ ] Snapshot → XLA HLO 변환
  - [ ] XLA compilation 및 execution
- [ ] ONNX 런타임
  - [ ] compiled/onnx.rs
  - [ ] Snapshot이 이미 ONNX 형식이면 ONNX Runtime session 생성
- [ ] TVM 런타임
  - [ ] compiled/tvm.rs
  - [ ] Snapshot → Relay IR 변환

**[std 전용 - 파일 I/O 필요]**
- [x] AOT Object 파일 생성 (`emit_object_file`)
- [x] AOT Assembly 파일 생성 (`emit_assembly`)
- [x] Shared library 생성 (`emit_shared_library`)
- [ ] Builder API 완성 (파일 기반 빌드)

**[커널 구현 - 각 백엔드별]**
- [x] CPU 커널: `hodu_cpu_{op}_{dtype}` 형식으로 리네이밍 (174d85e)
- [x] CUDA 커널: `hodu_cuda_{op}_{dtype}` 형식으로 리네이밍 (c6e4b67)
- [x] Metal 커널: `hodu_metal_{op}_{dtype}` 형식으로 리네이밍 (2142441)
- [ ] XLA 런타임: 별도 처리 방식 설계 필요
