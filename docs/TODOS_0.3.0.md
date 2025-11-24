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
  - `open()`: 기록 시작
  - `close()`: 기록 종료
  - `add_target()`: 추적할 출력 텐서 추가
  - `capture()`: CapturedOps → Snapshot 변환 → Script 반환

```rust
let board = CaptureBoard::new();
board.open();
// ... tensor operations ...
board.add_target("output", tensor);
board.close();
let script = board.capture(); // Returns Script
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

**Script::compile()** - Runtime/Device에 맞게 실행 가능한 형태로 컴파일
- Runtime이 Device를 지원하는지 검증 (`runtime.is_supported(device)`)
- Snapshot IR 검증 (dtype/device 호환성, shape 정합성)
- IR 최적화 (상수 폴딩, 연산 융합 등)
- 컴파일된 상태로 전환 (실행 가능)

```rust
// Default: CPU + HODU runtime
script.compile()?;

// With config
let config = CompileConfig::new()
    .device(Device::CUDA(0))
    .runtime(Runtime::XLA);
script.compile_with(config)?;
```

### Phase 4: Execute or Build

**Phase 4-1: run() - JIT 실행** (no_std / std)

**메모리 내 컴파일 및 즉시 실행**

- `compile()`된 Script를 즉시 실행
- 입력 텐서를 받아 출력 텐서 반환

```rust
script.compile()?;
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
```bash
# 공유 라이브러리 생성 (.so, .dylib, .dll)
hodu --build model.hdss -o libmodel.so

# ONNX 모델을 공유 라이브러리로 컴파일
hodu --build model.onnx -o libmodel.so

# 실행 파일 생성
hodu --build model.hdss --binary -o model_exec

# 특정 타겟 아키텍처 지정
hodu --build model.hdss --target x86_64-unknown-linux-gnu -o libmodel.so

# ARM64 크로스 컴파일
hodu --build model.hdss --target aarch64-apple-darwin -o libmodel.dylib

# 최적화 레벨 지정
hodu --build model.hdss -O3 -o libmodel.so

# LLVM IR만 출력 (디버깅용)
hodu --build model.hdss --emit-llvm -o model.ll
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
- [x] LLVM module 검증 및 최적화 패스 적용
- [ ] JIT 실행 엔진 구현 (메모리 내 실행)

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
