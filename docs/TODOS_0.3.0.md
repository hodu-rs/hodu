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

### Phase 1: Capture

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

### Phase 2: Save/Load

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

### Phase 3: Compile

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

**Phase 4-1: run()** - 컴파일된 Script를 JIT 실행
- `compile()`된 Script를 즉시 실행
- 입력 텐서를 받아 출력 텐서 반환

```rust
script.compile()?;
let outputs = script.run(&inputs)?;
```

**Phase 4-2: Builder + build()** - AOT 컴파일하여 네이티브 바이너리/라이브러리 생성
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
- [ ] `inkwell` crate 추가 (LLVM wrapper for Rust)
- [ ] Snapshot → LLVM IR 변환 로직 구현
- [ ] Op별 LLVM instruction 생성 함수 구현
- [ ] LLVM 최적화 패스 적용
- [ ] 타겟별 코드 생성 및 링킹
