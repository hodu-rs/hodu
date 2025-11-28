# Hodu 0.3.0-2 TODO

## 플러그인 아키텍처 설계

### 목표

hodu_core를 순수 IR 생성(Script/Snapshot)까지만 유지하고, 컴파일/실행은 플러그인 시스템으로 분리하여 동적 로딩 가능하게 함.

### 아키텍처 개요

```
┌─────────────────────────────────────────────────────┐
│  Format Plugin (모델 포맷)                            │
│  - onnx, safetensors, gguf, pytorch, ...            │
└─────────────────────────────────────────────────────┘
                      ↓ load
┌─────────────────────────────────────────────────────┐
│  hodu_core                                          │
│  - Script / Snapshot IR (플랫폼 독립적)                │
│  - CaptureBoard (연산 그래프 캡처)                      │
└─────────────────────────────────────────────────────┘
                      ↓ compile/execute
┌─────────────────────────────────────────────────────┐
│  Backend Plugin (컴파일 + 실행)                        │
│  - llvm: LLVM JIT/AOT → CPU/CUDA/ROCm               │
│  - metal: MSL → Metal                               │
│  - xla: XLA 컴파일러 → CPU/GPU/TPU                    │
│  - onnxruntime: ONNX Runtime 위임                    │
│  - tvm: TVM 컴파일러                                  │
│  - interp: 순수 인터프리터 (builtin)                    │
└─────────────────────────────────────────────────────┘
                      ↓ targets
┌─────────────────────────────────────────────────────┐
│  Device (하드웨어)                                    │
│  - CPU, CUDA, Metal, ROCm, TPU, ...                 │
└─────────────────────────────────────────────────────┘
```

### 컴파일 체인 (GPU)

```
Script (Snapshot IR)
       ↓
┌──────────────────────────────────────────────────────┐
│  Backend Plugin                                      │
│                                                      │
│  ┌─────────────┐    ┌─────────────┐                  │
│  │ Host Code   │    │ Device Code │                  │
│  │ (CPU)       │    │ (GPU)       │                  │
│  └─────────────┘    └─────────────┘                  │
│        ↓                  ↓                          │
│     LLVM IR           ┌───┴───┐                      │
│        ↓              ↓       ↓                      │
│   Native Code      PTX      MSL/AIR                  │
│   (x86/arm)         ↓         ↓                      │
│                   cubin    metallib                  │
└──────────────────────────────────────────────────────┘
       ↓                 ↓           ↓
      CPU              CUDA        Metal
```

### Device별 IR 및 출력 포맷

| Device | IR | 출력 포맷 옵션 |
|--------|-----|---------------|
| CPU | LLVM IR | `.o`, `.so`/`.dylib`/`.dll`, `.a`/`.lib`, 실행파일, `.ll`, `.bc`, `.s` |
| CUDA | PTX / LLVM NVPTX | `.ptx`, `.cubin`, `.fatbin` |
| Metal | MSL → AIR | `.metal`, `.air`, `.metallib` |
| ROCm | LLVM AMDGPU | `.s`, `.hsaco` |
| Vulkan/OpenCL | SPIR-V | `.spv` |

---

## 플러그인 타입

### 1. Format Plugin

모델 포맷 로드/저장 담당.

```rust
pub trait FormatPlugin: Send + Sync {
    /// 플러그인 이름
    fn name(&self) -> &str;

    /// 지원하는 확장자 목록
    fn extensions(&self) -> &[&str];

    /// 파일 → Script 로드
    fn load(&self, path: &Path) -> Result<Script>;

    /// Script → 파일 저장 (선택적)
    fn save(&self, script: &Script, path: &Path) -> Result<()>;
}
```

**예시 플러그인:**
- `hodu-format-onnx`: `.onnx` 파일 로드/저장
- `hodu-format-safetensors`: `.safetensors` 파일 로드
- `hodu-format-gguf`: `.gguf` 파일 로드 (llama.cpp 호환)
- `hodu-format-pytorch`: `.pt`, `.pth` 파일 로드

### 2. Backend Plugin

컴파일 + 실행 담당. 각 Backend는 여러 Device를 지원할 수 있음.

```rust
pub trait BackendPlugin: Send + Sync {
    /// 플러그인 이름
    fn name(&self) -> &str;

    /// 지원하는 디바이스 목록
    fn supported_devices(&self) -> &[Device];

    /// 디바이스별 지원하는 출력 포맷
    fn supported_formats(&self, device: Device) -> &[OutputFormat];

    /// JIT 컴파일 (메모리에 로드)
    fn compile(&self, script: &Script, device: Device) -> Result<CompiledModule>;

    /// AOT 빌드 (파일로 출력)
    fn build(
        &self,
        script: &Script,
        device: Device,
        format: OutputFormat,
        path: &Path,
    ) -> Result<()>;

    /// 실행
    fn execute(
        &self,
        module: &CompiledModule,
        inputs: &[(&str, &Tensor)],
    ) -> Result<HashMap<String, Tensor>>;
}
```

**예시 플러그인:**
- `hodu-backend-llvm`: CPU/CUDA/ROCm 지원 (LLVM 기반)
- `hodu-backend-metal`: Metal 지원 (MSL 기반)
- `hodu-backend-xla`: XLA 런타임 (CPU/GPU/TPU)
- `hodu-backend-onnxruntime`: ONNX Runtime 위임
- `hodu-backend-tvm`: TVM 컴파일러
- `hodu-backend-interp`: 순수 인터프리터 (builtin, 기본 제공)

---

## 출력 포맷 정의

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    // === CPU (Native) ===
    Object,           // .o
    SharedLib,        // .so / .dylib / .dll
    StaticLib,        // .a / .lib
    Executable,       // 실행파일

    // === LLVM IR (디버깅용) ===
    LlvmIR,           // .ll (텍스트)
    LlvmBitcode,      // .bc (바이너리)
    Assembly,         // .s

    // === CUDA ===
    Ptx,              // .ptx (텍스트 IR)
    Cubin,            // .cubin (단일 아키텍처 바이너리)
    Fatbin,           // .fatbin (멀티 아키텍처)

    // === Metal ===
    Msl,              // .metal (소스 코드)
    Air,              // .air (IR)
    Metallib,         // .metallib (바이너리)

    // === ROCm ===
    Hsaco,            // .hsaco (AMD GPU 바이너리)

    // === Portable ===
    SpirV,            // .spv (Vulkan/OpenCL)
}

impl OutputFormat {
    /// 확장자로부터 포맷 추론
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "o" => Some(Self::Object),
            "so" | "dylib" | "dll" => Some(Self::SharedLib),
            "a" | "lib" => Some(Self::StaticLib),
            "ll" => Some(Self::LlvmIR),
            "bc" => Some(Self::LlvmBitcode),
            "s" => Some(Self::Assembly),
            "ptx" => Some(Self::Ptx),
            "cubin" => Some(Self::Cubin),
            "fatbin" => Some(Self::Fatbin),
            "metal" => Some(Self::Msl),
            "air" => Some(Self::Air),
            "metallib" => Some(Self::Metallib),
            "hsaco" => Some(Self::Hsaco),
            "spv" => Some(Self::SpirV),
            _ => None,
        }
    }

    /// 기본 확장자 반환
    pub fn extension(&self) -> &str {
        match self {
            Self::Object => "o",
            Self::SharedLib => {
                #[cfg(target_os = "linux")]
                { "so" }
                #[cfg(target_os = "macos")]
                { "dylib" }
                #[cfg(target_os = "windows")]
                { "dll" }
            },
            Self::StaticLib => {
                #[cfg(any(target_os = "linux", target_os = "macos"))]
                { "a" }
                #[cfg(target_os = "windows")]
                { "lib" }
            },
            Self::Executable => "",
            Self::LlvmIR => "ll",
            Self::LlvmBitcode => "bc",
            Self::Assembly => "s",
            Self::Ptx => "ptx",
            Self::Cubin => "cubin",
            Self::Fatbin => "fatbin",
            Self::Msl => "metal",
            Self::Air => "air",
            Self::Metallib => "metallib",
            Self::Hsaco => "hsaco",
            Self::SpirV => "spv",
        }
    }
}
```

---

## 플러그인 시스템 구현

### 디렉토리 구조

```
~/.hodu/
├── plugins/
│   ├── hodu-backend-llvm.dylib
│   ├── hodu-backend-metal.dylib
│   ├── hodu-format-onnx.dylib
│   └── ...
├── config.toml
└── cache/
```

### 플러그인 ABI (C ABI 안정성)

`abi_stable` crate 사용하여 Rust ABI 안정화.

```rust
// hodu_plugin_api/src/lib.rs

use abi_stable::StableAbi;

#[repr(C)]
#[derive(StableAbi)]
pub struct PluginInfo {
    pub name: RString,
    pub version: RString,
    pub plugin_type: PluginType,
}

#[repr(C)]
#[derive(StableAbi)]
pub enum PluginType {
    Format,
    Backend,
}

// 모든 플러그인이 export해야 하는 함수
#[no_mangle]
pub extern "C" fn hodu_plugin_info() -> PluginInfo;

#[no_mangle]
pub extern "C" fn hodu_plugin_init() -> RBox<dyn BackendPlugin>;
// 또는
pub extern "C" fn hodu_plugin_init() -> RBox<dyn FormatPlugin>;
```

### 플러그인 로더

```rust
// hodu/src/plugin/loader.rs

use abi_stable::library::RootModule;
use libloading::Library;

pub struct PluginManager {
    backends: HashMap<String, Box<dyn BackendPlugin>>,
    formats: HashMap<String, Box<dyn FormatPlugin>>,
    libraries: Vec<Library>,  // Keep libraries alive
}

impl PluginManager {
    pub fn new() -> Self { ... }

    /// 플러그인 디렉토리 스캔 및 로드
    pub fn load_all(&mut self, plugin_dir: &Path) -> Result<()> { ... }

    /// 특정 플러그인 로드
    pub fn load(&mut self, path: &Path) -> Result<()> { ... }

    /// Backend 플러그인 가져오기
    pub fn backend(&self, name: &str) -> Option<&dyn BackendPlugin> { ... }

    /// Format 플러그인 가져오기 (확장자로)
    pub fn format_for_extension(&self, ext: &str) -> Option<&dyn FormatPlugin> { ... }
}
```

---

## CLI 명령어

### 플러그인 관리

```bash
# 설치된 플러그인 목록
hodu plugin list
> Backends:
>   interp      1.0.0  [cpu]  (builtin)
>   llvm        1.0.0  [cpu, cuda, rocm]
>   metal       1.0.0  [metal]
> Formats:
>   hdss        1.0.0  [.hdss]  (builtin)
>   onnx        1.0.0  [.onnx]

# 플러그인 상세 정보
hodu plugin info llvm
> Backend: llvm
> Version: 1.0.0
> Devices:
>   cpu   → [object, shared, static, executable, llvm-ir, llvm-bc, asm]
>   cuda  → [ptx, cubin, fatbin, llvm-ir]
>   rocm  → [hsaco, llvm-ir, asm]

# 플러그인 설치
hodu plugin add xla
hodu plugin add onnx

# 플러그인 제거
hodu plugin remove xla

# 사용 가능한 플러그인 검색
hodu plugin search cuda
> Available:
>   hodu-backend-llvm    1.0.0  (supports cuda)
>   hodu-backend-xla     1.0.0  (supports cuda)
```

### 모델 실행 (JIT)

```bash
# 기본 실행 (interp backend, cpu)
hodu run model.hdss --input input.json

# Backend/Device 지정
hodu run model.hdss --backend llvm --device cpu
hodu run model.hdss --backend llvm --device cuda:0
hodu run model.hdss --backend metal --device metal
hodu run model.hdss --backend xla --device tpu

# ONNX 모델 실행 (onnx format plugin 필요)
hodu run model.onnx --backend onnxruntime --device cpu

# 출력 저장
hodu run model.hdss --input input.json -o output.json
hodu run model.hdss --input input.json -o output.npy
```

### AOT 빌드

```bash
# 공유 라이브러리 (확장자로 포맷 자동 추론)
hodu build model.hdss -o libmodel.so --backend llvm --device cpu
hodu build model.hdss -o libmodel.dylib --backend llvm --device cpu

# 포맷 명시적 지정
hodu build model.hdss -o model --backend llvm --device cpu --format shared

# CUDA PTX/cubin
hodu build model.hdss -o model.ptx --backend llvm --device cuda
hodu build model.hdss -o model.cubin --backend llvm --device cuda --format cubin
hodu build model.hdss -o model.fatbin --backend llvm --device cuda --format fatbin

# Metal
hodu build model.hdss -o model.metal --backend metal --device metal --format msl
hodu build model.hdss -o model.metallib --backend metal --device metal

# 디버깅용 IR 출력
hodu build model.hdss -o model.ll --backend llvm --device cpu --format llvm-ir
hodu build model.hdss -o model.s --backend llvm --device cpu --format asm

# 크로스 컴파일
hodu build model.hdss -o libmodel.so --backend llvm --device cpu --target x86_64-unknown-linux-gnu
hodu build model.hdss -o libmodel.so --backend llvm --device cpu --target aarch64-apple-darwin

# 최적화 레벨
hodu build model.hdss -o libmodel.so -O0
hodu build model.hdss -o libmodel.so -O3
```

### 모델 정보/변환

```bash
# 모델 정보
hodu info model.hdss
hodu info model.onnx

# 포맷 변환
hodu convert model.onnx -o model.hdss
hodu convert model.hdss -o model.onnx
```

---

## Crate 구조

```
hodu/                           # CLI + 플러그인 로더
├── src/
│   ├── main.rs
│   ├── cli.rs
│   └── plugin/
│       ├── mod.rs
│       ├── loader.rs
│       └── manager.rs

hodu_core/                      # 순수 IR (Script/Snapshot)
├── src/
│   ├── script/
│   │   ├── capture/           # CaptureBoard
│   │   └── snapshot.rs        # Snapshot IR
│   └── ...

hodu_plugin_api/                # 플러그인 인터페이스 (공유)
├── src/
│   ├── lib.rs
│   ├── backend.rs             # BackendPlugin trait
│   ├── format.rs              # FormatPlugin trait
│   └── types.rs               # Device, OutputFormat, etc.

# Backend 플러그인들
hodu-backend-llvm/
hodu-backend-metal/
hodu-backend-xla/
hodu-backend-onnxruntime/
hodu-backend-tvm/
hodu-backend-interp/           # builtin (hodu에 포함)

# Format 플러그인들
hodu-format-onnx/
hodu-format-safetensors/
hodu-format-gguf/
hodu-format-pytorch/
```

---

## 구현 계획

### Phase 1: Core 분리

- [ ] hodu_core에서 builder/, compiled/ 제거
- [ ] hodu_core는 Script/Snapshot/CaptureBoard만 유지
- [ ] 기존 Script의 compile/run 메서드 제거

### Phase 2: Plugin API 설계

- [ ] hodu_plugin_api crate 생성
- [ ] BackendPlugin trait 정의
- [ ] FormatPlugin trait 정의
- [ ] Device, OutputFormat enum 정의
- [ ] abi_stable 적용

### Phase 3: 플러그인 로더 구현

- [ ] PluginManager 구현
- [ ] 동적 라이브러리 로딩 (libloading)
- [ ] 플러그인 디렉토리 관리 (~/.hodu/plugins/)
- [ ] 버전 호환성 체크

### Phase 4: Backend 플러그인 구현

- [ ] hodu-backend-interp (builtin, 순수 인터프리터)
- [ ] hodu-backend-llvm (기존 코드 이전)
  - [ ] CPU codegen
  - [ ] CUDA codegen (PTX)
  - [ ] ROCm codegen
- [ ] hodu-backend-metal
  - [ ] MSL codegen
  - [ ] Metal runtime

### Phase 5: Format 플러그인 구현

- [ ] hdss format (builtin)
- [ ] hodu-format-onnx

### Phase 6: CLI 업데이트

- [ ] `hodu plugin` 서브커맨드
- [ ] `hodu run` 업데이트 (backend/device 옵션)
- [ ] `hodu build` 업데이트 (format 옵션)
- [ ] 플러그인 레지스트리 연동 (선택적)

---

## 참고: Backend별 지원 매트릭스

| Backend | CPU | CUDA | ROCm | Metal | TPU |
|---------|-----|------|------|-------|-----|
| interp | O | - | - | - | - |
| llvm | O | O | O | - | - |
| metal | - | - | - | O | - |
| xla | O | O | - | - | O |
| onnxruntime | O | O | O | - | - |
| tvm | O | O | O | O | - |

| Backend | .so/.dylib | .ptx | .cubin | .metallib | .spv |
|---------|------------|------|--------|-----------|------|
| llvm | O | O | O | - | - |
| metal | - | - | - | O | - |
| xla | - | - | - | - | - |
