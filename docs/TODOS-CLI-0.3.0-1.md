# Hodu CLI & Plugin System 0.3.0-1

## 목표

1. **hodu_plugin_sdk**를 플러그인 개발용 SDK crate로 새로 제공 (기존 hodu_plugin에서 분리/rename)
2. 누구나 hodu_plugin_sdk만 의존해서 Backend/Format 플러그인 만들 수 있게
3. `hodu plugin install` 명령으로 crates.io 또는 GitHub에서 플러그인 설치
4. 설치 시 소스에서 빌드하여 `~/.hodu/plugins/`에 저장

---

## 프로젝트 구조

### Monorepo 유지 (hodu/)

```
hodu/                           # 메인 workspace
├── Cargo.toml
├── crates/
│   ├── hodu_core/              # 코어 라이브러리
│   ├── hodu_plugin_sdk/        # 플러그인 SDK (NEW, hodu_plugin에서 분리)
│   ├── hodu_nn/
│   ├── hodu_utils/
│   ├── hodu_compat/
│   ├── hodu_internal/
│   └── hodu_*_kernels/
├── cli/                        # CLI crate (src/cli.rs에서 분리)
│   ├── Cargo.toml
│   └── src/
└── src/lib.rs                  # hodu 라이브러리
```

### 플러그인 별도 repo

```
hodu-plugins/                   # 별도 repository
├── hodu-backend-cpu/
├── hodu-backend-metal/
├── hodu-format-onnx/
├── hodu-format-npy/
└── hodu-format-safetensors/
```

**플러그인 분리 이유:**
- 커뮤니티 플러그인과 동일한 구조
- `hodu_plugin_sdk`만 의존 (hodu_core 직접 의존 X)
- 독립 버전 관리/배포

---

## Builtin vs Plugin

| 기능 | 유형 | 설명 |
|------|------|------|
| **hdss 포맷** | builtin | Snapshot 직렬화 (기본 모델 포맷) |
| **hdt 포맷** | builtin | Tensor 바이너리 포맷 |
| **json 포맷** | builtin | Tensor JSON 포맷 |
| **interp backend** | builtin | 인터프리터 (플러그인 없이 CPU 실행) |
| ONNX | plugin | hodu-format-onnx |
| safetensors | plugin | hodu-format-safetensors |
| npy/npz | plugin | hodu-format-npy |
| CPU backend | plugin | hodu-backend-cpu (AOT 컴파일) |
| Metal backend | plugin | hodu-backend-metal |
| CUDA backend | plugin | hodu-backend-cuda |

---

## 플러그인 타입 2종

| 타입 | 역할 | CLI | 예시 |
|------|------|-----|------|
| **BackendPlugin** | 실행(Runner) + 빌드(Builder) | `hodu run`, `hodu build` | hodu-backend-cpu, hodu-backend-metal |
| **FormatPlugin** | 파일 ↔ Snapshot/Tensor 변환 | 자동 (확장자 기반) | hodu-format-onnx, hodu-format-npy |

### BackendPlugin Capabilities

하나의 플러그인이 Runner/Builder 중 하나 또는 둘 다 제공 가능:

| 플러그인 | Runner | Builder | 설명 |
|----------|:------:|:-------:|------|
| `hodu-backend-cpu` | ✅ | ✅ | CPU 실행 + 네이티브 빌드 |
| `hodu-backend-metal` | ✅ | ✅ | Metal 실행 + metallib 빌드 |
| `hodu-backend-cuda` | ✅ | ✅ | CUDA 실행 + ptx/cubin 빌드 |
| `hodu-backend-interp` | ✅ | ❌ | 인터프리터 (빌드 없음, builtin) |
| `hodu-backend-llvm` | ❌ | ✅ | AOT 전용 (크로스 컴파일) |

### FormatPlugin Capabilities

| 플러그인 | 확장자 | load_model | save_model | load_tensor | save_tensor |
|----------|--------|:----------:|:----------:|:-----------:|:-----------:|
| `hodu-format-onnx` | .onnx | ✅ | ✅ | ❌ | ❌ |
| `hodu-format-safetensors` | .safetensors | ❌ | ❌ | ✅ | ✅ |
| `hodu-format-npy` | .npy, .npz | ❌ | ❌ | ✅ | ✅ |
| `hodu-format-gguf` | .gguf | ✅ | ❌ | ❌ | ❌ |
| `hodu-format-pytorch` | .pt, .pth | ❌ | ❌ | ✅ | ❌ |

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│  hodu_plugin_sdk (crates.io) - 플러그인 SDK                   │
│  ───────────────────────────────────────────────            │
│  - BackendPlugin trait (Runner + Builder)                   │
│  - FormatPlugin trait                                       │
│  - BackendCapabilities, FormatCapabilities                  │
│  - DispatchManifest, build_metadata()                       │
│  - BuildTarget, OutputFormat, Device                        │
│  - hodu_core 타입 re-export (DType, Shape, Layout, Op 등)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 의존
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  플러그인들 (독립 crate, cdylib 빌드)                          │
│  ───────────────────────────────────────────────            │
│  Backend:                                                   │
│    hodu-backend-cpu       (공식, Runner+Builder)             │
│    hodu-backend-metal     (공식, Runner+Builder)             │
│    hodu-backend-cuda      (커뮤니티, Runner+Builder)          │
│    hodu-backend-llvm      (공식, Builder only)               │
│                                                             │
│  Format:                                                    │
│    hodu-format-onnx       (공식)                             │
│    hodu-format-safetensors (공식)                            │
│    hodu-format-npy        (공식)                             │
│    hodu-format-gguf       (커뮤니티)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ hodu plugin install
                              │ (소스 다운 → cargo build → dylib 복사)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  ~/.hodu/                                                   │
│  ├── plugins/                                               │
│  │   ├── hodu-backend-cpu.dylib                             │
│  │   ├── hodu-backend-metal.dylib                           │
│  │   ├── hodu-format-onnx.dylib                             │
│  │   └── hodu-format-npy.dylib                              │
│  ├── plugins.json          # 설치된 플러그인 메타데이터        │
│  └── config.toml           # CLI 설정                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ hodu run / hodu build
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  hodu CLI                                                   │
│  ───────────────────────────────────────────────            │
│  - 플러그인 로드 (libloading)                                 │
│  - 확장자로 FormatPlugin 자동 선택                            │
│  - device/target으로 BackendPlugin 자동 선택                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 플러그인 선택 로직

### Backend 선택

`hodu run --device <DEVICE>` 실행 시:

1. `--backend <NAME>` 명시적 지정 → 해당 플러그인 사용
2. 미지정 시 → `config.toml`의 `[backends].priority` 순서대로 검색
3. 해당 device 지원하는 첫 번째 플러그인 선택
4. 없으면 builtin interp 시도 (CPU만)
5. 그래도 없으면 에러 + 설치 안내

```bash
# 명시적 백엔드 선택
hodu run model.onnx --device cpu --backend llvm

# 자동 선택 (config.toml priority 따름)
hodu run model.onnx --device cpu
```

### Format 선택

파일 확장자 기반 자동 선택:

1. `config.toml`의 `[formats]`에 확장자 오버라이드 있으면 사용
2. 없으면 해당 확장자 지원하는 플러그인 검색
3. 여러 개면 먼저 설치된 것 사용
4. builtin 포맷(hdss, hdt, json)은 항상 우선

```bash
# .onnx → hodu-format-onnx 자동 선택
hodu run model.onnx -i x=input.npy

# .npy → hodu-format-npy 자동 선택 (입력 텐서)
```

### 선택 로직 디버깅

```bash
# 어떤 플러그인이 선택되는지 확인
hodu run model.onnx --device metal --dry-run

# 출력:
# Model format: onnx (hodu-format-onnx 0.3.0)
# Input x.npy: npy (hodu-format-npy 0.1.0)
# Backend: metal (hodu-backend-metal 0.3.0)
#
# Would execute with above configuration.
```

---

## 캐싱 전략

### 컴파일 캐시

AOT 컴파일 결과를 캐싱하여 재실행 시 컴파일 생략:

```
~/.hodu/cache/
├── compile/
│   ├── <model_hash>_<device>_<backend>.cache
│   └── ...
└── metadata.json
```

**캐시 키 구성:**
- 모델 파일 해시 (SHA256)
- 디바이스 (cpu, metal, cuda:0)
- 백엔드 플러그인 이름 + 버전
- 최적화 레벨

```rust
// CLI 레벨 캐싱
fn get_or_compile(
    snapshot: &Snapshot,
    backend: &dyn BackendPlugin,
    device: Device,
) -> HoduResult<CachedModule> {
    let cache_key = compute_cache_key(snapshot, backend, device);

    if let Some(cached) = cache.get(&cache_key) {
        return Ok(cached);
    }

    let module = backend.compile(snapshot, device)?;
    cache.put(&cache_key, &module)?;
    Ok(module)
}
```

### 캐시 관리

```bash
# 캐시 상태 확인
hodu cache status
# Cache: 156 MB / 1024 MB (15%)
# Entries: 23

# 캐시 정리
hodu cache clean           # 오래된 항목 정리
hodu cache clean --all     # 전체 삭제

# 특정 모델 캐시 삭제
hodu cache clean model.onnx
```

### config.toml 설정

```toml
[build]
cache_enabled = true
cache_max_size = 1024      # MB
cache_ttl = 30             # 일 (미사용 시 삭제)
```

---

## Plugin Traits

### BackendPlugin

```rust
pub trait BackendPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;

    /// 이 플러그인이 제공하는 기능
    fn capabilities(&self) -> BackendCapabilities;

    // === Runner 기능 (hodu run) ===

    /// Runner로서 지원하는 디바이스
    fn supported_devices(&self) -> Vec<Device>;

    /// 모델 실행
    fn run(
        &self,
        snapshot: &Snapshot,
        device: Device,
        inputs: &[(&str, TensorData)],
    ) -> HoduResult<HashMap<String, TensorData>>;

    // === Builder 기능 (hodu build) ===

    /// Builder로서 지원하는 타겟
    fn supported_targets(&self) -> Vec<BuildTarget>;

    /// 특정 타겟에서 지원하는 출력 포맷
    fn supported_formats(&self, target: &BuildTarget) -> Vec<OutputFormat>;

    /// AOT 빌드
    fn build(
        &self,
        snapshot: &Snapshot,
        target: &BuildTarget,
        format: OutputFormat,
        output: &Path,
    ) -> HoduResult<()>;
}

pub struct BackendCapabilities {
    pub runner: bool,    // hodu run 지원 여부
    pub builder: bool,   // hodu build 지원 여부
}

pub struct BuildTarget {
    pub triple: String,   // x86_64-unknown-linux-gnu, aarch64-apple-darwin, ...
    pub device: Device,   // CPU, Metal, CUDA
}
```

### FormatPlugin

```rust
pub trait FormatPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;

    /// 이 플러그인이 제공하는 기능
    fn capabilities(&self) -> FormatCapabilities;

    /// 지원하는 파일 확장자
    fn supported_extensions(&self) -> Vec<&str>;

    // === 모델 로드/저장 ===

    /// 파일 → Snapshot
    fn load_model(&self, path: &Path) -> HoduResult<Snapshot>;
    fn load_model_from_bytes(&self, data: &[u8]) -> HoduResult<Snapshot>;

    /// Snapshot → 파일
    fn save_model(&self, snapshot: &Snapshot, path: &Path) -> HoduResult<()>;
    fn save_model_to_bytes(&self, snapshot: &Snapshot) -> HoduResult<Vec<u8>>;

    // === 텐서 로드/저장 ===

    /// 파일 → TensorData
    fn load_tensor(&self, path: &Path) -> HoduResult<TensorData>;
    fn load_tensor_from_bytes(&self, data: &[u8]) -> HoduResult<TensorData>;

    /// TensorData → 파일
    fn save_tensor(&self, tensor: &TensorData, path: &Path) -> HoduResult<()>;
    fn save_tensor_to_bytes(&self, tensor: &TensorData) -> HoduResult<Vec<u8>>;
}

pub struct FormatCapabilities {
    pub load_model: bool,
    pub save_model: bool,
    pub load_tensor: bool,
    pub save_tensor: bool,
}
```

---

## CLI 명령어

### hodu run

모델 실행 (BackendPlugin의 Runner 기능 사용)

```bash
hodu run [OPTIONS] <MODEL>

Arguments:
  <MODEL>  모델 파일 (.onnx, .hdss, ...)

Options:
  -d, --device <DEVICE>     실행 디바이스 (cpu, metal, cuda:0) [default: cpu]
  -i, --input <INPUT>       입력 텐서 (name=path), 반복 가능
  -I, --inputs <INPUTS>     입력 텐서들 (콤마 구분)
  -f, --format <FORMAT>     출력 포맷 (pretty, json, hdt) [default: pretty]
  -o, --output-dir <DIR>    출력 디렉토리 (hdt 포맷용)
      --benchmark [N]       벤치마크 모드 (N회 실행, 기본 10)
      --profile             프로파일링 모드
      --warmup <N>          워밍업 실행 횟수 [default: 3]
  -h, --help                도움말
```

**예시:**
```bash
# 기본 실행 (CPU, builtin interp)
hodu run model.onnx -i x=input.npy

# Metal GPU 실행
hodu run model.onnx -i x=input.npy --device metal

# CUDA 실행
hodu run model.onnx -i x=input.npy --device cuda:0

# 여러 입력
hodu run model.onnx -i a=a.npy -i b=b.safetensors --device metal
```

**벤치마크:**
```bash
hodu run model.onnx -i x=input.npy --benchmark

# Warmup: 3 runs
# Benchmark: 10 runs
#
# Results:
#   Mean:   12.34 ms
#   Std:     0.82 ms
#   Min:    11.21 ms
#   Max:    14.08 ms
#   P50:    12.15 ms
#   P95:    13.89 ms
#   P99:    14.02 ms

# 50회 벤치마크
hodu run model.onnx -i x=input.npy --benchmark 50 --warmup 5
```

**프로파일링:**
```bash
hodu run model.onnx -i x=input.npy --profile

# Profile:
#   Total:        15.42 ms
#   ─────────────────────────
#   Load model:    1.23 ms (  8.0%)
#   Compile:       0.00 ms (  0.0%)  [cached]
#   Execute:      14.19 ms ( 92.0%)
#
# Top operators:
#   Conv_0:        3.21 ms ( 22.6%)
#   Conv_1:        2.87 ms ( 20.2%)
#   MatMul_0:      1.94 ms ( 13.7%)
#   Conv_2:        1.52 ms ( 10.7%)
#   ...
#
# Memory:
#   Peak:         182.4 MB
#   Weights:       44.8 MB
#   Activations:  137.6 MB

hodu run model.onnx -i x=input.npy --profile --format json
# {"total_ms": 15.42, "load_ms": 1.23, "compile_ms": 0.0, ...}
```

### hodu build

AOT 빌드 (BackendPlugin의 Builder 기능 사용)

```bash
hodu build [OPTIONS] <MODEL>

Arguments:
  <MODEL>  모델 파일 (.onnx, .hdss, ...)

Options:
  -o, --output <OUTPUT>     출력 파일 경로
  -t, --target <TARGET>     타겟 triple (x86_64-unknown-linux-gnu, ...) [default: 현재 시스템]
  -d, --device <DEVICE>     타겟 디바이스 (cpu, metal, cuda) [default: cpu]
  -f, --format <FORMAT>     출력 포맷 (sharedlib, staticlib, object, metallib, ptx, ...)
  -O, --opt-level <LEVEL>   최적화 레벨 (0-3) [default: 2]
      --standalone          독립 실행파일 생성
  -h, --help                도움말
```

**예시:**
```bash
# 현재 시스템용 shared library
hodu build model.onnx -o libmodel.dylib

# Linux x86_64 크로스 컴파일
hodu build model.onnx -o libmodel.so --target x86_64-unknown-linux-gnu

# Windows 크로스 컴파일
hodu build model.onnx -o model.dll --target x86_64-pc-windows-msvc

# Metal library
hodu build model.onnx -o model.metallib --device metal

# CUDA PTX
hodu build model.onnx -o model.ptx --device cuda

# 독립 실행파일
hodu build model.onnx -o inference --standalone
```

**최적화 레벨:**
```bash
hodu build model.onnx -o model.dylib -O0   # 최적화 없음 (빠른 컴파일)
hodu build model.onnx -o model.dylib -O1   # 기본 (상수 폴딩)
hodu build model.onnx -o model.dylib -O2   # 표준 (+ 연산자 융합) [default]
hodu build model.onnx -o model.dylib -O3   # 최대 (+ 레이아웃 최적화, 느린 컴파일)

# 적용되는 최적화 확인
hodu build model.onnx -o model.dylib -O2 --verbose
# Optimizations applied:
#   ✓ Constant folding: 12 ops eliminated
#   ✓ Operator fusion: 8 patterns matched
#     - Conv+BatchNorm: 4
#     - Conv+ReLU: 3
#     - MatMul+Add: 1
#   ✓ Dead code elimination: 3 ops removed
#   Total: 152 ops → 129 ops
```

### --standalone 상세

독립 실행파일은 모델과 런타임을 하나의 바이너리로 패키징:

**생성되는 구조:**
```
inference (실행파일)
├── embedded model (serialized snapshot)
├── embedded weights
└── minimal runtime (interpreter or compiled kernels)
```

**사용법:**
```bash
# 빌드
hodu build model.onnx -o inference --standalone

# 실행
./inference --input x=input.bin --output output.bin
./inference --input x=input.bin --output-format json
./inference --help

# 옵션
./inference [OPTIONS]

Options:
  -i, --input <NAME=PATH>   입력 텐서 (바이너리 파일)
  -o, --output <PATH>       출력 디렉토리 또는 파일
  -f, --output-format       출력 포맷 (bin, json) [default: bin]
  --list-inputs             입력 텐서 정보 출력
  --list-outputs            출력 텐서 정보 출력
  -h, --help                도움말
```

**입출력 포맷:**
```bash
# 입력: raw binary (shape/dtype은 모델에서 가져옴)
./inference -i x=input.bin -o output/

# 출력 파일들
output/
├── output_0.bin    # raw f32 bytes
└── manifest.json   # shape, dtype 정보

# JSON 출력
./inference -i x=input.bin -f json
# {"output_0": {"shape": [1, 10], "dtype": "f32", "data": [0.1, 0.2, ...]}}
```

**크기 최적화:**
```bash
# 기본 (interpreter 포함)
hodu build model.onnx -o inference --standalone
# → ~2MB + model size

# 최적화 (release + strip)
hodu build model.onnx -o inference --standalone --release --strip
# → ~500KB + model size

# weights 외부화 (큰 모델용)
hodu build model.onnx -o inference --standalone --external-weights
# → inference (코드만) + model.weights (별도 파일)
```

### hodu inspect

모델 파일 정보 확인:

```bash
hodu inspect [OPTIONS] <MODEL>

Arguments:
  <MODEL>  모델 파일 (.onnx, .hdss, .gguf, ...)

Options:
  -v, --verbose           상세 정보 출력
  -f, --format <FORMAT>   출력 포맷 (pretty, json) [default: pretty]
  -h, --help              도움말
```

**예시:**
```bash
hodu inspect model.onnx

# Model: model.onnx
# Format: ONNX (hodu-format-onnx 0.3.0)
# Size: 45.2 MB
#
# Inputs:
#   x: f32[1, 3, 224, 224]
#
# Outputs:
#   output: f32[1, 1000]
#
# Operations: 152
#   Conv: 53, BatchNorm: 52, ReLU: 49, ...

hodu inspect model.onnx --verbose

# ... (위 내용 + 상세 op 목록, 메모리 추정치 등)
#
# Estimated memory: ~180 MB (inference)
# Weights size: 44.8 MB
# Intermediate tensors: ~135 MB (peak)

hodu inspect model.onnx --format json
# {"name": "model.onnx", "format": "onnx", "size_bytes": 47420416, ...}
```

### hodu convert

모델 포맷 변환:

```bash
hodu convert [OPTIONS] <INPUT> -o <OUTPUT>

Arguments:
  <INPUT>   입력 모델 파일

Options:
  -o, --output <OUTPUT>   출력 파일 경로 (필수)
  -f, --format <FORMAT>   출력 포맷 (확장자에서 자동 추론)
  -h, --help              도움말
```

**예시:**
```bash
# ONNX → HDSS
hodu convert model.onnx -o model.hdss

# GGUF → HDSS
hodu convert model.gguf -o model.hdss

# 포맷 명시
hodu convert model.onnx -o model.bin --format hdss
```

### hodu version

버전 정보 출력:

```bash
hodu version

# hodu 0.3.0
# hodu_plugin_sdk 0.3.0
# Platform: aarch64-apple-darwin
# Rust: 1.75.0
#
# Installed plugins:
#   hodu-backend-cpu    0.3.0  crates.io
#   hodu-backend-metal  0.3.0  crates.io
#   hodu-format-onnx    0.3.0  crates.io

hodu version --short
# 0.3.0

hodu version --json
# {"version": "0.3.0", "sdk_version": "0.3.0", "platform": "aarch64-apple-darwin", ...}
```

### hodu completions

쉘 자동완성 스크립트 생성:

```bash
# Bash
hodu completions bash > ~/.bash_completion.d/hodu
# 또는
hodu completions bash | sudo tee /etc/bash_completion.d/hodu

# Zsh
hodu completions zsh > ~/.zfunc/_hodu
# ~/.zshrc에 추가: fpath=(~/.zfunc $fpath)

# Fish
hodu completions fish > ~/.config/fish/completions/hodu.fish

# PowerShell
hodu completions powershell > $PROFILE.CurrentUserAllHosts
```

### hodu plugin install

```bash
# crates.io에서 설치
hodu plugin install cpu              # hodu-backend-cpu
hodu plugin install metal            # hodu-backend-metal
hodu plugin install onnx             # hodu-format-onnx
hodu plugin install hodu-backend-cpu # 풀네임

# 특정 버전
hodu plugin install cpu@0.3.0

# GitHub에서 설치
hodu plugin install --git https://github.com/user/hodu-backend-tpu
hodu plugin install --git https://github.com/user/hodu-backend-tpu --tag v0.1.0

# 로컬 경로에서 빌드
hodu plugin install --path ./my-plugin
```

### hodu plugin list

```bash
hodu plugin list

# 출력:
# Backend plugins:
#   cpu       0.3.0  [Runner, Builder]  devices: cpu     crates.io
#   metal     0.3.0  [Runner, Builder]  devices: metal   crates.io
#   llvm      0.3.0  [Builder]          targets: *       github:user/repo
#
# Format plugins:
#   onnx          0.3.0  [load_model, save_model]      .onnx          crates.io
#   safetensors   0.1.0  [load_tensor, save_tensor]    .safetensors   crates.io
#   npy           0.1.0  [load_tensor, save_tensor]    .npy, .npz     crates.io
```

### hodu plugin remove

```bash
hodu plugin remove cpu
hodu plugin remove hodu-format-onnx
```

### hodu plugin update

```bash
hodu plugin update          # 모든 플러그인
hodu plugin update cpu      # 특정 플러그인
```

### hodu plugin search

crates.io에서 사용 가능한 플러그인 검색:

```bash
hodu plugin search onnx
# Backend plugins:
#   (none)
#
# Format plugins:
#   hodu-format-onnx  0.3.0  [load_model, save_model]  .onnx

hodu plugin search --type backend
# Backend plugins:
#   hodu-backend-cpu    0.3.0  [Runner, Builder]  cpu
#   hodu-backend-metal  0.3.0  [Runner, Builder]  metal
#   hodu-backend-cuda   0.2.0  [Runner, Builder]  cuda
#   hodu-backend-llvm   0.1.0  [Builder]          cpu (cross-compile)

hodu plugin search --type format
# Format plugins:
#   hodu-format-onnx         0.3.0  [load_model, save_model]  .onnx
#   hodu-format-safetensors  0.1.0  [load_tensor, save_tensor]  .safetensors
#   hodu-format-npy          0.1.0  [load_tensor, save_tensor]  .npy, .npz
#   hodu-format-gguf         0.1.0  [load_model]  .gguf
```

### hodu plugin freeze

현재 설치된 플러그인 상태를 잠금 파일로 저장:

```bash
# 현재 상태 저장
hodu plugin freeze > hodu-plugins.lock

# 잠금 파일 내용
cat hodu-plugins.lock
# {
#   "sdk_version": "0.3.0",
#   "created_at": "2024-01-15T10:30:00Z",
#   "platform": "aarch64-apple-darwin",
#   "plugins": [
#     {"name": "hodu-backend-cpu", "version": "0.3.0", "source": "crates.io"},
#     {"name": "hodu-backend-metal", "version": "0.3.0", "source": "crates.io"},
#     {"name": "hodu-format-onnx", "version": "0.3.0", "source": "crates.io"}
#   ]
# }

# 잠금 파일에서 복원
hodu plugin install --from hodu-plugins.lock

# CI/CD에서 재현 가능한 환경
hodu plugin install --from hodu-plugins.lock --exact
```

### hodu plugin bundle

추천 플러그인 번들 설치:

```bash
# 추천 플러그인 (기본)
hodu plugin install --recommended
# Installing: cpu, onnx, npy, safetensors...

# 용도별 번들
hodu plugin install --bundle basic       # cpu, onnx, npy
hodu plugin install --bundle apple       # metal, cpu, onnx, safetensors
hodu plugin install --bundle nvidia      # cuda, cpu, onnx, safetensors
hodu plugin install --bundle dev         # cpu, onnx, npy, safetensors (모든 포맷)

# 번들 목록 확인
hodu plugin bundle list
# Available bundles:
#   basic   - cpu, onnx, npy
#   apple   - metal, cpu, onnx, safetensors
#   nvidia  - cuda, cpu, onnx, safetensors
#   dev     - cpu, onnx, npy, safetensors
```

### 오프라인 설치

인터넷 연결 없이 플러그인 설치:

```bash
# 1. 온라인 환경에서 crate 파일 다운로드
hodu plugin download cpu -o ./plugins/
hodu plugin download onnx -o ./plugins/
# → plugins/hodu-backend-cpu-0.3.0.crate
# → plugins/hodu-format-onnx-0.3.0.crate

# 2. 오프라인 환경에서 설치
hodu plugin install --offline ./plugins/hodu-backend-cpu-0.3.0.crate
hodu plugin install --offline ./plugins/hodu-format-onnx-0.3.0.crate

# 캐시된 소스에서 재빌드 (버전 업그레이드 없이)
hodu plugin rebuild cpu
hodu plugin rebuild --all
```

---

## 디렉토리 구조

```
~/.hodu/
├── plugins/
│   ├── hodu-backend-cpu.dylib
│   ├── hodu-backend-metal.dylib
│   ├── hodu-format-onnx.dylib
│   ├── hodu-format-safetensors.dylib
│   └── hodu-format-npy.dylib
├── plugins.json              # 설치된 플러그인 정보
├── config.toml               # CLI 설정
└── cache/                    # 빌드 캐시
```

### plugins.json

```json
{
  "plugins": [
    {
      "name": "hodu-backend-cpu",
      "version": "0.3.0",
      "type": "backend",
      "capabilities": {
        "runner": true,
        "builder": true
      },
      "devices": ["cpu"],
      "targets": ["x86_64-unknown-linux-gnu", "x86_64-apple-darwin", "aarch64-apple-darwin"],
      "library": "hodu-backend-cpu.dylib",
      "source": "crates.io",
      "installed_at": "2024-01-15T10:30:00Z"
    },
    {
      "name": "hodu-backend-metal",
      "version": "0.3.0",
      "type": "backend",
      "capabilities": {
        "runner": true,
        "builder": true
      },
      "devices": ["metal"],
      "targets": ["metal"],
      "library": "hodu-backend-metal.dylib",
      "source": "crates.io",
      "installed_at": "2024-01-15T10:35:00Z"
    },
    {
      "name": "hodu-format-onnx",
      "version": "0.3.0",
      "type": "format",
      "capabilities": {
        "load_model": true,
        "save_model": true,
        "load_tensor": false,
        "save_tensor": false
      },
      "extensions": [".onnx"],
      "library": "hodu-format-onnx.dylib",
      "source": "crates.io",
      "installed_at": "2024-01-15T10:40:00Z"
    },
    {
      "name": "hodu-format-npy",
      "version": "0.1.0",
      "type": "format",
      "capabilities": {
        "load_model": false,
        "save_model": false,
        "load_tensor": true,
        "save_tensor": true
      },
      "extensions": [".npy", ".npz"],
      "library": "hodu-format-npy.dylib",
      "source": "crates.io",
      "installed_at": "2024-01-15T10:45:00Z"
    }
  ]
}
```

### config.toml

```toml
# ~/.hodu/config.toml

[defaults]
# 기본 실행 디바이스
device = "cpu"
# 출력 포맷
output_format = "pretty"

[backends]
# 같은 디바이스 지원하는 플러그인이 여러 개일 때 우선순위
# 먼저 나온 것이 우선
priority = ["metal", "cpu", "interp"]

[formats]
# 확장자별 플러그인 오버라이드 (선택사항)
# 기본적으로 확장자 매칭으로 자동 선택
# ".npy" = "my-custom-npy"

[build]
# 기본 최적화 레벨
opt_level = 2
# 빌드 캐시 활성화
cache_enabled = true
# 캐시 최대 크기 (MB)
cache_max_size = 1024

[plugin]
# 플러그인 자동 업데이트 체크
auto_update_check = true
# 업데이트 체크 주기 (일)
update_check_interval = 7

[runtime]
# CPU 스레드 수 (0 = 자동 감지)
num_threads = 0
# 최대 메모리 사용량 (MB, 0 = 무제한)
max_memory = 0
# GPU 메모리 비율 (Metal/CUDA)
gpu_memory_fraction = 0.9
# 메모리 풀 사용
use_memory_pool = true

[logging]
# 로그 레벨: error, warn, info, debug, trace
level = "info"
# 로그 파일 (선택사항)
# file = "~/.hodu/hodu.log"
# 로그 포맷: pretty, json, compact
format = "pretty"
# 모듈별 로그 레벨 (선택사항)
# [logging.modules]
# hodu_plugin = "debug"
# hodu_core = "warn"
```

**환경 변수:**
```bash
# 로그 레벨 오버라이드
HODU_LOG=debug hodu run model.onnx

# 모듈별 로그 레벨
HODU_LOG=warn,hodu_plugin=debug hodu run model.onnx

# 스레드 수 오버라이드
HODU_THREADS=4 hodu run model.onnx

# 메모리 제한
HODU_MAX_MEMORY=2048 hodu run model.onnx
```

---

## hodu_plugin_sdk 구조

```
hodu_plugin_sdk/
├── src/
│   ├── lib.rs
│   ├── backend.rs       # BackendPlugin trait, BackendCapabilities
│   ├── format.rs        # FormatPlugin trait, FormatCapabilities
│   ├── artifact.rs      # CompiledArtifact, OutputFormat
│   ├── target.rs        # BuildTarget, Device
│   ├── dispatch.rs      # DispatchManifest, TensorSpec, KernelDispatch
│   ├── metadata.rs      # build_metadata(), op_to_kernel_name()
│   ├── manager.rs       # PluginManager
│   ├── interp.rs        # InterpBackend (builtin)
│   └── reexports.rs     # hodu_core 타입 re-export (DType, Shape, Layout, Op, ...)
```

### DispatchManifest 공통화

플러그인들이 공유하는 컴파일 로직:

```rust
// hodu_plugin_sdk/src/dispatch.rs
pub struct DispatchManifest { ... }
pub struct TensorSpec { ... }
pub struct KernelDispatch { ... }

impl DispatchManifest {
    pub fn from_snapshot(snapshot: &Snapshot) -> Self { ... }
}

// kernel name 생성 (prefix 파라미터화)
pub fn op_to_kernel_name(op: &Op, dtype: DType, prefix: &str) -> String {
    format!("{}{}_{}", prefix, op, dtype)
}
```

---

## 플러그인 개발 예시

### Backend Plugin (hodu-backend-tpu)

```toml
# Cargo.toml
[package]
name = "hodu-backend-tpu"
version = "0.1.0"

[lib]
crate-type = ["cdylib"]

[dependencies]
hodu_plugin_sdk = "0.3"
```

```rust
// src/lib.rs
use hodu_plugin_sdk::*;

pub struct TpuBackend;

impl BackendPlugin for TpuBackend {
    fn name(&self) -> &str { "tpu" }
    fn version(&self) -> &str { env!("CARGO_PKG_VERSION") }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            runner: true,
            builder: false,  // Runner만 지원
        }
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::TPU]
    }

    fn run(&self, snapshot: &Snapshot, device: Device, inputs: &[(&str, TensorData)])
        -> HoduResult<HashMap<String, TensorData>>
    {
        // TPU 실행 로직
        todo!()
    }

    // Builder 미지원
    fn supported_targets(&self) -> Vec<BuildTarget> { vec![] }
    fn supported_formats(&self, _: &BuildTarget) -> Vec<OutputFormat> { vec![] }
    fn build(&self, _: &Snapshot, _: &BuildTarget, _: OutputFormat, _: &Path) -> HoduResult<()> {
        Err(HoduError::UnsupportedOperation("TPU backend does not support building".into()))
    }
}

// FFI entry points
#[no_mangle]
pub extern "C" fn hodu_backend_plugin_create() -> *mut BackendPluginHandle {
    BackendPluginHandle::from_boxed(Box::new(TpuBackend))
}

#[no_mangle]
pub unsafe extern "C" fn hodu_backend_plugin_destroy(ptr: *mut BackendPluginHandle) {
    if !ptr.is_null() {
        drop(BackendPluginHandle::into_boxed(ptr));
    }
}
```

### Format Plugin (hodu-format-gguf)

```rust
// src/lib.rs
use hodu_plugin_sdk::*;

pub struct GgufFormat;

impl FormatPlugin for GgufFormat {
    fn name(&self) -> &str { "gguf" }
    fn version(&self) -> &str { env!("CARGO_PKG_VERSION") }

    fn capabilities(&self) -> FormatCapabilities {
        FormatCapabilities {
            load_model: true,
            save_model: false,  // 읽기 전용
            load_tensor: false,
            save_tensor: false,
        }
    }

    fn supported_extensions(&self) -> Vec<&str> {
        vec!["gguf"]
    }

    fn load_model(&self, path: &Path) -> HoduResult<Snapshot> {
        // GGUF 파싱 로직
        todo!()
    }

    // 나머지는 미지원
    fn load_model_from_bytes(&self, _: &[u8]) -> HoduResult<Snapshot> { todo!() }
    fn save_model(&self, _: &Snapshot, _: &Path) -> HoduResult<()> {
        Err(HoduError::UnsupportedOperation("GGUF format is read-only".into()))
    }
    fn save_model_to_bytes(&self, _: &Snapshot) -> HoduResult<Vec<u8>> {
        Err(HoduError::UnsupportedOperation("GGUF format is read-only".into()))
    }
    fn load_tensor(&self, _: &Path) -> HoduResult<TensorData> {
        Err(HoduError::UnsupportedOperation("GGUF does not support tensor loading".into()))
    }
    fn save_tensor(&self, _: &TensorData, _: &Path) -> HoduResult<()> {
        Err(HoduError::UnsupportedOperation("GGUF does not support tensor saving".into()))
    }
}

// FFI entry points
#[no_mangle]
pub extern "C" fn hodu_format_plugin_create() -> *mut FormatPluginHandle {
    FormatPluginHandle::from_boxed(Box::new(GgufFormat))
}

#[no_mangle]
pub unsafe extern "C" fn hodu_format_plugin_destroy(ptr: *mut FormatPluginHandle) {
    if !ptr.is_null() {
        drop(FormatPluginHandle::into_boxed(ptr));
    }
}
```

---

## 버전 호환성 정책

### Semantic Versioning

```
hodu_plugin_sdk 버전: MAJOR.MINOR.PATCH

- MAJOR: Breaking API 변경
- MINOR: 하위 호환되는 기능 추가
- PATCH: 버그 수정
```

### 플러그인 호환성

```toml
# 플러그인 Cargo.toml
[dependencies]
hodu_plugin_sdk = "0.3"    # 0.3.x 모두 호환
```

- 소스 빌드 방식이라 ABI 문제 없음
- `hodu_plugin_sdk` major 버전 내에서 하위 호환 유지
- Major 버전 업 시 플러그인 재빌드 필요

### SDK 버전 검증 (FFI)

플러그인 로드 시점에 SDK 버전 호환성 검증:

```rust
// hodu_plugin_sdk에서 제공하는 매크로
#[macro_export]
macro_rules! export_plugin_metadata {
    () => {
        #[no_mangle]
        pub extern "C" fn hodu_plugin_sdk_version() -> *const std::ffi::c_char {
            concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const std::ffi::c_char
        }

        #[no_mangle]
        pub extern "C" fn hodu_plugin_sdk_major() -> u32 {
            env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap_or(0)
        }
    };
}

// 플러그인에서 사용
use hodu_plugin_sdk::export_plugin_metadata;
export_plugin_metadata!();
```

**CLI 로드 시 검증:**

```rust
fn load_plugin(path: &Path) -> HoduResult<Plugin> {
    let lib = unsafe { libloading::Library::new(path)? };

    // SDK 버전 체크
    let sdk_major: libloading::Symbol<fn() -> u32> =
        unsafe { lib.get(b"hodu_plugin_sdk_major")? };

    let plugin_major = sdk_major();
    let cli_major = env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>().unwrap();

    if plugin_major != cli_major {
        return Err(HoduError::IncompatiblePlugin {
            plugin_version: format!("{}.x", plugin_major),
            cli_version: format!("{}.x", cli_major),
            suggestion: "hodu plugin install <name> --force".to_string(),
        });
    }

    // 플러그인 로드 계속...
}
```

---

## 에러 처리 및 UX

### 플러그인 없을 때

```bash
$ hodu run model.onnx --device metal

Error: Backend 'metal' not found

To install:
  hodu plugin install metal

Available backends:
  interp (builtin)
  cpu

Available format plugins for .onnx:
  (none installed)

To install ONNX support:
  hodu plugin install onnx
```

### 플러그인 빌드 실패

`hodu plugin install` 중 cargo build 실패 시:

```bash
$ hodu plugin install cuda

Downloading hodu-backend-cuda 0.2.0...
Building...

Build failed!

error[E0463]: can't find crate `cuda_sys`
  --> src/lib.rs:1:1

Possible causes:
  - Missing system dependency: CUDA toolkit

To fix:
  1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
  2. Retry: hodu plugin install cuda

Build log: ~/.hodu/logs/hodu-backend-cuda-build.log
```

### 플러그인 로드 실패

```bash
$ hodu run model.onnx --device cpu

Error: Failed to load plugin 'hodu-backend-cpu'

Cause: Symbol not found: hodu_backend_plugin_create
  This usually means the plugin was built with an incompatible SDK version.

To fix:
  hodu plugin install cpu --force   # 강제 재빌드
```

### 플러그인 크래시 처리

플러그인 실행 중 panic 발생 시:

```bash
$ hodu run model.onnx --device metal

Error: Plugin 'hodu-backend-metal' crashed

  thread 'main' panicked at 'index out of bounds'
  note: run with `RUST_BACKTRACE=1` for a backtrace

This is likely a bug in the plugin. Please report to:
  https://github.com/prunee/hodu-plugins/issues

Workaround:
  - Try a different backend: hodu run model.onnx --device cpu
  - Downgrade plugin: hodu plugin install metal@0.2.0
```

**구현 참고:**
- FFI 경계에서 `catch_unwind` 사용하여 panic 캡처
- 플러그인 크래시가 CLI 전체를 죽이지 않도록 처리
- 상세 에러 정보를 로그 파일에 저장

```rust
// plugin loader에서
pub fn call_plugin_safely<T, F>(f: F) -> HoduResult<T>
where
    F: FnOnce() -> T + std::panic::UnwindSafe,
{
    std::panic::catch_unwind(f)
        .map_err(|e| HoduError::PluginCrash(format_panic_info(e)))
}
```

### hodu doctor 명령어

시스템 상태 및 플러그인 점검:

```bash
$ hodu doctor

System:
  ✓ Platform: aarch64-apple-darwin
  ✓ Rust: 1.75.0

Backends:
  ✓ interp (builtin)
  ✓ cpu 0.3.0
  ✗ metal (not installed)
  ✗ cuda (not installed, requires CUDA toolkit)

Formats:
  ✓ hdss, hdt, json (builtin)
  ✓ onnx 0.3.0
  ✗ npy (not installed)
  ✗ safetensors (not installed)

Plugin Health:
  ✓ hodu-backend-cpu: OK
  ✓ hodu-format-onnx: OK
  ! hodu-format-npy: SDK version mismatch (0.2.0 vs 0.3.0)
    → Run: hodu plugin install npy --force
```

---

## 시스템 의존성

### 플러그인별 의존성

| 플러그인 | 필수 | 선택 |
|----------|------|------|
| hodu-backend-cpu | clang | openblas |
| hodu-backend-metal | Xcode (macOS only) | - |
| hodu-backend-cuda | CUDA toolkit | cuDNN |
| hodu-format-onnx | - | - |
| hodu-format-npy | - | - |

### 설치 시 의존성 체크

```bash
$ hodu plugin install cuda

Checking dependencies...
✗ CUDA toolkit not found

Install CUDA toolkit first:
  https://developer.nvidia.com/cuda-downloads

Then retry:
  hodu plugin install cuda
```

### 플러그인 메타데이터

```toml
# hodu-backend-cpu/Cargo.toml
[package.metadata.hodu]
type = "backend"
system_deps = ["clang"]
system_deps_optional = ["openblas"]
```

---

## UX 및 안정성

### 프로그레스 표시

긴 작업에 대한 진행 상황 표시:

```bash
$ hodu plugin install cuda

Downloading hodu-backend-cuda 0.2.0...
[████████████████████░░░░░░░░░░] 67% (12.3 MB / 18.4 MB)

Building...
[████████████████████████░░░░░░] 80% Compiling cuda_kernels

$ hodu build large_model.onnx -o model.dylib

Loading model...
[████████████████████████████████] 100%

Optimizing...
[██████████░░░░░░░░░░░░░░░░░░░░] 33% Operator fusion

Compiling...
[██████░░░░░░░░░░░░░░░░░░░░░░░░] 20% (Layer 45/230)
```

**구현 참고:**
- `indicatif` crate 사용
- stderr로 출력 (stdout은 결과용)
- `--quiet` 옵션으로 비활성화
- CI 환경 자동 감지 (TTY 없으면 간소화)

### 인터럽트 처리 (Ctrl+C)

```bash
$ hodu plugin install cuda
Downloading...
^C
Interrupted. Cleaning up...
  - Removed partial download
Done.

$ hodu build model.onnx -o model.dylib
Compiling...
^C
Interrupted. Cleaning up...
  - Removed temporary files
  - Released GPU memory
Done.
```

**구현 참고:**
```rust
use ctrlc;

ctrlc::set_handler(move || {
    eprintln!("\nInterrupted. Cleaning up...");
    CLEANUP_FLAG.store(true, Ordering::SeqCst);
})?;

// 긴 작업에서 주기적으로 체크
if CLEANUP_FLAG.load(Ordering::SeqCst) {
    cleanup_resources()?;
    std::process::exit(130);  // 128 + SIGINT(2)
}
```

### 플러그인 검증

crates.io 공식 플러그인과 서드파티 구분:

```bash
$ hodu plugin list

Backend plugins:
  cpu       0.3.0  ✓ official   crates.io
  metal     0.3.0  ✓ official   crates.io
  my-backend 0.1.0              github:user/repo

Format plugins:
  onnx      0.3.0  ✓ official   crates.io
  custom    0.1.0              local
```

**검증되지 않은 플러그인 설치 시:**
```bash
$ hodu plugin install --git https://github.com/unknown/plugin

⚠ Warning: This plugin is from an unverified source.
  - Source: github:unknown/plugin
  - Not published on crates.io
  - No security audit

Install anyway? [y/N] y

Installing...
```

**신뢰 옵션:**
```bash
# 경고 없이 설치
hodu plugin install --git https://github.com/user/plugin --trust

# config.toml에서 신뢰할 소스 설정
[plugin.trusted_sources]
github = ["prunee/*", "hodu-community/*"]
```

### hodu self-update

CLI 자체 업데이트:

```bash
$ hodu self-update

Current version: 0.3.0
Checking for updates...

New version available: 0.3.1

Changelog:
  - Fixed plugin loading on Windows
  - Added --profile option to hodu run
  - Performance improvements

Update now? [Y/n] y

Downloading hodu 0.3.1...
Installing...

Updated successfully!
Please restart your shell or run: hash -r

$ hodu self-update --check
Update available: 0.3.0 → 0.3.1
Run 'hodu self-update' to install.

$ hodu self-update --force
# 확인 없이 즉시 업데이트
```

**config.toml 설정:**
```toml
[updates]
# 자동 업데이트 체크 (시작 시)
auto_check = true
# 체크 주기 (일)
check_interval = 7
# pre-release 포함
include_prerelease = false
```

---

## 네트워크 설정

### 프록시 지원

기업 환경 등에서 프록시를 통해 인터넷 접근이 필요한 경우:

**config.toml:**
```toml
[network]
# HTTP/HTTPS 프록시
proxy = "http://proxy.company.com:8080"

# 프록시 제외 대상
no_proxy = ["localhost", "127.0.0.1", "*.internal.company.com"]

# 타임아웃 (초)
timeout = 30

# SSL 인증서 검증 비활성화 (자체 서명 인증서 사용 시)
# 보안 위험! 꼭 필요한 경우만 사용
# insecure = true
```

**환경 변수 (config.toml보다 우선):**
```bash
# 표준 환경 변수 지원 (대부분의 도구와 호환)
HTTPS_PROXY=http://proxy:8080 hodu plugin install cpu
HTTP_PROXY=http://proxy:8080 hodu plugin install cpu
NO_PROXY=localhost,*.internal hodu plugin install cpu

# 인증이 필요한 프록시
HTTPS_PROXY=http://user:password@proxy:8080 hodu plugin install cpu
```

**영향받는 명령어:**
- `hodu plugin install <NAME>` (crates.io)
- `hodu plugin install --git <URL>` (GitHub)
- `hodu plugin search`
- `hodu plugin update`
- `hodu self-update`

**구현 참고:**
```rust
// reqwest가 환경 변수 자동 지원
let client = reqwest::Client::builder()
    .proxy(reqwest::Proxy::all(&config.network.proxy)?)
    .timeout(Duration::from_secs(config.network.timeout))
    .danger_accept_invalid_certs(config.network.insecure)
    .build()?;
```

---

## 플러그인 개발 워크플로우

### 로컬 개발

```bash
# 1. 플러그인 프로젝트 생성
cargo new hodu-backend-mydevice --lib
cd hodu-backend-mydevice

# 2. Cargo.toml 설정
cat >> Cargo.toml << 'EOF'
[lib]
crate-type = ["cdylib"]

[dependencies]
hodu_plugin_sdk = "0.3"

[package.metadata.hodu]
type = "backend"
EOF

# 3. 구현...

# 4. 로컬 빌드 & 설치
hodu plugin install --path . --debug

# 5. 테스트
hodu run test_model.onnx --device mydevice
```

### 디버그 모드

```bash
# 디버그 빌드 (최적화 없음, 디버그 심볼 포함)
hodu plugin install --path ./my-plugin --debug

# 설치된 플러그인 확인
hodu plugin list
# ...
# mydevice  0.1.0 (dev)  [Runner]  devices: mydevice  local:/path/to/my-plugin

# 로그 레벨 설정
HODU_LOG=debug hodu run model.onnx --device mydevice
```

### 플러그인 테스트

```bash
# 플러그인 기본 검증 (로드, 메타데이터, 기본 호출)
hodu plugin test ./my-plugin

# 출력:
# Testing hodu-backend-mydevice...
#   ✓ Build succeeded
#   ✓ Plugin loads correctly
#   ✓ SDK version compatible (0.3.x)
#   ✓ name() returns "mydevice"
#   ✓ capabilities() = {runner: true, builder: false}
#   ✓ supported_devices() = [MyDevice]
#
# Basic tests passed!

# 모델로 통합 테스트
hodu plugin test ./my-plugin --model test_model.onnx --input x=test_input.npy
# ...
#   ✓ run() completed without error
#   ✓ Output shapes match expected
#
# All tests passed!
```

### 예제 플러그인 템플릿

```bash
# 템플릿에서 새 플러그인 생성
hodu plugin new my-custom-backend --type backend
hodu plugin new my-custom-format --type format

# 생성되는 구조
my-custom-backend/
├── Cargo.toml
├── src/
│   └── lib.rs          # BackendPlugin 구현 템플릿
├── tests/
│   └── integration.rs  # 통합 테스트
└── README.md
```

### CI/CD 권장 구성

```yaml
# .github/workflows/test.yml
name: Test Plugin

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Install hodu CLI
        run: cargo install hodu

      - name: Build plugin
        run: cargo build --release

      - name: Test plugin
        run: hodu plugin test .

      - name: Integration test
        run: |
          hodu plugin install --path .
          hodu run tests/model.onnx --device mydevice -i x=tests/input.npy
```

### 릴리즈 체크리스트

```bash
# 1. 버전 업데이트
# Cargo.toml: version = "0.2.0"

# 2. 로컬 테스트
cargo test
hodu plugin test .

# 3. 문서 확인
cargo doc --open

# 4. crates.io 배포
cargo publish

# 5. 설치 테스트
hodu plugin install my-plugin@0.2.0
hodu doctor
```

---

## 구현 계획

### Phase 1: hodu_plugin_sdk 생성

- [ ] `hodu_plugin` → `hodu_plugin_sdk` rename/분리
- [ ] `CompilerPlugin` + `RuntimePlugin` → `BackendPlugin` 통합
- [ ] `BackendCapabilities` 추가 (runner/builder 지원 여부)
- [ ] `FormatPlugin` trait 추가 (bytes 버전 포함)
- [ ] `FormatCapabilities` 추가
- [ ] `DispatchManifest`를 hodu_plugin_sdk로 이동
- [ ] `build_metadata()`, `op_to_kernel_name()` 이동
- [ ] FFI handle 타입 정리 (`BackendPluginHandle`, `FormatPluginHandle`)
- [ ] `export_plugin_metadata!` 매크로 (SDK 버전 검증용)
- [ ] hodu_core 타입 re-export 정리

### Phase 2: CLI 구조 개선

- [ ] `src/cli.rs` → `cli/` crate로 분리
- [ ] `#[path = "cli/..."]` 비표준 문법 제거
- [ ] `~/.hodu/config.toml` 지원 (logging, runtime 포함)
- [ ] `hodu doctor` 명령어 추가
- [ ] `hodu version` 명령어 추가
- [ ] `hodu completions` 명령어 추가 (bash/zsh/fish/powershell)
- [ ] `hodu self-update` 명령어 추가
- [ ] 플러그인 없을 때 친절한 에러 메시지
- [ ] 플러그인 크래시 처리 (`catch_unwind`)
- [ ] 인터럽트 처리 (`ctrlc`, Ctrl+C 정리)
- [ ] 프로그레스 표시 (`indicatif`)
- [ ] 환경 변수 지원 (HODU_LOG, HODU_THREADS, HODU_MAX_MEMORY)

### Phase 3: 플러그인 repo 분리

- [ ] `hodu-plugins/` 별도 repository 생성
- [ ] `hodu-compiler-cpu` + `hodu-runtime-cpu` → `hodu-backend-cpu`
- [ ] `hodu-compiler-metal` + `hodu-runtime-metal` → `hodu-backend-metal`
- [ ] `hodu_onnx` → `hodu-format-onnx` (플러그인화)
- [ ] 중복 코드 제거 (dispatch.rs → hodu_plugin_sdk)

### Phase 4: CLI plugin 명령어

- [ ] `hodu plugin list`
- [ ] `hodu plugin install --path <PATH>` (로컬)
- [ ] `hodu plugin install --path <PATH> --debug` (디버그 빌드)
- [ ] `hodu plugin remove <NAME>`
- [ ] `hodu plugin search <QUERY>` (crates.io 검색)
- [ ] `hodu plugin test <PATH>` (플러그인 검증)
- [ ] `hodu plugin new <NAME> --type <TYPE>` (템플릿 생성)
- [ ] `hodu plugin freeze` (잠금 파일 생성)
- [ ] `hodu plugin install --from <LOCK>` (잠금 파일에서 복원)
- [ ] `hodu plugin install --bundle <NAME>` (번들 설치)
- [ ] `hodu plugin download <NAME>` (오프라인용 다운로드)
- [ ] `hodu plugin install --offline <FILE>` (오프라인 설치)
- [ ] `hodu plugin rebuild <NAME>` (재빌드)
- [ ] `~/.hodu/plugins/` 디렉토리 구조
- [ ] `plugins.json` 관리
- [ ] SDK 버전 호환성 검증

### Phase 5: 원격 설치

- [ ] `hodu plugin install <NAME>` (crates.io)
- [ ] `hodu plugin install --git <URL>` (GitHub)
- [ ] `hodu plugin update`
- [ ] `hodu plugin install --force` (강제 재빌드)
- [ ] `hodu plugin install --trust` (검증 생략)
- [ ] 시스템 의존성 체크 (clang, Xcode, CUDA 등)
- [ ] 빌드 실패 시 친절한 에러 메시지
- [ ] 비공식 플러그인 경고 표시
- [ ] 프록시 지원 (`[network]` config, 환경 변수)

### Phase 6: hodu run/build 개선

- [ ] FormatPlugin 자동 선택 (확장자 기반)
- [ ] BackendPlugin 자동 선택 (device + priority 기반)
- [ ] `--backend <NAME>` 명시적 선택 옵션
- [ ] `--dry-run` 플러그인 선택 디버깅
- [ ] `--benchmark` 벤치마크 모드
- [ ] `--profile` 프로파일링 모드
- [ ] `hodu build` 명령어 구현
- [ ] `-O, --opt-level` 최적화 레벨 (0-3)
- [ ] 그래프 최적화 (상수 폴딩, 연산자 융합, DCE)
- [ ] `--target` 크로스 컴파일 지원
- [ ] `--standalone` 독립 실행파일 생성
- [ ] `hodu inspect` 명령어 (모델 정보 확인)
- [ ] `hodu convert` 명령어 (포맷 변환)

### Phase 7: 캐싱

- [ ] `~/.hodu/cache/` 컴파일 캐시
- [ ] 캐시 키 (model hash + device + backend + opt_level)
- [ ] `hodu cache status`
- [ ] `hodu cache clean`
- [ ] config.toml 캐시 설정

### Phase 8: 추가 플러그인

- [ ] `hodu-format-safetensors`
- [ ] `hodu-format-npy`
- [ ] `hodu-backend-llvm` (크로스 컴파일용)

### Phase 9: 문서화

- [ ] 플러그인 개발 가이드
- [ ] hodu_plugin_sdk API 문서
- [ ] 예제 플러그인 템플릿
- [ ] CI/CD 설정 가이드

---

## 플러그인 명명 규칙

| 패턴 | 예시 |
|------|------|
| `hodu-backend-{name}` | hodu-backend-cpu, hodu-backend-metal, hodu-backend-cuda |
| `hodu-format-{name}` | hodu-format-onnx, hodu-format-npy, hodu-format-safetensors |

CLI에서 약칭 지원:
- `cpu` → `hodu-backend-cpu`
- `metal` → `hodu-backend-metal`
- `onnx` → `hodu-format-onnx`
- `npy` → `hodu-format-npy`
