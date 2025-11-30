# Hodu CLI & Plugin System 0.3.0-1 Progress

## Phase 1: hodu_plugin_sdk 생성

- [x] `hodu_plugin` → `hodu_plugin_sdk` rename/분리
- [x] `CompilerPlugin` + `RuntimePlugin` → `BackendPlugin` 통합
- [x] `BackendCapabilities` 추가 (runner/builder 지원 여부)
- [x] `FormatPlugin` trait 추가 (bytes 버전 포함)
- [x] `FormatCapabilities` 추가
- [ ] `DispatchManifest`를 hodu_plugin_sdk로 이동
- [ ] `build_metadata()`, `op_to_kernel_name()` 이동
- [x] FFI handle 타입 정리 (`BackendPluginHandle`, `FormatPluginHandle`)
- [x] `export_backend_plugin!`, `export_format_plugin!` 매크로
- [x] hodu_core 타입 re-export 정리

## Phase 2: CLI 구조 개선

- [x] `src/cli/` 모듈 구조 생성
- [x] `Cargo.toml`에 `[lib]` + `[[bin]]` 설정
- [x] `src/main.rs` CLI 진입점
- [x] `src/cli/mod.rs` - Cli, Commands 정의
- [x] `src/cli/commands/run.rs` - hodu run
- [x] `src/cli/commands/build.rs` - hodu build
- [x] `src/cli/commands/inspect.rs` - hodu inspect
- [x] `src/cli/commands/plugin.rs` - hodu plugin
- [x] `src/cli/commands/version.rs` - hodu version
- [ ] `~/.hodu/config.toml` 지원 (logging, runtime 포함)
- [ ] `hodu doctor` 명령어 추가
- [ ] `hodu completions` 명령어 추가 (bash/zsh/fish/powershell)
- [ ] `hodu self-update` 명령어 추가
- [ ] 플러그인 없을 때 친절한 에러 메시지
- [ ] 플러그인 크래시 처리 (`catch_unwind`)
- [ ] 인터럽트 처리 (`ctrlc`, Ctrl+C 정리)
- [ ] 프로그레스 표시 (`indicatif`)
- [ ] 환경 변수 지원 (HODU_LOG, HODU_THREADS, HODU_MAX_MEMORY)

## Phase 3: 플러그인 repo 분리

- [x] `hodu-plugins/` 별도 repository 생성
- [x] `hodu-backend-interp` 플러그인 구현
- [ ] `hodu-compiler-cpu` + `hodu-runtime-cpu` → `hodu-backend-cpu`
- [ ] `hodu-compiler-metal` + `hodu-runtime-metal` → `hodu-backend-metal`
- [ ] `hodu_onnx` → `hodu-format-onnx` (플러그인화)
- [ ] 중복 코드 제거 (dispatch.rs → hodu_plugin_sdk)

## Phase 4: CLI plugin 명령어

- [ ] `~/.hodu/plugins/` 디렉토리 구조
- [ ] `plugins.json` 스키마 정의
- [ ] 플러그인 로더 구현 (libloading)
- [ ] `hodu plugin list` 구현
- [ ] `hodu plugin install --path <PATH>` (로컬)
- [ ] `hodu plugin install --path <PATH> --debug` (디버그 빌드)
- [ ] `hodu plugin remove <NAME>`
- [ ] SDK 버전 호환성 검증
- [ ] `hodu plugin search <QUERY>` (crates.io 검색)
- [ ] `hodu plugin test <PATH>` (플러그인 검증)
- [ ] `hodu plugin new <NAME> --type <TYPE>` (템플릿 생성)
- [ ] `hodu plugin freeze` (잠금 파일 생성)
- [ ] `hodu plugin install --from <LOCK>` (잠금 파일에서 복원)
- [ ] `hodu plugin install --bundle <NAME>` (번들 설치)
- [ ] `hodu plugin download <NAME>` (오프라인용 다운로드)
- [ ] `hodu plugin install --offline <FILE>` (오프라인 설치)
- [ ] `hodu plugin rebuild <NAME>` (재빌드)

## Phase 5: 원격 설치

- [ ] `hodu plugin install <NAME>` (crates.io)
- [ ] `hodu plugin install --git <URL>` (GitHub)
- [ ] `hodu plugin update`
- [ ] `hodu plugin install --force` (강제 재빌드)
- [ ] `hodu plugin install --trust` (검증 생략)
- [ ] 시스템 의존성 체크 (clang, Xcode, CUDA 등)
- [ ] 빌드 실패 시 친절한 에러 메시지
- [ ] 비공식 플러그인 경고 표시
- [ ] 프록시 지원 (`[network]` config, 환경 변수)

## Phase 6: hodu run/build 개선

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

## Phase 7: 캐싱

- [ ] `~/.hodu/cache/` 컴파일 캐시
- [ ] 캐시 키 (model hash + device + backend + opt_level)
- [ ] `hodu cache status`
- [ ] `hodu cache clean`
- [ ] config.toml 캐시 설정

## Phase 8: 추가 플러그인

- [ ] `hodu-format-safetensors`
- [ ] `hodu-format-npy`
- [ ] `hodu-backend-llvm` (크로스 컴파일용)

## Phase 9: 문서화

- [ ] 플러그인 개발 가이드
- [ ] hodu_plugin_sdk API 문서
- [ ] 예제 플러그인 템플릿
- [ ] CI/CD 설정 가이드
