# hodu 프로젝트 로드맵

## 프로젝트 개요

**hodu**는 Rust 기반의 ML 컴파일러 프레임워크로, ONNX 모델과 HoduScript를 최적화된 네이티브 바이너리로 컴파일하고, 학습과 추론을 위한 Rust 프론트엔드 API를 제공합니다.

## 핵심 목표

### 1. 멀티 포맷 컴파일러

```bash
# ONNX 모델 컴파일
hodu --build model.onnx -o model.so

# HoduScript 컴파일
hodu --build model.hodu -o model.so
```

**기능**
- ONNX 모델을 최적화된 네이티브 라이브러리(.so, .dylib, .dll)로 컴파일
- HoduScript (자체 DSL)를 네이티브 바이너리로 컴파일
- 다양한 백엔드(XLA, IREE, MLX 등)를 활용한 최적화
- 플랫폼별 최적화된 바이너리 출력

### 2. HoduScript

- ML 모델 정의를 위한 자체 도메인 특화 언어
- 직관적이고 표현력 있는 문법
- ONNX보다 유연한 모델 정의 및 커스터마이징
- Rust 네이티브 타입 시스템과의 긴밀한 통합

### 3. Rust 프론트엔드 API

학습(Training)과 추론(Inference)을 위한 통합 인터페이스 제공

**특징**
- 컴파일된 모델을 쉽게 사용할 수 있는 ergonomic API
- Zero-cost abstractions로 성능 손실 없는 추상화
- 타입 안전성과 메모리 안전성 보장
- **PyTorch/TensorFlow처럼 일반적인 ML 라이브러리로도 사용 가능**

### 4. 패키징 시스템

- 모델, 가중치, 메타데이터를 하나의 패키지로 번들링
- 크로스 플랫폼 배포 지원
- Edge 디바이스 친화적인 경량 패키지

## 프로젝트 구조

### hodu (Core Library & Compiler)

- ML 컴파일러 코어
- ONNX 파서 및 HoduScript 컴파일러
- 다양한 백엔드 통합 및 추상화
- 독립 실행 가능한 CLI 바이너리
- Rust 프론트엔드 API 라이브러리

### hodugaki (Distributed Computing Binary)

- hodu를 기반으로 한 분산 컴퓨팅 시스템
- Edge-Cloud 협업
- 분산 학습 및 추론

## 아키텍처 전략

### Multi-backend Dispatcher

Rust를 추상화 레이어로 사용하여 다양한 컴퓨팅 백엔드 통합:

**현재 지원**
- HODU
- XLA (별도 레포: [hodu-rs/hodu_xla](https://github.com/hodu-rs/hodu_xla))

**추가 예정**
- IREE
- MLX (Apple Silicon 최적화)

### 컴파일 파이프라인

```
Input (ONNX/HoduScript)
    ↓
Frontend Parser
    ↓
IR (Intermediate Representation)
    ↓
Optimizer
    ↓
Backend Compiler (HODU/XLA/IREE/MLX)
    ↓
Native Binary (.so/.dylib/.dll)
```

## 핵심 가치

### Compile-time Optimization

- AOT(Ahead-of-Time) 컴파일로 최대 성능
- 백엔드별 최적화 전략 적용
- 플랫폼 특화 최적화

### Developer Experience

- 직관적인 CLI 인터페이스
- 타입 안전한 Rust API
- 명확한 에러 메시지
- 풍부한 문서화

### Edge-Friendly

- 경량 런타임
- 최소한의 의존성
- 리소스 제약 환경 최적화

### Cross-Platform

- Linux, macOS, Windows 지원
- ARM, x86 아키텍처 호환
- 임베디드 환경 지원
- 다양한 가속기 지원 (CPU, GPU, NPU)

### Production-Ready

- Rust의 메모리 안전성
- Zero-cost abstractions
- 예측 가능한 성능
