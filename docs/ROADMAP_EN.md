# Hodu Project Roadmap

## Project Overview

**Hodu** is a Rust-based ML compiler framework that compiles ONNX models and HoduScript into optimized native binaries, while providing a Rust frontend API for training and inference.

## Core Objectives

### 1. Multi-Format Compiler

```bash
# Compile ONNX model
hodu --build model.onnx -o model.so

# Compile HoduScript
hodu --build model.hodu -o model.so
```

**Features**
- Compile ONNX models into optimized native libraries (.so, .dylib, .dll)
- Compile HoduScript (proprietary DSL) into native binaries
- Leverage various backends (XLA, IREE, MLX, etc.) for optimization
- Generate platform-optimized binaries

### 2. HoduScript

- Domain-specific language for ML model definition
- Intuitive and expressive syntax
- More flexible model definition and customization than ONNX
- Tight integration with Rust's native type system

### 3. Rust Frontend API

Unified interface for training and inference

**Features**
- Ergonomic API for easy use of compiled models
- Zero-cost abstractions with no performance overhead
- Type safety and memory safety guarantees
- **Can be used as a general ML library like PyTorch/TensorFlow**

### 4. Packaging System

- Bundle models, weights, and metadata into a single package
- Cross-platform deployment support
- Lightweight packages optimized for edge devices

## Project Structure

### hodu (Core Library & Compiler)

- ML compiler core
- ONNX parser and HoduScript compiler
- Multi-backend integration and abstraction
- Standalone CLI binary
- Rust frontend API library

### hodugaki (Distributed Computing Binary)

- Distributed computing system based on hodu
- Edge-Cloud collaboration
- Distributed training and inference

## Architecture Strategy

### Multi-backend Dispatcher

Using Rust as an abstraction layer to integrate various computing backends:

**Currently Supported**
- HODU
- XLA (separate repo: [hodu-rs/hodu_xla](https://github.com/hodu-rs/hodu_xla))

**Planned**
- IREE
- MLX (Apple Silicon optimization)

### Compilation Pipeline

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

## Core Values

### Compile-time Optimization

- Maximum performance with AOT (Ahead-of-Time) compilation
- Backend-specific optimization strategies
- Platform-specific optimizations

### Developer Experience

- Intuitive CLI interface
- Type-safe Rust API
- Clear error messages
- Comprehensive documentation

### Edge-Friendly

- Lightweight runtime
- Minimal dependencies
- Optimized for resource-constrained environments

### Cross-Platform

- Linux, macOS, Windows support
- ARM, x86 architecture compatibility
- Embedded environment support
- Various accelerator support (CPU, GPU, NPU)

### Production-Ready

- Rust's memory safety
- Zero-cost abstractions
- Predictable performance
