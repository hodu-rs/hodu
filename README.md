# Hodu, a user-friendly ML framework built in Rust.

[![Crates.io](https://img.shields.io/crates/v/hodu.svg)](https://crates.io/crates/hodu)
[![Doc.rs](https://docs.rs/hodu/badge.svg)](https://docs.rs/hodu)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/hodu-rs/hodu#license)

> **Hodu** (호두) is a Korean word meaning "walnut".

## About Hodu

<img align="right" src="assets/hodu/hodu_avatar.png" alt="Hodu Avatar" width="200"/>

**Hodu** is a machine learning library built with user convenience at its core, designed for both rapid prototyping and seamless production deployment—including embedded environments.

### Core Differentiators

Built on **Rust's foundation of memory safety and zero-cost abstractions**, Hodu offers unique advantages:

- **Hybrid Execution Model**: Seamlessly switch between dynamic execution for rapid prototyping and static computation graphs for optimized production deployment
- **Memory Safety by Design**: Leverage Rust's ownership system to eliminate common ML deployment issues like memory leaks and data races
- **Embedded-First Architecture**: Full `no_std` support enables ML inference on microcontrollers and resource-constrained devices
- **Zero-Cost Abstractions**: High-level APIs that compile down to efficient machine code without runtime overhead

### Dual Backend Architecture

- **HODU Backend**: Pure Rust implementation with `no_std` support for embedded environments
  - CPU operations with SIMD optimization
  - CUDA GPU acceleration (with `cuda` feature)
  - Metal GPU support for macOS (with `metal` feature)
- **XLA Backend**: JIT compilation via [OpenXLA/PJRT](https://github.com/openxla/xla) (requires `std`)
  - Advanced graph-level optimizations
  - CPU and CUDA device support
  - Production-grade performance for static computation graphs

> [!WARNING]
>
> This is a personal learning and development project. As such:
> - The framework is under active development
> - Features may be experimental or incomplete
> - Functionality is not guaranteed for production use
>
> It is recommended to use the latest version.

> [!CAUTION]
>
> **Current Development Status**:
> - CUDA GPU support is not yet fully implemented and is under active development
> - SIMD optimizations are not yet implemented and are under active development

## Get started

Here are some examples that demonstrate matrix multiplication using both dynamic execution and static computation graphs.

### Dynamic Execution

This example shows direct tensor operations that are executed immediately:

```rust
use hodu::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set the runtime device (CPU, CUDA, Metal)
    set_runtime_device(Device::CPU);

    // Create random tensors
    let a = Tensor::randn(&[2, 3], 0f32, 1.)?;
    let b = Tensor::randn(&[3, 4], 0f32, 1.)?;

    // Matrix multiplication
    let c = a.matmul(&b)?;

    println!("{}", c);
    println!("{:?}", c);

    Ok(())
}
```

With the `cuda` feature enabled, you can use CUDA in dynamic execution with the following setting:

```diff
- set_runtime_device(Device::CPU);
+ set_runtime_device(Device::CUDA(0));
```

### Static Computation Graphs

For more complex workflows or when you need reusable computation graphs, you can use the Builder pattern:

```rust
use hodu::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new computation graph
    let builder = Builder::new("matmul_example".to_string());
    builder.start()?;

    // Define input placeholders
    let a = Tensor::input("a", &[2, 3])?;
    let b = Tensor::input("b", &[3, 4])?;

    // Define computation
    let c = a.matmul(&b)?;

    // Mark output
    builder.add_output("result", c)?;
    builder.end()?;

    // Build and execute script
    let mut script = builder.build()?;

    // Provide actual data
    let a_data = Tensor::randn(&[2, 3], 0f32, 1.)?;
    let b_data = Tensor::randn(&[3, 4], 0f32, 1.)?;
    script.add_input("a", a_data);
    script.add_input("b", b_data);

    // Execute and get results
    let output = script.run()?;
    println!("{}", output["result"]);
    println!("{:?}", output["result"]);

    Ok(())
}
```

With the `cuda` feature enabled, you can use CUDA in static computation graphs with the following setting:

```diff
let mut script = builder.build()?;
+ script.set_device(Device::CUDA(0));
```

With the `xla` feature enabled, you can use XLA in static computation graphs with the following setting:

```diff
let mut script = builder.build()?;
+ script.set_backend(Backend::XLA);
```

## Features

### Default Features
| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `std` | Standard library support | - |
| `serde` | Serialization/deserialization support | - |
| `rayon` | Parallel processing support | `std` |

### Optional Features
| Feature | Description | Dependencies | Required Features |
|---------|-------------|--------------|-------------------|
| `cuda` | NVIDIA CUDA GPU support | CUDA toolkit | - |
| `metal` | Apple Metal GPU support | Metal framework | `std` |
| `xla` | Google XLA compiler backend | XLA libraries | `std` |

#### XLA Feature Requirements

Building with the `xla` feature requires:
- **LLVM** and **Clang** installed on your system
- **RAM**: 8GB+ free memory
- **Disk Space**: 20GB+ free storage

## Supported Platforms

### Standard Environments

| Target Triple | Backend | Device | Features | Status |
|--------------|---------|--------|----------|--------|
| x86_64-unknown-linux-gnu | HODU | CPU | `std` | ✅ Stable |
| | HODU | CUDA | `std`, `cuda` | 🚧 In Development |
| | XLA | CPU | `std`, `xla` | ✅ Stable |
| | XLA | CUDA | `std`, `xla`, `cuda` | 🚧 In Development |
| aarch64-unknown-linux-gnu | HODU | CPU | `std` | ✅ Stable |
| | XLA | CPU | `std`, `xla` | ✅ Stable |
| x86_64-apple-darwin | HODU | CPU | `std` | 🧪 Experimental |
| | XLA | CPU | `std`, `xla` | 🚧 In Development |
| aarch64-apple-darwin | HODU | CPU | `std` | ✅ Stable |
| | HODU | Metal | `std`, `metal` | 🧪 Experimental |
| | XLA | CPU | `std`, `xla` | ✅ Stable |
| x86_64-pc-windows-msvc | HODU | CPU | `std` | ✅ Stable |
| | HODU | CUDA | `std`, `cuda` | 🚧 In Development |
| | XLA | CPU | `std`, `xla` | 🚧 In Development |
| | XLA | CUDA | `std`, `xla`, `cuda` | 🚧 In Development |

### Embedded Environments

🧪 **Experimental**: Embedded platforms (ARM Cortex-M, RISC-V, Embedded Linux) are supported via `no_std` feature but are experimental and not extensively tested in production environments.

## Docs

[CHANGELOG](CHANGELOG.md) - Project changelog and version history

[TODOS](TODOS.md) - Planned features and improvements

[CONTRIBUTING](CONTRIBUTING.md) - Contribution guide

Guide
- [Tensor Creation Guide (Korean)](docs/tensor_creation_ko.md) - 텐서 생성 가이드
- [Tensor Creation Guide (English)](docs/tensor_creation_en.md) - Tensor creation guide
- [Tensor Data Type Guide](docs/tensor_dtype.md) - Tensor data type guide
- [Tensor Operations Guide](docs/tensor_ops.md) - Tensor operations guide (only English)
- [Neural Network Modules Guide (Korean)](docs/tensor_nn_modules_ko.md) - 신경망 모듈 가이드
- [Neural Network Modules Guide (English)](docs/tensor_nn_modules_en.md) - Neural network modules guide
- [Tensor Utils Guide (Korean)](docs/tensor_utils_ko.md) - 텐서 유틸리티 가이드 (DataLoader, Dataset, Sampler)
- [Tensor Utils Guide (English)](docs/tensor_utils_en.md) - Tensor utilities guide (DataLoader, Dataset, Sampler)
- [Builder/Script Guide (Korean)](docs/builder_and_script_ko.md) - Builder/Script 가이드
- [Builder/Script Guide (English)](docs/builder_and_script_en.md) - Builder/Script guide
- [Gradient Tape Management Guide (Korean)](docs/tensor_gradient_tape_ko.md) - 그래디언트 테이프 관리 가이드
- [Gradient Tape Management Guide (English)](docs/tensor_gradient_tape_en.md) - Gradient tape management guide

## Related Projects

Here are some other Rust ML frameworks you might find interesting:

- [maidenx](https://github.com/miniex/maidenx) - The predecessor project to Hodu
- [cetana](https://github.com/SkuldNorniern/cetana) - An advanced machine learning library empowering developers to build intelligent applications with ease.

## Inspired by

Hodu draws inspiration from the following amazing projects:

- [maidenx](https://github.com/miniex/maidenx) - The predecessor project to Hodu
- [candle](https://github.com/huggingface/candle) - Minimalist ML framework for Rust
- [GoMlx](https://github.com/gomlx/gomlx) - An Accelerated Machine Learning Framework For Go

## Credits

Hodu Character Design: Created by <a href="https://github.com/SkuldNorniern">Eira</a>
