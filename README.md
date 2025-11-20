# Hodu, a user-friendly ML framework built in Rust.

[![hodu - Crates.io](https://img.shields.io/crates/v/hodu.svg)](https://crates.io/crates/hodu)
[![hodu - Doc.rs](https://docs.rs/hodu/badge.svg)](https://docs.rs/hodu)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/hodu-rs/hodu#license)

> **Hodu** (호두) is a Korean word meaning "walnut".

## About Hodu

<img align="right" src="assets/hodu/hodu_avatar.png" alt="Hodu Avatar" width="200"/>

**Hodu** is a machine learning framework built in Rust that provides both a library for embedding ML in Rust applications and a standalone compiler & runtime for executing ML.

### Key Features

- **Dual Interface**: Use as a Rust library or standalone CLI compiler & runtime
- **Hybrid Execution**: Dynamic execution for prototyping, static graphs for production
- **Embedded Support**: Full `no_std` support for microcontrollers
- **Multiple Backends**: CPU, CUDA, Metal GPU acceleration
- **Two Compilers**: HODU (embedded-friendly) and XLA (performance-optimized)

> [!WARNING]
>
> This is a personal learning and development project. As such:
> - The cli and framework are under active development
> - Features may be experimental or incomplete
> - Functionality is not guaranteed for production use
>
> It is recommended to use the latest version.

## Components

### hodu (Library)

The main ML framework library for Rust applications.

```bash
cargo add hodu
```

```rust
use hodu::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_runtime_device(Device::CPU);

    let a = Tensor::randn(&[2, 3], 0f32, 1.)?;
    let b = Tensor::randn(&[3, 4], 0f32, 1.)?;
    let c = a.matmul(&b)?;

    println!("{}", c);
    Ok(())
}
```

See [hodu-lib/README.md](hodu-lib/README.md) for usage examples and features.

### hodu-cli (Compiler & Runtime)

Command-line tool for compiling and executing Hodu scripts.

```bash
cargo install hodu-cli
```

```bash
hodu build ~
```

See [hodu-cli/README.md](hodu-cli/README.md) for CLI documentation.

## Get started

### Requirements

**Required**
- Rust 1.90.0 or later (latest stable version recommended)

**Optional**
- **OpenBLAS 0.3.30+** - For optimized linear algebra operations on CPU
  - macOS: `brew install openblas gfortran`
  - Linux: `sudo apt install libopenblas-dev pkg-config gfortran`
  - Windows: Install via vcpkg or MinGW

- **LLVM/Clang** - Required when building with the `xla` feature
  - macOS: `brew install llvm`
  - Linux: `sudo apt install llvm clang`
  - Windows: Install from LLVM releases

- **CUDA Toolkit** - Required when using the `cuda` feature
  - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

- **Xcode Command Line Tools** - Required when using the `metal` feature on macOS
  - `xcode-select --install`

## Supported Platforms

### Standard Environments

| Target Triple | Backend | Device | Features | Status |
|--------------|---------|--------|----------|--------|
| x86_64-unknown-linux-gnu | HODU | CPU | `std` | ✅ Stable |
| | HODU | CUDA | `std`, `cuda` | ✅ Stable |
| | XLA | CPU | `std`, `xla` | ✅ Stable |
| | XLA | CUDA | `std`, `xla`, `cuda` | 🚧 In Development |
| aarch64-unknown-linux-gnu | HODU | CPU | `std` | ✅ Stable |
| | XLA | CPU | `std`, `xla` | ✅ Stable |
| x86_64-apple-darwin | HODU | CPU | `std` | 🧪 Experimental |
| | XLA | CPU | `std`, `xla` | 🚧 In Development |
| aarch64-apple-darwin | HODU | CPU | `std` | ✅ Stable |
| | HODU | Metal | `std`, `metal` | 🧪 Experimental |
| | XLA | CPU | `std`, `xla` | ✅ Stable |
| x86_64-pc-windows-msvc | HODU | CPU | `std` | 🧪 Experimental |
| | HODU | CUDA | `std`, `cuda` | 🧪 Experimental |
| | XLA | CPU | `std`, `xla` | 🚧 In Development |
| | XLA | CUDA | `std`, `xla`, `cuda` | 🚧 In Development |

### Embedded Environments

🧪 **Experimental**: Embedded platforms are supported but are experimental and not extensively tested in production environments.

> **Note**: Development should be done in a standard (std) host environment. Cross-compilation for embedded targets is supported.

| Target Triple | Backend | Device | Features | Status |
|--------------|---------|--------|----------|--------|
| thumbv7em-none-eabihf | HODU | CPU | (no default) | 🧪 Experimental |
| aarch64-unknown-none | HODU | CPU | (no default) | 🧪 Experimental |
| | HODU | CUDA | `cuda` | 🧪 Experimental (Jetson) |
| armv7a-none-eabi | HODU | CPU | (no default) | 🧪 Experimental |

## Docs

[TODOS](docs/TODOS.md) - Planned features and improvements

## Credits

Hodu Character Design: Created by <a href="https://github.com/SkuldNorniern">Eira</a>
