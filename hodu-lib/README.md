# Hodu

[![hodu - Crates.io](https://img.shields.io/crates/v/hodu.svg)](https://crates.io/crates/hodu)
[![hodu - Doc.rs](https://docs.rs/hodu/badge.svg)](https://docs.rs/hodu)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/hodu-rs/hodu#license)

A user-friendly ML framework built in Rust for rapid prototyping and embedded deployment.

## Installation

```toml
[dependencies]
hodu = "0.2"
```

## Quick Start

### Dynamic Execution

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

### Static Computation Graphs

```rust
use hodu::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let builder = Builder::new("matmul_example".to_string());
    builder.start()?;

    let a = Tensor::input("a", &[2, 3])?;
    let b = Tensor::input("b", &[3, 4])?;
    let c = a.matmul(&b)?;

    builder.add_output("result", c)?;
    builder.end()?;

    let mut script = builder.build()?;
    script.set_input("a", Tensor::randn(&[2, 3], 0f32, 1.)?);
    script.set_input("b", Tensor::randn(&[3, 4], 0f32, 1.)?);

    let output = script.run()?;
    println!("{}", output["result"]);
    Ok(())
}
```

## Features

### Default
- `std` - Standard library support
- `serde` - Serialization support
- `rayon` - Parallel processing

### Optional
- `openblas` - OpenBLAS acceleration
- `cuda` - NVIDIA CUDA GPU support
- `metal` - Apple Metal GPU support
- `xla` - Google XLA compiler backend

### Data Types
- `f8e5m2`, `f64`, `u16`, `u64`, `i16`, `i64` - Additional data type support

## Documentation

For full documentation, visit [docs.rs/hodu](https://docs.rs/hodu)

For the complete project overview, see the [main repository](https://github.com/hodu-rs/hodu)
