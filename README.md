# Hodu, a user-friendly ML framework built in Rust.

<p align="center">
    <img src="./assets/hodu/hodu_avatar.png" alt="Hodu Avatar" width="180"/>
</p>

> **Hodu** (호두) is a Korean word meaning "walnut".

> [!WARNING]
>
> This is a personal learning and development project. As such:
> - The library is under active development
> - Features may be experimental or incomplete
> - Functionality is not guaranteed for production use
>
> It is recommended to use the latest version.

## About Hodu

**Hodu** is a Rust-based ML toolkit designed for ease of use, from prototyping to deployment.

- **hodu-lib**: Build and train models with a simple, intuitive API
- **hodu-cli**: Run inference and convert models with a single command
- **Plugin system**: Extend with custom formats and backends

## Components

### [hodu-lib](./hodu-lib/README.md)

Core ML library for tensor operations and model building. Use it as a Rust dependency for building ML applications.

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

### [hodu-cli](./hodu-cli/README.md)

Command-line interface for model inference, conversion, and deployment.

```bash
hodu run model.hdss --input x=input.hdt --save ./output
hodu run model.onnx --input x=input.npy
```

### hodu-gui (PLANNED)

GUI application for model visualization and editing.

### [hodu-plugin-sdk](./hodu-plugin-sdk/README.md)

SDK for building format and backend plugins. Create custom plugins to support new model formats or execution backends.

