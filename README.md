# Hodu, a user-friendly ML toolkit built in Rust.

> **Hodu** (호두) is a Korean word meaning "walnut".

<p align="center">
    <img src="./assets/hodu/hodu_avatar.png" alt="Hodu Avatar" width="256" height="256" />
</p>

**Hodu** is a Rust-based ML toolkit designed for ease of use, from prototyping to deployment.

Built on Rust's foundation of memory safety and zero-cost abstractions, Hodu provides a simple API for tensor operations and model building through **hodu-lib**, while **hodu-cli** handles model inference, format conversion, and deployment with a single command.

The plugin system allows you to extend Hodu with custom model formats and execution backends, making it adaptable to various workflows and hardware targets.

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
# Run inference
hodu run model.onnx --input x=input.npy
hodu run model.hdss --inputs x=input1.hdt,y=input2.json

# Build executable
hodu build model.hdss -o model

# Build library
hodu build model.onnx -o model.dylib
```

### hodu-gui (PLANNED)

GUI application for model visualization and editing.

### [hodu-plugin-sdk](./hodu-plugin-sdk/README.md)

SDK for building format and backend plugins. Create custom plugins to support new model formats or execution backends.

## License

This project is licensed under the [BSD-3-Clause License](./LICENSE).
