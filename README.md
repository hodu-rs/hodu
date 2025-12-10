# Hodu, a Rust ML toolkit

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](./LICENSE)

> **Hodu** (호두) is a Korean word meaning "walnut".

<p align="center">
    <img src="./assets/hodu/sd_logo.png" alt="Hodu Avatar" width="384" />
</p>

Tensor ops, model building, inference, deployment. All in one.

## Packages

### [hodu/cli](./hodu-cli/README.md)

[![hodu-cli](https://img.shields.io/crates/v/hodu.svg?label=hodu/cli)](https://crates.io/crates/hodu-cli)

Run, convert, build models from the command line.

```bash
$ cargo install hodu-cli
```

```bash
$ hodu plugin install aot-cpu

$ hodu run model.onnx --input x=input.npy
$ hodu build model.hdss -o model
$ hodu build model.onnx -o model.dylib
```

### [hodu/lib](./hodu-lib/README.md)

[![hodu](https://img.shields.io/crates/v/hodu.svg?label=hodu/lib)](https://crates.io/crates/hodu)

Core library for tensors and models.

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

### [hodu-plugin-sdk](./hodu-plugin-sdk/README.md)

[![hodu-plugin-sdk](https://img.shields.io/crates/v/hodu-plugin-sdk.svg?label=hodu-plugin-sdk)](https://crates.io/crates/hodu-plugin-sdk)

Build your own format/backend plugins. JSON-RPC over stdio.

```bash
$ curl -fsSL https://raw.githubusercontent.com/daminstudio/hodu/main/hodu-plugin-sdk/new.sh | sh
```

Official plugins: [hodu-plugins](https://github.com/daminstudio/hodu-plugins)

## Ideology

### Why another ML library?

The Rust ecosystem already has great ML libraries like [Candle](https://github.com/huggingface/candle) and [Burn](https://github.com/tracel-ai/burn). But we saw a different problem.

Most ML workflows look like this: write a model in Python, export to ONNX, convert to an inference runtime, then deploy. Each step uses different tools, different formats, different constraints. Deploying a single model means juggling Python, C++, and platform-specific SDKs.

Hodu unifies this pipeline into one ecosystem.

### One ecosystem, unified

**hodu** (cli) follows a simple principle: one command gets you what you need. `hodu run` for inference, `hodu build` for native binaries, `hodu inspect` to examine models. No config files or build scripts required.

**hodu** (lib) gives you a familiar PyTorch-style API for building models. Operations can be captured into a computation graph for optimization and compilation. Every kernel is implemented from scratch—no external ML runtime dependencies.

**hodu-plugin-sdk** keeps Hodu open for extension. Plugins run as separate processes communicating via JSON-RPC, so you can write them in any language. Need a new model format or hardware backend? Build a plugin without touching Hodu's core.

### What we believe

- **Tools should get out of your way.** Focus on models, not configuration.
- **One language, one ecosystem.** Prototype to production in Rust, no language switching.
- **Extension should be open.** We can't support everything, so anyone can add what they need.

## Contributing

> **Note**: The project structure is still being established. For now, please open an issue first before submitting pull requests. This helps us coordinate work and avoid conflicts during this early stage.

Bug reports, feature requests, and feedback are always welcome!

Check out [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the [BSD-3-Clause License](./LICENSE).

### Assets

All icons, characters, and visual assets are created by **해꿈(sundream)** and © Han Damin. All rights reserved. These assets may not be used, modified, or distributed for any purpose, commercial or non-commercial, without explicit written permission from the copyright holder. See [Assets License](./assets/hodu/LICENSE) for details.

For licensing inquiries, contact: [miniex@daminstudio.net](mailto:miniex@daminstudio.net)
