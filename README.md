<div align="center">
  <img src="assets/hodu/hodu_avatar.png" alt="Hodu Avatar" width="200"/>
   <br>
   <sub><i>Character by: <a href="https://github.com/SkuldNorniern">Eira</a></i></sub>

  <h1>hodu</h1>
  <p>Hodu is a user-friendly ML framework built in Rust for rapid prototyping and embedded deployment.</p>

  [![Crates.io](https://img.shields.io/crates/v/hodu.svg)](https://crates.io/crates/hodu)
  [![Doc.rs](https://docs.rs/hodu/badge.svg)](https://docs.rs/hodu)
  [![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/hodu-rs/hodu#license)
</div>

> **Hodu** (호두) is a Korean word meaning "walnut".

## About hodu

`hodu` is a machine learning library built with user convenience at its core, designed for both rapid prototyping and seamless production deployment—including embedded environments.

While Hodu shares similarities with PyTorch and TensorFlow, it brings a unique approach to ML workflows that prioritizes ease of use without sacrificing performance.

For static computation graphs, we leverage powerful backends like XLA to provide fast, optimized just-in-time compilation that keeps your models running at peak efficiency.

> [!WARNING]
>
> This is a personal learning and development project. As such:
> - The framework is under active development
> - Features may be experimental or incomplete
> - Functionality is not guaranteed for production use
> 
> It is recommended to use the latest version.

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
