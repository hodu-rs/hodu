# Script Mode Guide

## What is Script?

Script is a feature that enables optimized execution by pre-compiling computation graphs. Unlike dynamic execution mode, operations are first converted to IR (Intermediate Representation), then optimized and executed by backends (HODU, XLA).

## Basic Usage

### 1. Building a Script

```rust
use hodu::prelude::*;

// Create and start builder
let builder = Builder::new("my_script".to_string());
builder.start()?;

// Define operations in script mode
let x = Tensor::input("x", &[2, 3])?;
let y = Tensor::input("y", &[3, 4])?;
let result = x.matmul(&y)?;

// Register outputs
builder.add_output("result", result)?;
builder.end()?;

// Build script
let mut script = builder.build()?;
```

### 2. Running a Script

```rust
// Prepare input data
let x_data = Tensor::randn(&[2, 3], DType::F32)?;
let y_data = Tensor::randn(&[3, 4], DType::F32)?;

// Add inputs
script.add_input("x", x_data);
script.add_input("y", y_data);

// Execute
let outputs = script.run()?;
let result = &outputs["result"];
```

## Compilation Caching

Script caches compilation results to improve performance on repeated executions.

### Automatic Compilation

```rust
let mut script = builder.build()?;

// First run() automatically compiles
script.add_input("x", x_data);
let output1 = script.run()?;  // Compile + execute

// Subsequent run() uses cached compilation
script.add_input("x", x_data2);
let output2 = script.run()?;  // Execute only
```

### Explicit Compilation

```rust
let mut script = builder.build()?;

// Explicitly pre-compile (warm-up)
println!("Compiling...");
script.compile()?;
println!("Compilation done!");

// Subsequent run() only executes
script.add_input("x", x_data);
let output = script.run()?;  // Execute only
```

### Measuring Compilation Time

```rust
use std::time::Instant;

let mut script = builder.build()?;
script.add_input("x", x_data);

// Measure compilation time
let compile_start = Instant::now();
script.compile()?;
let compile_time = compile_start.elapsed();
println!("Compilation: {:?}", compile_time);

// Measure execution time
let run_start = Instant::now();
let output = script.run()?;
let run_time = run_start.elapsed();
println!("Execution: {:?}", run_time);
```

## Backend Selection

### HODU Backend (Default)

```rust
let mut script = builder.build()?;
script.set_backend(Backend::HODU);  // Default
```

### XLA Backend

```rust
#[cfg(feature = "xla")]
{
    let mut script = builder.build()?;
    script.set_backend(Backend::XLA);
}
```

**Note**: Changing the backend invalidates cached compilation.

```rust
let mut script = builder.build()?;
script.compile()?;  // Compile with HODU

script.set_backend(Backend::XLA);  // Cache invalidated!
// Next run() will recompile with XLA
```

## Training in Script Mode

Special care is needed when writing training loops in script mode.

### Training Loop Example

```rust
let builder = Builder::new("training".to_string());
builder.start()?;

let mut linear = Linear::new(3, 1, true, DType::F32)?;
let mse_loss = MSE::new();
let mut optimizer = SGD::new(0.01);

let input = Tensor::input("input", &[100, 3])?;
let target = Tensor::input("target", &[100, 1])?;

let epochs = 1000;
let mut final_loss = Tensor::full(&[], 0.0)?;

for _ in 0..epochs {
    let pred = linear.forward(&input)?;
    let loss = mse_loss.forward((&pred, &target))?;

    // Backward - compute gradients
    loss.backward()?;

    // Optimizer update
    optimizer.step(&mut linear.parameters())?;
    optimizer.zero_grad(&mut linear.parameters())?;

    final_loss = loss;
}

builder.add_output("loss", final_loss)?;
builder.end()?;

let mut script = builder.build()?;
```

### Characteristics of Training Scripts

1. **Loop Unrolling**: Loops are fully unrolled during script building.
   - 1000 epochs = 1000 forward + backward + update operations recorded in graph
   - Graph size can become very large

2. **Gradient Tape**: Gradient tape works even in builder mode.
   - `backward()` actually computes gradients and records to tape
   - Operations with `requires_grad=true` must be recorded to tape

3. **Compilation Time**: Large graphs take long to compile.
   - XLA: Complex optimization process takes longer
   - HODU: Relatively faster compilation

## Important Notes

### 1. Transpose and Gradient

Shape operations like `transpose()` in builder mode **must also be recorded to gradient tape**:

```rust
// Correct implementation (ops.rs)
if builder::is_builder_active() {
    let (tensor_id, result) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);
    let op = Op::Shape(op::ShapeOp::Transpose, self.id());
    register_operation_in_builder(op.clone(), tensor_id, ...);

    if self.is_requires_grad() {
        gradient::record_operation(tensor_id, op, vec![self.id()])?;  // Required!
    }

    Ok(result)
}
```

Forgetting this causes "Gradient not computed" error during backward.

### 2. Script is Optimized for Inference

Script is more suitable for **inference** than training:

```rust
// Inference script
let builder = Builder::new("inference".to_string());
builder.start()?;

let input = Tensor::input("input", &[1, 784])?;
let output = model.forward(&input)?;  // requires_grad=false

builder.add_output("output", output)?;
builder.end()?;

let mut script = builder.build()?;
script.compile()?;  // Fast compilation

// Reuse multiple times
for batch in batches {
    script.add_input("input", batch);
    let result = script.run()?;  // Fast execution
}
```

### 3. Cache Invalidation Conditions

Compilation cache is invalidated when:

- `set_backend()` is called
- `set_device()` is called

```rust
let mut script = builder.build()?;
script.compile()?;  // Compilation complete

script.set_device(Device::CUDA(0));  // Cache invalidated!
// Next run() will recompile
```

### 4. Script Reuse

Same script can be executed multiple times with different inputs:

```rust
let mut script = builder.build()?;
script.compile()?;  // Compile once

for i in 0..100 {
    let input_data = generate_input(i);
    script.clear_inputs();  // Remove previous inputs
    script.add_input("x", input_data);
    let output = script.run()?;  // Fast execution
}
```

## Performance Tips

### 1. Warm-up with Explicit Compilation

```rust
// Pre-compile at program start
let mut script = builder.build()?;
script.add_input("x", dummy_input);
script.compile()?;  // Warm-up

// Actual data processing is fast
for batch in data {
    script.add_input("x", batch);
    let output = script.run()?;
}
```

### 2. Backend Selection

- **HODU**: Fast compilation, general-purpose performance
- **XLA**: Slow compilation, optimized execution (especially GPU)

```rust
// CPU inference: HODU recommended
script.set_backend(Backend::HODU);

// GPU inference: XLA recommended (compile once, run many times)
#[cfg(feature = "xla")]
{
    script.set_backend(Backend::XLA);
    script.set_device(Device::CUDA(0));
    script.compile()?;  // Takes time
    // Subsequent executions are very fast
}
```

### 3. Use Dynamic Mode for Training

Large training loops are more efficient with dynamic execution than script:

```rust
// Direct execution without script
let mut optimizer = SGD::new(0.01);

for epoch in 0..1000 {
    let pred = model.forward(&input)?;
    let loss = compute_loss(&pred, &target)?;
    loss.backward()?;
    optimizer.step(&mut model.parameters())?;
    optimizer.zero_grad(&mut model.parameters())?;
}
```

## Saving and Loading Scripts

```rust
#[cfg(all(feature = "serde", feature = "std"))]
{
    // Save script
    script.save("model.hoduscript")?;

    // Load script
    let mut loaded_script = Script::load("model.hoduscript")?;
    loaded_script.set_backend(Backend::XLA);
    loaded_script.compile()?;

    // Use
    loaded_script.add_input("x", input_data);
    let output = loaded_script.run()?;
}
```

## Summary

| Item | Description |
|------|-------------|
| **Purpose** | Inference optimization, repeated execution of fixed graphs |
| **Advantages** | Fast execution after compilation, backend optimization |
| **Disadvantages** | Compilation time, inefficient for large training loops |
| **Caching** | `compile()` result automatically cached, reused on repeated runs |
| **Invalidation** | When `set_backend()` or `set_device()` is called |
| **Gradient** | Training possible but dynamic mode recommended |
