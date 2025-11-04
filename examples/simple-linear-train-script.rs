use hodu::{nn::SGD, prelude::*};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_data: Vec<Vec<f32>> = (0..10000)
        .map(|i| {
            vec![
                (i % 100) as f32 / 100.0,
                ((i % 100) + 1) as f32 / 100.0,
                ((i % 100) + 2) as f32 / 100.0,
            ]
        })
        .collect();
    let target_data: Vec<Vec<f32>> = (0..10000).map(|i| vec![((i % 100) * 10) as f32 / 1000.0]).collect();

    let input_tensor = Tensor::new(input_data)?;
    let target_tensor = Tensor::new(target_data)?;

    // Build script using new API
    let builder = Builder::new("linear_training".to_string());
    builder.start()?;

    let mut linear = Linear::new(3, 1, true, DType::F32)?;
    let mse_loss = MSELoss::new();
    let mut optimizer = SGD::new(0.01);

    let input = Tensor::input("input", [10000, 3])?;
    input.requires_grad()?;
    let target = Tensor::input("target", [10000, 1])?;

    let epochs = 1000;
    let mut final_loss = Tensor::scalar(0.0)?;

    for _ in 0..epochs {
        let pred = linear.forward(&input)?;
        let loss = mse_loss.forward((&pred, &target))?;

        loss.backward()?;

        optimizer.step(&mut linear.parameters())?;
        optimizer.zero_grad(&mut linear.parameters())?;

        final_loss = loss;
    }

    let params = linear.parameters();
    builder.add_output("loss", final_loss)?;
    builder.add_output("weight", *params[0])?;
    builder.add_output("bias", *params[1])?;

    builder.end()?;

    // Build module and create script
    let module = builder.build()?;
    let mut script = Script::new(module);

    // Set device
    #[cfg(feature = "metal")]
    script.set_device(Device::Metal);

    // Set compiler type
    #[cfg(feature = "xla")]
    script.set_compiler(Compiler::XLA);

    // Set runtime inputs
    script.set_input("input", input_tensor);
    script.set_input("target", target_tensor);

    println!("Compiling script...");
    let compile_start = Instant::now();
    script.compile()?;
    let compile_elapsed = compile_start.elapsed();
    println!("Compilation time: {:?}", compile_elapsed);

    println!("Running script...");
    let run_start = Instant::now();
    let output = script.run()?;
    let run_elapsed = run_start.elapsed();

    println!("Loss: {}", output["loss"]);
    println!("Weight: {}", output["weight"]);
    println!("Bias: {}", output["bias"]);
    println!("Execution time: {:?}", run_elapsed);
    println!("Total time: {:?}", compile_elapsed + run_elapsed);

    Ok(())
}
