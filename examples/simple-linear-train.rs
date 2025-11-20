use hodu::{nn::SGD, prelude::*};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    set_runtime_device(Device::CUDA(0));
    #[cfg(feature = "metal")]
    set_runtime_device(Device::Metal);

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

    let input = Tensor::new(input_data)?;
    let target = Tensor::new(target_data)?;
    input.requires_grad()?;

    let linear = Linear::new(3, 1, true, DType::F32)?;
    let mse_loss = MSELoss::new();
    let mut optimizer = SGD::new(0.01);
    let epochs = 1000;

    let mut hundred_epochs_start = Instant::now();

    for epoch in 0..epochs {
        let pred = linear.forward(&input)?;
        let loss = mse_loss.forward((&pred, &target))?;
        loss.backward()?;

        optimizer.step(&linear.parameters())?;
        optimizer.zero_grad(&linear.parameters())?;

        if (epoch + 1) % 100 == 0 {
            let hundred_elapsed = hundred_epochs_start.elapsed();
            let params = linear.parameters();
            println!(
                "Epoch {}: Loss = {}, 100 Epochs Time = {:?}, Weight = {}, Bias = {}",
                epoch + 1,
                loss,
                hundred_elapsed,
                params[0],
                params.get(1).unwrap()
            );
            hundred_epochs_start = Instant::now();
        }
    }

    Ok(())
}
