use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Conv2d, Conv2dConfig, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::env;
use std::time::Instant;

// Statistical utilities
fn trimmed_mean(times: &mut [f64], trim_ratio: f64) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let trim_count = (times.len() as f64 * trim_ratio) as usize;
    if trim_count > 0 {
        let trimmed = &times[trim_count..times.len() - trim_count];
        trimmed.iter().sum::<f64>() / trimmed.len() as f64
    } else {
        times.iter().sum::<f64>() / times.len() as f64
    }
}

#[derive(Debug, Clone, Copy)]
enum BenchMode {
    DynamicCPU,
    #[cfg(feature = "metal")]
    DynamicMetal,
    #[cfg(feature = "cuda")]
    DynamicCuda,
}

impl BenchMode {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "dynamic-cpu" => Some(Self::DynamicCPU),
            #[cfg(feature = "metal")]
            "dynamic-metal" => Some(Self::DynamicMetal),
            #[cfg(feature = "cuda")]
            "dynamic-cuda" => Some(Self::DynamicCuda),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::DynamicCPU => "Dynamic CPU",
            #[cfg(feature = "metal")]
            Self::DynamicMetal => "Dynamic Metal",
            #[cfg(feature = "cuda")]
            Self::DynamicCuda => "Dynamic CUDA",
        }
    }

    fn device(&self) -> Result<Device, Box<dyn std::error::Error>> {
        match self {
            Self::DynamicCPU => Ok(Device::Cpu),
            #[cfg(feature = "metal")]
            Self::DynamicMetal => Ok(Device::new_metal(0)?),
            #[cfg(feature = "cuda")]
            Self::DynamicCuda => Ok(Device::new_cuda(0)?),
        }
    }
}

// Simple CNN model
struct SimpleCNN {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
}

impl SimpleCNN {
    fn new(vb: VarBuilder) -> Result<Self, Box<dyn std::error::Error>> {
        let conv_config = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };

        let conv1 = candle_nn::conv2d(3, 32, 3, conv_config, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 3, conv_config, vb.pp("conv2"))?;

        let fc1 = candle_nn::linear(64 * 8 * 8, 128, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(128, 10, vb.pp("fc2"))?;

        Ok(Self { conv1, conv2, fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Conv1 + ReLU + Pool
        let x = self.conv1.forward(x)?;
        let x = x.relu()?;
        let x = x.max_pool2d_with_stride(2, 2)?;

        // Conv2 + ReLU + Pool
        let x = self.conv2.forward(&x)?;
        let x = x.relu()?;
        let x = x.max_pool2d_with_stride(2, 2)?;

        // Flatten
        let batch_size = x.dim(0)?;
        let x = x.reshape((batch_size, 64 * 8 * 8))?;

        // FC1 + ReLU
        let x = self.fc1.forward(&x)?;
        let x = x.relu()?;

        // FC2
        self.fc2.forward(&x).map_err(|e| e.into())
    }
}

fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let log_softmax = candle_nn::ops::log_softmax(logits, 1)?;
    let nll = log_softmax.gather(&targets.unsqueeze(1)?, 1)?;
    let loss = nll.neg()?.mean_all()?;
    Ok(loss)
}

fn benchmark_training(
    mode: BenchMode,
    batch_size: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let device = mode.device()?;

    // Create model with VarMap
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = SimpleCNN::new(vb)?;

    // Optimizer
    let params = ParamsAdamW {
        lr: 0.001,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

    // Random data
    let x = Tensor::randn(0f32, 1f32, (batch_size, 3, 32, 32), &device)?;
    let target = (Tensor::rand(0f32, 1f32, (batch_size,), &device)? * 10.0)?.to_dtype(DType::U32)?;

    // Warmup
    for _ in 0..warmup {
        let logits = model.forward(&x)?;
        let loss = cross_entropy_loss(&logits, &target)?;
        optimizer.backward_step(&loss)?;
    }

    // Benchmark full training step (forward + backward + optimizer step)
    let mut step_times = Vec::with_capacity(iterations);
    let bench_start = Instant::now();

    for i in 0..iterations {
        let start = Instant::now();
        let logits = model.forward(&x)?;
        let loss = cross_entropy_loss(&logits, &target)?;
        optimizer.backward_step(&loss)?;
        step_times.push(start.elapsed().as_secs_f64());

        if bench_start.elapsed().as_secs_f64() > 60.0 {
            return Err(format!("TIMEOUT: Exceeded 60 seconds after {} iterations", i + 1).into());
        }
    }

    Ok(trimmed_mean(&mut step_times, 0.1))
}

fn run_benchmark(
    mode: BenchMode,
    configs: &[usize],
    warmup: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("mode={}", mode.name());
    println!("warmup={}", warmup);
    println!("iterations={}", iterations);

    let mut timed_out = false;

    for &batch_size in configs {
        if timed_out {
            println!("{},TIMEOUT", batch_size);
            continue;
        }

        let result = benchmark_training(mode, batch_size, warmup, iterations);

        match result {
            Ok(step_time) => {
                println!("{},time_ms={:.6}", batch_size, step_time * 1000.0);
            },
            Err(e) if e.to_string().contains("TIMEOUT") => {
                println!("{},TIMEOUT", batch_size);
                timed_out = true;
            },
            Err(_) => {
                println!("{},ERROR", batch_size);
            },
        }
    }

    Ok(())
}

fn print_usage() {
    println!("Usage: bench <mode>");
    println!("\nAvailable modes:");
    println!("  dynamic-cpu     - Dynamic execution on CPU");
    #[cfg(feature = "cuda")]
    println!("  dynamic-cuda    - Dynamic execution on CUDA");
    #[cfg(feature = "metal")]
    println!("  dynamic-metal   - Dynamic execution on Metal");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let mode = match BenchMode::from_str(&args[1]) {
        Some(m) => m,
        None => {
            eprintln!("Error: Invalid mode '{}'", args[1]);
            print_usage();
            std::process::exit(1);
        },
    };

    let warmup = if args.len() > 2 {
        args[2].parse().unwrap_or(5)
    } else {
        5
    };

    let iterations = if args.len() > 3 {
        args[3].parse().unwrap_or(100)
    } else {
        100
    };

    // CNN training configs: batch sizes for 32x32 RGB images
    let configs = [16, 32, 64];

    run_benchmark(mode, &configs, warmup, iterations)?;

    Ok(())
}
