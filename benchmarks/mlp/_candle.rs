use candle_core::{DType, Device, Tensor};
use std::env;
use std::time::Instant;

// Statistical utilities
fn median(times: &mut [f64]) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = times.len();
    if len % 2 == 0 {
        (times[len / 2 - 1] + times[len / 2]) / 2.0
    } else {
        times[len / 2]
    }
}

fn trimmed_mean(times: &mut [f64], trim_ratio: f64) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let trim_count = (times.len() as f64 * trim_ratio) as usize;
    let trimmed = &times[trim_count..times.len() - trim_count];
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
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

// MLP Block implementation
struct MLPBlock {
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
    fc3_w: Tensor,
    fc3_b: Tensor,
    projection: Option<Tensor>,
}

impl MLPBlock {
    fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        device: &Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Xavier initialization
        let k_in = (1.0 / in_features as f32).sqrt();
        let k_hidden = (1.0 / hidden_features as f32).sqrt();

        let fc1_w = Tensor::randn(0f32, 1f32, (in_features, hidden_features), device)?
            .to_dtype(DType::F32)?
            .affine(k_in as f64, 0.0)?;
        let fc1_b = Tensor::zeros((hidden_features,), DType::F32, device)?;

        let fc2_w = Tensor::randn(0f32, 1f32, (hidden_features, hidden_features), device)?
            .to_dtype(DType::F32)?
            .affine(k_hidden as f64, 0.0)?;
        let fc2_b = Tensor::zeros((hidden_features,), DType::F32, device)?;

        let fc3_w = Tensor::randn(0f32, 1f32, (hidden_features, out_features), device)?
            .to_dtype(DType::F32)?
            .affine(k_hidden as f64, 0.0)?;
        let fc3_b = Tensor::zeros((out_features,), DType::F32, device)?;

        let projection = if in_features != out_features {
            Some(
                Tensor::randn(0f32, 1f32, (in_features, out_features), device)?
                    .to_dtype(DType::F32)?
                    .affine(k_in as f64, 0.0)?,
            )
        } else {
            None
        };

        Ok(Self {
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            fc3_w,
            fc3_b,
            projection,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Save input for residual connection
        let identity = if let Some(ref proj) = self.projection {
            x.matmul(proj)?
        } else {
            x.clone()
        };

        // Forward through layers
        let mut out = x.matmul(&self.fc1_w)?.broadcast_add(&self.fc1_b)?;
        out = out.gelu()?;
        out = out.matmul(&self.fc2_w)?.broadcast_add(&self.fc2_b)?;
        out = out.gelu()?;
        out = out.matmul(&self.fc3_w)?.broadcast_add(&self.fc3_b)?;

        // Residual connection
        out.broadcast_add(&identity).map_err(|e| e.into())
    }
}

fn benchmark_dynamic(
    mode: BenchMode,
    batch_size: usize,
    in_features: usize,
    hidden_features: usize,
    out_features: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let device = mode.device()?;

    let mlp = MLPBlock::new(in_features, hidden_features, out_features, &device)?;
    let x = Tensor::randn(0f32, 1f32, (batch_size, in_features), &device)?.to_dtype(DType::F32)?;

    // Warmup
    for _ in 0..warmup {
        let _ = mlp.forward(&x)?;
    }

    // Benchmark - collect individual iteration times
    let mut times = Vec::with_capacity(iterations);
    let bench_start = Instant::now();
    for i in 0..iterations {
        let start = Instant::now();
        let _ = mlp.forward(&x)?;
        times.push(start.elapsed().as_secs_f64());

        // Check timeout after each iteration
        let total_elapsed = bench_start.elapsed();
        if total_elapsed.as_secs_f64() > 2.0 {
            return Err(format!("TIMEOUT: Exceeded 2 seconds after {} iterations", i + 1).into());
        }
    }

    // Use trimmed mean (remove top/bottom 10%)
    Ok(trimmed_mean(&mut times, 0.1))
}

fn run_benchmark(
    mode: BenchMode,
    configs: &[(usize, usize, usize, usize)],
    warmup: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("mode={}", mode.name());
    println!("warmup={}", warmup);
    println!("iterations={}", iterations);

    let mut timed_out = false;

    for (batch_size, in_features, hidden_features, out_features) in configs {
        // If we already timed out, skip remaining benchmarks
        if timed_out {
            println!(
                "{}x{}x{}x{},TIMEOUT",
                batch_size, in_features, hidden_features, out_features
            );
            continue;
        }

        let result = benchmark_dynamic(
            mode,
            *batch_size,
            *in_features,
            *hidden_features,
            *out_features,
            warmup,
            iterations,
        );

        match result {
            Ok(time) => {
                println!(
                    "{}x{}x{}x{},time_ms={:.6}ms",
                    batch_size,
                    in_features,
                    hidden_features,
                    out_features,
                    time * 1000.0
                );
            },
            Err(e) if e.to_string().contains("TIMEOUT") => {
                println!(
                    "{}x{}x{}x{},TIMEOUT",
                    batch_size, in_features, hidden_features, out_features
                );
                timed_out = true;
            },
            Err(e) => return Err(e),
        }
    }

    Ok(())
}

fn print_usage() {
    println!("Usage: candle <mode>");
    println!("\nAvailable modes:");
    println!("  dynamic-cpu     - Dynamic execution on CPU");
    #[cfg(feature = "metal")]
    println!("  dynamic-metal   - Dynamic execution on Metal");
    #[cfg(feature = "cuda")]
    println!("  dynamic-cuda    - Dynamic execution on CUDA");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
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

    // MLP configs: (batch_size, in_features, hidden_features, out_features)
    let configs = [(32, 256, 512, 256), (64, 512, 1024, 512), (128, 768, 2048, 1024)];

    let warmup = 10;
    let iterations = 30;

    run_benchmark(mode, &configs, warmup, iterations)?;

    Ok(())
}
