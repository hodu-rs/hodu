use candle_core::{DType, Device, Tensor};
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

fn benchmark_dynamic(
    mode: BenchMode,
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let device = mode.device()?;

    let a = Tensor::randn(0f32, 1f32, (m, k), &device)?.to_dtype(DType::F32)?;
    let b = Tensor::randn(0f32, 1f32, (k, n), &device)?.to_dtype(DType::F32)?;

    // Warmup
    for _ in 0..warmup {
        let _ = a.matmul(&b)?;
    }

    // Benchmark - collect individual iteration times
    let mut times = Vec::with_capacity(iterations);
    let bench_start = Instant::now();
    for i in 0..iterations {
        let start = Instant::now();
        let _ = a.matmul(&b)?;
        times.push(start.elapsed().as_secs_f64());

        // Check timeout after each iteration
        let total_elapsed = bench_start.elapsed();
        if total_elapsed.as_secs_f64() > 10.0 {
            return Err(format!("TIMEOUT: Exceeded 10 seconds after {} iterations", i + 1).into());
        }
    }

    // Use trimmed mean (remove top/bottom 10%)
    Ok(trimmed_mean(&mut times, 0.1))
}

fn run_benchmark(
    mode: BenchMode,
    configs: &[(usize, usize, usize)],
    warmup: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("mode={}", mode.name());
    println!("warmup={}", warmup);
    println!("iterations={}", iterations);

    let mut timed_out = false;

    for (m, k, n) in configs {
        // If we already timed out, skip remaining benchmarks
        if timed_out {
            println!("{}x{}x{},TIMEOUT", m, k, n);
            continue;
        }

        let result = benchmark_dynamic(mode, *m, *k, *n, warmup, iterations);

        match result {
            Ok(time) => {
                println!("{}x{}x{},time_ms={:.6}ms", m, k, n, time * 1000.0);
            },
            Err(e) if e.to_string().contains("TIMEOUT") => {
                println!("{}x{}x{},TIMEOUT", m, k, n);
                timed_out = true; // Mark as timed out to skip remaining benchmarks
            },
            Err(_) => {
                println!("{}x{}x{},ERROR", m, k, n);
            },
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

    let configs = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)];

    let warmup = 100;
    let iterations = 100;

    run_benchmark(mode, &configs, warmup, iterations)?;

    Ok(())
}
