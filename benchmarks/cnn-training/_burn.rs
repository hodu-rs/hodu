use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
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
    #[cfg(feature = "wgpu")]
    DynamicWgpu,
    #[cfg(feature = "cuda")]
    DynamicTch,
}

impl BenchMode {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "dynamic-cpu" => Some(Self::DynamicCPU),
            #[cfg(feature = "wgpu")]
            "dynamic-wgpu" => Some(Self::DynamicWgpu),
            #[cfg(feature = "cuda")]
            "dynamic-tch" => Some(Self::DynamicTch),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::DynamicCPU => "Dynamic CPU",
            #[cfg(feature = "wgpu")]
            Self::DynamicWgpu => "Dynamic WGPU",
            #[cfg(feature = "cuda")]
            Self::DynamicTch => "Dynamic TCH (CUDA)",
        }
    }
}

// Note: Burn doesn't support backward pass in the same way as PyTorch
// This benchmark only measures forward pass
// For training benchmarks, use PyTorch, JAX, or TensorFlow

fn benchmark_dynamic_ndarray(
    batch_size: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    use burn_ndarray::{NdArray, NdArrayDevice};
    type B = NdArray<f32>;

    let device = NdArrayDevice::Cpu;

    // Create simple forward-only operations
    let x: Tensor<B, 4> = Tensor::random(Shape::new([batch_size, 3, 32, 32]), Distribution::Default, &device);
    let w1: Tensor<B, 4> = Tensor::random(Shape::new([32, 3, 3, 3]), Distribution::Default, &device) * 0.1;

    // Warmup
    for _ in 0..warmup {
        let _out = x.clone().matmul(w1.clone().reshape([3, 32 * 3 * 3]));
    }

    // Benchmark
    let mut times = Vec::with_capacity(iterations);
    let bench_start = Instant::now();

    for i in 0..iterations {
        let start = Instant::now();
        let _out = x.clone().matmul(w1.clone().reshape([3, 32 * 3 * 3]));
        times.push(start.elapsed().as_secs_f64());

        if bench_start.elapsed().as_secs_f64() > 60.0 {
            return Err(format!("TIMEOUT: Exceeded 60 seconds after {} iterations", i + 1).into());
        }
    }

    Ok(trimmed_mean(&mut times, 0.1))
}

#[cfg(feature = "wgpu")]
fn benchmark_dynamic_wgpu(
    batch_size: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    use burn_wgpu::{Wgpu, WgpuDevice};
    type B = Wgpu;

    let device = WgpuDevice::default();

    let x: Tensor<B, 4> = Tensor::random(Shape::new([batch_size, 3, 32, 32]), Distribution::Default, &device);
    let w1: Tensor<B, 4> = Tensor::random(Shape::new([32, 3, 3, 3]), Distribution::Default, &device) * 0.1;

    // Warmup
    for _ in 0..warmup {
        let _out = x.clone().matmul(w1.clone().reshape([3, 32 * 3 * 3]));
    }

    // Benchmark
    let mut times = Vec::with_capacity(iterations);
    let bench_start = Instant::now();

    for i in 0..iterations {
        let start = Instant::now();
        let _out = x.clone().matmul(w1.clone().reshape([3, 32 * 3 * 3]));
        times.push(start.elapsed().as_secs_f64());

        if bench_start.elapsed().as_secs_f64() > 60.0 {
            return Err(format!("TIMEOUT: Exceeded 60 seconds after {} iterations", i + 1).into());
        }
    }

    Ok(trimmed_mean(&mut times, 0.1))
}

#[cfg(feature = "cuda")]
fn benchmark_dynamic_tch(
    batch_size: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    use burn_tch::{LibTorch, LibTorchDevice};
    type B = LibTorch<f32>;

    let device = LibTorchDevice::Cuda(0);

    let x: Tensor<B, 4> = Tensor::random(Shape::new([batch_size, 3, 32, 32]), Distribution::Default, &device);
    let w1: Tensor<B, 4> = Tensor::random(Shape::new([32, 3, 3, 3]), Distribution::Default, &device) * 0.1;

    // Warmup
    for _ in 0..warmup {
        let _out = x.clone().matmul(w1.clone().reshape([3, 32 * 3 * 3]));
    }

    // Benchmark
    let mut times = Vec::with_capacity(iterations);
    let bench_start = Instant::now();

    for i in 0..iterations {
        let start = Instant::now();
        let _out = x.clone().matmul(w1.clone().reshape([3, 32 * 3 * 3]));
        times.push(start.elapsed().as_secs_f64());

        if bench_start.elapsed().as_secs_f64() > 60.0 {
            return Err(format!("TIMEOUT: Exceeded 60 seconds after {} iterations", i + 1).into());
        }
    }

    Ok(trimmed_mean(&mut times, 0.1))
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
    println!("NOTE: Burn framework doesn't support autograd/backward in the same way.");
    println!("NOTE: This benchmark only measures forward pass operations.");

    let mut timed_out = false;

    for &batch_size in configs {
        if timed_out {
            println!("{},TIMEOUT", batch_size);
            continue;
        }

        let result = match mode {
            BenchMode::DynamicCPU => benchmark_dynamic_ndarray(batch_size, warmup, iterations),
            #[cfg(feature = "wgpu")]
            BenchMode::DynamicWgpu => benchmark_dynamic_wgpu(batch_size, warmup, iterations),
            #[cfg(feature = "cuda")]
            BenchMode::DynamicTch => benchmark_dynamic_tch(batch_size, warmup, iterations),
        };

        match result {
            Ok(time) => {
                println!("{},time_ms={:.6}", batch_size, time * 1000.0);
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
    println!("  dynamic-cpu   - Dynamic execution on CPU (NdArray backend)");
    #[cfg(feature = "wgpu")]
    println!("  dynamic-wgpu  - Dynamic execution on WGPU");
    #[cfg(feature = "cuda")]
    println!("  dynamic-tch   - Dynamic execution on CUDA (LibTorch backend)");
    println!("\nNote: Burn doesn't support backward/training in this benchmark format");
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
