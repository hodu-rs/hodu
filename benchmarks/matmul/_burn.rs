use burn::tensor::{Distribution, Shape, Tensor};
use std::env;
use std::time::Instant;

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

fn benchmark_dynamic_ndarray(
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    use burn_ndarray::{NdArray, NdArrayDevice};

    type B = NdArray<f32>;

    let device = NdArrayDevice::Cpu;

    let a: Tensor<B, 2> = Tensor::random(Shape::new([m, k]), Distribution::Default, &device);
    let b: Tensor<B, 2> = Tensor::random(Shape::new([k, n]), Distribution::Default, &device);

    // Warmup
    for _ in 0..warmup {
        let _ = a.clone().matmul(b.clone());
    }

    // Benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let _ = a.clone().matmul(b.clone());

        // Check timeout after each iteration
        let elapsed = start.elapsed();
        if elapsed.as_secs_f64() > 1.0 {
            return Err(format!("TIMEOUT: Exceeded 1 second after {} iterations", i + 1).into());
        }
    }
    let elapsed = start.elapsed();

    Ok(elapsed.as_secs_f64() / iterations as f64)
}

#[cfg(feature = "wgpu")]
fn benchmark_dynamic_wgpu(
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    use burn_wgpu::{Wgpu, WgpuDevice};

    type B = Wgpu;

    let device = WgpuDevice::default();

    let a: Tensor<B, 2> = Tensor::random(Shape::new([m, k]), Distribution::Default, &device);
    let b: Tensor<B, 2> = Tensor::random(Shape::new([k, n]), Distribution::Default, &device);

    // Warmup
    for _ in 0..warmup {
        let _ = a.clone().matmul(b.clone());
    }

    // Benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let c = a.clone().matmul(b.clone());
        // Force synchronization by reading a value
        let _ = c.clone().into_data();

        // Check timeout after each iteration
        let elapsed = start.elapsed();
        if elapsed.as_secs_f64() > 1.0 {
            return Err(format!("TIMEOUT: Exceeded 1 second after {} iterations", i + 1).into());
        }
    }
    let elapsed = start.elapsed();

    Ok(elapsed.as_secs_f64() / iterations as f64)
}

#[cfg(feature = "cuda")]
fn benchmark_dynamic_tch(
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    use burn_tch::{LibTorch, LibTorchDevice};

    type B = LibTorch;

    let device = LibTorchDevice::Cuda(0);

    let a: Tensor<B, 2> = Tensor::random(Shape::new([m, k]), Distribution::Default, &device);
    let b: Tensor<B, 2> = Tensor::random(Shape::new([k, n]), Distribution::Default, &device);

    // Warmup
    for _ in 0..warmup {
        let _ = a.clone().matmul(b.clone());
    }

    // Benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let c = a.clone().matmul(b.clone());
        // Force synchronization by reading a value
        let _ = c.clone().into_data();

        // Check timeout after each iteration
        let elapsed = start.elapsed();
        if elapsed.as_secs_f64() > 1.0 {
            return Err(format!("TIMEOUT: Exceeded 1 second after {} iterations", i + 1).into());
        }
    }
    let elapsed = start.elapsed();

    Ok(elapsed.as_secs_f64() / iterations as f64)
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

        let result = match mode {
            BenchMode::DynamicCPU => benchmark_dynamic_ndarray(*m, *k, *n, warmup, iterations),
            #[cfg(feature = "wgpu")]
            BenchMode::DynamicWgpu => benchmark_dynamic_wgpu(*m, *k, *n, warmup, iterations),
            #[cfg(feature = "cuda")]
            BenchMode::DynamicTch => benchmark_dynamic_tch(*m, *k, *n, warmup, iterations),
        };

        match result {
            Ok(time) => {
                println!("{}x{}x{},time_ms={:.6}ms", m, k, n, time * 1000.0);
            },
            Err(e) if e.to_string().contains("TIMEOUT") => {
                println!("{}x{}x{},TIMEOUT", m, k, n);
                timed_out = true; // Mark as timed out to skip remaining benchmarks
            },
            Err(e) => {
                eprintln!("Error for {}x{}x{}: {}", m, k, n, e);
                println!("{}x{}x{},ERROR", m, k, n);
            },
        }
    }

    Ok(())
}

fn print_usage() {
    println!("Usage: burn <mode>");
    println!("\nAvailable modes:");
    println!("  dynamic-cpu     - Dynamic execution on CPU");
    #[cfg(feature = "wgpu")]
    println!("  dynamic-wgpu    - Dynamic execution on WGPU (Metal/Vulkan/CUDA)");
    #[cfg(feature = "cuda")]
    println!("  dynamic-tch     - Dynamic execution on LibTorch CUDA");
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

    let warmup = 5;
    let iterations = 10;

    run_benchmark(mode, &configs, warmup, iterations)?;

    Ok(())
}
