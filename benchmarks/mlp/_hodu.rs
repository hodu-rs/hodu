use hodu::prelude::*;
use std::env;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
enum BenchMode {
    DynamicCPU,
    #[cfg(feature = "cuda")]
    DynamicCUDA,
    #[cfg(feature = "metal")]
    DynamicMetal,
    StaticCPU,
    #[cfg(feature = "cuda")]
    StaticCUDA,
    #[cfg(feature = "metal")]
    StaticMetal,
    #[cfg(feature = "xla")]
    StaticXLACPU,
}

impl BenchMode {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "dynamic-cpu" => Some(Self::DynamicCPU),
            #[cfg(feature = "cuda")]
            "dynamic-cuda" => Some(Self::DynamicCUDA),
            #[cfg(feature = "metal")]
            "dynamic-metal" => Some(Self::DynamicMetal),
            "static-cpu" => Some(Self::StaticCPU),
            #[cfg(feature = "cuda")]
            "static-cuda" => Some(Self::StaticCUDA),
            #[cfg(feature = "metal")]
            "static-metal" => Some(Self::StaticMetal),
            #[cfg(feature = "xla")]
            "static-xla-cpu" => Some(Self::StaticXLACPU),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::DynamicCPU => "Dynamic CPU",
            #[cfg(feature = "cuda")]
            Self::DynamicCUDA => "Dynamic CUDA",
            #[cfg(feature = "metal")]
            Self::DynamicMetal => "Dynamic Metal",
            Self::StaticCPU => "Static CPU",
            #[cfg(feature = "cuda")]
            Self::StaticCUDA => "Static CUDA",
            #[cfg(feature = "metal")]
            Self::StaticMetal => "Static Metal",
            #[cfg(feature = "xla")]
            Self::StaticXLACPU => "Static XLA CPU",
        }
    }
}

// MLP Block with residual connection
struct MLPBlock {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    projection: Option<Linear>, // for residual connection when dimensions don't match
    gelu: Gelu,
}

impl MLPBlock {
    fn new(in_features: usize, hidden_features: usize, out_features: usize, dtype: DType) -> HoduResult<Self> {
        let fc1 = Linear::new(in_features, hidden_features, true, dtype)?;
        let fc2 = Linear::new(hidden_features, hidden_features, true, dtype)?;
        let fc3 = Linear::new(hidden_features, out_features, true, dtype)?;
        let gelu = Gelu::new();

        // Add projection if input/output dimensions don't match
        let projection = if in_features != out_features {
            Some(Linear::new(in_features, out_features, false, dtype)?)
        } else {
            None
        };

        Ok(Self {
            fc1,
            fc2,
            fc3,
            projection,
            gelu,
        })
    }

    fn forward(&self, x: &Tensor) -> HoduResult<Tensor> {
        // Save input for residual connection
        let identity = if let Some(ref proj) = self.projection {
            proj.forward(x)?
        } else {
            x.clone()
        };

        // Forward through layers
        let x = self.fc1.forward(x)?;
        let x = self.gelu.forward(&x)?;
        let x = self.fc2.forward(&x)?;
        let x = self.gelu.forward(&x)?;
        let x = self.fc3.forward(&x)?;

        // Residual connection
        x.add(&identity)
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
    match mode {
        BenchMode::DynamicCPU => {
            set_runtime_device(Device::CPU);
        },
        #[cfg(feature = "cuda")]
        BenchMode::DynamicCUDA => {
            set_runtime_device(Device::CUDA(0));
        },
        #[cfg(feature = "metal")]
        BenchMode::DynamicMetal => {
            set_runtime_device(Device::Metal);
        },
        _ => unreachable!(),
    }

    let mlp = MLPBlock::new(in_features, hidden_features, out_features, DType::F32)?;
    let x = Tensor::randn(&[batch_size, in_features], 0f32, 1.)?;

    // Warmup
    for _ in 0..warmup {
        let _ = mlp.forward(&x)?;
    }

    // Benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let _ = mlp.forward(&x)?;

        // Check timeout after each iteration
        let elapsed = start.elapsed();
        if elapsed.as_secs_f64() > 1.0 {
            return Err(format!("TIMEOUT: Exceeded 1 second after {} iterations", i + 1).into());
        }
    }
    let elapsed = start.elapsed();

    Ok(elapsed.as_secs_f64() / iterations as f64)
}

fn benchmark_static(
    mode: BenchMode,
    batch_size: usize,
    in_features: usize,
    hidden_features: usize,
    out_features: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let x_data = Tensor::randn(&[batch_size, in_features], 0f32, 1.)?;

    let mlp = MLPBlock::new(in_features, hidden_features, out_features, DType::F32)?;

    let builder = Builder::new("mlp_bench".to_string());
    builder.start()?;

    let x = Tensor::input("x", &[batch_size, in_features])?;

    // Build static graph
    let result = mlp.forward(&x)?;

    builder.add_output("result", result)?;
    builder.end()?;

    let mut script = builder.build()?;

    match mode {
        BenchMode::StaticCPU => {
            script.set_device(Device::CPU);
        },
        #[cfg(feature = "cuda")]
        BenchMode::StaticCUDA => {
            script.set_device(Device::CUDA(0));
        },
        #[cfg(feature = "metal")]
        BenchMode::StaticMetal => {
            script.set_device(Device::Metal);
        },
        #[cfg(feature = "xla")]
        BenchMode::StaticXLACPU => {
            script.set_device(Device::CPU);
            script.set_compiler(Compiler::XLA);
        },
        _ => unreachable!(),
    }

    script.set_input("x", x_data.clone());

    // Compile
    script.compile()?;

    // Warmup
    for _ in 0..warmup {
        let _ = script.run()?;
    }

    // Benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let _ = script.run()?;

        // Check timeout after each iteration
        let start_iteration = Instant::now();
        let elapsed_iteration = start_iteration.elapsed();
        if elapsed_iteration.as_secs_f64() > 1.0 {
            return Err(format!("TIMEOUT: Exceeded 1 second after {} iterations", i + 1).into());
        }
    }
    let elapsed = start.elapsed();

    Ok(elapsed.as_secs_f64() / iterations as f64)
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

        let result = match mode {
            BenchMode::DynamicCPU => benchmark_dynamic(
                mode,
                *batch_size,
                *in_features,
                *hidden_features,
                *out_features,
                warmup,
                iterations,
            ),
            #[cfg(feature = "cuda")]
            BenchMode::DynamicCUDA => benchmark_dynamic(
                mode,
                *batch_size,
                *in_features,
                *hidden_features,
                *out_features,
                warmup,
                iterations,
            ),
            #[cfg(feature = "metal")]
            BenchMode::DynamicMetal => benchmark_dynamic(
                mode,
                *batch_size,
                *in_features,
                *hidden_features,
                *out_features,
                warmup,
                iterations,
            ),
            BenchMode::StaticCPU => benchmark_static(
                mode,
                *batch_size,
                *in_features,
                *hidden_features,
                *out_features,
                warmup,
                iterations,
            ),
            #[cfg(feature = "cuda")]
            BenchMode::StaticCUDA => benchmark_static(
                mode,
                *batch_size,
                *in_features,
                *hidden_features,
                *out_features,
                warmup,
                iterations,
            ),
            #[cfg(feature = "metal")]
            BenchMode::StaticMetal => benchmark_static(
                mode,
                *batch_size,
                *in_features,
                *hidden_features,
                *out_features,
                warmup,
                iterations,
            ),
            #[cfg(feature = "xla")]
            BenchMode::StaticXLACPU => benchmark_static(
                mode,
                *batch_size,
                *in_features,
                *hidden_features,
                *out_features,
                warmup,
                iterations,
            ),
        };

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
                timed_out = true; // Mark as timed out to skip remaining benchmarks
            },
            Err(e) => return Err(e),
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
    println!("  static-cpu      - Static computation graph on CPU");
    #[cfg(feature = "cuda")]
    println!("  static-cuda     - Static computation graph on CUDA");
    #[cfg(feature = "metal")]
    println!("  static-metal    - Static computation graph on Metal");
    #[cfg(feature = "xla")]
    println!("  static-xla-cpu  - Static computation graph with XLA backend on CPU");
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
    // Tests 3-layer MLP with GELU activation and residual connections
    let configs = [
        (32, 256, 512, 256),    // Small: input projection not needed (same in/out)
        (64, 512, 1024, 512),   // Medium
        (128, 768, 2048, 1024), // Large: different in/out features (needs projection)
    ];

    let warmup = 5;
    let iterations = 10;

    run_benchmark(mode, &configs, warmup, iterations)?;

    Ok(())
}
