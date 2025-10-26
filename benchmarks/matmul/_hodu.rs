use hodu::prelude::*;
use std::env;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
enum BenchMode {
    DynamicCPU,
    #[cfg(feature = "metal")]
    DynamicMetal,
    StaticCPU,
    #[cfg(feature = "metal")]
    StaticMetal,
    #[cfg(feature = "xla")]
    StaticXLA,
}

impl BenchMode {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "dynamic-cpu" => Some(Self::DynamicCPU),
            #[cfg(feature = "metal")]
            "dynamic-metal" => Some(Self::DynamicMetal),
            "static-cpu" => Some(Self::StaticCPU),
            #[cfg(feature = "metal")]
            "static-metal" => Some(Self::StaticMetal),
            #[cfg(feature = "xla")]
            "static-xla" => Some(Self::StaticXLA),
            _ => None,
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::DynamicCPU => "Dynamic CPU",
            #[cfg(feature = "metal")]
            Self::DynamicMetal => "Dynamic Metal",
            Self::StaticCPU => "Static CPU",
            #[cfg(feature = "metal")]
            Self::StaticMetal => "Static Metal",
            #[cfg(feature = "xla")]
            Self::StaticXLA => "Static XLA",
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
    match mode {
        BenchMode::DynamicCPU => {
            set_runtime_device(Device::CPU);
        },
        #[cfg(feature = "metal")]
        BenchMode::DynamicMetal => {
            set_runtime_device(Device::Metal);
        },
        _ => unreachable!(),
    }

    let a = Tensor::randn(&[m, k], 0f32, 1.)?;
    let b = Tensor::randn(&[k, n], 0f32, 1.)?;

    // Warmup
    for _ in 0..warmup {
        let _ = a.matmul(&b)?;
    }

    // Benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let _ = a.matmul(&b)?;

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
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let a_data = Tensor::randn(&[m, k], 0f32, 1.)?;
    let b_data = Tensor::randn(&[k, n], 0f32, 1.)?;

    let builder = Builder::new("matmul_bench".to_string());
    builder.start()?;

    let a = Tensor::input("a", &[m, k])?;
    let b = Tensor::input("b", &[k, n])?;
    let result = a.matmul(&b)?;

    builder.add_output("result", result)?;
    builder.end()?;

    let mut script = builder.build()?;

    match mode {
        BenchMode::StaticCPU => {
            script.set_device(Device::CPU);
        },
        #[cfg(feature = "metal")]
        BenchMode::StaticMetal => {
            script.set_device(Device::Metal);
        },
        #[cfg(feature = "xla")]
        BenchMode::StaticXLA => {
            script.set_backend(Backend::XLA);
        },
        _ => unreachable!(),
    }

    script.add_input("a", a_data.clone());
    script.add_input("b", b_data.clone());

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
            BenchMode::DynamicCPU => benchmark_dynamic(mode, *m, *k, *n, warmup, iterations),
            #[cfg(feature = "metal")]
            BenchMode::DynamicMetal => benchmark_dynamic(mode, *m, *k, *n, warmup, iterations),
            BenchMode::StaticCPU => benchmark_static(mode, *m, *k, *n, warmup, iterations),
            #[cfg(feature = "metal")]
            BenchMode::StaticMetal => benchmark_static(mode, *m, *k, *n, warmup, iterations),
            #[cfg(feature = "xla")]
            BenchMode::StaticXLA => benchmark_static(mode, *m, *k, *n, warmup, iterations),
        };

        match result {
            Ok(time) => {
                println!("{}x{}x{},time_ms={:.6}ms", m, k, n, time * 1000.0);
            },
            Err(e) if e.to_string().contains("TIMEOUT") => {
                println!("{}x{}x{},TIMEOUT", m, k, n);
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
    #[cfg(feature = "metal")]
    println!("  dynamic-metal   - Dynamic execution on Metal");
    println!("  static-cpu      - Static computation graph on CPU");
    #[cfg(feature = "metal")]
    println!("  static-metal    - Static computation graph on Metal");
    #[cfg(feature = "xla")]
    println!("  static-xla      - Static computation graph with XLA backend");
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

    let configs = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ];

    let warmup = 5;
    let iterations = 10;

    run_benchmark(mode, &configs, warmup, iterations)?;

    Ok(())
}
