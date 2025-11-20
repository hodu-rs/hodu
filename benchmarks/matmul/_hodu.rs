use hodu::prelude::*;
use std::env;
use std::time::Instant;

// Memory measurement utilities
#[cfg(target_os = "linux")]
fn get_memory_usage_kb() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            return parts.get(1)?.parse().ok();
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn get_memory_usage_kb() -> Option<usize> {
    use std::process::Command;
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let rss_str = String::from_utf8(output.stdout).ok()?;
    rss_str.trim().parse().ok()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn get_memory_usage_kb() -> Option<usize> {
    None
}

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

fn benchmark_dynamic(
    mode: BenchMode,
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iterations: usize,
) -> Result<(f64, Option<usize>), Box<dyn std::error::Error>> {
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

    let a = Tensor::randn(&[m, k], 0f32, 1.)?;
    let b = Tensor::randn(&[k, n], 0f32, 1.)?;

    // Warmup
    for _ in 0..warmup {
        let _ = a.matmul(&b)?;
    }

    let mem_before = get_memory_usage_kb();

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

    let mem_after = get_memory_usage_kb();
    let mem_used = match (mem_before, mem_after) {
        (Some(before), Some(after)) => Some(after.saturating_sub(before)),
        _ => mem_after,
    };

    // Use trimmed mean (remove top/bottom 10%)
    Ok((trimmed_mean(&mut times, 0.1), mem_used))
}

fn benchmark_static(
    mode: BenchMode,
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iterations: usize,
) -> Result<(f64, Option<usize>), Box<dyn std::error::Error>> {
    let a_data = Tensor::randn(&[m, k], 0f32, 1.)?;
    let b_data = Tensor::randn(&[k, n], 0f32, 1.)?;

    let builder = Builder::new("matmul_bench".to_string());
    builder.start()?;

    let a = Tensor::input("a", &[m, k], DType::F32)?;
    let b = Tensor::input("b", &[k, n], DType::F32)?;
    let result = a.matmul(&b)?;

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

    script.set_input("a", a_data.clone());
    script.set_input("b", b_data.clone());

    // Compile
    script.compile()?;

    // Warmup
    for _ in 0..warmup {
        let _ = script.run()?;
    }

    let mem_before = get_memory_usage_kb();

    // Benchmark - collect individual iteration times
    let mut times = Vec::with_capacity(iterations);
    let bench_start = Instant::now();
    for i in 0..iterations {
        let start = Instant::now();
        let _ = script.run()?;
        times.push(start.elapsed().as_secs_f64());

        // Check timeout after each iteration
        let total_elapsed = bench_start.elapsed();
        if total_elapsed.as_secs_f64() > 10.0 {
            return Err(format!("TIMEOUT: Exceeded 10 seconds after {} iterations", i + 1).into());
        }
    }

    let mem_after = get_memory_usage_kb();
    let mem_used = match (mem_before, mem_after) {
        (Some(before), Some(after)) => Some(after.saturating_sub(before)),
        _ => mem_after,
    };

    // Use trimmed mean (remove top/bottom 10%)
    Ok((trimmed_mean(&mut times, 0.1), mem_used))
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
            #[cfg(feature = "cuda")]
            BenchMode::DynamicCUDA => benchmark_dynamic(mode, *m, *k, *n, warmup, iterations),
            #[cfg(feature = "metal")]
            BenchMode::DynamicMetal => benchmark_dynamic(mode, *m, *k, *n, warmup, iterations),
            BenchMode::StaticCPU => benchmark_static(mode, *m, *k, *n, warmup, iterations),
            #[cfg(feature = "cuda")]
            BenchMode::StaticCUDA => benchmark_static(mode, *m, *k, *n, warmup, iterations),
            #[cfg(feature = "metal")]
            BenchMode::StaticMetal => benchmark_static(mode, *m, *k, *n, warmup, iterations),
            #[cfg(feature = "xla")]
            BenchMode::StaticXLACPU => benchmark_static(mode, *m, *k, *n, warmup, iterations),
        };

        match result {
            Ok((time, mem)) => {
                if let Some(mem_kb) = mem {
                    println!("{}x{}x{},time_ms={:.6}ms,mem_kb={}", m, k, n, time * 1000.0, mem_kb);
                } else {
                    println!("{}x{}x{},time_ms={:.6}ms", m, k, n, time * 1000.0);
                }
            },
            Err(e) if e.to_string().contains("TIMEOUT") => {
                println!("{}x{}x{},TIMEOUT", m, k, n);
                timed_out = true;
            },
            Err(_) => {
                println!("{}x{}x{},ERROR", m, k, n);
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

    // Parse warmup and iterations from command line, with defaults
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

    let configs = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)];

    run_benchmark(mode, &configs, warmup, iterations)?;

    Ok(())
}
