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
}

impl BenchMode {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "dynamic-cpu" => Some(Self::DynamicCPU),
            #[cfg(feature = "cuda")]
            "dynamic-cuda" => Some(Self::DynamicCUDA),
            #[cfg(feature = "metal")]
            "dynamic-metal" => Some(Self::DynamicMetal),
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
        }
    }
}

// Simple CNN model for image classification
struct SimpleCNN {
    conv1: Conv2D,
    conv2: Conv2D,
    fc1: Linear,
    fc2: Linear,
    relu: ReLU,
    pool: MaxPool2D,
}

impl SimpleCNN {
    fn new(dtype: DType) -> HoduResult<Self> {
        // Conv layers: 3 -> 32 -> 64 channels
        let conv1 = Conv2D::new(3, 32, 3, 1, 1, 1, true, dtype)?;
        let conv2 = Conv2D::new(32, 64, 3, 1, 1, 1, true, dtype)?;

        // FC layers: 64*8*8 -> 128 -> 10
        let fc1 = Linear::new(64 * 8 * 8, 128, true, dtype)?;
        let fc2 = Linear::new(128, 10, true, dtype)?;

        let relu = ReLU::new();
        let pool = MaxPool2D::new(2, 2, 0);

        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            relu,
            pool,
        })
    }

    fn forward(&self, x: &Tensor) -> HoduResult<Tensor> {
        // Input: [batch, 3, 32, 32]
        let x = self.conv1.forward(x)?; // [batch, 32, 32, 32]
        let x = self.relu.forward(&x)?;
        let x = self.pool.forward(&x)?; // [batch, 32, 16, 16]

        let x = self.conv2.forward(&x)?; // [batch, 64, 16, 16]
        let x = self.relu.forward(&x)?;
        let x = self.pool.forward(&x)?; // [batch, 64, 8, 8]

        // Flatten
        let batch_size = x.shape()[0];
        let x = x.reshape(&[batch_size, 64 * 8 * 8])?;

        let x = self.fc1.forward(&x)?; // [batch, 128]
        let x = self.relu.forward(&x)?;

        self.fc2.forward(&x) // [batch, 10]
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
}

fn benchmark_training(
    mode: BenchMode,
    batch_size: usize,
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
    }

    let mut model = SimpleCNN::new(DType::F32)?;

    // Optimizer
    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    // Loss function
    let loss_fn = CrossEntropyLoss::new();

    // Random data
    let x = Tensor::randn(&[batch_size, 3, 32, 32], 0f32, 1.)?;
    x.requires_grad()?;

    // Create random target indices (0-9)
    let target_data: Vec<i32> = (0..batch_size).map(|i| ((i * 7) % 10) as i32).collect();
    let target = Tensor::from_slice(target_data, &[batch_size])?;

    // Warmup
    for _ in 0..warmup {
        let logits = model.forward(&x)?;
        let loss = loss_fn.forward((&logits, &target))?;
        loss.backward()?;
        let mut params = model.parameters();
        optimizer.step(&mut params)?;
        optimizer.zero_grad(&mut params)?;
    }

    let mem_before = get_memory_usage_kb();

    // Benchmark full training step (forward + backward + optimizer step)
    let mut step_times = Vec::with_capacity(iterations);
    let bench_start = Instant::now();
    for i in 0..iterations {
        let start = Instant::now();
        let logits = model.forward(&x)?;
        let loss = loss_fn.forward((&logits, &target))?;
        loss.backward()?;
        let mut params = model.parameters();
        optimizer.step(&mut params)?;
        optimizer.zero_grad(&mut params)?;
        step_times.push(start.elapsed().as_secs_f64());

        if bench_start.elapsed().as_secs_f64() > 60.0 {
            return Err(format!("TIMEOUT: Exceeded 60 seconds after {} iterations", i + 1).into());
        }
    }

    let mem_after = get_memory_usage_kb();
    let mem_used = match (mem_before, mem_after) {
        (Some(before), Some(after)) => Some(after.saturating_sub(before)),
        _ => mem_after,
    };

    Ok((trimmed_mean(&mut step_times, 0.1), mem_used))
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
            Ok((step_time, mem)) => {
                if let Some(mem_kb) = mem {
                    println!("{},time_ms={:.6},mem_kb={}", batch_size, step_time * 1000.0, mem_kb);
                } else {
                    println!("{},time_ms={:.6}", batch_size, step_time * 1000.0);
                }
            },
            Err(e) if e.to_string().contains("TIMEOUT") => {
                println!("{},TIMEOUT", batch_size);
                timed_out = true;
            },
            Err(e) => {
                eprintln!("Error for batch_size {}: {}", batch_size, e);
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
