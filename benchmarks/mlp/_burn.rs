use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
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

// MLP Block implementation
struct MLPBlock<B: Backend> {
    fc1_w: Tensor<B, 2>,
    fc1_b: Tensor<B, 1>,
    fc2_w: Tensor<B, 2>,
    fc2_b: Tensor<B, 1>,
    fc3_w: Tensor<B, 2>,
    fc3_b: Tensor<B, 1>,
    projection: Option<Tensor<B, 2>>,
}

impl<B: Backend> MLPBlock<B> {
    fn new(in_features: usize, hidden_features: usize, out_features: usize, device: &B::Device) -> Self {
        // Xavier initialization
        let k_in = (1.0 / in_features as f32).sqrt();
        let k_hidden = (1.0 / hidden_features as f32).sqrt();

        let fc1_w = Tensor::random(
            Shape::new([in_features, hidden_features]),
            Distribution::Default,
            device,
        ) * k_in;
        let fc1_b = Tensor::zeros(Shape::new([hidden_features]), device);

        let fc2_w = Tensor::random(
            Shape::new([hidden_features, hidden_features]),
            Distribution::Default,
            device,
        ) * k_hidden;
        let fc2_b = Tensor::zeros(Shape::new([hidden_features]), device);

        let fc3_w = Tensor::random(
            Shape::new([hidden_features, out_features]),
            Distribution::Default,
            device,
        ) * k_hidden;
        let fc3_b = Tensor::zeros(Shape::new([out_features]), device);

        let projection = if in_features != out_features {
            Some(Tensor::random(Shape::new([in_features, out_features]), Distribution::Default, device) * k_in)
        } else {
            None
        };

        Self {
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            fc3_w,
            fc3_b,
            projection,
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        use burn::tensor::activation;

        // Save input for residual connection
        let identity = if let Some(ref proj) = self.projection {
            x.clone().matmul(proj.clone())
        } else {
            x.clone()
        };

        // Forward through layers - Burn uses activation module
        let mut out = x.matmul(self.fc1_w.clone());
        // Broadcast add bias: unsqueeze to [1, hidden] then broadcast
        let bias1 = self.fc1_b.clone().unsqueeze_dim(0);
        out = out + bias1;
        out = activation::gelu(out);

        out = out.matmul(self.fc2_w.clone());
        let bias2 = self.fc2_b.clone().unsqueeze_dim(0);
        out = out + bias2;
        out = activation::gelu(out);

        out = out.matmul(self.fc3_w.clone());
        let bias3 = self.fc3_b.clone().unsqueeze_dim(0);
        out = out + bias3;

        // Residual connection
        out + identity
    }
}

fn benchmark_dynamic_ndarray(
    batch_size: usize,
    in_features: usize,
    hidden_features: usize,
    out_features: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    use burn_ndarray::{NdArray, NdArrayDevice};

    type B = NdArray<f32>;

    let device = NdArrayDevice::Cpu;

    let mlp = MLPBlock::<B>::new(in_features, hidden_features, out_features, &device);
    let x: Tensor<B, 2> = Tensor::random(Shape::new([batch_size, in_features]), Distribution::Default, &device);

    // Warmup
    for _ in 0..warmup {
        let _ = mlp.forward(x.clone());
    }

    // Benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let _ = mlp.forward(x.clone());

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
    batch_size: usize,
    in_features: usize,
    hidden_features: usize,
    out_features: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    use burn_wgpu::{Wgpu, WgpuDevice};

    type B = Wgpu;

    let device = WgpuDevice::default();

    let mlp = MLPBlock::<B>::new(in_features, hidden_features, out_features, &device);
    let x: Tensor<B, 2> = Tensor::random(Shape::new([batch_size, in_features]), Distribution::Default, &device);

    // Warmup
    for _ in 0..warmup {
        let _ = mlp.forward(x.clone());
    }

    // Benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let c = mlp.forward(x.clone());
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
    batch_size: usize,
    in_features: usize,
    hidden_features: usize,
    out_features: usize,
    warmup: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    use burn_tch::{LibTorch, LibTorchDevice};

    type B = LibTorch;

    let device = LibTorchDevice::Cuda(0);

    let mlp = MLPBlock::<B>::new(in_features, hidden_features, out_features, &device);
    let x: Tensor<B, 2> = Tensor::random(Shape::new([batch_size, in_features]), Distribution::Default, &device);

    // Warmup
    for _ in 0..warmup {
        let _ = mlp.forward(x.clone());
    }

    // Benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let c = mlp.forward(x.clone());
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
            BenchMode::DynamicCPU => benchmark_dynamic_ndarray(
                *batch_size,
                *in_features,
                *hidden_features,
                *out_features,
                warmup,
                iterations,
            ),
            #[cfg(feature = "wgpu")]
            BenchMode::DynamicWgpu => benchmark_dynamic_wgpu(
                *batch_size,
                *in_features,
                *hidden_features,
                *out_features,
                warmup,
                iterations,
            ),
            #[cfg(feature = "cuda")]
            BenchMode::DynamicTch => benchmark_dynamic_tch(
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
                timed_out = true;
            },
            Err(e) => return Err(e),
        }
    }

    Ok(())
}

fn print_usage() {
    println!("Usage: burn <mode>");
    println!("\nAvailable modes:");
    println!("  dynamic-cpu     - Dynamic execution on CPU");
    #[cfg(feature = "wgpu")]
    println!("  dynamic-wgpu    - Dynamic execution on WGPU");
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

    // MLP configs: (batch_size, in_features, hidden_features, out_features)
    let configs = [(32, 256, 512, 256), (64, 512, 1024, 512), (128, 768, 2048, 1024)];

    let warmup = 5;
    let iterations = 10;

    run_benchmark(mode, &configs, warmup, iterations)?;

    Ok(())
}
