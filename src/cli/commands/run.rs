use clap::Args;
use std::path::PathBuf;

#[derive(Args)]
pub struct RunArgs {
    /// Model file (.onnx, .hdss, etc.)
    pub model: PathBuf,

    /// Input tensors (name=path), can be repeated
    #[arg(short, long = "input", value_name = "NAME=PATH")]
    pub inputs: Vec<String>,

    /// Execution device (cpu, metal, cuda:0)
    #[arg(short, long, default_value = "cpu")]
    pub device: String,

    /// Backend plugin to use (auto-select if not specified)
    #[arg(long)]
    pub backend: Option<String>,

    /// Output format (pretty, json, hdt)
    #[arg(short, long, default_value = "pretty")]
    pub format: String,

    /// Output directory (for hdt format)
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// Benchmark mode (run N times)
    #[arg(long, value_name = "N")]
    pub benchmark: Option<Option<usize>>,

    /// Warmup runs before benchmark
    #[arg(long, default_value = "3")]
    pub warmup: usize,

    /// Profile execution
    #[arg(long)]
    pub profile: bool,

    /// Dry run (show what would be executed)
    #[arg(long)]
    pub dry_run: bool,
}

pub fn execute(args: RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.dry_run {
        println!("Model: {}", args.model.display());
        println!("Device: {}", args.device);
        println!("Backend: {}", args.backend.as_deref().unwrap_or("auto"));
        println!("Inputs: {:?}", args.inputs);
        println!("\nWould execute with above configuration.");
        return Ok(());
    }

    // TODO: Implement actual run logic
    // 1. Load model using FormatPlugin (based on extension)
    // 2. Parse inputs using FormatPlugin
    // 3. Select BackendPlugin (based on device)
    // 4. Execute model
    // 5. Output results

    println!("hodu run: not yet implemented");
    println!("Model: {}", args.model.display());
    println!("Device: {}", args.device);

    Ok(())
}
