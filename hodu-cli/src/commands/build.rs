//! Build command - AOT compile models using backend plugins
//!
//! This command uses JSON-RPC based plugins to compile models.

use crate::plugins::{PluginManager, PluginRegistry};
use clap::Args;
use hodu_plugin_sdk::{BuildTarget, Snapshot};
use std::path::{Path, PathBuf};

#[derive(Args)]
pub struct BuildArgs {
    /// Model file (.onnx, .hdss, etc.)
    pub model: PathBuf,

    /// Output file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Target triple (default: current system)
    #[arg(short, long)]
    pub target: Option<String>,

    /// Target device (cpu, metal, cuda::0)
    #[arg(short, long, default_value = "cpu")]
    pub device: String,

    /// Output format (sharedlib, staticlib, object, metallib, ptx)
    #[arg(short, long)]
    pub format: Option<String>,

    /// Optimization level (0-3)
    #[arg(short = 'O', long, default_value = "2")]
    pub opt_level: u8,

    /// Generate standalone executable
    #[arg(long)]
    pub standalone: bool,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn execute(args: BuildArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.model.exists() {
        return Err(format!("Model file not found: {}", args.model.display()).into());
    }

    let extension = args
        .model
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    let registry_path = PluginRegistry::default_path()?;
    let registry = PluginRegistry::load(&registry_path)?;

    // Find model format plugin if needed
    let format_plugin = match extension.as_deref() {
        Some("hdss") => None,
        Some(ext) => {
            let plugin = registry.find_model_format_by_extension(ext);
            if plugin.is_none() {
                return Err(format!("No model format plugin found for .{}", ext).into());
            }
            plugin
        },
        None => return Err("Model file has no extension".into()),
    };

    // Normalize device (lowercase)
    let device = args.device.to_lowercase();

    // Find builder backend plugin
    let backend_entry = find_builder_backend(&device, &registry)?;

    if args.verbose {
        println!("Model: {}", args.model.display());
        println!("Output: {}", args.output.display());
        println!("Backend: {} v{}", backend_entry.name, backend_entry.version);
        println!("Device: {}", device);
    }

    // Create plugin manager
    let mut manager = PluginManager::new()?;

    // Load model (using format plugin if needed)
    let snapshot_path = if let Some(format_entry) = format_plugin {
        let client = manager.get_plugin(&format_entry.name)?;
        let result = client.load_model(args.model.to_str().unwrap())?;
        PathBuf::from(result.snapshot_path)
    } else {
        args.model.clone()
    };

    // Load snapshot for validation
    let _snapshot = Snapshot::load(&snapshot_path)?;

    // Determine build format from arg or output extension
    let format = determine_format(&args.format, &args.output);

    // Determine build target
    let build_target = match &args.target {
        Some(triple) => BuildTarget::new(triple.clone(), device.clone()),
        None => BuildTarget::host(device.clone()),
    };

    if args.verbose {
        println!("Building...");
    }

    // Call backend.build via JSON-RPC
    let client = manager.get_plugin(&backend_entry.name)?;
    client.build(
        snapshot_path.to_str().unwrap(),
        &build_target.triple,
        &build_target.device,
        &format,
        args.output.to_str().unwrap(),
    )?;

    println!("Built: {}", args.output.display());

    Ok(())
}

fn find_builder_backend<'a>(
    device: &str,
    registry: &'a PluginRegistry,
) -> Result<&'a crate::plugins::PluginEntry, Box<dyn std::error::Error>> {
    for plugin in registry.backends() {
        if plugin.capabilities.builder == Some(true)
            && plugin.capabilities.devices.iter().any(|d| d.to_lowercase() == device)
        {
            return Ok(plugin);
        }
    }

    Err(format!(
        "No builder backend found for device '{}'\n\nInstalled backends:\n{}",
        device,
        registry
            .backends()
            .map(|p| format!(
                "  {} - devices: {}, builder: {}",
                p.name,
                p.capabilities.devices.join(", "),
                p.capabilities.builder.unwrap_or(false)
            ))
            .collect::<Vec<_>>()
            .join("\n")
    )
    .into())
}

fn determine_format(format_arg: &Option<String>, output: &Path) -> String {
    if let Some(fmt) = format_arg {
        return fmt.to_lowercase();
    }

    let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "so" | "dylib" | "dll" => "sharedlib".to_string(),
        "a" | "lib" => "staticlib".to_string(),
        "o" | "obj" => "object".to_string(),
        "metallib" => "metallib".to_string(),
        "ptx" => "ptx".to_string(),
        "cubin" => "cubin".to_string(),
        "ll" => "llvmir".to_string(),
        "bc" => "llvmbitcode".to_string(),
        "wgsl" => "wgsl".to_string(),
        "spv" => "spirv".to_string(),
        "" => "executable".to_string(),
        _ => "sharedlib".to_string(),
    }
}
