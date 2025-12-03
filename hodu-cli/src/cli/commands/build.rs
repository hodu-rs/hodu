use crate::cli::plugin::{LoadedBackendPlugin, LoadedFormatPlugin, PluginRegistry};
use clap::Args;
use hodu_plugin_sdk::{BuildFormat, BuildTarget, Device, Snapshot};
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

    /// Target device (cpu, metal, cuda)
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
    let plugins_dir = PluginRegistry::plugins_dir()?;

    // Find format plugin if needed
    let format_plugin_lib = match extension.as_deref() {
        Some("hdss") => None,
        Some(ext) => {
            let plugin = registry.find_format_by_extension(ext);
            if plugin.is_none() {
                return Err(format!("No format plugin found for .{}", ext).into());
            }
            plugin.map(|p| p.library.clone())
        },
        None => return Err("Model file has no extension".into()),
    };

    // Parse device
    let device = parse_device(&args.device)?;

    // Find builder backend plugin
    let backend_entry = find_builder_backend(&device, &registry)?;

    if args.verbose {
        println!("Model: {}", args.model.display());
        println!("Output: {}", args.output.display());
        println!("Backend: {} v{}", backend_entry.name, backend_entry.version);
        println!("Device: {:?}", device);
    }

    // Load model
    let snapshot = load_model(&args.model, format_plugin_lib.as_ref(), &plugins_dir)?;

    // Determine build format
    let format = determine_format(&args.format, &args.output)?;

    // Determine build target
    let build_target = match &args.target {
        Some(triple) => BuildTarget::new(triple.clone(), device),
        None => BuildTarget::host(device),
    };

    // Load backend plugin
    let lib_path = plugins_dir.join(&backend_entry.library);
    let loaded_backend = LoadedBackendPlugin::load(&lib_path)?;

    if args.verbose {
        println!("Building...");
    }

    // Call build
    loaded_backend
        .plugin()
        .build(&snapshot, &build_target, format, &args.output)
        .map_err(|e| format!("Build failed: {}", e))?;

    println!("Built: {}", args.output.display());

    Ok(())
}

fn load_model(
    path: &Path,
    format_plugin_lib: Option<&String>,
    plugins_dir: &Path,
) -> Result<Snapshot, Box<dyn std::error::Error>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "hdss" => Snapshot::load(path).map_err(|e| format!("Failed to load snapshot: {}", e).into()),
        _ => {
            if let Some(lib_name) = format_plugin_lib {
                let lib_path = plugins_dir.join(lib_name);
                let loaded_format = LoadedFormatPlugin::load(&lib_path)?;
                loaded_format
                    .plugin()
                    .load_model(path)
                    .map_err(|e| format!("Format plugin failed: {}", e).into())
            } else {
                Err(format!("No format plugin for .{}", ext).into())
            }
        },
    }
}

fn parse_device(device_str: &str) -> Result<Device, Box<dyn std::error::Error>> {
    match device_str.to_lowercase().as_str() {
        "cpu" => Ok(Device::CPU),
        "metal" => Ok(Device::Metal),
        s if s.starts_with("cuda") => {
            if let Some(idx_str) = s.strip_prefix("cuda:") {
                let idx: usize = idx_str.parse()?;
                Ok(Device::CUDA(idx))
            } else {
                Ok(Device::CUDA(0))
            }
        },
        _ => Err(format!("Unknown device: {}", device_str).into()),
    }
}

fn find_builder_backend<'a>(
    device: &Device,
    registry: &'a PluginRegistry,
) -> Result<&'a crate::cli::plugin::PluginEntry, Box<dyn std::error::Error>> {
    let device_str = format!("{:?}", device);

    // Find backend with builder capability for this device
    for plugin in registry.backends() {
        if plugin.capabilities.builder == Some(true) && plugin.capabilities.devices.iter().any(|d| d == &device_str) {
            return Ok(plugin);
        }
    }

    Err(format!(
        "No builder backend found for device '{}'\n\nInstalled backends:\n{}",
        device_str,
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

fn determine_format(format_arg: &Option<String>, output: &Path) -> Result<BuildFormat, Box<dyn std::error::Error>> {
    if let Some(fmt) = format_arg {
        return match fmt.to_lowercase().as_str() {
            "sharedlib" | "so" | "dylib" | "dll" => Ok(BuildFormat::SharedLib),
            "staticlib" | "a" | "lib" => Ok(BuildFormat::StaticLib),
            "object" | "o" => Ok(BuildFormat::Object),
            "executable" | "exe" => Ok(BuildFormat::Executable),
            "metallib" => Ok(BuildFormat::MetalLib),
            "ptx" => Ok(BuildFormat::PTX),
            _ => Err(format!("Unknown format: {}", fmt).into()),
        };
    }

    // Infer from output extension
    let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "so" | "dylib" | "dll" => Ok(BuildFormat::SharedLib),
        "a" | "lib" => Ok(BuildFormat::StaticLib),
        "o" | "obj" => Ok(BuildFormat::Object),
        "metallib" => Ok(BuildFormat::MetalLib),
        "ptx" => Ok(BuildFormat::PTX),
        "" => Ok(BuildFormat::Executable),
        _ => Ok(BuildFormat::SharedLib),
    }
}
