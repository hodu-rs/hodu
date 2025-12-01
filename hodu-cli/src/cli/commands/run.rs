use crate::cli::plugin::{LoadedBackendPlugin, PluginRegistry};
use clap::Args;
use hodu_cli_plugin_sdk::Device;
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
    // Check if model file exists
    if !args.model.exists() {
        return Err(format!("Model file not found: {}", args.model.display()).into());
    }

    // Get model extension
    let extension = args
        .model
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    // Load plugin registry
    let registry_path = PluginRegistry::default_path()?;
    let registry = PluginRegistry::load(&registry_path)?;

    // Check for format plugin (for non-builtin formats)
    let format_plugin = match extension.as_deref() {
        Some("hdss") | Some("hdt") | Some("json") => {
            // Builtin formats
            None
        },
        Some(ext) => {
            let plugin = registry.find_format_by_extension(ext);
            if plugin.is_none() {
                return Err(friendly_format_error(ext, &registry).into());
            }
            plugin
        },
        None => {
            return Err("Model file has no extension. Cannot determine format.".into());
        },
    };

    // Parse device
    let device = parse_device(&args.device)?;

    // Find backend plugin
    let backend_plugin = find_backend_plugin(&args.backend, &device, &registry)?;

    if args.dry_run {
        println!("Model: {}", args.model.display());
        println!(
            "Format: {} ({})",
            extension.as_deref().unwrap_or("unknown"),
            format_plugin.map(|p| p.name.as_str()).unwrap_or("builtin")
        );
        println!("Device: {:?}", device);
        println!("Backend: {}", backend_plugin.name);
        println!("Inputs: {:?}", args.inputs);
        println!("\nWould execute with above configuration.");
        return Ok(());
    }

    // Load the backend plugin
    let plugins_dir = PluginRegistry::plugins_dir()?;
    let lib_path = plugins_dir.join(&backend_plugin.library);

    if !lib_path.exists() {
        return Err(format!(
            "Backend plugin library not found: {}\n\nTry reinstalling: hodu plugin install {} --force",
            lib_path.display(),
            backend_plugin.name
        )
        .into());
    }

    let loaded_backend = LoadedBackendPlugin::load(&lib_path)?;

    // TODO: Load model, parse inputs, execute
    // For now, just show that we successfully loaded the plugin
    println!("Model: {}", args.model.display());
    println!(
        "Backend: {} v{}",
        loaded_backend.plugin().name(),
        loaded_backend.plugin().version()
    );
    println!("Device: {:?}", device);
    println!();
    println!("Model execution not yet implemented.");
    println!("Plugin loaded successfully - ready for execution logic.");

    Ok(())
}

fn parse_device(device_str: &str) -> Result<Device, Box<dyn std::error::Error>> {
    let device_lower = device_str.to_lowercase();
    match device_lower.as_str() {
        "cpu" => Ok(Device::CPU),
        "metal" => Ok(Device::Metal),
        s if s.starts_with("cuda") => {
            // Parse cuda:N format
            if let Some(idx_str) = s.strip_prefix("cuda:") {
                let idx: usize = idx_str.parse().map_err(|_| format!("Invalid CUDA device index: {}", idx_str))?;
                Ok(Device::CUDA(idx))
            } else if s == "cuda" {
                Ok(Device::CUDA(0))
            } else {
                Err(format!("Invalid device format: {}", device_str).into())
            }
        },
        _ => Err(format!(
            "Unknown device: '{}'\n\nSupported devices:\n  cpu    - CPU execution\n  metal  - Apple Metal GPU\n  cuda   - NVIDIA CUDA GPU (cuda:0, cuda:1, ...)",
            device_str
        )
        .into()),
    }
}

fn find_backend_plugin<'a>(
    backend_name: &Option<String>,
    device: &Device,
    registry: &'a PluginRegistry,
) -> Result<&'a crate::cli::plugin::PluginEntry, Box<dyn std::error::Error>> {
    let device_str = format!("{:?}", device);

    // If backend explicitly specified, find it
    if let Some(name) = backend_name {
        // Try exact match first
        if let Some(plugin) = registry.find(name) {
            return Ok(plugin);
        }
        // Try with hodu-backend- prefix
        let prefixed = format!("hodu-backend-{}", name);
        if let Some(plugin) = registry.find(&prefixed) {
            return Ok(plugin);
        }
        // Try just the name for short aliases like "interp"
        for plugin in registry.backends() {
            if plugin.name.ends_with(&format!("-{}", name)) {
                return Ok(plugin);
            }
        }
        return Err(format!("Backend '{}' not found.\n\n{}", name, suggest_backends(registry)).into());
    }

    // Auto-select based on device
    if let Some(plugin) = registry.find_backend_by_device(&device_str) {
        return Ok(plugin);
    }

    // No plugin found for this device
    Err(friendly_backend_error(device, registry).into())
}

fn friendly_format_error(extension: &str, registry: &PluginRegistry) -> String {
    let mut msg = format!("No plugin found for '.{}' format.\n\n", extension);

    // Suggest installation
    let suggestions = match extension {
        "onnx" => Some(("onnx", "hodu plugin install onnx")),
        "npy" | "npz" => Some(("npy", "hodu plugin install npy")),
        "safetensors" => Some(("safetensors", "hodu plugin install safetensors")),
        "gguf" => Some(("gguf", "hodu plugin install gguf")),
        "pt" | "pth" => Some(("pytorch", "hodu plugin install pytorch")),
        _ => None,
    };

    if let Some((name, cmd)) = suggestions {
        msg.push_str(&format!("To add {} support:\n  {}\n\n", name, cmd));
    }

    // Show installed format plugins
    let formats: Vec<_> = registry.formats().collect();
    if !formats.is_empty() {
        msg.push_str("Installed format plugins:\n");
        for p in formats {
            let exts = p.capabilities.extensions.join(", ");
            msg.push_str(&format!("  {} - {}\n", p.name, exts));
        }
    } else {
        msg.push_str("No format plugins installed.\n");
    }

    msg.push_str("\nBuiltin formats: .hdss, .hdt, .json");
    msg
}

fn friendly_backend_error(device: &Device, registry: &PluginRegistry) -> String {
    let device_str = format!("{:?}", device);
    let mut msg = format!("No backend plugin found for device '{}'.\n\n", device_str);

    // Suggest installation based on device
    match device {
        Device::CPU => {
            msg.push_str("To install a CPU backend:\n");
            msg.push_str("  hodu plugin install interp-cpu  # Interpreter (no compilation)\n");
            msg.push_str("  hodu plugin install cpu         # AOT compiled (faster)\n\n");
        },
        Device::Metal => {
            msg.push_str("To install Metal backend:\n");
            msg.push_str("  hodu plugin install metal\n\n");
        },
        Device::CUDA(_) => {
            msg.push_str("To install CUDA backend:\n");
            msg.push_str("  hodu plugin install cuda\n\n");
        },
        _ => {},
    }

    msg.push_str(&suggest_backends(registry));
    msg
}

fn suggest_backends(registry: &PluginRegistry) -> String {
    let mut msg = String::new();

    let backends: Vec<_> = registry.backends().collect();
    if !backends.is_empty() {
        msg.push_str("Installed backend plugins:\n");
        for p in backends {
            let devices = p.capabilities.devices.join(", ");
            msg.push_str(&format!("  {} - devices: {}\n", p.name, devices));
        }
    } else {
        msg.push_str("No backend plugins installed.\n");
        msg.push_str("\nTo get started:\n");
        msg.push_str("  hodu plugin install interp-cpu  # CPU interpreter\n");
        msg.push_str("  hodu plugin install cpu         # CPU AOT compiled\n");
        msg.push_str("  hodu plugin install metal       # Apple Metal GPU\n");
        msg.push_str("  hodu plugin install cuda        # NVIDIA CUDA GPU\n");
    }

    msg
}
