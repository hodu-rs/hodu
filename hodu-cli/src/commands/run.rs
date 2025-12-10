//! Run command - execute model inference using plugins
//!
//! This command uses JSON-RPC based plugins to load models and run inference.

use crate::output;
use crate::plugins::{PluginManager, PluginRegistry};
use crate::tensor::{load_tensor_file, save_outputs};
use clap::Args;
use hodu_core::format::hdt;
use hodu_core::snapshot::Snapshot;
use hodu_core::tensor::Tensor;
use hodu_core::types::{DType, Device as CoreDevice, Shape};
use hodu_plugin::rpc::TensorInput;
use hodu_plugin::{Device, PluginDType, TensorData};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Convert a path to a string, returning an error if the path is not valid UTF-8
fn path_to_str(path: &Path) -> Result<&str, Box<dyn std::error::Error>> {
    path.to_str()
        .ok_or_else(|| format!("Invalid UTF-8 in path: {}", path.display()).into())
}

#[derive(Args)]
pub struct RunArgs {
    /// Model file (.onnx, .hdss, etc.)
    pub model: PathBuf,

    /// Input tensor (name=path), can be repeated
    #[arg(short, long = "input", value_name = "NAME=PATH")]
    pub input: Vec<String>,

    /// Input tensors (comma-separated: a=path,b=path)
    #[arg(long = "inputs", value_name = "INPUTS", value_delimiter = ',')]
    pub inputs: Vec<String>,

    /// Execution device (cpu, metal, cuda::0)
    #[arg(short, long, default_value = "cpu")]
    pub device: String,

    /// Backend plugin to use (auto-select if not specified)
    #[arg(long)]
    pub backend: Option<String>,

    /// Output format (pretty, json)
    #[arg(short, long, default_value = "pretty")]
    pub format: String,

    /// Save outputs to directory
    #[arg(long)]
    pub save: Option<PathBuf>,

    /// Save format (hdt, json, or format plugin extension)
    #[arg(long, default_value = "hdt")]
    pub save_format: String,

    /// Dry run (show what would be executed)
    #[arg(long)]
    pub dry_run: bool,

    /// Suppress all output
    #[arg(short, long)]
    pub quiet: bool,

    /// Timeout in seconds for plugin operations (default: 300)
    #[arg(long, value_name = "SECONDS")]
    pub timeout: Option<u64>,
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

    // Check for model format plugin (for non-builtin formats)
    let format_plugin = match extension.as_deref() {
        Some("hdss") | Some("hdt") | Some("json") => {
            // Builtin formats
            None
        },
        Some(ext) => {
            let plugin = registry.find_model_format_by_extension(ext);
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

    // Combine --input and --inputs arguments
    let all_inputs: Vec<String> = args.input.iter().chain(args.inputs.iter()).cloned().collect();

    if args.dry_run {
        println!(
            "Model format: {} ({})",
            extension.as_deref().unwrap_or("unknown"),
            format_plugin
                .map(|p| format!("{} {}", p.name, p.version))
                .unwrap_or_else(|| "builtin".to_string())
        );
        for input_arg in &all_inputs {
            if let Some((name, path)) = input_arg.split_once('=') {
                let ext = std::path::Path::new(path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown");
                println!("Input {}: {} (builtin)", name, ext);
            }
        }
        println!(
            "Backend: {} ({} {})",
            device, backend_plugin.name, backend_plugin.version
        );
        println!();
        println!("Would execute with above configuration.");
        return Ok(());
    }

    // Create plugin manager with optional timeout
    let mut manager = match args.timeout {
        Some(secs) => PluginManager::with_timeout(secs)?,
        None => PluginManager::new()?,
    };

    // Load model (using format plugin if needed)
    let model_name = args.model.file_name().unwrap_or_default().to_string_lossy();
    let snapshot_path = if let Some(format_entry) = format_plugin {
        // Use format plugin to convert to snapshot
        output::loading(&format!("{}", model_name));
        let client = manager.get_plugin(&format_entry.name)?;
        let result = client.load_model(path_to_str(&args.model)?)?;
        PathBuf::from(result.snapshot_path)
    } else {
        // Builtin format - model is already a snapshot
        args.model.clone()
    };

    // Load the snapshot
    let snapshot = Snapshot::load(&snapshot_path)?;

    // Parse input tensors
    let inputs = parse_inputs(&all_inputs, &snapshot)?;

    // Save input tensors to temp files and create TensorInput refs
    let mut input_refs = Vec::new();
    for (name, tensor_data) in &inputs {
        let temp_path = std::env::temp_dir().join(format!("hodu_input_{}_{}.hdt", name, std::process::id()));
        save_tensor_data(tensor_data, &temp_path)?;
        input_refs.push(TensorInput {
            name: name.clone(),
            path: temp_path.to_string_lossy().to_string(),
        });
    }

    // Run inference using backend plugin
    // First, spawn the backend plugin and get cancellation handle
    let _ = manager.get_plugin(&backend_plugin.name)?; // Ensure plugin is running
    let cancel_handle = manager.get_cancellation_handle(&backend_plugin.name);

    // Set up Ctrl+C handler for cancellation
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_clone = Arc::clone(&cancelled);
    if let Some(handle) = cancel_handle {
        ctrlc::set_handler(move || {
            cancelled_clone.store(true, Ordering::SeqCst);
            eprintln!("\nCancelling...");
            let _ = handle.cancel();
        })
        .ok();
    }

    // Compute cache key from snapshot content
    let snapshot_content = std::fs::read(&snapshot_path).map_err(|e| format!("Failed to read snapshot: {}", e))?;
    let mut hasher = Sha256::new();
    hasher.update(&snapshot_content);
    hasher.update(current_target_triple().as_bytes());
    let snapshot_hash = hex::encode(hasher.finalize());

    // Determine library extension and cache path
    let lib_ext = if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };

    let cache_dir = dirs::home_dir()
        .ok_or("Could not determine home directory")?
        .join(".hodu")
        .join("cache")
        .join(&backend_plugin.name);
    std::fs::create_dir_all(&cache_dir)?;
    let library_path = cache_dir.join(format!("{}.{}", snapshot_hash, lib_ext));

    // Build if not cached
    let backend_client = manager.get_plugin(&backend_plugin.name)?;
    let start = std::time::Instant::now();
    if !library_path.exists() {
        output::compiling(&format!("{} ({})", model_name, device));
        backend_client.build(
            path_to_str(&snapshot_path)?,
            &current_target_triple(),
            &device,
            "sharedlib",
            path_to_str(&library_path)?,
        )?;
    } else {
        output::cached(&format!("{}", model_name));
    }

    // Run with cached library
    output::running(&format!("{} ({})", model_name, device));
    let result = backend_client.run(
        path_to_str(&library_path)?,
        path_to_str(&snapshot_path)?,
        &device,
        input_refs,
    )?;
    let duration = start.elapsed().as_secs_f64();
    if !args.quiet {
        output::finished(&format!("inference in {}", output::format_duration(duration)));
    }

    // Check if was cancelled
    if cancelled.load(Ordering::SeqCst) {
        return Err("Operation cancelled by user".into());
    }

    // Load output tensors from paths
    let mut outputs: HashMap<String, TensorData> = HashMap::new();
    for output_ref in result.outputs {
        let tensor_data = load_tensor_data(&output_ref.path)?;
        outputs.insert(output_ref.name, tensor_data);
    }

    // Save outputs if requested
    if let Some(save_dir) = &args.save {
        save_outputs(&outputs, save_dir, &args.save_format)?;
    }

    // Output results
    if !args.quiet {
        output_results(&outputs, &args)?;
    }

    Ok(())
}

fn parse_inputs(
    input_args: &[String],
    snapshot: &Snapshot,
) -> Result<HashMap<String, TensorData>, Box<dyn std::error::Error>> {
    let mut inputs = HashMap::new();

    for arg in input_args {
        let parts: Vec<&str> = arg.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid input format: '{}'. Expected: name=path", arg).into());
        }

        let name = parts[0];
        let path = expand_path(parts[1]);

        if !path.exists() {
            return Err(format!("Input file not found: {}", path.display()).into());
        }

        let input_spec = snapshot.inputs.iter().find(|i| i.name == name).ok_or_else(|| {
            format!(
                "Unknown input '{}'. Available: {:?}",
                name,
                snapshot.inputs.iter().map(|i| &i.name).collect::<Vec<_>>()
            )
        })?;

        let tensor_data = load_tensor_file(&path, input_spec.shape.dims(), core_dtype_to_plugin(input_spec.dtype))?;
        inputs.insert(name.to_string(), tensor_data);
    }

    for input_spec in &snapshot.inputs {
        if !inputs.contains_key(&input_spec.name) {
            return Err(format!(
                "Missing required input '{}' ({:?}, {:?})",
                input_spec.name,
                input_spec.shape.dims(),
                input_spec.dtype
            )
            .into());
        }
    }

    Ok(inputs)
}

fn output_results(outputs: &HashMap<String, TensorData>, args: &RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.format.as_str() {
        "json" => {
            let json_outputs: HashMap<String, serde_json::Value> = outputs
                .iter()
                .map(|(name, data)| {
                    (
                        name.clone(),
                        serde_json::json!({
                            "shape": data.shape,
                            "dtype": data.dtype.name(),
                        }),
                    )
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json_outputs)?);
        },
        _ => {
            let mut names: Vec<_> = outputs.keys().collect();
            names.sort();
            for name in names {
                let data = &outputs[name];
                let dtype = plugin_dtype_to_core(data.dtype);
                let shape = Shape::new(&data.shape);
                let tensor = Tensor::from_bytes(&data.data, shape, dtype, CoreDevice::CPU)?;
                // Colored ">" prefix, white name
                if output::supports_color() {
                    println!("{}>{}  {}", output::colors::BOLD_YELLOW, output::colors::RESET, name);
                } else {
                    println!(">  {}", name);
                }
                println!("{}", tensor);
            }
        },
    }

    Ok(())
}

fn expand_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(stripped);
        }
    }
    PathBuf::from(path)
}

fn parse_device(device_str: &str) -> Result<Device, Box<dyn std::error::Error>> {
    // Normalize to lowercase with :: separator for device index
    let device = device_str.to_lowercase();

    // Handle common patterns
    match device.as_str() {
        "cpu" | "metal" | "vulkan" | "webgpu" => Ok(device),
        s if s.starts_with("cuda") => {
            if s == "cuda" {
                Ok("cuda::0".to_string())
            } else if let Some(idx_str) = s.strip_prefix("cuda::") {
                // Validate index
                idx_str
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid CUDA device index: {}", idx_str))?;
                Ok(device)
            } else if let Some(idx_str) = s.strip_prefix("cuda:") {
                // Convert cuda:N to cuda::N
                idx_str
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid CUDA device index: {}", idx_str))?;
                Ok(format!("cuda::{}", idx_str))
            } else {
                Err(format!("Invalid device format: {}", device_str).into())
            }
        },
        s if s.starts_with("rocm") => {
            if s == "rocm" {
                Ok("rocm::0".to_string())
            } else if let Some(idx_str) = s.strip_prefix("rocm::") {
                idx_str
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid ROCm device index: {}", idx_str))?;
                Ok(device)
            } else {
                Err(format!("Invalid device format: {}", device_str).into())
            }
        },
        // Allow any other device string for extensibility
        _ => Ok(device),
    }
}

fn find_backend_plugin<'a>(
    backend_name: &Option<String>,
    device: &Device,
    registry: &'a PluginRegistry,
) -> Result<&'a crate::plugins::PluginEntry, Box<dyn std::error::Error>> {
    if let Some(name) = backend_name {
        if let Some(plugin) = registry.find(name) {
            return Ok(plugin);
        }
        let prefixed = format!("hodu-backend-{}", name);
        if let Some(plugin) = registry.find(&prefixed) {
            return Ok(plugin);
        }
        return Err(format!("Backend '{}' not found.", name).into());
    }

    if let Some(plugin) = registry.find_backend_by_device(device) {
        return Ok(plugin);
    }

    Err(friendly_backend_error(device, registry).into())
}

fn friendly_format_error(extension: &str, registry: &PluginRegistry) -> String {
    let mut msg = format!("No model format plugin found for '.{}' format.\n\n", extension);
    let formats: Vec<_> = registry.model_formats().collect();
    if !formats.is_empty() {
        msg.push_str("Installed model format plugins:\n");
        for p in formats {
            msg.push_str(&format!(
                "  {} - {}\n",
                p.name,
                p.capabilities.model_extensions.join(", ")
            ));
        }
    }
    msg.push_str("\nBuiltin formats: .hdss");
    msg
}

fn friendly_backend_error(device: &Device, registry: &PluginRegistry) -> String {
    let mut msg = format!("No backend plugin found for device '{}'.\n\n", device);

    let backends: Vec<_> = registry.backends().collect();
    if !backends.is_empty() {
        msg.push_str("Installed backend plugins:\n");
        for p in backends {
            msg.push_str(&format!(
                "  {} - devices: {}\n",
                p.name,
                p.capabilities.devices.join(", ")
            ));
        }
    }

    msg
}

fn current_target_triple() -> String {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    return "x86_64-unknown-linux-gnu".to_string();
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    return "aarch64-unknown-linux-gnu".to_string();
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    return "x86_64-apple-darwin".to_string();
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    return "aarch64-apple-darwin".to_string();
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    return "x86_64-pc-windows-msvc".to_string();
    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "linux"),
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "windows"),
    )))]
    return String::new();
}

fn load_tensor_data(path: &str) -> Result<TensorData, Box<dyn std::error::Error>> {
    let tensor = hdt::load(path).map_err(|e| format!("Failed to load HDT: {}", e))?;
    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let dtype = core_dtype_to_plugin(tensor.dtype());
    let data = tensor
        .to_bytes()
        .map_err(|e| format!("Failed to get tensor bytes: {}", e))?;
    Ok(TensorData::new(data, shape, dtype))
}

fn core_dtype_to_plugin(dtype: DType) -> PluginDType {
    match dtype {
        DType::BOOL => PluginDType::BOOL,
        DType::F8E4M3 => PluginDType::F8E4M3,
        DType::F8E5M2 => PluginDType::F8E5M2,
        DType::BF16 => PluginDType::BF16,
        DType::F16 => PluginDType::F16,
        DType::F32 => PluginDType::F32,
        DType::F64 => PluginDType::F64,
        DType::U8 => PluginDType::U8,
        DType::U16 => PluginDType::U16,
        DType::U32 => PluginDType::U32,
        DType::U64 => PluginDType::U64,
        DType::I8 => PluginDType::I8,
        DType::I16 => PluginDType::I16,
        DType::I32 => PluginDType::I32,
        DType::I64 => PluginDType::I64,
    }
}

fn plugin_dtype_to_core(dtype: PluginDType) -> DType {
    match dtype {
        PluginDType::BOOL => DType::BOOL,
        PluginDType::F8E4M3 => DType::F8E4M3,
        PluginDType::F8E5M2 => DType::F8E5M2,
        PluginDType::BF16 => DType::BF16,
        PluginDType::F16 => DType::F16,
        PluginDType::F32 => DType::F32,
        PluginDType::F64 => DType::F64,
        PluginDType::U8 => DType::U8,
        PluginDType::U16 => DType::U16,
        PluginDType::U32 => DType::U32,
        PluginDType::U64 => DType::U64,
        PluginDType::I8 => DType::I8,
        PluginDType::I16 => DType::I16,
        PluginDType::I32 => DType::I32,
        PluginDType::I64 => DType::I64,
        _ => DType::F32, // fallback for future dtypes
    }
}

fn save_tensor_data(
    tensor_data: &TensorData,
    path: impl AsRef<std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use hodu_core::format::hdt;

    let shape = Shape::new(&tensor_data.shape);
    let dtype = plugin_dtype_to_core(tensor_data.dtype);
    let tensor = Tensor::from_bytes(&tensor_data.data, shape, dtype, CoreDevice::CPU).map_err(|e| e.to_string())?;
    hdt::save(&tensor, path).map_err(|e| e.to_string())?;
    Ok(())
}
