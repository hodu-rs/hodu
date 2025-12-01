use crate::cli::plugin::{LoadedBackendPlugin, LoadedFormatPlugin, PluginRegistry};
use clap::Args;
use hodu_cli_plugin_sdk::{Device, SdkDType, Snapshot, TensorData};
use std::collections::HashMap;
use std::path::PathBuf;

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

    /// Execution device (cpu, metal, cuda:0)
    #[arg(short, long, default_value = "cpu")]
    pub device: String,

    /// Backend plugin to use (auto-select if not specified)
    #[arg(long)]
    pub backend: Option<String>,

    /// Output format (pretty, json)
    #[arg(short, long, default_value = "pretty")]
    pub format: String,

    /// Show detailed output info (shape, dtype)
    #[arg(long)]
    pub dirty: bool,

    /// Save outputs to directory
    #[arg(long)]
    pub save: Option<PathBuf>,

    /// Save format (hdt, json, or format plugin extension)
    #[arg(long, default_value = "hdt")]
    pub save_format: String,

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

    /// Suppress all output
    #[arg(short, long)]
    pub quiet: bool,
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
            "Backend: {:?} ({} {})",
            device, backend_plugin.name, backend_plugin.version
        );
        println!();
        println!("Would execute with above configuration.");
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

    // Load model
    let snapshot = load_model(&args.model, format_plugin.map(|p| &p.library), &plugins_dir)?;

    // Parse input tensors
    let inputs = parse_inputs(&all_inputs, &snapshot)?;
    let input_refs: Vec<(&str, TensorData)> = inputs.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

    // Run model
    let outputs = loaded_backend
        .plugin()
        .run(&snapshot, device, &input_refs)
        .map_err(|e| format!("Execution failed: {}", e))?;

    // Save outputs if requested
    if let Some(save_dir) = &args.save {
        save_outputs(&outputs, save_dir, &args.save_format, &registry, &plugins_dir)?;
    }

    // Output results
    if !args.quiet {
        output_results(&outputs, &args)?;
    }

    Ok(())
}

fn load_model(
    path: &PathBuf,
    format_plugin_lib: Option<&String>,
    plugins_dir: &PathBuf,
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
                    .map_err(|e| format!("Format plugin failed to load model: {}", e).into())
            } else {
                Err(format!("No format plugin available for .{}", ext).into())
            }
        },
    }
}

fn save_outputs(
    outputs: &HashMap<String, TensorData>,
    save_dir: &PathBuf,
    format: &str,
    registry: &PluginRegistry,
    plugins_dir: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    use hodu_cli_plugin_sdk::{hdt, json, CoreDevice, Shape, Tensor};

    // Create directory if it doesn't exist
    std::fs::create_dir_all(save_dir)?;

    let format_lower = format.to_lowercase();

    // Check if it's a builtin format or needs a plugin
    let is_builtin = matches!(format_lower.as_str(), "hdt" | "json");

    // Find format plugin if not builtin
    let format_plugin = if !is_builtin {
        let plugin = registry.find_format_by_extension(&format_lower);
        if plugin.is_none() {
            return Err(format!(
                "Unknown save format: '{}'\n\nBuiltin formats: hdt, json\nInstalled format plugins: {:?}",
                format,
                registry
                    .formats()
                    .map(|p| p.capabilities.extensions.join(", "))
                    .collect::<Vec<_>>()
            )
            .into());
        }
        plugin
    } else {
        None
    };

    for (name, data) in outputs {
        let file_path = save_dir.join(format!("{}.{}", name, format_lower));

        match format_lower.as_str() {
            "hdt" | "json" => {
                let dtype = sdk_dtype_to_core_dtype(data.dtype);
                let shape = Shape::new(&data.shape);
                let tensor = Tensor::from_bytes(&data.data, shape, dtype, CoreDevice::CPU)
                    .map_err(|e| format!("Failed to create tensor: {}", e))?;
                if format_lower == "hdt" {
                    hdt::save(&tensor, &file_path).map_err(|e| format!("Failed to save HDT: {}", e))?;
                } else {
                    json::save(&tensor, &file_path).map_err(|e| format!("Failed to save JSON: {}", e))?;
                }
            },
            _ => {
                // Use format plugin
                if let Some(plugin_entry) = format_plugin {
                    let lib_path = plugins_dir.join(&plugin_entry.library);
                    let loaded_format = LoadedFormatPlugin::load(&lib_path)?;
                    loaded_format
                        .plugin()
                        .save_tensor(data, &file_path)
                        .map_err(|e| format!("Format plugin failed to save tensor: {}", e))?;
                }
            },
        }
    }

    Ok(())
}

fn sdk_dtype_to_core_dtype(dtype: SdkDType) -> hodu_cli_plugin_sdk::DType {
    use hodu_cli_plugin_sdk::DType;
    match dtype {
        SdkDType::Bool => DType::BOOL,
        SdkDType::F8E4M3 => DType::F8E4M3,
        SdkDType::F8E5M2 => DType::F8E5M2,
        SdkDType::BF16 => DType::BF16,
        SdkDType::F16 => DType::F16,
        SdkDType::F32 => DType::F32,
        SdkDType::F64 => DType::F64,
        SdkDType::U8 => DType::U8,
        SdkDType::U16 => DType::U16,
        SdkDType::U32 => DType::U32,
        SdkDType::U64 => DType::U64,
        SdkDType::I8 => DType::I8,
        SdkDType::I16 => DType::I16,
        SdkDType::I32 => DType::I32,
        SdkDType::I64 => DType::I64,
        _ => DType::F32, // fallback
    }
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

        let tensor_data = load_tensor_file(&path, &input_spec.shape.dims(), input_spec.dtype.into())?;
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

fn load_tensor_file(
    path: &PathBuf,
    expected_shape: &[usize],
    expected_dtype: SdkDType,
) -> Result<TensorData, Box<dyn std::error::Error>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "hdt" => load_tensor_hdt(path, expected_shape, expected_dtype),
        "json" => load_tensor_json(path, expected_shape, expected_dtype),
        _ => Err(format!("Unsupported tensor format: .{}\nSupported: .hdt, .json", ext).into()),
    }
}

fn load_tensor_hdt(
    path: &PathBuf,
    expected_shape: &[usize],
    expected_dtype: SdkDType,
) -> Result<TensorData, Box<dyn std::error::Error>> {
    use hodu_cli_plugin_sdk::hdt;

    let tensor = hdt::load(path).map_err(|e| format!("Failed to load HDT file: {}", e))?;

    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let dtype: SdkDType = tensor.dtype().into();

    // Verify shape
    if shape != expected_shape {
        return Err(format!(
            "Shape mismatch: file has {:?}, model expects {:?}",
            shape, expected_shape
        )
        .into());
    }

    // Verify dtype
    if dtype != expected_dtype {
        return Err(format!(
            "DType mismatch: file has {}, model expects {}",
            dtype.name(),
            expected_dtype.name()
        )
        .into());
    }

    let data = tensor
        .to_bytes()
        .map_err(|e| format!("Failed to get tensor bytes: {}", e))?;

    Ok(TensorData::new(data, shape, dtype))
}

fn load_tensor_json(
    path: &PathBuf,
    expected_shape: &[usize],
    expected_dtype: SdkDType,
) -> Result<TensorData, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let obj = json
        .as_object()
        .ok_or("JSON must be an object with shape, dtype, data")?;

    // Parse shape
    let shape: Vec<usize> = obj
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or("Missing 'shape' field")?
        .iter()
        .map(|v| v.as_u64().map(|n| n as usize).ok_or("Invalid shape dimension"))
        .collect::<Result<Vec<_>, _>>()?;

    // Verify shape matches expected
    if shape != expected_shape {
        return Err(format!(
            "Shape mismatch: file has {:?}, model expects {:?}",
            shape, expected_shape
        )
        .into());
    }

    // Parse dtype
    let dtype_str = obj
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'dtype' field")?;
    let dtype = str_to_sdk_dtype(dtype_str)?;

    // Verify dtype matches expected
    if dtype != expected_dtype {
        return Err(format!(
            "DType mismatch: file has {}, model expects {}",
            dtype.name(),
            expected_dtype.name()
        )
        .into());
    }

    // Parse data
    let data_arr = obj
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or("Missing 'data' field")?;

    let data = json_array_to_bytes(data_arr, dtype)?;

    Ok(TensorData::new(data, shape, dtype))
}

fn str_to_sdk_dtype(s: &str) -> Result<SdkDType, Box<dyn std::error::Error>> {
    match s.to_lowercase().as_str() {
        "bool" => Ok(SdkDType::Bool),
        "f8e4m3" => Ok(SdkDType::F8E4M3),
        "f8e5m2" => Ok(SdkDType::F8E5M2),
        "bf16" => Ok(SdkDType::BF16),
        "f16" => Ok(SdkDType::F16),
        "f32" => Ok(SdkDType::F32),
        "f64" => Ok(SdkDType::F64),
        "u8" => Ok(SdkDType::U8),
        "u16" => Ok(SdkDType::U16),
        "u32" => Ok(SdkDType::U32),
        "u64" => Ok(SdkDType::U64),
        "i8" => Ok(SdkDType::I8),
        "i16" => Ok(SdkDType::I16),
        "i32" => Ok(SdkDType::I32),
        "i64" => Ok(SdkDType::I64),
        _ => Err(format!("Unknown dtype: {}", s).into()),
    }
}

fn json_array_to_bytes(arr: &[serde_json::Value], dtype: SdkDType) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(arr.len() * dtype.size_in_bytes());

    match dtype {
        SdkDType::Bool => {
            for v in arr {
                let b = v.as_bool().ok_or("Expected bool")?;
                bytes.push(if b { 1 } else { 0 });
            }
        },
        SdkDType::F8E4M3 => {
            for v in arr {
                let f = v.as_f64().ok_or("Expected number")? as f32;
                bytes.push(float8::F8E4M3::from_f32(f).to_bits());
            }
        },
        SdkDType::F8E5M2 => {
            for v in arr {
                let f = v.as_f64().ok_or("Expected number")? as f32;
                bytes.push(float8::F8E5M2::from_f32(f).to_bits());
            }
        },
        SdkDType::BF16 => {
            for v in arr {
                let f = v.as_f64().ok_or("Expected number")?;
                bytes.extend_from_slice(&half::bf16::from_f64(f).to_bits().to_le_bytes());
            }
        },
        SdkDType::F16 => {
            for v in arr {
                let f = v.as_f64().ok_or("Expected number")?;
                bytes.extend_from_slice(&half::f16::from_f64(f).to_bits().to_le_bytes());
            }
        },
        SdkDType::F32 => {
            for v in arr {
                let f = v.as_f64().ok_or("Expected number")? as f32;
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        },
        SdkDType::F64 => {
            for v in arr {
                let f = v.as_f64().ok_or("Expected number")?;
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        },
        SdkDType::U8 => {
            for v in arr {
                let n = v.as_u64().ok_or("Expected integer")? as u8;
                bytes.push(n);
            }
        },
        SdkDType::U16 => {
            for v in arr {
                let n = v.as_u64().ok_or("Expected integer")? as u16;
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        },
        SdkDType::U32 => {
            for v in arr {
                let n = v.as_u64().ok_or("Expected integer")? as u32;
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        },
        SdkDType::U64 => {
            for v in arr {
                let n = v.as_u64().ok_or("Expected integer")?;
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        },
        SdkDType::I8 => {
            for v in arr {
                let n = v.as_i64().ok_or("Expected integer")? as i8;
                bytes.push(n as u8);
            }
        },
        SdkDType::I16 => {
            for v in arr {
                let n = v.as_i64().ok_or("Expected integer")? as i16;
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        },
        SdkDType::I32 => {
            for v in arr {
                let n = v.as_i64().ok_or("Expected integer")? as i32;
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        },
        SdkDType::I64 => {
            for v in arr {
                let n = v.as_i64().ok_or("Expected integer")?;
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        },
        _ => return Err(format!("Unsupported dtype for JSON input: {}", dtype.name()).into()),
    }

    Ok(bytes)
}

fn output_results(outputs: &HashMap<String, TensorData>, args: &RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.format.as_str() {
        "json" => {
            let json_outputs: HashMap<String, serde_json::Value> = outputs
                .iter()
                .map(|(name, data)| {
                    let values = bytes_to_json_values(&data.data, data.dtype);
                    (
                        name.clone(),
                        serde_json::json!({
                            "shape": data.shape,
                            "dtype": data.dtype.name(),
                            "data": values
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
                if args.dirty {
                    println!("{}:", name);
                    println!("  shape: {:?}", data.shape);
                    println!("  dtype: {}", data.dtype.name());
                    print_tensor_preview(&data.data, &data.shape, data.dtype, "  ");
                } else {
                    let values = format_tensor_values(&data.data, &data.shape, data.dtype);
                    println!("{}: {}", name, values);
                }
            }
        },
    }

    Ok(())
}

fn bytes_to_json_values(data: &[u8], dtype: SdkDType) -> Vec<serde_json::Value> {
    match dtype {
        SdkDType::Bool => data.iter().map(|&b| serde_json::json!(b != 0)).collect(),
        SdkDType::F8E4M3 => data
            .iter()
            .map(|&b| serde_json::json!(f32::from(float8::F8E4M3::from_bits(b))))
            .collect(),
        SdkDType::F8E5M2 => data
            .iter()
            .map(|&b| serde_json::json!(f32::from(float8::F8E5M2::from_bits(b))))
            .collect(),
        SdkDType::BF16 => data
            .chunks_exact(2)
            .map(|b| serde_json::json!(half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()))
            .collect(),
        SdkDType::F16 => data
            .chunks_exact(2)
            .map(|b| serde_json::json!(half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()))
            .collect(),
        SdkDType::F32 => data
            .chunks_exact(4)
            .map(|b| serde_json::json!(f32::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::F64 => data
            .chunks_exact(8)
            .map(|b| serde_json::json!(f64::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::U8 => data.iter().map(|&b| serde_json::json!(b)).collect(),
        SdkDType::U16 => data
            .chunks_exact(2)
            .map(|b| serde_json::json!(u16::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::U32 => data
            .chunks_exact(4)
            .map(|b| serde_json::json!(u32::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::U64 => data
            .chunks_exact(8)
            .map(|b| serde_json::json!(u64::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::I8 => data.iter().map(|&b| serde_json::json!(b as i8)).collect(),
        SdkDType::I16 => data
            .chunks_exact(2)
            .map(|b| serde_json::json!(i16::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::I32 => data
            .chunks_exact(4)
            .map(|b| serde_json::json!(i32::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::I64 => data
            .chunks_exact(8)
            .map(|b| serde_json::json!(i64::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        _ => vec![serde_json::json!(format!("<{} bytes, unsupported dtype>", data.len()))],
    }
}

fn print_tensor_preview(data: &[u8], shape: &[usize], dtype: SdkDType, prefix: &str) {
    let numel: usize = shape.iter().product();
    let max_display = 10;
    let values = collect_tensor_values(data, dtype, max_display);

    let preview = if numel > max_display {
        format!("[{}, ... ({} more)]", values.join(", "), numel - max_display)
    } else {
        format!("[{}]", values.join(", "))
    };
    println!("{}data: {}", prefix, preview);
}

fn format_tensor_values(data: &[u8], shape: &[usize], dtype: SdkDType) -> String {
    let numel: usize = shape.iter().product();
    let max_display = 20;
    let values = collect_tensor_values(data, dtype, max_display);

    if numel > max_display {
        format!("[{}, ... ({} more)]", values.join(", "), numel - max_display)
    } else {
        format!("[{}]", values.join(", "))
    }
}

fn collect_tensor_values(data: &[u8], dtype: SdkDType, max_display: usize) -> Vec<String> {
    match dtype {
        SdkDType::Bool => data.iter().take(max_display).map(|&b| format!("{}", b != 0)).collect(),
        SdkDType::F8E4M3 => data
            .iter()
            .take(max_display)
            .map(|&b| format!("{:.4}", f32::from(float8::F8E4M3::from_bits(b))))
            .collect(),
        SdkDType::F8E5M2 => data
            .iter()
            .take(max_display)
            .map(|&b| format!("{:.4}", f32::from(float8::F8E5M2::from_bits(b))))
            .collect(),
        SdkDType::BF16 => data
            .chunks_exact(2)
            .take(max_display)
            .map(|b| {
                format!(
                    "{:.4}",
                    half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()
                )
            })
            .collect(),
        SdkDType::F16 => data
            .chunks_exact(2)
            .take(max_display)
            .map(|b| {
                format!(
                    "{:.4}",
                    half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()
                )
            })
            .collect(),
        SdkDType::F32 => data
            .chunks_exact(4)
            .take(max_display)
            .map(|b| format!("{:.4}", f32::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::F64 => data
            .chunks_exact(8)
            .take(max_display)
            .map(|b| format!("{:.4}", f64::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::U8 => data.iter().take(max_display).map(|&b| format!("{}", b)).collect(),
        SdkDType::U16 => data
            .chunks_exact(2)
            .take(max_display)
            .map(|b| format!("{}", u16::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::U32 => data
            .chunks_exact(4)
            .take(max_display)
            .map(|b| format!("{}", u32::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::U64 => data
            .chunks_exact(8)
            .take(max_display)
            .map(|b| format!("{}", u64::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::I8 => data.iter().take(max_display).map(|&b| format!("{}", b as i8)).collect(),
        SdkDType::I16 => data
            .chunks_exact(2)
            .take(max_display)
            .map(|b| format!("{}", i16::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::I32 => data
            .chunks_exact(4)
            .take(max_display)
            .map(|b| format!("{}", i32::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        SdkDType::I64 => data
            .chunks_exact(8)
            .take(max_display)
            .map(|b| format!("{}", i64::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        _ => vec![format!("<{} bytes>", data.len())],
    }
}

fn expand_path(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(&path[2..]);
        }
    }
    PathBuf::from(path)
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
