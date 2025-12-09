//! Convert command - convert models and tensors between formats

use crate::output;
use crate::plugins::{PluginManager, PluginRegistry};
use clap::Args;
use std::path::{Path, PathBuf};

/// Convert a path to a string, returning an error if the path is not valid UTF-8
fn path_to_str(path: &Path) -> Result<&str, Box<dyn std::error::Error>> {
    path.to_str()
        .ok_or_else(|| format!("Invalid UTF-8 in path: {}", path.display()).into())
}

#[derive(Args)]
pub struct ConvertArgs {
    /// Input file
    pub input: PathBuf,

    /// Output file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn execute(args: ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.input.exists() {
        return Err(format!("Input file not found: {}", args.input.display()).into());
    }

    let input_ext = args
        .input
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .ok_or("Input file has no extension")?;

    let output_ext = args
        .output
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .ok_or("Output file has no extension")?;

    let registry_path = PluginRegistry::default_path()?;
    let registry = PluginRegistry::load(&registry_path)?;

    // Determine conversion type (model or tensor)
    let is_model = is_model_format(&input_ext) || is_model_format(&output_ext);

    if args.verbose {
        println!("Input: {} (.{})", args.input.display(), input_ext);
        println!("Output: {} (.{})", args.output.display(), output_ext);
        println!("Type: {}", if is_model { "model" } else { "tensor" });
    }

    let mut manager = PluginManager::new()?;

    output::converting(&format!(
        "{} -> .{}",
        args.input.file_name().unwrap_or_default().to_string_lossy(),
        output_ext
    ));

    if is_model {
        convert_model(&args, &input_ext, &output_ext, &registry, &mut manager)
    } else {
        convert_tensor(&args, &input_ext, &output_ext, &registry, &mut manager)
    }
}

fn is_model_format(ext: &str) -> bool {
    matches!(ext, "hdss" | "onnx" | "pb" | "tflite" | "mlmodel")
}

fn convert_model(
    args: &ConvertArgs,
    input_ext: &str,
    output_ext: &str,
    registry: &PluginRegistry,
    manager: &mut PluginManager,
) -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Load input model to snapshot
    let snapshot_path = if input_ext == "hdss" {
        // Already a snapshot
        args.input.clone()
    } else {
        // Use model format plugin to load
        let plugin = registry
            .find_model_format_by_extension(input_ext)
            .ok_or_else(|| format!("No model format plugin for .{}", input_ext))?;

        if !plugin.capabilities.load_model.unwrap_or(false) {
            return Err(format!("Plugin {} doesn't support loading models", plugin.name).into());
        }

        let client = manager.get_plugin(&plugin.name)?;
        let result = client.load_model(path_to_str(&args.input)?)?;
        PathBuf::from(result.snapshot_path)
    };

    // Step 2: Save to output format
    if output_ext == "hdss" {
        // Just copy the snapshot
        std::fs::copy(&snapshot_path, &args.output)?;
    } else {
        // Use model format plugin to save
        let plugin = registry
            .find_model_format_by_extension(output_ext)
            .ok_or_else(|| format!("No model format plugin for .{}", output_ext))?;

        if !plugin.capabilities.save_model.unwrap_or(false) {
            return Err(format!("Plugin {} doesn't support saving models", plugin.name).into());
        }

        let client = manager.get_plugin(&plugin.name)?;
        client.save_model(path_to_str(&snapshot_path)?, path_to_str(&args.output)?)?;
    }

    output::finished(&format!(
        "{} -> {}",
        args.input.file_name().unwrap_or_default().to_string_lossy(),
        args.output.file_name().unwrap_or_default().to_string_lossy()
    ));
    Ok(())
}

fn convert_tensor(
    args: &ConvertArgs,
    input_ext: &str,
    output_ext: &str,
    registry: &PluginRegistry,
    manager: &mut PluginManager,
) -> Result<(), Box<dyn std::error::Error>> {
    use hodu_core::format::json;
    use hodu_core::tensor::Tensor;
    use hodu_core::types::{Device as CoreDevice, Shape};
    use hodu_plugin::TensorData;

    // Step 1: Load input tensor
    let tensor_data = match input_ext {
        "hdt" => load_tensor_data_from_path(&args.input)?,
        "json" => {
            let tensor = json::load(&args.input).map_err(|e| e.to_string())?;
            let shape = tensor.shape().dims().to_vec();
            let dtype = core_dtype_to_plugin(tensor.dtype());
            let data = tensor.to_bytes().map_err(|e| e.to_string())?;
            TensorData::new(data, shape, dtype)
        },
        _ => {
            // Use tensor format plugin
            let plugin = registry
                .find_tensor_format_by_extension(input_ext)
                .ok_or_else(|| format!("No tensor format plugin for .{}", input_ext))?;

            if !plugin.capabilities.load_tensor.unwrap_or(false) {
                return Err(format!("Plugin {} doesn't support loading tensors", plugin.name).into());
            }

            let client = manager.get_plugin(&plugin.name)?;
            let result = client.load_tensor(path_to_str(&args.input)?)?;
            load_tensor_data_from_path(&result.tensor_path)?
        },
    };

    // Step 2: Save to output format
    match output_ext {
        "hdt" => {
            save_tensor_data(&tensor_data, &args.output)?;
        },
        "json" => {
            let shape = Shape::new(&tensor_data.shape);
            let dtype = plugin_dtype_to_core(tensor_data.dtype);
            let tensor =
                Tensor::from_bytes(&tensor_data.data, shape, dtype, CoreDevice::CPU).map_err(|e| e.to_string())?;
            json::save(&tensor, &args.output).map_err(|e| e.to_string())?;
        },
        _ => {
            // Use tensor format plugin
            let plugin = registry
                .find_tensor_format_by_extension(output_ext)
                .ok_or_else(|| format!("No tensor format plugin for .{}", output_ext))?;

            if !plugin.capabilities.save_tensor.unwrap_or(false) {
                return Err(format!("Plugin {} doesn't support saving tensors", plugin.name).into());
            }

            // Save to temp hdt first
            let temp_path = std::env::temp_dir().join(format!("hodu_convert_{}.hdt", std::process::id()));
            save_tensor_data(&tensor_data, &temp_path)?;

            let client = manager.get_plugin(&plugin.name)?;
            client.save_tensor(path_to_str(&temp_path)?, path_to_str(&args.output)?)?;

            let _ = std::fs::remove_file(&temp_path);
        },
    }

    output::finished(&format!(
        "{} -> {}",
        args.input.file_name().unwrap_or_default().to_string_lossy(),
        args.output.file_name().unwrap_or_default().to_string_lossy()
    ));
    Ok(())
}

fn load_tensor_data_from_path(path: impl AsRef<Path>) -> Result<hodu_plugin::TensorData, Box<dyn std::error::Error>> {
    use hodu_core::format::hdt;
    use hodu_plugin::TensorData;

    let tensor = hdt::load(path).map_err(|e| format!("Failed to load HDT: {}", e))?;
    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let dtype = core_dtype_to_plugin(tensor.dtype());
    let data = tensor
        .to_bytes()
        .map_err(|e| format!("Failed to get tensor bytes: {}", e))?;
    Ok(TensorData::new(data, shape, dtype))
}

fn save_tensor_data(
    tensor_data: &hodu_plugin::TensorData,
    path: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use hodu_core::format::hdt;
    use hodu_core::tensor::Tensor;
    use hodu_core::types::{Device as CoreDevice, Shape};

    let shape = Shape::new(&tensor_data.shape);
    let dtype = plugin_dtype_to_core(tensor_data.dtype);
    let tensor = Tensor::from_bytes(&tensor_data.data, shape, dtype, CoreDevice::CPU).map_err(|e| e.to_string())?;
    hdt::save(&tensor, path).map_err(|e| e.to_string())?;
    Ok(())
}

fn core_dtype_to_plugin(dtype: hodu_core::types::DType) -> hodu_plugin::PluginDType {
    use hodu_core::types::DType;
    use hodu_plugin::PluginDType;
    match dtype {
        DType::BOOL => PluginDType::Bool,
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

fn plugin_dtype_to_core(dtype: hodu_plugin::PluginDType) -> hodu_core::types::DType {
    use hodu_core::types::DType;
    use hodu_plugin::PluginDType;
    match dtype {
        PluginDType::Bool => DType::BOOL,
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
