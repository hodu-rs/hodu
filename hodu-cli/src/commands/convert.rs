//! Convert command - convert models and tensors between formats

use crate::output;
use crate::plugins::{PluginManager, PluginRegistry};
use clap::Args;
use std::path::PathBuf;

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
        let result = client.load_model(args.input.to_str().unwrap())?;
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
        client.save_model(snapshot_path.to_str().unwrap(), args.output.to_str().unwrap())?;
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
    use hodu_plugin_sdk::{json, TensorData};

    // Step 1: Load input tensor
    let tensor_data = match input_ext {
        "hdt" => TensorData::load(&args.input)?,
        "json" => {
            let tensor = json::load(&args.input).map_err(|e| e.to_string())?;
            let shape = tensor.shape().dims().to_vec();
            let dtype = tensor.dtype().into();
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
            let result = client.load_tensor(args.input.to_str().unwrap())?;
            TensorData::load(&result.tensor_path)?
        },
    };

    // Step 2: Save to output format
    match output_ext {
        "hdt" => {
            tensor_data.save(&args.output)?;
        },
        "json" => {
            use hodu_plugin_sdk::{CoreDevice, Shape, Tensor};
            let shape = Shape::new(&tensor_data.shape);
            let dtype = tensor_data.core_dtype();
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
            tensor_data.save(&temp_path)?;

            let client = manager.get_plugin(&plugin.name)?;
            client.save_tensor(temp_path.to_str().unwrap(), args.output.to_str().unwrap())?;

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
