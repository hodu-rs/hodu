use crate::cli::plugin::{LoadedFormatPlugin, PluginRegistry};
use clap::Args;
use hodu_cli_plugin_sdk::{hdt, Snapshot, Tensor};
use std::path::{Path, PathBuf};

#[derive(Args)]
pub struct InspectArgs {
    /// File to inspect (.hdss, .hdt, .json, .onnx, etc.)
    pub file: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Output format (pretty, json)
    #[arg(short, long, default_value = "pretty")]
    pub format: String,
}

pub fn execute(args: InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.file.exists() {
        return Err(format!("File not found: {}", args.file.display()).into());
    }

    let ext = args
        .file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "hdss" => inspect_hdss(&args),
        "hdt" => inspect_hdt(&args),
        "json" => inspect_json_tensor(&args),
        _ => inspect_with_plugin(&args, &ext),
    }
}

fn inspect_hdss(args: &InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = Snapshot::load(&args.file).map_err(|e| format!("Failed to load snapshot: {}", e))?;

    if args.format == "json" {
        println!("{}", serde_json::to_string_pretty(&snapshot)?);
        return Ok(());
    }

    // Pretty format
    if let Some(name) = &snapshot.name {
        println!("Model: {}", name);
    } else {
        println!("Model: {}", args.file.display());
    }

    println!("Inputs ({}):", snapshot.inputs.len());
    for input in &snapshot.inputs {
        println!("  - {} : {:?} ({:?})", input.name, input.shape.dims(), input.dtype);
    }
    println!();

    if !snapshot.constants.is_empty() {
        println!("Constants ({}):", snapshot.constants.len());
        for constant in &snapshot.constants {
            let name = constant.name.as_deref().unwrap_or("(unnamed)");
            println!(
                "  - {} : {:?} ({:?}, {} bytes)",
                name,
                constant.shape.dims(),
                constant.dtype,
                constant.data.len()
            );
        }
        println!();
    }

    println!("Outputs ({}):", snapshot.targets.len());
    for target in &snapshot.targets {
        println!("  - {}", target.name);
    }
    println!();

    println!("Nodes ({}):", snapshot.nodes.len());
    if args.verbose {
        for (i, node) in snapshot.nodes.iter().enumerate() {
            println!("  [{}] {:?}", i, node.op);
            println!("      inputs: {:?}", node.input_ids);
            println!("      output: {:?} ({:?})", node.output_id, node.output_dtype);
        }
    } else {
        for node in &snapshot.nodes {
            println!("  - {:?}", node.op);
        }
    }

    Ok(())
}

fn inspect_hdt(args: &InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let tensor = hdt::load(&args.file).map_err(|e| format!("Failed to load HDT: {}", e))?;
    print_tensor_info(&tensor, &args.file, args.format == "json")
}

fn inspect_json_tensor(args: &InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    use hodu_cli_plugin_sdk::json;
    let tensor = json::load(&args.file).map_err(|e| format!("Failed to load JSON tensor: {}", e))?;
    print_tensor_info(&tensor, &args.file, args.format == "json")
}

fn print_tensor_info(tensor: &Tensor, path: &Path, as_json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let tensor_shape = tensor.shape();
    let shape = tensor_shape.dims();
    let dtype = tensor.dtype();
    let numel: usize = shape.iter().product();
    let size_bytes = tensor.to_bytes().map(|b| b.len()).unwrap_or(0);

    if as_json {
        println!(
            "{}",
            serde_json::json!({
                "file": path.display().to_string(),
                "shape": shape,
                "dtype": format!("{:?}", dtype),
                "numel": numel,
                "size_bytes": size_bytes
            })
        );
    } else {
        println!("File: {}", path.display());
        println!("Shape: {:?}", shape);
        println!("DType: {:?}", dtype);
        println!("Elements: {}", numel);
        println!("Size: {} bytes", size_bytes);
    }

    Ok(())
}

fn inspect_with_plugin(args: &InspectArgs, ext: &str) -> Result<(), Box<dyn std::error::Error>> {
    let registry_path = PluginRegistry::default_path()?;
    let registry = PluginRegistry::load(&registry_path)?;

    let plugin_entry = registry.find_format_by_extension(ext).ok_or_else(|| {
        format!(
            "No plugin found for '.{}' format.\n\nBuiltin formats: .hdss, .hdt, .json\n\nInstall a format plugin:\n  hodu plugin install onnx\n  hodu plugin install safetensors",
            ext
        )
    })?;

    let plugins_dir = PluginRegistry::plugins_dir()?;
    let lib_path = plugins_dir.join(&plugin_entry.library);
    let loaded_plugin = LoadedFormatPlugin::load(&lib_path)?;

    // Try to load as model first
    if plugin_entry.capabilities.load_model.unwrap_or(false) {
        let snapshot = loaded_plugin
            .plugin()
            .load_model(&args.file)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        if args.format == "json" {
            println!("{}", serde_json::to_string_pretty(&snapshot)?);
        } else {
            if let Some(name) = &snapshot.name {
                println!("Model: {}", name);
            } else {
                println!("Model: {}", args.file.display());
            }
            println!("Format: {} ({})", ext, plugin_entry.name);

            println!("Inputs ({}):", snapshot.inputs.len());
            for input in &snapshot.inputs {
                println!("  - {} : {:?} ({:?})", input.name, input.shape.dims(), input.dtype);
            }
            println!();

            println!("Outputs ({}):", snapshot.targets.len());
            for target in &snapshot.targets {
                println!("  - {}", target.name);
            }
            println!();

            println!("Nodes: {}", snapshot.nodes.len());
            if args.verbose {
                for (i, node) in snapshot.nodes.iter().enumerate() {
                    println!("  [{}] {:?}", i, node.op);
                }
            }
        }
        return Ok(());
    }

    // Try to load as tensor
    if plugin_entry.capabilities.load_tensor.unwrap_or(false) {
        let tensor_data = loaded_plugin
            .plugin()
            .load_tensor(&args.file)
            .map_err(|e| format!("Failed to load tensor: {}", e))?;

        if args.format == "json" {
            println!(
                "{}",
                serde_json::json!({
                    "file": args.file.display().to_string(),
                    "shape": tensor_data.shape,
                    "dtype": tensor_data.dtype.name(),
                    "size_bytes": tensor_data.data.len()
                })
            );
        } else {
            println!("File: {}", args.file.display());
            println!("Shape: {:?}", tensor_data.shape);
            println!("DType: {}", tensor_data.dtype.name());
            println!("Size: {} bytes", tensor_data.data.len());
        }
        return Ok(());
    }

    Err(format!(
        "Plugin '{}' doesn't support loading models or tensors",
        plugin_entry.name
    )
    .into())
}
