use clap::Args;
use hodu_cli_plugin_sdk::Snapshot;
use std::path::PathBuf;

#[derive(Args)]
pub struct InspectArgs {
    /// Model file (.onnx, .hdss, .gguf, etc.)
    pub model: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Output format (pretty, json)
    #[arg(short, long, default_value = "pretty")]
    pub format: String,
}

pub fn execute(args: InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let ext = args.model.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "hdss" => inspect_hdss(&args),
        _ => {
            // TODO: Use FormatPlugin for other formats
            println!("Unsupported format: .{}", ext);
            println!("Supported formats: .hdss");
            Ok(())
        },
    }
}

fn inspect_hdss(args: &InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = Snapshot::load(&args.model).map_err(|e| format!("Failed to load snapshot: {}", e))?;

    if args.format == "json" {
        println!("{}", serde_json::to_string_pretty(&snapshot)?);
        return Ok(());
    }

    // Pretty format
    println!("=== Hodu Snapshot ===");
    if let Some(name) = &snapshot.name {
        println!("Name: {}", name);
    }
    println!();

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
