//! Compile command - compile .hdss to target format

use crate::common::{format_extension, parse_device, parse_output_format};
use clap::Args;
use hodu_core::{format::hdss, script::Script};
use hodu_plugin::{HoduError, HoduResult, PluginManager};
use std::path::PathBuf;

#[derive(Args)]
pub struct CompileArgs {
    /// Path to the .hdss file
    pub path: PathBuf,

    /// Output file path
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Target device (cpu, metal, cuda:0)
    #[arg(short, long, default_value = "cpu")]
    pub device: String,

    /// Output format (sharedlib, msl, metallib, ptx, cubin, llvm-ir, object)
    #[arg(short, long, default_value = "sharedlib")]
    pub format: String,

    /// Path to compiler plugin (.dylib/.so/.dll)
    #[arg(short, long)]
    pub plugin: Option<PathBuf>,
}

pub fn execute(args: CompileArgs) -> HoduResult<()> {
    let device = parse_device(&args.device)?;
    let format = parse_output_format(&args.format)?;

    // Load snapshot
    let snapshot = hdss::load(&args.path)?;
    let script = Script::new(snapshot);

    // Initialize plugin manager
    let mut manager = PluginManager::with_default_dir()?;
    manager.load_all()?;

    // Load specific plugin if provided
    if let Some(plugin_path) = &args.plugin {
        manager.load_compiler(plugin_path)?;
    }

    // Find compiler for the device
    let compiler = manager.compilers().find(|c| c.supports_device(device)).ok_or_else(|| {
        HoduError::BackendError(format!(
            "No compiler found for {:?}. Available: {:?}",
            device,
            manager.compiler_names()
        ))
    })?;

    // Determine output path
    let output_path = args
        .output
        .unwrap_or_else(|| args.path.with_extension(format_extension(format)));

    // Compile
    compiler.build(&script, device, format, &output_path)?;

    println!("Compiled {} -> {}", args.path.display(), output_path.display());

    Ok(())
}
