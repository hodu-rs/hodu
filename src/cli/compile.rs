//! Compile command - compile .hdss to target format

use hodu_core::script::Script;
use hodu_format::hdss;
use hodu_plugin::{Device, HoduError, HoduResult, OutputFormat, PluginManager};
use std::path::PathBuf;

fn parse_device(s: &str) -> HoduResult<Device> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(Device::CPU),
        #[cfg(feature = "cuda")]
        s if s.starts_with("cuda:") => {
            let id: usize = s[5..]
                .parse()
                .map_err(|_| HoduError::InvalidArgument("Invalid CUDA device ID".into()))?;
            Ok(Device::CUDA(id))
        },
        "metal" => Ok(Device::Metal),
        _ => Err(HoduError::InvalidArgument(format!(
            "Unknown device: {}. Use 'cpu', 'metal'{}",
            s,
            if cfg!(feature = "cuda") { ", 'cuda:N'" } else { "" },
        ))),
    }
}

fn parse_format(s: &str) -> HoduResult<OutputFormat> {
    match s.to_lowercase().as_str() {
        "msl" => Ok(OutputFormat::Msl),
        "air" => Ok(OutputFormat::Air),
        "metallib" => Ok(OutputFormat::Metallib),
        "ptx" => Ok(OutputFormat::Ptx),
        "cubin" => Ok(OutputFormat::Cubin),
        "llvm-ir" | "llvm" => Ok(OutputFormat::LlvmIR),
        "object" | "obj" => Ok(OutputFormat::Object),
        _ => Err(HoduError::InvalidArgument(format!(
            "Unknown output format: {}. Use 'msl', 'air', 'metallib', 'ptx', 'cubin', 'llvm-ir', 'object'",
            s
        ))),
    }
}

pub fn execute(
    path: PathBuf,
    output: Option<PathBuf>,
    device_str: &str,
    format_str: &str,
    plugin_path: Option<PathBuf>,
) -> HoduResult<()> {
    let device = parse_device(device_str)?;
    let format = parse_format(format_str)?;

    // Load snapshot
    let snapshot = hdss::load(&path)?;
    let script = Script::new(snapshot);

    // Initialize plugin manager
    let mut manager = PluginManager::with_default_dir()?;
    manager.load_all()?;

    // Load specific plugin if provided
    if let Some(plugin_path) = plugin_path {
        manager.load_compiler(&plugin_path)?;
    }

    // Find compiler for the device
    let compiler = manager.compilers().find(|c| c.supports_device(device)).ok_or_else(|| {
        HoduError::BackendError(format!(
            "No compiler found for device {:?}. Available compilers: {:?}",
            device,
            manager.compiler_names()
        ))
    })?;

    // Determine output path
    let output_path = output.unwrap_or_else(|| {
        let ext = match format {
            OutputFormat::Msl => "metal",
            OutputFormat::Air => "air",
            OutputFormat::Metallib => "metallib",
            OutputFormat::Ptx => "ptx",
            OutputFormat::Cubin => "cubin",
            OutputFormat::LlvmIR => "ll",
            OutputFormat::Object => "o",
            _ => "bin",
        };
        path.with_extension(ext)
    });

    // Compile
    compiler.build(&script, device, format, &output_path)?;

    println!("Compiled {} -> {}", path.display(), output_path.display());

    Ok(())
}
