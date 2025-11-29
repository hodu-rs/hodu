//! Common CLI utilities

use hodu_core::{
    format::{hdt, json},
    tensor::Tensor,
};
use hodu_plugin::{Device, HoduError, HoduResult};
use std::path::Path;

/// Parse device string to Device enum
pub fn parse_device(s: &str) -> HoduResult<Device> {
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
            "Unknown device: '{}'. Supported: cpu, metal{}",
            s,
            if cfg!(feature = "cuda") { ", cuda:N" } else { "" },
        ))),
    }
}

/// Parse output format string
pub fn parse_output_format(s: &str) -> HoduResult<hodu_plugin::OutputFormat> {
    use hodu_plugin::OutputFormat;
    match s.to_lowercase().as_str() {
        // CPU formats
        "sharedlib" | "shared" | "dylib" | "so" => Ok(OutputFormat::SharedLib),
        "object" | "obj" | "o" => Ok(OutputFormat::Object),
        "staticlib" | "static" | "a" => Ok(OutputFormat::StaticLib),
        // Metal formats
        "msl" | "metal" => Ok(OutputFormat::Msl),
        "air" => Ok(OutputFormat::Air),
        "metallib" => Ok(OutputFormat::Metallib),
        // CUDA formats
        "ptx" => Ok(OutputFormat::Ptx),
        "cubin" => Ok(OutputFormat::Cubin),
        "fatbin" => Ok(OutputFormat::Fatbin),
        // LLVM formats
        "llvm-ir" | "llvm" | "ll" => Ok(OutputFormat::LlvmIR),
        "llvm-bc" | "bc" => Ok(OutputFormat::LlvmBitcode),
        "asm" | "s" => Ok(OutputFormat::Assembly),
        _ => Err(HoduError::InvalidArgument(format!(
            "Unknown output format: '{}'. Supported: sharedlib, object, msl, metallib, ptx, cubin, llvm-ir",
            s
        ))),
    }
}

/// Get file extension for output format
pub fn format_extension(format: hodu_plugin::OutputFormat) -> &'static str {
    use hodu_plugin::OutputFormat;
    match format {
        OutputFormat::Object => "o",
        OutputFormat::SharedLib => {
            #[cfg(target_os = "macos")]
            {
                "dylib"
            }
            #[cfg(target_os = "windows")]
            {
                "dll"
            }
            #[cfg(not(any(target_os = "macos", target_os = "windows")))]
            {
                "so"
            }
        },
        OutputFormat::StaticLib => "a",
        OutputFormat::Executable => "exe",
        OutputFormat::LlvmIR => "ll",
        OutputFormat::LlvmBitcode => "bc",
        OutputFormat::Assembly => "s",
        OutputFormat::Ptx => "ptx",
        OutputFormat::Cubin => "cubin",
        OutputFormat::Fatbin => "fatbin",
        OutputFormat::Msl => "metal",
        OutputFormat::Air => "air",
        OutputFormat::Metallib => "metallib",
        OutputFormat::Hsaco => "hsaco",
        OutputFormat::SpirV => "spv",
        OutputFormat::HoduSnapshot => "hdss",
    }
}

/// Load tensor from file (supports .hdt and .json)
pub fn load_tensor(path: &Path) -> HoduResult<Tensor> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "hdt" => hdt::load(path),
        "json" => json::load(path),
        _ => Err(HoduError::InvalidArgument(format!(
            "Unsupported tensor format: '{}'. Use .hdt or .json",
            ext
        ))),
    }
}

/// Parse input argument "name=path.hdt"
pub fn parse_input_arg(s: &str) -> HoduResult<(String, std::path::PathBuf)> {
    let parts: Vec<&str> = s.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(HoduError::InvalidArgument(format!(
            "Invalid input format: '{}'. Use name=path.hdt or name=path.json",
            s
        )));
    }
    Ok((parts[0].to_string(), std::path::PathBuf::from(parts[1])))
}
