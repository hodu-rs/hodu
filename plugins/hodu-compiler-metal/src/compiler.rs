//! Metal Compiler Plugin implementation

use crate::dispatch::DispatchManifest;
use crate::xcrun;
use hodu_plugin::{
    ArtifactDType, ArtifactSymbols, ArtifactTensorInfo, CompiledArtifact, CompilerPlugin, Device,
    HoduError, HoduResult, OutputFormat, Script,
};
use std::io::Write;
use std::path::Path;

/// Metal Compiler Plugin
///
/// Compiles Hodu Scripts/Snapshots to Metal artifacts
///
/// Supported dtypes: bool, bf16, f16, f32, u8, u16, u32, u64, i8, i16, i32, i64
/// NOT supported: f8e4m3, f8e5m2, f64 (Metal hardware limitation)
pub struct MetalCompiler {
    /// Path to bundled Metal kernels (optional, uses embedded if None)
    kernel_path: Option<std::path::PathBuf>,
}

impl MetalCompiler {
    /// Create a new Metal compiler
    pub fn new() -> Self {
        Self { kernel_path: None }
    }

    /// Create with custom kernel path
    pub fn with_kernel_path(kernel_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            kernel_path: Some(kernel_path.into()),
        }
    }

    /// Get the embedded Metal kernel source
    fn get_kernel_source(&self) -> HoduResult<String> {
        // Bundled kernels from hodu_metal_kernels
        Ok(include_str!(concat!(env!("OUT_DIR"), "/bundled_kernels.metal")).to_string())
    }

    /// Compile snapshot to artifact
    fn compile_snapshot(
        &self,
        snapshot: &hodu_core::script::Snapshot,
        _device: Device,
    ) -> HoduResult<CompiledArtifact> {
        // Generate dispatch manifest
        let manifest = DispatchManifest::from_snapshot(snapshot);

        // Create temp directory for compilation
        let temp_dir = std::env::temp_dir().join(format!("hodu_metal_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| HoduError::BackendError(format!("Failed to create temp dir: {}", e)))?;

        // Write kernel source
        let metal_path = temp_dir.join("kernels.metal");
        let kernel_source = self.get_kernel_source()?;
        std::fs::write(&metal_path, &kernel_source)
            .map_err(|e| HoduError::BackendError(format!("Failed to write metal source: {}", e)))?;

        // Compile to metallib
        let metallib_path = temp_dir.join("kernels.metallib");
        xcrun::compile_metal_to_metallib(&metal_path, &metallib_path)?;

        // Read compiled metallib
        let metallib_data = std::fs::read(&metallib_path)
            .map_err(|e| HoduError::BackendError(format!("Failed to read metallib: {}", e)))?;

        // Clean up temp directory
        let _ = std::fs::remove_dir_all(&temp_dir);

        // Create artifact with metallib + manifest
        let manifest_json = manifest.to_json();

        // Pack: [manifest_len (8 bytes)][manifest_json][metallib_data]
        let mut artifact_data = Vec::new();
        artifact_data.extend_from_slice(&(manifest_json.len() as u64).to_le_bytes());
        artifact_data.extend_from_slice(&manifest_json);
        artifact_data.extend_from_slice(&metallib_data);

        // Build artifact
        let mut artifact = CompiledArtifact::new(OutputFormat::Metallib, Device::Metal, artifact_data);

        // Add input info
        for input in &manifest.inputs {
            artifact.inputs.push(ArtifactTensorInfo {
                name: input.name.clone(),
                shape: input.shape.clone(),
                dtype: parse_dtype(&input.dtype)?,
            });
        }

        // Add output info
        for output in &manifest.outputs {
            artifact.outputs.push(ArtifactTensorInfo {
                name: output.name.clone(),
                shape: output.shape.clone(),
                dtype: ArtifactDType::F32, // Default, will be updated by runtime
            });
        }

        artifact.symbols = Some(ArtifactSymbols {
            entry_point: "dispatch_graph".to_string(),
            exports: vec![],
        });

        Ok(artifact)
    }
}

impl Default for MetalCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl CompilerPlugin for MetalCompiler {
    fn name(&self) -> &str {
        "metal"
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::Metal]
    }

    fn supported_formats(&self, device: Device) -> Vec<OutputFormat> {
        match device {
            Device::Metal => vec![
                OutputFormat::Msl,
                OutputFormat::Air,
                OutputFormat::Metallib,
            ],
            _ => vec![],
        }
    }

    fn compile(&self, script: &Script, device: Device) -> HoduResult<CompiledArtifact> {
        if device != Device::Metal {
            return Err(HoduError::UnsupportedDevice(device));
        }

        if !xcrun::is_available() {
            return Err(HoduError::BackendError(
                "xcrun metal not available. Are you on macOS with Xcode installed?".into(),
            ));
        }

        // Get snapshot from script
        self.compile_snapshot(script.snapshot(), device)
    }

    fn build(
        &self,
        script: &Script,
        device: Device,
        format: OutputFormat,
        path: &Path,
    ) -> HoduResult<()> {
        if device != Device::Metal {
            return Err(HoduError::UnsupportedDevice(device));
        }

        let snapshot = script.snapshot();

        match format {
            OutputFormat::Msl => {
                // Output MSL source code
                let kernel_source = self.get_kernel_source()?;
                let manifest = DispatchManifest::from_snapshot(snapshot);
                let manifest_json = serde_json::to_string_pretty(&manifest)
                    .map_err(|e| HoduError::BackendError(format!("Failed to serialize manifest: {}", e)))?;

                let mut file = std::fs::File::create(path)
                    .map_err(|e| HoduError::BackendError(format!("Failed to create file: {}", e)))?;

                writeln!(file, "// Hodu Metal Compiler Output")?;
                writeln!(file, "// Dispatch Manifest (JSON):")?;
                for line in manifest_json.lines() {
                    writeln!(file, "// {}", line)?;
                }
                writeln!(file)?;
                write!(file, "{}", kernel_source)?;

                Ok(())
            }
            OutputFormat::Air => {
                // Compile to AIR
                let temp_dir = std::env::temp_dir().join(format!("hodu_metal_{}", std::process::id()));
                std::fs::create_dir_all(&temp_dir)?;

                let metal_path = temp_dir.join("kernels.metal");
                let kernel_source = self.get_kernel_source()?;
                std::fs::write(&metal_path, &kernel_source)?;

                xcrun::compile_metal_to_air(&metal_path, path)?;

                let _ = std::fs::remove_dir_all(&temp_dir);
                Ok(())
            }
            OutputFormat::Metallib => {
                // Compile to metallib
                let artifact = self.compile_snapshot(&snapshot, device)?;

                // Extract metallib from artifact (skip manifest)
                let manifest_len = u64::from_le_bytes(artifact.data[0..8].try_into().unwrap()) as usize;
                let metallib_data = &artifact.data[8 + manifest_len..];

                std::fs::write(path, metallib_data)
                    .map_err(|e| HoduError::BackendError(format!("Failed to write metallib: {}", e)))?;

                // Also write manifest alongside
                let manifest_path = path.with_extension("manifest.json");
                let manifest_json = &artifact.data[8..8 + manifest_len];
                std::fs::write(manifest_path, manifest_json)?;

                Ok(())
            }
            _ => Err(HoduError::UnsupportedOperation(format!(
                "Metal compiler does not support {:?} format",
                format
            ))),
        }
    }
}

/// Parse dtype string to ArtifactDType
/// Metal does NOT support: f8e4m3, f8e5m2, f64
fn parse_dtype(s: &str) -> HoduResult<ArtifactDType> {
    match s.to_lowercase().as_str() {
        "bool" => Ok(ArtifactDType::Bool),
        "bf16" => Ok(ArtifactDType::BF16),
        "f16" => Ok(ArtifactDType::F16),
        "f32" => Ok(ArtifactDType::F32),
        "u8" => Ok(ArtifactDType::U8),
        "u16" => Ok(ArtifactDType::U16),
        "u32" => Ok(ArtifactDType::U32),
        "u64" => Ok(ArtifactDType::U64),
        "i8" => Ok(ArtifactDType::I8),
        "i16" => Ok(ArtifactDType::I16),
        "i32" => Ok(ArtifactDType::I32),
        "i64" => Ok(ArtifactDType::I64),
        // Unsupported types
        "f8e4m3" | "f8e5m2" => Err(HoduError::BackendError(
            format!("Metal does not support {} dtype", s),
        )),
        "f64" => Err(HoduError::BackendError(
            "Metal does not support f64 dtype".into(),
        )),
        _ => Err(HoduError::BackendError(
            format!("Unknown dtype: {}", s),
        )),
    }
}

// Plugin entry points for dynamic loading
// Note: We return a double-boxed pointer to properly handle trait object fat pointers across FFI
#[no_mangle]
pub extern "C" fn hodu_compiler_plugin_create() -> *mut Box<dyn CompilerPlugin> {
    let boxed: Box<dyn CompilerPlugin> = Box::new(MetalCompiler::new());
    Box::into_raw(Box::new(boxed))
}

#[no_mangle]
pub unsafe extern "C" fn hodu_compiler_plugin_destroy(ptr: *mut Box<dyn CompilerPlugin>) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}
