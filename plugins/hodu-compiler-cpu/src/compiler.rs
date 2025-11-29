//! CPU Compiler Plugin implementation
//!
//! Implements CompilerPlugin trait for CPU backend.

use crate::codegen::generate_c_code;
use crate::dispatch::DispatchManifest;
use crate::toolchain::{find_compiler, CCompiler};
use hodu_core::script::Script;
use hodu_plugin::{
    ArtifactDType, ArtifactSymbols, ArtifactTensorInfo, CompiledArtifact, CompilerPlugin, Device, HoduError,
    HoduResult, OutputFormat,
};
use std::fs;
use std::path::{Path, PathBuf};

/// CPU Compiler Plugin
pub struct CpuCompiler {
    /// C compiler to use (lazily initialized)
    compiler: Option<CCompiler>,
    /// Path to libhodu_cpu_kernels static library
    kernels_lib_path: Option<PathBuf>,
    /// Optimization level (0-3)
    opt_level: u8,
}

impl Default for CpuCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuCompiler {
    pub fn new() -> Self {
        Self {
            compiler: None,
            kernels_lib_path: None,
            opt_level: 2,
        }
    }

    pub fn with_compiler(mut self, compiler: CCompiler) -> Self {
        self.compiler = Some(compiler);
        self
    }

    pub fn with_kernels_lib(mut self, path: PathBuf) -> Self {
        self.kernels_lib_path = Some(path);
        self
    }

    pub fn with_opt_level(mut self, level: u8) -> Self {
        self.opt_level = level.min(3);
        self
    }

    fn get_compiler(&self) -> HoduResult<CCompiler> {
        match &self.compiler {
            Some(c) => Ok(*c),
            None => find_compiler(),
        }
    }

    fn find_kernels_lib(&self) -> PathBuf {
        self.kernels_lib_path.clone().unwrap_or_else(|| {
            // Try to find in standard locations
            // 1. Environment variable
            if let Ok(path) = std::env::var("HODU_CPU_KERNELS_LIB") {
                return PathBuf::from(path);
            }
            // 2. Relative to executable
            if let Ok(exe) = std::env::current_exe() {
                if let Some(parent) = exe.parent() {
                    let lib_path = parent.join("libhodu_cpu_kernels.a");
                    if lib_path.exists() {
                        return lib_path;
                    }
                }
            }
            // 3. Default path (will likely fail, but gives informative error)
            PathBuf::from("libhodu_cpu_kernels.a")
        })
    }
}

impl CompilerPlugin for CpuCompiler {
    fn name(&self) -> &str {
        "cpu"
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::CPU]
    }

    fn supported_formats(&self, device: Device) -> Vec<OutputFormat> {
        match device {
            Device::CPU => vec![OutputFormat::SharedLib],
            _ => vec![],
        }
    }

    fn compile(&self, script: &Script, device: Device) -> HoduResult<CompiledArtifact> {
        if device != Device::CPU {
            return Err(HoduError::UnsupportedDevice(device));
        }

        // Find C compiler
        let compiler = self.get_compiler()?;

        // Get snapshot from script
        let snapshot = script.snapshot();

        // Generate dispatch manifest
        let manifest = DispatchManifest::from_snapshot(&snapshot);

        // Generate C code
        let c_code = generate_c_code(&manifest);

        // Create temp directory for compilation
        let temp_dir = std::env::temp_dir().join(format!("hodu_cpu_{}", std::process::id()));
        fs::create_dir_all(&temp_dir)?;

        // Write C source
        let c_source_path = temp_dir.join("hodu_compiled.c");
        fs::write(&c_source_path, &c_code)?;

        // Output shared library path
        let output_name = snapshot.name.as_ref().map(|s| s.as_str()).unwrap_or("hodu_compiled");
        let lib_ext = if cfg!(target_os = "windows") {
            "dll"
        } else if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let output_path = temp_dir.join(format!("lib{}.{}", output_name, lib_ext));

        // Find kernels library
        let kernels_lib = self.find_kernels_lib();

        // Compile to shared library
        compiler.compile_to_shared_lib(&c_source_path, &output_path, &kernels_lib, self.opt_level)?;

        // Read compiled library
        let lib_data = fs::read(&output_path)?;

        // Build artifact
        let mut artifact = CompiledArtifact::new(OutputFormat::SharedLib, Device::CPU, lib_data);

        // Add input info
        for input in &manifest.inputs {
            artifact.inputs.push(ArtifactTensorInfo {
                name: input.name.clone(),
                shape: input.shape.clone(),
                dtype: string_to_artifact_dtype(&input.dtype),
            });
        }

        // Add output info
        for output in &manifest.outputs {
            artifact.outputs.push(ArtifactTensorInfo {
                name: output.name.clone(),
                shape: output.shape.clone(),
                dtype: string_to_artifact_dtype(&output.dtype),
            });
        }

        // Set symbols
        artifact.symbols = Some(ArtifactSymbols {
            entry_point: "hodu_execute".to_string(),
            exports: vec![
                "hodu_init".to_string(),
                "hodu_execute".to_string(),
                "hodu_cleanup".to_string(),
            ],
        });

        // Cleanup temp directory
        let _ = fs::remove_dir_all(&temp_dir);

        Ok(artifact)
    }

    fn build(&self, script: &Script, device: Device, format: OutputFormat, path: &Path) -> HoduResult<()> {
        if device != Device::CPU {
            return Err(HoduError::UnsupportedDevice(device));
        }

        if format != OutputFormat::SharedLib {
            return Err(HoduError::BackendError(format!(
                "CPU compiler only supports SharedLib format, got {:?}",
                format
            )));
        }

        // Find C compiler
        let compiler = self.get_compiler()?;

        // Get snapshot from script
        let snapshot = script.snapshot();

        // Generate dispatch manifest
        let manifest = DispatchManifest::from_snapshot(&snapshot);

        // Generate C code
        let c_code = generate_c_code(&manifest);

        // Create temp directory for compilation
        let temp_dir = std::env::temp_dir().join(format!("hodu_cpu_{}", std::process::id()));
        fs::create_dir_all(&temp_dir)?;

        // Write C source
        let c_source_path = temp_dir.join("hodu_compiled.c");
        fs::write(&c_source_path, &c_code)?;

        // Find kernels library
        let kernels_lib = self.find_kernels_lib();

        // Compile to shared library
        compiler.compile_to_shared_lib(&c_source_path, path, &kernels_lib, self.opt_level)?;

        // Cleanup temp directory
        let _ = fs::remove_dir_all(&temp_dir);

        Ok(())
    }
}

fn string_to_artifact_dtype(dtype: &str) -> ArtifactDType {
    match dtype {
        "bool" => ArtifactDType::Bool,
        "f8e4m3" => ArtifactDType::F8E4M3,
        "f8e5m2" => ArtifactDType::F8E5M2,
        "bf16" => ArtifactDType::BF16,
        "f16" => ArtifactDType::F16,
        "f32" => ArtifactDType::F32,
        "f64" => ArtifactDType::F64,
        "u8" => ArtifactDType::U8,
        "u16" => ArtifactDType::U16,
        "u32" => ArtifactDType::U32,
        "u64" => ArtifactDType::U64,
        "i8" => ArtifactDType::I8,
        "i16" => ArtifactDType::I16,
        "i32" => ArtifactDType::I32,
        "i64" => ArtifactDType::I64,
        _ => ArtifactDType::F32,
    }
}

// Plugin entry points for dynamic loading
#[no_mangle]
pub extern "C" fn hodu_compiler_plugin_create() -> *mut hodu_plugin::CompilerPluginHandle {
    let boxed: Box<dyn CompilerPlugin> = Box::new(CpuCompiler::new());
    hodu_plugin::CompilerPluginHandle::from_boxed(boxed)
}

#[no_mangle]
pub unsafe extern "C" fn hodu_compiler_plugin_destroy(ptr: *mut hodu_plugin::CompilerPluginHandle) {
    if !ptr.is_null() {
        drop(hodu_plugin::CompilerPluginHandle::into_boxed(ptr));
    }
}
