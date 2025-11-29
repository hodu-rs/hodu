//! CPU Runtime implementation

use hodu_plugin::{
    CompiledArtifact, DType, Device, ExecutableModule, ExecutableModuleInner, HoduError,
    HoduResult, OutputFormat, RuntimePlugin, TensorData,
};
use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::path::Path;

/// CPU Runtime Plugin
pub struct CpuRuntime;

impl CpuRuntime {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimePlugin for CpuRuntime {
    fn name(&self) -> &str {
        "cpu"
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::CPU]
    }

    fn loadable_formats(&self, device: Device) -> Vec<OutputFormat> {
        match device {
            Device::CPU => vec![OutputFormat::SharedLib],
            _ => vec![],
        }
    }

    fn load(&self, artifact: &CompiledArtifact, device: Device) -> HoduResult<ExecutableModule> {
        if device != Device::CPU {
            return Err(HoduError::UnsupportedDevice(device));
        }

        if artifact.format != OutputFormat::SharedLib {
            return Err(HoduError::BackendError(format!(
                "CpuRuntime only supports SharedLib format, got {:?}",
                artifact.format
            )));
        }

        // Write artifact data to temp file
        let temp_dir = std::env::temp_dir().join(format!("hodu_cpu_rt_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir)?;

        let lib_ext = if cfg!(target_os = "windows") {
            "dll"
        } else if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let lib_path = temp_dir.join(format!("libhodu_module.{}", lib_ext));
        std::fs::write(&lib_path, &artifact.data)?;

        // Load library
        let library = unsafe {
            Library::new(&lib_path)
                .map_err(|e| HoduError::BackendError(format!("Failed to load library: {}", e)))?
        };

        // Build input/output specs from artifact
        let inputs: Vec<TensorSpec> = artifact
            .inputs
            .iter()
            .map(|i| TensorSpec {
                name: i.name.clone(),
                shape: i.shape.clone(),
                dtype: artifact_dtype_to_string(i.dtype),
            })
            .collect();

        let outputs: Vec<TensorSpec> = artifact
            .outputs
            .iter()
            .map(|o| TensorSpec {
                name: o.name.clone(),
                shape: o.shape.clone(),
                dtype: artifact_dtype_to_string(o.dtype),
            })
            .collect();

        Ok(ExecutableModule::new(CpuExecutable {
            library,
            inputs,
            outputs,
            _temp_dir: temp_dir,
        }))
    }

    fn load_file(&self, path: &Path, device: Device) -> HoduResult<ExecutableModule> {
        if device != Device::CPU {
            return Err(HoduError::UnsupportedDevice(device));
        }

        // Load library directly from path
        let library = unsafe {
            Library::new(path)
                .map_err(|e| HoduError::BackendError(format!("Failed to load library: {}", e)))?
        };

        // For file loading, we need a manifest file
        // Look for .manifest.json next to the library
        let manifest_path = path.with_extension("manifest.json");
        let manifest_json = std::fs::read(&manifest_path).map_err(|e| {
            HoduError::BackendError(format!(
                "Failed to read manifest at {}: {}",
                manifest_path.display(),
                e
            ))
        })?;

        let manifest: Manifest = serde_json::from_slice(&manifest_json)
            .map_err(|e| HoduError::BackendError(format!("Failed to parse manifest: {}", e)))?;

        Ok(ExecutableModule::new(CpuExecutable {
            library,
            inputs: manifest.inputs,
            outputs: manifest.outputs,
            _temp_dir: std::path::PathBuf::new(), // No temp dir for file loading
        }))
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Manifest {
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TensorSpec {
    name: String,
    shape: Vec<usize>,
    dtype: String,
}

/// Executable CPU module
struct CpuExecutable {
    library: Library,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    _temp_dir: std::path::PathBuf,
}

impl ExecutableModuleInner for CpuExecutable {
    fn execute(&self, inputs: &[(&str, TensorData)]) -> HoduResult<HashMap<String, TensorData>> {
        // Create input map
        let input_map: HashMap<&str, &TensorData> = inputs.iter().map(|(n, d)| (*n, d)).collect();

        // Get function pointers
        type InitFn = unsafe extern "C" fn() -> i32;
        type ExecuteFn = unsafe extern "C" fn(...) -> i32;
        type CleanupFn = unsafe extern "C" fn();

        let init: Symbol<InitFn> = unsafe {
            self.library
                .get(b"hodu_init")
                .map_err(|e| HoduError::BackendError(format!("Failed to get hodu_init: {}", e)))?
        };

        let execute: Symbol<ExecuteFn> = unsafe {
            self.library
                .get(b"hodu_execute")
                .map_err(|e| HoduError::BackendError(format!("Failed to get hodu_execute: {}", e)))?
        };

        let cleanup: Symbol<CleanupFn> = unsafe {
            self.library
                .get(b"hodu_cleanup")
                .map_err(|e| HoduError::BackendError(format!("Failed to get hodu_cleanup: {}", e)))?
        };

        // Initialize
        let ret = unsafe { init() };
        if ret != 0 {
            return Err(HoduError::BackendError(format!(
                "hodu_init failed with code {}",
                ret
            )));
        }

        // Prepare input pointers
        let mut input_ptrs: Vec<*const u8> = Vec::new();
        for input_spec in &self.inputs {
            let tensor_data = input_map.get(input_spec.name.as_str()).ok_or_else(|| {
                HoduError::BackendError(format!("Missing input: {}", input_spec.name))
            })?;
            input_ptrs.push(tensor_data.data.as_ptr());
        }

        // Allocate output buffers
        let mut output_buffers: Vec<Vec<u8>> = Vec::new();
        for output_spec in &self.outputs {
            let size: usize = output_spec.shape.iter().product();
            let byte_size = size * dtype_size(&output_spec.dtype);
            output_buffers.push(vec![0u8; byte_size]);
        }

        // Execute - this is tricky because we need variadic function call
        // We'll use a workaround by building the call dynamically
        let ret = unsafe {
            match (self.inputs.len(), self.outputs.len()) {
                (1, 1) => {
                    type Fn1_1 = unsafe extern "C" fn(*const u8, *mut u8) -> i32;
                    let f: Symbol<Fn1_1> = self.library.get(b"hodu_execute").unwrap();
                    f(input_ptrs[0], output_buffers[0].as_mut_ptr())
                }
                (2, 1) => {
                    type Fn2_1 = unsafe extern "C" fn(*const u8, *const u8, *mut u8) -> i32;
                    let f: Symbol<Fn2_1> = self.library.get(b"hodu_execute").unwrap();
                    f(
                        input_ptrs[0],
                        input_ptrs[1],
                        output_buffers[0].as_mut_ptr(),
                    )
                }
                (3, 1) => {
                    type Fn3_1 =
                        unsafe extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> i32;
                    let f: Symbol<Fn3_1> = self.library.get(b"hodu_execute").unwrap();
                    f(
                        input_ptrs[0],
                        input_ptrs[1],
                        input_ptrs[2],
                        output_buffers[0].as_mut_ptr(),
                    )
                }
                (1, 2) => {
                    type Fn1_2 = unsafe extern "C" fn(*const u8, *mut u8, *mut u8) -> i32;
                    let f: Symbol<Fn1_2> = self.library.get(b"hodu_execute").unwrap();
                    f(
                        input_ptrs[0],
                        output_buffers[0].as_mut_ptr(),
                        output_buffers[1].as_mut_ptr(),
                    )
                }
                (2, 2) => {
                    type Fn2_2 =
                        unsafe extern "C" fn(*const u8, *const u8, *mut u8, *mut u8) -> i32;
                    let f: Symbol<Fn2_2> = self.library.get(b"hodu_execute").unwrap();
                    f(
                        input_ptrs[0],
                        input_ptrs[1],
                        output_buffers[0].as_mut_ptr(),
                        output_buffers[1].as_mut_ptr(),
                    )
                }
                _ => {
                    return Err(HoduError::BackendError(format!(
                        "Unsupported input/output count: {}/{}",
                        self.inputs.len(),
                        self.outputs.len()
                    )));
                }
            }
        };

        // Cleanup
        unsafe { cleanup() };

        if ret != 0 {
            return Err(HoduError::BackendError(format!(
                "hodu_execute failed with code {}",
                ret
            )));
        }

        // Build output map
        let mut outputs: HashMap<String, TensorData> = HashMap::new();
        for (i, output_spec) in self.outputs.iter().enumerate() {
            let dtype = parse_dtype(&output_spec.dtype)?;
            outputs.insert(
                output_spec.name.clone(),
                TensorData::new(output_buffers[i].clone(), output_spec.shape.clone(), dtype),
            );
        }

        Ok(outputs)
    }
}

// CpuExecutable is Send+Sync safe
unsafe impl Send for CpuExecutable {}
unsafe impl Sync for CpuExecutable {}

fn dtype_size(dtype: &str) -> usize {
    match dtype {
        "bool" | "u8" | "i8" | "f8e4m3" | "f8e5m2" => 1,
        "bf16" | "f16" | "u16" | "i16" => 2,
        "f32" | "u32" | "i32" => 4,
        "f64" | "u64" | "i64" => 8,
        _ => 4,
    }
}

fn parse_dtype(dtype_str: &str) -> HoduResult<DType> {
    match dtype_str.to_lowercase().as_str() {
        "bool" => Ok(DType::BOOL),
        "bf16" => Ok(DType::BF16),
        "f16" => Ok(DType::F16),
        "f32" => Ok(DType::F32),
        "u8" => Ok(DType::U8),
        "u32" => Ok(DType::U32),
        "i8" => Ok(DType::I8),
        "i32" => Ok(DType::I32),
        _ => Ok(DType::F32),
    }
}

fn artifact_dtype_to_string(dtype: hodu_plugin::ArtifactDType) -> String {
    match dtype {
        hodu_plugin::ArtifactDType::Bool => "bool".to_string(),
        hodu_plugin::ArtifactDType::F8E4M3 => "f8e4m3".to_string(),
        hodu_plugin::ArtifactDType::F8E5M2 => "f8e5m2".to_string(),
        hodu_plugin::ArtifactDType::BF16 => "bf16".to_string(),
        hodu_plugin::ArtifactDType::F16 => "f16".to_string(),
        hodu_plugin::ArtifactDType::F32 => "f32".to_string(),
        hodu_plugin::ArtifactDType::F64 => "f64".to_string(),
        hodu_plugin::ArtifactDType::U8 => "u8".to_string(),
        hodu_plugin::ArtifactDType::U16 => "u16".to_string(),
        hodu_plugin::ArtifactDType::U32 => "u32".to_string(),
        hodu_plugin::ArtifactDType::U64 => "u64".to_string(),
        hodu_plugin::ArtifactDType::I8 => "i8".to_string(),
        hodu_plugin::ArtifactDType::I16 => "i16".to_string(),
        hodu_plugin::ArtifactDType::I32 => "i32".to_string(),
        hodu_plugin::ArtifactDType::I64 => "i64".to_string(),
    }
}

// Plugin entry points
#[no_mangle]
pub extern "C" fn hodu_runtime_plugin_create() -> *mut hodu_plugin::RuntimePluginHandle {
    let boxed: Box<dyn RuntimePlugin> = Box::new(CpuRuntime::new());
    hodu_plugin::RuntimePluginHandle::from_boxed(boxed)
}

#[no_mangle]
pub unsafe extern "C" fn hodu_runtime_plugin_destroy(ptr: *mut hodu_plugin::RuntimePluginHandle) {
    if !ptr.is_null() {
        drop(hodu_plugin::RuntimePluginHandle::into_boxed(ptr));
    }
}
