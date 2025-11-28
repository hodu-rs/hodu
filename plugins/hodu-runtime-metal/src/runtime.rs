//! Metal Runtime implementation

use crate::dispatch::{dtype_size, DispatchManifest};
use hodu_plugin::{
    CompiledArtifact, DType, Device, ExecutableModule, ExecutableModuleInner, HoduError, HoduResult, OutputFormat,
    RuntimePlugin, TensorData,
};
use metal::{CommandQueue, ComputePipelineState, Device as MTLDevice, MTLSize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Metal Runtime Plugin
pub struct MetalRuntime {
    device: MTLDevice,
    command_queue: CommandQueue,
}

impl MetalRuntime {
    pub fn new() -> HoduResult<Self> {
        let device =
            MTLDevice::system_default().ok_or_else(|| HoduError::BackendError("No Metal device found".into()))?;
        let command_queue = device.new_command_queue();

        Ok(Self { device, command_queue })
    }
}

impl Default for MetalRuntime {
    fn default() -> Self {
        Self::new().expect("Failed to create Metal runtime")
    }
}

impl RuntimePlugin for MetalRuntime {
    fn name(&self) -> &str {
        "metal"
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::Metal]
    }

    fn loadable_formats(&self, device: Device) -> Vec<OutputFormat> {
        match device {
            Device::Metal => vec![OutputFormat::Metallib],
            _ => vec![],
        }
    }

    fn load(&self, artifact: &CompiledArtifact, device: Device) -> HoduResult<ExecutableModule> {
        if device != Device::Metal {
            return Err(HoduError::UnsupportedDevice(device));
        }

        // Unpack artifact: [manifest_len (8 bytes)][manifest_json][metallib_data]
        if artifact.data.len() < 8 {
            return Err(HoduError::BackendError("Invalid artifact data".into()));
        }

        let manifest_len = u64::from_le_bytes(artifact.data[0..8].try_into().unwrap()) as usize;
        let manifest_json = &artifact.data[8..8 + manifest_len];
        let metallib_data = &artifact.data[8 + manifest_len..];

        let manifest = DispatchManifest::from_json(manifest_json)
            .ok_or_else(|| HoduError::BackendError("Failed to parse manifest".into()))?;

        // Load metallib
        let library = self
            .device
            .new_library_with_data(metallib_data)
            .map_err(|e| HoduError::BackendError(format!("Failed to load metallib: {}", e)))?;

        // Create pipeline states for all kernels
        let mut pipelines: HashMap<String, ComputePipelineState> = HashMap::new();
        for dispatch in &manifest.dispatches {
            if dispatch.kernel_name == "noop" {
                continue;
            }
            if pipelines.contains_key(&dispatch.kernel_name) {
                continue;
            }

            let function = library.get_function(&dispatch.kernel_name, None).map_err(|e| {
                HoduError::BackendError(format!("Failed to get function '{}': {}", dispatch.kernel_name, e))
            })?;

            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| {
                    HoduError::BackendError(format!(
                        "Failed to create pipeline for '{}': {}",
                        dispatch.kernel_name, e
                    ))
                })?;

            pipelines.insert(dispatch.kernel_name.clone(), pipeline);
        }

        Ok(ExecutableModule::new(MetalExecutable {
            device: self.device.clone(),
            command_queue: self.command_queue.clone(),
            manifest,
            pipelines: Arc::new(pipelines),
        }))
    }

    fn load_file(&self, path: &Path, device: Device) -> HoduResult<ExecutableModule> {
        if device != Device::Metal {
            return Err(HoduError::UnsupportedDevice(device));
        }

        // Load metallib file
        let metallib_data =
            std::fs::read(path).map_err(|e| HoduError::BackendError(format!("Failed to read metallib: {}", e)))?;

        // Load manifest file (same path with .manifest.json extension)
        let manifest_path = path.with_extension("manifest.json");
        let manifest_json = std::fs::read(&manifest_path).map_err(|e| {
            HoduError::BackendError(format!("Failed to read manifest at {}: {}", manifest_path.display(), e))
        })?;

        let manifest = DispatchManifest::from_json(&manifest_json)
            .ok_or_else(|| HoduError::BackendError("Failed to parse manifest".into()))?;

        // Load metallib
        let library = self
            .device
            .new_library_with_data(&metallib_data)
            .map_err(|e| HoduError::BackendError(format!("Failed to load metallib: {}", e)))?;

        // Create pipeline states
        let mut pipelines: HashMap<String, ComputePipelineState> = HashMap::new();
        for dispatch in &manifest.dispatches {
            if dispatch.kernel_name == "noop" {
                continue;
            }
            if pipelines.contains_key(&dispatch.kernel_name) {
                continue;
            }

            let function = library.get_function(&dispatch.kernel_name, None).map_err(|e| {
                HoduError::BackendError(format!("Failed to get function '{}': {}", dispatch.kernel_name, e))
            })?;

            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| {
                    HoduError::BackendError(format!(
                        "Failed to create pipeline for '{}': {}",
                        dispatch.kernel_name, e
                    ))
                })?;

            pipelines.insert(dispatch.kernel_name.clone(), pipeline);
        }

        Ok(ExecutableModule::new(MetalExecutable {
            device: self.device.clone(),
            command_queue: self.command_queue.clone(),
            manifest,
            pipelines: Arc::new(pipelines),
        }))
    }
}

/// Executable Metal module
struct MetalExecutable {
    device: MTLDevice,
    command_queue: CommandQueue,
    manifest: DispatchManifest,
    pipelines: Arc<HashMap<String, ComputePipelineState>>,
}

impl ExecutableModuleInner for MetalExecutable {
    fn execute(&self, inputs: &[(&str, TensorData)]) -> HoduResult<HashMap<String, TensorData>> {
        // Create input map
        let input_map: HashMap<&str, &TensorData> = inputs.iter().map(|(n, d)| (*n, d)).collect();

        // Allocate all buffers
        let mut buffers: Vec<metal::Buffer> = Vec::with_capacity(self.manifest.num_buffers);

        // First pass: calculate buffer sizes
        let mut buffer_sizes: Vec<usize> = vec![0; self.manifest.num_buffers];

        for input in &self.manifest.inputs {
            let size: usize = input.shape.iter().product();
            let byte_size = size * dtype_size(&input.dtype);
            buffer_sizes[input.buffer_id] = byte_size;
        }

        for constant in &self.manifest.constants {
            buffer_sizes[constant.buffer_id] = constant.data.len();
        }

        for dispatch in &self.manifest.dispatches {
            // Estimate output buffer size from grid_size
            let byte_size = dispatch.grid_size * 4; // Assume f32 for now
            if buffer_sizes[dispatch.output_buffer] < byte_size {
                buffer_sizes[dispatch.output_buffer] = byte_size;
            }
        }

        // Allocate buffers
        for size in &buffer_sizes {
            let size = if *size == 0 { 16 } else { *size }; // Minimum 16 bytes
            let buffer = self
                .device
                .new_buffer(size as u64, metal::MTLResourceOptions::StorageModeShared);
            buffers.push(buffer);
        }

        // Copy input data
        for input_spec in &self.manifest.inputs {
            let tensor_data = input_map
                .get(input_spec.name.as_str())
                .ok_or_else(|| HoduError::BackendError(format!("Missing input: {}", input_spec.name)))?;

            let buffer = &buffers[input_spec.buffer_id];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor_data.data.as_ptr(),
                    buffer.contents() as *mut u8,
                    tensor_data.data.len(),
                );
            }
        }

        // Copy constant data
        for constant in &self.manifest.constants {
            let buffer = &buffers[constant.buffer_id];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    constant.data.as_ptr(),
                    buffer.contents() as *mut u8,
                    constant.data.len(),
                );
            }
        }

        // Execute dispatches
        let command_buffer = self.command_queue.new_command_buffer();

        for dispatch in &self.manifest.dispatches {
            if dispatch.kernel_name == "noop" {
                // Copy input to output for noop
                if !dispatch.input_buffers.is_empty() {
                    let src = &buffers[dispatch.input_buffers[0]];
                    let dst = &buffers[dispatch.output_buffer];
                    let blit = command_buffer.new_blit_command_encoder();
                    blit.copy_from_buffer(src, 0, dst, 0, src.length().min(dst.length()));
                    blit.end_encoding();
                }
                continue;
            }

            let pipeline = self
                .pipelines
                .get(&dispatch.kernel_name)
                .ok_or_else(|| HoduError::BackendError(format!("Pipeline not found: {}", dispatch.kernel_name)))?;

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);

            // Set input buffers
            for (i, &buf_id) in dispatch.input_buffers.iter().enumerate() {
                encoder.set_buffer(i as u64, Some(&buffers[buf_id]), 0);
            }

            // Set output buffer
            let output_idx = dispatch.input_buffers.len();
            encoder.set_buffer(output_idx as u64, Some(&buffers[dispatch.output_buffer]), 0);

            // Set metadata buffer
            let metadata_bytes: Vec<u8> = dispatch
                .metadata
                .iter()
                .flat_map(|&x| (x as u64).to_le_bytes())
                .collect();
            let metadata_buffer = self.device.new_buffer_with_data(
                metadata_bytes.as_ptr() as *const _,
                metadata_bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer((output_idx + 1) as u64, Some(&metadata_buffer), 0);

            // Dispatch
            let thread_group_size =
                MTLSize::new(pipeline.thread_execution_width().min(dispatch.grid_size as u64), 1, 1);
            let grid_size = MTLSize::new(dispatch.grid_size as u64, 1, 1);

            encoder.dispatch_threads(grid_size, thread_group_size);
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read outputs
        let mut outputs: HashMap<String, TensorData> = HashMap::new();

        for output_spec in &self.manifest.outputs {
            let buffer = &buffers[output_spec.buffer_id];
            let size: usize = output_spec.shape.iter().product();
            let byte_size = size * dtype_size(&output_spec.dtype);

            let mut data = vec![0u8; byte_size];
            unsafe {
                std::ptr::copy_nonoverlapping(buffer.contents() as *const u8, data.as_mut_ptr(), byte_size);
            }

            let dtype = parse_dtype(&output_spec.dtype)?;
            outputs.insert(
                output_spec.name.clone(),
                TensorData::new(data, output_spec.shape.clone(), dtype),
            );
        }

        Ok(outputs)
    }
}

// Marked as Send+Sync (Metal handles are thread-safe)
unsafe impl Send for MetalExecutable {}
unsafe impl Sync for MetalExecutable {}

fn parse_dtype(dtype_str: &str) -> HoduResult<DType> {
    match dtype_str.to_lowercase().as_str() {
        "bool" => Ok(DType::BOOL),
        "bf16" => Ok(DType::BF16),
        "f16" => Ok(DType::F16),
        "f32" => Ok(DType::F32),
        "u8" => Ok(DType::U8),
        "u16" => Ok(DType::U16),
        "u32" => Ok(DType::U32),
        "u64" => Ok(DType::U64),
        "i8" => Ok(DType::I8),
        "i16" => Ok(DType::I16),
        "i32" => Ok(DType::I32),
        "i64" => Ok(DType::I64),
        _ => Ok(DType::F32),
    }
}

// Plugin entry points
// Using opaque handle type for FFI safety
#[no_mangle]
pub extern "C" fn hodu_runtime_plugin_create() -> *mut hodu_plugin::RuntimePluginHandle {
    match MetalRuntime::new() {
        Ok(runtime) => {
            let boxed: Box<dyn RuntimePlugin> = Box::new(runtime);
            hodu_plugin::RuntimePluginHandle::from_boxed(boxed)
        },
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn hodu_runtime_plugin_destroy(ptr: *mut hodu_plugin::RuntimePluginHandle) {
    if !ptr.is_null() {
        drop(hodu_plugin::RuntimePluginHandle::into_boxed(ptr));
    }
}
