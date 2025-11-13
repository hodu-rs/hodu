use super::{base, instance::CompilerT, types::*};
use crate::{
    be::storage::BackendStorage,
    be_cpu::storage::CpuStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    script::builder::ir::*,
    types::{Compiler, Device},
};

/// HODU native compiler
#[derive(Debug)]
pub struct HoduCompiler {
    device: Device,
}

impl HoduCompiler {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Convert constant data to target device storage
    fn convert_constants_to_device(
        &self,
        constant_data: &HashMap<crate::tensor::TensorId, ConstantData>,
        device: Device,
    ) -> HoduResult<HashMap<crate::tensor::TensorId, Arc<BackendStorage>>> {
        let mut constant_storages = HashMap::new();

        for (tensor_id, constant) in constant_data {
            // Create CPU storage from constant data
            let cpu_storage = CpuStorage::from_bytes(&constant.data, constant.dtype)?;

            // Convert to target device
            let storage = match device {
                Device::CPU => BackendStorage::CPU(cpu_storage),
                #[cfg(feature = "cuda")]
                Device::CUDA(device_id) => BackendStorage::CUDA(
                    crate::be_cuda::storage::CudaStorage::from_cpu_storage(&cpu_storage, device_id)?,
                ),
                #[cfg(feature = "metal")]
                Device::Metal => {
                    BackendStorage::Metal(crate::be_metal::storage::MetalStorage::from_cpu_storage(&cpu_storage)?)
                },
                #[allow(unreachable_patterns)]
                _ => return Err(HoduError::UnsupportedDevice(device)),
            };

            constant_storages.insert(*tensor_id, Arc::new(storage));
        }

        Ok(constant_storages)
    }
}

impl CompilerT for HoduCompiler {
    fn compiler_type(&self) -> Compiler {
        Compiler::HODU
    }

    fn device(&self) -> Device {
        self.device
    }

    fn compile(&self, module: &Module, options: CompileOptions) -> HoduResult<CompiledModule> {
        self.validate(module)?;

        // Find the main function (forward or first function ending with _main)
        let function = base::find_main_function(module)
            .ok_or_else(|| HoduError::MissingFunction("no main function found in module".to_string()))?;

        // Build value to tensor mapping
        let mut value_to_tensor = HashMap::new();

        // Build execution plan (this will populate value_to_tensor)
        let execution_plan = base::build_execution_plan(function, &mut value_to_tensor)?;

        // Extract input/output mapping
        let (input_mapping, output_mapping) = base::extract_input_output_mapping(function);

        // Extract value layouts and dtypes
        let (value_layouts, value_dtypes) = base::extract_value_info(function);

        // Copy constant data
        let constant_data = module.constants.clone();

        // Pre-convert constants to target device
        let constant_storages = self.convert_constants_to_device(&constant_data, options.device)?;

        Ok(CompiledModule {
            module: module.clone(),
            execution_plan,
            input_mapping,
            output_mapping,
            constant_data,
            constant_storages,
            value_layouts,
            value_dtypes,
            value_to_tensor,
            compiler: Compiler::HODU,
            device: options.device,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hodu_compiler_creation() {
        let compiler = HoduCompiler::new(Device::CPU);
        assert_eq!(compiler.compiler_type(), Compiler::HODU);
        assert_eq!(compiler.device(), Device::CPU);
    }

    #[test]
    fn test_compile_empty_module_fails() {
        let module = Module::new("empty".to_string());
        let compiler = HoduCompiler::new(Device::CPU);
        let options = CompileOptions::default();
        let result = compiler.compile(&module, options);
        assert!(result.is_err());
    }
}
