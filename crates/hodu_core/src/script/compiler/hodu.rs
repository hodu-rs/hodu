use super::{base, instance::CompilerT, types::*};
use crate::{
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

        Ok(CompiledModule {
            module: module.clone(),
            execution_plan,
            input_mapping,
            output_mapping,
            constant_data,
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
