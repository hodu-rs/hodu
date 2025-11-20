use super::{base, instance::CompilerT, types::*};
use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    script::builder::ir::*,
    types::{Compiler, Device},
};

/// XLA compiler (feature-gated)
#[derive(Debug)]
pub struct XlaCompiler {
    device: Device,
}

impl XlaCompiler {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl CompilerT for XlaCompiler {
    fn compiler_type(&self) -> Compiler {
        Compiler::XLA
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

        // XLA doesn't use constant_storages (XLA executor builds constants directly)
        let constant_storages = HashMap::new();

        // Calculate max ValueId for efficient storage allocation during execution
        let max_value_id = execution_plan
            .iter()
            .flat_map(|instr| instr.inputs.iter().chain(iter::once(&instr.result)))
            .chain(input_mapping.values())
            .chain(output_mapping.values())
            .map(|vid| vid.0)
            .max()
            .unwrap_or(0);

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
            compiler: Compiler::XLA,
            device: options.device,
            max_value_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xla_compiler_creation() {
        let compiler = XlaCompiler::new(Device::CPU);
        assert_eq!(compiler.compiler_type(), Compiler::XLA);
        assert_eq!(compiler.device(), Device::CPU);
    }

    #[test]
    fn test_xla_compile_not_implemented() {
        let module = Module::new("test".to_string());
        let compiler = XlaCompiler::new(Device::CPU);
        let options = CompileOptions::default();
        let result = compiler.compile(&module, options);
        assert!(result.is_err());
    }
}
