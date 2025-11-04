use super::{instance::CompilerT, types::*};
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    script::builder::ir::*,
    tensor::TensorId,
    types::{Compiler, DType, Device, Layout},
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

    fn build_execution_plan(
        &self,
        function: &Function,
        value_to_tensor: &mut HashMap<ValueId, TensorId>,
    ) -> HoduResult<Vec<CompiledInstruction>> {
        let mut execution_plan = Vec::new();

        // Traverse blocks in order
        for block in &function.blocks {
            for instruction in &block.instructions {
                match instruction {
                    Instruction::Compute {
                        result,
                        op,
                        inputs,
                        attributes,
                    } => {
                        execution_plan.push(CompiledInstruction {
                            op: op.clone(),
                            inputs: inputs.clone(),
                            result: *result,
                            attributes: attributes.clone(),
                        });
                    },
                    Instruction::LoadConstant { result, tensor_id } => {
                        // Store the mapping from value_id to tensor_id
                        value_to_tensor.insert(*result, *tensor_id);

                        // Mark as constant load operation
                        let mut attrs = HashMap::new();
                        attrs.insert("is_constant".to_string(), Attribute::Bool(true));

                        execution_plan.push(CompiledInstruction {
                            op: crate::ops::Op::Dummy,
                            inputs: vec![],
                            result: *result,
                            attributes: attrs,
                        });
                    },
                    Instruction::Phi { .. } => {
                        // TODO: Handle Phi nodes properly
                    },
                }
            }
        }

        Ok(execution_plan)
    }

    fn extract_input_output_mapping(
        &self,
        function: &Function,
    ) -> (HashMap<String, ValueId>, HashMap<String, ValueId>) {
        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        for param in &function.signature.inputs {
            input_mapping.insert(param.name.clone(), param.value_id);
        }

        for param in &function.signature.outputs {
            output_mapping.insert(param.name.clone(), param.value_id);
        }

        (input_mapping, output_mapping)
    }

    fn extract_value_info(&self, function: &Function) -> (HashMap<ValueId, Layout>, HashMap<ValueId, DType>) {
        let mut value_layouts = HashMap::new();
        let mut value_dtypes = HashMap::new();

        // From function signature
        for param in &function.signature.inputs {
            if let Some(ref shape) = param.shape {
                value_layouts.insert(param.value_id, Layout::from_shape(shape));
            }
            if let Some(dtype) = param.dtype {
                value_dtypes.insert(param.value_id, dtype);
            }
        }

        for param in &function.signature.outputs {
            if let Some(ref shape) = param.shape {
                value_layouts.insert(param.value_id, Layout::from_shape(shape));
            }
            if let Some(dtype) = param.dtype {
                value_dtypes.insert(param.value_id, dtype);
            }
        }

        // From value info
        for (value_id, info) in &function.value_info {
            if let Some(ref layout) = info.layout {
                value_layouts.insert(*value_id, layout.clone());
            }
            if let Some(dtype) = info.dtype {
                value_dtypes.insert(*value_id, dtype);
            }
        }

        (value_layouts, value_dtypes)
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
        let function = module
            .functions
            .iter()
            .find(|f| f.name == "forward" || f.name.ends_with("_main"))
            .ok_or_else(|| HoduError::InternalError("No main function found in module".to_string()))?;

        // Build value to tensor mapping
        let mut value_to_tensor = HashMap::new();

        // Build execution plan (this will populate value_to_tensor)
        let execution_plan = self.build_execution_plan(function, &mut value_to_tensor)?;

        // Extract input/output mapping
        let (input_mapping, output_mapping) = self.extract_input_output_mapping(function);

        // Extract value layouts and dtypes
        let (value_layouts, value_dtypes) = self.extract_value_info(function);

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
            compiler: Compiler::XLA,
            device: options.device,
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
