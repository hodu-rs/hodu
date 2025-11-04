use super::{instance::ExecutorT, types::*};
use crate::{
    be::storage::BackendStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    script::{
        builder::ir::{Attribute, ValueId},
        compiler::CompiledModule,
    },
    tensor::from_storage,
    types::{Compiler, Device, Layout},
};

/// HODU native executor
#[derive(Debug)]
pub struct HoduExecutor {
    device: Device,
}

impl HoduExecutor {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Validate that inputs match the compiled module's requirements
    fn validate_inputs(&self, compiled: &CompiledModule, inputs: &ExecutionInputs<'_>) -> HoduResult<()> {
        for (name, _value_id) in &compiled.input_mapping {
            if !inputs.contains_key(name.as_str()) {
                return Err(HoduError::InternalError(format!("Missing required input: {}", name)));
            }
        }
        Ok(())
    }
}

impl ExecutorT for HoduExecutor {
    fn compiler_type(&self) -> Compiler {
        Compiler::HODU
    }

    fn device(&self) -> Device {
        self.device
    }

    fn execute(&self, compiled: &CompiledModule, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        self.validate_inputs(compiled, &inputs)?;

        // Map value_id to storage for runtime
        let mut value_storage: HashMap<ValueId, Arc<BackendStorage>> = HashMap::new();

        // Load input values
        for (name, value_id) in &compiled.input_mapping {
            let tensor = inputs
                .get(name.as_str())
                .ok_or_else(|| HoduError::InternalError(format!("Missing input: {}", name)))?;

            let storage = tensor.with_storage(|s| Ok(Arc::new(s.clone())))?;

            value_storage.insert(*value_id, storage);
        }

        // Execute instructions in order
        for instr in &compiled.execution_plan {
            // Check if this is a constant load
            if let Some(Attribute::Bool(true)) = instr.attributes.get("is_constant") {
                // Load constant from value_to_tensor mapping
                if let Some(tensor_id) = compiled.value_to_tensor.get(&instr.result) {
                    // Get the tensor and extract storage
                    let tensor = crate::tensor::tensor_from_id(*tensor_id);
                    let storage = if tensor.device() != self.device {
                        // Convert to target device
                        let cpu_storage = tensor.with_storage(|s| s.to_cpu_storage())?;
                        let new_storage = match self.device {
                            Device::CPU => BackendStorage::CPU(cpu_storage),
                            #[cfg(feature = "metal")]
                            Device::Metal => BackendStorage::Metal(
                                crate::be_metal::storage::MetalStorage::from_cpu_storage(&cpu_storage)?,
                            ),
                            #[allow(unreachable_patterns)]
                            _ => {
                                return Err(HoduError::InternalError(format!(
                                    "Unsupported device: {:?}",
                                    self.device
                                )))
                            },
                        };
                        Arc::new(new_storage)
                    } else {
                        tensor.with_storage(|s| Ok(Arc::new(s.clone())))?
                    };
                    value_storage.insert(instr.result, storage);
                    continue;
                }
                return Err(HoduError::InternalError("Failed to load constant".to_string()));
            }

            // Get input storages for this operation
            let input_storages: Vec<Arc<BackendStorage>> = instr
                .inputs
                .iter()
                .filter_map(|vid| value_storage.get(vid).cloned())
                .collect();

            if input_storages.len() != instr.inputs.len() {
                return Err(HoduError::InternalError(format!(
                    "Missing inputs for operation: {:?}. Expected {}, got {}",
                    instr.op,
                    instr.inputs.len(),
                    input_storages.len()
                )));
            }

            // Get input layouts
            let input_layouts: Vec<Layout> = instr
                .inputs
                .iter()
                .filter_map(|vid| compiled.value_layouts.get(vid).cloned())
                .collect();

            if input_layouts.len() != instr.inputs.len() {
                return Err(HoduError::InternalError(format!(
                    "Missing layouts for operation: {:?}. Expected {}, got {}",
                    instr.op,
                    instr.inputs.len(),
                    input_layouts.len()
                )));
            }

            // Execute the operation
            let result_storage =
                self.execute_operation(&instr.op, &input_storages, &input_layouts, &instr.attributes)?;
            value_storage.insert(instr.result, Arc::new(result_storage));
        }

        // Collect outputs
        let mut outputs = HashMap::new();
        for (name, value_id) in &compiled.output_mapping {
            let storage = value_storage
                .get(value_id)
                .ok_or_else(|| HoduError::InternalError(format!("Missing output: {}", name)))?;

            let layout = compiled
                .value_layouts
                .get(value_id)
                .ok_or_else(|| HoduError::InternalError(format!("Missing layout for output: {}", name)))?
                .clone();

            let tensor = from_storage((**storage).clone(), layout, true, false);
            outputs.insert(name.clone(), tensor);
        }

        Ok(outputs)
    }
}

impl HoduExecutor {
    /// Execute a single operation
    fn execute_operation(
        &self,
        op: &Op,
        inputs: &[Arc<BackendStorage>],
        layouts: &[Layout],
        attributes: &HashMap<String, Attribute>,
    ) -> HoduResult<BackendStorage> {
        use crate::ops::*;

        match op {
            Op::Binary(_) => {
                if inputs.len() != 2 || layouts.len() != 2 {
                    return Err(HoduError::InternalError(format!(
                        "Binary operation requires 2 inputs and layouts, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                inputs[0].call_binary(&inputs[1], &layouts[0], &layouts[1], op.clone())
            },

            Op::BinaryLogical(_) => {
                if inputs.len() != 2 || layouts.len() != 2 {
                    return Err(HoduError::InternalError(format!(
                        "BinaryLogical operation requires 2 inputs and layouts, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                inputs[0].call_binary_logical(&inputs[1], &layouts[0], &layouts[1], op.clone())
            },

            Op::Cmp(_) => {
                if inputs.len() != 2 || layouts.len() != 2 {
                    return Err(HoduError::InternalError(format!(
                        "Cmp operation requires 2 inputs and layouts, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                inputs[0].call_cmp(&inputs[1], &layouts[0], &layouts[1], op.clone())
            },

            Op::CmpScalar(_) => {
                if inputs.len() != 1 || layouts.len() != 1 {
                    return Err(HoduError::InternalError(format!(
                        "CmpScalar operation requires 1 input and layout, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                let scalar = attributes
                    .get("scalar")
                    .and_then(|a| if let Attribute::Scalar(s) = a { Some(*s) } else { None })
                    .ok_or_else(|| HoduError::InternalError("Missing scalar attribute".to_string()))?;
                inputs[0].call_cmp_scalar(&layouts[0], scalar, op.clone())
            },

            Op::Unary(_) => {
                if inputs.len() != 1 || layouts.len() != 1 {
                    return Err(HoduError::InternalError(format!(
                        "Unary operation requires 1 input and layout, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                inputs[0].call_unary(&layouts[0], op.clone())
            },

            Op::UnaryLogical(_) => {
                if inputs.len() != 1 || layouts.len() != 1 {
                    return Err(HoduError::InternalError(format!(
                        "UnaryLogical operation requires 1 input and layout, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                inputs[0].call_unary_logical(&layouts[0], op.clone())
            },

            Op::UnaryScalar(_) => {
                if inputs.len() != 1 || layouts.len() != 1 {
                    return Err(HoduError::InternalError(format!(
                        "UnaryScalar operation requires 1 input and layout, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                let scalar = attributes
                    .get("scalar")
                    .and_then(|a| if let Attribute::Scalar(s) = a { Some(*s) } else { None })
                    .ok_or_else(|| HoduError::InternalError("Missing scalar attribute".to_string()))?;
                inputs[0].call_unary_scalar(&layouts[0], scalar, op.clone())
            },

            Op::Matrix(MatrixOp::Matmul) => {
                if inputs.len() != 2 || layouts.len() != 2 {
                    return Err(HoduError::InternalError(format!(
                        "Matmul operation requires 2 inputs and layouts, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                inputs[0].call_matmul(&inputs[1], &layouts[0], &layouts[1], op.clone())
            },

            Op::Matrix(MatrixOp::Dot) => {
                if inputs.len() != 2 || layouts.len() != 2 {
                    return Err(HoduError::InternalError(format!(
                        "Dot operation requires 2 inputs and layouts, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                inputs[0].call_dot(&inputs[1], &layouts[0], &layouts[1], op.clone())
            },

            Op::Reduce(_) => {
                if inputs.len() != 1 || layouts.len() != 1 {
                    return Err(HoduError::InternalError(format!(
                        "Reduce operation requires 1 input and layout, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                let dims: Vec<u32> = attributes
                    .get("dims")
                    .and_then(|a| {
                        if let Attribute::Scalars(s) = a {
                            Some(s.iter().map(|sc| sc.to_u32()).collect())
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing dims attribute".to_string()))?;
                let keep_dim = attributes
                    .get("keep_dim")
                    .and_then(|a| if let Attribute::Bool(b) = a { Some(*b) } else { None })
                    .unwrap_or(false);
                inputs[0].call_reduce(&layouts[0], &dims, keep_dim, op.clone())
            },

            Op::Concat(_) => {
                if inputs.is_empty() {
                    return Err(HoduError::InternalError(
                        "Concat requires at least one input".to_string(),
                    ));
                }
                let dim = attributes
                    .get("dim")
                    .and_then(|a| match a {
                        Attribute::U32(d) => Some(*d),
                        Attribute::Scalar(s) => Some(s.to_u32()),
                        _ => None,
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing dim attribute".to_string()))?;
                let other_storages: Vec<&BackendStorage> = inputs[1..].iter().map(|s| s.as_ref()).collect();
                let other_layouts: Vec<&Layout> = layouts[1..].iter().collect();
                inputs[0].call_concat(&other_storages, &other_layouts, dim, op.clone())
            },

            Op::Split(_) => {
                if inputs.len() != 1 || layouts.len() != 1 {
                    return Err(HoduError::InternalError(format!(
                        "Split operation requires 1 input and layout, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                let dim = attributes
                    .get("dim")
                    .and_then(|a| match a {
                        Attribute::U32(d) => Some(*d),
                        Attribute::Scalar(s) => Some(s.to_u32()),
                        _ => None,
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing dim attribute".to_string()))?;
                let start = attributes
                    .get("start")
                    .and_then(|a| match a {
                        Attribute::U32(s) => Some(*s),
                        Attribute::Scalar(s) => Some(s.to_u32()),
                        _ => None,
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing start attribute".to_string()))?;
                let size = attributes
                    .get("size")
                    .and_then(|a| match a {
                        Attribute::U32(s) => Some(*s),
                        Attribute::Scalar(s) => Some(s.to_u32()),
                        _ => None,
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing size attribute".to_string()))?;
                inputs[0].call_split(&layouts[0], dim, start, size, op.clone())
            },

            Op::Indexing(IndexingOp::IndexSelect) => {
                if inputs.len() != 2 || layouts.len() != 2 {
                    return Err(HoduError::InternalError(format!(
                        "IndexSelect requires 2 inputs and layouts, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                let dim = attributes
                    .get("dim")
                    .and_then(|a| match a {
                        Attribute::U32(d) => Some(*d),
                        Attribute::Scalar(s) => Some(s.to_u32()),
                        _ => None,
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing dim attribute".to_string()))?;
                inputs[0].call_index_select(&layouts[0], &inputs[1], &layouts[1], dim, op.clone())
            },

            Op::Indexing(IndexingOp::IndexPut) => {
                if inputs.len() != 3 || layouts.len() != 3 {
                    return Err(HoduError::InternalError(format!(
                        "IndexPut requires 3 inputs and layouts, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                let dim = attributes
                    .get("dim")
                    .and_then(|a| match a {
                        Attribute::U32(d) => Some(*d),
                        Attribute::Scalar(s) => Some(s.to_u32()),
                        _ => None,
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing dim attribute".to_string()))?;
                inputs[0].call_index_put(
                    &layouts[0],
                    &inputs[1],
                    &layouts[1],
                    &inputs[2],
                    &layouts[2],
                    dim,
                    op.clone(),
                )
            },

            Op::Indexing(IndexingOp::Gather) => {
                if inputs.len() != 2 || layouts.len() != 2 {
                    return Err(HoduError::InternalError(format!(
                        "Gather requires 2 inputs and layouts, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                let dim = attributes
                    .get("dim")
                    .and_then(|a| match a {
                        Attribute::U32(d) => Some(*d),
                        Attribute::Scalar(s) => Some(s.to_u32()),
                        _ => None,
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing dim attribute".to_string()))?;
                inputs[0].call_gather(&layouts[0], &inputs[1], &layouts[1], dim, op.clone())
            },

            Op::Indexing(IndexingOp::Scatter)
            | Op::Indexing(IndexingOp::ScatterAdd)
            | Op::Indexing(IndexingOp::ScatterMax)
            | Op::Indexing(IndexingOp::ScatterMin) => {
                if inputs.len() != 3 || layouts.len() != 3 {
                    return Err(HoduError::InternalError(format!(
                        "Scatter operation requires 3 inputs and layouts, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                let dim = attributes
                    .get("dim")
                    .and_then(|a| match a {
                        Attribute::U32(d) => Some(*d),
                        Attribute::Scalar(s) => Some(s.to_u32()),
                        _ => None,
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing dim attribute".to_string()))?;
                inputs[0].call_scatter(
                    &layouts[0],
                    &inputs[1],
                    &layouts[1],
                    &inputs[2],
                    &layouts[2],
                    dim,
                    op.clone(),
                )
            },

            Op::Conv(_) => {
                if inputs.len() != 2 || layouts.len() != 2 {
                    return Err(HoduError::InternalError(format!(
                        "Conv operation requires 2 inputs and layouts, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                let stride: Vec<u32> = attributes
                    .get("stride")
                    .and_then(|a| {
                        if let Attribute::Scalars(s) = a {
                            Some(s.iter().map(|sc| sc.to_u32()).collect())
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing stride attribute".to_string()))?;
                let padding: Vec<u32> = attributes
                    .get("padding")
                    .and_then(|a| {
                        if let Attribute::Scalars(s) = a {
                            Some(s.iter().map(|sc| sc.to_u32()).collect())
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing padding attribute".to_string()))?;
                let dilation: Vec<u32> = attributes
                    .get("dilation")
                    .and_then(|a| {
                        if let Attribute::Scalars(s) = a {
                            Some(s.iter().map(|sc| sc.to_u32()).collect())
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing dilation attribute".to_string()))?;
                inputs[0].call_conv(
                    &layouts[0],
                    &inputs[1],
                    &layouts[1],
                    &stride,
                    &padding,
                    &dilation,
                    op.clone(),
                )
            },

            Op::Windowing(_) => {
                if inputs.len() != 1 || layouts.len() != 1 {
                    return Err(HoduError::InternalError(format!(
                        "Windowing operation requires 1 input and layout, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                let window_shape: Vec<u32> = attributes
                    .get("window_shape")
                    .and_then(|a| {
                        if let Attribute::Scalars(s) = a {
                            Some(s.iter().map(|sc| sc.to_u32()).collect())
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing window_shape attribute".to_string()))?;
                let strides: Vec<u32> = attributes
                    .get("strides")
                    .and_then(|a| {
                        if let Attribute::Scalars(s) = a {
                            Some(s.iter().map(|sc| sc.to_u32()).collect())
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing strides attribute".to_string()))?;
                let padding: Vec<u32> = attributes
                    .get("padding")
                    .and_then(|a| {
                        if let Attribute::Scalars(s) = a {
                            Some(s.iter().map(|sc| sc.to_u32()).collect())
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| HoduError::InternalError("Missing padding attribute".to_string()))?;
                inputs[0].call_reduce_window(&layouts[0], &window_shape, &strides, &padding, op.clone())
            },

            Op::Shape(_) | Op::ShapeScalars(_) => {
                // Shape operations don't modify storage, just return input storage
                if inputs.len() != 1 {
                    return Err(HoduError::InternalError(format!(
                        "Shape operation requires 1 input, got {}",
                        inputs.len()
                    )));
                }
                Ok(inputs[0].as_ref().clone())
            },

            Op::Cast(CastOp::ToDType) => {
                if inputs.len() != 1 || layouts.len() != 1 {
                    return Err(HoduError::InternalError(format!(
                        "Cast operation requires 1 input and layout, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                let target_dtype = attributes
                    .get("dtype")
                    .and_then(|a| if let Attribute::DType(dt) = a { Some(*dt) } else { None })
                    .ok_or_else(|| HoduError::InternalError("Missing dtype attribute".to_string()))?;
                inputs[0].to_dtype(&layouts[0], target_dtype)
            },

            Op::Memory(MemoryOp::Contiguous) => {
                if inputs.len() != 1 || layouts.len() != 1 {
                    return Err(HoduError::InternalError(format!(
                        "Contiguous operation requires 1 input and layout, got {} and {}",
                        inputs.len(),
                        layouts.len()
                    )));
                }
                inputs[0].contiguous(&layouts[0])
            },

            Op::Dummy => {
                // Dummy op just returns the first input
                if let Some(input) = inputs.first() {
                    Ok((**input).clone())
                } else {
                    Err(HoduError::InternalError(
                        "Dummy operation requires one input".to_string(),
                    ))
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hodu_executor_creation() {
        let executor = HoduExecutor::new(Device::CPU);
        assert_eq!(executor.compiler_type(), Compiler::HODU);
        assert_eq!(executor.device(), Device::CPU);
    }
}
