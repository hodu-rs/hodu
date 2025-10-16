use crate::{
    backends::{
        executor::{CompileOptions, ExecutionInputs, ExecutionOutputs, ExecutorT},
        op::{
            BinaryLogicalOp, BinaryOp, CastOp, CmpOp, CmpScalarOp, IndexingOp, MatrixOp, MemoryOp, Op, ReduceOp,
            ShapeOp, UnaryLogicalOp, UnaryOp, UnaryScalarOp,
        },
        script::{ir::ScriptIR, Script},
    },
    compat::*,
    error::{HoduError, HoduResult},
    tensor::{from_storage, Tensor, TensorId},
    types::{device::Device, dtype::DType, layout::Layout},
};
use hodu_xla::{ElementType, Literal, PjRtClient, PjRtLoadedExecutable, PrimitiveType, XlaBuilder, XlaOp};
use std::collections::HashMap;
use std::sync::Arc;

// Thread-safe wrapper for PjRtLoadedExecutable
// SAFETY: XLA's PjRtLoadedExecutable is thread-safe in practice,
// but the Rust bindings don't expose Send/Sync markers
struct ThreadSafeExecutable(PjRtLoadedExecutable);

unsafe impl Send for ThreadSafeExecutable {}
unsafe impl Sync for ThreadSafeExecutable {}

impl std::ops::Deref for ThreadSafeExecutable {
    type Target = PjRtLoadedExecutable;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Thread-safe wrapper for PjRtClient
// SAFETY: XLA's PjRtClient is thread-safe in practice,
// but the Rust bindings don't expose Send/Sync markers
struct ThreadSafeClient(PjRtClient);

unsafe impl Send for ThreadSafeClient {}
unsafe impl Sync for ThreadSafeClient {}

impl std::ops::Deref for ThreadSafeClient {
    type Target = PjRtClient;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Helper function to convert XLA errors to HoduError
fn xla_error_to_hodu_error(err: hodu_xla::Error) -> HoduError {
    HoduError::InternalError(format!("XLA error: {:?}", err))
}

pub struct XlaExecutor {
    client: Arc<ThreadSafeClient>,
    device: Device,
}

impl XlaExecutor {
    // Helper method to create multiply computation (cached per executor instance)
    fn create_multiply_computation() -> HoduResult<hodu_xla::XlaComputation> {
        let builder = XlaBuilder::new("multiply_computation");
        let lhs = builder
            .parameter(0, ElementType::F32, &[], "lhs")
            .map_err(xla_error_to_hodu_error)?;
        let rhs = builder
            .parameter(1, ElementType::F32, &[], "rhs")
            .map_err(xla_error_to_hodu_error)?;
        let result = lhs.mul_(&rhs).map_err(xla_error_to_hodu_error)?;
        result.build().map_err(xla_error_to_hodu_error)
    }

    // Helper method to convert PrimitiveType to ElementType
    fn element_type_to_element_type(element_type: PrimitiveType) -> HoduResult<ElementType> {
        match element_type {
            PrimitiveType::Pred => Ok(ElementType::Pred),
            PrimitiveType::S8 => Ok(ElementType::S8),
            PrimitiveType::S16 => Ok(ElementType::S16),
            PrimitiveType::S32 => Ok(ElementType::S32),
            PrimitiveType::S64 => Ok(ElementType::S64),
            PrimitiveType::U8 => Ok(ElementType::U8),
            PrimitiveType::U16 => Ok(ElementType::U16),
            PrimitiveType::U32 => Ok(ElementType::U32),
            PrimitiveType::U64 => Ok(ElementType::U64),
            PrimitiveType::F16 => Ok(ElementType::F16),
            PrimitiveType::F32 => Ok(ElementType::F32),
            PrimitiveType::Bf16 => Ok(ElementType::Bf16),
            PrimitiveType::F64 => Ok(ElementType::F64),
            PrimitiveType::C64 => Ok(ElementType::C64),
            PrimitiveType::C128 => Ok(ElementType::C128),
            _ => Err(HoduError::InternalError(format!(
                "Cannot convert PrimitiveType {:?} to ElementType",
                element_type
            ))),
        }
    }
}

pub struct XlaCompiledScript {
    executable: Arc<ThreadSafeExecutable>,
    input_mapping: HashMap<String, TensorId>,
    output_mapping: HashMap<String, TensorId>,
    tensor_layouts: HashMap<TensorId, Layout>,
    tensor_dtypes: HashMap<TensorId, DType>,
}

impl XlaExecutor {
    /// Convert DType to XLA ElementType
    fn dtype_to_element_type(dtype: DType) -> HoduResult<ElementType> {
        match dtype {
            DType::F32 => Ok(ElementType::F32),
            DType::F64 => Ok(ElementType::F64),
            DType::I32 => Ok(ElementType::S32),
            DType::I64 => Ok(ElementType::S64),
            DType::U32 => Ok(ElementType::U32),
            DType::U64 => Ok(ElementType::U64),
            DType::BOOL => Ok(ElementType::Pred),
            DType::F16 => Ok(ElementType::F16),
            DType::BF16 => Ok(ElementType::Bf16),
            DType::I8 => Ok(ElementType::S8),
            DType::I16 => Ok(ElementType::S16),
            DType::U8 => Ok(ElementType::U8),
            DType::U16 => Ok(ElementType::U16),
            DType::F8E4M3 | DType::F8E5M2 => Err(HoduError::InternalError(format!(
                "XLA does not support {:?} dtype",
                dtype
            ))),
        }
    }

    fn collect_tensor_layouts(&self, script_ir: &ScriptIR) -> HashMap<TensorId, Layout> {
        // Estimate capacity: node layouts + inputs + outputs
        let estimated_layout_count = script_ir.graph.topology.nodes.len() * 2
            + script_ir.graph.metadata.inputs.len()
            + script_ir.graph.metadata.outputs.len();
        let mut tensor_layouts = HashMap::with_capacity(estimated_layout_count);

        for node in &script_ir.graph.topology.nodes {
            for (layout, &tensor_id) in node.input_layouts.iter().zip(&node.input_tensors) {
                tensor_layouts.insert(tensor_id, layout.clone());
            }
            for (layout, &tensor_id) in node.output_layouts.iter().zip(&node.output_tensors) {
                tensor_layouts.insert(tensor_id, layout.clone());
            }
        }

        for input in &script_ir.graph.metadata.inputs {
            if let Some(tensor_info) = script_ir.graph.metadata.tensor_info.get(&input.tensor_id) {
                if let Some(ref shape) = tensor_info.shape {
                    // Avoid unnecessary Vec allocation
                    let shape_usize: Vec<usize> = shape.iter().map(|s| s.unwrap_or(1)).collect();
                    tensor_layouts.insert(input.tensor_id, Layout::from_shape(&shape_usize));
                }
            }
        }

        for output in &script_ir.graph.metadata.outputs {
            if let Some(tensor_info) = script_ir.graph.metadata.tensor_info.get(&output.tensor_id) {
                if let Some(ref shape) = tensor_info.shape {
                    // Avoid unnecessary Vec allocation
                    let shape_usize: Vec<usize> = shape.iter().map(|s| s.unwrap_or(1)).collect();
                    tensor_layouts.insert(output.tensor_id, Layout::from_shape(&shape_usize));
                }
            }
        }

        tensor_layouts
    }

    fn collect_tensor_dtypes(&self, script_ir: &ScriptIR, script: &Script) -> HashMap<TensorId, DType> {
        let mut tensor_dtypes = HashMap::with_capacity(script_ir.graph.metadata.tensor_info.len());

        for (&tensor_id, tensor_info) in &script_ir.graph.metadata.tensor_info {
            if let Some(dtype) = tensor_info.dtype {
                tensor_dtypes.insert(tensor_id, dtype);
            }
        }

        // Override dtypes for input tensors with actual runtime input dtypes
        let runtime_inputs = script.get_inputs();
        for input in &script_ir.graph.metadata.inputs {
            if let Some(tensor) = runtime_inputs.get(&input.name) {
                let actual_dtype = tensor.get_dtype();
                tensor_dtypes.insert(input.tensor_id, actual_dtype);
            }
        }

        tensor_dtypes
    }

    fn build_xla_computation(
        &self,
        compiled_script: &XlaCompiledScript,
        script_ir: &ScriptIR,
    ) -> HoduResult<Arc<ThreadSafeExecutable>> {
        let builder = XlaBuilder::new("xla_computation");
        // Pre-allocate for inputs + constants + nodes
        let estimated_ops_count = compiled_script.input_mapping.len()
            + script_ir.graph.metadata.constants.len()
            + script_ir.graph.topology.nodes.len();
        let mut xla_ops = HashMap::with_capacity(estimated_ops_count);

        // Create parameters for inputs in order (sort references to avoid cloning)
        let mut input_pairs: Vec<_> = compiled_script.input_mapping.iter().collect();
        input_pairs.sort_by_key(|(name, _)| *name); // Sort by reference

        for (i, (_input_name, &tensor_id)) in input_pairs.iter().enumerate() {
            if let Some(layout) = compiled_script.tensor_layouts.get(&tensor_id) {
                let dtype = compiled_script
                    .tensor_dtypes
                    .get(&tensor_id)
                    .copied()
                    .unwrap_or(DType::F32);
                let element_type = Self::dtype_to_element_type(dtype)?;
                let dims: Vec<i64> = layout.get_shape().iter().map(|&d| d as i64).collect();

                let param_name = format!("input_{}", i);
                let xla_param = builder
                    .parameter(i as i64, element_type, &dims, &param_name)
                    .map_err(xla_error_to_hodu_error)?;

                xla_ops.insert(tensor_id, xla_param);
            }
        }

        // Process constants and create constant operations
        for (tensor_id, constant_node) in &script_ir.graph.metadata.constants {
            let constant_op = self.convert_constant_to_xla_op(&builder, constant_node)?;
            xla_ops.insert(*tensor_id, constant_op);
        }

        // Process all nodes in execution order
        for &node_id in &script_ir.graph.execution_plan.execution_order {
            let node = &script_ir.graph.topology.nodes[node_id.0];

            // Get input operations
            let input_ops: Vec<XlaOp> = node
                .input_tensors
                .iter()
                .map(|&tensor_id| {
                    xla_ops.get(&tensor_id).cloned().ok_or_else(|| {
                        HoduError::InternalError(format!(
                            "Input tensor {:?} not found for node {:?}",
                            tensor_id, node_id
                        ))
                    })
                })
                .collect::<HoduResult<Vec<_>>>()?;

            // Execute the operation and store result
            let result_op = self.execute_xla_operation(&builder, &node.operation, &input_ops, compiled_script, node)?;

            // Store the result for output tensor
            if let Some(&output_tensor_id) = node.output_tensors.first() {
                xla_ops.insert(output_tensor_id, result_op);
            }
        }

        // Handle multiple outputs by creating a tuple
        let mut output_names: Vec<_> = compiled_script.output_mapping.keys().cloned().collect();
        output_names.sort(); // Ensure consistent ordering

        let computation = if output_names.len() == 1 {
            // Single output case
            let output_tensor_id = compiled_script
                .output_mapping
                .get(&output_names[0])
                .ok_or_else(|| HoduError::InternalError("Output tensor mapping not found".to_string()))?;

            let final_op = xla_ops
                .get(output_tensor_id)
                .ok_or_else(|| HoduError::InternalError("Final output operation not found".to_string()))?;

            final_op.build().map_err(xla_error_to_hodu_error)?
        } else {
            // Multiple output case - create tuple
            let mut output_ops = Vec::new();
            for output_name in &output_names {
                let output_tensor_id = compiled_script.output_mapping.get(output_name).ok_or_else(|| {
                    HoduError::InternalError(format!("Output tensor mapping not found for {}", output_name))
                })?;

                let output_op = xla_ops.get(output_tensor_id).ok_or_else(|| {
                    HoduError::InternalError(format!("Output operation not found for {}", output_name))
                })?;

                output_ops.push(output_op.clone());
            }

            // Create tuple of outputs using builder
            let tuple_op = builder.tuple(&output_ops).map_err(xla_error_to_hodu_error)?;
            tuple_op.build().map_err(xla_error_to_hodu_error)?
        };
        let executable = self.client.compile(&computation).map_err(xla_error_to_hodu_error)?;
        Ok(Arc::new(ThreadSafeExecutable(executable)))
    }

    fn execute_xla_operation(
        &self,
        builder: &XlaBuilder,
        operation: &Op,
        input_ops: &[XlaOp],
        _compiled_script: &XlaCompiledScript,
        current_node: &crate::backends::script::ir::GraphNode,
    ) -> HoduResult<XlaOp> {
        match operation {
            // Binary Operations
            Op::Binary(op, _, _) => {
                if input_ops.len() != 2 {
                    return Err(HoduError::InternalError(
                        "Binary operation requires exactly 2 inputs".to_string(),
                    ));
                }
                let result = match op {
                    BinaryOp::Add => input_ops[0].add_(&input_ops[1]),
                    BinaryOp::Sub => input_ops[0].sub_(&input_ops[1]),
                    BinaryOp::Mul => input_ops[0].mul_(&input_ops[1]),
                    BinaryOp::Div => input_ops[0].div_(&input_ops[1]),
                    BinaryOp::Pow => input_ops[0].pow(&input_ops[1]),
                    BinaryOp::Maximum => input_ops[0].max(&input_ops[1]),
                    BinaryOp::Minimum => input_ops[0].min(&input_ops[1]),
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // Binary Logical Operations
            Op::BinaryLogical(op, _, _) => {
                if input_ops.len() != 2 {
                    return Err(HoduError::InternalError(
                        "Binary logical operation requires exactly 2 inputs".to_string(),
                    ));
                }
                let result = match op {
                    BinaryLogicalOp::LogicalAnd => input_ops[0].and(&input_ops[1]),
                    BinaryLogicalOp::LogicalOr => input_ops[0].or(&input_ops[1]),
                    BinaryLogicalOp::LogicalXor => input_ops[0].xor(&input_ops[1]),
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // Comparison Operations
            Op::Cmp(op, _, _) => {
                if input_ops.len() != 2 {
                    return Err(HoduError::InternalError(
                        "Comparison operation requires exactly 2 inputs".to_string(),
                    ));
                }
                let result = match op {
                    CmpOp::Eq => input_ops[0].eq(&input_ops[1]),
                    CmpOp::Ne => input_ops[0].ne(&input_ops[1]),
                    CmpOp::Lt => input_ops[0].lt(&input_ops[1]),
                    CmpOp::Le => input_ops[0].le(&input_ops[1]),
                    CmpOp::Gt => input_ops[0].gt(&input_ops[1]),
                    CmpOp::Ge => input_ops[0].ge(&input_ops[1]),
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // Comparison with Scalar Operations
            Op::CmpScalar(op, _, scalar) => {
                if input_ops.len() != 1 {
                    return Err(HoduError::InternalError(
                        "Scalar comparison operation requires exactly 1 input".to_string(),
                    ));
                }
                let scalar_value = scalar.to_f32();
                let scalar_op = builder.constant_r0(scalar_value).map_err(xla_error_to_hodu_error)?;
                let result = match op {
                    CmpScalarOp::EqScalar => input_ops[0].eq(&scalar_op),
                    CmpScalarOp::NeScalar => input_ops[0].ne(&scalar_op),
                    CmpScalarOp::LtScalar => input_ops[0].lt(&scalar_op),
                    CmpScalarOp::LeScalar => input_ops[0].le(&scalar_op),
                    CmpScalarOp::GtScalar => input_ops[0].gt(&scalar_op),
                    CmpScalarOp::GeScalar => input_ops[0].ge(&scalar_op),
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // Unary Operations (simplified set)
            Op::Unary(op, _) => {
                if input_ops.len() != 1 {
                    return Err(HoduError::InternalError(
                        "Unary operation requires exactly 1 input".to_string(),
                    ));
                }
                let result = match op {
                    UnaryOp::Neg => input_ops[0].neg(),
                    UnaryOp::Abs => input_ops[0].abs(),
                    UnaryOp::Sign => input_ops[0].sign(),
                    UnaryOp::Square => input_ops[0].mul_(&input_ops[0]),
                    UnaryOp::Relu => {
                        let zero = builder.constant_r0(0.0f32).map_err(xla_error_to_hodu_error)?;
                        input_ops[0].max(&zero)
                    },
                    UnaryOp::Sigmoid => input_ops[0].logistic(), // Use XLA's builtin logistic
                    UnaryOp::Tanh => input_ops[0].tanh(),
                    UnaryOp::Gelu => {
                        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                        let sqrt_2_over_pi = builder
                            .constant_r0(0.7978845608028654f32)
                            .map_err(xla_error_to_hodu_error)?; // sqrt(2/π)
                        let gelu_coeff = builder.constant_r0(0.044715f32).map_err(xla_error_to_hodu_error)?;
                        let half = builder.constant_r0(0.5f32).map_err(xla_error_to_hodu_error)?;
                        let one = builder.constant_r0(1.0f32).map_err(xla_error_to_hodu_error)?;

                        let x = &input_ops[0];
                        let x_cubed = x
                            .mul_(x)
                            .map_err(xla_error_to_hodu_error)?
                            .mul_(x)
                            .map_err(xla_error_to_hodu_error)?;
                        let inner = x
                            .add_(&gelu_coeff.mul_(&x_cubed).map_err(xla_error_to_hodu_error)?)
                            .map_err(xla_error_to_hodu_error)?;
                        let tanh_arg = sqrt_2_over_pi.mul_(&inner).map_err(xla_error_to_hodu_error)?;
                        let tanh_result = tanh_arg.tanh().map_err(xla_error_to_hodu_error)?;
                        let one_plus_tanh = one.add_(&tanh_result).map_err(xla_error_to_hodu_error)?;
                        half.mul_(x).map_err(xla_error_to_hodu_error)?.mul_(&one_plus_tanh)
                    },
                    UnaryOp::Sin => input_ops[0].sin(),
                    UnaryOp::Cos => input_ops[0].cos(),
                    UnaryOp::Tan => {
                        let sin_val = input_ops[0].sin().map_err(xla_error_to_hodu_error)?;
                        let cos_val = input_ops[0].cos().map_err(xla_error_to_hodu_error)?;
                        sin_val.div_(&cos_val)
                    },
                    UnaryOp::Ln => input_ops[0].log(),
                    UnaryOp::Log10 => {
                        let ln_val = input_ops[0].log().map_err(xla_error_to_hodu_error)?;
                        let ln_10 = builder
                            .constant_r0(2.302585092994046f32)
                            .map_err(xla_error_to_hodu_error)?; // ln(10)
                        ln_val.div_(&ln_10)
                    },
                    UnaryOp::Log2 => {
                        let ln_val = input_ops[0].log().map_err(xla_error_to_hodu_error)?;
                        let ln_2 = builder
                            .constant_r0(0.6931471805599453f32)
                            .map_err(xla_error_to_hodu_error)?; // ln(2)
                        ln_val.div_(&ln_2)
                    },
                    UnaryOp::Exp => input_ops[0].exp(),
                    UnaryOp::Exp10 => {
                        let ln_10 = builder
                            .constant_r0(2.302585092994046f32)
                            .map_err(xla_error_to_hodu_error)?; // ln(10)
                        let scaled = input_ops[0].mul_(&ln_10).map_err(xla_error_to_hodu_error)?;
                        scaled.exp()
                    },
                    UnaryOp::Exp2 => {
                        let ln_2 = builder
                            .constant_r0(0.6931471805599453f32)
                            .map_err(xla_error_to_hodu_error)?; // ln(2)
                        let scaled = input_ops[0].mul_(&ln_2).map_err(xla_error_to_hodu_error)?;
                        scaled.exp()
                    },
                    UnaryOp::Softplus => {
                        // softplus(x) = log(1 + exp(x))
                        let one = builder.constant_r0(1.0f32).map_err(xla_error_to_hodu_error)?;
                        let exp_x = input_ops[0].exp().map_err(xla_error_to_hodu_error)?;
                        let one_plus_exp = one.add_(&exp_x).map_err(xla_error_to_hodu_error)?;
                        one_plus_exp.log()
                    },
                    UnaryOp::Recip => {
                        let one = builder.constant_r0(1.0f32).map_err(xla_error_to_hodu_error)?;
                        one.div_(&input_ops[0])
                    },
                    UnaryOp::Sqrt => input_ops[0].sqrt(),
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // Unary Logical Operations
            Op::UnaryLogical(op, _) => {
                if input_ops.len() != 1 {
                    return Err(HoduError::InternalError(
                        "Unary logical operation requires exactly 1 input".to_string(),
                    ));
                }
                let result = match op {
                    UnaryLogicalOp::LogicalNot => input_ops[0].not(),
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // Unary Scalar Operations
            Op::UnaryScalar(op, _, scalar) => {
                if input_ops.len() != 1 {
                    return Err(HoduError::InternalError(
                        "Unary scalar operation requires exactly 1 input".to_string(),
                    ));
                }
                let scalar_value = scalar.to_f32();
                let scalar_op = builder.constant_r0(scalar_value).map_err(xla_error_to_hodu_error)?;
                let result = match op {
                    UnaryScalarOp::AddScalar => input_ops[0].add_(&scalar_op),
                    UnaryScalarOp::SubScalar => input_ops[0].sub_(&scalar_op),
                    UnaryScalarOp::MulScalar => input_ops[0].mul_(&scalar_op),
                    UnaryScalarOp::DivScalar => input_ops[0].div_(&scalar_op),
                    UnaryScalarOp::PowScalar => input_ops[0].pow(&scalar_op),
                    UnaryScalarOp::MaximumScalar => input_ops[0].max(&scalar_op),
                    UnaryScalarOp::MinimumScalar => input_ops[0].min(&scalar_op),
                    UnaryScalarOp::LeakyRelu => {
                        let zero = builder.constant_r0(0.0f32).map_err(xla_error_to_hodu_error)?;
                        let positive_part = input_ops[0].max(&zero).map_err(xla_error_to_hodu_error)?;
                        let negative_part = input_ops[0].min(&zero).map_err(xla_error_to_hodu_error)?;
                        let scaled_negative = negative_part.mul_(&scalar_op).map_err(xla_error_to_hodu_error)?;
                        positive_part.add_(&scaled_negative)
                    },
                    UnaryScalarOp::Elu => {
                        // ELU: x if x > 0, else α * (exp(x) - 1)
                        let zero = builder.constant_r0(0.0f32).map_err(xla_error_to_hodu_error)?;
                        let one = builder.constant_r0(1.0f32).map_err(xla_error_to_hodu_error)?;
                        let exp_part = input_ops[0].exp().map_err(xla_error_to_hodu_error)?;
                        let elu_negative = scalar_op
                            .mul_(&exp_part.sub_(&one).map_err(xla_error_to_hodu_error)?)
                            .map_err(xla_error_to_hodu_error)?;
                        let condition = input_ops[0].gt(&zero).map_err(xla_error_to_hodu_error)?;
                        condition.select(&input_ops[0], &elu_negative)
                    },
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // Matrix Operations
            Op::Matrix(op, _, _) => {
                if input_ops.len() != 2 {
                    return Err(HoduError::InternalError(
                        "Matrix operation requires exactly 2 inputs".to_string(),
                    ));
                }

                match op {
                    MatrixOp::Matmul => {
                        // Use XLA's matmul which supports batched operations and broadcasting (1D, 2D, ND)
                        input_ops[0].matmul(&input_ops[1]).map_err(xla_error_to_hodu_error)
                    },
                    MatrixOp::Dot => {
                        // Simple dot operation (1D/2D only)
                        // Use XLA's basic dot operation
                        input_ops[0].dot(&input_ops[1]).map_err(xla_error_to_hodu_error)
                    },
                }
            },

            // Reduce Operations
            Op::Reduce(reduce_op, _, dims_scalars) => {
                if input_ops.len() != 1 {
                    return Err(HoduError::InternalError(
                        "Reduce operation requires exactly 1 input".to_string(),
                    ));
                }

                // Extract dimensions from scalar array
                let dims: Vec<i64> = dims_scalars.iter().map(|scalar| scalar.to_u64() as i64).collect();

                match reduce_op {
                    ReduceOp::Sum => {
                        if dims.is_empty() {
                            // Sum all elements
                            let input_shape = input_ops[0].shape().map_err(xla_error_to_hodu_error)?;
                            let all_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                                _ => {
                                    return Err(HoduError::InternalError("Expected array shape for reduce".to_string()))
                                },
                            };
                            input_ops[0].reduce_sum(&all_dims, false)
                        } else {
                            input_ops[0].reduce_sum(&dims, false)
                        }
                        .map_err(xla_error_to_hodu_error)
                    },
                    ReduceOp::Mean => {
                        if dims.is_empty() {
                            // Mean of all elements
                            let input_shape = input_ops[0].shape().map_err(xla_error_to_hodu_error)?;
                            let all_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                                _ => {
                                    return Err(HoduError::InternalError("Expected array shape for reduce".to_string()))
                                },
                            };
                            input_ops[0].reduce_mean(&all_dims, false)
                        } else {
                            input_ops[0].reduce_mean(&dims, false)
                        }
                        .map_err(xla_error_to_hodu_error)
                    },
                    ReduceOp::Max => {
                        if dims.is_empty() {
                            // Max of all elements
                            let input_shape = input_ops[0].shape().map_err(xla_error_to_hodu_error)?;
                            let all_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                                _ => {
                                    return Err(HoduError::InternalError("Expected array shape for reduce".to_string()))
                                },
                            };
                            input_ops[0].reduce_max(&all_dims, false)
                        } else {
                            input_ops[0].reduce_max(&dims, false)
                        }
                        .map_err(xla_error_to_hodu_error)
                    },
                    ReduceOp::Min => {
                        if dims.is_empty() {
                            // Min of all elements
                            let input_shape = input_ops[0].shape().map_err(xla_error_to_hodu_error)?;
                            let all_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                                _ => {
                                    return Err(HoduError::InternalError("Expected array shape for reduce".to_string()))
                                },
                            };
                            input_ops[0].reduce_min(&all_dims, false)
                        } else {
                            input_ops[0].reduce_min(&dims, false)
                        }
                        .map_err(xla_error_to_hodu_error)
                    },
                    ReduceOp::Prod => {
                        // Create multiplication computation (still better than inline creation)
                        let one = builder.constant_r0(1.0f32).map_err(xla_error_to_hodu_error)?;
                        let prod_computation = Self::create_multiply_computation()?;

                        if dims.is_empty() {
                            // Product of all elements
                            let input_shape = input_ops[0].shape().map_err(xla_error_to_hodu_error)?;
                            let all_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                                _ => {
                                    return Err(HoduError::InternalError("Expected array shape for reduce".to_string()))
                                },
                            };
                            input_ops[0].reduce(one, prod_computation, &all_dims, false)
                        } else {
                            input_ops[0].reduce(one, prod_computation, &dims, false)
                        }
                        .map_err(xla_error_to_hodu_error)
                    },
                    ReduceOp::Std => {
                        // XLA doesn't have built-in std, so we implement it as sqrt(variance)
                        let mean_op = if dims.is_empty() {
                            let input_shape = input_ops[0].shape().map_err(xla_error_to_hodu_error)?;
                            let all_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                                _ => {
                                    return Err(HoduError::InternalError("Expected array shape for reduce".to_string()))
                                },
                            };
                            input_ops[0]
                                .reduce_mean(&all_dims, true)
                                .map_err(xla_error_to_hodu_error)?
                        } else {
                            input_ops[0].reduce_mean(&dims, true).map_err(xla_error_to_hodu_error)?
                        };

                        // Variance: mean((x - mean)^2)
                        let diff = input_ops[0].sub_(&mean_op).map_err(xla_error_to_hodu_error)?;
                        let squared = diff.mul_(&diff).map_err(xla_error_to_hodu_error)?;
                        let variance = if dims.is_empty() {
                            let input_shape = input_ops[0].shape().map_err(xla_error_to_hodu_error)?;
                            let all_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                                _ => {
                                    return Err(HoduError::InternalError("Expected array shape for reduce".to_string()))
                                },
                            };
                            squared.reduce_mean(&all_dims, false).map_err(xla_error_to_hodu_error)?
                        } else {
                            squared.reduce_mean(&dims, false).map_err(xla_error_to_hodu_error)?
                        };

                        // Standard deviation: sqrt(variance)
                        variance.sqrt().map_err(xla_error_to_hodu_error)
                    },
                    ReduceOp::Var => {
                        // XLA doesn't have built-in variance, so we implement it as mean((x - mean)^2)
                        let mean_op = if dims.is_empty() {
                            let input_shape = input_ops[0].shape().map_err(xla_error_to_hodu_error)?;
                            let all_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                                _ => {
                                    return Err(HoduError::InternalError("Expected array shape for reduce".to_string()))
                                },
                            };
                            input_ops[0]
                                .reduce_mean(&all_dims, true)
                                .map_err(xla_error_to_hodu_error)?
                        } else {
                            input_ops[0].reduce_mean(&dims, true).map_err(xla_error_to_hodu_error)?
                        };

                        // Variance: mean((x - mean)^2)
                        let diff = input_ops[0].sub_(&mean_op).map_err(xla_error_to_hodu_error)?;
                        let squared = diff.mul_(&diff).map_err(xla_error_to_hodu_error)?;
                        if dims.is_empty() {
                            let input_shape = input_ops[0].shape().map_err(xla_error_to_hodu_error)?;
                            let all_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                                _ => {
                                    return Err(HoduError::InternalError("Expected array shape for reduce".to_string()))
                                },
                            };
                            squared.reduce_mean(&all_dims, false)
                        } else {
                            squared.reduce_mean(&dims, false)
                        }
                        .map_err(xla_error_to_hodu_error)
                    },
                    ReduceOp::Norm => {
                        // L2 norm: sqrt(sum(x^2))
                        let squared = input_ops[0].mul_(&input_ops[0]).map_err(xla_error_to_hodu_error)?;
                        let sum_squared = if dims.is_empty() {
                            let input_shape = input_ops[0].shape().map_err(xla_error_to_hodu_error)?;
                            let all_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                                _ => {
                                    return Err(HoduError::InternalError("Expected array shape for reduce".to_string()))
                                },
                            };
                            squared.reduce_sum(&all_dims, false).map_err(xla_error_to_hodu_error)?
                        } else {
                            squared.reduce_sum(&dims, false).map_err(xla_error_to_hodu_error)?
                        };

                        // L2 norm: sqrt(sum_squared)
                        sum_squared.sqrt().map_err(xla_error_to_hodu_error)
                    },
                    ReduceOp::ArgMax => {
                        // Implement argmax using: find max value, then find its index
                        if dims.is_empty() {
                            return Err(HoduError::InternalError(
                                "ArgMax requires a dimension to reduce over".to_string(),
                            ));
                        }
                        if dims.len() > 1 {
                            return Err(HoduError::InternalError(
                                "ArgMax only supports single dimension reduction".to_string(),
                            ));
                        }

                        let dim = dims[0];
                        let input = &input_ops[0];
                        let shape = input.array_shape().map_err(xla_error_to_hodu_error)?;
                        let input_dims: Vec<i64> = shape.dims().iter().map(|&d| d as i64).collect();
                        let builder = input.builder();

                        // Step 1: Find the max value along the dimension
                        let max_val = input.reduce_max(&[dim], true).map_err(xla_error_to_hodu_error)?;

                        // Step 2: Create iota tensor for indices
                        let indices = builder
                            .iota(hodu_xla::ElementType::S64, &input_dims, dim)
                            .map_err(xla_error_to_hodu_error)?;

                        // Step 3: Create mask where input equals max (broadcast max to input shape)
                        let mask = input.eq(&max_val).map_err(xla_error_to_hodu_error)?;

                        // Step 4: Convert mask to int64 and multiply with indices
                        let mask_i64 = mask
                            .convert(hodu_xla::PrimitiveType::S64)
                            .map_err(xla_error_to_hodu_error)?;
                        let masked_indices = mask_i64.mul_(&indices).map_err(xla_error_to_hodu_error)?;

                        // Step 5: For non-matching positions, set to a large value so they don't affect min
                        let large_val = builder.c0(i64::MAX).map_err(xla_error_to_hodu_error)?;
                        let inv_mask = mask.not().map_err(xla_error_to_hodu_error)?;
                        let inv_mask_i64 = inv_mask
                            .convert(hodu_xla::PrimitiveType::S64)
                            .map_err(xla_error_to_hodu_error)?;
                        let large_vals = inv_mask_i64.mul_(&large_val).map_err(xla_error_to_hodu_error)?;
                        let adjusted_indices = masked_indices.add_(&large_vals).map_err(xla_error_to_hodu_error)?;

                        // Step 6: Take minimum to get the first occurrence
                        adjusted_indices
                            .reduce_min(&[dim], false)
                            .map_err(xla_error_to_hodu_error)
                    },
                    ReduceOp::ArgMin => {
                        // Implement argmin using: find min value, then find its index
                        if dims.is_empty() {
                            return Err(HoduError::InternalError(
                                "ArgMin requires a dimension to reduce over".to_string(),
                            ));
                        }
                        if dims.len() > 1 {
                            return Err(HoduError::InternalError(
                                "ArgMin only supports single dimension reduction".to_string(),
                            ));
                        }

                        let dim = dims[0];
                        let input = &input_ops[0];
                        let shape = input.array_shape().map_err(xla_error_to_hodu_error)?;
                        let input_dims: Vec<i64> = shape.dims().iter().map(|&d| d as i64).collect();
                        let builder = input.builder();

                        // Step 1: Find the min value along the dimension
                        let min_val = input.reduce_min(&[dim], true).map_err(xla_error_to_hodu_error)?;

                        // Step 2: Create iota tensor for indices
                        let indices = builder
                            .iota(hodu_xla::ElementType::S64, &input_dims, dim)
                            .map_err(xla_error_to_hodu_error)?;

                        // Step 3: Create mask where input equals min (broadcast min to input shape)
                        let mask = input.eq(&min_val).map_err(xla_error_to_hodu_error)?;

                        // Step 4: Convert mask to int64 and multiply with indices
                        let mask_i64 = mask
                            .convert(hodu_xla::PrimitiveType::S64)
                            .map_err(xla_error_to_hodu_error)?;
                        let masked_indices = mask_i64.mul_(&indices).map_err(xla_error_to_hodu_error)?;

                        // Step 5: For non-matching positions, set to a large value so they don't affect min
                        let large_val = builder.c0(i64::MAX).map_err(xla_error_to_hodu_error)?;
                        let inv_mask = mask.not().map_err(xla_error_to_hodu_error)?;
                        let inv_mask_i64 = inv_mask
                            .convert(hodu_xla::PrimitiveType::S64)
                            .map_err(xla_error_to_hodu_error)?;
                        let large_vals = inv_mask_i64.mul_(&large_val).map_err(xla_error_to_hodu_error)?;
                        let adjusted_indices = masked_indices.add_(&large_vals).map_err(xla_error_to_hodu_error)?;

                        // Step 6: Take minimum to get the first occurrence
                        adjusted_indices
                            .reduce_min(&[dim], false)
                            .map_err(xla_error_to_hodu_error)
                    },
                }
            },

            // Concat Operation
            Op::Concat(_concat_op, _input_tensor_ids, params) => {
                if input_ops.is_empty() {
                    return Err(HoduError::InternalError(
                        "Concat operation requires at least 1 input".to_string(),
                    ));
                }

                // Extract dimension from params
                if params.is_empty() {
                    return Err(HoduError::InternalError(
                        "Concat operation requires dimension parameter".to_string(),
                    ));
                }

                let dim = params[0].to_u64() as i64;

                // Use XLA's concat_in_dim operation
                input_ops[0]
                    .concat_in_dim(&input_ops[1..], dim)
                    .map_err(xla_error_to_hodu_error)
            },

            // Split Operation
            Op::Split(_split_op, _input_tensor_id, size_scalars, output_index) => {
                if input_ops.len() != 1 {
                    return Err(HoduError::InternalError(
                        "Split operation requires exactly 1 input".to_string(),
                    ));
                }

                // Extract dimension and sizes from the operation metadata
                // The dimension is stored in the node metadata
                // For now, we need to get it from the output layout comparison
                let input_layout = current_node
                    .input_layouts
                    .first()
                    .ok_or_else(|| HoduError::InternalError("Missing input layout for split".to_string()))?;
                let output_layout = current_node
                    .output_layouts
                    .first()
                    .ok_or_else(|| HoduError::InternalError("Missing output layout for split".to_string()))?;

                // Find which dimension changed by comparing shapes
                let input_shape = input_layout.get_shape();
                let output_shape = output_layout.get_shape();

                let mut split_dim = None;
                for (i, (&in_size, &out_size)) in input_shape.iter().zip(output_shape.iter()).enumerate() {
                    if in_size != out_size {
                        split_dim = Some(i as i64);
                        break;
                    }
                }

                let dim = split_dim
                    .ok_or_else(|| HoduError::InternalError("Could not determine split dimension".to_string()))?;

                // Extract sizes from size_scalars (skip first element which is the dimension)
                let sizes: Vec<i64> = size_scalars.iter().skip(1).map(|s| s.to_u64() as i64).collect();

                // Calculate split indices (cumulative sum)
                let mut split_indices = Vec::with_capacity(sizes.len() - 1);
                let mut cumsum = 0i64;
                for &size in &sizes[..sizes.len() - 1] {
                    cumsum += size;
                    split_indices.push(cumsum);
                }

                // Calculate start and limit for this output slice
                let start_offset = if *output_index == 0 {
                    0
                } else {
                    split_indices[*output_index - 1]
                };

                let size = sizes[*output_index];

                // Use slice_in_dim operation for single dimension slicing
                let sliced = input_ops[0]
                    .slice_in_dim(start_offset, start_offset + size, 1, dim)
                    .map_err(xla_error_to_hodu_error)?;

                // Reshape to maintain dimensions when size is 1 (XLA may squeeze it)
                // Get the target shape from output layout
                if let Some(target_layout) = current_node.output_layouts.first() {
                    let target_shape: Vec<i64> = target_layout.get_shape().iter().map(|&d| d as i64).collect();
                    sliced.reshape(&target_shape).map_err(xla_error_to_hodu_error)
                } else {
                    Ok(sliced)
                }
            },

            // Indexing Operations
            Op::Indexing(indexing_op, _tensor_ids, params) => {
                match indexing_op {
                    IndexingOp::IndexSelect => {
                        // input_ops: [self, indices]
                        if input_ops.len() != 2 {
                            return Err(HoduError::InternalError("IndexSelect requires 2 inputs".to_string()));
                        }

                        // Extract dim from params
                        let dim = params
                            .first()
                            .ok_or_else(|| {
                                HoduError::InternalError("IndexSelect requires dimension parameter".to_string())
                            })?
                            .to_u64() as i64;

                        // Use XLA's take operation for index_select
                        input_ops[0].take(&input_ops[1], dim).map_err(xla_error_to_hodu_error)
                    },

                    IndexingOp::Gather => {
                        // input_ops: [self, indices]
                        if input_ops.len() != 2 {
                            return Err(HoduError::InternalError("Gather requires 2 inputs".to_string()));
                        }

                        // Extract dim from params
                        let dim = params
                            .first()
                            .ok_or_else(|| HoduError::InternalError("Gather requires dimension parameter".to_string()))?
                            .to_u64() as i64;

                        // Get shapes
                        let base_shape = input_ops[0].array_shape().map_err(xla_error_to_hodu_error)?;
                        let indices_shape = input_ops[1].array_shape().map_err(xla_error_to_hodu_error)?;

                        let base_rank = base_shape.dims().len();
                        let indices_rank = indices_shape.dims().len();
                        let base_dims = base_shape.dims();
                        let indices_dims = indices_shape.dims();

                        // For multidimensional gather, we need to implement PyTorch semantics
                        // out[i][j] = input[indices[i][j]][j] for dim=0
                        // out[i][j] = input[i][indices[i][j]] for dim=1
                        if indices_rank > 1 && base_rank > 1 {
                            // Flatten input and indices
                            let base_size: i64 = base_dims.iter().product();
                            let indices_size: i64 = indices_dims.iter().product();

                            let base_flat = input_ops[0].reshape(&[base_size]).map_err(xla_error_to_hodu_error)?;
                            let indices_flat =
                                input_ops[1].reshape(&[indices_size]).map_err(xla_error_to_hodu_error)?;

                            // Calculate strides (same pattern as scatter)
                            let base_strides: Vec<i64> = {
                                let mut strides = vec![1i64; base_rank];
                                for i in (0..base_rank - 1).rev() {
                                    strides[i] = strides[i + 1] * base_dims[i + 1];
                                }
                                strides
                            };

                            let indices_strides: Vec<i64> = {
                                let mut strides = vec![1i64; indices_rank];
                                for i in (0..indices_rank - 1).rev() {
                                    strides[i] = strides[i + 1] * indices_dims[i + 1];
                                }
                                strides
                            };

                            let positions: Vec<i64> = (0..indices_size).collect();
                            let positions_op = builder.constant_r1(&positions).map_err(xla_error_to_hodu_error)?;
                            let positions_i32 = positions_op
                                .convert(PrimitiveType::S32)
                                .map_err(xla_error_to_hodu_error)?;
                            let indices_i32 = indices_flat
                                .convert(PrimitiveType::S32)
                                .map_err(xla_error_to_hodu_error)?;

                            let zero_i32 = builder.constant_r0(0i32).map_err(xla_error_to_hodu_error)?;
                            let mut flat_indices_i32 =
                                zero_i32.broadcast(&[indices_size]).map_err(xla_error_to_hodu_error)?;

                            // Build flat index for gathering from base
                            for d in 0..base_rank {
                                if d == dim as usize {
                                    // Use indices[p] for this dimension
                                    let base_stride_i32 = builder
                                        .constant_r0(base_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let contribution =
                                        indices_i32.mul_(&base_stride_i32).map_err(xla_error_to_hodu_error)?;
                                    flat_indices_i32 =
                                        flat_indices_i32.add_(&contribution).map_err(xla_error_to_hodu_error)?;
                                } else {
                                    // Extract coordinate from position p in indices tensor
                                    let indices_stride_i32 = builder
                                        .constant_r0(indices_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let indices_dim_i32 = builder
                                        .constant_r0(indices_dims[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;

                                    let coord_quotient = positions_i32
                                        .div_(&indices_stride_i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let coord_remainder_quotient =
                                        coord_quotient.div_(&indices_dim_i32).map_err(xla_error_to_hodu_error)?;
                                    let coord = coord_quotient
                                        .sub_(
                                            &coord_remainder_quotient
                                                .mul_(&indices_dim_i32)
                                                .map_err(xla_error_to_hodu_error)?,
                                        )
                                        .map_err(xla_error_to_hodu_error)?;

                                    let base_stride_i32 = builder
                                        .constant_r0(base_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let contribution = coord.mul_(&base_stride_i32).map_err(xla_error_to_hodu_error)?;
                                    flat_indices_i32 =
                                        flat_indices_i32.add_(&contribution).map_err(xla_error_to_hodu_error)?;
                                }
                            }

                            // Perform gather using take (simpler for 1D indexing)
                            let flat_indices_i64 = flat_indices_i32
                                .convert(PrimitiveType::S64)
                                .map_err(xla_error_to_hodu_error)?;

                            let result_flat = base_flat.take(&flat_indices_i64, 0).map_err(xla_error_to_hodu_error)?;

                            result_flat.reshape(indices_dims).map_err(xla_error_to_hodu_error)
                        } else {
                            // 1D case - use simple take
                            input_ops[0].take(&input_ops[1], dim).map_err(xla_error_to_hodu_error)
                        }
                    },

                    IndexingOp::Scatter => {
                        // input_ops: [self, indices, src]
                        if input_ops.len() != 3 {
                            return Err(HoduError::InternalError("Scatter requires 3 inputs".to_string()));
                        }

                        // Extract dim from params
                        let dim = params
                            .first()
                            .ok_or_else(|| {
                                HoduError::InternalError("Scatter requires dimension parameter".to_string())
                            })?
                            .to_u64() as i64;

                        // Create update computation that replaces values
                        let element_type = input_ops[0].ty().map_err(xla_error_to_hodu_error)?;
                        let element_type = Self::element_type_to_element_type(element_type)?;

                        let update_builder = XlaBuilder::new("scatter_update");
                        let _old = update_builder
                            .parameter(0, element_type, &[], "old")
                            .map_err(xla_error_to_hodu_error)?;
                        let new = update_builder
                            .parameter(1, element_type, &[], "new")
                            .map_err(xla_error_to_hodu_error)?;
                        let update_computation = new.build().map_err(xla_error_to_hodu_error)?;

                        // Get shapes for scatter configuration
                        let base_shape = input_ops[0].array_shape().map_err(xla_error_to_hodu_error)?;
                        let indices_shape = input_ops[1].array_shape().map_err(xla_error_to_hodu_error)?;
                        let src_shape = input_ops[2].array_shape().map_err(xla_error_to_hodu_error)?;

                        let base_rank = base_shape.dims().len();
                        let indices_rank = indices_shape.dims().len();
                        let base_dims = base_shape.dims();
                        let indices_dims = indices_shape.dims();
                        let _src_dims = src_shape.dims();

                        // For multidimensional scatter, convert PyTorch semantics to XLA
                        if indices_rank > 1 && base_rank > 1 {
                            // PyTorch: src[i,j,...] -> input[indices[i,j,...], j, ...]
                            // XLA: requires flattened indices with proper offset calculation

                            // Flatten base, indices, src
                            let base_size: i64 = base_dims.iter().product();
                            let indices_size: i64 = indices_dims.iter().product();

                            let base_flat = input_ops[0].reshape(&[base_size]).map_err(xla_error_to_hodu_error)?;
                            let indices_flat =
                                input_ops[1].reshape(&[indices_size]).map_err(xla_error_to_hodu_error)?;
                            let src_flat = input_ops[2].reshape(&[indices_size]).map_err(xla_error_to_hodu_error)?;

                            // Calculate strides for mapping positions
                            // For scatter along dim, we need to map position p in flattened indices/src
                            // to a position in flattened base.
                            //
                            // Key insight: indices and src have the same shape, and that shape
                            // tells us how to "traverse" the base tensor.
                            //
                            // Strides in base tensor (for computing flat index from coordinates)
                            let base_strides: Vec<i64> = {
                                let mut strides = vec![1i64; base_rank];
                                for i in (0..base_rank - 1).rev() {
                                    strides[i] = strides[i + 1] * base_dims[i + 1];
                                }
                                strides
                            };

                            // Strides in indices tensor (for decomposing position p)
                            let indices_strides: Vec<i64> = {
                                let mut strides = vec![1i64; indices_rank];
                                for i in (0..indices_rank - 1).rev() {
                                    strides[i] = strides[i + 1] * indices_dims[i + 1];
                                }
                                strides
                            };

                            // For each position p, compute coordinates in indices tensor,
                            // then map to flat index in base
                            let positions: Vec<i64> = (0..indices_size).collect();
                            let positions_op = builder.constant_r1(&positions).map_err(xla_error_to_hodu_error)?;
                            let positions_i32 = positions_op
                                .convert(PrimitiveType::S32)
                                .map_err(xla_error_to_hodu_error)?;
                            let indices_i32 = indices_flat
                                .convert(PrimitiveType::S32)
                                .map_err(xla_error_to_hodu_error)?;

                            // Build flat index calculation step by step
                            // flat_idx = sum over all dims: coord[d] * base_strides[d]
                            // where coord[d] = (p / indices_strides[d]) % indices_dims[d] for d != dim
                            //       coord[dim] = indices[p]
                            let zero_i32 = builder.constant_r0(0i32).map_err(xla_error_to_hodu_error)?;
                            let mut flat_indices_i32 =
                                zero_i32.broadcast(&[indices_size]).map_err(xla_error_to_hodu_error)?;

                            for d in 0..base_rank {
                                if d == dim as usize {
                                    // Use indices[p] for this dimension
                                    let base_stride_i32 = builder
                                        .constant_r0(base_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let contribution =
                                        indices_i32.mul_(&base_stride_i32).map_err(xla_error_to_hodu_error)?;
                                    flat_indices_i32 =
                                        flat_indices_i32.add_(&contribution).map_err(xla_error_to_hodu_error)?;
                                } else {
                                    // Extract coordinate from position p
                                    let indices_stride_i32 = builder
                                        .constant_r0(indices_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let indices_dim_i32 = builder
                                        .constant_r0(indices_dims[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;

                                    let coord_quotient = positions_i32
                                        .div_(&indices_stride_i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let coord_remainder_quotient =
                                        coord_quotient.div_(&indices_dim_i32).map_err(xla_error_to_hodu_error)?;
                                    let coord = coord_quotient
                                        .sub_(
                                            &coord_remainder_quotient
                                                .mul_(&indices_dim_i32)
                                                .map_err(xla_error_to_hodu_error)?,
                                        )
                                        .map_err(xla_error_to_hodu_error)?;

                                    let base_stride_i32 = builder
                                        .constant_r0(base_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let contribution = coord.mul_(&base_stride_i32).map_err(xla_error_to_hodu_error)?;
                                    flat_indices_i32 =
                                        flat_indices_i32.add_(&contribution).map_err(xla_error_to_hodu_error)?;
                                }
                            }
                            let flat_indices_reshaped = flat_indices_i32
                                .reshape(&[indices_size, 1])
                                .map_err(xla_error_to_hodu_error)?;

                            // Perform 1D scatter
                            let result_flat = base_flat
                                .scatter(
                                    &flat_indices_reshaped,
                                    &src_flat,
                                    update_computation,
                                    &[],
                                    &[0],
                                    &[0],
                                    Some(1),
                                    false,
                                    false,
                                )
                                .map_err(xla_error_to_hodu_error)?;

                            // Reshape back to original shape
                            result_flat.reshape(base_dims).map_err(xla_error_to_hodu_error)
                        } else {
                            // 1D case - simple scatter
                            let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                            let inserted_window_dims = vec![dim];
                            let scatter_dims_to_operand_dims = vec![dim];
                            let index_vector_dim = Some(indices_rank as i64);

                            let mut indices_dims_plus_1 = indices_shape.dims().to_vec();
                            indices_dims_plus_1.push(1);
                            let indices_reshaped = input_ops[1]
                                .reshape(&indices_dims_plus_1)
                                .map_err(xla_error_to_hodu_error)?;

                            input_ops[0]
                                .scatter(
                                    &indices_reshaped,
                                    &input_ops[2],
                                    update_computation,
                                    &update_window_dims,
                                    &inserted_window_dims,
                                    &scatter_dims_to_operand_dims,
                                    index_vector_dim,
                                    false,
                                    false,
                                )
                                .map_err(xla_error_to_hodu_error)
                        }
                    },

                    IndexingOp::ScatterAdd => {
                        // input_ops: [self, indices, src]
                        if input_ops.len() != 3 {
                            return Err(HoduError::InternalError("ScatterAdd requires 3 inputs".to_string()));
                        }

                        // Extract dim from params
                        let dim = params
                            .first()
                            .ok_or_else(|| {
                                HoduError::InternalError("ScatterAdd requires dimension parameter".to_string())
                            })?
                            .to_u64() as i64;

                        // Create add computation
                        let element_type = input_ops[0].ty().map_err(xla_error_to_hodu_error)?;
                        let element_type = Self::element_type_to_element_type(element_type)?;

                        let add_builder = XlaBuilder::new("scatter_add");
                        let old = add_builder
                            .parameter(0, element_type, &[], "old")
                            .map_err(xla_error_to_hodu_error)?;
                        let new = add_builder
                            .parameter(1, element_type, &[], "new")
                            .map_err(xla_error_to_hodu_error)?;
                        let sum = old.add_(&new).map_err(xla_error_to_hodu_error)?;
                        let add_computation = sum.build().map_err(xla_error_to_hodu_error)?;

                        // Get shapes for scatter configuration
                        let base_shape = input_ops[0].array_shape().map_err(xla_error_to_hodu_error)?;
                        let indices_shape = input_ops[1].array_shape().map_err(xla_error_to_hodu_error)?;
                        let _src_shape = input_ops[2].array_shape().map_err(xla_error_to_hodu_error)?;

                        let base_rank = base_shape.dims().len();
                        let indices_rank = indices_shape.dims().len();
                        let base_dims = base_shape.dims();
                        let indices_dims = indices_shape.dims();

                        // For multidimensional scatter, convert PyTorch semantics to XLA
                        if indices_rank > 1 && base_rank > 1 {
                            // Same approach as Scatter: flatten, compute indices, scatter, reshape
                            let base_size: i64 = base_dims.iter().product();
                            let indices_size: i64 = indices_dims.iter().product();

                            let base_flat = input_ops[0].reshape(&[base_size]).map_err(xla_error_to_hodu_error)?;
                            let indices_flat =
                                input_ops[1].reshape(&[indices_size]).map_err(xla_error_to_hodu_error)?;
                            let src_flat = input_ops[2].reshape(&[indices_size]).map_err(xla_error_to_hodu_error)?;

                            // Calculate strides (same as Scatter)
                            let base_strides: Vec<i64> = {
                                let mut strides = vec![1i64; base_rank];
                                for i in (0..base_rank - 1).rev() {
                                    strides[i] = strides[i + 1] * base_dims[i + 1];
                                }
                                strides
                            };

                            let indices_strides: Vec<i64> = {
                                let mut strides = vec![1i64; indices_rank];
                                for i in (0..indices_rank - 1).rev() {
                                    strides[i] = strides[i + 1] * indices_dims[i + 1];
                                }
                                strides
                            };

                            let positions: Vec<i64> = (0..indices_size).collect();
                            let positions_op = builder.constant_r1(&positions).map_err(xla_error_to_hodu_error)?;
                            let positions_i32 = positions_op
                                .convert(PrimitiveType::S32)
                                .map_err(xla_error_to_hodu_error)?;
                            let indices_i32 = indices_flat
                                .convert(PrimitiveType::S32)
                                .map_err(xla_error_to_hodu_error)?;

                            let zero_i32 = builder.constant_r0(0i32).map_err(xla_error_to_hodu_error)?;
                            let mut flat_indices_i32 =
                                zero_i32.broadcast(&[indices_size]).map_err(xla_error_to_hodu_error)?;

                            for d in 0..base_rank {
                                if d == dim as usize {
                                    let base_stride_i32 = builder
                                        .constant_r0(base_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let contribution =
                                        indices_i32.mul_(&base_stride_i32).map_err(xla_error_to_hodu_error)?;
                                    flat_indices_i32 =
                                        flat_indices_i32.add_(&contribution).map_err(xla_error_to_hodu_error)?;
                                } else {
                                    let indices_stride_i32 = builder
                                        .constant_r0(indices_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let indices_dim_i32 = builder
                                        .constant_r0(indices_dims[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;

                                    let coord_quotient = positions_i32
                                        .div_(&indices_stride_i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let coord_remainder_quotient =
                                        coord_quotient.div_(&indices_dim_i32).map_err(xla_error_to_hodu_error)?;
                                    let coord = coord_quotient
                                        .sub_(
                                            &coord_remainder_quotient
                                                .mul_(&indices_dim_i32)
                                                .map_err(xla_error_to_hodu_error)?,
                                        )
                                        .map_err(xla_error_to_hodu_error)?;

                                    let base_stride_i32 = builder
                                        .constant_r0(base_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let contribution = coord.mul_(&base_stride_i32).map_err(xla_error_to_hodu_error)?;
                                    flat_indices_i32 =
                                        flat_indices_i32.add_(&contribution).map_err(xla_error_to_hodu_error)?;
                                }
                            }

                            let flat_indices_reshaped = flat_indices_i32
                                .reshape(&[indices_size, 1])
                                .map_err(xla_error_to_hodu_error)?;

                            // Perform 1D scatter with add
                            let result_flat = base_flat
                                .scatter(
                                    &flat_indices_reshaped,
                                    &src_flat,
                                    add_computation,
                                    &[],
                                    &[0],
                                    &[0],
                                    Some(1),
                                    false,
                                    false,
                                )
                                .map_err(xla_error_to_hodu_error)?;

                            result_flat.reshape(base_dims).map_err(xla_error_to_hodu_error)
                        } else {
                            // 1D case - simple scatter
                            let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                            let inserted_window_dims = vec![dim];
                            let scatter_dims_to_operand_dims = vec![dim];
                            let index_vector_dim = Some(indices_rank as i64);

                            let mut indices_dims_plus_1 = indices_shape.dims().to_vec();
                            indices_dims_plus_1.push(1);
                            let indices_reshaped = input_ops[1]
                                .reshape(&indices_dims_plus_1)
                                .map_err(xla_error_to_hodu_error)?;

                            input_ops[0]
                                .scatter(
                                    &indices_reshaped,
                                    &input_ops[2],
                                    add_computation,
                                    &update_window_dims,
                                    &inserted_window_dims,
                                    &scatter_dims_to_operand_dims,
                                    index_vector_dim,
                                    false,
                                    false,
                                )
                                .map_err(xla_error_to_hodu_error)
                        }
                    },

                    IndexingOp::ScatterMax => {
                        // input_ops: [self, indices, src]
                        if input_ops.len() != 3 {
                            return Err(HoduError::InternalError("ScatterMax requires 3 inputs".to_string()));
                        }

                        // Extract dim from params
                        let dim = params
                            .first()
                            .ok_or_else(|| {
                                HoduError::InternalError("ScatterMax requires dimension parameter".to_string())
                            })?
                            .to_u64() as i64;

                        // Create max computation
                        let element_type = input_ops[0].ty().map_err(xla_error_to_hodu_error)?;
                        let element_type = Self::element_type_to_element_type(element_type)?;

                        let max_builder = XlaBuilder::new("scatter_max");
                        let old = max_builder
                            .parameter(0, element_type, &[], "old")
                            .map_err(xla_error_to_hodu_error)?;
                        let new = max_builder
                            .parameter(1, element_type, &[], "new")
                            .map_err(xla_error_to_hodu_error)?;
                        let maximum = old.max(&new).map_err(xla_error_to_hodu_error)?;
                        let max_computation = maximum.build().map_err(xla_error_to_hodu_error)?;

                        // Get shapes for scatter configuration
                        let base_shape = input_ops[0].array_shape().map_err(xla_error_to_hodu_error)?;
                        let indices_shape = input_ops[1].array_shape().map_err(xla_error_to_hodu_error)?;
                        let _src_shape = input_ops[2].array_shape().map_err(xla_error_to_hodu_error)?;

                        let base_rank = base_shape.dims().len();
                        let indices_rank = indices_shape.dims().len();
                        let base_dims = base_shape.dims();
                        let indices_dims = indices_shape.dims();

                        // For multidimensional scatter, convert PyTorch semantics to XLA
                        if indices_rank > 1 && base_rank > 1 {
                            let base_size: i64 = base_dims.iter().product();
                            let indices_size: i64 = indices_dims.iter().product();

                            let base_flat = input_ops[0].reshape(&[base_size]).map_err(xla_error_to_hodu_error)?;
                            let indices_flat =
                                input_ops[1].reshape(&[indices_size]).map_err(xla_error_to_hodu_error)?;
                            let src_flat = input_ops[2].reshape(&[indices_size]).map_err(xla_error_to_hodu_error)?;

                            let base_strides: Vec<i64> = {
                                let mut strides = vec![1i64; base_rank];
                                for i in (0..base_rank - 1).rev() {
                                    strides[i] = strides[i + 1] * base_dims[i + 1];
                                }
                                strides
                            };

                            let indices_strides: Vec<i64> = {
                                let mut strides = vec![1i64; indices_rank];
                                for i in (0..indices_rank - 1).rev() {
                                    strides[i] = strides[i + 1] * indices_dims[i + 1];
                                }
                                strides
                            };

                            let positions: Vec<i64> = (0..indices_size).collect();
                            let positions_op = builder.constant_r1(&positions).map_err(xla_error_to_hodu_error)?;
                            let positions_i32 = positions_op
                                .convert(PrimitiveType::S32)
                                .map_err(xla_error_to_hodu_error)?;
                            let indices_i32 = indices_flat
                                .convert(PrimitiveType::S32)
                                .map_err(xla_error_to_hodu_error)?;

                            let zero_i32 = builder.constant_r0(0i32).map_err(xla_error_to_hodu_error)?;
                            let mut flat_indices_i32 =
                                zero_i32.broadcast(&[indices_size]).map_err(xla_error_to_hodu_error)?;

                            for d in 0..base_rank {
                                if d == dim as usize {
                                    let base_stride_i32 = builder
                                        .constant_r0(base_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let contribution =
                                        indices_i32.mul_(&base_stride_i32).map_err(xla_error_to_hodu_error)?;
                                    flat_indices_i32 =
                                        flat_indices_i32.add_(&contribution).map_err(xla_error_to_hodu_error)?;
                                } else {
                                    let indices_stride_i32 = builder
                                        .constant_r0(indices_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let indices_dim_i32 = builder
                                        .constant_r0(indices_dims[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;

                                    let coord_quotient = positions_i32
                                        .div_(&indices_stride_i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let coord_remainder_quotient =
                                        coord_quotient.div_(&indices_dim_i32).map_err(xla_error_to_hodu_error)?;
                                    let coord = coord_quotient
                                        .sub_(
                                            &coord_remainder_quotient
                                                .mul_(&indices_dim_i32)
                                                .map_err(xla_error_to_hodu_error)?,
                                        )
                                        .map_err(xla_error_to_hodu_error)?;

                                    let base_stride_i32 = builder
                                        .constant_r0(base_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let contribution = coord.mul_(&base_stride_i32).map_err(xla_error_to_hodu_error)?;
                                    flat_indices_i32 =
                                        flat_indices_i32.add_(&contribution).map_err(xla_error_to_hodu_error)?;
                                }
                            }

                            let flat_indices_reshaped = flat_indices_i32
                                .reshape(&[indices_size, 1])
                                .map_err(xla_error_to_hodu_error)?;

                            let result_flat = base_flat
                                .scatter(
                                    &flat_indices_reshaped,
                                    &src_flat,
                                    max_computation,
                                    &[],
                                    &[0],
                                    &[0],
                                    Some(1),
                                    false,
                                    false,
                                )
                                .map_err(xla_error_to_hodu_error)?;

                            result_flat.reshape(base_dims).map_err(xla_error_to_hodu_error)
                        } else {
                            // 1D case
                            let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                            let inserted_window_dims = vec![dim];
                            let scatter_dims_to_operand_dims = vec![dim];
                            let index_vector_dim = Some(indices_rank as i64);

                            let mut indices_dims_plus_1 = indices_shape.dims().to_vec();
                            indices_dims_plus_1.push(1);
                            let indices_reshaped = input_ops[1]
                                .reshape(&indices_dims_plus_1)
                                .map_err(xla_error_to_hodu_error)?;

                            input_ops[0]
                                .scatter(
                                    &indices_reshaped,
                                    &input_ops[2],
                                    max_computation,
                                    &update_window_dims,
                                    &inserted_window_dims,
                                    &scatter_dims_to_operand_dims,
                                    index_vector_dim,
                                    false,
                                    false,
                                )
                                .map_err(xla_error_to_hodu_error)
                        }
                    },

                    IndexingOp::ScatterMin => {
                        // input_ops: [self, indices, src]
                        if input_ops.len() != 3 {
                            return Err(HoduError::InternalError("ScatterMin requires 3 inputs".to_string()));
                        }

                        // Extract dim from params
                        let dim = params
                            .first()
                            .ok_or_else(|| {
                                HoduError::InternalError("ScatterMin requires dimension parameter".to_string())
                            })?
                            .to_u64() as i64;

                        // Create min computation
                        let element_type = input_ops[0].ty().map_err(xla_error_to_hodu_error)?;
                        let element_type = Self::element_type_to_element_type(element_type)?;

                        let min_builder = XlaBuilder::new("scatter_min");
                        let old = min_builder
                            .parameter(0, element_type, &[], "old")
                            .map_err(xla_error_to_hodu_error)?;
                        let new = min_builder
                            .parameter(1, element_type, &[], "new")
                            .map_err(xla_error_to_hodu_error)?;
                        let minimum = old.min(&new).map_err(xla_error_to_hodu_error)?;
                        let min_computation = minimum.build().map_err(xla_error_to_hodu_error)?;

                        // Get shapes for scatter configuration
                        let base_shape = input_ops[0].array_shape().map_err(xla_error_to_hodu_error)?;
                        let indices_shape = input_ops[1].array_shape().map_err(xla_error_to_hodu_error)?;
                        let _src_shape = input_ops[2].array_shape().map_err(xla_error_to_hodu_error)?;

                        let base_rank = base_shape.dims().len();
                        let indices_rank = indices_shape.dims().len();
                        let base_dims = base_shape.dims();
                        let indices_dims = indices_shape.dims();

                        // For multidimensional scatter, convert PyTorch semantics to XLA
                        if indices_rank > 1 && base_rank > 1 {
                            let base_size: i64 = base_dims.iter().product();
                            let indices_size: i64 = indices_dims.iter().product();

                            let base_flat = input_ops[0].reshape(&[base_size]).map_err(xla_error_to_hodu_error)?;
                            let indices_flat =
                                input_ops[1].reshape(&[indices_size]).map_err(xla_error_to_hodu_error)?;
                            let src_flat = input_ops[2].reshape(&[indices_size]).map_err(xla_error_to_hodu_error)?;

                            let base_strides: Vec<i64> = {
                                let mut strides = vec![1i64; base_rank];
                                for i in (0..base_rank - 1).rev() {
                                    strides[i] = strides[i + 1] * base_dims[i + 1];
                                }
                                strides
                            };

                            let indices_strides: Vec<i64> = {
                                let mut strides = vec![1i64; indices_rank];
                                for i in (0..indices_rank - 1).rev() {
                                    strides[i] = strides[i + 1] * indices_dims[i + 1];
                                }
                                strides
                            };

                            let positions: Vec<i64> = (0..indices_size).collect();
                            let positions_op = builder.constant_r1(&positions).map_err(xla_error_to_hodu_error)?;
                            let positions_i32 = positions_op
                                .convert(PrimitiveType::S32)
                                .map_err(xla_error_to_hodu_error)?;
                            let indices_i32 = indices_flat
                                .convert(PrimitiveType::S32)
                                .map_err(xla_error_to_hodu_error)?;

                            let zero_i32 = builder.constant_r0(0i32).map_err(xla_error_to_hodu_error)?;
                            let mut flat_indices_i32 =
                                zero_i32.broadcast(&[indices_size]).map_err(xla_error_to_hodu_error)?;

                            for d in 0..base_rank {
                                if d == dim as usize {
                                    let base_stride_i32 = builder
                                        .constant_r0(base_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let contribution =
                                        indices_i32.mul_(&base_stride_i32).map_err(xla_error_to_hodu_error)?;
                                    flat_indices_i32 =
                                        flat_indices_i32.add_(&contribution).map_err(xla_error_to_hodu_error)?;
                                } else {
                                    let indices_stride_i32 = builder
                                        .constant_r0(indices_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let indices_dim_i32 = builder
                                        .constant_r0(indices_dims[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;

                                    let coord_quotient = positions_i32
                                        .div_(&indices_stride_i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let coord_remainder_quotient =
                                        coord_quotient.div_(&indices_dim_i32).map_err(xla_error_to_hodu_error)?;
                                    let coord = coord_quotient
                                        .sub_(
                                            &coord_remainder_quotient
                                                .mul_(&indices_dim_i32)
                                                .map_err(xla_error_to_hodu_error)?,
                                        )
                                        .map_err(xla_error_to_hodu_error)?;

                                    let base_stride_i32 = builder
                                        .constant_r0(base_strides[d] as i32)
                                        .map_err(xla_error_to_hodu_error)?;
                                    let contribution = coord.mul_(&base_stride_i32).map_err(xla_error_to_hodu_error)?;
                                    flat_indices_i32 =
                                        flat_indices_i32.add_(&contribution).map_err(xla_error_to_hodu_error)?;
                                }
                            }

                            let flat_indices_reshaped = flat_indices_i32
                                .reshape(&[indices_size, 1])
                                .map_err(xla_error_to_hodu_error)?;

                            let result_flat = base_flat
                                .scatter(
                                    &flat_indices_reshaped,
                                    &src_flat,
                                    min_computation,
                                    &[],
                                    &[0],
                                    &[0],
                                    Some(1),
                                    false,
                                    false,
                                )
                                .map_err(xla_error_to_hodu_error)?;

                            result_flat.reshape(base_dims).map_err(xla_error_to_hodu_error)
                        } else {
                            // 1D case
                            let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                            let inserted_window_dims = vec![dim];
                            let scatter_dims_to_operand_dims = vec![dim];
                            let index_vector_dim = Some(indices_rank as i64);

                            let mut indices_dims_plus_1 = indices_shape.dims().to_vec();
                            indices_dims_plus_1.push(1);
                            let indices_reshaped = input_ops[1]
                                .reshape(&indices_dims_plus_1)
                                .map_err(xla_error_to_hodu_error)?;

                            input_ops[0]
                                .scatter(
                                    &indices_reshaped,
                                    &input_ops[2],
                                    min_computation,
                                    &update_window_dims,
                                    &inserted_window_dims,
                                    &scatter_dims_to_operand_dims,
                                    index_vector_dim,
                                    false,
                                    false,
                                )
                                .map_err(xla_error_to_hodu_error)
                        }
                    },
                }
            },

            // Shape Operations
            Op::Shape(op, _) => {
                if input_ops.len() != 1 {
                    return Err(HoduError::InternalError(
                        "View operation requires exactly 1 input".to_string(),
                    ));
                }
                match op {
                    ShapeOp::Reshape => {
                        // Get target shape from output layout
                        if let Some(target_layout) = current_node.output_layouts.first() {
                            let target_shape: Vec<i64> = target_layout.get_shape().iter().map(|&d| d as i64).collect();
                            input_ops[0].reshape(&target_shape).map_err(xla_error_to_hodu_error)
                        } else {
                            // Fallback: return input unchanged
                            Ok(input_ops[0].clone())
                        }
                    },
                    ShapeOp::Flatten => {
                        // Get input shape and flatten to 1D
                        let input_shape = input_ops[0].shape().map_err(xla_error_to_hodu_error)?;
                        let total_size = match input_shape {
                            hodu_xla::Shape::Array(array_shape) => array_shape.dims().iter().product::<i64>(),
                            _ => {
                                return Err(HoduError::InternalError(
                                    "Expected array shape for flatten operation".to_string(),
                                ))
                            },
                        };

                        let flat_shape = vec![total_size];
                        input_ops[0].reshape(&flat_shape).map_err(xla_error_to_hodu_error)
                    },
                    ShapeOp::Squeeze => {
                        // Get target shape from output layout (dimensions of size 1 removed)
                        if let Some(target_layout) = current_node.output_layouts.first() {
                            let target_shape: Vec<i64> = target_layout.get_shape().iter().map(|&d| d as i64).collect();
                            input_ops[0].reshape(&target_shape).map_err(xla_error_to_hodu_error)
                        } else {
                            // Fallback: return input unchanged
                            Ok(input_ops[0].clone())
                        }
                    },
                    ShapeOp::Unsqueeze => {
                        // Get target shape from output layout (dimensions of size 1 added)
                        if let Some(target_layout) = current_node.output_layouts.first() {
                            let target_shape: Vec<i64> = target_layout.get_shape().iter().map(|&d| d as i64).collect();
                            input_ops[0].reshape(&target_shape).map_err(xla_error_to_hodu_error)
                        } else {
                            // Fallback: return input unchanged
                            Ok(input_ops[0].clone())
                        }
                    },
                    ShapeOp::Broadcast => {
                        // Get target shape from output layout
                        if let Some(target_layout) = current_node.output_layouts.first() {
                            let target_shape: Vec<i64> = target_layout.get_shape().iter().map(|&d| d as i64).collect();

                            // Use XLA's broadcast_in_dim for explicit broadcasting
                            // For now, assume we're broadcasting to larger dimensions (left-padding with 1s)
                            let input_shape = input_ops[0].shape().map_err(xla_error_to_hodu_error)?;
                            let input_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                                _ => {
                                    return Err(HoduError::InternalError(
                                        "Expected array shape for broadcast operation".to_string(),
                                    ))
                                },
                            };

                            if input_dims == target_shape {
                                // No broadcasting needed
                                Ok(input_ops[0].clone())
                            } else {
                                // Create broadcast dimensions mapping
                                let input_rank = input_dims.len();
                                let target_rank = target_shape.len();

                                if input_rank <= target_rank {
                                    // Standard broadcasting: map input dimensions to the rightmost target dimensions
                                    let broadcast_dims: Vec<i64> =
                                        (target_rank - input_rank..target_rank).map(|i| i as i64).collect();

                                    input_ops[0]
                                        .broadcast_in_dim(&target_shape, &broadcast_dims)
                                        .map_err(xla_error_to_hodu_error)
                                } else {
                                    // Input has more dimensions than target, this shouldn't happen in normal broadcasting
                                    Err(HoduError::InternalError(
                                        "Cannot broadcast tensor to smaller shape".to_string(),
                                    ))
                                }
                            }
                        } else {
                            // Fallback: return input unchanged
                            Ok(input_ops[0].clone())
                        }
                    },
                    ShapeOp::Transpose => {
                        // Get input and output shapes
                        let input_layout = current_node.input_layouts.first().ok_or_else(|| {
                            HoduError::InternalError("Missing input layout for transpose".to_string())
                        })?;
                        let output_layout = current_node.output_layouts.first().ok_or_else(|| {
                            HoduError::InternalError("Missing output layout for transpose".to_string())
                        })?;

                        let input_shape = input_layout.get_shape();
                        let output_shape = output_layout.get_shape();
                        let ndim = input_layout.get_ndim();

                        if ndim < 2 {
                            return Err(HoduError::InternalError(
                                "Cannot transpose tensor with fewer than 2 dimensions".to_string(),
                            ));
                        }

                        // Find which dimensions were swapped by comparing shapes
                        let mut dim1 = None;
                        let mut dim2 = None;
                        for i in 0..ndim {
                            for j in (i + 1)..ndim {
                                if input_shape[i] == output_shape[j] && input_shape[j] == output_shape[i] {
                                    // Check if all other dimensions match
                                    let mut all_match = true;
                                    for k in 0..ndim {
                                        if k != i && k != j && input_shape[k] != output_shape[k] {
                                            all_match = false;
                                            break;
                                        }
                                    }
                                    if all_match {
                                        dim1 = Some(i);
                                        dim2 = Some(j);
                                        break;
                                    }
                                }
                            }
                            if dim1.is_some() {
                                break;
                            }
                        }

                        match (dim1, dim2) {
                            (Some(d1), Some(d2)) => {
                                // Create permutation array for transpose
                                let mut permutation: Vec<i64> = (0..ndim as i64).collect();
                                permutation.swap(d1, d2);

                                input_ops[0].transpose(&permutation).map_err(xla_error_to_hodu_error)
                            },
                            _ => {
                                // Fallback: assume last two dimensions for matrix transpose
                                let mut permutation: Vec<i64> = (0..ndim as i64).collect();
                                permutation.swap(ndim - 2, ndim - 1);

                                input_ops[0].transpose(&permutation).map_err(xla_error_to_hodu_error)
                            },
                        }
                    },
                    ShapeOp::Permute => {
                        // Get input and output shapes
                        let input_layout = current_node
                            .input_layouts
                            .first()
                            .ok_or_else(|| HoduError::InternalError("Missing input layout for permute".to_string()))?;
                        let output_layout = current_node
                            .output_layouts
                            .first()
                            .ok_or_else(|| HoduError::InternalError("Missing output layout for permute".to_string()))?;

                        let input_shape = input_layout.get_shape();
                        let output_shape = output_layout.get_shape();
                        let ndim = input_layout.get_ndim();

                        // Find the permutation by comparing input and output shapes
                        let mut permutation = vec![0i64; ndim];
                        for i in 0..ndim {
                            // Find which output dimension corresponds to input dimension i
                            for j in 0..ndim {
                                if input_shape[i] == output_shape[j] {
                                    // Check if this is not already assigned
                                    let already_used = permutation[..i].contains(&(j as i64));
                                    if !already_used {
                                        permutation[i] = j as i64;
                                        break;
                                    }
                                }
                            }
                        }

                        input_ops[0].transpose(&permutation).map_err(xla_error_to_hodu_error)
                    },
                }
            },

            // Cast Operations
            Op::Cast(op, _) => {
                if input_ops.len() != 1 {
                    return Err(HoduError::InternalError(
                        "Cast operation requires exactly 1 input".to_string(),
                    ));
                }
                match op {
                    CastOp::ToDType => {
                        // Get target dtype from output tensor metadata
                        if let Some(output_tensor_id) = current_node.output_tensors.first() {
                            // Get target dtype from compiled script tensor_dtypes
                            let target_dtype = _compiled_script
                                .tensor_dtypes
                                .get(output_tensor_id)
                                .copied()
                                .unwrap_or(DType::F32); // Default to F32 if not specified

                            // Convert DType to XLA PrimitiveType
                            let target_element_type = match target_dtype {
                                DType::BOOL => PrimitiveType::Pred,
                                DType::F8E4M3 | DType::F8E5M2 => {
                                    return Err(HoduError::InternalError(
                                        "F8 types not supported in XLA backend".to_string(),
                                    ));
                                },
                                DType::BF16 => PrimitiveType::Bf16,
                                DType::F16 => PrimitiveType::F16,
                                DType::F32 => PrimitiveType::F32,
                                DType::F64 => PrimitiveType::F64,
                                DType::U8 => PrimitiveType::U8,
                                DType::U16 => PrimitiveType::U16,
                                DType::U32 => PrimitiveType::U32,
                                DType::U64 => PrimitiveType::U64,
                                DType::I8 => PrimitiveType::S8,
                                DType::I16 => PrimitiveType::S16,
                                DType::I32 => PrimitiveType::S32,
                                DType::I64 => PrimitiveType::S64,
                            };

                            let result = input_ops[0]
                                .convert(target_element_type)
                                .map_err(xla_error_to_hodu_error)?;

                            Ok(result)
                        } else {
                            Err(HoduError::InternalError(
                                "Cast operation has no output tensor".to_string(),
                            ))
                        }
                    },
                }
            },

            // Memory Operations
            Op::Memory(op, _) => {
                match op {
                    MemoryOp::Contiguous => {
                        if input_ops.len() != 1 {
                            return Err(HoduError::InternalError(
                                "Contiguous operation requires exactly 1 input".to_string(),
                            ));
                        }
                        // XLA tensors are always contiguous in memory
                        // So we can just return the input unchanged
                        Ok(input_ops[0].clone())
                    },
                }
            },

            // For now, return error for unsupported operations
            _ => Err(HoduError::InternalError(format!(
                "Operation {:?} not yet implemented for XLA backend",
                operation
            ))),
        }
    }

    /// Convert Hodu tensor to XLA literal
    fn tensor_to_literal(&self, tensor: &Tensor) -> HoduResult<Literal> {
        let layout = tensor.get_layout();
        let shape = layout.get_shape();
        let dims: Vec<usize> = shape.to_vec();

        tensor.with_storage(|storage| {
            let cpu_storage = storage
                .to_cpu_storage()
                .map_err(|e| HoduError::InternalError(format!("Failed to convert to CPU storage: {:?}", e)))?;

            use crate::backends::be_hodu::cpu::storage::CpuStorage;
            match cpu_storage {
                CpuStorage::BOOL(ref data) => {
                    // Convert bool to u8 as XLA doesn't support bool NativeType directly
                    let u8_data: Vec<u8> = data.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect();
                    if dims.is_empty() {
                        Ok(Literal::scalar(u8_data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(&u8_data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::Pred, &dims, &u8_data)
                                .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::BF16(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::Bf16, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<half::bf16>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::F16(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::F16, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<half::f16>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::F32(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::F32, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<f32>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::F64(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::F64, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<f64>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::U8(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::U8, &dims, data)
                                .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::U16(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::U16, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<u16>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::U32(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::U32, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<u32>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::U64(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::U64, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<u64>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::I8(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::S8, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<i8>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::I16(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::S16, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<i16>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::I32(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::S32, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<i32>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::I64(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::S64, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<i64>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                _ => Err(HoduError::InternalError(
                    "Unsupported tensor data type in XLA conversion".to_string(),
                )),
            }
        })
    }

    /// Convert XLA literal to Hodu tensor
    fn literal_to_tensor(&self, literal: &Literal, dtype: DType) -> HoduResult<Tensor> {
        use crate::backends::be_hodu::cpu::storage::CpuStorage;
        use crate::backends::be_hodu::storage::HoduStorage;

        let array_shape = literal
            .array_shape()
            .map_err(|e| HoduError::InternalError(format!("Failed to get array shape from literal: {:?}", e)))?;

        let dims: Vec<usize> = array_shape.dims().iter().map(|&d| d as usize).collect();

        let cpu_storage = match dtype {
            DType::BOOL => {
                // Convert from u8 back to bool
                let u8_data = literal.to_vec::<u8>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract u8 data for bool from literal: {:?}", e))
                })?;
                let bool_data: Vec<bool> = u8_data.iter().map(|&b| b != 0).collect();
                CpuStorage::BOOL(bool_data)
            },
            DType::BF16 => {
                let data = literal.to_vec::<half::bf16>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract bf16 data from literal: {:?}", e))
                })?;
                CpuStorage::BF16(data)
            },
            DType::F16 => {
                let data = literal.to_vec::<half::f16>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract f16 data from literal: {:?}", e))
                })?;
                CpuStorage::F16(data)
            },
            DType::F32 => {
                let data = literal.to_vec::<f32>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract f32 data from literal: {:?}", e))
                })?;
                CpuStorage::F32(data)
            },
            DType::F64 => {
                let data = literal.to_vec::<f64>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract f64 data from literal: {:?}", e))
                })?;
                CpuStorage::F64(data)
            },
            DType::U8 => {
                let data = literal.to_vec::<u8>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract u8 data from literal: {:?}", e))
                })?;
                CpuStorage::U8(data)
            },
            DType::U16 => {
                let data = literal.to_vec::<u16>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract u16 data from literal: {:?}", e))
                })?;
                CpuStorage::U16(data)
            },
            DType::U32 => {
                let data = literal.to_vec::<u32>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract u32 data from literal: {:?}", e))
                })?;
                CpuStorage::U32(data)
            },
            DType::U64 => {
                let data = literal.to_vec::<u64>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract u64 data from literal: {:?}", e))
                })?;
                CpuStorage::U64(data)
            },
            DType::I8 => {
                let data = literal.to_vec::<i8>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract i8 data from literal: {:?}", e))
                })?;
                CpuStorage::I8(data)
            },
            DType::I16 => {
                let data = literal.to_vec::<i16>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract i16 data from literal: {:?}", e))
                })?;
                CpuStorage::I16(data)
            },
            DType::I32 => {
                let data = literal.to_vec::<i32>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract i32 data from literal: {:?}", e))
                })?;
                CpuStorage::I32(data)
            },
            DType::I64 => {
                let data = literal.to_vec::<i64>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract i64 data from literal: {:?}", e))
                })?;
                CpuStorage::I64(data)
            },
            _ => {
                return Err(HoduError::InternalError(format!(
                    "Unsupported dtype for XLA conversion: {:?}",
                    dtype
                )));
            },
        };

        let hodu_storage = HoduStorage::CPU(cpu_storage);
        let layout = Layout::from_shape(&dims);
        let tensor = from_storage(hodu_storage, layout, true);
        Ok(tensor)
    }

    pub fn new(device: Device) -> HoduResult<Self> {
        let client = match device {
            Device::CPU => {
                Arc::new(ThreadSafeClient(PjRtClient::cpu().map_err(|e| {
                    HoduError::InternalError(format!("Failed to create CPU client: {:?}", e))
                })?))
            },
            Device::CUDA(_idx) => {
                #[cfg(feature = "cuda")]
                {
                    Arc::new(ThreadSafeClient(PjRtClient::gpu(_idx as f64, true).map_err(|e| {
                        HoduError::InternalError(format!("Failed to create GPU client for device {}: {:?}", _idx, e))
                    })?))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(HoduError::InternalError("CUDA support not compiled".to_string()));
                }
            },
            _ => Arc::new(ThreadSafeClient(PjRtClient::cpu().map_err(|e| {
                HoduError::InternalError(format!("Failed to create fallback CPU client: {:?}", e))
            })?)),
        };

        Ok(Self { client, device })
    }

    fn xla_element_type_to_dtype(&self, element_type: hodu_xla::ElementType) -> Option<DType> {
        use hodu_xla::ElementType;
        match element_type {
            ElementType::F32 => Some(DType::F32),
            ElementType::F64 => Some(DType::F64),
            ElementType::F16 => Some(DType::F16),
            ElementType::Bf16 => Some(DType::BF16),
            ElementType::U8 => Some(DType::U8),
            ElementType::U16 => Some(DType::U16),
            ElementType::U32 => Some(DType::U32),
            ElementType::U64 => Some(DType::U64),
            ElementType::S8 => Some(DType::I8),
            ElementType::S16 => Some(DType::I16),
            ElementType::S32 => Some(DType::I32),
            ElementType::S64 => Some(DType::I64),
            ElementType::Pred => Some(DType::BOOL),
            _ => None,
        }
    }
}

unsafe impl Send for XlaExecutor {}
unsafe impl Sync for XlaExecutor {}

impl ExecutorT for XlaExecutor {
    type CompiledScript = XlaCompiledScript;

    fn backend_name(&self) -> &'static str {
        "xla"
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::CPU]
    }

    fn current_device(&self) -> Device {
        self.device
    }

    fn compile(&mut self, script: &Script, _options: CompileOptions) -> HoduResult<Self::CompiledScript> {
        let script_ir = script
            .get_ir()
            .ok_or_else(|| HoduError::ScriptValidationFailed("Script has no IR".to_string()))?;
        script_ir.validate().map_err(HoduError::ScriptValidationFailed)?;

        // Create input and output mappings exactly like HoduExecutor
        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        for input in &script_ir.graph.metadata.inputs {
            input_mapping.insert(input.name.clone(), input.tensor_id);
        }
        for output in &script_ir.graph.metadata.outputs {
            output_mapping.insert(output.name.clone(), output.tensor_id);
        }

        let tensor_layouts = self.collect_tensor_layouts(script_ir);
        let tensor_dtypes = self.collect_tensor_dtypes(script_ir, script);

        // Create dummy executable first
        let dummy_client = ThreadSafeClient(PjRtClient::cpu().map_err(xla_error_to_hodu_error)?);
        let dummy_builder = XlaBuilder::new("dummy");
        let dummy_param = dummy_builder
            .parameter(0, ElementType::F32, &[1], "dummy")
            .map_err(xla_error_to_hodu_error)?;
        let dummy_computation = dummy_param.build().map_err(xla_error_to_hodu_error)?;
        let dummy_executable = dummy_client
            .compile(&dummy_computation)
            .map_err(xla_error_to_hodu_error)?;

        let compiled_script = XlaCompiledScript {
            executable: Arc::new(ThreadSafeExecutable(dummy_executable)), // Dummy executable, will be replaced
            input_mapping,
            output_mapping,
            tensor_layouts,
            tensor_dtypes,
        };

        // Build the actual XLA computation
        let executable = self.build_xla_computation(&compiled_script, script_ir)?;

        Ok(XlaCompiledScript {
            executable,
            input_mapping: compiled_script.input_mapping,
            output_mapping: compiled_script.output_mapping,
            tensor_layouts: compiled_script.tensor_layouts,
            tensor_dtypes: compiled_script.tensor_dtypes,
        })
    }

    fn execute(&self, compiled: &Self::CompiledScript, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        // Convert inputs to XLA literals using input_mapping (exactly like HoduExecutor)
        let mut xla_inputs = Vec::new();

        // Get input names in consistent order (sort for consistent ordering)
        let mut input_names: Vec<_> = compiled.input_mapping.keys().cloned().collect();
        input_names.sort();

        for input_name in input_names.iter() {
            let tensor = inputs
                .get(input_name.as_str())
                .ok_or_else(|| HoduError::InternalError(format!("Input {} not found", input_name)))?;

            let literal = self.tensor_to_literal(tensor).map_err(|e| {
                HoduError::InternalError(format!("Failed to convert input {} to literal: {}", input_name, e))
            })?;
            xla_inputs.push(literal);
        }

        // Execute the computation
        let result_buffers = compiled
            .executable
            .execute::<hodu_xla::Literal>(&xla_inputs)
            .map_err(|e| HoduError::InternalError(format!("Failed to execute XLA computation: {:?}", e)))?;

        // Convert results back to tensors using output_mapping
        let mut outputs = HashMap::new();

        // Get output names in consistent order (to match compile order)
        let mut output_names: Vec<_> = compiled.output_mapping.keys().cloned().collect();
        output_names.sort(); // Must match the order used in compile

        if output_names.len() == 1 {
            // Single output case
            let result_literal = result_buffers[0][0]
                .to_literal_sync()
                .map_err(|e| HoduError::InternalError(format!("Failed to convert result to literal: {:?}", e)))?;

            let output_name = &output_names[0];

            // Get expected dtype from tensor_dtypes mapping
            let output_tensor_id = compiled
                .output_mapping
                .get(output_name)
                .ok_or_else(|| HoduError::InternalError(format!("Output tensor ID not found for {}", output_name)))?;
            let expected_dtype = compiled
                .tensor_dtypes
                .get(output_tensor_id)
                .copied()
                .unwrap_or(DType::F32);

            // For operations that change dtype (like argmax/argmin), detect actual dtype from literal
            let actual_dtype =
                self.xla_element_type_to_dtype(result_literal.element_type().map_err(|e| {
                    HoduError::InternalError(format!("Failed to get element type from literal: {:?}", e))
                })?);
            let dtype_to_use = actual_dtype.unwrap_or(expected_dtype);

            let tensor = self
                .literal_to_tensor(&result_literal, dtype_to_use)
                .map_err(|e| HoduError::InternalError(format!("Failed to convert literal to tensor: {}", e)))?;

            outputs.insert(output_name.clone(), tensor);
        } else {
            // Multiple output case - access tuple elements directly
            if result_buffers[0].len() != output_names.len() {
                return Err(HoduError::InternalError(format!(
                    "Tuple has {} elements but expected {} outputs",
                    result_buffers[0].len(),
                    output_names.len()
                )));
            }

            for (i, output_name) in output_names.iter().enumerate() {
                let element_literal = result_buffers[0][i].to_literal_sync().map_err(|e| {
                    HoduError::InternalError(format!("Failed to convert tuple element {} to literal: {:?}", i, e))
                })?;

                // Get expected dtype from tensor_dtypes mapping
                let output_tensor_id = compiled.output_mapping.get(output_name).ok_or_else(|| {
                    HoduError::InternalError(format!("Output tensor ID not found for {}", output_name))
                })?;
                let expected_dtype = compiled
                    .tensor_dtypes
                    .get(output_tensor_id)
                    .copied()
                    .unwrap_or(DType::F32);

                let tensor = self.literal_to_tensor(&element_literal, expected_dtype).map_err(|e| {
                    HoduError::InternalError(format!("Failed to convert literal {} to tensor: {}", i, e))
                })?;

                outputs.insert(output_name.clone(), tensor);
            }
        }

        Ok(outputs)
    }

    fn cleanup(&mut self) -> HoduResult<()> {
        Ok(())
    }
}

impl XlaExecutor {
    fn convert_constant_to_xla_op(
        &self,
        builder: &XlaBuilder,
        constant: &crate::backends::script::ir::ConstantNode,
    ) -> HoduResult<XlaOp> {
        use crate::types::dtype::DType;

        // Decompress data if needed
        let data = match &constant.compression {
            #[cfg(all(feature = "serde", feature = "std"))]
            Some(crate::backends::script::ir::CompressionType::Gzip) => {
                let mut decoder = flate2::read::GzDecoder::new(&constant.data[..]);
                let mut decompressed = Vec::new();
                std::io::Read::read_to_end(&mut decoder, &mut decompressed)
                    .map_err(|e| HoduError::DecompressionError(e.to_string()))?;
                decompressed
            },
            #[cfg(not(all(feature = "serde", feature = "std")))]
            Some(crate::backends::script::ir::CompressionType::Gzip) => {
                return Err(HoduError::InternalError(
                    "Gzip decompression requires both 'serde' and 'std' features to be enabled".to_string(),
                ));
            },
            Some(crate::backends::script::ir::CompressionType::None) => constant.data.clone(),
            Some(crate::backends::script::ir::CompressionType::Zstd) => {
                return Err(HoduError::InternalError(
                    "Zstd decompression not implemented for XLA".to_string(),
                ));
            },
            None => constant.data.clone(),
        };

        let dims: Vec<i64> = constant.shape.iter().map(|&d| d as i64).collect();

        // Convert DType to XLA ElementType
        let element_type = Self::dtype_to_element_type(constant.dtype)?;

        // Create XLA constant based on dtype
        let xla_op = match constant.dtype {
            DType::F32 => {
                let values: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::F64 => {
                let values: Vec<f64> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                        ])
                    })
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::I32 => {
                let values: Vec<i32> = data
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::I64 => {
                let values: Vec<i64> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                        ])
                    })
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::I16 => {
                let values: Vec<i16> = data
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::I8 => {
                let values: Vec<i8> = data.iter().map(|&b| b as i8).collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::U32 => {
                let values: Vec<u32> = data
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::U64 => {
                let values: Vec<u64> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                        ])
                    })
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::U16 => {
                let values: Vec<u16> = data
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::U8 => self.create_xla_constant(builder, &data, &dims, element_type),
            DType::BOOL => {
                let literal = hodu_xla::Literal::create_from_shape_and_untyped_data(
                    element_type,
                    &dims.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                    &data,
                )
                .map_err(xla_error_to_hodu_error)?;
                Ok(builder.constant_literal(&literal).map_err(xla_error_to_hodu_error)?)
            },
            // For F16 and BF16, we'll create them using literal directly since they may not implement NativeType
            DType::F16 => {
                let literal = hodu_xla::Literal::create_from_shape_and_untyped_data(
                    element_type,
                    &dims.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                    &data,
                )
                .map_err(xla_error_to_hodu_error)?;
                Ok(builder.constant_literal(&literal).map_err(xla_error_to_hodu_error)?)
            },
            DType::BF16 => {
                let literal = hodu_xla::Literal::create_from_shape_and_untyped_data(
                    element_type,
                    &dims.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                    &data,
                )
                .map_err(xla_error_to_hodu_error)?;
                Ok(builder.constant_literal(&literal).map_err(xla_error_to_hodu_error)?)
            },
            DType::F8E4M3 | DType::F8E5M2 => {
                return Err(HoduError::InternalError(format!(
                    "XLA does not support {:?} dtype",
                    constant.dtype
                )));
            },
        }?;

        Ok(xla_op)
    }

    fn create_xla_constant<T: hodu_xla::NativeType + Copy>(
        &self,
        builder: &XlaBuilder,
        values: &[T],
        dims: &[i64],
        element_type: ElementType,
    ) -> HoduResult<XlaOp> {
        if dims.is_empty() {
            // Scalar constant
            builder.constant_r0(values[0]).map_err(xla_error_to_hodu_error)
        } else if dims.len() == 1 {
            // 1D constant
            builder.constant_r1(values).map_err(xla_error_to_hodu_error)
        } else {
            // Multi-dimensional constant - use literal creation
            let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
            let literal = self.create_literal_from_values(values, &shape, element_type)?;
            builder.constant_literal(&literal).map_err(xla_error_to_hodu_error)
        }
    }

    fn create_literal_from_values<T>(
        &self,
        values: &[T],
        shape: &[usize],
        element_type: ElementType,
    ) -> HoduResult<hodu_xla::Literal>
    where
        T: Copy,
    {
        use std::mem;

        // Convert values to bytes based on type
        let data_bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, mem::size_of_val(values)) };

        hodu_xla::Literal::create_from_shape_and_untyped_data(element_type, shape, data_bytes)
            .map_err(xla_error_to_hodu_error)
    }
}
