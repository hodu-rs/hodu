use super::{helpers::xla_error_to_hodu_error, ThreadSafeClient, ThreadSafeExecutable, XlaCompiledScript, XlaExecutor};
use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    op::{
        BinaryLogicalOp, BinaryOp, CastOp, CmpOp, CmpScalarOp, ConvOp, IndexingOp, MatrixOp, MemoryOp, Op, ReduceOp,
        ShapeOp, ShapeScalarsOp, UnaryLogicalOp, UnaryOp, UnaryScalarOp, WindowingOp,
    },
    scalar::Scalar,
    script::{ir::ScriptIR, Script},
    tensor::{from_storage, Tensor, TensorId},
    types::{device::Device, dtype::DType, layout::Layout},
};
use hodu_xla::{ElementType, Literal, PjRtClient, PrimitiveType, XlaBuilder, XlaOp};
use std::collections::HashMap;
use std::f32;
use std::sync::Arc;

impl XlaExecutor {
    /// Convert DType to XLA ElementType
    pub(super) fn dtype_to_element_type(dtype: DType) -> HoduResult<ElementType> {
        match dtype {
            DType::BOOL => Ok(ElementType::Pred),
            DType::BF16 => Ok(ElementType::Bf16),
            DType::F16 => Ok(ElementType::F16),
            DType::F32 => Ok(ElementType::F32),
            DType::F64 => Ok(ElementType::F64),
            #[cfg(feature = "u8")]
            DType::U8 => Ok(ElementType::U8),
            DType::U16 => Ok(ElementType::U16),
            #[cfg(feature = "u32")]
            DType::U32 => Ok(ElementType::U32),
            #[cfg(feature = "u64")]
            DType::U64 => Ok(ElementType::U64),
            DType::I8 => Ok(ElementType::S8),
            #[cfg(feature = "i16")]
            DType::I16 => Ok(ElementType::S16),
            DType::I32 => Ok(ElementType::S32),
            #[cfg(feature = "i64")]
            DType::I64 => Ok(ElementType::S64),
            _ => Err(HoduError::InternalError(format!(
                "XLA does not support {:?} dtype",
                dtype
            ))),
        }
    }

    pub(super) fn collect_tensor_layouts(&self, script_ir: &ScriptIR) -> HashMap<TensorId, Layout> {
        // Estimate capacity: node layouts + inputs + outputs
        let estimated_layout_count = script_ir.graph.topology.nodes.len() * 2
            + script_ir.graph.metadata.inputs.len()
            + script_ir.graph.metadata.outputs.len();
        let mut tensor_layouts = HashMap::with_capacity(estimated_layout_count);

        #[cfg(feature = "rayon")]
        {
            let node_layouts: Vec<_> = script_ir
                .graph
                .topology
                .nodes
                .par_iter()
                .flat_map(|node| {
                    let mut layouts = Vec::new();
                    for (layout, &tensor_id) in node.input_layouts.iter().zip(&node.input_tensors) {
                        layouts.push((tensor_id, layout.clone()));
                    }
                    for (layout, &tensor_id) in node.output_layouts.iter().zip(&node.output_tensors) {
                        layouts.push((tensor_id, layout.clone()));
                    }
                    layouts
                })
                .collect();

            for (tensor_id, layout) in node_layouts {
                tensor_layouts.insert(tensor_id, layout);
            }
        }
        #[cfg(not(feature = "rayon"))]
        {
            for node in &script_ir.graph.topology.nodes {
                for (layout, &tensor_id) in node.input_layouts.iter().zip(&node.input_tensors) {
                    tensor_layouts.insert(tensor_id, layout.clone());
                }
                for (layout, &tensor_id) in node.output_layouts.iter().zip(&node.output_tensors) {
                    tensor_layouts.insert(tensor_id, layout.clone());
                }
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

    pub(super) fn collect_tensor_dtypes(&self, script_ir: &ScriptIR, script: &Script) -> HashMap<TensorId, DType> {
        let mut tensor_dtypes = HashMap::with_capacity(script_ir.graph.metadata.tensor_info.len());

        #[cfg(feature = "rayon")]
        {
            let dtypes: Vec<_> = script_ir
                .graph
                .metadata
                .tensor_info
                .par_iter()
                .filter_map(|(&tensor_id, tensor_info)| tensor_info.dtype.map(|dtype| (tensor_id, dtype)))
                .collect();

            for (tensor_id, dtype) in dtypes {
                tensor_dtypes.insert(tensor_id, dtype);
            }
        }
        #[cfg(not(feature = "rayon"))]
        {
            for (&tensor_id, tensor_info) in &script_ir.graph.metadata.tensor_info {
                if let Some(dtype) = tensor_info.dtype {
                    tensor_dtypes.insert(tensor_id, dtype);
                }
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

    pub(super) fn build_xla_computation(
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

    pub(super) fn execute_xla_operation(
        &self,
        builder: &XlaBuilder,
        operation: &Op,
        input_ops: &[XlaOp],
        _compiled_script: &XlaCompiledScript,
        current_node: &crate::script::ir::GraphNode,
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
                    UnaryOp::Sqrt => input_ops[0].sqrt(),
                    UnaryOp::Recip => {
                        let one = builder.constant_r0(1.0f32).map_err(xla_error_to_hodu_error)?;
                        one.div_(&input_ops[0])
                    },

                    UnaryOp::Relu => {
                        let zero = builder.constant_r0(0.0f32).map_err(xla_error_to_hodu_error)?;
                        input_ops[0].max(&zero)
                    },
                    UnaryOp::Sigmoid => input_ops[0].logistic(), // Use XLA's builtin logistic
                    UnaryOp::Tanh => input_ops[0].tanh(),
                    UnaryOp::Gelu => {
                        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                        let sqrt_2_over_pi = builder.constant_r0(0.797_884_6_f32).map_err(xla_error_to_hodu_error)?; // sqrt(2/π)
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
                    UnaryOp::Softplus => {
                        // softplus(x) = log(1 + exp(x))
                        let one = builder.constant_r0(1.0f32).map_err(xla_error_to_hodu_error)?;
                        let exp_x = input_ops[0].exp().map_err(xla_error_to_hodu_error)?;
                        let one_plus_exp = one.add_(&exp_x).map_err(xla_error_to_hodu_error)?;
                        one_plus_exp.log()
                    },
                    UnaryOp::Silu => {
                        // silu(x) = x * sigmoid(x)
                        let x = &input_ops[0];
                        let sigmoid = x.logistic().map_err(xla_error_to_hodu_error)?;
                        x.mul_(&sigmoid)
                    },
                    UnaryOp::Mish => {
                        // mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
                        let one = builder.constant_r0(1.0f32).map_err(xla_error_to_hodu_error)?;
                        let x = &input_ops[0];
                        let exp_x = x.exp().map_err(xla_error_to_hodu_error)?;
                        let one_plus_exp = one.add_(&exp_x).map_err(xla_error_to_hodu_error)?;
                        let softplus = one_plus_exp.log().map_err(xla_error_to_hodu_error)?;
                        let tanh_softplus = softplus.tanh().map_err(xla_error_to_hodu_error)?;
                        x.mul_(&tanh_softplus)
                    },

                    UnaryOp::Sin => input_ops[0].sin(),
                    UnaryOp::Cos => input_ops[0].cos(),
                    UnaryOp::Tan => {
                        let sin_val = input_ops[0].sin().map_err(xla_error_to_hodu_error)?;
                        let cos_val = input_ops[0].cos().map_err(xla_error_to_hodu_error)?;
                        sin_val.div_(&cos_val)
                    },

                    UnaryOp::Exp => input_ops[0].exp(),
                    UnaryOp::Exp2 => {
                        let ln_2 = builder
                            .constant_r0(f32::consts::LN_2)
                            .map_err(xla_error_to_hodu_error)?; // ln(2)
                        let scaled = input_ops[0].mul_(&ln_2).map_err(xla_error_to_hodu_error)?;
                        scaled.exp()
                    },
                    UnaryOp::Exp10 => {
                        let ln_10 = builder
                            .constant_r0(f32::consts::LN_10)
                            .map_err(xla_error_to_hodu_error)?; // ln(10)
                        let scaled = input_ops[0].mul_(&ln_10).map_err(xla_error_to_hodu_error)?;
                        scaled.exp()
                    },
                    UnaryOp::Ln => input_ops[0].log(),
                    UnaryOp::Log2 => {
                        let ln_val = input_ops[0].log().map_err(xla_error_to_hodu_error)?;
                        let ln_2 = builder
                            .constant_r0(f32::consts::LN_2)
                            .map_err(xla_error_to_hodu_error)?; // ln(2)
                        ln_val.div_(&ln_2)
                    },
                    UnaryOp::Log10 => {
                        let ln_val = input_ops[0].log().map_err(xla_error_to_hodu_error)?;
                        let ln_10 = builder
                            .constant_r0(f32::consts::LN_10)
                            .map_err(xla_error_to_hodu_error)?; // ln(10)
                        ln_val.div_(&ln_10)
                    },
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
                    UnaryScalarOp::Prelu => {
                        // PReLU: x if x > 0, else α * x
                        let zero = builder.constant_r0(0.0f32).map_err(xla_error_to_hodu_error)?;
                        let negative_part = scalar_op.mul_(&input_ops[0]).map_err(xla_error_to_hodu_error)?;
                        let condition = input_ops[0].gt(&zero).map_err(xla_error_to_hodu_error)?;
                        condition.select(&input_ops[0], &negative_part)
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
            Op::Reduce(reduce_op, _, keep_dim, dims_scalars) => {
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
                            input_ops[0].reduce_sum(&all_dims, *keep_dim)
                        } else {
                            input_ops[0].reduce_sum(&dims, *keep_dim)
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
                            input_ops[0].reduce_mean(&all_dims, *keep_dim)
                        } else {
                            input_ops[0].reduce_mean(&dims, *keep_dim)
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
                            input_ops[0].reduce_max(&all_dims, *keep_dim)
                        } else {
                            input_ops[0].reduce_max(&dims, *keep_dim)
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
                            input_ops[0].reduce_min(&all_dims, *keep_dim)
                        } else {
                            input_ops[0].reduce_min(&dims, *keep_dim)
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
                            input_ops[0].reduce(one, prod_computation, &all_dims, *keep_dim)
                        } else {
                            input_ops[0].reduce(one, prod_computation, &dims, *keep_dim)
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
                            squared
                                .reduce_mean(&all_dims, *keep_dim)
                                .map_err(xla_error_to_hodu_error)?
                        } else {
                            squared.reduce_mean(&dims, *keep_dim).map_err(xla_error_to_hodu_error)?
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
                            squared.reduce_mean(&all_dims, *keep_dim)
                        } else {
                            squared.reduce_mean(&dims, *keep_dim)
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
                            squared
                                .reduce_sum(&all_dims, *keep_dim)
                                .map_err(xla_error_to_hodu_error)?
                        } else {
                            squared.reduce_sum(&dims, *keep_dim).map_err(xla_error_to_hodu_error)?
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
                        let input_dims: Vec<i64> = shape.dims().to_vec();
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
                            .reduce_min(&[dim], *keep_dim)
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
                        let input_dims: Vec<i64> = shape.dims().to_vec();
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
                            .reduce_min(&[dim], *keep_dim)
                            .map_err(xla_error_to_hodu_error)
                    },
                    ReduceOp::Any => {
                        let input = &input_ops[0];

                        // Check the actual XLA type, not the tensor dtype
                        let input_shape = input.shape().map_err(xla_error_to_hodu_error)?;
                        let is_already_bool = match &input_shape {
                            hodu_xla::Shape::Array(array_shape) => {
                                matches!(array_shape.element_type(), ElementType::Pred)
                            },
                            _ => false,
                        };

                        // If input is already bool (Pred type), use it directly; otherwise convert to bool
                        let as_bool = if is_already_bool {
                            input.clone()
                        } else {
                            // Get tensor dtype to determine conversion strategy
                            let input_tensor_id = current_node.input_tensors[0];
                            let input_tensor = crate::tensor::tensor_from_id(input_tensor_id);
                            let input_dtype = input_tensor.get_dtype();

                            let builder = input.builder();
                            let input_as_float = if input_dtype.is_int() {
                                input.convert(PrimitiveType::F32).map_err(xla_error_to_hodu_error)?
                            } else {
                                input.clone()
                            };
                            let zero = builder.c0(0.0f32).map_err(xla_error_to_hodu_error)?;
                            input_as_float.ne(&zero).map_err(xla_error_to_hodu_error)?
                        };

                        // Use logical OR reduction
                        if dims.is_empty() {
                            // Reduce over all dimensions
                            let input_shape = input.shape().map_err(xla_error_to_hodu_error)?;
                            let all_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                                _ => {
                                    return Err(HoduError::InternalError("Expected array shape for reduce".to_string()))
                                },
                            };
                            as_bool.reduce_or(&all_dims, *keep_dim)
                        } else {
                            as_bool.reduce_or(&dims, *keep_dim)
                        }
                        .map_err(xla_error_to_hodu_error)
                    },
                    ReduceOp::All => {
                        let input = &input_ops[0];

                        // Check the actual XLA type, not the tensor dtype
                        let input_shape = input.shape().map_err(xla_error_to_hodu_error)?;
                        let is_already_bool = match &input_shape {
                            hodu_xla::Shape::Array(array_shape) => {
                                matches!(array_shape.element_type(), ElementType::Pred)
                            },
                            _ => false,
                        };

                        // If input is already bool (Pred type), use it directly; otherwise convert to bool
                        let as_bool = if is_already_bool {
                            input.clone()
                        } else {
                            // Get tensor dtype to determine conversion strategy
                            let input_tensor_id = current_node.input_tensors[0];
                            let input_tensor = crate::tensor::tensor_from_id(input_tensor_id);
                            let input_dtype = input_tensor.get_dtype();

                            let builder = input.builder();
                            let input_as_float = if input_dtype.is_int() {
                                input.convert(PrimitiveType::F32).map_err(xla_error_to_hodu_error)?
                            } else {
                                input.clone()
                            };
                            let zero = builder.c0(0.0f32).map_err(xla_error_to_hodu_error)?;
                            input_as_float.ne(&zero).map_err(xla_error_to_hodu_error)?
                        };

                        // Use logical AND reduction
                        if dims.is_empty() {
                            // Reduce over all dimensions
                            let input_shape = input.shape().map_err(xla_error_to_hodu_error)?;
                            let all_dims: Vec<i64> = match input_shape {
                                hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                                _ => {
                                    return Err(HoduError::InternalError("Expected array shape for reduce".to_string()))
                                },
                            };
                            as_bool.reduce_and(&all_dims, *keep_dim)
                        } else {
                            as_bool.reduce_and(&dims, *keep_dim)
                        }
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
                    IndexingOp::IndexPut => {
                        // input_ops: [self, indices, values]
                        if input_ops.len() != 3 {
                            return Err(HoduError::InternalError("IndexPut requires 3 inputs".to_string()));
                        }

                        // Extract dim from params
                        let dim = params
                            .first()
                            .ok_or_else(|| {
                                HoduError::InternalError("IndexPut requires dimension parameter".to_string())
                            })?
                            .to_u64() as i64;

                        // Create update computation that replaces values
                        let element_type = input_ops[0].ty().map_err(xla_error_to_hodu_error)?;
                        let element_type = Self::element_type_to_element_type(element_type)?;

                        let update_builder = XlaBuilder::new("index_put_update");
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
                        let values_shape = input_ops[2].array_shape().map_err(xla_error_to_hodu_error)?;

                        let base_rank = base_shape.dims().len();
                        let indices_rank = indices_shape.dims().len();
                        let base_dims = base_shape.dims();
                        let indices_dims = indices_shape.dims();
                        let _values_dims = values_shape.dims();

                        // For multidimensional index_put, convert PyTorch semantics to XLA
                        if indices_rank > 1 && base_rank > 1 {
                            // PyTorch: values[i,j,...] -> input[indices[i,j,...], j, ...]
                            // XLA: requires flattened indices with proper offset calculation

                            // Flatten base, indices, values
                            let base_size: i64 = base_dims.iter().product();
                            let indices_size: i64 = indices_dims.iter().product();

                            let base_flat = input_ops[0].reshape(&[base_size]).map_err(xla_error_to_hodu_error)?;
                            let indices_flat =
                                input_ops[1].reshape(&[indices_size]).map_err(xla_error_to_hodu_error)?;
                            let values_flat = input_ops[2].reshape(&[indices_size]).map_err(xla_error_to_hodu_error)?;

                            // Calculate strides for mapping positions
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
                                    &values_flat,
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

            // Convolution operations
            Op::Conv(conv_op, _, _, params) => {
                if input_ops.len() != 2 {
                    return Err(HoduError::InternalError(
                        "Convolution operation requires exactly 2 inputs (input, weight)".to_string(),
                    ));
                }

                match conv_op {
                    ConvOp::Conv1d => {
                        // Extract parameters
                        if params.len() < 8 {
                            return Err(HoduError::InternalError("Conv1d requires 8 parameters".to_string()));
                        }

                        let stride = params[6].to_u64() as i64;
                        let padding = params[5].to_u64() as i64;
                        let dilation = params[7].to_u64() as i64;

                        // Conv1d: input [N, C, L], kernel [Co, Ci, K]
                        // XLA dimension numbers:
                        // - input_batch_dimension: 0 (N)
                        // - input_feature_dimension: 1 (C)
                        // - input_spatial_dimensions: [2] (L)
                        // - kernel_output_feature_dimension: 0 (Co)
                        // - kernel_input_feature_dimension: 1 (Ci)
                        // - kernel_spatial_dimensions: [2] (K)

                        input_ops[0]
                            .conv_general_dilated(
                                &input_ops[1],
                                &[stride],             // window strides
                                &[(padding, padding)], // padding (low, high)
                                &[],                   // lhs_dilation
                                &[dilation],           // rhs_dilation
                                0,                     // input_batch_dimension
                                1,                     // input_feature_dimension
                                &[2],                  // input_spatial_dimensions
                                1,                     // kernel_input_feature_dimension
                                0,                     // kernel_output_feature_dimension
                                &[2],                  // kernel_spatial_dimensions
                                1,                     // feature_group_count
                                1,                     // batch_group_count
                            )
                            .map_err(xla_error_to_hodu_error)
                    },
                    ConvOp::Conv2d => {
                        // Extract parameters
                        if params.len() < 10 {
                            return Err(HoduError::InternalError("Conv2d requires 10 parameters".to_string()));
                        }

                        let stride = params[8].to_u64() as i64;
                        let padding = params[7].to_u64() as i64;
                        let dilation = params[9].to_u64() as i64;

                        // Conv2d: input [N, C, H, W], kernel [Co, Ci, Kh, Kw]
                        // XLA dimension numbers:
                        // - input_batch_dimension: 0 (N)
                        // - input_feature_dimension: 1 (C)
                        // - input_spatial_dimensions: [2, 3] (H, W)
                        // - kernel_output_feature_dimension: 0 (Co)
                        // - kernel_input_feature_dimension: 1 (Ci)
                        // - kernel_spatial_dimensions: [2, 3] (Kh, Kw)

                        input_ops[0]
                            .conv_general_dilated(
                                &input_ops[1],
                                &[stride, stride],                         // window strides
                                &[(padding, padding), (padding, padding)], // padding
                                &[],                                       // lhs_dilation
                                &[dilation, dilation],                     // rhs_dilation
                                0,                                         // input_batch_dimension
                                1,                                         // input_feature_dimension
                                &[2, 3],                                   // input_spatial_dimensions
                                1,                                         // kernel_input_feature_dimension
                                0,                                         // kernel_output_feature_dimension
                                &[2, 3],                                   // kernel_spatial_dimensions
                                1,                                         // feature_group_count
                                1,                                         // batch_group_count
                            )
                            .map_err(xla_error_to_hodu_error)
                    },
                    ConvOp::Conv3d => {
                        // Extract parameters
                        if params.len() < 12 {
                            return Err(HoduError::InternalError("Conv3d requires 12 parameters".to_string()));
                        }

                        let stride = params[10].to_u64() as i64;
                        let padding = params[9].to_u64() as i64;
                        let dilation = params[11].to_u64() as i64;

                        // Conv3d: input [N, C, D, H, W], kernel [Co, Ci, Kd, Kh, Kw]
                        // XLA dimension numbers:
                        // - input_batch_dimension: 0 (N)
                        // - input_feature_dimension: 1 (C)
                        // - input_spatial_dimensions: [2, 3, 4] (D, H, W)
                        // - kernel_output_feature_dimension: 0 (Co)
                        // - kernel_input_feature_dimension: 1 (Ci)
                        // - kernel_spatial_dimensions: [2, 3, 4] (Kd, Kh, Kw)

                        input_ops[0]
                            .conv_general_dilated(
                                &input_ops[1],
                                &[stride, stride, stride], // window strides
                                &[(padding, padding), (padding, padding), (padding, padding)], // padding
                                &[],                       // lhs_dilation
                                &[dilation, dilation, dilation], // rhs_dilation
                                0,                         // input_batch_dimension
                                1,                         // input_feature_dimension
                                &[2, 3, 4],                // input_spatial_dimensions
                                1,                         // kernel_input_feature_dimension
                                0,                         // kernel_output_feature_dimension
                                &[2, 3, 4],                // kernel_spatial_dimensions
                                1,                         // feature_group_count
                                1,                         // batch_group_count
                            )
                            .map_err(xla_error_to_hodu_error)
                    },
                    ConvOp::ConvTranspose1d => {
                        // Extract parameters
                        if params.len() < 9 {
                            return Err(HoduError::InternalError(
                                "ConvTranspose1d requires 9 parameters".to_string(),
                            ));
                        }

                        let kernel_size = params[4].to_u64() as i64;
                        let padding = params[5].to_u64() as i64;
                        let output_padding = params[6].to_u64() as i64;
                        let stride = params[7].to_u64() as i64;
                        let dilation = params[8].to_u64() as i64;

                        // ConvTranspose1d: input [N, Ci, L], kernel [Ci, Co, K]
                        // Transposed convolution padding calculation
                        // XLA formula: out_len = (in_len - 1) * lhs_dilation + kernel_size - pad_low - pad_high
                        // HODU formula: out_len = (in_len - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
                        //
                        // For stride >= 2: use lhs_dilation = stride, padding = (kernel_size-1)/2
                        // For stride == 1: need to adjust padding to match HODU semantics

                        // For transposed convolution with stride >= 2, use lhs_dilation
                        // For stride == 1, transposed conv is equivalent to regular conv with adjusted padding
                        let use_lhs_dilation = stride > 1;

                        let (xla_lhs_dilation, xla_padding_low, xla_padding_high) = if use_lhs_dilation {
                            // For stride > 1, use lhs_dilation = stride
                            // XLA with lhs_dilation: out = (in-1)*lhs_dilation + 2 - kernel + pad_total (empirical formula)
                            // HODU formula: out = (in-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1
                            //              = (in-1)*stride + kernel - 2*padding + output_padding (with dilation=1)
                            //
                            // Matching: (in-1)*stride + 2 - kernel + pad_total = (in-1)*stride + kernel - 2*padding + output_padding
                            //          2 - kernel + pad_total = kernel - 2*padding + output_padding
                            //          pad_total = 2*kernel - 2 - 2*padding + output_padding

                            let pad_total = 2 * kernel_size - 2 - 2 * padding + output_padding;
                            let pad_low = pad_total / 2;
                            let pad_high = pad_total - pad_low;

                            (vec![stride], pad_low, pad_high)
                        } else {
                            // For stride=1, use empty lhs_dilation (regular conv mode)
                            // Need to adjust padding to match HODU output size
                            // HODU: out = (in-1)*1 + dilation*(K-1) + 1 - 2*padding + output_padding
                            //      = in + dilation*(K-1) - 2*padding + output_padding
                            // XLA (no dilation):  out = (in - K + pad_low + pad_high) / window_stride + 1
                            //                        = in - K + pad_total + 1  (with window_stride=1)
                            //
                            // Matching: in - K + pad_total + 1 = in + dilation*(K-1) - 2*padding + output_padding
                            //          pad_total = K - 1 + dilation*(K-1) - 2*padding + output_padding
                            //          pad_total = dilation*K - 1 + K - 1 - 2*padding + output_padding
                            // For dilation=1: pad_total = K + K - 2 - 2*padding + output_padding
                            //                = 2*K - 2 - 2*padding + output_padding

                            let pad_total = dilation * kernel_size + kernel_size - 2 - 2 * padding + output_padding;
                            let pad_low = pad_total / 2;
                            let pad_high = pad_total - pad_low;

                            (vec![], pad_low, pad_high)
                        };

                        // Reverse kernel weights for transposed convolution (only for stride=1)
                        let kernel_op = if use_lhs_dilation {
                            // For stride > 1, don't reverse (lhs_dilation handles it)
                            &input_ops[1]
                        } else {
                            // For stride = 1, reverse the kernel
                            &input_ops[1].rev(&[2]).map_err(xla_error_to_hodu_error)?
                        };

                        input_ops[0]
                            .conv_general_dilated(
                                kernel_op,
                                &[1],                                   // window strides = 1 for transpose
                                &[(xla_padding_low, xla_padding_high)], // XLA padding
                                &xla_lhs_dilation,                      // lhs_dilation (empty for stride=1)
                                &[dilation],                            // rhs_dilation (kernel dilation)
                                0,                                      // input_batch_dimension
                                1,                                      // input_feature_dimension
                                &[2],                                   // input_spatial_dimensions
                                0,                                      // kernel_input_feature_dimension
                                1,                                      // kernel_output_feature_dimension
                                &[2],                                   // kernel_spatial_dimensions
                                1,                                      // feature_group_count
                                1,                                      // batch_group_count
                            )
                            .map_err(xla_error_to_hodu_error)
                    },
                    ConvOp::ConvTranspose2d => {
                        // Extract parameters
                        if params.len() < 11 {
                            return Err(HoduError::InternalError(
                                "ConvTranspose2d requires 11 parameters".to_string(),
                            ));
                        }

                        let kernel_height = params[3].to_u64() as i64;
                        let kernel_width = params[4].to_u64() as i64;
                        let padding = params[7].to_u64() as i64;
                        let output_padding = params[8].to_u64() as i64;
                        let stride = params[9].to_u64() as i64;
                        let dilation = params[10].to_u64() as i64;

                        // ConvTranspose2d: input [N, Ci, H, W], kernel [Ci, Co, Kh, Kw]
                        let use_lhs_dilation = stride > 1;

                        let (
                            xla_lhs_dilation,
                            xla_padding_h_low,
                            xla_padding_h_high,
                            xla_padding_w_low,
                            xla_padding_w_high,
                        ) = if use_lhs_dilation {
                            // For stride > 1
                            let pad_h_total = 2 * kernel_height - 2 - 2 * padding + output_padding;
                            let pad_h_low = pad_h_total / 2;
                            let pad_h_high = pad_h_total - pad_h_low;
                            let pad_w_total = 2 * kernel_width - 2 - 2 * padding + output_padding;
                            let pad_w_low = pad_w_total / 2;
                            let pad_w_high = pad_w_total - pad_w_low;
                            (vec![stride, stride], pad_h_low, pad_h_high, pad_w_low, pad_w_high)
                        } else {
                            // For stride = 1
                            let pad_h_total =
                                dilation * kernel_height + kernel_height - 2 - 2 * padding + output_padding;
                            let pad_h_low = pad_h_total / 2;
                            let pad_h_high = pad_h_total - pad_h_low;
                            let pad_w_total = dilation * kernel_width + kernel_width - 2 - 2 * padding + output_padding;
                            let pad_w_low = pad_w_total / 2;
                            let pad_w_high = pad_w_total - pad_w_low;
                            (vec![], pad_h_low, pad_h_high, pad_w_low, pad_w_high)
                        };

                        let kernel_op = if use_lhs_dilation {
                            &input_ops[1]
                        } else {
                            &input_ops[1].rev(&[2, 3]).map_err(xla_error_to_hodu_error)?
                        };

                        input_ops[0]
                            .conv_general_dilated(
                                kernel_op,
                                &[1, 1],
                                &[
                                    (xla_padding_h_low, xla_padding_h_high),
                                    (xla_padding_w_low, xla_padding_w_high),
                                ],
                                &xla_lhs_dilation,
                                &[dilation, dilation],
                                0,       // input_batch_dimension
                                1,       // input_feature_dimension
                                &[2, 3], // input_spatial_dimensions
                                0,       // kernel_input_feature_dimension
                                1,       // kernel_output_feature_dimension
                                &[2, 3], // kernel_spatial_dimensions
                                1,       // feature_group_count
                                1,       // batch_group_count
                            )
                            .map_err(xla_error_to_hodu_error)
                    },
                    ConvOp::ConvTranspose3d => {
                        // Extract parameters
                        if params.len() < 13 {
                            return Err(HoduError::InternalError(
                                "ConvTranspose3d requires 13 parameters".to_string(),
                            ));
                        }

                        let kernel_depth = params[4].to_u64() as i64;
                        let kernel_height = params[5].to_u64() as i64;
                        let kernel_width = params[6].to_u64() as i64;
                        let padding = params[9].to_u64() as i64;
                        let output_padding = params[10].to_u64() as i64;
                        let stride = params[11].to_u64() as i64;
                        let dilation = params[12].to_u64() as i64;

                        // ConvTranspose3d: input [N, Ci, D, H, W], kernel [Ci, Co, Kd, Kh, Kw]
                        let use_lhs_dilation = stride > 1;

                        let (
                            xla_lhs_dilation,
                            xla_padding_d_low,
                            xla_padding_d_high,
                            xla_padding_h_low,
                            xla_padding_h_high,
                            xla_padding_w_low,
                            xla_padding_w_high,
                        ) = if use_lhs_dilation {
                            // For stride > 1
                            let pad_d_total = 2 * kernel_depth - 2 - 2 * padding + output_padding;
                            let pad_d_low = pad_d_total / 2;
                            let pad_d_high = pad_d_total - pad_d_low;
                            let pad_h_total = 2 * kernel_height - 2 - 2 * padding + output_padding;
                            let pad_h_low = pad_h_total / 2;
                            let pad_h_high = pad_h_total - pad_h_low;
                            let pad_w_total = 2 * kernel_width - 2 - 2 * padding + output_padding;
                            let pad_w_low = pad_w_total / 2;
                            let pad_w_high = pad_w_total - pad_w_low;
                            (
                                vec![stride, stride, stride],
                                pad_d_low,
                                pad_d_high,
                                pad_h_low,
                                pad_h_high,
                                pad_w_low,
                                pad_w_high,
                            )
                        } else {
                            // For stride = 1
                            let pad_d_total = dilation * kernel_depth + kernel_depth - 2 - 2 * padding + output_padding;
                            let pad_d_low = pad_d_total / 2;
                            let pad_d_high = pad_d_total - pad_d_low;
                            let pad_h_total =
                                dilation * kernel_height + kernel_height - 2 - 2 * padding + output_padding;
                            let pad_h_low = pad_h_total / 2;
                            let pad_h_high = pad_h_total - pad_h_low;
                            let pad_w_total = dilation * kernel_width + kernel_width - 2 - 2 * padding + output_padding;
                            let pad_w_low = pad_w_total / 2;
                            let pad_w_high = pad_w_total - pad_w_low;
                            (
                                vec![],
                                pad_d_low,
                                pad_d_high,
                                pad_h_low,
                                pad_h_high,
                                pad_w_low,
                                pad_w_high,
                            )
                        };

                        let kernel_op = if use_lhs_dilation {
                            &input_ops[1]
                        } else {
                            &input_ops[1].rev(&[2, 3, 4]).map_err(xla_error_to_hodu_error)?
                        };

                        input_ops[0]
                            .conv_general_dilated(
                                kernel_op,
                                &[1, 1, 1],
                                &[
                                    (xla_padding_d_low, xla_padding_d_high),
                                    (xla_padding_h_low, xla_padding_h_high),
                                    (xla_padding_w_low, xla_padding_w_high),
                                ],
                                &xla_lhs_dilation,
                                &[dilation, dilation, dilation],
                                0,          // input_batch_dimension
                                1,          // input_feature_dimension
                                &[2, 3, 4], // input_spatial_dimensions
                                0,          // kernel_input_feature_dimension
                                1,          // kernel_output_feature_dimension
                                &[2, 3, 4], // kernel_spatial_dimensions
                                1,          // feature_group_count
                                1,          // batch_group_count
                            )
                            .map_err(xla_error_to_hodu_error)
                    },
                    ConvOp::Conv1dGradWeight => {
                        // Extract parameters
                        if params.len() < 8 {
                            return Err(HoduError::InternalError(
                                "Conv1dGradWeight requires 8 parameters".to_string(),
                            ));
                        }

                        let stride = params[6].to_u64() as i64;
                        let padding = params[5].to_u64() as i64;
                        let dilation = params[7].to_u64() as i64;

                        // Conv1dGradWeight computes weight gradient
                        // input: [N, Ci, L], grad_output: [N, Co, L_out]
                        // output: [Co, Ci, K]
                        // This is a convolution with special dimension configuration
                        input_ops[0]
                            .conv_general_dilated(
                                &input_ops[1],
                                &[stride],             // window strides
                                &[(padding, padding)], // padding (low, high)
                                &[],                   // lhs_dilation
                                &[dilation],           // rhs_dilation
                                1,                     // input_batch_dimension -> feature
                                0,                     // input_feature_dimension -> batch
                                &[2],                  // input_spatial_dimensions
                                0,                     // kernel_input_feature_dimension -> batch
                                1,                     // kernel_output_feature_dimension -> feature
                                &[2],                  // kernel_spatial_dimensions
                                1,                     // feature_group_count
                                1,                     // batch_group_count
                            )
                            .map_err(xla_error_to_hodu_error)
                    },
                    ConvOp::Conv2dGradWeight => {
                        // Extract parameters
                        if params.len() < 10 {
                            return Err(HoduError::InternalError(
                                "Conv2dGradWeight requires 10 parameters".to_string(),
                            ));
                        }

                        let stride = params[8].to_u64() as i64;
                        let padding = params[7].to_u64() as i64;
                        let dilation = params[9].to_u64() as i64;

                        // Conv2dGradWeight computes weight gradient
                        // input: [N, Ci, H, W], grad_output: [N, Co, H_out, W_out]
                        // output: [Co, Ci, Kh, Kw]
                        input_ops[0]
                            .conv_general_dilated(
                                &input_ops[1],
                                &[stride, stride],                         // window strides
                                &[(padding, padding), (padding, padding)], // padding
                                &[],                                       // lhs_dilation
                                &[dilation, dilation],                     // rhs_dilation
                                1,                                         // input_batch_dimension -> feature
                                0,                                         // input_feature_dimension -> batch
                                &[2, 3],                                   // input_spatial_dimensions
                                0,                                         // kernel_input_feature_dimension -> batch
                                1,                                         // kernel_output_feature_dimension -> feature
                                &[2, 3],                                   // kernel_spatial_dimensions
                                1,                                         // feature_group_count
                                1,                                         // batch_group_count
                            )
                            .map_err(xla_error_to_hodu_error)
                    },
                    ConvOp::Conv3dGradWeight => {
                        // Extract parameters
                        if params.len() < 12 {
                            return Err(HoduError::InternalError(
                                "Conv3dGradWeight requires 12 parameters".to_string(),
                            ));
                        }

                        let stride = params[10].to_u64() as i64;
                        let padding = params[9].to_u64() as i64;
                        let dilation = params[11].to_u64() as i64;

                        // Conv3dGradWeight computes weight gradient
                        // input: [N, Ci, D, H, W], grad_output: [N, Co, D_out, H_out, W_out]
                        // output: [Co, Ci, Kd, Kh, Kw]
                        input_ops[0]
                            .conv_general_dilated(
                                &input_ops[1],
                                &[stride, stride, stride], // window strides
                                &[(padding, padding), (padding, padding), (padding, padding)], // padding
                                &[],                       // lhs_dilation
                                &[dilation, dilation, dilation], // rhs_dilation
                                1,                         // input_batch_dimension -> feature
                                0,                         // input_feature_dimension -> batch
                                &[2, 3, 4],                // input_spatial_dimensions
                                0,                         // kernel_input_feature_dimension -> batch
                                1,                         // kernel_output_feature_dimension -> feature
                                &[2, 3, 4],                // kernel_spatial_dimensions
                                1,                         // feature_group_count
                                1,                         // batch_group_count
                            )
                            .map_err(xla_error_to_hodu_error)
                    },
                    ConvOp::ConvTranspose1dGradWeight => {
                        // Extract parameters
                        if params.len() < 9 {
                            return Err(HoduError::InternalError(
                                "ConvTranspose1dGradWeight requires 9 parameters".to_string(),
                            ));
                        }

                        let stride = params[7].to_u64() as i64;
                        let padding = params[5].to_u64() as i64;
                        let output_padding = params[6].to_u64() as i64;
                        let dilation = params[8].to_u64() as i64;

                        // ConvTranspose1dGradWeight: similar to Conv1dGradWeight but with special handling
                        input_ops[0]
                            .conv_general_dilated(
                                &input_ops[1],
                                &[stride],
                                &[(padding, padding + output_padding)],
                                &[],
                                &[dilation],
                                1,    // input_batch_dimension -> feature
                                0,    // input_feature_dimension -> batch
                                &[2], // input_spatial_dimensions
                                0,    // kernel_input_feature_dimension -> batch
                                1,    // kernel_output_feature_dimension -> feature
                                &[2], // kernel_spatial_dimensions
                                1,    // feature_group_count
                                1,    // batch_group_count
                            )
                            .map_err(xla_error_to_hodu_error)
                    },
                    ConvOp::ConvTranspose2dGradWeight => {
                        // Extract parameters
                        if params.len() < 11 {
                            return Err(HoduError::InternalError(
                                "ConvTranspose2dGradWeight requires 11 parameters".to_string(),
                            ));
                        }

                        let stride = params[9].to_u64() as i64;
                        let padding = params[7].to_u64() as i64;
                        let output_padding = params[8].to_u64() as i64;
                        let dilation = params[10].to_u64() as i64;

                        // ConvTranspose2dGradWeight: similar to Conv2dGradWeight but with special handling
                        input_ops[0]
                            .conv_general_dilated(
                                &input_ops[1],
                                &[stride, stride],
                                &[(padding, padding + output_padding), (padding, padding + output_padding)],
                                &[],
                                &[dilation, dilation],
                                1,       // input_batch_dimension -> feature
                                0,       // input_feature_dimension -> batch
                                &[2, 3], // input_spatial_dimensions
                                0,       // kernel_input_feature_dimension -> batch
                                1,       // kernel_output_feature_dimension -> feature
                                &[2, 3], // kernel_spatial_dimensions
                                1,       // feature_group_count
                                1,       // batch_group_count
                            )
                            .map_err(xla_error_to_hodu_error)
                    },
                    ConvOp::ConvTranspose3dGradWeight => {
                        // Extract parameters
                        if params.len() < 13 {
                            return Err(HoduError::InternalError(
                                "ConvTranspose3dGradWeight requires 13 parameters".to_string(),
                            ));
                        }

                        let stride = params[11].to_u64() as i64;
                        let padding = params[9].to_u64() as i64;
                        let output_padding = params[10].to_u64() as i64;
                        let dilation = params[12].to_u64() as i64;

                        // ConvTranspose3dGradWeight: similar to Conv3dGradWeight but with special handling
                        input_ops[0]
                            .conv_general_dilated(
                                &input_ops[1],
                                &[stride, stride, stride],
                                &[
                                    (padding, padding + output_padding),
                                    (padding, padding + output_padding),
                                    (padding, padding + output_padding),
                                ],
                                &[],
                                &[dilation, dilation, dilation],
                                1,          // input_batch_dimension -> feature
                                0,          // input_feature_dimension -> batch
                                &[2, 3, 4], // input_spatial_dimensions
                                0,          // kernel_input_feature_dimension -> batch
                                1,          // kernel_output_feature_dimension -> feature
                                &[2, 3, 4], // kernel_spatial_dimensions
                                1,          // feature_group_count
                                1,          // batch_group_count
                            )
                            .map_err(xla_error_to_hodu_error)
                    },
                }
            },

            // Windowing Operations
            Op::Windowing(op, _, params) => {
                if input_ops.len() != 1 {
                    return Err(HoduError::InternalError(
                        "Windowing operation requires exactly 1 input".to_string(),
                    ));
                }

                match op {
                    WindowingOp::ReduceWindow => {
                        // Unpack parameters: rank, window_shape[rank], strides[rank], padding[rank*2], reduction_type
                        if params.is_empty() {
                            return Err(HoduError::InternalError("ReduceWindow requires parameters".to_string()));
                        }

                        let rank = match params[0] {
                            Scalar::I32(r) => r as usize,
                            _ => return Err(HoduError::InternalError("Expected I32 for rank".to_string())),
                        };

                        let expected_len = 1 + rank + rank + (rank * 2) + 1;
                        if params.len() != expected_len {
                            return Err(HoduError::InternalError(format!(
                                "ReduceWindow requires {} parameters, got {}",
                                expected_len,
                                params.len()
                            )));
                        }

                        // Extract window_shape
                        let mut window_shape = Vec::with_capacity(rank);
                        for i in 0..rank {
                            match params[1 + i] {
                                Scalar::I32(v) => window_shape.push(v as usize),
                                _ => return Err(HoduError::InternalError("Expected I32 for window_shape".to_string())),
                            }
                        }

                        // Extract strides
                        let mut strides = Vec::with_capacity(rank);
                        for i in 0..rank {
                            match params[1 + rank + i] {
                                Scalar::I32(v) => strides.push(v as usize),
                                _ => return Err(HoduError::InternalError("Expected I32 for strides".to_string())),
                            }
                        }

                        // Extract padding
                        let mut padding = Vec::with_capacity(rank);
                        for i in 0..rank {
                            let pad_lo = match params[1 + rank + rank + (i * 2)] {
                                Scalar::I32(v) => v as usize,
                                _ => return Err(HoduError::InternalError("Expected I32 for padding low".to_string())),
                            };
                            let pad_hi = match params[1 + rank + rank + (i * 2) + 1] {
                                Scalar::I32(v) => v as usize,
                                _ => return Err(HoduError::InternalError("Expected I32 for padding high".to_string())),
                            };
                            padding.push((pad_lo, pad_hi));
                        }

                        // Extract reduction type
                        let reduction_type = match params[1 + rank + rank + (rank * 2)] {
                            Scalar::I32(v) => v,
                            _ => return Err(HoduError::InternalError("Expected I32 for reduction type".to_string())),
                        };

                        // Get dtype from input tensor
                        if current_node.input_tensors.is_empty() {
                            return Err(HoduError::InternalError(
                                "ReduceWindow requires input tensor".to_string(),
                            ));
                        }
                        let input_tensor_id = current_node.input_tensors[0];
                        let input_dtype = _compiled_script
                            .tensor_dtypes
                            .get(&input_tensor_id)
                            .copied()
                            .unwrap_or(DType::F32); // Default to F32 if not specified
                        let dtype = Self::dtype_to_element_type(input_dtype)?;

                        let input = &input_ops[0];

                        // Create initial value and reduction computation based on reduction type
                        let (init_value, reduction_comp) = match reduction_type {
                            0 => {
                                // Max
                                let init = builder.min_value(dtype).map_err(xla_error_to_hodu_error)?;
                                let max_builder = hodu_xla::XlaBuilder::new("Max");
                                let x = max_builder
                                    .parameter(0, dtype, &[], "x")
                                    .map_err(xla_error_to_hodu_error)?;
                                let y = max_builder
                                    .parameter(1, dtype, &[], "y")
                                    .map_err(xla_error_to_hodu_error)?;
                                let comp = x
                                    .max(&y)
                                    .map_err(xla_error_to_hodu_error)?
                                    .build()
                                    .map_err(xla_error_to_hodu_error)?;
                                (init, comp)
                            },
                            1 => {
                                // Mean - use sum and divide later
                                let init = builder.zero(dtype).map_err(xla_error_to_hodu_error)?;
                                let add_builder = hodu_xla::XlaBuilder::new("Add");
                                let x = add_builder
                                    .parameter(0, dtype, &[], "x")
                                    .map_err(xla_error_to_hodu_error)?;
                                let y = add_builder
                                    .parameter(1, dtype, &[], "y")
                                    .map_err(xla_error_to_hodu_error)?;
                                let comp = x
                                    .add_(&y)
                                    .map_err(xla_error_to_hodu_error)?
                                    .build()
                                    .map_err(xla_error_to_hodu_error)?;
                                (init, comp)
                            },
                            2 => {
                                // Sum
                                let init = builder.zero(dtype).map_err(xla_error_to_hodu_error)?;
                                let add_builder = hodu_xla::XlaBuilder::new("Add");
                                let x = add_builder
                                    .parameter(0, dtype, &[], "x")
                                    .map_err(xla_error_to_hodu_error)?;
                                let y = add_builder
                                    .parameter(1, dtype, &[], "y")
                                    .map_err(xla_error_to_hodu_error)?;
                                let comp = x
                                    .add_(&y)
                                    .map_err(xla_error_to_hodu_error)?
                                    .build()
                                    .map_err(xla_error_to_hodu_error)?;
                                (init, comp)
                            },
                            3 => {
                                // Min
                                let init = builder.max_value(dtype).map_err(xla_error_to_hodu_error)?;
                                let min_builder = hodu_xla::XlaBuilder::new("Min");
                                let x = min_builder
                                    .parameter(0, dtype, &[], "x")
                                    .map_err(xla_error_to_hodu_error)?;
                                let y = min_builder
                                    .parameter(1, dtype, &[], "y")
                                    .map_err(xla_error_to_hodu_error)?;
                                let comp = x
                                    .min(&y)
                                    .map_err(xla_error_to_hodu_error)?
                                    .build()
                                    .map_err(xla_error_to_hodu_error)?;
                                (init, comp)
                            },
                            _ => {
                                return Err(HoduError::InternalError(format!(
                                    "Unknown reduction type: {}",
                                    reduction_type
                                )));
                            },
                        };

                        // Convert to i64 for XLA API
                        let window_shape_i64: Vec<i64> = window_shape.iter().map(|&v| v as i64).collect();
                        let strides_i64: Vec<i64> = strides.iter().map(|&v| v as i64).collect();
                        let padding_i64: Vec<(i64, i64)> =
                            padding.iter().map(|&(lo, hi)| (lo as i64, hi as i64)).collect();

                        // Apply reduce_window
                        let result = input
                            .reduce_window(
                                init_value,
                                reduction_comp,
                                &window_shape_i64,
                                &strides_i64,
                                &padding_i64,
                            )
                            .map_err(xla_error_to_hodu_error)?;

                        // For mean reduction, divide by window size
                        if reduction_type == 1 {
                            let window_size: usize = window_shape.iter().product();
                            let window_size_scalar = match dtype {
                                hodu_xla::ElementType::F16 => builder
                                    .constant_r0(half::f16::from_f32(window_size as f32))
                                    .map_err(xla_error_to_hodu_error)?,
                                hodu_xla::ElementType::Bf16 => builder
                                    .constant_r0(half::bf16::from_f32(window_size as f32))
                                    .map_err(xla_error_to_hodu_error)?,
                                hodu_xla::ElementType::F32 => builder
                                    .constant_r0(window_size as f32)
                                    .map_err(xla_error_to_hodu_error)?,
                                hodu_xla::ElementType::F64 => builder
                                    .constant_r0(window_size as f64)
                                    .map_err(xla_error_to_hodu_error)?,
                                _ => {
                                    return Err(HoduError::UnsupportedDType {
                                        dtype: input_dtype,
                                        op: "reduce_window with Mean reduction".to_string(),
                                    });
                                },
                            };
                            (result / window_size_scalar).map_err(xla_error_to_hodu_error)
                        } else {
                            Ok(result)
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
                            for (j, &out_dim) in output_shape.iter().enumerate().take(ndim) {
                                if input_shape[i] == out_dim {
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

            // Shape Scalars Operations
            Op::ShapeScalars(op, _, scalars) => {
                if input_ops.len() != 1 {
                    return Err(HoduError::InternalError(
                        "ShapeScalars operation requires exactly 1 input".to_string(),
                    ));
                }
                match op {
                    ShapeScalarsOp::Slice => {
                        // Extract slice parameters: [dim, start, end_or_max, step]
                        if scalars.len() < 4 {
                            return Err(HoduError::InternalError(
                                "Slice requires 4 scalar parameters".to_string(),
                            ));
                        }

                        let dim = scalars[0].to_i32() as usize;
                        let start = scalars[1].to_i64();
                        let end_value = scalars[2].to_i32();
                        let end = if end_value == i32::MAX {
                            None
                        } else {
                            Some(scalars[2].to_i64())
                        };
                        let stride = scalars[3].to_i64();

                        // Get input shape to compute actual indices
                        let input_layout = current_node
                            .input_layouts
                            .first()
                            .ok_or_else(|| HoduError::InternalError("Missing input layout for slice".to_string()))?;
                        let input_shape = input_layout.get_shape();

                        // Normalize negative indices
                        let dim_size = input_shape[dim] as i64;
                        let start_idx = if start < 0 { dim_size + start } else { start };
                        let end_idx = end
                            .map(|e| if e < 0 { dim_size + e } else { e })
                            .unwrap_or(if stride > 0 { dim_size } else { -1 });

                        // Use XLA slice operation
                        input_ops[0]
                            .slice_in_dim(start_idx, end_idx, stride, dim as i64)
                            .map_err(xla_error_to_hodu_error)
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
                                #[cfg(feature = "u8")]
                                DType::U8 => PrimitiveType::U8,
                                DType::U16 => PrimitiveType::U16,
                                #[cfg(feature = "u32")]
                                DType::U32 => PrimitiveType::U32,
                                #[cfg(feature = "u64")]
                                DType::U64 => PrimitiveType::U64,
                                DType::I8 => PrimitiveType::S8,
                                #[cfg(feature = "i16")]
                                DType::I16 => PrimitiveType::S16,
                                DType::I32 => PrimitiveType::S32,
                                #[cfg(feature = "i64")]
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
    pub(super) fn tensor_to_literal(&self, tensor: &Tensor) -> HoduResult<Literal> {
        let layout = tensor.get_layout();
        let shape = layout.get_shape();
        let dims: Vec<usize> = shape.to_vec();

        tensor.with_storage(|storage| {
            let cpu_storage = storage
                .to_cpu_storage()
                .map_err(|e| HoduError::InternalError(format!("Failed to convert to CPU storage: {:?}", e)))?;

            use crate::be_hodu::cpu::storage::CpuStorage;
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
                #[cfg(feature = "u8")]
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
                #[cfg(feature = "u32")]
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
                #[cfg(feature = "u64")]
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
                #[cfg(feature = "i16")]
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
                #[cfg(feature = "i64")]
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
    pub(super) fn literal_to_tensor(&self, literal: &Literal, dtype: DType) -> HoduResult<Tensor> {
        use crate::be_hodu::cpu::storage::CpuStorage;
        use crate::be_hodu::storage::HoduStorage;

        let array_shape = literal
            .array_shape()
            .map_err(|e| HoduError::InternalError(format!("Failed to get array shape from literal: {:?}", e)))?;

        let dims: Vec<usize> = array_shape.dims().iter().map(|&d| d as usize).collect();

        let cpu_storage = match dtype {
            DType::BOOL => {
                // XLA Pred type should be read as bool directly
                let bool_data = literal.to_vec::<bool>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract bool data from literal: {:?}", e))
                })?;
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
            #[cfg(feature = "u8")]
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
            #[cfg(feature = "u32")]
            DType::U32 => {
                let data = literal.to_vec::<u32>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract u32 data from literal: {:?}", e))
                })?;
                CpuStorage::U32(data)
            },
            #[cfg(feature = "u64")]
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
            #[cfg(feature = "i16")]
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
            #[cfg(feature = "i64")]
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

    pub(super) fn xla_element_type_to_dtype(&self, element_type: hodu_xla::ElementType) -> Option<DType> {
        use hodu_xla::ElementType;
        match element_type {
            ElementType::F32 => Some(DType::F32),
            ElementType::F64 => Some(DType::F64),
            ElementType::F16 => Some(DType::F16),
            ElementType::Bf16 => Some(DType::BF16),
            #[cfg(feature = "u8")]
            ElementType::U8 => Some(DType::U8),
            ElementType::U16 => Some(DType::U16),
            #[cfg(feature = "u32")]
            ElementType::U32 => Some(DType::U32),
            #[cfg(feature = "u64")]
            ElementType::U64 => Some(DType::U64),
            ElementType::S8 => Some(DType::I8),
            #[cfg(feature = "i16")]
            ElementType::S16 => Some(DType::I16),
            ElementType::S32 => Some(DType::I32),
            #[cfg(feature = "i64")]
            ElementType::S64 => Some(DType::I64),
            ElementType::Pred => Some(DType::BOOL),
            _ => None,
        }
    }
}
