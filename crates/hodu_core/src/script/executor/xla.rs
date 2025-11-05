use super::{instance::ExecutorT, types::*};
use crate::{
    be::storage::BackendStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    script::{
        builder::ir::Attribute,
        compiler::{CompiledInstruction, CompiledModule},
    },
    tensor::from_storage,
    types::{Compiler, Device},
};
use std::collections::HashMap;

/// XLA executor (feature-gated)
///
/// Note: This executor requires the `xla` feature and `hodu_xla` crate to be available.
/// For now, it provides a basic structure but requires full XLA integration.
#[derive(Debug)]
pub struct XlaExecutor {
    device: Device,
}

impl XlaExecutor {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Build and execute XLA computation from the compiled module
    /// This is called at runtime for each execution
    #[cfg(feature = "xla")]
    fn build_and_execute_xla(
        &self,
        compiled: &CompiledModule,
        inputs: &ExecutionInputs<'_>,
    ) -> HoduResult<ExecutionOutputs> {
        use hodu_xla::{PjRtClient, XlaBuilder};

        // Create XLA client
        let client = match self.device {
            Device::CPU => PjRtClient::cpu()
                .map_err(|e| HoduError::InternalError(format!("Failed to create XLA CPU client: {:?}", e)))?,
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => PjRtClient::gpu(0.95, true)
                .map_err(|e| HoduError::InternalError(format!("Failed to create XLA GPU client: {:?}", e)))?,
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => {
                return Err(HoduError::InternalError(format!(
                    "Device {:?} not supported for XLA",
                    self.device
                )))
            },
        };

        // Build XLA computation
        let builder = XlaBuilder::new("computation");
        let mut xla_ops = HashMap::new();

        // Create parameters for inputs
        let mut input_names: Vec<_> = compiled.input_mapping.keys().cloned().collect();
        input_names.sort();

        for (i, input_name) in input_names.iter().enumerate() {
            if let Some(&value_id) = compiled.input_mapping.get(input_name) {
                let _tensor = inputs
                    .get(input_name.as_str())
                    .ok_or_else(|| HoduError::InternalError(format!("Missing input: {}", input_name)))?;

                // Get layout and dtype
                let layout = compiled
                    .value_layouts
                    .get(&value_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Missing layout for input: {}", input_name)))?;
                let dtype = compiled
                    .value_dtypes
                    .get(&value_id)
                    .copied()
                    .unwrap_or(crate::types::DType::F32);

                // Convert dtype to ElementType
                let element_type = dtype_to_element_type(dtype)?;
                let dims: Vec<i64> = layout.shape().dims().iter().map(|&d| d as i64).collect();

                let param = builder
                    .parameter(i as i64, element_type, &dims, &format!("input_{}", i))
                    .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;

                xla_ops.insert(value_id, param);
            }
        }

        // Process constants
        for (tensor_id, constant_data) in &compiled.constant_data {
            // Find value_id for this tensor_id
            if let Some((&value_id, _)) = compiled.value_to_tensor.iter().find(|(_, &tid)| tid == *tensor_id) {
                // Create constant XLA op
                let constant_op = create_constant_op(&builder, constant_data)?;
                xla_ops.insert(value_id, constant_op);
            }
        }

        // Execute instructions
        for instr in &compiled.execution_plan {
            // Skip constant loads
            if let Some(crate::script::builder::ir::Attribute::Bool(true)) = instr.attributes.get("is_constant") {
                continue;
            }

            // Get input ops
            let input_ops: Vec<_> = instr
                .inputs
                .iter()
                .filter_map(|vid| xla_ops.get(vid).cloned())
                .collect();

            if input_ops.len() != instr.inputs.len() {
                return Err(HoduError::InternalError(format!(
                    "Missing input ops for instruction, expected {}, got {}",
                    instr.inputs.len(),
                    input_ops.len()
                )));
            }

            // Execute operation
            let result_op = execute_xla_op(&builder, &instr.op, &input_ops, &instr.attributes, compiled, instr)?;
            xla_ops.insert(instr.result, result_op);
        }

        // Build output computation
        let mut output_names: Vec<_> = compiled.output_mapping.keys().cloned().collect();
        output_names.sort();

        let computation = if output_names.len() == 1 {
            // Single output
            let output_name = &output_names[0];
            let value_id = compiled
                .output_mapping
                .get(output_name)
                .ok_or_else(|| HoduError::InternalError(format!("Missing output mapping: {}", output_name)))?;
            let output_op = xla_ops
                .get(value_id)
                .ok_or_else(|| HoduError::InternalError(format!("Missing output op for: {}", output_name)))?;
            output_op
                .build()
                .map_err(|e| HoduError::InternalError(format!("Failed to build XLA computation: {:?}", e)))?
        } else {
            // Multiple outputs - create tuple
            let output_ops: Vec<_> = output_names
                .iter()
                .filter_map(|name| {
                    compiled
                        .output_mapping
                        .get(name)
                        .and_then(|vid| xla_ops.get(vid).cloned())
                })
                .collect();

            if output_ops.len() != output_names.len() {
                return Err(HoduError::InternalError("Missing some output ops".to_string()));
            }

            let tuple_op = builder
                .tuple(&output_ops)
                .map_err(|e| HoduError::InternalError(format!("Failed to create tuple: {:?}", e)))?;
            tuple_op
                .build()
                .map_err(|e| HoduError::InternalError(format!("Failed to build XLA computation: {:?}", e)))?
        };

        // Compile and execute
        let executable = client
            .compile(&computation)
            .map_err(|e| HoduError::InternalError(format!("Failed to compile XLA computation: {:?}", e)))?;

        // Convert inputs to literals
        let input_literals: Vec<_> = input_names
            .iter()
            .filter_map(|name| {
                inputs
                    .get(name.as_str())
                    .and_then(|tensor| tensor_to_literal(tensor).ok())
            })
            .collect();

        // Execute
        let result_buffers = executable
            .execute::<hodu_xla::Literal>(&input_literals)
            .map_err(|e| HoduError::InternalError(format!("Failed to execute XLA computation: {:?}", e)))?;

        // Convert results back to tensors
        let mut outputs = HashMap::new();

        if output_names.len() == 1 {
            let result_literal = result_buffers[0][0]
                .to_literal_sync()
                .map_err(|e| HoduError::InternalError(format!("Failed to get result literal: {:?}", e)))?;

            let output_name = &output_names[0];
            let value_id = compiled
                .output_mapping
                .get(output_name)
                .ok_or_else(|| HoduError::InternalError(format!("Missing output mapping: {}", output_name)))?;
            let dtype = compiled
                .value_dtypes
                .get(value_id)
                .copied()
                .unwrap_or(crate::types::DType::F32);

            let tensor = literal_to_tensor(&result_literal, dtype)?;
            outputs.insert(output_name.clone(), tensor);
        } else {
            for (i, output_name) in output_names.iter().enumerate() {
                let element_literal = result_buffers[0][i]
                    .to_literal_sync()
                    .map_err(|e| HoduError::InternalError(format!("Failed to get tuple element: {:?}", e)))?;

                let value_id = compiled
                    .output_mapping
                    .get(output_name)
                    .ok_or_else(|| HoduError::InternalError(format!("Missing output mapping: {}", output_name)))?;
                let dtype = compiled
                    .value_dtypes
                    .get(value_id)
                    .copied()
                    .unwrap_or(crate::types::DType::F32);

                let tensor = literal_to_tensor(&element_literal, dtype)?;
                outputs.insert(output_name.clone(), tensor);
            }
        }

        Ok(outputs)
    }
}

impl ExecutorT for XlaExecutor {
    fn compiler_type(&self) -> Compiler {
        Compiler::XLA
    }

    fn device(&self) -> Device {
        self.device
    }

    fn execute(&self, compiled: &CompiledModule, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        // Validate inputs
        for name in compiled.input_mapping.keys() {
            if !inputs.contains_key(name.as_str()) {
                return Err(HoduError::InternalError(format!("Missing required input: {}", name)));
            }
        }

        #[cfg(feature = "xla")]
        {
            self.build_and_execute_xla(compiled, &inputs)
        }

        #[cfg(not(feature = "xla"))]
        {
            Err(HoduError::NotImplemented(
                "XLA feature not enabled - rebuild with --features xla".to_string(),
            ))
        }
    }
}

// Helper functions for XLA operations

#[cfg(feature = "xla")]
fn dtype_to_element_type(dtype: crate::types::DType) -> HoduResult<hodu_xla::ElementType> {
    use crate::types::DType;
    use hodu_xla::ElementType;

    match dtype {
        DType::BOOL => Ok(ElementType::Pred),
        DType::F8E4M3 => Err(HoduError::InternalError("F8E4M3 not supported by XLA".to_string())),
        #[cfg(feature = "f8e5m2")]
        DType::F8E5M2 => Err(HoduError::InternalError("F8E5M2 not supported by XLA".to_string())),
        DType::BF16 => Ok(ElementType::Bf16),
        DType::F16 => Ok(ElementType::F16),
        DType::F32 => Ok(ElementType::F32),
        #[cfg(feature = "f64")]
        DType::F64 => Ok(ElementType::F64),
        DType::U8 => Ok(ElementType::U8),
        #[cfg(feature = "u16")]
        DType::U16 => Ok(ElementType::U16),
        DType::U32 => Ok(ElementType::U32),
        #[cfg(feature = "u64")]
        DType::U64 => Ok(ElementType::U64),
        DType::I8 => Ok(ElementType::S8),
        #[cfg(feature = "i16")]
        DType::I16 => Ok(ElementType::S16),
        DType::I32 => Ok(ElementType::S32),
        #[cfg(feature = "i64")]
        DType::I64 => Ok(ElementType::S64),
    }
}

#[cfg(feature = "xla")]
fn create_constant_op(
    builder: &hodu_xla::XlaBuilder,
    constant_data: &crate::script::builder::ir::ConstantData,
) -> HoduResult<hodu_xla::XlaOp> {
    use crate::be_cpu::storage::CpuStorage;

    // Parse the data bytes into appropriate CPU storage
    let cpu_storage = CpuStorage::from_bytes(&constant_data.data, constant_data.dtype)
        .map_err(|e| HoduError::InternalError(format!("Failed to parse constant data: {:?}", e)))?;

    // Convert to XLA literal
    let literal = cpu_storage_to_literal(&cpu_storage)?;

    // Reshape the literal to match the constant's shape
    let dims: Vec<i64> = constant_data.shape.dims().iter().map(|&d| d as i64).collect();
    let reshaped = literal
        .reshape(&dims)
        .map_err(|e| HoduError::InternalError(format!("Failed to reshape constant: {:?}", e)))?;

    // Convert literal to constant op
    builder
        .constant_literal(&reshaped)
        .map_err(|e| HoduError::InternalError(format!("Failed to create constant op: {:?}", e)))
}

#[cfg(feature = "xla")]
fn execute_xla_op(
    builder: &hodu_xla::XlaBuilder,
    op: &Op,
    inputs: &[hodu_xla::XlaOp],
    attributes: &HashMap<String, Attribute>,
    compiled: &CompiledModule,
    instr: &CompiledInstruction,
) -> HoduResult<hodu_xla::XlaOp> {
    use crate::ops::*;

    match op {
        // Binary operations
        Op::Binary(BinaryOp::Add) => binary_op_check(inputs, 2, |i| i[0].add_(&i[1]), "add"),
        Op::Binary(BinaryOp::Sub) => binary_op_check(inputs, 2, |i| i[0].sub_(&i[1]), "sub"),
        Op::Binary(BinaryOp::Mul) => binary_op_check(inputs, 2, |i| i[0].mul_(&i[1]), "mul"),
        Op::Binary(BinaryOp::Div) => binary_op_check(inputs, 2, |i| i[0].div_(&i[1]), "div"),
        Op::Binary(BinaryOp::Pow) => binary_op_check(inputs, 2, |i| i[0].pow(&i[1]), "pow"),
        Op::Binary(BinaryOp::Maximum) => binary_op_check(inputs, 2, |i| i[0].max(&i[1]), "max"),
        Op::Binary(BinaryOp::Minimum) => binary_op_check(inputs, 2, |i| i[0].min(&i[1]), "min"),

        // BinaryLogical operations
        Op::BinaryLogical(BinaryLogicalOp::LogicalAnd) => binary_op_check(inputs, 2, |i| i[0].and(&i[1]), "and"),
        Op::BinaryLogical(BinaryLogicalOp::LogicalOr) => binary_op_check(inputs, 2, |i| i[0].or(&i[1]), "or"),
        Op::BinaryLogical(BinaryLogicalOp::LogicalXor) => binary_op_check(inputs, 2, |i| i[0].xor(&i[1]), "xor"),

        // Cmp operations
        Op::Cmp(CmpOp::Eq) => binary_op_check(inputs, 2, |i| i[0].eq(&i[1]), "eq"),
        Op::Cmp(CmpOp::Ne) => binary_op_check(inputs, 2, |i| i[0].ne(&i[1]), "ne"),
        Op::Cmp(CmpOp::Lt) => binary_op_check(inputs, 2, |i| i[0].lt(&i[1]), "lt"),
        Op::Cmp(CmpOp::Le) => binary_op_check(inputs, 2, |i| i[0].le(&i[1]), "le"),
        Op::Cmp(CmpOp::Gt) => binary_op_check(inputs, 2, |i| i[0].gt(&i[1]), "gt"),
        Op::Cmp(CmpOp::Ge) => binary_op_check(inputs, 2, |i| i[0].ge(&i[1]), "ge"),

        // CmpScalar operations
        Op::CmpScalar(CmpScalarOp::EqScalar) => scalar_op(builder, inputs, attributes, |i, s| i[0].eq(&s), "eq_scalar"),
        Op::CmpScalar(CmpScalarOp::NeScalar) => scalar_op(builder, inputs, attributes, |i, s| i[0].ne(&s), "ne_scalar"),
        Op::CmpScalar(CmpScalarOp::LtScalar) => scalar_op(builder, inputs, attributes, |i, s| i[0].lt(&s), "lt_scalar"),
        Op::CmpScalar(CmpScalarOp::LeScalar) => scalar_op(builder, inputs, attributes, |i, s| i[0].le(&s), "le_scalar"),
        Op::CmpScalar(CmpScalarOp::GtScalar) => scalar_op(builder, inputs, attributes, |i, s| i[0].gt(&s), "gt_scalar"),
        Op::CmpScalar(CmpScalarOp::GeScalar) => scalar_op(builder, inputs, attributes, |i, s| i[0].ge(&s), "ge_scalar"),

        // Unary operations
        Op::Unary(UnaryOp::Neg) => unary_op_check(inputs, |i| i[0].neg(), "neg"),
        Op::Unary(UnaryOp::Abs) => unary_op_check(inputs, |i| i[0].abs(), "abs"),
        Op::Unary(UnaryOp::Sign) => unary_op_check(inputs, |i| i[0].sign(), "sign"),
        Op::Unary(UnaryOp::Square) => unary_op_check(inputs, |i| i[0].mul_(&i[0]), "square"),
        Op::Unary(UnaryOp::Sqrt) => unary_op_check(inputs, |i| i[0].sqrt(), "sqrt"),
        Op::Unary(UnaryOp::Recip) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Recip requires 1 input".to_string()));
            }
            let one = builder
                .constant_r0(1.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create one: {:?}", e)))?;
            one.div_(&inputs[0])
                .map_err(|e| HoduError::InternalError(format!("XLA recip failed: {:?}", e)))
        },
        Op::Unary(UnaryOp::Relu) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Relu requires 1 input".to_string()));
            }
            let zero = builder
                .constant_r0(0.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create zero: {:?}", e)))?;
            inputs[0]
                .max(&zero)
                .map_err(|e| HoduError::InternalError(format!("XLA relu failed: {:?}", e)))
        },
        Op::Unary(UnaryOp::Sigmoid) => unary_op_check(inputs, |i| i[0].logistic(), "sigmoid"),
        Op::Unary(UnaryOp::Tanh) => unary_op_check(inputs, |i| i[0].tanh(), "tanh"),
        Op::Unary(UnaryOp::Sin) => unary_op_check(inputs, |i| i[0].sin(), "sin"),
        Op::Unary(UnaryOp::Cos) => unary_op_check(inputs, |i| i[0].cos(), "cos"),
        Op::Unary(UnaryOp::Tan) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Tan requires 1 input".to_string()));
            }
            let sin_val = inputs[0]
                .sin()
                .map_err(|e| HoduError::InternalError(format!("XLA sin failed: {:?}", e)))?;
            let cos_val = inputs[0]
                .cos()
                .map_err(|e| HoduError::InternalError(format!("XLA cos failed: {:?}", e)))?;
            sin_val
                .div_(&cos_val)
                .map_err(|e| HoduError::InternalError(format!("XLA tan failed: {:?}", e)))
        },
        Op::Unary(UnaryOp::Exp) => unary_op_check(inputs, |i| i[0].exp(), "exp"),
        Op::Unary(UnaryOp::Exp2) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Exp2 requires 1 input".to_string()));
            }
            let ln2 = builder
                .constant_r0(2.0f32.ln())
                .map_err(|e| HoduError::InternalError(format!("Failed to create ln2: {:?}", e)))?;
            let scaled = inputs[0]
                .mul_(&ln2)
                .map_err(|e| HoduError::InternalError(format!("XLA mul failed: {:?}", e)))?;
            scaled
                .exp()
                .map_err(|e| HoduError::InternalError(format!("XLA exp2 failed: {:?}", e)))
        },
        Op::Unary(UnaryOp::Exp10) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Exp10 requires 1 input".to_string()));
            }
            let ln10 = builder
                .constant_r0(10.0f32.ln())
                .map_err(|e| HoduError::InternalError(format!("Failed to create ln10: {:?}", e)))?;
            let scaled = inputs[0]
                .mul_(&ln10)
                .map_err(|e| HoduError::InternalError(format!("XLA mul failed: {:?}", e)))?;
            scaled
                .exp()
                .map_err(|e| HoduError::InternalError(format!("XLA exp10 failed: {:?}", e)))
        },
        Op::Unary(UnaryOp::Ln) => unary_op_check(inputs, |i| i[0].log(), "ln"),
        Op::Unary(UnaryOp::Log2) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Log2 requires 1 input".to_string()));
            }
            let ln_val = inputs[0]
                .log()
                .map_err(|e| HoduError::InternalError(format!("XLA log failed: {:?}", e)))?;
            let ln2 = builder
                .constant_r0(2.0f32.ln())
                .map_err(|e| HoduError::InternalError(format!("Failed to create ln2: {:?}", e)))?;
            ln_val
                .div_(&ln2)
                .map_err(|e| HoduError::InternalError(format!("XLA log2 failed: {:?}", e)))
        },
        Op::Unary(UnaryOp::Log10) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Log10 requires 1 input".to_string()));
            }
            let ln_val = inputs[0]
                .log()
                .map_err(|e| HoduError::InternalError(format!("XLA log failed: {:?}", e)))?;
            let ln10 = builder
                .constant_r0(10.0f32.ln())
                .map_err(|e| HoduError::InternalError(format!("Failed to create ln10: {:?}", e)))?;
            ln_val
                .div_(&ln10)
                .map_err(|e| HoduError::InternalError(format!("XLA log10 failed: {:?}", e)))
        },
        Op::Unary(UnaryOp::Gelu) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Gelu requires 1 input".to_string()));
            }
            // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            let half = builder
                .constant_r0(0.5f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create constant: {:?}", e)))?;
            let one = builder
                .constant_r0(1.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create constant: {:?}", e)))?;
            let sqrt_2_pi = builder
                .constant_r0((2.0f32 / std::f32::consts::PI).sqrt())
                .map_err(|e| HoduError::InternalError(format!("Failed to create constant: {:?}", e)))?;
            let c = builder
                .constant_r0(0.044715f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create constant: {:?}", e)))?;

            let x3 = inputs[0]
                .mul_(&inputs[0])
                .and_then(|x2| x2.mul_(&inputs[0]))
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            let cx3 = c
                .mul_(&x3)
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            let inner = inputs[0]
                .add_(&cx3)
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            let scaled = sqrt_2_pi
                .mul_(&inner)
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            let tanh_val = scaled
                .tanh()
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            let one_plus = one
                .add_(&tanh_val)
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            let x_mul = inputs[0]
                .mul_(&one_plus)
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            half.mul_(&x_mul)
                .map_err(|e| HoduError::InternalError(format!("XLA gelu failed: {:?}", e)))
        },
        Op::Unary(UnaryOp::Softplus) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Softplus requires 1 input".to_string()));
            }
            // softplus(x) = ln(1 + exp(x))
            let one = builder
                .constant_r0(1.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create one: {:?}", e)))?;
            let exp_x = inputs[0]
                .exp()
                .map_err(|e| HoduError::InternalError(format!("XLA exp failed: {:?}", e)))?;
            let one_plus_exp = one
                .add_(&exp_x)
                .map_err(|e| HoduError::InternalError(format!("XLA add failed: {:?}", e)))?;
            one_plus_exp
                .log()
                .map_err(|e| HoduError::InternalError(format!("XLA softplus failed: {:?}", e)))
        },
        Op::Unary(UnaryOp::Silu) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Silu requires 1 input".to_string()));
            }
            // silu(x) = x * sigmoid(x)
            let sigmoid_x = inputs[0]
                .logistic()
                .map_err(|e| HoduError::InternalError(format!("XLA logistic failed: {:?}", e)))?;
            inputs[0]
                .mul_(&sigmoid_x)
                .map_err(|e| HoduError::InternalError(format!("XLA silu failed: {:?}", e)))
        },
        Op::Unary(UnaryOp::Mish) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Mish requires 1 input".to_string()));
            }
            // mish(x) = x * tanh(softplus(x))
            let one = builder
                .constant_r0(1.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create one: {:?}", e)))?;
            let exp_x = inputs[0]
                .exp()
                .map_err(|e| HoduError::InternalError(format!("XLA exp failed: {:?}", e)))?;
            let one_plus_exp = one
                .add_(&exp_x)
                .map_err(|e| HoduError::InternalError(format!("XLA add failed: {:?}", e)))?;
            let ln_val = one_plus_exp
                .log()
                .map_err(|e| HoduError::InternalError(format!("XLA log failed: {:?}", e)))?;
            let tanh_val = ln_val
                .tanh()
                .map_err(|e| HoduError::InternalError(format!("XLA tanh failed: {:?}", e)))?;
            inputs[0]
                .mul_(&tanh_val)
                .map_err(|e| HoduError::InternalError(format!("XLA mish failed: {:?}", e)))
        },

        // UnaryLogical operations
        Op::UnaryLogical(UnaryLogicalOp::LogicalNot) => unary_op_check(inputs, |i| i[0].not(), "logical_not"),

        // UnaryScalar operations
        Op::UnaryScalar(UnaryScalarOp::AddScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].add_(&s), "add_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::SubScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].sub_(&s), "sub_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::MulScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].mul_(&s), "mul_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::DivScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].div_(&s), "div_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::PowScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].pow(&s), "pow_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::MaximumScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].max(&s), "maximum_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::MinimumScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].min(&s), "minimum_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::LeakyRelu) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("LeakyRelu requires 1 input".to_string()));
            }
            let alpha = get_scalar_from_attributes(attributes)?;
            let zero = builder
                .constant_r0(0.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create zero: {:?}", e)))?;
            let alpha_op = builder
                .constant_r0(alpha)
                .map_err(|e| HoduError::InternalError(format!("Failed to create alpha: {:?}", e)))?;
            let cond = inputs[0]
                .gt(&zero)
                .map_err(|e| HoduError::InternalError(format!("XLA gt failed: {:?}", e)))?;
            let neg_part = inputs[0]
                .mul_(&alpha_op)
                .map_err(|e| HoduError::InternalError(format!("XLA mul failed: {:?}", e)))?;
            cond.select(&inputs[0], &neg_part)
                .map_err(|e| HoduError::InternalError(format!("XLA leaky_relu failed: {:?}", e)))
        },
        Op::UnaryScalar(UnaryScalarOp::Elu) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Elu requires 1 input".to_string()));
            }
            let alpha = get_scalar_from_attributes(attributes)?;
            let zero = builder
                .constant_r0(0.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create zero: {:?}", e)))?;
            let alpha_op = builder
                .constant_r0(alpha)
                .map_err(|e| HoduError::InternalError(format!("Failed to create alpha: {:?}", e)))?;
            let one = builder
                .constant_r0(1.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create one: {:?}", e)))?;
            let cond = inputs[0]
                .gt(&zero)
                .map_err(|e| HoduError::InternalError(format!("XLA gt failed: {:?}", e)))?;
            let exp_x = inputs[0]
                .exp()
                .map_err(|e| HoduError::InternalError(format!("XLA exp failed: {:?}", e)))?;
            let exp_minus_one = exp_x
                .sub_(&one)
                .map_err(|e| HoduError::InternalError(format!("XLA sub failed: {:?}", e)))?;
            let neg_part = alpha_op
                .mul_(&exp_minus_one)
                .map_err(|e| HoduError::InternalError(format!("XLA mul failed: {:?}", e)))?;
            cond.select(&inputs[0], &neg_part)
                .map_err(|e| HoduError::InternalError(format!("XLA elu failed: {:?}", e)))
        },
        Op::UnaryScalar(UnaryScalarOp::Prelu) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Prelu requires 1 input".to_string()));
            }
            let alpha = get_scalar_from_attributes(attributes)?;
            let zero = builder
                .constant_r0(0.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create zero: {:?}", e)))?;
            let alpha_op = builder
                .constant_r0(alpha)
                .map_err(|e| HoduError::InternalError(format!("Failed to create alpha: {:?}", e)))?;
            let cond = inputs[0]
                .gt(&zero)
                .map_err(|e| HoduError::InternalError(format!("XLA gt failed: {:?}", e)))?;
            let neg_part = inputs[0]
                .mul_(&alpha_op)
                .map_err(|e| HoduError::InternalError(format!("XLA mul failed: {:?}", e)))?;
            cond.select(&inputs[0], &neg_part)
                .map_err(|e| HoduError::InternalError(format!("XLA prelu failed: {:?}", e)))
        },

        // Matrix operations
        Op::Matrix(MatrixOp::Matmul) => binary_op_check(inputs, 2, |i| i[0].dot(&i[1]), "matmul"),
        Op::Matrix(MatrixOp::Dot) => binary_op_check(inputs, 2, |i| i[0].dot(&i[1]), "dot"),

        // Reduce operations
        Op::Reduce(reduce_op) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Reduce requires input".to_string()));
            }

            // Extract keep_dim from attributes
            let keep_dim = attributes
                .get("keep_dim")
                .and_then(|attr| if let Attribute::Bool(b) = attr { Some(*b) } else { None })
                .unwrap_or(false);

            // Extract dimensions from attributes
            let dims: Vec<i64> = attributes
                .get("dims")
                .and_then(|attr| {
                    if let Attribute::IntArray(arr) = attr {
                        Some(arr.iter().map(|&d| d as i64).collect())
                    } else {
                        None
                    }
                })
                .unwrap_or_default();

            match reduce_op {
                ReduceOp::Sum => if dims.is_empty() {
                    let input_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("XLA error: {:?}", e)))?;
                    let all_dims: Vec<i64> = match input_shape {
                        hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                        _ => return Err(HoduError::InternalError("Expected array shape for reduce".to_string())),
                    };
                    inputs[0].reduce_sum(&all_dims, keep_dim)
                } else {
                    inputs[0].reduce_sum(&dims, keep_dim)
                }
                .map_err(|e| HoduError::InternalError(format!("XLA reduce_sum failed: {:?}", e))),
                ReduceOp::Mean => if dims.is_empty() {
                    let input_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("XLA error: {:?}", e)))?;
                    let all_dims: Vec<i64> = match input_shape {
                        hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                        _ => return Err(HoduError::InternalError("Expected array shape for reduce".to_string())),
                    };
                    inputs[0].reduce_mean(&all_dims, keep_dim)
                } else {
                    inputs[0].reduce_mean(&dims, keep_dim)
                }
                .map_err(|e| HoduError::InternalError(format!("XLA reduce_mean failed: {:?}", e))),
                ReduceOp::Max => if dims.is_empty() {
                    let input_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("XLA error: {:?}", e)))?;
                    let all_dims: Vec<i64> = match input_shape {
                        hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                        _ => return Err(HoduError::InternalError("Expected array shape for reduce".to_string())),
                    };
                    inputs[0].reduce_max(&all_dims, keep_dim)
                } else {
                    inputs[0].reduce_max(&dims, keep_dim)
                }
                .map_err(|e| HoduError::InternalError(format!("XLA reduce_max failed: {:?}", e))),
                ReduceOp::Min => if dims.is_empty() {
                    let input_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("XLA error: {:?}", e)))?;
                    let all_dims: Vec<i64> = match input_shape {
                        hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                        _ => return Err(HoduError::InternalError("Expected array shape for reduce".to_string())),
                    };
                    inputs[0].reduce_min(&all_dims, keep_dim)
                } else {
                    inputs[0].reduce_min(&dims, keep_dim)
                }
                .map_err(|e| HoduError::InternalError(format!("XLA reduce_min failed: {:?}", e))),
                _ => Err(HoduError::InternalError(format!(
                    "Reduce operation {:?} not yet implemented",
                    reduce_op
                ))),
            }
        },

        // Concat operations
        Op::Concat(_concat_op) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Concat requires at least 1 input".to_string()));
            }

            // Extract dimension from attributes
            let dim = attributes
                .get("dim")
                .and_then(|attr| {
                    if let Attribute::Int(i) = attr {
                        Some(*i as i64)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Concat requires dim attribute".to_string()))?;

            inputs[0]
                .concat_in_dim(&inputs[1..], dim)
                .map_err(|e| HoduError::InternalError(format!("XLA concat failed: {:?}", e)))
        },

        // Split operations
        Op::Split(_split_op) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Split requires exactly 1 input".to_string()));
            }

            // Extract dimension from attributes
            let dim = attributes
                .get("dim")
                .and_then(|attr| {
                    if let Attribute::Int(i) = attr {
                        Some(*i as i64)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Split requires dim attribute".to_string()))?;

            // Extract sizes from attributes
            let sizes: Vec<i64> = attributes
                .get("sizes")
                .and_then(|attr| {
                    if let Attribute::IntArray(arr) = attr {
                        Some(arr.iter().map(|&d| d as i64).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Split requires sizes attribute".to_string()))?;

            // Extract output_index from attributes
            let output_index = attributes
                .get("output_index")
                .and_then(|attr| {
                    if let Attribute::Int(i) = attr {
                        Some(*i as usize)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Split requires output_index attribute".to_string()))?;

            // Calculate split indices (cumulative sum)
            let mut split_indices = Vec::with_capacity(sizes.len() - 1);
            let mut cumsum = 0i64;
            for &size in &sizes[..sizes.len() - 1] {
                cumsum += size;
                split_indices.push(cumsum);
            }

            // Calculate start and limit for this output slice
            let start_offset = if output_index == 0 {
                0
            } else {
                split_indices[output_index - 1]
            };

            let size = sizes[output_index];

            // Use slice_in_dim operation
            inputs[0]
                .slice_in_dim(start_offset, start_offset + size, 1, dim)
                .map_err(|e| HoduError::InternalError(format!("XLA slice_in_dim failed: {:?}", e)))
        },

        // Indexing operations
        Op::Indexing(indexing_op) => {
            match indexing_op {
                IndexingOp::IndexSelect => {
                    if inputs.len() != 2 {
                        return Err(HoduError::InternalError("IndexSelect requires 2 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("IndexSelect requires dim attribute".to_string()))?;
                    inputs[0]
                        .take(&inputs[1], dim)
                        .map_err(|e| HoduError::InternalError(format!("XLA take failed: {:?}", e)))
                },
                IndexingOp::IndexPut => {
                    if inputs.len() != 3 {
                        return Err(HoduError::InternalError("IndexPut requires 3 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("IndexPut requires dim attribute".to_string()))?;

                    // Create update computation
                    let update_builder = hodu_xla::XlaBuilder::new("index_put_update");
                    let shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let element_type = match shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.element_type(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let _old = update_builder
                        .parameter(0, element_type, &[], "old")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let new = update_builder
                        .parameter(1, element_type, &[], "new")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let update_computation = new
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build computation: {:?}", e)))?;

                    let indices_shape = inputs[1]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let indices_rank = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let base_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let base_rank = match &base_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                    let inserted_window_dims = vec![dim];
                    let scatter_dims_to_operand_dims = vec![dim];
                    let index_vector_dim = Some(indices_rank as i64);

                    let indices_dims_vec = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let mut indices_dims_plus_1 = indices_dims_vec;
                    indices_dims_plus_1.push(1);
                    let indices_reshaped = inputs[1]
                        .reshape(&indices_dims_plus_1)
                        .map_err(|e| HoduError::InternalError(format!("Failed to reshape: {:?}", e)))?;

                    inputs[0]
                        .scatter(
                            &indices_reshaped,
                            &inputs[2],
                            update_computation,
                            &update_window_dims,
                            &inserted_window_dims,
                            &scatter_dims_to_operand_dims,
                            index_vector_dim,
                            false,
                            false,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA scatter failed: {:?}", e)))
                },
                IndexingOp::Gather => {
                    if inputs.len() != 2 {
                        return Err(HoduError::InternalError("Gather requires 2 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("Gather requires dim attribute".to_string()))?;
                    inputs[0]
                        .take(&inputs[1], dim)
                        .map_err(|e| HoduError::InternalError(format!("XLA take failed: {:?}", e)))
                },
                IndexingOp::Scatter => {
                    if inputs.len() != 3 {
                        return Err(HoduError::InternalError("Scatter requires 3 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("Scatter requires dim attribute".to_string()))?;

                    // Create update computation
                    let update_builder = hodu_xla::XlaBuilder::new("scatter_update");
                    let shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let element_type = match shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.element_type(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let _old = update_builder
                        .parameter(0, element_type, &[], "old")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let new = update_builder
                        .parameter(1, element_type, &[], "new")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let update_computation = new
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build computation: {:?}", e)))?;

                    let indices_shape = inputs[1]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let indices_rank = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let base_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let base_rank = match &base_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                    let inserted_window_dims = vec![dim];
                    let scatter_dims_to_operand_dims = vec![dim];
                    let index_vector_dim = Some(indices_rank as i64);

                    let indices_dims_vec = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let mut indices_dims_plus_1 = indices_dims_vec;
                    indices_dims_plus_1.push(1);
                    let indices_reshaped = inputs[1]
                        .reshape(&indices_dims_plus_1)
                        .map_err(|e| HoduError::InternalError(format!("Failed to reshape: {:?}", e)))?;

                    inputs[0]
                        .scatter(
                            &indices_reshaped,
                            &inputs[2],
                            update_computation,
                            &update_window_dims,
                            &inserted_window_dims,
                            &scatter_dims_to_operand_dims,
                            index_vector_dim,
                            false,
                            false,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA scatter failed: {:?}", e)))
                },
                IndexingOp::ScatterAdd => {
                    if inputs.len() != 3 {
                        return Err(HoduError::InternalError("ScatterAdd requires 3 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("ScatterAdd requires dim attribute".to_string()))?;

                    // Create add computation
                    let add_builder = hodu_xla::XlaBuilder::new("scatter_add");
                    let shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let element_type = match shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.element_type(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let old = add_builder
                        .parameter(0, element_type, &[], "old")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let new = add_builder
                        .parameter(1, element_type, &[], "new")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let sum = old
                        .add_(&new)
                        .map_err(|e| HoduError::InternalError(format!("Failed to add: {:?}", e)))?;
                    let add_computation = sum
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build computation: {:?}", e)))?;

                    let indices_shape = inputs[1]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let indices_rank = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let base_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let base_rank = match &base_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                    let inserted_window_dims = vec![dim];
                    let scatter_dims_to_operand_dims = vec![dim];
                    let index_vector_dim = Some(indices_rank as i64);

                    let indices_dims_vec = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let mut indices_dims_plus_1 = indices_dims_vec;
                    indices_dims_plus_1.push(1);
                    let indices_reshaped = inputs[1]
                        .reshape(&indices_dims_plus_1)
                        .map_err(|e| HoduError::InternalError(format!("Failed to reshape: {:?}", e)))?;

                    inputs[0]
                        .scatter(
                            &indices_reshaped,
                            &inputs[2],
                            add_computation,
                            &update_window_dims,
                            &inserted_window_dims,
                            &scatter_dims_to_operand_dims,
                            index_vector_dim,
                            false,
                            false,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA scatter failed: {:?}", e)))
                },
                IndexingOp::ScatterMax => {
                    if inputs.len() != 3 {
                        return Err(HoduError::InternalError("ScatterMax requires 3 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("ScatterMax requires dim attribute".to_string()))?;

                    // Create max computation
                    let max_builder = hodu_xla::XlaBuilder::new("scatter_max");
                    let shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let element_type = match shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.element_type(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let old = max_builder
                        .parameter(0, element_type, &[], "old")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let new = max_builder
                        .parameter(1, element_type, &[], "new")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let maximum = old
                        .max(&new)
                        .map_err(|e| HoduError::InternalError(format!("Failed to max: {:?}", e)))?;
                    let max_computation = maximum
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build computation: {:?}", e)))?;

                    let indices_shape = inputs[1]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let indices_rank = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let base_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let base_rank = match &base_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                    let inserted_window_dims = vec![dim];
                    let scatter_dims_to_operand_dims = vec![dim];
                    let index_vector_dim = Some(indices_rank as i64);

                    let indices_dims_vec = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let mut indices_dims_plus_1 = indices_dims_vec;
                    indices_dims_plus_1.push(1);
                    let indices_reshaped = inputs[1]
                        .reshape(&indices_dims_plus_1)
                        .map_err(|e| HoduError::InternalError(format!("Failed to reshape: {:?}", e)))?;

                    inputs[0]
                        .scatter(
                            &indices_reshaped,
                            &inputs[2],
                            max_computation,
                            &update_window_dims,
                            &inserted_window_dims,
                            &scatter_dims_to_operand_dims,
                            index_vector_dim,
                            false,
                            false,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA scatter failed: {:?}", e)))
                },
                IndexingOp::ScatterMin => {
                    if inputs.len() != 3 {
                        return Err(HoduError::InternalError("ScatterMin requires 3 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("ScatterMin requires dim attribute".to_string()))?;

                    // Create min computation
                    let min_builder = hodu_xla::XlaBuilder::new("scatter_min");
                    let shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let element_type = match shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.element_type(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let old = min_builder
                        .parameter(0, element_type, &[], "old")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let new = min_builder
                        .parameter(1, element_type, &[], "new")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let minimum = old
                        .min(&new)
                        .map_err(|e| HoduError::InternalError(format!("Failed to min: {:?}", e)))?;
                    let min_computation = minimum
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build computation: {:?}", e)))?;

                    let indices_shape = inputs[1]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let indices_rank = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let base_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let base_rank = match &base_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                    let inserted_window_dims = vec![dim];
                    let scatter_dims_to_operand_dims = vec![dim];
                    let index_vector_dim = Some(indices_rank as i64);

                    let indices_dims_vec = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let mut indices_dims_plus_1 = indices_dims_vec;
                    indices_dims_plus_1.push(1);
                    let indices_reshaped = inputs[1]
                        .reshape(&indices_dims_plus_1)
                        .map_err(|e| HoduError::InternalError(format!("Failed to reshape: {:?}", e)))?;

                    inputs[0]
                        .scatter(
                            &indices_reshaped,
                            &inputs[2],
                            min_computation,
                            &update_window_dims,
                            &inserted_window_dims,
                            &scatter_dims_to_operand_dims,
                            index_vector_dim,
                            false,
                            false,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA scatter failed: {:?}", e)))
                },
            }
        },

        // Conv operations
        Op::Conv(conv_op) => {
            if inputs.len() != 2 {
                return Err(HoduError::InternalError(
                    "Conv requires exactly 2 inputs (input, weight)".to_string(),
                ));
            }

            match conv_op {
                ConvOp::Conv1d => {
                    let stride = attributes
                        .get("stride")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let padding = attributes
                        .get("padding")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    let dilation = attributes
                        .get("dilation")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);

                    inputs[0]
                        .conv_general_dilated(
                            &inputs[1],
                            &[stride],
                            &[(padding, padding)],
                            &[],
                            &[dilation],
                            0,
                            1,
                            &[2],
                            1,
                            0,
                            &[2],
                            1,
                            1,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA conv1d failed: {:?}", e)))
                },
                ConvOp::Conv2d => {
                    let stride = attributes
                        .get("stride")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let padding = attributes
                        .get("padding")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    let dilation = attributes
                        .get("dilation")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);

                    inputs[0]
                        .conv_general_dilated(
                            &inputs[1],
                            &[stride, stride],
                            &[(padding, padding), (padding, padding)],
                            &[],
                            &[dilation, dilation],
                            0,
                            1,
                            &[2, 3],
                            1,
                            0,
                            &[2, 3],
                            1,
                            1,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA conv2d failed: {:?}", e)))
                },
                ConvOp::Conv3d => {
                    let stride = attributes
                        .get("stride")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let padding = attributes
                        .get("padding")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    let dilation = attributes
                        .get("dilation")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);

                    inputs[0]
                        .conv_general_dilated(
                            &inputs[1],
                            &[stride, stride, stride],
                            &[(padding, padding), (padding, padding), (padding, padding)],
                            &[],
                            &[dilation, dilation, dilation],
                            0,
                            1,
                            &[2, 3, 4],
                            1,
                            0,
                            &[2, 3, 4],
                            1,
                            1,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA conv3d failed: {:?}", e)))
                },
                ConvOp::ConvTranspose1d => {
                    let kernel_size = attributes
                        .get("kernel_size")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let padding = attributes
                        .get("padding")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    let output_padding = attributes
                        .get("output_padding")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    let stride = attributes
                        .get("stride")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let dilation = attributes
                        .get("dilation")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);

                    let use_lhs_dilation = stride > 1;
                    let (xla_lhs_dilation, xla_padding_low, xla_padding_high) = if use_lhs_dilation {
                        let pad_total = 2 * kernel_size - 2 - 2 * padding + output_padding;
                        let pad_low = pad_total / 2;
                        let pad_high = pad_total - pad_low;
                        (vec![stride], pad_low, pad_high)
                    } else {
                        let pad_total = dilation * kernel_size + kernel_size - 2 - 2 * padding + output_padding;
                        let pad_low = pad_total / 2;
                        let pad_high = pad_total - pad_low;
                        (vec![], pad_low, pad_high)
                    };

                    let kernel_op = if use_lhs_dilation {
                        &inputs[1]
                    } else {
                        &inputs[1]
                            .rev(&[2])
                            .map_err(|e| HoduError::InternalError(format!("XLA rev failed: {:?}", e)))?
                    };

                    inputs[0]
                        .conv_general_dilated(
                            kernel_op,
                            &[1],
                            &[(xla_padding_low, xla_padding_high)],
                            &xla_lhs_dilation,
                            &[dilation],
                            0,
                            1,
                            &[2],
                            0,
                            1,
                            &[2],
                            1,
                            1,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA conv_transpose1d failed: {:?}", e)))
                },
                ConvOp::ConvTranspose2d => {
                    let kernel_height = attributes
                        .get("kernel_height")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let kernel_width = attributes
                        .get("kernel_width")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let padding = attributes
                        .get("padding")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    let output_padding = attributes
                        .get("output_padding")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    let stride = attributes
                        .get("stride")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let dilation = attributes
                        .get("dilation")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);

                    let use_lhs_dilation = stride > 1;
                    let (
                        xla_lhs_dilation,
                        xla_padding_h_low,
                        xla_padding_h_high,
                        xla_padding_w_low,
                        xla_padding_w_high,
                    ) = if use_lhs_dilation {
                        let pad_h_total = 2 * kernel_height - 2 - 2 * padding + output_padding;
                        let pad_h_low = pad_h_total / 2;
                        let pad_h_high = pad_h_total - pad_h_low;
                        let pad_w_total = 2 * kernel_width - 2 - 2 * padding + output_padding;
                        let pad_w_low = pad_w_total / 2;
                        let pad_w_high = pad_w_total - pad_w_low;
                        (vec![stride, stride], pad_h_low, pad_h_high, pad_w_low, pad_w_high)
                    } else {
                        let pad_h_total = dilation * kernel_height + kernel_height - 2 - 2 * padding + output_padding;
                        let pad_h_low = pad_h_total / 2;
                        let pad_h_high = pad_h_total - pad_h_low;
                        let pad_w_total = dilation * kernel_width + kernel_width - 2 - 2 * padding + output_padding;
                        let pad_w_low = pad_w_total / 2;
                        let pad_w_high = pad_w_total - pad_w_low;
                        (vec![], pad_h_low, pad_h_high, pad_w_low, pad_w_high)
                    };

                    let kernel_op = if use_lhs_dilation {
                        &inputs[1]
                    } else {
                        &inputs[1]
                            .rev(&[2, 3])
                            .map_err(|e| HoduError::InternalError(format!("XLA rev failed: {:?}", e)))?
                    };

                    inputs[0]
                        .conv_general_dilated(
                            kernel_op,
                            &[1, 1],
                            &[
                                (xla_padding_h_low, xla_padding_h_high),
                                (xla_padding_w_low, xla_padding_w_high),
                            ],
                            &xla_lhs_dilation,
                            &[dilation, dilation],
                            0,
                            1,
                            &[2, 3],
                            0,
                            1,
                            &[2, 3],
                            1,
                            1,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA conv_transpose2d failed: {:?}", e)))
                },
                ConvOp::ConvTranspose3d => {
                    let kernel_depth = attributes
                        .get("kernel_depth")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let kernel_height = attributes
                        .get("kernel_height")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let kernel_width = attributes
                        .get("kernel_width")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let padding = attributes
                        .get("padding")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    let output_padding = attributes
                        .get("output_padding")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    let stride = attributes
                        .get("stride")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);
                    let dilation = attributes
                        .get("dilation")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1);

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
                        let pad_d_total = dilation * kernel_depth + kernel_depth - 2 - 2 * padding + output_padding;
                        let pad_d_low = pad_d_total / 2;
                        let pad_d_high = pad_d_total - pad_d_low;
                        let pad_h_total = dilation * kernel_height + kernel_height - 2 - 2 * padding + output_padding;
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
                        &inputs[1]
                    } else {
                        &inputs[1]
                            .rev(&[2, 3, 4])
                            .map_err(|e| HoduError::InternalError(format!("XLA rev failed: {:?}", e)))?
                    };

                    inputs[0]
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
                            0,
                            1,
                            &[2, 3, 4],
                            0,
                            1,
                            &[2, 3, 4],
                            1,
                            1,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA conv_transpose3d failed: {:?}", e)))
                },
                _ => Err(HoduError::InternalError(format!(
                    "Conv operation {:?} not yet implemented",
                    conv_op
                ))),
            }
        },

        // Windowing operations
        Op::Windowing(windowing_op) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Windowing requires input".to_string()));
            }

            // Extract window_shape from attributes
            let window_shape: Vec<usize> = attributes
                .get("window_shape")
                .and_then(|attr| {
                    if let Attribute::IntArray(arr) = attr {
                        Some(arr.iter().map(|&d| d as usize).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Windowing requires window_shape attribute".to_string()))?;

            // Extract strides from attributes
            let strides: Vec<usize> = attributes
                .get("strides")
                .and_then(|attr| {
                    if let Attribute::IntArray(arr) = attr {
                        Some(arr.iter().map(|&d| d as usize).collect())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| vec![1; window_shape.len()]);

            // Extract padding from attributes - assuming it's stored as a flat array [pad_lo_0, pad_hi_0, pad_lo_1, pad_hi_1, ...]
            let padding_flat: Vec<i64> = attributes
                .get("padding")
                .and_then(|attr| {
                    if let Attribute::IntArray(arr) = attr {
                        Some(arr.iter().map(|&d| d as i64).collect())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| vec![0; window_shape.len() * 2]);

            let mut padding = Vec::with_capacity(window_shape.len());
            for i in 0..window_shape.len() {
                let pad_lo = padding_flat[i * 2] as usize;
                let pad_hi = padding_flat[i * 2 + 1] as usize;
                padding.push((pad_lo, pad_hi));
            }

            // Get element type from shape
            let shape = inputs[0]
                .shape()
                .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
            let element_type = match shape {
                hodu_xla::Shape::Array(array_shape) => array_shape.element_type(),
                _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
            };

            let input = &inputs[0];

            // Create initial value and reduction computation based on windowing operation
            let (init_value, reduction_comp, is_mean) = match windowing_op {
                WindowingOp::ReduceWindowMax => {
                    let init = builder
                        .min_value(element_type)
                        .map_err(|e| HoduError::InternalError(format!("Failed to create min_value: {:?}", e)))?;
                    let max_builder = hodu_xla::XlaBuilder::new("Max");
                    let x = max_builder
                        .parameter(0, element_type, &[], "x")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let y = max_builder
                        .parameter(1, element_type, &[], "y")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let comp = x
                        .max(&y)
                        .map_err(|e| HoduError::InternalError(format!("Failed to max: {:?}", e)))?
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build: {:?}", e)))?;
                    (init, comp, false)
                },
                WindowingOp::ReduceWindowMin => {
                    let init = builder
                        .max_value(element_type)
                        .map_err(|e| HoduError::InternalError(format!("Failed to create max_value: {:?}", e)))?;
                    let min_builder = hodu_xla::XlaBuilder::new("Min");
                    let x = min_builder
                        .parameter(0, element_type, &[], "x")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let y = min_builder
                        .parameter(1, element_type, &[], "y")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let comp = x
                        .min(&y)
                        .map_err(|e| HoduError::InternalError(format!("Failed to min: {:?}", e)))?
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build: {:?}", e)))?;
                    (init, comp, false)
                },
                WindowingOp::ReduceWindowSum | WindowingOp::ReduceWindowMean => {
                    let init = builder
                        .zero(element_type)
                        .map_err(|e| HoduError::InternalError(format!("Failed to create zero: {:?}", e)))?;
                    let add_builder = hodu_xla::XlaBuilder::new("Add");
                    let x = add_builder
                        .parameter(0, element_type, &[], "x")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let y = add_builder
                        .parameter(1, element_type, &[], "y")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let comp = x
                        .add_(&y)
                        .map_err(|e| HoduError::InternalError(format!("Failed to add: {:?}", e)))?
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build: {:?}", e)))?;
                    let is_mean = matches!(windowing_op, WindowingOp::ReduceWindowMean);
                    (init, comp, is_mean)
                },
            };

            // Convert to i64
            let window_shape_i64: Vec<i64> = window_shape.iter().map(|&v| v as i64).collect();
            let strides_i64: Vec<i64> = strides.iter().map(|&v| v as i64).collect();
            let padding_i64: Vec<(i64, i64)> = padding.iter().map(|&(lo, hi)| (lo as i64, hi as i64)).collect();

            // Apply reduce_window
            let result = input
                .reduce_window(
                    init_value,
                    reduction_comp,
                    &window_shape_i64,
                    &strides_i64,
                    &padding_i64,
                )
                .map_err(|e| HoduError::InternalError(format!("XLA reduce_window failed: {:?}", e)))?;

            // For mean, divide by window size
            if is_mean {
                let window_size: usize = window_shape.iter().product();
                let window_size_scalar = builder
                    .constant_r0(window_size as f32)
                    .map_err(|e| HoduError::InternalError(format!("Failed to create constant: {:?}", e)))?;
                result
                    .div_(&window_size_scalar)
                    .map_err(|e| HoduError::InternalError(format!("XLA div failed: {:?}", e)))
            } else {
                Ok(result)
            }
        },

        // Cast operations
        Op::Cast(CastOp::ToDType) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("ToDType requires input".to_string()));
            }

            // Get target dtype from output value_dtypes
            let target_dtype = compiled
                .value_dtypes
                .get(&instr.result)
                .ok_or_else(|| HoduError::InternalError("Output dtype not found".to_string()))?;

            // Convert DType to XLA PrimitiveType
            let target_element_type = match target_dtype {
                crate::types::DType::BOOL => hodu_xla::PrimitiveType::Pred,
                crate::types::DType::BF16 => hodu_xla::PrimitiveType::Bf16,
                crate::types::DType::F16 => hodu_xla::PrimitiveType::F16,
                crate::types::DType::F32 => hodu_xla::PrimitiveType::F32,
                #[cfg(feature = "f64")]
                crate::types::DType::F64 => hodu_xla::PrimitiveType::F64,
                crate::types::DType::U8 => hodu_xla::PrimitiveType::U8,
                #[cfg(feature = "u16")]
                crate::types::DType::U16 => hodu_xla::PrimitiveType::U16,
                crate::types::DType::U32 => hodu_xla::PrimitiveType::U32,
                #[cfg(feature = "u64")]
                crate::types::DType::U64 => hodu_xla::PrimitiveType::U64,
                crate::types::DType::I8 => hodu_xla::PrimitiveType::S8,
                #[cfg(feature = "i16")]
                crate::types::DType::I16 => hodu_xla::PrimitiveType::S16,
                crate::types::DType::I32 => hodu_xla::PrimitiveType::S32,
                #[cfg(feature = "i64")]
                crate::types::DType::I64 => hodu_xla::PrimitiveType::S64,
                _ => {
                    return Err(HoduError::InternalError(format!(
                        "XLA does not support dtype: {:?}",
                        target_dtype
                    )))
                },
            };

            inputs[0]
                .convert(target_element_type)
                .map_err(|e| HoduError::InternalError(format!("XLA convert failed: {:?}", e)))
        },

        // Memory operations
        Op::Memory(MemoryOp::Contiguous) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Contiguous requires input".to_string()));
            }
            // In XLA, all data is contiguous
            Ok(inputs[0].clone())
        },

        // Shape operations
        Op::Shape(ShapeOp::Reshape) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Reshape requires input".to_string()));
            }

            // Get target shape from output layout
            let output_layout = compiled
                .value_layouts
                .get(&instr.result)
                .ok_or_else(|| HoduError::InternalError("Output layout not found".to_string()))?;
            let target_shape: Vec<i64> = output_layout.shape().dims().iter().map(|&d| d as i64).collect();

            inputs[0]
                .reshape(&target_shape)
                .map_err(|e| HoduError::InternalError(format!("XLA reshape failed: {:?}", e)))
        },
        Op::Shape(ShapeOp::Flatten) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Flatten requires input".to_string()));
            }
            let input_shape = inputs[0]
                .shape()
                .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
            let total_size = match input_shape {
                hodu_xla::Shape::Array(array_shape) => array_shape.dims().iter().product::<i64>(),
                _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
            };
            inputs[0]
                .reshape(&[total_size])
                .map_err(|e| HoduError::InternalError(format!("XLA flatten failed: {:?}", e)))
        },
        Op::Shape(ShapeOp::Squeeze) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Squeeze requires input".to_string()));
            }
            // Get current shape and remove dimensions of size 1
            let input_shape = inputs[0]
                .shape()
                .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
            let dims = match input_shape {
                hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
            };
            let squeezed_dims: Vec<i64> = dims.iter().filter(|&&d| d != 1).copied().collect();
            if squeezed_dims.is_empty() {
                // If all dimensions are 1, keep at least one
                inputs[0]
                    .reshape(&[1])
                    .map_err(|e| HoduError::InternalError(format!("XLA squeeze failed: {:?}", e)))
            } else {
                inputs[0]
                    .reshape(&squeezed_dims)
                    .map_err(|e| HoduError::InternalError(format!("XLA squeeze failed: {:?}", e)))
            }
        },
        Op::Shape(ShapeOp::Unsqueeze) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Unsqueeze requires input".to_string()));
            }

            // Get target shape from output layout
            let output_layout = compiled
                .value_layouts
                .get(&instr.result)
                .ok_or_else(|| HoduError::InternalError("Output layout not found".to_string()))?;
            let target_dims: Vec<i64> = output_layout.shape().dims().iter().map(|&d| d as i64).collect();

            inputs[0]
                .reshape(&target_dims)
                .map_err(|e| HoduError::InternalError(format!("XLA unsqueeze failed: {:?}", e)))
        },
        Op::Shape(ShapeOp::Broadcast) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Broadcast requires input".to_string()));
            }

            // Get target shape from output layout
            let output_layout = compiled
                .value_layouts
                .get(&instr.result)
                .ok_or_else(|| HoduError::InternalError("Output layout not found".to_string()))?;
            let target_i64: Vec<i64> = output_layout.shape().dims().iter().map(|&d| d as i64).collect();

            let input_shape = inputs[0]
                .shape()
                .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
            let input_dims = match input_shape {
                hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
            };

            if input_dims == target_i64 {
                Ok(inputs[0].clone())
            } else {
                let input_rank = input_dims.len();
                let target_rank = target_i64.len();

                if input_rank <= target_rank {
                    let broadcast_dims: Vec<i64> = (target_rank - input_rank..target_rank).map(|i| i as i64).collect();
                    inputs[0]
                        .broadcast_in_dim(&target_i64, &broadcast_dims)
                        .map_err(|e| HoduError::InternalError(format!("XLA broadcast failed: {:?}", e)))
                } else {
                    Err(HoduError::InternalError(
                        "Cannot broadcast to smaller shape".to_string(),
                    ))
                }
            }
        },
        Op::Shape(ShapeOp::Transpose) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Transpose requires input".to_string()));
            }

            // Default transpose: swap last two dimensions
            let shape = inputs[0]
                .shape()
                .ok()
                .and_then(|s| match s {
                    hodu_xla::Shape::Array(array_shape) => {
                        let rank = array_shape.dims().len();
                        if rank < 2 {
                            None
                        } else {
                            let mut perm: Vec<i64> = (0..rank as i64).collect();
                            perm.swap(rank - 2, rank - 1);
                            Some(perm)
                        }
                    },
                    _ => None,
                })
                .ok_or_else(|| HoduError::InternalError("Failed to compute transpose permutation".to_string()))?;

            inputs[0]
                .transpose(&shape)
                .map_err(|e| HoduError::InternalError(format!("XLA transpose failed: {:?}", e)))
        },
        Op::Shape(ShapeOp::Permute) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Permute requires input".to_string()));
            }

            // Get permutation from attributes
            let perm_i64: Vec<i64> = attributes
                .get("perm")
                .and_then(|attr| {
                    if let Attribute::IntArray(arr) = attr {
                        Some(arr.iter().map(|&d| d as i64).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Permute requires perm attribute".to_string()))?;

            inputs[0]
                .transpose(&perm_i64)
                .map_err(|e| HoduError::InternalError(format!("XLA permute failed: {:?}", e)))
        },

        // ShapeScalars operations
        Op::ShapeScalars(ShapeScalarsOp::Slice) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Slice requires input".to_string()));
            }

            // Extract slice parameters from attributes
            let dim = attributes
                .get("dim")
                .and_then(|attr| {
                    if let Attribute::Int(i) = attr {
                        Some(*i as i64)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Slice requires dim attribute".to_string()))?;

            let start = attributes
                .get("start")
                .and_then(|attr| {
                    if let Attribute::Int(i) = attr {
                        Some(*i as i64)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Slice requires start attribute".to_string()))?;

            let end_value = attributes
                .get("end")
                .and_then(|attr| if let Attribute::Int(i) = attr { Some(*i) } else { None })
                .ok_or_else(|| HoduError::InternalError("Slice requires end attribute".to_string()))?;

            let stride = attributes
                .get("stride")
                .and_then(|attr| {
                    if let Attribute::Int(i) = attr {
                        Some(*i as i64)
                    } else {
                        None
                    }
                })
                .unwrap_or(1);

            // Get input shape to compute actual indices
            let input_shape = inputs[0]
                .shape()
                .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
            let dims = match input_shape {
                hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
            };

            let dim_size = dims[dim as usize];
            let start_idx = if start < 0 { dim_size + start } else { start };
            let end_idx = if end_value == i32::MAX {
                if stride > 0 {
                    dim_size
                } else {
                    -1
                }
            } else {
                let end = end_value as i64;
                if end < 0 {
                    dim_size + end
                } else {
                    end
                }
            };

            inputs[0]
                .slice_in_dim(start_idx, end_idx, stride, dim)
                .map_err(|e| HoduError::InternalError(format!("XLA slice_in_dim failed: {:?}", e)))
        },

        Op::Dummy => {
            if inputs.is_empty() {
                builder
                    .constant_r0(0.0f32)
                    .map_err(|e| HoduError::InternalError(format!("Failed to create dummy op: {:?}", e)))
            } else {
                Ok(inputs[0].clone())
            }
        },
    }
}

#[cfg(feature = "xla")]
fn unary_op_check<F>(inputs: &[hodu_xla::XlaOp], f: F, op_name: &str) -> HoduResult<hodu_xla::XlaOp>
where
    F: FnOnce(&[hodu_xla::XlaOp]) -> Result<hodu_xla::XlaOp, hodu_xla::Error>,
{
    if inputs.len() != 1 {
        return Err(HoduError::InternalError(format!("{} requires 1 input", op_name)));
    }
    f(inputs).map_err(|e| HoduError::InternalError(format!("XLA {} failed: {:?}", op_name, e)))
}

#[cfg(feature = "xla")]
fn binary_op_check<F>(inputs: &[hodu_xla::XlaOp], expected: usize, f: F, op_name: &str) -> HoduResult<hodu_xla::XlaOp>
where
    F: FnOnce(&[hodu_xla::XlaOp]) -> Result<hodu_xla::XlaOp, hodu_xla::Error>,
{
    if inputs.len() != expected {
        return Err(HoduError::InternalError(format!(
            "{} requires {} inputs",
            op_name, expected
        )));
    }
    f(inputs).map_err(|e| HoduError::InternalError(format!("XLA {} failed: {:?}", op_name, e)))
}

#[cfg(feature = "xla")]
fn scalar_op<F>(
    builder: &hodu_xla::XlaBuilder,
    inputs: &[hodu_xla::XlaOp],
    attributes: &HashMap<String, Attribute>,
    f: F,
    op_name: &str,
) -> HoduResult<hodu_xla::XlaOp>
where
    F: FnOnce(&[hodu_xla::XlaOp], hodu_xla::XlaOp) -> Result<hodu_xla::XlaOp, hodu_xla::Error>,
{
    if inputs.len() != 1 {
        return Err(HoduError::InternalError(format!("{} requires 1 input", op_name)));
    }
    let scalar = get_scalar_from_attributes(attributes)?;
    let scalar_op = builder
        .constant_r0(scalar)
        .map_err(|e| HoduError::InternalError(format!("Failed to create scalar constant: {:?}", e)))?;
    f(inputs, scalar_op).map_err(|e| HoduError::InternalError(format!("XLA {} failed: {:?}", op_name, e)))
}

#[cfg(feature = "xla")]
fn get_scalar_from_attributes(attributes: &HashMap<String, Attribute>) -> HoduResult<f32> {
    match attributes.get("scalar") {
        Some(Attribute::Scalar(scalar)) => Ok(scalar.to_f32()),
        _ => Err(HoduError::InternalError("Missing scalar attribute".to_string())),
    }
}

#[cfg(feature = "xla")]
fn tensor_to_literal(tensor: &crate::tensor::Tensor) -> HoduResult<hodu_xla::Literal> {
    tensor.with_storage(|storage| match storage {
        BackendStorage::CPU(cpu_storage) => cpu_storage_to_literal(cpu_storage),
        #[cfg(any(feature = "cuda", feature = "metal"))]
        _ => Err(HoduError::InternalError(
            "Only CPU storage supported for XLA conversion".to_string(),
        )),
    })
}

#[cfg(feature = "xla")]
fn cpu_storage_to_literal(storage: &crate::be_cpu::storage::CpuStorage) -> HoduResult<hodu_xla::Literal> {
    use crate::be_cpu::storage::CpuStorage;
    use hodu_xla::Literal;

    match storage {
        CpuStorage::BOOL(data) => Ok(Literal::vec1(data)),
        CpuStorage::F8E4M3(_) => Err(HoduError::InternalError("F8E4M3 not supported by XLA".to_string())),
        #[cfg(feature = "f8e5m2")]
        CpuStorage::F8E5M2(_) => Err(HoduError::InternalError("F8E5M2 not supported by XLA".to_string())),
        CpuStorage::BF16(data) => Ok(Literal::vec1(data)),
        CpuStorage::F16(data) => Ok(Literal::vec1(data)),
        CpuStorage::F32(data) => Ok(Literal::vec1(data)),
        #[cfg(feature = "f64")]
        CpuStorage::F64(data) => Ok(Literal::vec1(data)),
        CpuStorage::U8(data) => Ok(Literal::vec1(data)),
        #[cfg(feature = "u16")]
        CpuStorage::U16(data) => Ok(Literal::vec1(data)),
        CpuStorage::U32(data) => Ok(Literal::vec1(data)),
        #[cfg(feature = "u64")]
        CpuStorage::U64(data) => Ok(Literal::vec1(data)),
        CpuStorage::I8(data) => Ok(Literal::vec1(data)),
        #[cfg(feature = "i16")]
        CpuStorage::I16(data) => Ok(Literal::vec1(data)),
        CpuStorage::I32(data) => Ok(Literal::vec1(data)),
        #[cfg(feature = "i64")]
        CpuStorage::I64(data) => Ok(Literal::vec1(data)),
    }
}

#[cfg(feature = "xla")]
fn literal_to_tensor(literal: &hodu_xla::Literal, dtype: crate::types::DType) -> HoduResult<crate::tensor::Tensor> {
    use crate::be::storage::BackendStorage;
    use crate::be_cpu::storage::CpuStorage;
    use crate::types::{DType, Layout};

    // Get shape from literal
    let shape_result = literal
        .shape()
        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
    let shape: Vec<usize> = match &shape_result {
        hodu_xla::Shape::Array(array_shape) => array_shape.dims().iter().map(|&d| d as usize).collect(),
        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
    };

    // Convert to CPU storage based on dtype
    let cpu_storage = match dtype {
        DType::BOOL => {
            let data = literal
                .to_vec::<bool>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract BOOL data: {:?}", e)))?;
            CpuStorage::BOOL(data)
        },
        DType::F8E4M3 => return Err(HoduError::InternalError("F8E4M3 not supported by XLA".to_string())),
        #[cfg(feature = "f8e5m2")]
        DType::F8E5M2 => return Err(HoduError::InternalError("F8E5M2 not supported by XLA".to_string())),
        DType::BF16 => {
            let data = literal
                .to_vec::<half::bf16>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract BF16 data: {:?}", e)))?;
            CpuStorage::BF16(data)
        },
        DType::F16 => {
            let data = literal
                .to_vec::<half::f16>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract F16 data: {:?}", e)))?;
            CpuStorage::F16(data)
        },
        DType::F32 => {
            let data = literal
                .to_vec::<f32>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract F32 data: {:?}", e)))?;
            CpuStorage::F32(data)
        },
        #[cfg(feature = "f64")]
        DType::F64 => {
            let data = literal
                .to_vec::<f64>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract F64 data: {:?}", e)))?;
            CpuStorage::F64(data)
        },
        DType::U8 => {
            let data = literal
                .to_vec::<u8>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract U8 data: {:?}", e)))?;
            CpuStorage::U8(data)
        },
        #[cfg(feature = "u16")]
        DType::U16 => {
            let data = literal
                .to_vec::<u16>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract U16 data: {:?}", e)))?;
            CpuStorage::U16(data)
        },
        DType::U32 => {
            let data = literal
                .to_vec::<u32>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract U32 data: {:?}", e)))?;
            CpuStorage::U32(data)
        },
        #[cfg(feature = "u64")]
        DType::U64 => {
            let data = literal
                .to_vec::<u64>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract U64 data: {:?}", e)))?;
            CpuStorage::U64(data)
        },
        DType::I8 => {
            let data = literal
                .to_vec::<i8>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract I8 data: {:?}", e)))?;
            CpuStorage::I8(data)
        },
        #[cfg(feature = "i16")]
        DType::I16 => {
            let data = literal
                .to_vec::<i16>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract I16 data: {:?}", e)))?;
            CpuStorage::I16(data)
        },
        DType::I32 => {
            let data = literal
                .to_vec::<i32>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract I32 data: {:?}", e)))?;
            CpuStorage::I32(data)
        },
        #[cfg(feature = "i64")]
        DType::I64 => {
            let data = literal
                .to_vec::<i64>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract I64 data: {:?}", e)))?;
            CpuStorage::I64(data)
        },
    };

    let shape_u32: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
    let shape_obj = crate::types::Shape::new(&shape_u32);
    let layout = Layout::from_shape(&shape_obj);
    Ok(from_storage(BackendStorage::CPU(cpu_storage), layout, true, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xla_executor_creation() {
        let executor = XlaExecutor::new(Device::CPU);
        assert_eq!(executor.compiler_type(), Compiler::XLA);
        assert_eq!(executor.device(), Device::CPU);
    }
}
