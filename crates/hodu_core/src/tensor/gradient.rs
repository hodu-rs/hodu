mod utils;

use crate::{
    backends::op::{
        BinaryLogicalOp, BinaryOp, CmpOp, CmpScalarOp, MatrixOp, Op, ReduceOp, ShapeOp, UnaryLogicalOp, UnaryOp,
        UnaryScalarOp,
    },
    compat::*,
    error::{HoduError, HoduResult},
    scalar::Scalar,
    tensor::{self, set_grad_tensor_id, tensor_from_id, TensorId},
};
use utils::*;

pub trait VjpCompute {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        output: TensorId,
        grad_output: TensorId,
        scalar: Option<Scalar>,
    ) -> HoduResult<Vec<TensorId>>;
}

impl VjpCompute for BinaryOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        _scalar: Option<Scalar>,
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            BinaryOp::Add => Ok(vec![grad_output, grad_output]),
            BinaryOp::Sub => Ok(vec![grad_output, create_neg_tensor(grad_output)?]),
            BinaryOp::Mul => Ok(vec![
                create_mul_tensor(grad_output, inputs[1])?,
                create_mul_tensor(grad_output, inputs[0])?,
            ]),
            BinaryOp::Div => {
                let grad_a = create_div_tensor(grad_output, inputs[1])?;
                let b_squared = create_mul_tensor(inputs[1], inputs[1])?;
                let grad_output_mul_a = create_mul_tensor(grad_output, inputs[0])?;
                let grad_a_div_b_sq = create_div_tensor(grad_output_mul_a, b_squared)?;
                let neg_grad_a_div_b_sq = create_neg_tensor(grad_a_div_b_sq)?;
                Ok(vec![grad_a, neg_grad_a_div_b_sq])
            },
            BinaryOp::Pow => {
                // d/dx (x^y) = y * x^(y-1)
                // d/dy (x^y) = x^y * ln(x)
                let x = inputs[0];
                let y = inputs[1];

                let x_tensor = tensor_from_id(x);
                let dtype = x_tensor.get_dtype();
                let one = create_scalar_for_dtype(1.0, dtype);

                let grad_x = create_mul_tensor(
                    grad_output,
                    create_mul_tensor(y, create_pow_tensor(x, create_sub_scalar_tensor(y, one)?)?)?,
                )?;
                let grad_y = create_mul_tensor(grad_output, create_mul_tensor(_output, create_ln_tensor(x)?)?)?;

                Ok(vec![grad_x, grad_y])
            },
            BinaryOp::Maximum => {
                // gradient goes to the larger input
                let a_ge_b = create_ge_tensor(inputs[0], inputs[1])?; // a >= b mask
                let b_gt_a = create_gt_tensor(inputs[1], inputs[0])?; // b > a mask
                Ok(vec![
                    create_mul_tensor(grad_output, a_ge_b)?, // grad to a if a >= b
                    create_mul_tensor(grad_output, b_gt_a)?, // grad to b if b > a
                ])
            },
            BinaryOp::Minimum => {
                // gradient goes to the smaller input
                let a_le_b = create_le_tensor(inputs[0], inputs[1])?; // a <= b mask
                let b_lt_a = create_lt_tensor(inputs[1], inputs[0])?; // b < a mask
                Ok(vec![
                    create_mul_tensor(grad_output, a_le_b)?, // grad to a if a <= b
                    create_mul_tensor(grad_output, b_lt_a)?, // grad to b if b < a
                ])
            },
        }
    }
}

impl VjpCompute for BinaryLogicalOp {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _scalar: Option<Scalar>,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(format!("BinaryLogicalOp::{self:?}")))
    }
}

impl VjpCompute for CmpOp {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _scalar: Option<Scalar>,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(format!("CmpOp::{self:?}")))
    }
}

impl VjpCompute for CmpScalarOp {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _scalar: Option<Scalar>,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(format!("CmpScalarOp::{self:?}")))
    }
}

impl VjpCompute for UnaryOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        _scalar: Option<Scalar>,
    ) -> HoduResult<Vec<TensorId>> {
        let input = inputs[0];
        match self {
            UnaryOp::Neg => Ok(vec![create_neg_tensor(grad_output)?]),
            UnaryOp::Abs => {
                // d/dx |x| = sign(x)
                let sign_input = create_sign_tensor(input)?;
                Ok(vec![create_mul_tensor(grad_output, sign_input)?])
            },
            UnaryOp::Sign => {
                // d/dx sign(x) = 0 (discontinuous)
                Ok(vec![create_zeros_like_tensor(input)?])
            },
            UnaryOp::Square => {
                // d/dx x^2 = 2x
                let two_x = create_add_tensor(input, input)?;
                Ok(vec![create_mul_tensor(grad_output, two_x)?])
            },
            UnaryOp::Relu => {
                // d/dx relu(x) = 1 if x > 0, else 0
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let zero = create_scalar_for_dtype(0.0, dtype);
                let mask = create_gt_scalar_tensor(input, zero)?;
                Ok(vec![create_mul_tensor(grad_output, mask)?])
            },
            UnaryOp::Sigmoid => {
                // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                let sigmoid_output = _output;
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let one = create_scalar_for_dtype(1.0, dtype);
                let one_minus_sigmoid = create_sub_scalar_tensor(sigmoid_output, one)?;
                let neg_one_minus_sigmoid = create_neg_tensor(one_minus_sigmoid)?;
                let derivative = create_mul_tensor(sigmoid_output, neg_one_minus_sigmoid)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Tanh => {
                // d/dx tanh(x) = 1 - tanh^2(x)
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let one = create_scalar_for_dtype(1.0, dtype);
                let tanh_squared = create_mul_tensor(_output, _output)?;
                let derivative = create_sub_scalar_tensor(tanh_squared, one)?;
                let neg_derivative = create_neg_tensor(derivative)?;
                Ok(vec![create_mul_tensor(grad_output, neg_derivative)?])
            },
            UnaryOp::Gelu => {
                // GELU derivative is complex, approximation: d/dx ≈ 0.5 * (1 + tanh(...))
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let sqrt_2_over_pi = create_scalar_for_dtype(0.797884, dtype);
                let one = create_scalar_for_dtype(1.0, dtype);
                let half = create_scalar_for_dtype(0.5, dtype);
                let approx_derivative = create_mul_scalar_tensor(input, sqrt_2_over_pi)?;
                let tanh_part = create_tanh_tensor(approx_derivative)?;
                let one_plus_tanh = create_add_scalar_tensor(tanh_part, one)?;
                let derivative = create_mul_scalar_tensor(one_plus_tanh, half)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Sin => {
                // d/dx sin(x) = cos(x)
                let cos_input = create_cos_tensor(input)?;
                Ok(vec![create_mul_tensor(grad_output, cos_input)?])
            },
            UnaryOp::Cos => {
                // d/dx cos(x) = -sin(x)
                let sin_input = create_sin_tensor(input)?;
                let neg_sin_input = create_neg_tensor(sin_input)?;
                Ok(vec![create_mul_tensor(grad_output, neg_sin_input)?])
            },
            UnaryOp::Tan => {
                // d/dx tan(x) = sec^2(x) = 1 + tan^2(x)
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let one = create_scalar_for_dtype(1.0, dtype);
                let tan_squared = create_mul_tensor(_output, _output)?;
                let derivative = create_add_scalar_tensor(tan_squared, one)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Ln => {
                // d/dx ln(x) = 1/x
                let derivative = create_recip_tensor(input)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Log10 => {
                // d/dx log10(x) = 1/(x * ln(10))
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let ln_10 = create_scalar_for_dtype(core::f32::consts::LN_10, dtype);
                let x_ln_10 = create_mul_scalar_tensor(input, ln_10)?;
                let derivative = create_recip_tensor(x_ln_10)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Log2 => {
                // d/dx log2(x) = 1/(x * ln(2))
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let ln_2 = create_scalar_for_dtype(core::f32::consts::LN_2, dtype);
                let x_ln_2 = create_mul_scalar_tensor(input, ln_2)?;
                let derivative = create_recip_tensor(x_ln_2)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Exp => {
                // d/dx exp(x) = exp(x)
                Ok(vec![create_mul_tensor(grad_output, _output)?])
            },
            UnaryOp::Exp10 => {
                // d/dx 10^x = 10^x * ln(10)
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let ln_10 = create_scalar_for_dtype(core::f32::consts::LN_10, dtype);
                let derivative = create_mul_scalar_tensor(_output, ln_10)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Exp2 => {
                // d/dx 2^x = 2^x * ln(2)
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let ln_2 = create_scalar_for_dtype(core::f32::consts::LN_2, dtype);
                let derivative = create_mul_scalar_tensor(_output, ln_2)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Softplus => {
                // d/dx softplus(x) = d/dx ln(1 + exp(x)) = exp(x)/(1 + exp(x)) = sigmoid(x)
                let sigmoid_input = create_sigmoid_tensor(input)?;
                Ok(vec![create_mul_tensor(grad_output, sigmoid_input)?])
            },
            UnaryOp::Recip => {
                // d/dx (1/x) = -1/x^2
                let x_squared = create_mul_tensor(input, input)?;
                let neg_recip_x_squared = create_neg_tensor(create_recip_tensor(x_squared)?)?;
                Ok(vec![create_mul_tensor(grad_output, neg_recip_x_squared)?])
            },
            UnaryOp::Sqrt => {
                // d/dx sqrt(x) = 1/(2*sqrt(x)) = 0.5 * recip(sqrt(x))
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let half = create_scalar_for_dtype(0.5, dtype);
                let recip_sqrt = create_recip_tensor(_output)?; // 1/sqrt(x)
                let derivative = create_mul_scalar_tensor(recip_sqrt, half)?; // 0.5/sqrt(x)
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
        }
    }
}

impl VjpCompute for UnaryLogicalOp {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _scalar: Option<Scalar>,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(format!("UnaryLogicalOp::{self:?}")))
    }
}

impl VjpCompute for UnaryScalarOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        scalar: Option<Scalar>,
    ) -> HoduResult<Vec<TensorId>> {
        let input = inputs[0];
        match self {
            UnaryScalarOp::AddScalar => {
                // d/dx (x + c) = 1
                Ok(vec![grad_output])
            },
            UnaryScalarOp::SubScalar => {
                // d/dx (x - c) = 1
                Ok(vec![grad_output])
            },
            UnaryScalarOp::MulScalar => {
                // d/dx (x * c) = c
                if let Some(c) = scalar {
                    Ok(vec![create_mul_scalar_tensor(grad_output, c)?])
                } else {
                    Ok(vec![grad_output]) // fallback
                }
            },
            UnaryScalarOp::DivScalar => {
                // d/dx (x / c) = 1/c
                if let Some(c) = scalar {
                    let input_tensor = tensor_from_id(input);
                    let dtype = input_tensor.get_dtype();
                    let one = create_scalar_for_dtype(1.0, dtype);
                    // Create 1/c scalar with proper dtype
                    let c_value = match c {
                        Scalar::F32(v) => create_scalar_for_dtype(1.0 / v, dtype),
                        Scalar::F64(v) => create_scalar_for_dtype(1.0 / v as f32, dtype),
                        Scalar::F16(v) => create_scalar_for_dtype(1.0 / v.to_f32(), dtype),
                        Scalar::BF16(v) => create_scalar_for_dtype(1.0 / v.to_f32(), dtype),
                        Scalar::F8E4M3(v) => create_scalar_for_dtype(1.0 / v.to_f32(), dtype),
                        Scalar::F8E5M2(v) => create_scalar_for_dtype(1.0 / v.to_f32(), dtype),
                        _ => one,
                    };
                    Ok(vec![create_mul_scalar_tensor(grad_output, c_value)?])
                } else {
                    Ok(vec![grad_output]) // fallback
                }
            },
            UnaryScalarOp::PowScalar => {
                // d/dx (x^c) = c * x^(c-1)
                if let Some(c) = scalar {
                    let input_tensor = tensor_from_id(input);
                    let dtype = input_tensor.get_dtype();
                    let c_minus_one = match c {
                        Scalar::F32(v) => create_scalar_for_dtype(v - 1.0, dtype),
                        Scalar::F64(v) => create_scalar_for_dtype((v - 1.0) as f32, dtype),
                        Scalar::F16(v) => create_scalar_for_dtype(v.to_f32() - 1.0, dtype),
                        Scalar::BF16(v) => create_scalar_for_dtype(v.to_f32() - 1.0, dtype),
                        Scalar::F8E4M3(v) => create_scalar_for_dtype(v.to_f32() - 1.0, dtype),
                        Scalar::F8E5M2(v) => create_scalar_for_dtype(v.to_f32() - 1.0, dtype),
                        _ => create_scalar_for_dtype(0.0, dtype),
                    };
                    let x_power_c_minus_one = create_pow_scalar_tensor(input, c_minus_one)?;
                    let derivative = create_mul_scalar_tensor(x_power_c_minus_one, c)?;
                    Ok(vec![create_mul_tensor(grad_output, derivative)?])
                } else {
                    Ok(vec![grad_output]) // fallback
                }
            },
            UnaryScalarOp::MaximumScalar => {
                // d/dx max(x, scalar) = 1 if x >= scalar, else 0
                if let Some(c) = scalar {
                    let mask = create_ge_scalar_tensor(input, c)?;
                    Ok(vec![create_mul_tensor(grad_output, mask)?])
                } else {
                    Ok(vec![grad_output]) // fallback
                }
            },
            UnaryScalarOp::MinimumScalar => {
                // d/dx min(x, scalar) = 1 if x <= scalar, else 0
                if let Some(c) = scalar {
                    let mask = create_le_scalar_tensor(input, c)?;
                    Ok(vec![create_mul_tensor(grad_output, mask)?])
                } else {
                    Ok(vec![grad_output]) // fallback
                }
            },
            UnaryScalarOp::LeakyRelu => {
                // d/dx leaky_relu(x, alpha) = 1 if x > 0, else alpha
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let zero = create_scalar_for_dtype(0.0, dtype);
                let alpha = scalar.unwrap_or(create_scalar_for_dtype(0.01, dtype));
                let mask_pos = create_gt_scalar_tensor(input, zero)?;
                let mask_neg = create_le_scalar_tensor(input, zero)?;
                let alpha_grad = create_mul_scalar_tensor(mask_neg, alpha)?;
                let total_grad = create_add_tensor(mask_pos, alpha_grad)?;
                Ok(vec![create_mul_tensor(grad_output, total_grad)?])
            },
            UnaryScalarOp::Elu => {
                // d/dx elu(x, alpha) = 1 if x > 0, else alpha * exp(x)
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let zero = create_scalar_for_dtype(0.0, dtype);
                let alpha = scalar.unwrap_or(create_scalar_for_dtype(1.0, dtype));
                let mask_pos = create_gt_scalar_tensor(input, zero)?;
                let mask_neg = create_le_scalar_tensor(input, zero)?;
                let exp_input = create_exp_tensor(input)?;
                let alpha_exp = create_mul_scalar_tensor(exp_input, alpha)?;
                let neg_grad = create_mul_tensor(mask_neg, alpha_exp)?;
                let total_grad = create_add_tensor(mask_pos, neg_grad)?;
                Ok(vec![create_mul_tensor(grad_output, total_grad)?])
            },
        }
    }
}

impl VjpCompute for MatrixOp {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        _scalar: Option<Scalar>,
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            MatrixOp::Matmul => {
                // TODO: Need transpose operation to implement properly
                // For now, return simple approximation
                Ok(vec![grad_output, grad_output])
            },
        }
    }
}

impl VjpCompute for ReduceOp {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _scalar: Option<Scalar>,
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            _ => {
                todo!()
            },
        }
    }
}

impl VjpCompute for ShapeOp {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        _scalar: Option<Scalar>,
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            ShapeOp::Transpose => {
                // TODO: Implement transpose gradient (reverse the transpose)
                Ok(vec![grad_output])
            },
            ShapeOp::Broadcast => {
                // TODO: Implement broadcast gradient (sum over broadcasted dims)
                Ok(vec![grad_output])
            },
            _ => {
                todo!()
            },
        }
    }
}

static COMPUTATION_TAPE: Mutex<Vec<(TensorId, Op, Vec<TensorId>)>> = Mutex::new(Vec::new());

static IS_COMPUTING_GRADIENTS: AtomicBool = AtomicBool::new(false);

pub fn is_computing_gradients() -> bool {
    IS_COMPUTING_GRADIENTS.load(Ordering::Relaxed)
}

pub fn record_operation(output_id: TensorId, op: Op, input_ids: Vec<TensorId>) -> HoduResult<()> {
    if !is_computing_gradients() {
        let mut tape = {
            #[cfg(feature = "std")]
            {
                COMPUTATION_TAPE.lock().map_err(|_| HoduError::GradientTapeCorrupted)?
            }
            #[cfg(not(feature = "std"))]
            {
                COMPUTATION_TAPE.lock()
            }
        };
        tape.push((output_id, op, input_ids));
    }
    Ok(())
}

// pub fn clear_tape() {
//     match COMPUTATION_TAPE.lock() {
//         Ok(mut tape) => tape.clear(),
//         Err(e) => {
//             eprintln!("Warning: Failed to clear gradient tape: {e}");
//         },
//     }
// }

pub fn compute_gradients(loss_tensor_id: TensorId) -> HoduResult<()> {
    IS_COMPUTING_GRADIENTS.store(true, Ordering::Relaxed);

    let result = (|| -> HoduResult<()> {
        let loss_tensor = tensor_from_id(loss_tensor_id);
        let loss_layout = loss_tensor.get_layout();

        let loss_grad = tensor::Tensor::ones(loss_layout.get_shape(), loss_tensor.get_dtype())?;
        let loss_grad_id = loss_grad.id();

        set_grad_tensor_id(loss_tensor_id, loss_grad_id)?;

        let mut gradients: HashMap<TensorId, TensorId> = HashMap::new();
        gradients.insert(loss_tensor_id, loss_grad_id);

        let tape = {
            #[cfg(feature = "std")]
            {
                COMPUTATION_TAPE.lock().map_err(|_| HoduError::GradientTapeCorrupted)?
            }
            #[cfg(not(feature = "std"))]
            {
                COMPUTATION_TAPE.lock()
            }
        };

        for (output_id, op, input_ids) in tape.iter().rev() {
            if let Some(&grad_output) = gradients.get(output_id) {
                for &input_id in input_ids {
                    if tensor::get(input_id).is_none() {
                        return Err(HoduError::TensorNotFound(input_id));
                    }
                }

                if tensor::get(grad_output).is_none() {
                    return Err(HoduError::TensorNotFound(grad_output));
                }

                let input_grads = compute_vjp_for_op(op, input_ids, *output_id, grad_output)?;

                for (input_id, grad_id) in input_ids.iter().zip(input_grads.iter()) {
                    if tensor::get(*grad_id).is_none() {
                        return Err(HoduError::TensorNotFound(*grad_id));
                    }

                    let final_grad_id = if let Some(&existing_grad) = gradients.get(input_id) {
                        if tensor::get(existing_grad).is_none() {
                            return Err(HoduError::TensorNotFound(existing_grad));
                        }
                        create_add_tensor(existing_grad, *grad_id)?
                    } else {
                        *grad_id
                    };

                    gradients.insert(*input_id, final_grad_id);

                    let input_tensor = tensor_from_id(*input_id);
                    if input_tensor.is_requires_grad() {
                        set_grad_tensor_id(*input_id, final_grad_id)?;
                    }
                }
            }
        }

        Ok(())
    })();

    IS_COMPUTING_GRADIENTS.store(false, Ordering::Relaxed);

    result
}

fn compute_vjp_for_op(
    op: &Op,
    inputs: &[TensorId],
    output: TensorId,
    grad_output: TensorId,
) -> HoduResult<Vec<TensorId>> {
    match op {
        Op::Binary(binary_op, _, _) => binary_op.compute_vjp(inputs, output, grad_output, None),
        Op::BinaryLogical(binary_logical_op, _, _) => binary_logical_op.compute_vjp(inputs, output, grad_output, None),
        Op::Cmp(cmp_op, _, _) => cmp_op.compute_vjp(inputs, output, grad_output, None),
        Op::CmpScalar(cmp_scalar_op, _, scalar) => {
            cmp_scalar_op.compute_vjp(inputs, output, grad_output, Some(*scalar))
        },
        Op::Unary(unary_op, _) => unary_op.compute_vjp(inputs, output, grad_output, None),
        Op::UnaryLogical(unary_logical_op, _) => unary_logical_op.compute_vjp(inputs, output, grad_output, None),
        Op::UnaryScalar(unary_scalar_op, _, scalar) => {
            unary_scalar_op.compute_vjp(inputs, output, grad_output, Some(*scalar))
        },
        Op::Matrix(matrix_op, _, _) => matrix_op.compute_vjp(inputs, output, grad_output, None),
        Op::Reduce(reduce_op, _, _) => reduce_op.compute_vjp(inputs, output, grad_output, None),
        Op::Shape(shape_op, _) => shape_op.compute_vjp(inputs, output, grad_output, None),
        _ => Err(HoduError::VjpFunctionNotFound(format!("compute_vjp for {:?}", op))),
    }
}
