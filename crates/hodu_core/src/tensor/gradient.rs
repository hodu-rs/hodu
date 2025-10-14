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
use num_traits::float::Float;
use utils::*;

pub trait VjpCompute {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(format!("compute_vjp not implemented")))
    }

    fn compute_vjp_with_scalar(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _scalar: Scalar,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(format!(
            "compute_vjp_with_scalar not implemented"
        )))
    }

    fn compute_vjp_with_dims(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _dims: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(format!(
            "compute_vjp_with_dims not implemented"
        )))
    }
}

impl VjpCompute for BinaryOp {
    fn compute_vjp(&self, inputs: &[TensorId], _output: TensorId, grad_output: TensorId) -> HoduResult<Vec<TensorId>> {
        // Get input shapes for gradient reduction
        let input_a_shape = tensor_from_id(inputs[0]).get_layout().get_shape().to_vec();
        let input_b_shape = tensor_from_id(inputs[1]).get_layout().get_shape().to_vec();

        match self {
            BinaryOp::Add => Ok(vec![
                create_sum_to_shape_tensor(grad_output, &input_a_shape)?,
                create_sum_to_shape_tensor(grad_output, &input_b_shape)?,
            ]),
            BinaryOp::Sub => Ok(vec![
                create_sum_to_shape_tensor(grad_output, &input_a_shape)?,
                create_sum_to_shape_tensor(create_neg_tensor(grad_output)?, &input_b_shape)?,
            ]),
            BinaryOp::Mul => Ok(vec![
                create_sum_to_shape_tensor(create_mul_tensor(grad_output, inputs[1])?, &input_a_shape)?,
                create_sum_to_shape_tensor(create_mul_tensor(grad_output, inputs[0])?, &input_b_shape)?,
            ]),
            BinaryOp::Div => {
                let grad_a_raw = create_div_tensor(grad_output, inputs[1])?;
                let b_squared = create_mul_tensor(inputs[1], inputs[1])?;
                let grad_output_mul_a = create_mul_tensor(grad_output, inputs[0])?;
                let grad_b_div_b_sq = create_div_tensor(grad_output_mul_a, b_squared)?;
                let grad_b_raw = create_neg_tensor(grad_b_div_b_sq)?;
                Ok(vec![
                    create_sum_to_shape_tensor(grad_a_raw, &input_a_shape)?,
                    create_sum_to_shape_tensor(grad_b_raw, &input_b_shape)?,
                ])
            },
            BinaryOp::Pow => {
                // d/dx (x^y) = y * x^(y-1)
                // d/dy (x^y) = x^y * ln(x)
                let x = inputs[0];
                let y = inputs[1];

                let x_tensor = tensor_from_id(x);
                let dtype = x_tensor.get_dtype();
                let one = Scalar::one(dtype);

                let grad_x_raw = create_mul_tensor(
                    grad_output,
                    create_mul_tensor(y, create_pow_tensor(x, create_sub_scalar_tensor(y, one)?)?)?,
                )?;
                let grad_y_raw = create_mul_tensor(grad_output, create_mul_tensor(_output, create_ln_tensor(x)?)?)?;

                Ok(vec![
                    create_sum_to_shape_tensor(grad_x_raw, &input_a_shape)?,
                    create_sum_to_shape_tensor(grad_y_raw, &input_b_shape)?,
                ])
            },
            BinaryOp::Maximum => {
                // gradient goes to the larger input
                let a_ge_b = create_ge_tensor(inputs[0], inputs[1])?; // a >= b mask
                let b_gt_a = create_gt_tensor(inputs[1], inputs[0])?; // b > a mask
                let grad_a_raw = create_mul_tensor(grad_output, a_ge_b)?;
                let grad_b_raw = create_mul_tensor(grad_output, b_gt_a)?;
                Ok(vec![
                    create_sum_to_shape_tensor(grad_a_raw, &input_a_shape)?,
                    create_sum_to_shape_tensor(grad_b_raw, &input_b_shape)?,
                ])
            },
            BinaryOp::Minimum => {
                // gradient goes to the smaller input
                let a_le_b = create_le_tensor(inputs[0], inputs[1])?; // a <= b mask
                let b_lt_a = create_lt_tensor(inputs[1], inputs[0])?; // b < a mask
                let grad_a_raw = create_mul_tensor(grad_output, a_le_b)?;
                let grad_b_raw = create_mul_tensor(grad_output, b_lt_a)?;
                Ok(vec![
                    create_sum_to_shape_tensor(grad_a_raw, &input_a_shape)?,
                    create_sum_to_shape_tensor(grad_b_raw, &input_b_shape)?,
                ])
            },
        }
    }
}

impl VjpCompute for BinaryLogicalOp {}

impl VjpCompute for CmpOp {}

impl VjpCompute for CmpScalarOp {}

impl VjpCompute for UnaryOp {
    fn compute_vjp(&self, inputs: &[TensorId], _output: TensorId, grad_output: TensorId) -> HoduResult<Vec<TensorId>> {
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
                let zero = Scalar::zero(dtype);
                let mask = create_gt_scalar_tensor(input, zero)?;
                Ok(vec![create_mul_tensor(grad_output, mask)?])
            },
            UnaryOp::Sigmoid => {
                // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                let sigmoid_output = _output;
                let ones = create_ones_like_tensor(sigmoid_output)?;
                let one_minus_sigmoid = create_sub_tensor(ones, sigmoid_output)?; // 1 - sigmoid
                let derivative = create_mul_tensor(sigmoid_output, one_minus_sigmoid)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Tanh => {
                // d/dx tanh(x) = 1 - tanh^2(x)
                let tanh_squared = create_mul_tensor(_output, _output)?;
                let ones = create_ones_like_tensor(input)?;
                let derivative = create_sub_tensor(ones, tanh_squared)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Gelu => {
                // GELU(x) = x * Φ(x) where Φ is standard normal CDF
                // Using tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
                // d/dx GELU(x) ≈ 0.5 * (1 + tanh(z)) + x * 0.5 * sech²(z) * dz/dx
                // where z = √(2/π) * (x + 0.044715 * x³), dz/dx = √(2/π) * (1 + 3 * 0.044715 * x²)
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let sqrt_2_over_pi = Scalar::from_f32(0.797885, dtype);
                let coeff = Scalar::from_f32(0.044715, dtype);
                let half = Scalar::from_f32(0.5, dtype);

                // Compute x²
                let x_squared = create_mul_tensor(input, input)?;

                // Compute z = √(2/π) * (x + 0.044715 * x³)
                let x_cubed = create_mul_tensor(x_squared, input)?;
                let coeff_x_cubed = create_mul_scalar_tensor(x_cubed, coeff)?;
                let x_plus_coeff_x_cubed = create_add_tensor(input, coeff_x_cubed)?;
                let z = create_mul_scalar_tensor(x_plus_coeff_x_cubed, sqrt_2_over_pi)?;

                // Compute tanh(z)
                let tanh_z = create_tanh_tensor(z)?;

                // First part: 0.5 * (1 + tanh(z))
                let ones = create_ones_like_tensor(input)?;
                let one_plus_tanh_z = create_add_tensor(ones, tanh_z)?;
                let first_part = create_mul_scalar_tensor(one_plus_tanh_z, half)?;

                // Second part: x * 0.5 * sech²(z) * dz/dx
                // sech²(z) = 1 - tanh²(z)
                let tanh_z_squared = create_mul_tensor(tanh_z, tanh_z)?;
                let ones_for_sech = create_ones_like_tensor(input)?;
                let sech_squared_z = create_sub_tensor(ones_for_sech, tanh_z_squared)?;

                // dz/dx = √(2/π) * (1 + 3 * 0.044715 * x²)
                let three_coeff_scalar = Scalar::from_f32(3.0 * 0.044715, dtype);
                let three_coeff_x_squared = create_mul_scalar_tensor(x_squared, three_coeff_scalar)?;
                let ones_for_dz = create_ones_like_tensor(input)?;
                let one_plus_three_coeff_x_squared = create_add_tensor(ones_for_dz, three_coeff_x_squared)?;
                let dz_dx = create_mul_scalar_tensor(one_plus_three_coeff_x_squared, sqrt_2_over_pi)?;

                // Second part: x * 0.5 * sech²(z) * dz/dx
                let x_times_half = create_mul_scalar_tensor(input, half)?;
                let second_part_temp = create_mul_tensor(x_times_half, sech_squared_z)?;
                let second_part = create_mul_tensor(second_part_temp, dz_dx)?;

                // Total derivative = first_part + second_part
                let derivative = create_add_tensor(first_part, second_part)?;

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
                let one = Scalar::one(dtype);
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
                let ln_10 = Scalar::from_f32(core::f32::consts::LN_10, dtype);
                let x_ln_10 = create_mul_scalar_tensor(input, ln_10)?;
                let derivative = create_recip_tensor(x_ln_10)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Log2 => {
                // d/dx log2(x) = 1/(x * ln(2))
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let ln_2 = Scalar::from_f32(core::f32::consts::LN_2, dtype);
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
                let ln_10 = Scalar::from_f32(core::f32::consts::LN_10, dtype);
                let derivative = create_mul_scalar_tensor(_output, ln_10)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Exp2 => {
                // d/dx 2^x = 2^x * ln(2)
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let ln_2 = Scalar::from_f32(core::f32::consts::LN_2, dtype);
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
                let half = Scalar::from_f32(0.5, dtype);
                let recip_sqrt = create_recip_tensor(_output)?; // 1/sqrt(x)
                let derivative = create_mul_scalar_tensor(recip_sqrt, half)?; // 0.5/sqrt(x)
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
        }
    }
}

impl VjpCompute for UnaryLogicalOp {}

impl VjpCompute for UnaryScalarOp {
    fn compute_vjp_with_scalar(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        scalar: Scalar,
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
                Ok(vec![create_mul_scalar_tensor(grad_output, scalar)?])
            },
            UnaryScalarOp::DivScalar => {
                // d/dx (x / c) = 1/c
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                // Create 1/c scalar with proper dtype and check for division by zero
                let scalar_value = match scalar {
                    Scalar::F32(v) => {
                        if v.abs() < f32::EPSILON {
                            return Err(HoduError::InternalError(
                                "Division by zero in DivScalar gradient".to_string(),
                            ));
                        }
                        Scalar::from_f32(1.0 / v, dtype)
                    },
                    Scalar::F64(v) => {
                        if v.abs() < f64::EPSILON as f64 {
                            return Err(HoduError::InternalError(
                                "Division by zero in DivScalar gradient".to_string(),
                            ));
                        }
                        Scalar::from_f32(1.0 / v as f32, dtype)
                    },
                    Scalar::F16(v) => {
                        if v.abs() < half::f16::EPSILON {
                            return Err(HoduError::InternalError(
                                "Division by zero in DivScalar gradient".to_string(),
                            ));
                        }
                        Scalar::from_f32(1.0 / v.to_f32(), dtype)
                    },
                    Scalar::BF16(v) => {
                        if v.abs() < half::bf16::EPSILON {
                            return Err(HoduError::InternalError(
                                "Division by zero in DivScalar gradient".to_string(),
                            ));
                        }
                        Scalar::from_f32(1.0 / v.to_f32(), dtype)
                    },
                    Scalar::F8E4M3(v) => {
                        if v.abs() < float8::F8E4M3::EPSILON {
                            return Err(HoduError::InternalError(
                                "Division by zero in DivScalar gradient".to_string(),
                            ));
                        }
                        Scalar::from_f32(1.0 / v.to_f32(), dtype)
                    },
                    Scalar::F8E5M2(v) => {
                        if v.abs() < float8::F8E5M2::EPSILON {
                            return Err(HoduError::InternalError(
                                "Division by zero in DivScalar gradient".to_string(),
                            ));
                        }
                        Scalar::from_f32(1.0 / v.to_f32(), dtype)
                    },
                    _ => {
                        return Err(HoduError::InternalError(
                            "Unsupported scalar type in DivScalar gradient".to_string(),
                        ))
                    },
                };
                Ok(vec![create_mul_scalar_tensor(grad_output, scalar_value)?])
            },
            UnaryScalarOp::PowScalar => {
                // d/dx (x^c) = c * x^(c-1)
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let scalar_minus_one = match scalar {
                    Scalar::F32(v) => Scalar::from_f32(v - 1.0, dtype),
                    Scalar::F64(v) => Scalar::from_f32((v - 1.0) as f32, dtype),
                    Scalar::F16(v) => Scalar::from_f32(v.to_f32() - 1.0, dtype),
                    Scalar::BF16(v) => Scalar::from_f32(v.to_f32() - 1.0, dtype),
                    Scalar::F8E4M3(v) => Scalar::from_f32(v.to_f32() - 1.0, dtype),
                    Scalar::F8E5M2(v) => Scalar::from_f32(v.to_f32() - 1.0, dtype),
                    _ => Scalar::from_f32(0.0, dtype),
                };
                let x_power_scalar_minus_one = create_pow_scalar_tensor(input, scalar_minus_one)?;
                let derivative = create_mul_scalar_tensor(x_power_scalar_minus_one, scalar)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryScalarOp::MaximumScalar => {
                // d/dx max(x, scalar) = 1 if x >= scalar, else 0
                let mask = create_ge_scalar_tensor(input, scalar)?;
                Ok(vec![create_mul_tensor(grad_output, mask)?])
            },
            UnaryScalarOp::MinimumScalar => {
                // d/dx min(x, scalar) = 1 if x <= scalar, else 0
                let mask = create_le_scalar_tensor(input, scalar)?;
                Ok(vec![create_mul_tensor(grad_output, mask)?])
            },
            UnaryScalarOp::LeakyRelu => {
                // d/dx leaky_relu(x, alpha) = 1 if x > 0, else alpha
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let zero = Scalar::zero(dtype);
                let alpha = scalar;
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
                let zero = Scalar::zero(dtype);
                let alpha = scalar;
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
    fn compute_vjp(&self, inputs: &[TensorId], _output: TensorId, grad_output: TensorId) -> HoduResult<Vec<TensorId>> {
        match self {
            MatrixOp::Matmul => {
                // For matmul (ND batched with broadcasting):
                // dA = grad_output @ B^T
                // dB = A^T @ grad_output
                // Need to sum gradients to match original input shapes (handles broadcasting)
                let a = inputs[0];
                let b = inputs[1];

                let a_tensor = tensor_from_id(a);
                let b_tensor = tensor_from_id(b);
                let grad_tensor = tensor_from_id(grad_output);
                let a_shape = a_tensor.get_layout().get_shape().to_vec();
                let b_shape = b_tensor.get_layout().get_shape().to_vec();

                // dA = grad_output @ B^T
                let b_transposed = b_tensor.transpose(-2, -1)?;
                let grad_a_raw = grad_tensor.matmul(&b_transposed)?;

                // dB = A^T @ grad_output
                let a_transposed = a_tensor.transpose(-2, -1)?;
                let grad_b_raw = a_transposed.matmul(&grad_tensor)?;

                // Sum gradients to match original input shapes (handles broadcasting)
                let grad_a = create_sum_to_shape_tensor(grad_a_raw.id(), &a_shape)?;
                let grad_b = create_sum_to_shape_tensor(grad_b_raw.id(), &b_shape)?;

                Ok(vec![grad_a, grad_b])
            },
            MatrixOp::Dot => {
                // For simple dot (1D/2D only, no batching):
                // dA = grad_output @ B^T
                // dB = A^T @ grad_output
                // No broadcasting to handle, so no need for sum_to_shape
                let a = inputs[0];
                let b = inputs[1];

                let a_tensor = tensor_from_id(a);
                let b_tensor = tensor_from_id(b);
                let grad_tensor = tensor_from_id(grad_output);

                // dA = grad_output @ B^T
                let b_transposed = b_tensor.transpose(-2, -1)?;
                let grad_a = grad_tensor.dot(&b_transposed)?;

                // dB = A^T @ grad_output
                let a_transposed = a_tensor.transpose(-2, -1)?;
                let grad_b = a_transposed.dot(&grad_tensor)?;

                Ok(vec![grad_a.id(), grad_b.id()])
            },
        }
    }
}

impl VjpCompute for ReduceOp {
    fn compute_vjp_with_dims(
        &self,
        inputs: &[TensorId],
        output: TensorId,
        grad_output: TensorId,
        dims_scalars: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        let input = inputs[0];
        let input_tensor = tensor_from_id(input);
        let input_layout = input_tensor.get_layout();
        let input_shape = input_layout.get_shape();
        let dtype = input_tensor.get_dtype();

        // Convert dims_scalars to Vec<usize>
        let reduce_dims: Vec<usize> = if dims_scalars.is_empty() {
            // If no dims provided (fallback case), use all dimensions
            (0..input_shape.len()).collect()
        } else {
            dims_scalars.iter().map(|scalar| scalar.to_u32() as usize).collect()
        };

        match self {
            ReduceOp::Sum => {
                // d/dx sum(x) = 1 for all elements
                // Broadcast gradient back to input shape
                let grad_tensor = tensor_from_id(grad_output);
                let broadcasted = grad_tensor.broadcast(input_shape)?;
                Ok(vec![broadcasted.id()])
            },
            ReduceOp::Mean => {
                // d/dx mean(x) = 1/N where N is number of elements in reduced dimensions
                let reduce_elements = reduce_dims.iter().map(|&dim| input_shape[dim]).product::<usize>() as f32;
                let scale_scalar = Scalar::from_f32(1.0 / reduce_elements, dtype);

                let grad_tensor = tensor_from_id(grad_output);
                let scaled_grad = grad_tensor.mul_scalar(scale_scalar)?;
                let broadcasted = scaled_grad.broadcast(input_shape)?;
                Ok(vec![broadcasted.id()])
            },
            ReduceOp::Prod => {
                // d/dx prod(x) = prod(x) / x_i for each element x_i
                let output_tensor = tensor_from_id(output);
                let broadcasted_output = output_tensor.broadcast(input_shape)?;
                let derivative = broadcasted_output.div(&input_tensor)?;
                let grad_tensor = tensor_from_id(grad_output);
                let broadcasted_grad = grad_tensor.broadcast(input_shape)?;
                let result = derivative.mul(&broadcasted_grad)?;
                Ok(vec![result.id()])
            },
            ReduceOp::Norm => {
                // d/dx ||x||_2 = x / ||x||_2
                let output_tensor = tensor_from_id(output);
                let broadcasted_output = output_tensor.broadcast(input_shape)?;
                let derivative = input_tensor.div(&broadcasted_output)?;
                let grad_tensor = tensor_from_id(grad_output);
                let broadcasted_grad = grad_tensor.broadcast(input_shape)?;
                let result = derivative.mul(&broadcasted_grad)?;
                Ok(vec![result.id()])
            },
            ReduceOp::Var => {
                // d/dx var(x) = 2 * (x - mean(x)) / N where N is number of elements in reduced dimensions
                let mean_tensor = input_tensor.mean(&reduce_dims, false)?;
                let broadcasted_mean = mean_tensor.broadcast(input_shape)?;
                let diff = input_tensor.sub(&broadcasted_mean)?;
                let reduce_elements = reduce_dims.iter().map(|&dim| input_shape[dim]).product::<usize>() as f32;
                let scale = Scalar::from_f32(2.0 / reduce_elements, dtype);
                let derivative = diff.mul_scalar(scale)?;
                let grad_tensor = tensor_from_id(grad_output);
                let broadcasted_grad = grad_tensor.broadcast(input_shape)?;
                let result = derivative.mul(&broadcasted_grad)?;
                Ok(vec![result.id()])
            },
            ReduceOp::Std => {
                // d/dx std(x) = (x - mean(x)) / (N * std(x)) where N is number of elements in reduced dimensions
                let mean_tensor = input_tensor.mean(&reduce_dims, false)?;
                let broadcasted_mean = mean_tensor.broadcast(input_shape)?;
                let diff = input_tensor.sub(&broadcasted_mean)?;
                let output_tensor = tensor_from_id(output);
                let broadcasted_std = output_tensor.broadcast(input_shape)?;
                let reduce_elements = reduce_dims.iter().map(|&dim| input_shape[dim]).product::<usize>() as f32;
                let scale = Scalar::from_f32(1.0 / reduce_elements, dtype);
                let derivative = diff.div(&broadcasted_std)?.mul_scalar(scale)?;
                let grad_tensor = tensor_from_id(grad_output);
                let broadcasted_grad = grad_tensor.broadcast(input_shape)?;
                let result = derivative.mul(&broadcasted_grad)?;
                Ok(vec![result.id()])
            },
            ReduceOp::Max | ReduceOp::Min => {
                // These are marked as no-backprop, but if we need gradients:
                // Gradient flows only to the maximum/minimum elements
                let output_tensor = tensor_from_id(output);
                let broadcasted_output = output_tensor.broadcast(input_shape)?;
                let mask = input_tensor.eq(&broadcasted_output)?;
                let grad_tensor = tensor_from_id(grad_output);
                let broadcasted_grad = grad_tensor.broadcast(input_shape)?;
                let result = mask.mul(&broadcasted_grad)?;
                Ok(vec![result.id()])
            },
        }
    }
}

impl VjpCompute for ShapeOp {
    fn compute_vjp(&self, inputs: &[TensorId], _output: TensorId, grad_output: TensorId) -> HoduResult<Vec<TensorId>> {
        let input = inputs[0];
        let input_tensor = tensor_from_id(input);
        let input_layout = input_tensor.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_tensor = tensor_from_id(grad_output);
        let grad_layout = grad_tensor.get_layout();
        let grad_shape = grad_layout.get_shape();

        match self {
            ShapeOp::Reshape => {
                // Gradient just needs to be reshaped back to input shape
                let reshaped_grad = grad_tensor.reshape(input_shape)?;
                Ok(vec![reshaped_grad.id()])
            },
            ShapeOp::Flatten => {
                // Gradient needs to be reshaped back to original shape
                let reshaped_grad = grad_tensor.reshape(input_shape)?;
                Ok(vec![reshaped_grad.id()])
            },
            ShapeOp::Squeeze => {
                // Gradient needs to be unsqueezed back to original shape
                let unsqueezed_grad = grad_tensor.reshape(input_shape)?;
                Ok(vec![unsqueezed_grad.id()])
            },
            ShapeOp::Unsqueeze => {
                // Gradient needs to be squeezed back to original shape
                let squeezed_grad = grad_tensor.reshape(input_shape)?;
                Ok(vec![squeezed_grad.id()])
            },
            ShapeOp::Transpose => {
                // Reverse the transpose by finding which dimensions were swapped
                // Compare input and output shapes to determine the transpose dimensions
                if input_shape.len() != grad_shape.len() {
                    return Err(HoduError::InternalError(
                        "Input and gradient shapes must have same rank for transpose".to_string(),
                    ));
                }

                if input_shape.len() < 2 {
                    // For 0D or 1D tensors, transpose has no effect
                    Ok(vec![grad_output])
                } else {
                    // Find which dimensions were swapped by comparing shapes
                    let mut transpose_dims = None;

                    // Check common case: last two dimensions transposed
                    if input_shape.len() >= 2 {
                        let last_idx = input_shape.len() - 1;
                        let second_last_idx = input_shape.len() - 2;

                        if input_shape[last_idx] == grad_shape[second_last_idx]
                            && input_shape[second_last_idx] == grad_shape[last_idx]
                        {
                            // Check if all other dimensions match
                            let other_dims_match = input_shape[..second_last_idx]
                                .iter()
                                .zip(grad_shape[..second_last_idx].iter())
                                .all(|(a, b)| a == b);

                            if other_dims_match {
                                transpose_dims = Some((second_last_idx as i32, last_idx as i32));
                            }
                        }
                    }

                    // If we couldn't determine the transpose dimensions, fall back to (-2, -1)
                    let (dim1, dim2) = transpose_dims.unwrap_or((-2i32, -1i32));
                    let transposed_grad = grad_tensor.transpose(dim1, dim2)?;
                    Ok(vec![transposed_grad.id()])
                }
            },
            ShapeOp::Broadcast => {
                // Sum over the broadcasted dimensions to get back to original shape
                let mut result_grad = grad_tensor.clone();

                // Handle dimension differences (leading dimensions were added)
                if grad_shape.len() > input_shape.len() {
                    // Sum over leading dimensions that were added during broadcasting
                    let dims_to_sum: Vec<usize> = (0..(grad_shape.len() - input_shape.len())).collect();
                    for &dim in dims_to_sum.iter().rev() {
                        result_grad = result_grad.sum(&[dim], false)?;
                    }
                }

                // Handle size-1 dimensions that were broadcasted
                let current_layout = result_grad.get_layout();
                let current_shape = current_layout.get_shape();
                for (i, (&input_dim, &current_dim)) in input_shape.iter().zip(current_shape.iter()).enumerate() {
                    if input_dim == 1 && current_dim > 1 {
                        // This dimension was broadcasted from size 1, sum it back
                        result_grad = result_grad.sum(&[i], true)?; // keep_dim=true to maintain size 1
                    }
                }

                // Final reshape to ensure exact match
                let final_grad = result_grad.reshape(input_shape)?;
                Ok(vec![final_grad.id()])
            },
        }
    }
}

#[derive(Clone)]
struct TapeEntry {
    output_id: TensorId,
    op: Op,
    input_ids: Vec<TensorId>,
}

// Multiple tapes, one per context
#[cfg(feature = "std")]
static GRADIENT_TAPES: LazyLock<Mutex<HashMap<usize, Vec<TapeEntry>>>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert(0, Vec::new()); // Default context
    Mutex::new(map)
});

#[cfg(not(feature = "std"))]
static GRADIENT_TAPES: LazyLock<Mutex<HashMap<usize, Vec<TapeEntry>>>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert(0, Vec::new()); // Default context
    Mutex::new(map)
});

static CONTEXT_COUNTER: AtomicUsize = AtomicUsize::new(1); // Start from 1 (0 is default)

#[cfg(feature = "std")]
thread_local! {
    static CONTEXT_STACK: RefCell<Vec<usize>> = RefCell::new(vec![]);
}

#[cfg(not(feature = "std"))]
static CONTEXT_STACK: Mutex<Vec<usize>> = Mutex::new(Vec::new());

// Helper functions for context stack access
#[cfg(feature = "std")]
fn push_context(context_id: usize) {
    CONTEXT_STACK.with(|stack| {
        stack.borrow_mut().push(context_id);
    });
}

#[cfg(not(feature = "std"))]
fn push_context(context_id: usize) {
    CONTEXT_STACK.lock().push(context_id);
}

#[cfg(feature = "std")]
fn pop_context() {
    CONTEXT_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
}

#[cfg(not(feature = "std"))]
fn pop_context() {
    CONTEXT_STACK.lock().pop();
}

static IS_COMPUTING_GRADIENTS: AtomicBool = AtomicBool::new(false);

/// Auto-cleaning gradient context using RAII
pub struct GradientContext {
    context_id: usize,
}

impl GradientContext {
    pub fn new() -> Self {
        let context_id = CONTEXT_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Push to context stack
        push_context(context_id);

        // Initialize empty tape
        #[cfg(feature = "std")]
        {
            GRADIENT_TAPES.lock().unwrap().insert(context_id, Vec::new());
        }
        #[cfg(not(feature = "std"))]
        {
            GRADIENT_TAPES.lock().insert(context_id, Vec::new());
        }

        Self { context_id }
    }
}

impl Drop for GradientContext {
    fn drop(&mut self) {
        // Don't drop the default context (ID 0)
        if self.context_id == 0 {
            return;
        }

        // Pop from context stack
        pop_context();

        // Remove tape
        #[cfg(feature = "std")]
        {
            GRADIENT_TAPES.lock().unwrap().remove(&self.context_id);
        }
        #[cfg(not(feature = "std"))]
        {
            GRADIENT_TAPES.lock().remove(&self.context_id);
        }
    }
}

#[cfg(feature = "std")]
fn get_active_context() -> usize {
    CONTEXT_STACK.with(|stack| {
        stack.borrow().last().copied().unwrap_or(0) // Default to context 0
    })
}

#[cfg(not(feature = "std"))]
fn get_active_context() -> usize {
    CONTEXT_STACK.lock().last().copied().unwrap_or(0)
}

pub fn is_computing_gradients() -> bool {
    IS_COMPUTING_GRADIENTS.load(Ordering::Relaxed)
}

pub fn record_operation(output_id: TensorId, op: Op, input_ids: Vec<TensorId>) -> HoduResult<()> {
    if !is_computing_gradients() {
        let context_id = get_active_context();

        let mut tapes = {
            #[cfg(feature = "std")]
            {
                GRADIENT_TAPES.lock().map_err(|_| HoduError::GradientTapeCorrupted)?
            }
            #[cfg(not(feature = "std"))]
            {
                GRADIENT_TAPES.lock()
            }
        };

        if let Some(tape) = tapes.get_mut(&context_id) {
            tape.push(TapeEntry {
                output_id,
                op,
                input_ids,
            });
        }
    }
    Ok(())
}

// pub fn clear_all_tapes() {
//     #[cfg(feature = "std")]
//     {
//         if let Ok(mut tapes) = GRADIENT_TAPES.lock() {
//             for tape in tapes.values_mut() {
//                 tape.clear();
//             }
//         }
//     }
//     #[cfg(not(feature = "std"))]
//     {
//         let mut tapes = GRADIENT_TAPES.lock();
//         for tape in tapes.values_mut() {
//             tape.clear();
//         }
//     }
// }

/// Clear the default context tape (context 0)
pub fn clear_default_context_tape() {
    #[cfg(feature = "std")]
    {
        if let Ok(mut tapes) = GRADIENT_TAPES.lock() {
            if let Some(tape) = tapes.get_mut(&0) {
                tape.clear();
            }
        }
    }
    #[cfg(not(feature = "std"))]
    {
        let mut tapes = GRADIENT_TAPES.lock();
        if let Some(tape) = tapes.get_mut(&0) {
            tape.clear();
        }
    }
}

/// Clear tape for active context
pub fn clear_tape() {
    let context_id = get_active_context();

    #[cfg(feature = "std")]
    {
        if let Ok(mut tapes) = GRADIENT_TAPES.lock() {
            if let Some(tape) = tapes.get_mut(&context_id) {
                tape.clear();
            }
        }
    }
    #[cfg(not(feature = "std"))]
    {
        let mut tapes = GRADIENT_TAPES.lock();
        if let Some(tape) = tapes.get_mut(&context_id) {
            tape.clear();
        }
    }
}

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

        // Get tape from active context
        let context_id = get_active_context();
        let tape = {
            #[cfg(feature = "std")]
            {
                let tapes = GRADIENT_TAPES.lock().map_err(|_| HoduError::GradientTapeCorrupted)?;
                tapes.get(&context_id).cloned().unwrap_or_default()
            }
            #[cfg(not(feature = "std"))]
            {
                let tapes = GRADIENT_TAPES.lock();
                tapes.get(&context_id).cloned().unwrap_or_default()
            }
        };

        for entry in tape.iter().rev() {
            if let Some(&grad_output) = gradients.get(&entry.output_id) {
                for &input_id in &entry.input_ids {
                    if tensor::get(input_id).is_none() {
                        return Err(HoduError::TensorNotFound(input_id));
                    }
                }

                if tensor::get(grad_output).is_none() {
                    return Err(HoduError::TensorNotFound(grad_output));
                }

                let input_grads = compute_vjp_for_op(&entry.op, &entry.input_ids, entry.output_id, grad_output)?;

                for (input_id, grad_id) in entry.input_ids.iter().zip(input_grads.iter()) {
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
        Op::Binary(binary_op, _, _) => binary_op.compute_vjp(inputs, output, grad_output),
        Op::BinaryLogical(binary_logical_op, _, _) => binary_logical_op.compute_vjp(inputs, output, grad_output),
        Op::Cmp(cmp_op, _, _) => cmp_op.compute_vjp(inputs, output, grad_output),
        Op::CmpScalar(cmp_scalar_op, _, scalar) => {
            cmp_scalar_op.compute_vjp_with_scalar(inputs, output, grad_output, *scalar)
        },
        Op::Unary(unary_op, _) => unary_op.compute_vjp(inputs, output, grad_output),
        Op::UnaryLogical(unary_logical_op, _) => unary_logical_op.compute_vjp(inputs, output, grad_output),
        Op::UnaryScalar(unary_scalar_op, _, scalar) => {
            unary_scalar_op.compute_vjp_with_scalar(inputs, output, grad_output, *scalar)
        },
        Op::Matrix(matrix_op, _, _) => matrix_op.compute_vjp(inputs, output, grad_output),
        Op::Reduce(reduce_op, _, dims_scalars) => {
            reduce_op.compute_vjp_with_dims(inputs, output, grad_output, dims_scalars)
        },
        Op::Shape(shape_op, _) => shape_op.compute_vjp(inputs, output, grad_output),
        _ => Err(HoduError::VjpFunctionNotFound(format!("compute_vjp for {:?}", op))),
    }
}
