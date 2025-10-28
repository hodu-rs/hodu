mod utils;

use crate::{
    backends::op::{
        BinaryLogicalOp, BinaryOp, CmpOp, CmpScalarOp, ConcatOp, ConvOp, IndexingOp, MatrixOp, Op, ReduceOp, ShapeOp,
        SplitOp, UnaryLogicalOp, UnaryOp, UnaryScalarOp, WindowingOp,
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

    fn compute_vjp_with_scalars(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _scalars: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(format!(
            "compute_vjp_with_scalars not implemented"
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

    fn compute_vjp_with_split_info(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _params: &[Scalar],
        _output_index: usize,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(format!(
            "compute_vjp_with_split_info not implemented"
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
            UnaryOp::Silu => {
                // d/dx SiLU(x) = d/dx (x * sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                // = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                let sigmoid_x = create_sigmoid_tensor(input)?;
                let ones = create_ones_like_tensor(input)?;
                let one_minus_sigmoid = create_sub_tensor(ones, sigmoid_x)?;
                let x_times_one_minus_sigmoid = create_mul_tensor(input, one_minus_sigmoid)?;
                let one_plus_term = create_add_tensor(ones, x_times_one_minus_sigmoid)?;
                let derivative = create_mul_tensor(sigmoid_x, one_plus_term)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Swish => {
                // Swish is identical to SiLU, same derivative
                let sigmoid_x = create_sigmoid_tensor(input)?;
                let ones = create_ones_like_tensor(input)?;
                let one_minus_sigmoid = create_sub_tensor(ones, sigmoid_x)?;
                let x_times_one_minus_sigmoid = create_mul_tensor(input, one_minus_sigmoid)?;
                let one_plus_term = create_add_tensor(ones, x_times_one_minus_sigmoid)?;
                let derivative = create_mul_tensor(sigmoid_x, one_plus_term)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Mish => {
                // d/dx Mish(x) = d/dx (x * tanh(softplus(x)))
                // = tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)

                // softplus(x) = ln(1 + exp(x))
                let exp_x = create_exp_tensor(input)?;
                let ones = create_ones_like_tensor(input)?;
                let one_plus_exp_x = create_add_tensor(ones, exp_x)?;
                let softplus_x = create_ln_tensor(one_plus_exp_x)?;

                let tanh_softplus = create_tanh_tensor(softplus_x)?;
                let sigmoid_x = create_sigmoid_tensor(input)?;

                // sech²(softplus(x)) = 1 - tanh²(softplus(x))
                let tanh_softplus_squared = create_mul_tensor(tanh_softplus, tanh_softplus)?;
                let ones_for_sech = create_ones_like_tensor(input)?;
                let sech_squared_softplus = create_sub_tensor(ones_for_sech, tanh_softplus_squared)?;

                // x * sech²(softplus(x)) * sigmoid(x)
                let sech_times_sigmoid = create_mul_tensor(sech_squared_softplus, sigmoid_x)?;
                let x_times_term = create_mul_tensor(input, sech_times_sigmoid)?;

                // Total derivative = tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)
                let derivative = create_add_tensor(tanh_softplus, x_times_term)?;
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
            UnaryScalarOp::Prelu => {
                // d/dx prelu(x, alpha) = 1 if x > 0, else alpha
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.get_dtype();
                let zero = Scalar::zero(dtype);
                let alpha = scalar;
                let mask_pos = create_gt_scalar_tensor(input, zero)?;
                let mask_neg = create_le_scalar_tensor(input, zero)?;
                let ones_like_input = create_ones_like_tensor(input)?;
                let alpha_tensor = create_mul_scalar_tensor(ones_like_input, alpha)?;
                let neg_grad = create_mul_tensor(mask_neg, alpha_tensor)?;
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
            _ => Err(HoduError::InternalError(format!(
                "{:?} is not differentiable - cannot compute gradients for discrete index operations",
                self
            ))),
        }
    }
}

impl VjpCompute for ConcatOp {
    fn compute_vjp_with_dims(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        params: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        if params.is_empty() {
            return Err(HoduError::InternalError("Concat requires dim parameter".to_string()));
        }
        let dim = params[0];

        let sizes: Vec<usize> = inputs
            .iter()
            .map(|&id| {
                let tensor = tensor_from_id(id);
                tensor.get_layout().get_shape()[dim.to_u32() as usize]
            })
            .collect();

        let grad_tensor = tensor_from_id(grad_output);
        let grad_splits = grad_tensor.split(&sizes, dim)?;
        Ok(grad_splits.iter().map(|t| t.id()).collect())
    }
}

impl VjpCompute for SplitOp {
    fn compute_vjp_with_split_info(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        params: &[Scalar],
        output_index: usize,
    ) -> HoduResult<Vec<TensorId>> {
        if params.is_empty() {
            return Err(HoduError::InternalError("Split requires dim parameter".to_string()));
        }

        let input = inputs[0];
        let input_tensor = tensor_from_id(input);
        let input_layout = input_tensor.get_layout();
        let input_shape = input_layout.get_shape();
        let dtype = input_tensor.get_dtype();

        let dim = params[0].to_u32() as usize;
        let sizes: Vec<usize> = params[1..].iter().map(|s| s.to_u32() as usize).collect();

        // Create zero tensors for all splits
        let mut grad_pieces = Vec::new();
        for (idx, &size) in sizes.iter().enumerate() {
            if idx == output_index {
                // Use the actual gradient for this output
                grad_pieces.push(tensor_from_id(grad_output).clone());
            } else {
                // Create zeros for other outputs
                let mut piece_shape = input_shape.to_vec();
                piece_shape[dim] = size;
                let zeros = tensor::Tensor::zeros(&piece_shape, dtype)?;
                grad_pieces.push(zeros);
            }
        }

        // Concat all pieces back together
        let grad_refs: Vec<&tensor::Tensor> = grad_pieces.iter().collect();
        let result = tensor::Tensor::concat(&grad_refs, dim)?;
        Ok(vec![result.id()])
    }
}

impl VjpCompute for IndexingOp {
    fn compute_vjp_with_dims(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        params: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        if params.is_empty() {
            return Err(HoduError::InternalError(
                "Indexing operation requires dimension parameter".to_string(),
            ));
        }
        let dim = params[0].to_u32() as usize;

        match self {
            IndexingOp::IndexSelect => {
                // inputs: [self, indices]
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError("IndexSelect requires 2 inputs".to_string()));
                }

                let self_id = inputs[0];
                let indices_id = inputs[1];

                let self_tensor = tensor_from_id(self_id);
                let self_shape = self_tensor.get_layout().get_shape().to_vec();
                let dtype = self_tensor.get_dtype();

                // Create zero tensor with same shape as input
                let grad_self = tensor::Tensor::zeros(&self_shape, dtype)?;

                // Scatter the gradient back to the original positions
                let indices_tensor = tensor_from_id(indices_id);
                let grad_tensor = tensor_from_id(grad_output);
                let result = grad_self.scatter_add(dim, &indices_tensor, &grad_tensor)?;

                // IndexSelect doesn't have gradient w.r.t. indices
                Ok(vec![result.id()])
            },

            IndexingOp::Gather => {
                // inputs: [self, indices]
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError("Gather requires 2 inputs".to_string()));
                }

                let self_id = inputs[0];
                let indices_id = inputs[1];

                let self_tensor = tensor_from_id(self_id);
                let self_shape = self_tensor.get_layout().get_shape().to_vec();
                let dtype = self_tensor.get_dtype();

                // Create zero tensor with same shape as input
                let grad_self = tensor::Tensor::zeros(&self_shape, dtype)?;

                // Scatter the gradient back using scatter_add (accumulate for duplicate indices)
                let indices_tensor = tensor_from_id(indices_id);
                let grad_tensor = tensor_from_id(grad_output);
                let result = grad_self.scatter_add(dim, &indices_tensor, &grad_tensor)?;

                // Gather doesn't have gradient w.r.t. indices
                Ok(vec![result.id()])
            },

            IndexingOp::IndexPut => {
                // inputs: [self, values, indices]
                if inputs.len() != 3 {
                    return Err(HoduError::InternalError("IndexPut requires 3 inputs".to_string()));
                }

                let self_id = inputs[0];
                let _values_id = inputs[1];
                let indices_id = inputs[2];

                let self_tensor = tensor_from_id(self_id);
                let dtype = self_tensor.get_dtype();

                let indices_tensor = tensor_from_id(indices_id);
                let grad_tensor = tensor_from_id(grad_output);

                // Gradient w.r.t. self: everywhere except indexed positions
                // Create a mask: 1 everywhere, 0 at indexed positions
                let ones = tensor::Tensor::ones(self_tensor.get_layout().get_shape(), dtype)?;
                let zeros_at_indices = ones.index_put(
                    dim,
                    &indices_tensor,
                    &tensor::Tensor::zeros(indices_tensor.get_layout().get_shape(), dtype)?,
                )?;
                let grad_self = grad_tensor.mul(&zeros_at_indices)?;

                // Gradient w.r.t. values: gather from grad_output at indices
                let grad_values = grad_tensor.gather(dim, &indices_tensor)?;

                // IndexPut doesn't have gradient w.r.t. indices
                Ok(vec![grad_self.id(), grad_values.id()])
            },

            IndexingOp::Scatter => {
                // inputs: [self, src, indices]
                if inputs.len() != 3 {
                    return Err(HoduError::InternalError("Scatter requires 3 inputs".to_string()));
                }

                let self_id = inputs[0];
                let _src_id = inputs[1];
                let indices_id = inputs[2];

                let self_tensor = tensor_from_id(self_id);
                let dtype = self_tensor.get_dtype();

                let indices_tensor = tensor_from_id(indices_id);
                let grad_tensor = tensor_from_id(grad_output);

                // Gradient w.r.t. self: everywhere except scattered positions
                // Create a mask: 1 everywhere, 0 at scattered positions
                let ones = tensor::Tensor::ones(self_tensor.get_layout().get_shape(), dtype)?;
                let zeros_at_indices = ones.scatter(
                    dim,
                    &indices_tensor,
                    &tensor::Tensor::zeros(indices_tensor.get_layout().get_shape(), dtype)?,
                )?;
                let grad_self = grad_tensor.mul(&zeros_at_indices)?;

                // Gradient w.r.t. src: gather from grad_output at indices
                let grad_src = grad_tensor.gather(dim, &indices_tensor)?;

                // Scatter doesn't have gradient w.r.t. indices
                Ok(vec![grad_self.id(), grad_src.id()])
            },

            IndexingOp::ScatterAdd => {
                // inputs: [self, src, indices]
                if inputs.len() != 3 {
                    return Err(HoduError::InternalError("ScatterAdd requires 3 inputs".to_string()));
                }

                let indices_id = inputs[2];
                let indices_tensor = tensor_from_id(indices_id);
                let grad_tensor = tensor_from_id(grad_output);

                // Gradient w.r.t. self: full gradient (nothing is replaced, only added)
                let grad_self = grad_output;

                // Gradient w.r.t. src: gather from grad_output at indices
                let grad_src = grad_tensor.gather(dim, &indices_tensor)?;

                // ScatterAdd doesn't have gradient w.r.t. indices
                Ok(vec![grad_self, grad_src.id()])
            },

            IndexingOp::ScatterMax | IndexingOp::ScatterMin => {
                // inputs: [self, src, indices]
                if inputs.len() != 3 {
                    return Err(HoduError::InternalError(format!("{:?} requires 3 inputs", self)));
                }

                let self_id = inputs[0];
                let src_id = inputs[1];
                let indices_id = inputs[2];

                let self_tensor = tensor_from_id(self_id);
                let indices_tensor = tensor_from_id(indices_id);
                let src_tensor = tensor_from_id(src_id);
                let grad_tensor = tensor_from_id(grad_output);

                // Get the values at the scattered positions from output
                let scattered_values = grad_tensor.gather(dim, &indices_tensor)?;

                // Determine which values "won" (were selected)
                let src_won = match self {
                    IndexingOp::ScatterMax => {
                        // src won if src >= self at those positions
                        let self_at_indices = self_tensor.gather(dim, &indices_tensor)?;
                        src_tensor.ge(&self_at_indices)?
                    },
                    IndexingOp::ScatterMin => {
                        // src won if src <= self at those positions
                        let self_at_indices = self_tensor.gather(dim, &indices_tensor)?;
                        src_tensor.le(&self_at_indices)?
                    },
                    _ => unreachable!(),
                };

                // Gradient w.r.t. src: flows through where src won
                let grad_src = scattered_values.mul(&src_won)?;

                // Gradient w.r.t. self: flows through where self won
                // Create mask where self won (opposite of src_won)
                let dtype = self_tensor.get_dtype();
                let ones = tensor::Tensor::ones(src_won.get_layout().get_shape(), dtype)?;
                let self_won = ones.sub(&src_won)?;

                // Scatter the masked gradient back
                let grad_src_scattered = grad_tensor.gather(dim, &indices_tensor)?;
                let grad_self_at_indices = grad_src_scattered.mul(&self_won)?;

                let zeros = tensor::Tensor::zeros(self_tensor.get_layout().get_shape(), dtype)?;
                let grad_self = zeros.scatter_add(dim, &indices_tensor, &grad_self_at_indices)?;

                // Add gradient from positions not affected by scatter
                let grad_self_final = grad_self.add(&grad_tensor)?;

                Ok(vec![grad_self_final.id(), grad_src.id()])
            },
        }
    }
}

impl VjpCompute for ConvOp {
    fn compute_vjp_with_scalars(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        scalars: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            ConvOp::Conv1d => {
                // inputs: [input, weight]
                // Conv1d: input [N, Ci, L], weight [Co, Ci, K]
                // scalars: [batch_size, length_input, channels_output, channels_input, kernel_size, padding, stride, dilation]
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError("Conv1d requires 2 inputs".to_string()));
                }
                if scalars.len() < 8 {
                    return Err(HoduError::InternalError("Conv1d requires 8 parameters".to_string()));
                }

                let input = tensor_from_id(inputs[0]);
                let weight = tensor_from_id(inputs[1]);
                let grad_output_tensor = tensor_from_id(grad_output);

                let channels_output = scalars[2].to_u32() as usize;
                let channels_input = scalars[3].to_u32() as usize;
                let kernel_size = scalars[4].to_u32() as usize;
                let padding = scalars[5].to_u32() as usize;
                let stride = scalars[6].to_u32() as usize;
                let dilation = scalars[7].to_u32() as usize;

                // Gradient w.r.t. input: use transposed convolution
                // For conv1d: y = conv1d(x, w), grad_x = conv_transpose1d(grad_y, w, same params)
                // Weight for conv_transpose needs to be transposed: [Co, Ci, K] -> [Ci, Co, K]
                let weight_transposed = weight.transpose(0, 1)?; // Swap Co and Ci dimensions

                let grad_input =
                    grad_output_tensor.conv_transpose1d(&weight_transposed, stride, padding, 0, dilation)?;

                // Gradient w.r.t. weight: conv1d_grad_weight(input, grad_output, weight_shape)
                let weight_shape = vec![channels_output, channels_input, kernel_size];
                let grad_weight =
                    input.conv1d_grad_weight(&grad_output_tensor, &weight_shape, stride, padding, dilation)?;

                Ok(vec![grad_input.id(), grad_weight.id()])
            },
            ConvOp::Conv2d => {
                // inputs: [input, weight]
                // Conv2d: input [N, Ci, H, W], weight [Co, Ci, Kh, Kw]
                // scalars: [batch_size, input_height, input_width, kernel_height, kernel_width, channels_output, channels_input, padding, stride, dilation]
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError("Conv2d requires 2 inputs".to_string()));
                }
                if scalars.len() < 10 {
                    return Err(HoduError::InternalError("Conv2d requires 10 parameters".to_string()));
                }

                let input = tensor_from_id(inputs[0]);
                let weight = tensor_from_id(inputs[1]);
                let grad_output_tensor = tensor_from_id(grad_output);

                let kernel_height = scalars[3].to_u32() as usize;
                let kernel_width = scalars[4].to_u32() as usize;
                let channels_output = scalars[5].to_u32() as usize;
                let channels_input = scalars[6].to_u32() as usize;
                let padding = scalars[7].to_u32() as usize;
                let stride = scalars[8].to_u32() as usize;
                let dilation = scalars[9].to_u32() as usize;

                // Gradient w.r.t. input: use transposed convolution
                // For conv2d: y = conv2d(x, w), grad_x = conv_transpose2d(grad_y, w, same params)
                // Weight for conv_transpose needs to be transposed: [Co, Ci, Kh, Kw] -> [Ci, Co, Kh, Kw]
                let weight_transposed = weight.transpose(0, 1)?; // Swap Co and Ci dimensions

                let grad_input =
                    grad_output_tensor.conv_transpose2d(&weight_transposed, stride, padding, 0, dilation)?;

                // Gradient w.r.t. weight: conv2d_grad_weight(input, grad_output, weight_shape)
                let weight_shape_vec = vec![channels_output, channels_input, kernel_height, kernel_width];
                let grad_weight =
                    input.conv2d_grad_weight(&grad_output_tensor, &weight_shape_vec, stride, padding, dilation)?;

                Ok(vec![grad_input.id(), grad_weight.id()])
            },
            ConvOp::Conv3d => {
                // inputs: [input, weight]
                // Conv3d: input [N, Ci, D, H, W], weight [Co, Ci, Kd, Kh, Kw]
                // scalars: [batch_size, input_depth, input_height, input_width, kernel_depth, kernel_height, kernel_width, channels_output, channels_input, padding, stride, dilation]
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError("Conv3d requires 2 inputs".to_string()));
                }
                if scalars.len() < 12 {
                    return Err(HoduError::InternalError("Conv3d requires 12 parameters".to_string()));
                }

                let input = tensor_from_id(inputs[0]);
                let weight = tensor_from_id(inputs[1]);
                let grad_output_tensor = tensor_from_id(grad_output);

                let kernel_depth = scalars[4].to_u32() as usize;
                let kernel_height = scalars[5].to_u32() as usize;
                let kernel_width = scalars[6].to_u32() as usize;
                let channels_output = scalars[7].to_u32() as usize;
                let channels_input = scalars[8].to_u32() as usize;
                let padding = scalars[9].to_u32() as usize;
                let stride = scalars[10].to_u32() as usize;
                let dilation = scalars[11].to_u32() as usize;

                // Gradient w.r.t. input: use transposed convolution
                // For conv3d: y = conv3d(x, w), grad_x = conv_transpose3d(grad_y, w, same params)
                // Weight for conv_transpose needs to be transposed: [Co, Ci, Kd, Kh, Kw] -> [Ci, Co, Kd, Kh, Kw]
                let weight_transposed = weight.transpose(0, 1)?; // Swap Co and Ci dimensions

                let grad_input =
                    grad_output_tensor.conv_transpose3d(&weight_transposed, stride, padding, 0, dilation)?;

                // Gradient w.r.t. weight: conv3d_grad_weight(input, grad_output, weight_shape)
                let weight_shape = vec![
                    channels_output,
                    channels_input,
                    kernel_depth,
                    kernel_height,
                    kernel_width,
                ];
                let grad_weight =
                    input.conv3d_grad_weight(&grad_output_tensor, &weight_shape, stride, padding, dilation)?;

                Ok(vec![grad_input.id(), grad_weight.id()])
            },
            ConvOp::ConvTranspose1d => {
                // inputs: [input, weight]
                // ConvTranspose1d: input [N, Ci, L_in], weight [Ci, Co, K]
                // scalars: [batch_size, length_input, channels_output, channels_input, kernel_size, padding, output_padding, stride, dilation]
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose1d requires 2 inputs".to_string(),
                    ));
                }
                if scalars.len() < 9 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose1d requires 9 parameters".to_string(),
                    ));
                }

                let input = tensor_from_id(inputs[0]);
                let weight = tensor_from_id(inputs[1]);
                let grad_output_tensor = tensor_from_id(grad_output);

                let channels_output = scalars[2].to_u32() as usize;
                let channels_input = scalars[3].to_u32() as usize;
                let kernel_size = scalars[4].to_u32() as usize;
                let padding = scalars[5].to_u32() as usize;
                let _output_padding = scalars[6].to_u32() as usize;
                let stride = scalars[7].to_u32() as usize;
                let dilation = scalars[8].to_u32() as usize;

                // Gradient w.r.t. input: Use regular Conv1d
                // weight shape for conv1d is [Co, Ci, K], but we need to transpose channels
                let grad_input = grad_output_tensor.conv1d(&weight, stride, padding, dilation)?;

                // Gradient w.r.t. weight: conv_transpose1d_grad_weight(input, grad_output, weight_shape)
                let weight_shape = vec![channels_input, channels_output, kernel_size];
                let grad_weight = input.conv_transpose1d_grad_weight(
                    &grad_output_tensor,
                    &weight_shape,
                    stride,
                    padding,
                    _output_padding,
                    dilation,
                )?;

                Ok(vec![grad_input.id(), grad_weight.id()])
            },
            ConvOp::ConvTranspose2d => {
                // inputs: [input, weight]
                // ConvTranspose2d: input [N, Ci, H_in, W_in], weight [Ci, Co, Kh, Kw]
                // scalars: [batch_size, input_height, input_width, kernel_height, kernel_width, channels_output, channels_input, padding, output_padding, stride, dilation]
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose2d requires 2 inputs".to_string(),
                    ));
                }
                if scalars.len() < 11 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose2d requires 11 parameters".to_string(),
                    ));
                }

                let input = tensor_from_id(inputs[0]);
                let weight = tensor_from_id(inputs[1]);
                let grad_output_tensor = tensor_from_id(grad_output);

                let kernel_height = scalars[3].to_u32() as usize;
                let kernel_width = scalars[4].to_u32() as usize;
                let channels_output = scalars[5].to_u32() as usize;
                let channels_input = scalars[6].to_u32() as usize;
                let padding = scalars[7].to_u32() as usize;
                let output_padding = scalars[8].to_u32() as usize;
                let stride = scalars[9].to_u32() as usize;
                let dilation = scalars[10].to_u32() as usize;

                // Gradient w.r.t. input: Use regular Conv2d
                let grad_input = grad_output_tensor.conv2d(&weight, stride, padding, dilation)?;

                // Gradient w.r.t. weight: conv_transpose2d_grad_weight
                let weight_shape = vec![channels_input, channels_output, kernel_height, kernel_width];
                let grad_weight = input.conv_transpose2d_grad_weight(
                    &grad_output_tensor,
                    &weight_shape,
                    stride,
                    padding,
                    output_padding,
                    dilation,
                )?;

                Ok(vec![grad_input.id(), grad_weight.id()])
            },
            ConvOp::ConvTranspose3d => {
                // inputs: [input, weight]
                // ConvTranspose3d: input [N, Ci, D_in, H_in, W_in], weight [Ci, Co, Kd, Kh, Kw]
                // scalars: [batch_size, input_depth, input_height, input_width, kernel_depth, kernel_height, kernel_width, channels_output, channels_input, padding, output_padding, stride, dilation]
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose3d requires 2 inputs".to_string(),
                    ));
                }
                if scalars.len() < 13 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose3d requires 13 parameters".to_string(),
                    ));
                }

                let input = tensor_from_id(inputs[0]);
                let weight = tensor_from_id(inputs[1]);
                let grad_output_tensor = tensor_from_id(grad_output);

                let kernel_depth = scalars[4].to_u32() as usize;
                let kernel_height = scalars[5].to_u32() as usize;
                let kernel_width = scalars[6].to_u32() as usize;
                let channels_output = scalars[7].to_u32() as usize;
                let channels_input = scalars[8].to_u32() as usize;
                let padding = scalars[9].to_u32() as usize;
                let output_padding = scalars[10].to_u32() as usize;
                let stride = scalars[11].to_u32() as usize;
                let dilation = scalars[12].to_u32() as usize;

                // Gradient w.r.t. input: Use regular Conv3d
                let grad_input = grad_output_tensor.conv3d(&weight, stride, padding, dilation)?;

                // Gradient w.r.t. weight: conv_transpose3d_grad_weight
                let weight_shape = vec![
                    channels_input,
                    channels_output,
                    kernel_depth,
                    kernel_height,
                    kernel_width,
                ];
                let grad_weight = input.conv_transpose3d_grad_weight(
                    &grad_output_tensor,
                    &weight_shape,
                    stride,
                    padding,
                    output_padding,
                    dilation,
                )?;

                Ok(vec![grad_input.id(), grad_weight.id()])
            },
            // GradWeight operations don't need gradients (gradient of gradient not needed)
            ConvOp::Conv1dGradWeight
            | ConvOp::Conv2dGradWeight
            | ConvOp::Conv3dGradWeight
            | ConvOp::ConvTranspose1dGradWeight
            | ConvOp::ConvTranspose2dGradWeight
            | ConvOp::ConvTranspose3dGradWeight => Err(HoduError::InternalError(
                "Gradient of gradient weight operation not supported".to_string(),
            )),
        }
    }
}

impl VjpCompute for WindowingOp {
    fn compute_vjp_with_scalars(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        scalars: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            WindowingOp::ReduceWindow => {
                // Parameters: rank, window_shape[rank], strides[rank], padding[rank*2], reduction_type
                if scalars.is_empty() {
                    return Err(HoduError::InternalError("ReduceWindow requires parameters".to_string()));
                }

                let rank = scalars[0].to_u32() as usize;
                if scalars.len() < 1 + rank * 4 + 1 {
                    return Err(HoduError::InternalError(
                        "ReduceWindow requires sufficient parameters".to_string(),
                    ));
                }

                // Extract window_shape
                let window_shape: Vec<usize> = (0..rank).map(|i| scalars[1 + i].to_u32() as usize).collect();

                let reduction_type = scalars[1 + rank * 4].to_u32();

                let input = inputs[0];
                let input_tensor = tensor_from_id(input);
                let input_layout = input_tensor.get_layout();
                let input_shape = input_layout.get_shape();
                let dtype = input_tensor.get_dtype();

                // Check reduction type
                match reduction_type {
                    0 | 3 => {
                        // Max (0) or Min (3) - not differentiable
                        Err(HoduError::InternalError(
                            "Max and Min reductions are not differentiable (discrete operations)".to_string(),
                        ))
                    },
                    1 | 2 => {
                        // Mean (1) or Sum (2) - differentiable
                        // For pooling gradient, we need to broadcast the gradient back to input shape
                        let grad_tensor = tensor_from_id(grad_output);
                        let broadcasted_grad = grad_tensor.broadcast(input_shape)?;

                        if reduction_type == 1 {
                            // Mean: divide by window size
                            let window_size: usize = window_shape.iter().product();
                            let scale = Scalar::from_f32(1.0 / window_size as f32, dtype);
                            let scaled_grad = broadcasted_grad.mul_scalar(scale)?;
                            Ok(vec![scaled_grad.id()])
                        } else {
                            // Sum: just broadcast
                            Ok(vec![broadcasted_grad.id()])
                        }
                    },
                    _ => Err(HoduError::InternalError(format!(
                        "Unknown reduction type: {}",
                        reduction_type
                    ))),
                }
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
            ShapeOp::Permute => {
                // Reverse the permutation by finding the inverse permutation
                // If forward permutation is [a, b, c], then inverse is such that inverse[forward[i]] = i
                if input_shape.len() != grad_shape.len() {
                    return Err(HoduError::InternalError(
                        "Input and gradient shapes must have same rank for permute".to_string(),
                    ));
                }

                let ndim = input_shape.len();

                // Find the forward permutation by comparing input and grad shapes
                let mut forward_perm = vec![0; ndim];
                for i in 0..ndim {
                    for j in 0..ndim {
                        if input_shape[i] == grad_shape[j] {
                            // Check if this dimension is already used
                            let already_used = forward_perm[..i].contains(&j);
                            if !already_used {
                                forward_perm[i] = j;
                                break;
                            }
                        }
                    }
                }

                // Compute inverse permutation
                let mut inverse_perm = vec![0usize; ndim];
                for (i, &p) in forward_perm.iter().enumerate() {
                    inverse_perm[p] = i;
                }

                // Apply inverse permutation to gradient
                let permuted_grad = grad_tensor.permute(&inverse_perm)?;
                Ok(vec![permuted_grad.id()])
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
        Op::Reduce(reduce_op, _, _, dims_scalars) => {
            reduce_op.compute_vjp_with_dims(inputs, output, grad_output, dims_scalars)
        },
        Op::Concat(concat_op, _, params) => concat_op.compute_vjp_with_dims(inputs, output, grad_output, params),
        Op::Split(split_op, _, params, output_index) => {
            split_op.compute_vjp_with_split_info(inputs, output, grad_output, params, *output_index)
        },
        Op::Indexing(indexing_op, _, params) => indexing_op.compute_vjp_with_dims(inputs, output, grad_output, params),
        Op::Conv(conv_op, _, _, params) => conv_op.compute_vjp_with_scalars(inputs, output, grad_output, params),
        Op::Windowing(windowing_op, _, params) => {
            windowing_op.compute_vjp_with_scalars(inputs, output, grad_output, params)
        },
        Op::Shape(shape_op, _) => shape_op.compute_vjp(inputs, output, grad_output),
        _ => Err(HoduError::VjpFunctionNotFound(format!("compute_vjp for {:?}", op))),
    }
}
