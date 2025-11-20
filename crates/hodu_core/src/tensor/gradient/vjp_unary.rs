use super::{vjp_utils::*, VjpCompute};
use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    ops::{UnaryLogicalOp, UnaryOp, UnaryScalarOp},
    scalar::Scalar,
    tensor::{tensor_from_id, TensorId},
};
use num_traits::Float;

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
                let dtype = input_tensor.dtype();
                let zero = Scalar::zero(dtype);
                let mask = create_gt_scalar_tensor(input, zero)?;
                let grad_tensor = tensor_from_id(grad_output);
                let mask_f = tensor_from_id(mask).to_dtype(grad_tensor.dtype())?;
                Ok(vec![create_mul_tensor(grad_output, mask_f.id())?])
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
                let dtype = input_tensor.dtype();
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
                let dtype = input_tensor.dtype();
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
                let dtype = input_tensor.dtype();
                let ln_10 = Scalar::from_f32(core::f32::consts::LN_10, dtype);
                let x_ln_10 = create_mul_scalar_tensor(input, ln_10)?;
                let derivative = create_recip_tensor(x_ln_10)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Log2 => {
                // d/dx log2(x) = 1/(x * ln(2))
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.dtype();
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
                let dtype = input_tensor.dtype();
                let ln_10 = Scalar::from_f32(core::f32::consts::LN_10, dtype);
                let derivative = create_mul_scalar_tensor(_output, ln_10)?;
                Ok(vec![create_mul_tensor(grad_output, derivative)?])
            },
            UnaryOp::Exp2 => {
                // d/dx 2^x = 2^x * ln(2)
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.dtype();
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
                let dtype = input_tensor.dtype();
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
                let dtype = input_tensor.dtype();
                // Create 1/c scalar with proper dtype and check for division by zero
                let scalar_value = match scalar {
                    Scalar::F32(v) => {
                        if v.abs() < f32::EPSILON {
                            return Err(HoduError::GradientComputationFailed(
                                "Division by zero in DivScalar gradient".to_string(),
                            ));
                        }
                        Scalar::from_f32(1.0 / v, dtype)
                    },
                    #[cfg(feature = "f64")]
                    Scalar::F64(v) => {
                        if v.abs() < f64::EPSILON {
                            return Err(HoduError::GradientComputationFailed(
                                "Division by zero in DivScalar gradient".to_string(),
                            ));
                        }
                        Scalar::from_f32(1.0 / v as f32, dtype)
                    },
                    Scalar::F16(v) => {
                        if v.abs() < half::f16::EPSILON {
                            return Err(HoduError::GradientComputationFailed(
                                "Division by zero in DivScalar gradient".to_string(),
                            ));
                        }
                        Scalar::from_f32(1.0 / v.to_f32(), dtype)
                    },
                    Scalar::BF16(v) => {
                        if v.abs() < half::bf16::EPSILON {
                            return Err(HoduError::GradientComputationFailed(
                                "Division by zero in DivScalar gradient".to_string(),
                            ));
                        }
                        Scalar::from_f32(1.0 / v.to_f32(), dtype)
                    },
                    Scalar::F8E4M3(v) => {
                        if v.abs() < float8::F8E4M3::EPSILON {
                            return Err(HoduError::GradientComputationFailed(
                                "Division by zero in DivScalar gradient".to_string(),
                            ));
                        }
                        Scalar::from_f32(1.0 / v.to_f32(), dtype)
                    },
                    #[cfg(feature = "f8e5m2")]
                    Scalar::F8E5M2(v) => {
                        if v.abs() < float8::F8E5M2::EPSILON {
                            return Err(HoduError::GradientComputationFailed(
                                "Division by zero in DivScalar gradient".to_string(),
                            ));
                        }
                        Scalar::from_f32(1.0 / v.to_f32(), dtype)
                    },
                    _ => {
                        return Err(HoduError::GradientComputationFailed(
                            "Unsupported scalar type in DivScalar gradient".to_string(),
                        ))
                    },
                };
                Ok(vec![create_mul_scalar_tensor(grad_output, scalar_value)?])
            },
            UnaryScalarOp::PowScalar => {
                // d/dx (x^c) = c * x^(c-1)
                let input_tensor = tensor_from_id(input);
                let dtype = input_tensor.dtype();
                let scalar_minus_one = match scalar {
                    Scalar::F32(v) => Scalar::from_f32(v - 1.0, dtype),
                    #[cfg(feature = "f64")]
                    Scalar::F64(v) => Scalar::from_f32((v - 1.0) as f32, dtype),
                    Scalar::F16(v) => Scalar::from_f32(v.to_f32() - 1.0, dtype),
                    Scalar::BF16(v) => Scalar::from_f32(v.to_f32() - 1.0, dtype),
                    Scalar::F8E4M3(v) => Scalar::from_f32(v.to_f32() - 1.0, dtype),
                    #[cfg(feature = "f8e5m2")]
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
                let grad_tensor = tensor_from_id(grad_output);
                let mask_f = tensor_from_id(mask).to_dtype(grad_tensor.dtype())?;
                Ok(vec![create_mul_tensor(grad_output, mask_f.id())?])
            },
            UnaryScalarOp::MinimumScalar => {
                // d/dx min(x, scalar) = 1 if x <= scalar, else 0
                let mask = create_le_scalar_tensor(input, scalar)?;
                let grad_tensor = tensor_from_id(grad_output);
                let mask_f = tensor_from_id(mask).to_dtype(grad_tensor.dtype())?;
                Ok(vec![create_mul_tensor(grad_output, mask_f.id())?])
            },
            UnaryScalarOp::LeakyRelu => {
                // d/dx leaky_relu(x, alpha) = 1 if x > 0, else alpha
                let input_tensor = tensor_from_id(input);
                let grad_tensor = tensor_from_id(grad_output);
                let dtype = input_tensor.dtype();
                let zero = Scalar::zero(dtype);
                let alpha = scalar;
                let mask_pos = create_gt_scalar_tensor(input, zero)?;
                let mask_neg = create_le_scalar_tensor(input, zero)?;
                let mask_pos_f = tensor_from_id(mask_pos).to_dtype(grad_tensor.dtype())?;
                let mask_neg_f = tensor_from_id(mask_neg).to_dtype(grad_tensor.dtype())?;
                let alpha_grad = create_mul_scalar_tensor(mask_neg_f.id(), alpha)?;
                let total_grad = create_add_tensor(mask_pos_f.id(), alpha_grad)?;
                Ok(vec![create_mul_tensor(grad_output, total_grad)?])
            },
            UnaryScalarOp::Elu => {
                // d/dx elu(x, alpha) = 1 if x > 0, else alpha * exp(x)
                let input_tensor = tensor_from_id(input);
                let grad_tensor = tensor_from_id(grad_output);
                let dtype = input_tensor.dtype();
                let zero = Scalar::zero(dtype);
                let alpha = scalar;
                let mask_pos = create_gt_scalar_tensor(input, zero)?;
                let mask_neg = create_le_scalar_tensor(input, zero)?;
                let mask_pos_f = tensor_from_id(mask_pos).to_dtype(grad_tensor.dtype())?;
                let mask_neg_f = tensor_from_id(mask_neg).to_dtype(grad_tensor.dtype())?;
                let exp_input = create_exp_tensor(input)?;
                let alpha_exp = create_mul_scalar_tensor(exp_input, alpha)?;
                let neg_grad = create_mul_tensor(mask_neg_f.id(), alpha_exp)?;
                let total_grad = create_add_tensor(mask_pos_f.id(), neg_grad)?;
                Ok(vec![create_mul_tensor(grad_output, total_grad)?])
            },
            UnaryScalarOp::Prelu => {
                // d/dx prelu(x, alpha) = 1 if x > 0, else alpha
                let input_tensor = tensor_from_id(input);
                let grad_tensor = tensor_from_id(grad_output);
                let dtype = input_tensor.dtype();
                let zero = Scalar::zero(dtype);
                let alpha = scalar;
                let mask_pos = create_gt_scalar_tensor(input, zero)?;
                let mask_neg = create_le_scalar_tensor(input, zero)?;
                let mask_pos_f = tensor_from_id(mask_pos).to_dtype(grad_tensor.dtype())?;
                let mask_neg_f = tensor_from_id(mask_neg).to_dtype(grad_tensor.dtype())?;
                let ones_like_input = create_ones_like_tensor(input)?;
                let alpha_tensor = create_mul_scalar_tensor(ones_like_input, alpha)?;
                let neg_grad = create_mul_tensor(mask_neg_f.id(), alpha_tensor)?;
                let total_grad = create_add_tensor(mask_pos_f.id(), neg_grad)?;
                Ok(vec![create_mul_tensor(grad_output, total_grad)?])
            },
        }
    }
}
