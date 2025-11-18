use super::{utils::*, VjpCompute};
use crate::{
    error::HoduResult,
    layer::compat::*,
    ops::{BinaryLogicalOp, BinaryOp},
    scalar::Scalar,
    tensor::{tensor_from_id, TensorId},
};

impl VjpCompute for BinaryOp {
    fn compute_vjp(&self, inputs: &[TensorId], _output: TensorId, grad_output: TensorId) -> HoduResult<Vec<TensorId>> {
        // Get input shapes for gradient reduction
        let input_a = tensor_from_id(inputs[0]);
        let input_b = tensor_from_id(inputs[1]);
        let input_a_shape = input_a.shape();
        let input_b_shape = input_b.shape();

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
                let dtype = x_tensor.dtype();
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
                let a_ge_b = create_ge_tensor(inputs[0], inputs[1])?; // a >= b mask (bool)
                let b_gt_a = create_gt_tensor(inputs[1], inputs[0])?; // b > a mask (bool)
                let grad_tensor = tensor_from_id(grad_output);
                let a_ge_b_f = tensor_from_id(a_ge_b).to_dtype(grad_tensor.dtype())?;
                let b_gt_a_f = tensor_from_id(b_gt_a).to_dtype(grad_tensor.dtype())?;
                let grad_a_raw = create_mul_tensor(grad_output, a_ge_b_f.id())?;
                let grad_b_raw = create_mul_tensor(grad_output, b_gt_a_f.id())?;
                Ok(vec![
                    create_sum_to_shape_tensor(grad_a_raw, &input_a_shape)?,
                    create_sum_to_shape_tensor(grad_b_raw, &input_b_shape)?,
                ])
            },
            BinaryOp::Minimum => {
                // gradient goes to the smaller input
                let a_le_b = create_le_tensor(inputs[0], inputs[1])?; // a <= b mask (bool)
                let b_lt_a = create_lt_tensor(inputs[1], inputs[0])?; // b < a mask (bool)
                let grad_tensor = tensor_from_id(grad_output);
                let a_le_b_f = tensor_from_id(a_le_b).to_dtype(grad_tensor.dtype())?;
                let b_lt_a_f = tensor_from_id(b_lt_a).to_dtype(grad_tensor.dtype())?;
                let grad_a_raw = create_mul_tensor(grad_output, a_le_b_f.id())?;
                let grad_b_raw = create_mul_tensor(grad_output, b_lt_a_f.id())?;
                Ok(vec![
                    create_sum_to_shape_tensor(grad_a_raw, &input_a_shape)?,
                    create_sum_to_shape_tensor(grad_b_raw, &input_b_shape)?,
                ])
            },
        }
    }
}

impl VjpCompute for BinaryLogicalOp {}
