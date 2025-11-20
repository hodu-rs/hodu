use super::{vjp_utils::*, VjpCompute};
use crate::{
    error::HoduResult,
    layer::compat::*,
    ops::MatrixOp,
    tensor::{tensor_from_id, TensorId},
};

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
                let a_shape = a_tensor.shape();
                let b_shape = b_tensor.shape();

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
