use super::VjpCompute;
use crate::{
    error::HoduResult,
    ops::{LinalgOp, OpParams},
    tensor::{Tensor, TensorId},
};

impl VjpCompute for LinalgOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        output: TensorId,
        grad_output: TensorId,
        _op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            LinalgOp::Det => {
                // Gradient of determinant:
                // dL/dA = dL/d(det(A)) * det(A) * A^{-T}
                //       = grad_output * det_value * inv(A).transpose()
                //
                // For batched case:
                // grad_output: [...] (scalar per batch)
                // output (det): [...] (scalar per batch)
                // input A: [..., N, N]
                // result: [..., N, N]

                let input = Tensor::from_id(inputs[0]);
                let det_value = Tensor::from_id(output);
                let grad = Tensor::from_id(grad_output);

                // Compute A^{-1}
                let inv_a = input.inv()?;

                // Transpose the inverse: A^{-T}
                let inv_a_t = inv_a.transpose(-2, -1)?;

                // Scale by det(A) * grad_output
                // det_value and grad have shape [...], need to expand to [..., 1, 1]
                let scale = grad.mul(&det_value)?;
                let scale = scale.unsqueeze(-1)?.unsqueeze(-1)?;

                // grad_input = scale * inv_a_t
                let grad_input = inv_a_t.mul(&scale)?;

                Ok(vec![grad_input.id()])
            },
            LinalgOp::Inv => {
                // Gradient of matrix inverse:
                // dL/dA = -A^{-T} @ dL/d(A^{-1}) @ A^{-T}
                //       = -inv(A)^T @ grad_output @ inv(A)^T
                //
                // Since output = inv(A), we have:
                // dL/dA = -output^T @ grad_output @ output^T

                let inv_a = Tensor::from_id(output);
                let grad = Tensor::from_id(grad_output);

                // Transpose the output (inverse): A^{-T}
                let inv_a_t = inv_a.transpose(-2, -1)?;

                // Compute -inv(A)^T @ grad @ inv(A)^T
                let temp = inv_a_t.matmul(&grad)?;
                let grad_input = temp.matmul(&inv_a_t)?;
                let grad_input = grad_input.neg()?;

                Ok(vec![grad_input.id()])
            },
            LinalgOp::Trace => {
                // Gradient of trace:
                // trace(A) = sum of diagonal elements = A[0,0] + A[1,1] + ... + A[n-1,n-1]
                // d trace(A) / dA_ij = 1 if i == j, 0 otherwise
                // dL/dA = grad_output * I (identity matrix)
                //
                // For batched case:
                // grad_output: [...] (scalar per batch)
                // input A: [..., N, N]
                // result: [..., N, N]

                let input = Tensor::from_id(inputs[0]);
                let grad = Tensor::from_id(grad_output);

                // Get the matrix size from input shape
                let shape = input.shape();
                let ndim = shape.ndim();
                let n = shape.dims()[ndim - 1];

                // Create identity matrix with same batch dimensions
                let eye = Tensor::eye(n, input.dtype())?;

                // Expand grad to [..., 1, 1] for broadcasting
                let scale = grad.unsqueeze(-1)?.unsqueeze(-1)?;

                // grad_input = scale * I
                let grad_input = eye.mul(&scale)?;

                Ok(vec![grad_input.id()])
            },
        }
    }
}
