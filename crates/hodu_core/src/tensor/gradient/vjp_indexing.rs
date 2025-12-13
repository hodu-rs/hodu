use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    ops::{IndexingOp, OpParams},
    tensor::{tensor_from_id, Tensor, TensorId},
};

impl VjpCompute for IndexingOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        let dim = match op_params {
            OpParams::IndexSelect(p) => p.dim.to_usize(),
            OpParams::IndexPut(p) => p.dim.to_usize(),
            OpParams::Gather(p) => p.dim.to_usize(),
            OpParams::Scatter(p) => p.dim.to_usize(),
            OpParams::ScatterAdd(p) => p.dim.to_usize(),
            OpParams::ScatterMax(p) => p.dim.to_usize(),
            OpParams::ScatterMin(p) => p.dim.to_usize(),
            _ => {
                return Err(HoduError::VjpFunctionNotFound(
                    "IndexingOp requires IndexingParams".to_string(),
                ))
            },
        };

        match self {
            IndexingOp::IndexSelect => {
                // inputs: [self, indices]
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError("IndexSelect requires 2 inputs".to_string()));
                }

                let self_id = inputs[0];
                let indices_id = inputs[1];

                let self_tensor = tensor_from_id(self_id);
                let self_shape = self_tensor.shape();
                let dtype = self_tensor.dtype();

                // Create zero tensor with same shape as input
                let grad_self = Tensor::zeros(&self_shape, dtype)?;

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
                let self_shape = self_tensor.shape();
                let dtype = self_tensor.dtype();

                // Create zero tensor with same shape as input
                let grad_self = Tensor::zeros(&self_shape, dtype)?;

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
                let dtype = self_tensor.dtype();

                let indices_tensor = tensor_from_id(indices_id);
                let grad_tensor = tensor_from_id(grad_output);

                // Gradient w.r.t. self: everywhere except indexed positions
                // Create a mask: 1 everywhere, 0 at indexed positions
                let ones = Tensor::ones(self_tensor.shape(), dtype)?;
                let zeros_at_indices =
                    ones.index_put(dim, &indices_tensor, &Tensor::zeros(indices_tensor.shape(), dtype)?)?;
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
                let dtype = self_tensor.dtype();

                let indices_tensor = tensor_from_id(indices_id);
                let grad_tensor = tensor_from_id(grad_output);

                // Gradient w.r.t. self: everywhere except scattered positions
                // Create a mask: 1 everywhere, 0 at scattered positions
                let ones = Tensor::ones(self_tensor.shape(), dtype)?;
                let zeros_at_indices =
                    ones.scatter(dim, &indices_tensor, &Tensor::zeros(indices_tensor.shape(), dtype)?)?;
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
                let src_won_f = src_won.to_dtype(grad_tensor.dtype())?;
                let grad_src = scattered_values.mul(&src_won_f)?;

                // Gradient w.r.t. self: flows through where self won
                // Create mask where self won (opposite of src_won)
                let dtype = self_tensor.dtype();
                let ones = Tensor::ones(src_won_f.shape(), dtype)?;
                let self_won = ones.sub(&src_won_f)?;

                // Scatter the masked gradient back
                let grad_src_scattered = grad_tensor.gather(dim, &indices_tensor)?;
                let grad_self_at_indices = grad_src_scattered.mul(&self_won)?;

                let zeros = Tensor::zeros(self_tensor.shape(), dtype)?;
                let grad_self = zeros.scatter_add(dim, &indices_tensor, &grad_self_at_indices)?;

                // Add gradient from positions not affected by scatter
                let grad_self_final = grad_self.add(&grad_tensor)?;

                Ok(vec![grad_self_final.id(), grad_src.id()])
            },

            IndexingOp::Onehot => {
                // Onehot is non-differentiable
                Err(HoduError::VjpFunctionNotFound(
                    "Onehot operation does not support gradients".to_string(),
                ))
            },

            IndexingOp::Nonzero => {
                // Nonzero returns indices, which are not differentiable
                Err(HoduError::VjpFunctionNotFound(
                    "Nonzero operation does not support gradients".to_string(),
                ))
            },
            IndexingOp::Unique => {
                // Unique returns unique values/indices/counts, which are not differentiable
                Err(HoduError::VjpFunctionNotFound(
                    "Unique operation does not support gradients".to_string(),
                ))
            },
        }
    }
}
