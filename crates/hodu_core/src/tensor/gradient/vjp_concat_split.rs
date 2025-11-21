use super::VjpCompute;
use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    ops::{ConcatOp, ConcatParams, OpParams, SplitOp, SplitParams},
    tensor::{tensor_from_id, Tensor, TensorId},
    types::Shape,
};

impl VjpCompute for ConcatOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        let OpParams::Concat(ConcatParams { dim }) = op_params else {
            return Err(HoduError::VjpFunctionNotFound(
                "ConcatOp requires ConcatParams".to_string(),
            ));
        };
        let dim = *dim;

        let sizes: Vec<usize> = inputs
            .iter()
            .map(|&id| {
                let tensor = tensor_from_id(id);
                tensor.shape().dims()[dim.to_usize()] as usize
            })
            .collect();

        let grad_tensor = tensor_from_id(grad_output);
        let grad_splits = grad_tensor.split(&sizes, dim)?;
        Ok(grad_splits.iter().map(|t| t.id()).collect())
    }
}

impl VjpCompute for SplitOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        let OpParams::Split(SplitParams {
            dim,
            sizes,
            output_index,
        }) = op_params
        else {
            return Err(HoduError::VjpFunctionNotFound(
                "SplitOp requires SplitParams".to_string(),
            ));
        };

        let input = inputs[0];
        let input_tensor = tensor_from_id(input);
        let input_shape = input_tensor.shape();
        let dtype = input_tensor.dtype();

        let dim = dim.to_usize();
        let sizes: Vec<usize> = sizes.iter().map(|s| s.to_usize()).collect();
        let output_index = *output_index;

        // Create zero tensors for all splits
        let mut grad_pieces = Vec::new();
        for (idx, &size) in sizes.iter().enumerate() {
            if idx == output_index {
                // Use the actual gradient for this output
                grad_pieces.push(tensor_from_id(grad_output));
            } else {
                // Create zeros for other outputs
                let mut piece_shape_dims: Vec<usize> = input_shape.dims().to_vec();
                piece_shape_dims[dim] = size;
                let piece_shape = Shape::from(piece_shape_dims);
                let zeros = Tensor::zeros(&piece_shape, dtype)?;
                grad_pieces.push(zeros);
            }
        }

        // Concat all pieces back together
        let grad_refs: Vec<&Tensor> = grad_pieces.iter().collect();
        let result = Tensor::concat(&grad_refs, dim)?;
        Ok(vec![result.id()])
    }
}
