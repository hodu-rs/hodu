use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{ConcatOp, SplitOp},
    scalar::Scalar,
    tensor::{tensor_from_id, Tensor, TensorId},
    types::Shape,
};

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
                tensor.shape().dims()[dim.to_u32() as usize] as usize
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
        let input_shape = input_tensor.shape();
        let dtype = input_tensor.dtype();

        let dim = params[0].to_u32() as usize;
        let sizes: Vec<usize> = params[1..].iter().map(|s| s.to_u32() as usize).collect();

        // Create zero tensors for all splits
        let mut grad_pieces = Vec::new();
        for (idx, &size) in sizes.iter().enumerate() {
            if idx == output_index {
                // Use the actual gradient for this output
                grad_pieces.push(tensor_from_id(grad_output));
            } else {
                // Create zeros for other outputs
                let mut piece_shape_dims: Vec<u32> = input_shape.dims().to_vec();
                piece_shape_dims[dim] = size as u32;
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
