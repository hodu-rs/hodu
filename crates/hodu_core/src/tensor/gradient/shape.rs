use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{ShapeOp, ShapeScalarsOp},
    scalar::Scalar,
    tensor::{tensor_from_id, Tensor, TensorId},
};

impl VjpCompute for ShapeOp {
    fn compute_vjp(&self, inputs: &[TensorId], _output: TensorId, grad_output: TensorId) -> HoduResult<Vec<TensorId>> {
        let input = inputs[0];
        let input_tensor = tensor_from_id(input);
        let input_shape = input_tensor.shape();
        let grad_tensor = tensor_from_id(grad_output);
        let grad_shape = grad_tensor.shape();

        match self {
            ShapeOp::Reshape => {
                // Gradient just needs to be reshaped back to input shape
                let reshaped_grad = grad_tensor.reshape(&input_shape)?;
                Ok(vec![reshaped_grad.id()])
            },
            ShapeOp::Flatten => {
                // Gradient needs to be reshaped back to original shape
                let reshaped_grad = grad_tensor.reshape(&input_shape)?;
                Ok(vec![reshaped_grad.id()])
            },
            ShapeOp::Squeeze => {
                // Gradient needs to be unsqueezed back to original shape
                let unsqueezed_grad = grad_tensor.reshape(&input_shape)?;
                Ok(vec![unsqueezed_grad.id()])
            },
            ShapeOp::Unsqueeze => {
                // Gradient needs to be squeezed back to original shape
                let squeezed_grad = grad_tensor.reshape(&input_shape)?;
                Ok(vec![squeezed_grad.id()])
            },
            ShapeOp::Broadcast => {
                // Sum over the broadcasted dimensions to get back to original shape
                let mut result_grad = grad_tensor;

                // Handle dimension differences (leading dimensions were added)
                if grad_shape.dims().len() > input_shape.dims().len() {
                    // Sum over leading dimensions that were added during broadcasting
                    let dims_to_sum: Vec<usize> = (0..(grad_shape.dims().len() - input_shape.dims().len())).collect();
                    for &dim in dims_to_sum.iter().rev() {
                        result_grad = result_grad.sum(&[dim], false)?;
                    }
                }

                // Handle size-1 dimensions that were broadcasted
                let current_shape = result_grad.shape();
                for (i, (&input_dim, &current_dim)) in
                    input_shape.dims().iter().zip(current_shape.dims().iter()).enumerate()
                {
                    if input_dim == 1 && current_dim > 1 {
                        // This dimension was broadcasted from size 1, sum it back
                        result_grad = result_grad.sum(&[i], true)?; // keep_dim=true to maintain size 1
                    }
                }

                // Final reshape to ensure exact match
                let final_grad = result_grad.reshape(&input_shape)?;
                Ok(vec![final_grad.id()])
            },
            ShapeOp::Transpose => {
                // Reverse the transpose by finding which dimensions were swapped
                // Compare input and output shapes to determine the transpose dimensions
                if input_shape.dims().len() != grad_shape.dims().len() {
                    return Err(HoduError::InternalError(
                        "Input and gradient shapes must have same rank for transpose".to_string(),
                    ));
                }

                if input_shape.dims().len() < 2 {
                    // For 0D or 1D tensors, transpose has no effect
                    Ok(vec![grad_output])
                } else {
                    // Find which dimensions were swapped by comparing shapes
                    let mut transpose_dims = None;

                    // Check common case: last two dimensions transposed
                    if input_shape.dims().len() >= 2 {
                        let last_idx = input_shape.dims().len() - 1;
                        let second_last_idx = input_shape.dims().len() - 2;

                        if input_shape.dims()[last_idx] == grad_shape.dims()[second_last_idx]
                            && input_shape.dims()[second_last_idx] == grad_shape.dims()[last_idx]
                        {
                            // Check if all other dimensions match
                            let other_dims_match = input_shape.dims()[..second_last_idx]
                                .iter()
                                .zip(grad_shape.dims()[..second_last_idx].iter())
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
                if input_shape.dims().len() != grad_shape.dims().len() {
                    return Err(HoduError::InternalError(
                        "Input and gradient shapes must have same rank for permute".to_string(),
                    ));
                }

                let ndim = input_shape.dims().len();

                // Find the forward permutation by comparing input and grad shapes
                let mut forward_perm = vec![0; ndim];
                for i in 0..ndim {
                    for j in 0..ndim {
                        if input_shape.dims()[i] == grad_shape.dims()[j] {
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

impl VjpCompute for ShapeScalarsOp {
    fn compute_vjp_with_scalars(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        scalars: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        let input = inputs[0];
        let input_tensor = tensor_from_id(input);
        let input_shape = input_tensor.shape();
        let grad_tensor = tensor_from_id(grad_output);
        let grad_shape = grad_tensor.shape();

        match self {
            ShapeScalarsOp::Slice => {
                // Extract slice parameters from scalars: [dim, start, end_or_max, step]
                if scalars.len() < 4 {
                    return Err(HoduError::InternalError(
                        "Slice requires 4 scalar parameters".to_string(),
                    ));
                }

                let dim = scalars[0].to_i32() as usize;
                let start = scalars[1].to_i32() as isize;
                let end_value = scalars[2].to_i32();
                let end = if end_value == i32::MAX {
                    None
                } else {
                    Some(end_value as isize)
                };
                let step = scalars[3].to_i32() as isize;

                // Calculate slice indices
                let dim_size = input_shape.dims()[dim] as isize;
                let start_idx = if start < 0 { dim_size + start } else { start };
                let end_idx = end
                    .map(|e| if e < 0 { dim_size + e } else { e })
                    .unwrap_or(if step > 0 { dim_size } else { -1 });

                // Generate indices based on step direction
                let mut indices_vec = Vec::new();
                if step > 0 {
                    let mut idx = start_idx;
                    while idx < end_idx && idx < dim_size {
                        indices_vec.push(idx as i32);
                        idx += step;
                    }
                } else {
                    let mut idx = start_idx;
                    while idx > end_idx && idx >= 0 {
                        indices_vec.push(idx as i32);
                        idx += step;
                    }
                }

                // Validate that grad_shape matches expected slice output
                if grad_shape.dims().len() != input_shape.dims().len() {
                    return Err(HoduError::InternalError(
                        "Gradient shape rank must match input shape rank for slice".to_string(),
                    ));
                }
                if grad_shape.dims()[dim] as usize != indices_vec.len() {
                    return Err(HoduError::InternalError(format!(
                        "Gradient shape[{}]={} does not match expected slice size={}",
                        dim,
                        grad_shape.dims()[dim],
                        indices_vec.len()
                    )));
                }

                // Create indices tensor
                let indices_tensor = Tensor::new(indices_vec)?;

                // Create zero tensor with input shape
                let dtype = input_tensor.dtype();
                let grad_input = Tensor::zeros(&input_shape, dtype)?;

                // Use scatter_add to place gradients at the sliced positions
                let result = grad_input.scatter_add(dim, &indices_tensor, &grad_tensor)?;

                Ok(vec![result.id()])
            },
        }
    }
}
