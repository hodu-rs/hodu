use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    op_params::{OpParams, PaddingParams},
    ops::PaddingOp,
    tensor::{tensor_from_id, Tensor, TensorId},
    types::Shape,
};

impl VjpCompute for PaddingOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        let OpParams::Padding(PaddingParams { padding, .. }) = op_params else {
            return Err(HoduError::VjpFunctionNotFound(
                "PaddingOp requires PaddingParams".to_string(),
            ));
        };

        let input = inputs[0];
        let input_tensor = tensor_from_id(input);
        let input_shape = input_tensor.shape();

        let grad_tensor = tensor_from_id(grad_output);

        match self {
            PaddingOp::PadConstant => {
                // For constant padding, gradient is just a slice of grad_output
                // removing the padded regions
                let grad_input = slice_center(&grad_tensor, &input_shape, padding)?;
                Ok(vec![grad_input.id()])
            },
            PaddingOp::PadReflect => {
                // For reflect padding, gradient accumulates from reflected positions
                let grad_input = compute_reflect_grad(&grad_tensor, &input_shape, padding)?;
                Ok(vec![grad_input.id()])
            },
            PaddingOp::PadReplicate => {
                // For replicate padding, gradient accumulates at edge positions
                let grad_input = compute_replicate_grad(&grad_tensor, &input_shape, padding)?;
                Ok(vec![grad_input.id()])
            },
            PaddingOp::PadCircular => {
                // For circular padding, gradient wraps around
                let grad_input = compute_circular_grad(&grad_tensor, &input_shape, padding)?;
                Ok(vec![grad_input.id()])
            },
        }
    }
}

/// Slice out the center region from grad_output (removes padding)
fn slice_center(grad_output: &Tensor, input_shape: &Shape, padding: &[(usize, usize)]) -> HoduResult<Tensor> {
    let mut result = grad_output.clone();
    for (dim, (&(pad_before, _), &input_dim)) in padding.iter().zip(input_shape.dims().iter()).enumerate() {
        let start = pad_before as i64;
        let end = (pad_before + input_dim) as i64;
        result = result.slice(dim, start, Some(end), 1)?;
    }
    Ok(result)
}

/// Compute gradient for reflect padding
fn compute_reflect_grad(grad_output: &Tensor, input_shape: &Shape, padding: &[(usize, usize)]) -> HoduResult<Tensor> {
    // Start with center region (original input area)
    let mut grad_input = slice_center(grad_output, input_shape, padding)?;

    let input_dims = input_shape.dims();
    let ndim = input_dims.len();

    // For each dimension, add contributions from reflected regions
    for dim in 0..ndim {
        let (pad_before, pad_after) = padding[dim];
        let input_size = input_dims[dim];

        if pad_before == 0 && pad_after == 0 {
            continue;
        }

        // Left padding: reflected from positions [1..pad_before+1]
        // These map back to positions [1..min(pad_before, input_size-1)+1] in input
        if pad_before > 0 && input_size > 1 {
            let left_pad_grad = slice_dim(grad_output, dim, 0, pad_before, padding, input_dims)?;
            let flipped = left_pad_grad.flip(&[dim])?;

            // The reflection maps position i in left pad to position (pad_before - i) in input
            // After flip, position 0 maps to position 1, position 1 maps to position 2, etc.
            let target_len = pad_before.min(input_size - 1);
            let flipped_slice = if flipped.shape().dims()[dim] > target_len {
                slice_single_dim(
                    &flipped,
                    dim,
                    flipped.shape().dims()[dim] - target_len,
                    flipped.shape().dims()[dim],
                )?
            } else {
                flipped
            };

            let target = slice_single_dim(&grad_input, dim, 1, 1 + target_len)?;
            let added = target.add(&flipped_slice)?;
            grad_input = replace_single_dim(&grad_input, dim, 1, 1 + target_len, &added)?;
        }

        // Right padding: reflected from positions [input_size-2, input_size-pad_after-2]
        if pad_after > 0 && input_size > 1 {
            let output_size = grad_output.shape().dims()[dim];
            let right_pad_grad = slice_dim(
                grad_output,
                dim,
                output_size - pad_after,
                output_size,
                padding,
                input_dims,
            )?;
            let flipped = right_pad_grad.flip(&[dim])?;

            // After flip, these map to positions ending at input_size - 2
            let target_len = pad_after.min(input_size - 1);
            let start_pos = input_size - 1 - target_len;

            let flipped_slice = if flipped.shape().dims()[dim] > target_len {
                slice_single_dim(&flipped, dim, 0, target_len)?
            } else {
                flipped
            };

            let target = slice_single_dim(&grad_input, dim, start_pos, start_pos + target_len)?;
            let added = target.add(&flipped_slice)?;
            grad_input = replace_single_dim(&grad_input, dim, start_pos, start_pos + target_len, &added)?;
        }
    }

    Ok(grad_input)
}

/// Compute gradient for replicate padding
fn compute_replicate_grad(grad_output: &Tensor, input_shape: &Shape, padding: &[(usize, usize)]) -> HoduResult<Tensor> {
    // Start with center region
    let mut grad_input = slice_center(grad_output, input_shape, padding)?;

    let input_dims = input_shape.dims();
    let ndim = input_dims.len();

    // For each dimension, sum padding gradients into edge positions
    for dim in 0..ndim {
        let (pad_before, pad_after) = padding[dim];

        if pad_before == 0 && pad_after == 0 {
            continue;
        }

        // Left padding: all go to first element
        if pad_before > 0 {
            let left_pad_grad = slice_dim(grad_output, dim, 0, pad_before, padding, input_dims)?;
            let left_sum = left_pad_grad.sum(&[dim], true)?;

            let first = slice_single_dim(&grad_input, dim, 0, 1)?;
            let added = first.add(&left_sum)?;
            grad_input = replace_single_dim(&grad_input, dim, 0, 1, &added)?;
        }

        // Right padding: all go to last element
        if pad_after > 0 {
            let output_size = grad_output.shape().dims()[dim];
            let right_pad_grad = slice_dim(
                grad_output,
                dim,
                output_size - pad_after,
                output_size,
                padding,
                input_dims,
            )?;
            let right_sum = right_pad_grad.sum(&[dim], true)?;

            let input_size = input_dims[dim];
            let last = slice_single_dim(&grad_input, dim, input_size - 1, input_size)?;
            let added = last.add(&right_sum)?;
            grad_input = replace_single_dim(&grad_input, dim, input_size - 1, input_size, &added)?;
        }
    }

    Ok(grad_input)
}

/// Compute gradient for circular padding
fn compute_circular_grad(grad_output: &Tensor, input_shape: &Shape, padding: &[(usize, usize)]) -> HoduResult<Tensor> {
    // Start with center region
    let mut grad_input = slice_center(grad_output, input_shape, padding)?;

    let input_dims = input_shape.dims();
    let ndim = input_dims.len();

    // For each dimension, wrap padding gradients around
    for dim in 0..ndim {
        let (pad_before, pad_after) = padding[dim];
        let input_size = input_dims[dim];

        if pad_before == 0 && pad_after == 0 {
            continue;
        }

        // Left padding: comes from end of input (wraps around)
        // Left pad positions [0..pad_before] come from input positions [input_size - pad_before..input_size]
        if pad_before > 0 {
            let left_pad_grad = slice_dim(grad_output, dim, 0, pad_before, padding, input_dims)?;
            let start_pos = input_size - pad_before;

            let target = slice_single_dim(&grad_input, dim, start_pos, input_size)?;
            let added = target.add(&left_pad_grad)?;
            grad_input = replace_single_dim(&grad_input, dim, start_pos, input_size, &added)?;
        }

        // Right padding: comes from beginning of input
        // Right pad positions come from input positions [0..pad_after]
        if pad_after > 0 {
            let output_size = grad_output.shape().dims()[dim];
            let right_pad_grad = slice_dim(
                grad_output,
                dim,
                output_size - pad_after,
                output_size,
                padding,
                input_dims,
            )?;

            let target = slice_single_dim(&grad_input, dim, 0, pad_after)?;
            let added = target.add(&right_pad_grad)?;
            grad_input = replace_single_dim(&grad_input, dim, 0, pad_after, &added)?;
        }
    }

    Ok(grad_input)
}

/// Slice a single dimension of a tensor
fn slice_single_dim(tensor: &Tensor, dim: usize, start: usize, end: usize) -> HoduResult<Tensor> {
    tensor.slice(dim, start as i64, Some(end as i64), 1)
}

/// Slice from grad_output taking into account other dimensions are at center
fn slice_dim(
    grad_output: &Tensor,
    target_dim: usize,
    start: usize,
    end: usize,
    padding: &[(usize, usize)],
    input_dims: &[usize],
) -> HoduResult<Tensor> {
    let ndim = padding.len();
    let mut result = grad_output.clone();

    for dim in 0..ndim {
        if dim == target_dim {
            result = result.slice(dim, start as i64, Some(end as i64), 1)?;
        } else {
            let (pad_before, _) = padding[dim];
            let input_size = input_dims[dim];
            result = result.slice(dim, pad_before as i64, Some((pad_before + input_size) as i64), 1)?;
        }
    }

    Ok(result)
}

/// Replace a slice of the tensor along a single dimension
fn replace_single_dim(
    tensor: &Tensor,
    dim: usize,
    start: usize,
    end: usize,
    replacement: &Tensor,
) -> HoduResult<Tensor> {
    let shape = tensor.shape();
    let dim_size = shape.dims()[dim];

    // If replacing the entire dimension, just return replacement
    if start == 0 && end == dim_size {
        return Ok(replacement.clone());
    }

    // Build result by concatenating: [before, replacement, after]
    let mut parts: Vec<Tensor> = Vec::new();

    // Before slice
    if start > 0 {
        let before = slice_single_dim(tensor, dim, 0, start)?;
        parts.push(before);
    }

    // Replacement
    parts.push(replacement.clone());

    // After slice
    if end < dim_size {
        let after = slice_single_dim(tensor, dim, end, dim_size)?;
        parts.push(after);
    }

    // Concatenate along dim
    if parts.len() == 1 {
        Ok(parts.remove(0))
    } else {
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, dim)
    }
}
