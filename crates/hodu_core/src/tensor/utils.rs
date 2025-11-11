use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    tensor::Tensor,
};

pub fn broadcast_tensors2(a: &Tensor, b: &Tensor) -> HoduResult<(Tensor, Tensor)> {
    // Avoid cloning shapes by extracting only what we need
    let (a_dims, a_ndim, b_dims, b_ndim) = {
        let a_dims = a
            .with_shape(|s| s.dims().to_vec())
            .ok_or_else(|| HoduError::TensorNotFound(a.id()))?;
        let a_ndim = a_dims.len();
        let b_dims = b
            .with_shape(|s| s.dims().to_vec())
            .ok_or_else(|| HoduError::TensorNotFound(b.id()))?;
        let b_ndim = b_dims.len();
        (a_dims, a_ndim, b_dims, b_ndim)
    };

    let output_ndim = a_ndim.max(b_ndim);
    let mut output_dims = vec![0; output_ndim];

    // Compute output shape from right to left (broadcasting rules)
    for i in 0..output_ndim {
        let a_dim = if i < a_ndim { a_dims[a_ndim - 1 - i] } else { 1 };
        let b_dim = if i < b_ndim { b_dims[b_ndim - 1 - i] } else { 1 };

        if a_dim == 1 || b_dim == 1 || a_dim == b_dim {
            output_dims[output_ndim - 1 - i] = a_dim.max(b_dim);
        } else {
            // Only clone shapes for error messages
            return Err(HoduError::IncompatibleShapes {
                lhs: a.shape(),
                rhs: b.shape(),
                op: Op::Dummy,
            });
        }
    }

    let output_shape = crate::types::Shape::from(output_dims);
    let a_broadcasted = a.broadcast(&output_shape)?;
    let b_broadcasted = b.broadcast(&output_shape)?;

    Ok((a_broadcasted, b_broadcasted))
}

pub fn broadcast_tensors3(a: &Tensor, b: &Tensor, c: &Tensor) -> HoduResult<(Tensor, Tensor, Tensor)> {
    let (temp_ab, _) = broadcast_tensors2(a, b)?;
    let (temp_final, _) = broadcast_tensors2(&temp_ab, c)?;
    let final_shape = temp_final.shape();

    Ok((
        a.broadcast(&final_shape)?,
        b.broadcast(&final_shape)?,
        c.broadcast(&final_shape)?,
    ))
}
