use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    tensor::Tensor,
};

pub fn broadcast_tensors2(a: &Tensor, b: &Tensor) -> HoduResult<(Tensor, Tensor)> {
    let a_layout = a.get_layout();
    let b_layout = b.get_layout();
    let a_shape = a_layout.get_shape();
    let b_shape = b_layout.get_shape();
    let a_ndim = a_layout.get_ndim();
    let b_ndim = b_layout.get_ndim();

    let output_ndim = a_ndim.max(b_ndim);
    let mut output_shape = vec![0; output_ndim];

    // Compute output shape from right to left (broadcasting rules)
    for i in 0..output_ndim {
        let a_dim = if i < a_ndim { a_shape[a_ndim - 1 - i] } else { 1 };
        let b_dim = if i < b_ndim { b_shape[b_ndim - 1 - i] } else { 1 };

        if a_dim == 1 || b_dim == 1 || a_dim == b_dim {
            output_shape[output_ndim - 1 - i] = a_dim.max(b_dim);
        } else {
            return Err(HoduError::IncompatibleShapes {
                lhs: a_shape.to_vec(),
                rhs: b_shape.to_vec(),
                op: "broadcast_tensors".to_string(),
            });
        }
    }

    let a_broadcasted = a.broadcast(&output_shape)?;
    let b_broadcasted = b.broadcast(&output_shape)?;

    Ok((a_broadcasted, b_broadcasted))
}

pub fn broadcast_tensors3(a: &Tensor, b: &Tensor, c: &Tensor) -> HoduResult<(Tensor, Tensor, Tensor)> {
    let (temp_ab, _) = broadcast_tensors2(a, b)?;
    let (temp_final, _) = broadcast_tensors2(&temp_ab, c)?;
    let final_layout = temp_final.get_layout();
    let final_shape = final_layout.get_shape();

    Ok((
        a.broadcast(final_shape)?,
        b.broadcast(final_shape)?,
        c.broadcast(final_shape)?,
    ))
}
