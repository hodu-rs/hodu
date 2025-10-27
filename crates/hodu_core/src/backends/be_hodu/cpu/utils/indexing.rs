use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    types::layout::Layout,
};

pub fn index_select_map<T: Copy>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "index_select - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_offset = indices_layout.get_offset();
    let indices_size = indices_layout.get_size();
    let indices_strides = indices_layout.get_strides();
    let indices_shape = indices_layout.get_shape();

    // Output shape: replace the indexed dimension with indices size
    let mut output_shape = shape.to_vec();
    output_shape[dim] = indices_size;
    let output_size: usize = output_shape.iter().product();

    let mut result = Vec::with_capacity(output_size);
    let mut output_indices = vec![0; ndim];

    for _ in 0..output_size {
        // Calculate which index to use
        let index_pos = output_indices[dim];

        // Get the actual index value from indices tensor
        let mut indices_idx = indices_offset;
        if indices_layout.is_contiguous() {
            indices_idx += index_pos;
        } else {
            let mut tmp_pos = index_pos;
            for d in (0..indices_shape.len()).rev() {
                let i_dim = tmp_pos % indices_shape[d];
                indices_idx += i_dim * indices_strides[d];
                tmp_pos /= indices_shape[d];
            }
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        // Calculate flat index in source tensor
        let mut flat_index = offset;
        for i in 0..ndim {
            let actual_idx = if i == dim { idx as usize } else { output_indices[i] };
            flat_index += actual_idx * strides[i];
        }

        result.push(storage[flat_index]);

        // Increment output indices
        for i in (0..ndim).rev() {
            output_indices[i] += 1;
            if output_indices[i] < output_shape[i] {
                break;
            }
            output_indices[i] = 0;
        }
    }

    Ok(result)
}

pub fn index_put_map<T: Copy>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    values_storage: &[T],
    values_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "index_put - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_offset = indices_layout.get_offset();
    let indices_size = indices_layout.get_size();
    let indices_strides = indices_layout.get_strides();
    let indices_shape = indices_layout.get_shape();

    let values_shape = values_layout.get_shape();
    let values_strides = values_layout.get_strides();
    let values_offset = values_layout.get_offset();

    // Create a copy of the input storage
    let mut result = storage.to_vec();

    // Expected values shape: replace indexed dimension with indices size
    let mut expected_values_shape = shape.to_vec();
    expected_values_shape[dim] = indices_size;

    if values_shape != expected_values_shape {
        return Err(HoduError::InternalError(format!(
            "index_put - values shape {:?} does not match expected shape {:?}",
            values_shape, expected_values_shape
        )));
    }

    let values_size: usize = expected_values_shape.iter().product();
    let mut output_indices = vec![0; ndim];

    for _ in 0..values_size {
        // Calculate which index to use
        let index_pos = output_indices[dim];

        // Get the actual index value from indices tensor
        let mut indices_idx = indices_offset;
        if indices_layout.is_contiguous() {
            indices_idx += index_pos;
        } else {
            let mut tmp_pos = index_pos;
            for d in (0..indices_shape.len()).rev() {
                let i_dim = tmp_pos % indices_shape[d];
                indices_idx += i_dim * indices_strides[d];
                tmp_pos /= indices_shape[d];
            }
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        // Calculate flat index in result tensor
        let mut flat_index = offset;
        for i in 0..ndim {
            let actual_idx = if i == dim { idx as usize } else { output_indices[i] };
            flat_index += actual_idx * strides[i];
        }

        // Calculate flat index in values tensor
        let mut values_flat_index = values_offset;
        for i in 0..ndim {
            values_flat_index += output_indices[i] * values_strides[i];
        }

        // Put the value
        result[flat_index] = values_storage[values_flat_index];

        // Increment output indices
        for i in (0..ndim).rev() {
            output_indices[i] += 1;
            if output_indices[i] < expected_values_shape[i] {
                break;
            }
            output_indices[i] = 0;
        }
    }

    Ok(result)
}

pub fn gather_map<T: Copy>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "gather - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_shape = indices_layout.get_shape();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();
    let indices_ndim = indices_shape.len();

    if indices_ndim != ndim {
        return Err(HoduError::IncompatibleShapes {
            lhs: shape.to_vec(),
            rhs: indices_shape.to_vec(),
            op: "gather - indices must have same number of dimensions as input".to_string(),
        });
    }

    // Output has same shape as indices
    let output_size: usize = indices_shape.iter().product();
    let mut result = Vec::with_capacity(output_size);
    let mut output_indices = vec![0; ndim];

    for _ in 0..output_size {
        // Get index from indices tensor
        let mut indices_idx = indices_offset;
        for i in 0..ndim {
            indices_idx += output_indices[i] * indices_strides[i];
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        // Calculate flat index in source tensor
        let mut flat_index = offset;
        for i in 0..ndim {
            let actual_idx = if i == dim { idx as usize } else { output_indices[i] };
            flat_index += actual_idx * strides[i];
        }

        result.push(storage[flat_index]);

        // Increment output indices
        for i in (0..ndim).rev() {
            output_indices[i] += 1;
            if output_indices[i] < indices_shape[i] {
                break;
            }
            output_indices[i] = 0;
        }
    }

    Ok(result)
}

pub fn scatter_map<T: Copy>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    src_storage: &[T],
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "scatter - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_shape = indices_layout.get_shape();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let src_size = src_layout.get_size();

    // Check that indices and src have the same shape
    if indices_shape != src_shape {
        return Err(HoduError::IncompatibleShapes {
            lhs: indices_shape.to_vec(),
            rhs: src_shape.to_vec(),
            op: "scatter - indices and src must have the same shape".to_string(),
        });
    }

    // Create output by copying input
    let input_size = layout.get_size();
    let mut result = Vec::with_capacity(input_size);

    // Copy input to output
    if layout.is_contiguous() {
        result.extend_from_slice(&storage[offset..offset + input_size]);
    } else {
        let mut input_indices = vec![0; ndim];
        for _ in 0..input_size {
            let mut flat_index = offset;
            for i in 0..ndim {
                flat_index += input_indices[i] * strides[i];
            }
            result.push(storage[flat_index]);

            for i in (0..ndim).rev() {
                input_indices[i] += 1;
                if input_indices[i] < shape[i] {
                    break;
                }
                input_indices[i] = 0;
            }
        }
    }

    // Now scatter src values according to indices
    let mut scatter_indices = vec![0; ndim];
    for _ in 0..src_size {
        // Get index from indices tensor
        let mut indices_idx = indices_offset;
        for i in 0..ndim {
            indices_idx += scatter_indices[i] * indices_strides[i];
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        // Get source value
        let mut src_idx = src_offset;
        for i in 0..ndim {
            src_idx += scatter_indices[i] * src_strides[i];
        }
        let src_val = src_storage[src_idx];

        // Calculate flat index in output (contiguous result)
        let mut output_idx = 0;
        let mut stride = 1;
        for i in (0..ndim).rev() {
            let actual_idx = if i == dim { idx as usize } else { scatter_indices[i] };
            output_idx += actual_idx * stride;
            stride *= shape[i];
        }

        result[output_idx] = src_val;

        // Increment scatter indices
        for i in (0..ndim).rev() {
            scatter_indices[i] += 1;
            if scatter_indices[i] < src_shape[i] {
                break;
            }
            scatter_indices[i] = 0;
        }
    }

    Ok(result)
}

pub fn scatter_add_map<T: Copy + ops::Add<Output = T>>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    src_storage: &[T],
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "scatter_add - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_shape = indices_layout.get_shape();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let src_size = src_layout.get_size();

    if indices_shape != src_shape {
        return Err(HoduError::IncompatibleShapes {
            lhs: indices_shape.to_vec(),
            rhs: src_shape.to_vec(),
            op: "scatter_add - indices and src must have the same shape".to_string(),
        });
    }

    // Create output by copying input
    let input_size = layout.get_size();
    let mut result = Vec::with_capacity(input_size);

    if layout.is_contiguous() {
        result.extend_from_slice(&storage[offset..offset + input_size]);
    } else {
        let mut input_indices = vec![0; ndim];
        for _ in 0..input_size {
            let mut flat_index = offset;
            for i in 0..ndim {
                flat_index += input_indices[i] * strides[i];
            }
            result.push(storage[flat_index]);

            for i in (0..ndim).rev() {
                input_indices[i] += 1;
                if input_indices[i] < shape[i] {
                    break;
                }
                input_indices[i] = 0;
            }
        }
    }

    // Scatter-add src values according to indices
    let mut scatter_indices = vec![0; ndim];
    for _ in 0..src_size {
        let mut indices_idx = indices_offset;
        for i in 0..ndim {
            indices_idx += scatter_indices[i] * indices_strides[i];
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        let mut src_idx = src_offset;
        for i in 0..ndim {
            src_idx += scatter_indices[i] * src_strides[i];
        }
        let src_val = src_storage[src_idx];

        let mut output_idx = 0;
        let mut stride = 1;
        for i in (0..ndim).rev() {
            let actual_idx = if i == dim { idx as usize } else { scatter_indices[i] };
            output_idx += actual_idx * stride;
            stride *= shape[i];
        }

        result[output_idx] = result[output_idx] + src_val;

        for i in (0..ndim).rev() {
            scatter_indices[i] += 1;
            if scatter_indices[i] < src_shape[i] {
                break;
            }
            scatter_indices[i] = 0;
        }
    }

    Ok(result)
}

pub fn scatter_max_map<T: Copy + PartialOrd>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    src_storage: &[T],
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "scatter_max - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_shape = indices_layout.get_shape();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let src_size = src_layout.get_size();

    if indices_shape != src_shape {
        return Err(HoduError::IncompatibleShapes {
            lhs: indices_shape.to_vec(),
            rhs: src_shape.to_vec(),
            op: "scatter_max - indices and src must have the same shape".to_string(),
        });
    }

    // Create output by copying input
    let input_size = layout.get_size();
    let mut result = Vec::with_capacity(input_size);

    if layout.is_contiguous() {
        result.extend_from_slice(&storage[offset..offset + input_size]);
    } else {
        let mut input_indices = vec![0; ndim];
        for _ in 0..input_size {
            let mut flat_index = offset;
            for i in 0..ndim {
                flat_index += input_indices[i] * strides[i];
            }
            result.push(storage[flat_index]);

            for i in (0..ndim).rev() {
                input_indices[i] += 1;
                if input_indices[i] < shape[i] {
                    break;
                }
                input_indices[i] = 0;
            }
        }
    }

    // Scatter-max src values according to indices
    let mut scatter_indices = vec![0; ndim];
    for _ in 0..src_size {
        let mut indices_idx = indices_offset;
        for i in 0..ndim {
            indices_idx += scatter_indices[i] * indices_strides[i];
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        let mut src_idx = src_offset;
        for i in 0..ndim {
            src_idx += scatter_indices[i] * src_strides[i];
        }
        let src_val = src_storage[src_idx];

        let mut output_idx = 0;
        let mut stride = 1;
        for i in (0..ndim).rev() {
            let actual_idx = if i == dim { idx as usize } else { scatter_indices[i] };
            output_idx += actual_idx * stride;
            stride *= shape[i];
        }

        if src_val > result[output_idx] {
            result[output_idx] = src_val;
        }

        for i in (0..ndim).rev() {
            scatter_indices[i] += 1;
            if scatter_indices[i] < src_shape[i] {
                break;
            }
            scatter_indices[i] = 0;
        }
    }

    Ok(result)
}

pub fn scatter_min_map<T: Copy + PartialOrd>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    src_storage: &[T],
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "scatter_min - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_shape = indices_layout.get_shape();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let src_size = src_layout.get_size();

    if indices_shape != src_shape {
        return Err(HoduError::IncompatibleShapes {
            lhs: indices_shape.to_vec(),
            rhs: src_shape.to_vec(),
            op: "scatter_min - indices and src must have the same shape".to_string(),
        });
    }

    // Create output by copying input
    let input_size = layout.get_size();
    let mut result = Vec::with_capacity(input_size);

    if layout.is_contiguous() {
        result.extend_from_slice(&storage[offset..offset + input_size]);
    } else {
        let mut input_indices = vec![0; ndim];
        for _ in 0..input_size {
            let mut flat_index = offset;
            for i in 0..ndim {
                flat_index += input_indices[i] * strides[i];
            }
            result.push(storage[flat_index]);

            for i in (0..ndim).rev() {
                input_indices[i] += 1;
                if input_indices[i] < shape[i] {
                    break;
                }
                input_indices[i] = 0;
            }
        }
    }

    // Scatter-min src values according to indices
    let mut scatter_indices = vec![0; ndim];
    for _ in 0..src_size {
        let mut indices_idx = indices_offset;
        for i in 0..ndim {
            indices_idx += scatter_indices[i] * indices_strides[i];
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        let mut src_idx = src_offset;
        for i in 0..ndim {
            src_idx += scatter_indices[i] * src_strides[i];
        }
        let src_val = src_storage[src_idx];

        let mut output_idx = 0;
        let mut stride = 1;
        for i in (0..ndim).rev() {
            let actual_idx = if i == dim { idx as usize } else { scatter_indices[i] };
            output_idx += actual_idx * stride;
            stride *= shape[i];
        }

        if src_val < result[output_idx] {
            result[output_idx] = src_val;
        }

        for i in (0..ndim).rev() {
            scatter_indices[i] += 1;
            if scatter_indices[i] < src_shape[i] {
                break;
            }
            scatter_indices[i] = 0;
        }
    }

    Ok(result)
}
