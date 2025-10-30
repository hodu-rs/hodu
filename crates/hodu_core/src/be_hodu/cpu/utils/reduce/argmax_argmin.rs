#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]

use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    types::layout::Layout,
};

// Helper function to convert flat index to multi-dimensional indices
#[allow(dead_code)]
fn flat_to_indices(mut flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        indices[i] = flat_idx % shape[i];
        flat_idx /= shape[i];
    }
    indices
}

#[cfg(feature = "rayon")]
pub fn reduce_argmax<T: Copy + PartialOrd + Send + Sync>(
    storage: &[T],
    layout: &Layout,
    dim: i32,
    keep_dim: bool,
) -> HoduResult<(Vec<i32>, Vec<usize>)> {
    use rayon::prelude::*;

    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    // Handle negative dimension
    let actual_dim = if dim < 0 {
        (ndim as i32 + dim) as usize
    } else {
        dim as usize
    };

    if actual_dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "argmax - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    // Calculate output shape
    let mut output_shape = shape.to_vec();
    if keep_dim {
        output_shape[actual_dim] = 1;
    } else {
        output_shape.remove(actual_dim);
        if output_shape.is_empty() {
            output_shape = vec![1];
        }
    }

    let output_size = output_shape.iter().product::<usize>();

    // Iterate over output positions
    let result: Vec<i32> = (0..output_size)
        .into_par_iter()
        .map(|output_idx| {
            let mut max_val = storage[offset];
            let mut max_index = 0i32;
            let mut first = true;

            // Generate output indices
            let output_indices = flat_to_indices(output_idx, &output_shape);

            // Map output indices to input indices
            let mut input_indices = vec![0; ndim];
            let mut out_idx = 0;
            for in_dim in 0..ndim {
                if in_dim == actual_dim {
                    input_indices[in_dim] = 0; // Will iterate over this
                } else if !keep_dim || out_idx < output_indices.len() {
                    input_indices[in_dim] = if keep_dim {
                        output_indices[out_idx]
                    } else if out_idx < output_indices.len() {
                        output_indices[out_idx]
                    } else {
                        0
                    };
                    out_idx += 1;
                }
            }

            // Iterate along the reduction dimension
            for i in 0..shape[actual_dim] {
                input_indices[actual_dim] = i;

                // Calculate flat index
                let mut flat_index = offset;
                for (j, &idx) in input_indices.iter().enumerate() {
                    flat_index += idx * strides[j];
                }

                let val = storage[flat_index];
                if first || val > max_val {
                    max_val = val;
                    max_index = i as i32;
                    first = false;
                }
            }

            max_index
        })
        .collect();

    Ok((result, output_shape))
}

#[cfg(not(feature = "rayon"))]
pub fn reduce_argmax<T: Copy + PartialOrd + Send + Sync>(
    storage: &[T],
    layout: &Layout,
    dim: i32,
    keep_dim: bool,
) -> HoduResult<(Vec<i32>, Vec<usize>)> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    // Handle negative dimension
    let actual_dim = if dim < 0 {
        (ndim as i32 + dim) as usize
    } else {
        dim as usize
    };

    if actual_dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "argmax - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    // Calculate output shape
    let mut output_shape = shape.to_vec();
    if keep_dim {
        output_shape[actual_dim] = 1;
    } else {
        output_shape.remove(actual_dim);
        if output_shape.is_empty() {
            output_shape = vec![1];
        }
    }

    let output_size = output_shape.iter().product::<usize>();

    // Iterate over output positions
    let mut result = vec![0i32; output_size];

    for output_idx in 0..output_size {
        let mut max_val = storage[offset];
        let mut max_index = 0i32;
        let mut first = true;

        // Generate output indices
        let mut output_indices = Vec::with_capacity(output_shape.len());
        let mut temp = output_idx;
        for &dim_size in output_shape.iter().rev() {
            output_indices.push(temp % dim_size);
            temp /= dim_size;
        }
        output_indices.reverse();

        // Map output indices to input indices
        let mut input_indices = vec![0; ndim];
        let mut out_idx = 0;
        for in_dim in 0..ndim {
            if in_dim == actual_dim {
                input_indices[in_dim] = 0; // Will iterate over this
            } else if !keep_dim || out_idx < output_indices.len() {
                input_indices[in_dim] = if keep_dim {
                    output_indices[out_idx]
                } else if out_idx < output_indices.len() {
                    output_indices[out_idx]
                } else {
                    0
                };
                out_idx += 1;
            }
        }

        // Iterate along the reduction dimension
        for i in 0..shape[actual_dim] {
            input_indices[actual_dim] = i;

            // Calculate flat index
            let mut flat_index = offset;
            for (j, &idx) in input_indices.iter().enumerate() {
                flat_index += idx * strides[j];
            }

            let val = storage[flat_index];
            if first || val > max_val {
                max_val = val;
                max_index = i as i32;
                first = false;
            }
        }

        result[output_idx] = max_index;
    }

    Ok((result, output_shape))
}

#[cfg(feature = "rayon")]
pub fn reduce_argmin<T: Copy + PartialOrd + Send + Sync>(
    storage: &[T],
    layout: &Layout,
    dim: i32,
    keep_dim: bool,
) -> HoduResult<(Vec<i32>, Vec<usize>)> {
    use rayon::prelude::*;

    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    // Handle negative dimension
    let actual_dim = if dim < 0 {
        (ndim as i32 + dim) as usize
    } else {
        dim as usize
    };

    if actual_dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "argmin - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    // Calculate output shape
    let mut output_shape = shape.to_vec();
    if keep_dim {
        output_shape[actual_dim] = 1;
    } else {
        output_shape.remove(actual_dim);
        if output_shape.is_empty() {
            output_shape = vec![1];
        }
    }

    let output_size = output_shape.iter().product::<usize>();

    // Iterate over output positions
    let result: Vec<i32> = (0..output_size)
        .into_par_iter()
        .map(|output_idx| {
            let mut min_val = storage[offset];
            let mut min_index = 0i32;
            let mut first = true;

            // Generate output indices
            let output_indices = flat_to_indices(output_idx, &output_shape);

            // Map output indices to input indices
            let mut input_indices = vec![0; ndim];
            let mut out_idx = 0;
            for in_dim in 0..ndim {
                if in_dim == actual_dim {
                    input_indices[in_dim] = 0; // Will iterate over this
                } else if !keep_dim || out_idx < output_indices.len() {
                    input_indices[in_dim] = if keep_dim {
                        output_indices[out_idx]
                    } else if out_idx < output_indices.len() {
                        output_indices[out_idx]
                    } else {
                        0
                    };
                    out_idx += 1;
                }
            }

            // Iterate along the reduction dimension
            for i in 0..shape[actual_dim] {
                input_indices[actual_dim] = i;

                // Calculate flat index
                let mut flat_index = offset;
                for (j, &idx) in input_indices.iter().enumerate() {
                    flat_index += idx * strides[j];
                }

                let val = storage[flat_index];
                if first || val < min_val {
                    min_val = val;
                    min_index = i as i32;
                    first = false;
                }
            }

            min_index
        })
        .collect();

    Ok((result, output_shape))
}

#[cfg(not(feature = "rayon"))]
pub fn reduce_argmin<T: Copy + PartialOrd + Send + Sync>(
    storage: &[T],
    layout: &Layout,
    dim: i32,
    keep_dim: bool,
) -> HoduResult<(Vec<i32>, Vec<usize>)> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    // Handle negative dimension
    let actual_dim = if dim < 0 {
        (ndim as i32 + dim) as usize
    } else {
        dim as usize
    };

    if actual_dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "argmin - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    // Calculate output shape
    let mut output_shape = shape.to_vec();
    if keep_dim {
        output_shape[actual_dim] = 1;
    } else {
        output_shape.remove(actual_dim);
        if output_shape.is_empty() {
            output_shape = vec![1];
        }
    }

    let output_size = output_shape.iter().product::<usize>();

    // Iterate over output positions
    let mut result = vec![0i32; output_size];

    for output_idx in 0..output_size {
        let mut min_val = storage[offset];
        let mut min_index = 0i32;
        let mut first = true;

        // Generate output indices
        let mut output_indices = Vec::with_capacity(output_shape.len());
        let mut temp = output_idx;
        for &dim_size in output_shape.iter().rev() {
            output_indices.push(temp % dim_size);
            temp /= dim_size;
        }
        output_indices.reverse();

        // Map output indices to input indices
        let mut input_indices = vec![0; ndim];
        let mut out_idx = 0;
        for in_dim in 0..ndim {
            if in_dim == actual_dim {
                input_indices[in_dim] = 0; // Will iterate over this
            } else {
                #[allow(clippy::collapsible_else_if)]
                #[allow(clippy::if_same_then_else)]
                if !keep_dim || out_idx < output_indices.len() {
                    input_indices[in_dim] = if keep_dim {
                        output_indices[out_idx]
                    } else if out_idx < output_indices.len() {
                        output_indices[out_idx]
                    } else {
                        0
                    };
                    out_idx += 1;
                }
            }
        }

        // Iterate along the reduction dimension
        for i in 0..shape[actual_dim] {
            input_indices[actual_dim] = i;

            // Calculate flat index
            let mut flat_index = offset;
            for (j, &idx) in input_indices.iter().enumerate() {
                flat_index += idx * strides[j];
            }

            let val = storage[flat_index];
            if first || val < min_val {
                min_val = val;
                min_index = i as i32;
                first = false;
            }
        }

        result[output_idx] = min_index;
    }

    Ok((result, output_shape))
}
