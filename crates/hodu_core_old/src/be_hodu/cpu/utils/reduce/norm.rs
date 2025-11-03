#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]

use crate::{compat::*, error::HoduResult, types::layout::Layout};

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
pub fn reduce_norm<T>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)>
where
    T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + num_traits::Float + Send + Sync,
{
    use rayon::prelude::*;

    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    let reduce_dims: Vec<usize> = if dims.is_empty() {
        (0..ndim).collect()
    } else {
        dims.to_vec()
    };

    // Calculate output shape
    let mut output_shape = shape.to_vec();
    for &dim in &reduce_dims {
        if keep_dim {
            output_shape[dim] = 1;
        } else {
            output_shape[dim] = 0;
        }
    }
    if !keep_dim {
        output_shape.retain(|&size| size != 0);
        if output_shape.is_empty() {
            output_shape = vec![1];
        }
    }

    let output_size = output_shape.iter().product::<usize>();

    // Simple case: reduce all dimensions (L2 norm of entire tensor)
    if reduce_dims.len() == ndim {
        let total_elements = shape.iter().product::<usize>();

        let sum_squares = (0..total_elements)
            .into_par_iter()
            .map(|i| {
                let indices = flat_to_indices(i, shape);

                let mut flat_index = offset;
                for (j, &idx) in indices.iter().enumerate() {
                    flat_index += idx * strides[j];
                }

                let val = storage[flat_index];
                val * val
            })
            .reduce(|| T::default(), |a, b| a + b);

        let result = vec![sum_squares.sqrt()];
        return Ok((result, output_shape));
    }

    // Multi-dimensional case - compute L2 norm along specified dimensions
    let result: Vec<T> = (0..output_size)
        .into_par_iter()
        .map(|output_idx| {
            let output_indices = flat_to_indices(output_idx, &output_shape);

            let mut input_indices = vec![0; ndim];
            let mut output_dim_idx = 0;
            for input_dim in 0..ndim {
                if reduce_dims.contains(&input_dim) {
                    input_indices[input_dim] = 0;
                } else {
                    if output_dim_idx < output_indices.len() {
                        input_indices[input_dim] = output_indices[output_dim_idx];
                    }
                    output_dim_idx += 1;
                }
            }

            let reduced_sizes: Vec<usize> = reduce_dims.iter().map(|&d| shape[d]).collect();
            let total_reduced = reduced_sizes.iter().product::<usize>();

            let sum_squares = (0..total_reduced)
                .into_par_iter()
                .map(|reduced_idx| {
                    let mut temp_reduced_indices = Vec::new();
                    let mut temp = reduced_idx;
                    for &size in reduced_sizes.iter().rev() {
                        temp_reduced_indices.push(temp % size);
                        temp /= size;
                    }
                    temp_reduced_indices.reverse();

                    let mut local_input_indices = input_indices.clone();
                    for (i, &dim) in reduce_dims.iter().enumerate() {
                        local_input_indices[dim] = temp_reduced_indices[i];
                    }

                    let mut flat_index = offset;
                    for (j, &idx) in local_input_indices.iter().enumerate() {
                        flat_index += idx * strides[j];
                    }

                    let val = storage[flat_index];
                    val * val
                })
                .reduce(|| T::default(), |a, b| a + b);

            sum_squares.sqrt()
        })
        .collect();

    Ok((result, output_shape))
}

#[cfg(not(feature = "rayon"))]
pub fn reduce_norm<T>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)>
where
    T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + num_traits::Float + Send + Sync,
{
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    let reduce_dims: Vec<usize> = if dims.is_empty() {
        (0..ndim).collect()
    } else {
        dims.to_vec()
    };

    // Calculate output shape
    let mut output_shape = shape.to_vec();
    for &dim in &reduce_dims {
        if keep_dim {
            output_shape[dim] = 1;
        } else {
            output_shape[dim] = 0;
        }
    }
    if !keep_dim {
        output_shape.retain(|&size| size != 0);
        if output_shape.is_empty() {
            output_shape = vec![1];
        }
    }

    let output_size = output_shape.iter().product::<usize>();

    // Simple case: reduce all dimensions (L2 norm of entire tensor)
    if reduce_dims.len() == ndim {
        let mut sum_squares = T::default();
        let total_elements = shape.iter().product::<usize>();

        for i in 0..total_elements {
            let mut temp_indices = Vec::with_capacity(ndim);
            let mut temp = i;
            for &dim_size in shape.iter().rev() {
                temp_indices.push(temp % dim_size);
                temp /= dim_size;
            }
            temp_indices.reverse();

            let mut flat_index = offset;
            for (j, &idx) in temp_indices.iter().enumerate() {
                flat_index += idx * strides[j];
            }

            let val = storage[flat_index];
            sum_squares = sum_squares + val * val;
        }

        let result = vec![sum_squares.sqrt()];
        return Ok((result, output_shape));
    }

    // Multi-dimensional case - compute L2 norm along specified dimensions
    let mut result = vec![T::default(); output_size];

    for output_idx in 0..output_size {
        let mut sum_squares = T::default();

        let mut output_indices = Vec::with_capacity(output_shape.len());
        let mut temp = output_idx;
        for &dim_size in output_shape.iter().rev() {
            output_indices.push(temp % dim_size);
            temp /= dim_size;
        }
        output_indices.reverse();

        let mut input_indices = vec![0; ndim];
        let mut output_dim_idx = 0;
        for input_dim in 0..ndim {
            if reduce_dims.contains(&input_dim) {
                input_indices[input_dim] = 0;
            } else {
                if output_dim_idx < output_indices.len() {
                    input_indices[input_dim] = output_indices[output_dim_idx];
                }
                output_dim_idx += 1;
            }
        }

        let reduced_sizes: Vec<usize> = reduce_dims.iter().map(|&d| shape[d]).collect();
        let total_reduced = reduced_sizes.iter().product::<usize>();

        for reduced_idx in 0..total_reduced {
            let mut temp_reduced_indices = Vec::new();
            let mut temp = reduced_idx;
            for &size in reduced_sizes.iter().rev() {
                temp_reduced_indices.push(temp % size);
                temp /= size;
            }
            temp_reduced_indices.reverse();

            for (i, &dim) in reduce_dims.iter().enumerate() {
                input_indices[dim] = temp_reduced_indices[i];
            }

            let mut flat_index = offset;
            for (j, &idx) in input_indices.iter().enumerate() {
                flat_index += idx * strides[j];
            }

            let val = storage[flat_index];
            sum_squares = sum_squares + val * val;
        }

        result[output_idx] = sum_squares.sqrt();
    }

    Ok((result, output_shape))
}
