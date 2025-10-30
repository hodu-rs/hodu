#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]

use crate::{compat::*, error::HoduResult, types::layout::Layout};

#[cfg(feature = "rayon")]
const PARALLEL_THRESHOLD: usize = 4096;

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
pub fn reduce_std<T>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
    unbiased: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)>
where
    T: Copy
        + Default
        + ops::Add<Output = T>
        + ops::Sub<Output = T>
        + ops::Mul<Output = T>
        + ops::Div<Output = T>
        + num_traits::NumCast
        + Send
        + Sync,
    T: num_traits::Float, // This provides sqrt
{
    use rayon::prelude::*;

    let (var_result, output_shape) = reduce_var(storage, layout, dims, keep_dim, unbiased)?;

    if var_result.len() >= PARALLEL_THRESHOLD {
        let std_result = var_result.par_iter().map(|&x| x.sqrt()).collect();
        Ok((std_result, output_shape))
    } else {
        let std_result = var_result.into_iter().map(|x| x.sqrt()).collect();
        Ok((std_result, output_shape))
    }
}

#[cfg(not(feature = "rayon"))]
pub fn reduce_std<T>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
    unbiased: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)>
where
    T: Copy
        + Default
        + ops::Add<Output = T>
        + ops::Sub<Output = T>
        + ops::Mul<Output = T>
        + ops::Div<Output = T>
        + num_traits::NumCast
        + Send
        + Sync,
    T: num_traits::Float, // This provides sqrt
{
    let (var_result, output_shape) = reduce_var(storage, layout, dims, keep_dim, unbiased)?;

    let std_result = var_result.into_iter().map(|x| x.sqrt()).collect();
    Ok((std_result, output_shape))
}

#[cfg(feature = "rayon")]
pub fn reduce_var<T>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
    unbiased: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)>
where
    T: Copy
        + Default
        + ops::Add<Output = T>
        + ops::Sub<Output = T>
        + ops::Mul<Output = T>
        + ops::Div<Output = T>
        + num_traits::NumCast
        + Send
        + Sync,
{
    use rayon::prelude::*;

    // First calculate mean
    let (mean_result, output_shape) = crate::be_hodu::cpu::utils::reduce_mean(storage, layout, dims, keep_dim)?;

    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    let reduce_dims: Vec<usize> = if dims.is_empty() {
        (0..ndim).collect()
    } else {
        dims.to_vec()
    };

    let output_size = output_shape.iter().product::<usize>();

    // Calculate the number of elements that were reduced
    let n = reduce_dims.iter().map(|&d| shape[d]).product::<usize>();
    let denominator = if unbiased && n > 1 { n - 1 } else { n };
    let denom_val = T::from(denominator).unwrap();

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
        let mean_val = mean_result[0];
        let total_elements = shape.iter().product::<usize>();

        let var_sum = (0..total_elements)
            .into_par_iter()
            .map(|i| {
                let indices = flat_to_indices(i, shape);

                let mut flat_index = offset;
                for (j, &idx) in indices.iter().enumerate() {
                    flat_index += idx * strides[j];
                }

                let diff = storage[flat_index] - mean_val;
                diff * diff
            })
            .reduce(|| T::default(), |a, b| a + b);

        let result = vec![var_sum / denom_val];
        return Ok((result, output_shape));
    }

    // Multi-dimensional case
    let result: Vec<T> = (0..output_size)
        .into_par_iter()
        .map(|output_idx| {
            let mean_val = mean_result[output_idx];

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

            let var_sum = (0..total_reduced)
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

                    let diff = storage[flat_index] - mean_val;
                    diff * diff
                })
                .reduce(|| T::default(), |a, b| a + b);

            var_sum / denom_val
        })
        .collect();

    Ok((result, output_shape))
}

#[cfg(not(feature = "rayon"))]
pub fn reduce_var<T>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
    unbiased: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)>
where
    T: Copy
        + Default
        + ops::Add<Output = T>
        + ops::Sub<Output = T>
        + ops::Mul<Output = T>
        + ops::Div<Output = T>
        + num_traits::NumCast
        + Send
        + Sync,
{
    // First calculate mean
    let (mean_result, output_shape) = crate::be_hodu::cpu::utils::reduce_mean(storage, layout, dims, keep_dim)?;

    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    let reduce_dims: Vec<usize> = if dims.is_empty() {
        (0..ndim).collect()
    } else {
        dims.to_vec()
    };

    let output_size = output_shape.iter().product::<usize>();

    // Calculate the number of elements that were reduced
    let n = reduce_dims.iter().map(|&d| shape[d]).product::<usize>();
    let denominator = if unbiased && n > 1 { n - 1 } else { n };
    let denom_val = T::from(denominator).unwrap();

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
        let mean_val = mean_result[0];
        let mut var_sum = T::default();
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

            let diff = storage[flat_index] - mean_val;
            var_sum = var_sum + diff * diff;
        }

        let result = vec![var_sum / denom_val];
        return Ok((result, output_shape));
    }

    // Multi-dimensional case
    let mut result = vec![T::default(); output_size];

    for output_idx in 0..output_size {
        let mean_val = mean_result[output_idx];
        let mut var_sum = T::default();

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

            let diff = storage[flat_index] - mean_val;
            var_sum = var_sum + diff * diff;
        }

        result[output_idx] = var_sum / denom_val;
    }

    Ok((result, output_shape))
}
