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

pub fn reduce_max<T: Copy + PartialOrd + Send + Sync>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)> {
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

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
        #[cfg(feature = "rayon")]
        {
            let total_elements = shape.iter().product::<usize>();
            let max_val = (0..total_elements)
                .into_par_iter()
                .map(|i| {
                    let temp_indices = flat_to_indices(i, shape);

                    let mut flat_index = offset;
                    for (j, &idx) in temp_indices.iter().enumerate() {
                        flat_index += idx * strides[j];
                    }

                    storage[flat_index]
                })
                .reduce(|| storage[offset], |a, b| if a > b { a } else { b });

            let result = vec![max_val];
            return Ok((result, output_shape));
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut max_val = storage[offset];
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
                if val > max_val {
                    max_val = val;
                }
            }

            let result = vec![max_val];
            return Ok((result, output_shape));
        }
    }

    // Multi-dimensional case
    #[cfg(feature = "rayon")]
    {
        let result: Vec<T> = (0..output_size)
            .into_par_iter()
            .map(|output_idx| {
                let mut max_val = storage[offset];
                let mut first = true;

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

                for reduced_idx in 0..total_reduced {
                    let temp_reduced_indices = flat_to_indices(reduced_idx, &reduced_sizes);

                    for (i, &dim) in reduce_dims.iter().enumerate() {
                        input_indices[dim] = temp_reduced_indices[i];
                    }

                    let mut flat_index = offset;
                    for (j, &idx) in input_indices.iter().enumerate() {
                        flat_index += idx * strides[j];
                    }

                    let val = storage[flat_index];
                    if first || val > max_val {
                        max_val = val;
                        first = false;
                    }
                }

                max_val
            })
            .collect();

        Ok((result, output_shape))
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut result = vec![storage[offset]; output_size];

        for output_idx in 0..output_size {
            let mut max_val = storage[offset];
            let mut first = true;

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
                if first || val > max_val {
                    max_val = val;
                    first = false;
                }
            }

            result[output_idx] = max_val;
        }

        Ok((result, output_shape))
    }
}

pub fn reduce_min<T: Copy + PartialOrd + Send + Sync>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)> {
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

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
        #[cfg(feature = "rayon")]
        {
            let total_elements = shape.iter().product::<usize>();
            let min_val = (0..total_elements)
                .into_par_iter()
                .map(|i| {
                    let temp_indices = flat_to_indices(i, shape);

                    let mut flat_index = offset;
                    for (j, &idx) in temp_indices.iter().enumerate() {
                        flat_index += idx * strides[j];
                    }

                    storage[flat_index]
                })
                .reduce(|| storage[offset], |a, b| if a < b { a } else { b });

            let result = vec![min_val];
            return Ok((result, output_shape));
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut min_val = storage[offset];
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
                if val < min_val {
                    min_val = val;
                }
            }

            let result = vec![min_val];
            return Ok((result, output_shape));
        }
    }

    // Multi-dimensional case
    #[cfg(feature = "rayon")]
    {
        let result: Vec<T> = (0..output_size)
            .into_par_iter()
            .map(|output_idx| {
                let mut min_val = storage[offset];
                let mut first = true;

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

                for reduced_idx in 0..total_reduced {
                    let temp_reduced_indices = flat_to_indices(reduced_idx, &reduced_sizes);

                    for (i, &dim) in reduce_dims.iter().enumerate() {
                        input_indices[dim] = temp_reduced_indices[i];
                    }

                    let mut flat_index = offset;
                    for (j, &idx) in input_indices.iter().enumerate() {
                        flat_index += idx * strides[j];
                    }

                    let val = storage[flat_index];
                    if first || val < min_val {
                        min_val = val;
                        first = false;
                    }
                }

                min_val
            })
            .collect();

        Ok((result, output_shape))
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut result = vec![storage[offset]; output_size];

        for output_idx in 0..output_size {
            let mut min_val = storage[offset];
            let mut first = true;

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
                if first || val < min_val {
                    min_val = val;
                    first = false;
                }
            }

            result[output_idx] = min_val;
        }

        Ok((result, output_shape))
    }
}
