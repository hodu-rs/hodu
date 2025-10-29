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

pub fn reduce_any<T>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<bool>, Vec<usize>)>
where
    T: Copy + Default + PartialEq + Send + Sync,
{
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    // Handle empty dims (reduce all)
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
            output_shape[dim] = 0; // Mark for removal
        }
    }
    if !keep_dim {
        output_shape.retain(|&size| size != 0);
        // If all dimensions are reduced, output_shape becomes empty (scalar)
    }

    let output_size = if output_shape.is_empty() {
        1 // Scalar has size 1
    } else {
        output_shape.iter().product::<usize>()
    };

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
        let total_elements = shape.iter().product::<usize>();

        #[cfg(feature = "rayon")]
        {
            let any_true = (0..total_elements).into_par_iter().any(|i| {
                let indices = flat_to_indices(i, shape);

                let mut flat_index = offset;
                for (j, &idx) in indices.iter().enumerate() {
                    flat_index += idx * strides[j];
                }

                storage[flat_index] != T::default()
            });

            let result = vec![any_true];
            return Ok((result, output_shape));
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = vec![false];

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

                // Check if value is truthy (non-zero/non-default)
                if storage[flat_index] != T::default() {
                    result[0] = true;
                    return Ok((result, output_shape));
                }
            }

            return Ok((result, output_shape));
        }
    }

    // Multi-dimensional reduction
    #[cfg(feature = "rayon")]
    {
        let result: Vec<bool> = (0..output_size)
            .into_par_iter()
            .map(|output_idx| {
                // Generate indices for the output tensor
                let output_indices = flat_to_indices(output_idx, &output_shape);

                // Map output indices back to input indices
                let mut input_indices = vec![0; ndim];
                let mut output_dim_idx = 0;
                for input_dim in 0..ndim {
                    if reduce_dims.contains(&input_dim) {
                        input_indices[input_dim] = 0; // Will iterate over this
                    } else {
                        if output_dim_idx < output_indices.len() {
                            input_indices[input_dim] = output_indices[output_dim_idx];
                        }
                        output_dim_idx += 1;
                    }
                }

                // Iterate over reduced dimensions
                let reduced_sizes: Vec<usize> = reduce_dims.iter().map(|&d| shape[d]).collect();
                let total_reduced = reduced_sizes.iter().product::<usize>();

                (0..total_reduced).any(|reduced_idx| {
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

                    storage[flat_index] != T::default()
                })
            })
            .collect();

        Ok((result, output_shape))
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut result = vec![false; output_size];

        for output_idx in 0..output_size {
            // Generate indices for the output tensor
            let mut output_indices = Vec::with_capacity(output_shape.len());
            let mut temp = output_idx;
            for &dim_size in output_shape.iter().rev() {
                output_indices.push(temp % dim_size);
                temp /= dim_size;
            }
            output_indices.reverse();

            // Map output indices back to input indices
            let mut input_indices = vec![0; ndim];
            let mut output_dim_idx = 0;
            for input_dim in 0..ndim {
                if reduce_dims.contains(&input_dim) {
                    input_indices[input_dim] = 0; // Will iterate over this
                } else {
                    if output_dim_idx < output_indices.len() {
                        input_indices[input_dim] = output_indices[output_dim_idx];
                    }
                    output_dim_idx += 1;
                }
            }

            // Iterate over reduced dimensions
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

                // Set the reduced dimension indices
                for (i, &dim) in reduce_dims.iter().enumerate() {
                    input_indices[dim] = temp_reduced_indices[i];
                }

                // Calculate flat index
                let mut flat_index = offset;
                for (j, &idx) in input_indices.iter().enumerate() {
                    flat_index += idx * strides[j];
                }

                // Check if value is truthy
                if storage[flat_index] != T::default() {
                    result[output_idx] = true;
                    break; // Early exit for any
                }
            }
        }

        Ok((result, output_shape))
    }
}

pub fn reduce_all<T>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<bool>, Vec<usize>)>
where
    T: Copy + Default + PartialEq + Send + Sync,
{
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    // Handle empty dims (reduce all)
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
            output_shape[dim] = 0; // Mark for removal
        }
    }
    if !keep_dim {
        output_shape.retain(|&size| size != 0);
        // If all dimensions are reduced, output_shape becomes empty (scalar)
    }

    let output_size = if output_shape.is_empty() {
        1 // Scalar has size 1
    } else {
        output_shape.iter().product::<usize>()
    };

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
        let total_elements = shape.iter().product::<usize>();

        #[cfg(feature = "rayon")]
        {
            let all_true = (0..total_elements).into_par_iter().all(|i| {
                let indices = flat_to_indices(i, shape);

                let mut flat_index = offset;
                for (j, &idx) in indices.iter().enumerate() {
                    flat_index += idx * strides[j];
                }

                storage[flat_index] != T::default()
            });

            let result = vec![all_true];
            return Ok((result, output_shape));
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = vec![true];

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

                // Check if value is falsy (zero/default)
                if storage[flat_index] == T::default() {
                    result[0] = false;
                    return Ok((result, output_shape));
                }
            }

            return Ok((result, output_shape));
        }
    }

    // Multi-dimensional reduction
    #[cfg(feature = "rayon")]
    {
        let result: Vec<bool> = (0..output_size)
            .into_par_iter()
            .map(|output_idx| {
                // Generate indices for the output tensor
                let output_indices = flat_to_indices(output_idx, &output_shape);

                // Map output indices back to input indices
                let mut input_indices = vec![0; ndim];
                let mut output_dim_idx = 0;
                for input_dim in 0..ndim {
                    if reduce_dims.contains(&input_dim) {
                        input_indices[input_dim] = 0; // Will iterate over this
                    } else {
                        if output_dim_idx < output_indices.len() {
                            input_indices[input_dim] = output_indices[output_dim_idx];
                        }
                        output_dim_idx += 1;
                    }
                }

                // Iterate over reduced dimensions
                let reduced_sizes: Vec<usize> = reduce_dims.iter().map(|&d| shape[d]).collect();
                let total_reduced = reduced_sizes.iter().product::<usize>();

                (0..total_reduced).all(|reduced_idx| {
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

                    storage[flat_index] != T::default()
                })
            })
            .collect();

        Ok((result, output_shape))
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut result = vec![true; output_size];

        for output_idx in 0..output_size {
            // Generate indices for the output tensor
            let mut output_indices = Vec::with_capacity(output_shape.len());
            let mut temp = output_idx;
            for &dim_size in output_shape.iter().rev() {
                output_indices.push(temp % dim_size);
                temp /= dim_size;
            }
            output_indices.reverse();

            // Map output indices back to input indices
            let mut input_indices = vec![0; ndim];
            let mut output_dim_idx = 0;
            for input_dim in 0..ndim {
                if reduce_dims.contains(&input_dim) {
                    input_indices[input_dim] = 0; // Will iterate over this
                } else {
                    if output_dim_idx < output_indices.len() {
                        input_indices[input_dim] = output_indices[output_dim_idx];
                    }
                    output_dim_idx += 1;
                }
            }

            // Iterate over reduced dimensions
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

                // Set the reduced dimension indices
                for (i, &dim) in reduce_dims.iter().enumerate() {
                    input_indices[dim] = temp_reduced_indices[i];
                }

                // Calculate flat index
                let mut flat_index = offset;
                for (j, &idx) in input_indices.iter().enumerate() {
                    flat_index += idx * strides[j];
                }

                // Check if value is falsy
                if storage[flat_index] == T::default() {
                    result[output_idx] = false;
                    break; // Early exit for all
                }
            }
        }

        Ok((result, output_shape))
    }
}
