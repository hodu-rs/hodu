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

pub fn reduce_sum<T: Copy + Default + ops::Add<Output = T> + Send + Sync>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)> {
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
        if output_shape.is_empty() {
            output_shape = vec![1]; // Scalar result
        }
    }

    let output_size = output_shape.iter().product::<usize>();

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
        #[cfg(feature = "rayon")]
        {
            let total_elements = shape.iter().product::<usize>();
            let sum = (0..total_elements)
                .into_par_iter()
                .map(|i| {
                    let temp_indices = flat_to_indices(i, shape);

                    let mut flat_index = offset;
                    for (j, &idx) in temp_indices.iter().enumerate() {
                        flat_index += idx * strides[j];
                    }

                    storage[flat_index]
                })
                .reduce(|| T::default(), |a, b| a + b);

            let result = vec![sum];
            return Ok((result, output_shape));
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut sum = T::default();
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

                sum = sum + storage[flat_index];
            }

            let result = vec![sum];
            return Ok((result, output_shape));
        }
    }

    // Multi-dimensional reduction
    #[cfg(feature = "rayon")]
    {
        let result: Vec<T> = (0..output_size)
            .into_par_iter()
            .map(|output_idx| {
                let mut sum = T::default();

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

                for reduced_idx in 0..total_reduced {
                    let temp_reduced_indices = flat_to_indices(reduced_idx, &reduced_sizes);

                    // Set the reduced dimension indices
                    for (i, &dim) in reduce_dims.iter().enumerate() {
                        input_indices[dim] = temp_reduced_indices[i];
                    }

                    // Calculate flat index
                    let mut flat_index = offset;
                    for (j, &idx) in input_indices.iter().enumerate() {
                        flat_index += idx * strides[j];
                    }

                    sum = sum + storage[flat_index];
                }

                sum
            })
            .collect();

        Ok((result, output_shape))
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut result = vec![T::default(); output_size];

        for output_idx in 0..output_size {
            let mut sum = T::default();

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

                sum = sum + storage[flat_index];
            }

            result[output_idx] = sum;
        }

        Ok((result, output_shape))
    }
}

pub fn reduce_mean<T>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)>
where
    T: Copy + Default + ops::Add<Output = T> + ops::Div<Output = T> + Send + Sync,
    T: num_traits::NumCast,
{
    let (sum_result, output_shape) = reduce_sum(storage, layout, dims, keep_dim)?;

    // Calculate the number of elements that were reduced
    let shape = layout.get_shape();
    let reduce_dims: Vec<usize> = if dims.is_empty() {
        (0..shape.len()).collect()
    } else {
        dims.to_vec()
    };

    let count = reduce_dims.iter().map(|&d| shape[d]).product::<usize>();
    let count_val = T::from(count).unwrap();

    #[cfg(feature = "rayon")]
    {
        let mean_result = sum_result.par_iter().map(|&x| x / count_val).collect();
        Ok((mean_result, output_shape))
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mean_result = sum_result.into_iter().map(|x| x / count_val).collect();
        Ok((mean_result, output_shape))
    }
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

pub fn reduce_prod<T: Copy + ops::Mul<Output = T> + Send + Sync>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)>
where
    T: num_traits::One,
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

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
        #[cfg(feature = "rayon")]
        {
            let total_elements = shape.iter().product::<usize>();
            let prod_val = (0..total_elements)
                .into_par_iter()
                .map(|i| {
                    let temp_indices = flat_to_indices(i, shape);

                    let mut flat_index = offset;
                    for (j, &idx) in temp_indices.iter().enumerate() {
                        flat_index += idx * strides[j];
                    }

                    storage[flat_index]
                })
                .reduce(|| T::one(), |a, b| a * b);

            let result = vec![prod_val];
            return Ok((result, output_shape));
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut prod_val = T::one();
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

                prod_val = prod_val * storage[flat_index];
            }

            let result = vec![prod_val];
            return Ok((result, output_shape));
        }
    }

    // Multi-dimensional case
    #[cfg(feature = "rayon")]
    {
        let result: Vec<T> = (0..output_size)
            .into_par_iter()
            .map(|output_idx| {
                let mut prod_val = T::one();

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

                    prod_val = prod_val * storage[flat_index];
                }

                prod_val
            })
            .collect();

        Ok((result, output_shape))
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut result = vec![T::one(); output_size];

        for output_idx in 0..output_size {
            let mut prod_val = T::one();

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

                prod_val = prod_val * storage[flat_index];
            }

            result[output_idx] = prod_val;
        }

        Ok((result, output_shape))
    }
}

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

    #[cfg(feature = "rayon")]
    {
        let std_result = var_result.par_iter().map(|&x| x.sqrt()).collect();
        Ok((std_result, output_shape))
    }

    #[cfg(not(feature = "rayon"))]
    {
        let std_result = var_result.into_iter().map(|x| x.sqrt()).collect();
        Ok((std_result, output_shape))
    }
}

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
    let (mean_result, output_shape) = reduce_mean(storage, layout, dims, keep_dim)?;

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

        #[cfg(feature = "rayon")]
        {
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

        #[cfg(not(feature = "rayon"))]
        {
            let mut var_sum = T::default();

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
    }

    // Multi-dimensional case
    #[cfg(feature = "rayon")]
    {
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

        return Ok((result, output_shape));
    }

    #[cfg(not(feature = "rayon"))]
    {
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

        return Ok((result, output_shape));
    }
}

pub fn reduce_norm<T: Send + Sync>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)>
where
    T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + num_traits::Float, // This provides sqrt
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
        let total_elements = shape.iter().product::<usize>();

        #[cfg(feature = "rayon")]
        {
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

        #[cfg(not(feature = "rayon"))]
        {
            let mut sum_squares = T::default();

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
    }

    // Multi-dimensional case - compute L2 norm along specified dimensions
    #[cfg(feature = "rayon")]
    {
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

        return Ok((result, output_shape));
    }

    #[cfg(not(feature = "rayon"))]
    {
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

        return Ok((result, output_shape));
    }
}

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
    #[cfg(feature = "rayon")]
    {
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
                    } else {
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
                    if first || val > max_val {
                        max_val = val;
                        max_index = i as i32;
                        first = false;
                    }
                }

                max_index
            })
            .collect();

        return Ok((result, output_shape));
    }

    #[cfg(not(feature = "rayon"))]
    {
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
                } else {
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
                if first || val > max_val {
                    max_val = val;
                    max_index = i as i32;
                    first = false;
                }
            }

            result[output_idx] = max_index;
        }

        return Ok((result, output_shape));
    }
}

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
    #[cfg(feature = "rayon")]
    {
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
                    } else {
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

                min_index
            })
            .collect();

        return Ok((result, output_shape));
    }

    #[cfg(not(feature = "rayon"))]
    {
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

        return Ok((result, output_shape));
    }
}

pub fn reduce_any<T: Send + Sync>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<bool>, Vec<usize>)>
where
    T: Copy + Default + PartialEq,
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

        return Ok((result, output_shape));
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

        return Ok((result, output_shape));
    }
}

pub fn reduce_all<T: Send + Sync>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<bool>, Vec<usize>)>
where
    T: Copy + Default + PartialEq,
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

        return Ok((result, output_shape));
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

        return Ok((result, output_shape));
    }
}
