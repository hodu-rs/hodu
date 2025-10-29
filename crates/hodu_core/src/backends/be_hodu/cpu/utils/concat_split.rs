use crate::{compat::*, types::layout::Layout};

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

pub fn concat_map<T: Copy + Send + Sync>(
    first_storage: &[T],
    other_storages: &[&[T]],
    layouts: &[&Layout],
    dim: usize,
    output_shape: &[usize],
) -> Vec<T> {
    let total_size: usize = output_shape.iter().product();

    #[cfg(feature = "rayon")]
    {
        (0..total_size)
            .into_par_iter()
            .map(|flat_idx| {
                let indices = flat_to_indices(flat_idx, output_shape);

                // Determine which tensor to read from based on concat dimension index
                let mut cumulative_dim_size = 0;
                let mut tensor_idx = 0;
                let dim_index = indices[dim];

                for (i, layout) in layouts.iter().enumerate() {
                    let dim_size = layout.get_shape()[dim];
                    if dim_index < cumulative_dim_size + dim_size {
                        tensor_idx = i;
                        break;
                    }
                    cumulative_dim_size += dim_size;
                }

                // Adjust index for the selected tensor
                let mut local_indices = indices.clone();
                local_indices[dim] -= cumulative_dim_size;

                // Calculate flat index in the source tensor
                let layout = layouts[tensor_idx];
                let strides = layout.get_strides();
                let mut flat_index = layout.get_offset();
                for (idx, stride) in local_indices.iter().zip(strides.iter()) {
                    flat_index += idx * stride;
                }

                // Get data from appropriate storage
                if tensor_idx == 0 {
                    first_storage[flat_index]
                } else {
                    other_storages[tensor_idx - 1][flat_index]
                }
            })
            .collect()
    }

    #[cfg(not(feature = "rayon"))]
    {
        let ndim = output_shape.len();
        let mut result = Vec::with_capacity(total_size);
        let mut indices = vec![0; ndim];

        for _ in 0..total_size {
            // Determine which tensor to read from based on concat dimension index
            let mut cumulative_dim_size = 0;
            let mut tensor_idx = 0;
            let dim_index = indices[dim];

            for (i, layout) in layouts.iter().enumerate() {
                let dim_size = layout.get_shape()[dim];
                if dim_index < cumulative_dim_size + dim_size {
                    tensor_idx = i;
                    break;
                }
                cumulative_dim_size += dim_size;
            }

            // Adjust index for the selected tensor
            let mut local_indices = indices.clone();
            local_indices[dim] -= cumulative_dim_size;

            // Calculate flat index in the source tensor
            let layout = layouts[tensor_idx];
            let strides = layout.get_strides();
            let mut flat_index = layout.get_offset();
            for (idx, stride) in local_indices.iter().zip(strides.iter()) {
                flat_index += idx * stride;
            }

            // Get data from appropriate storage
            let value = if tensor_idx == 0 {
                first_storage[flat_index]
            } else {
                other_storages[tensor_idx - 1][flat_index]
            };
            result.push(value);

            // Increment indices (row-major order)
            let mut carry = 1;
            for i in (0..ndim).rev() {
                indices[i] += carry;
                if indices[i] < output_shape[i] {
                    carry = 0;
                    break;
                }
                indices[i] = 0;
            }
            if carry != 0 {
                break;
            }
        }

        result
    }
}

pub fn split_map<T: Copy + Send + Sync>(storage: &[T], layout: &Layout, dim: usize, sizes: &[usize]) -> Vec<Vec<T>> {
    let shape = layout.get_shape();
    let ndim = shape.len();
    let strides = layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let mut split_offset = 0usize;
        sizes
            .iter()
            .map(|&size| {
                let current_offset = split_offset;
                split_offset += size;

                // Calculate output shape for this split
                let mut output_shape = shape.to_vec();
                output_shape[dim] = size;
                let output_size: usize = output_shape.iter().product();

                (0..output_size)
                    .into_par_iter()
                    .map(|flat_idx| {
                        let indices = flat_to_indices(flat_idx, &output_shape);

                        // Calculate flat index in source tensor
                        let mut flat_index = layout.get_offset();
                        for i in 0..ndim {
                            let idx = if i == dim {
                                indices[i] + current_offset
                            } else {
                                indices[i]
                            };
                            flat_index += idx * strides[i];
                        }

                        storage[flat_index]
                    })
                    .collect()
            })
            .collect()
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut results = Vec::with_capacity(sizes.len());
        let mut split_offset = 0usize;

        for &size in sizes {
            // Calculate output shape for this split
            let mut output_shape = shape.to_vec();
            output_shape[dim] = size;
            let output_size: usize = output_shape.iter().product();

            let mut result = Vec::with_capacity(output_size);
            let mut indices = vec![0; ndim];

            for _ in 0..output_size {
                // Calculate flat index in source tensor
                let mut flat_index = layout.get_offset();
                for i in 0..ndim {
                    let idx = if i == dim {
                        indices[i] + split_offset
                    } else {
                        indices[i]
                    };
                    flat_index += idx * strides[i];
                }

                result.push(storage[flat_index]);

                // Increment indices
                let mut carry = 1;
                for i in (0..ndim).rev() {
                    indices[i] += carry;
                    if indices[i] < output_shape[i] {
                        carry = 0;
                        break;
                    }
                    indices[i] = 0;
                }
                if carry != 0 {
                    break;
                }
            }

            results.push(result);
            split_offset += size;
        }

        results
    }
}
