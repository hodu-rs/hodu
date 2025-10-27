use crate::{compat::*, types::layout::Layout};

pub fn concat_map<T: Copy>(
    first_storage: &[T],
    other_storages: &[&[T]],
    layouts: &[&Layout],
    dim: usize,
    output_shape: &[usize],
) -> Vec<T> {
    let ndim = output_shape.len();
    let total_size: usize = output_shape.iter().product();
    let mut result = Vec::with_capacity(total_size);

    // Create indices for iteration over all dimensions
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

pub fn split_map<T: Copy>(storage: &[T], layout: &Layout, dim: usize, sizes: &[usize]) -> Vec<Vec<T>> {
    let shape = layout.get_shape();
    let ndim = shape.len();
    let strides = layout.get_strides();

    let mut results = Vec::with_capacity(sizes.len());
    let mut split_offset = 0usize; // Track cumulative offset along split dimension

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
                    indices[i] + split_offset // Add split offset for the split dimension
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

        // Move to next split position along the split dimension
        split_offset += size;
    }

    results
}
