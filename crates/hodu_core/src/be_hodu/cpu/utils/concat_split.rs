use crate::{compat::*, types::layout::Layout};

#[cfg(feature = "rayon")]
const PARALLEL_THRESHOLD: usize = 4096;
#[cfg(feature = "rayon")]
const PARALLEL_CHUNK_SIZE: usize = 1024;

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
pub fn concat_map<T: Copy + Send + Sync>(
    first_storage: &[T],
    other_storages: &[&[T]],
    layouts: &[&Layout],
    dim: usize,
    output_shape: &[usize],
) -> Vec<T> {
    use rayon::prelude::*;

    let total_size: usize = output_shape.iter().product();

    if total_size >= PARALLEL_THRESHOLD {
        let chunks = total_size.div_ceil(PARALLEL_CHUNK_SIZE);
        (0..chunks)
            .into_par_iter()
            .flat_map(|chunk_idx| {
                let start = chunk_idx * PARALLEL_CHUNK_SIZE;
                let end = (start + PARALLEL_CHUNK_SIZE).min(total_size);
                let mut chunk_result = Vec::with_capacity(end - start);
                for flat_idx in start..end {
                    let indices = flat_to_indices(flat_idx, output_shape);

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

                    let mut local_indices = indices.clone();
                    local_indices[dim] -= cumulative_dim_size;

                    let layout = layouts[tensor_idx];
                    let strides = layout.get_strides();
                    let mut flat_index = layout.get_offset();
                    for (idx, stride) in local_indices.iter().zip(strides.iter()) {
                        flat_index += idx * stride;
                    }

                    let value = if tensor_idx == 0 {
                        first_storage[flat_index]
                    } else {
                        other_storages[tensor_idx - 1][flat_index]
                    };
                    chunk_result.push(value);
                }
                chunk_result
            })
            .collect()
    } else {
        let ndim = output_shape.len();
        let mut result = Vec::with_capacity(total_size);
        let mut indices = vec![0; ndim];

        for _ in 0..total_size {
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

            let mut local_indices = indices.clone();
            local_indices[dim] -= cumulative_dim_size;

            let layout = layouts[tensor_idx];
            let strides = layout.get_strides();
            let mut flat_index = layout.get_offset();
            for (idx, stride) in local_indices.iter().zip(strides.iter()) {
                flat_index += idx * stride;
            }

            let value = if tensor_idx == 0 {
                first_storage[flat_index]
            } else {
                other_storages[tensor_idx - 1][flat_index]
            };
            result.push(value);

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

#[cfg(not(feature = "rayon"))]
pub fn concat_map<T: Copy + Send + Sync>(
    first_storage: &[T],
    other_storages: &[&[T]],
    layouts: &[&Layout],
    dim: usize,
    output_shape: &[usize],
) -> Vec<T> {
    let total_size: usize = output_shape.iter().product();
    let ndim = output_shape.len();
    let mut result = Vec::with_capacity(total_size);
    let mut indices = vec![0; ndim];

    for _ in 0..total_size {
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

        let mut local_indices = indices.clone();
        local_indices[dim] -= cumulative_dim_size;

        let layout = layouts[tensor_idx];
        let strides = layout.get_strides();
        let mut flat_index = layout.get_offset();
        for (idx, stride) in local_indices.iter().zip(strides.iter()) {
            flat_index += idx * stride;
        }

        let value = if tensor_idx == 0 {
            first_storage[flat_index]
        } else {
            other_storages[tensor_idx - 1][flat_index]
        };
        result.push(value);

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

#[cfg(feature = "rayon")]
pub fn split_map<T: Copy + Send + Sync>(storage: &[T], layout: &Layout, dim: usize, sizes: &[usize]) -> Vec<Vec<T>> {
    use rayon::prelude::*;

    let shape = layout.get_shape();
    let ndim = shape.len();
    let strides = layout.get_strides();

    let mut split_offset = 0usize;
    sizes
        .iter()
        .map(|&size| {
            let current_offset = split_offset;
            split_offset += size;

            let mut output_shape = shape.to_vec();
            output_shape[dim] = size;
            let output_size: usize = output_shape.iter().product();

            if output_size >= PARALLEL_THRESHOLD {
                let chunks = output_size.div_ceil(PARALLEL_CHUNK_SIZE);
                (0..chunks)
                    .into_par_iter()
                    .flat_map(|chunk_idx| {
                        let start = chunk_idx * PARALLEL_CHUNK_SIZE;
                        let end = (start + PARALLEL_CHUNK_SIZE).min(output_size);
                        let mut chunk_result = Vec::with_capacity(end - start);
                        for flat_idx in start..end {
                            let indices = flat_to_indices(flat_idx, &output_shape);

                            let mut flat_index = layout.get_offset();
                            for i in 0..ndim {
                                let idx = if i == dim {
                                    indices[i] + current_offset
                                } else {
                                    indices[i]
                                };
                                flat_index += idx * strides[i];
                            }

                            chunk_result.push(storage[flat_index]);
                        }
                        chunk_result
                    })
                    .collect()
            } else {
                let mut result = Vec::with_capacity(output_size);
                let mut indices = vec![0; ndim];

                for _ in 0..output_size {
                    let mut flat_index = layout.get_offset();
                    for i in 0..ndim {
                        let idx = if i == dim {
                            indices[i] + current_offset
                        } else {
                            indices[i]
                        };
                        flat_index += idx * strides[i];
                    }

                    result.push(storage[flat_index]);

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
        })
        .collect()
}

#[cfg(not(feature = "rayon"))]
pub fn split_map<T: Copy + Send + Sync>(storage: &[T], layout: &Layout, dim: usize, sizes: &[usize]) -> Vec<Vec<T>> {
    let shape = layout.get_shape();
    let ndim = shape.len();
    let strides = layout.get_strides();

    let mut results = Vec::with_capacity(sizes.len());
    let mut split_offset = 0usize;

    for &size in sizes {
        let mut output_shape = shape.to_vec();
        output_shape[dim] = size;
        let output_size: usize = output_shape.iter().product();

        let mut result = Vec::with_capacity(output_size);
        let mut indices = vec![0; ndim];

        for _ in 0..output_size {
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
