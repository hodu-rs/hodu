use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    scalar::Scalar,
    types::layout::Layout,
};

pub fn binary_map<T: Copy, U: Copy, F: FnMut(T, T) -> U>(
    lhs_storage: &[T],
    rhs_storage: &[T],
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    mut f: F,
) -> Vec<U> {
    let lhs_offset = lhs_layout.get_offset();
    let rhs_offset = rhs_layout.get_offset();
    let lhs_size = lhs_layout.get_size();
    let rhs_size = rhs_layout.get_size();

    let output_size = lhs_size.max(rhs_size);
    let mut result = Vec::with_capacity(output_size);

    match (lhs_layout.is_contiguous(), rhs_layout.is_contiguous()) {
        (true, true) => {
            let lhs_slice = &lhs_storage[lhs_offset..lhs_offset + lhs_size];
            let rhs_slice = &rhs_storage[rhs_offset..rhs_offset + rhs_size];

            for i in 0..output_size {
                let l = lhs_slice[i % lhs_size];
                let r = rhs_slice[i % rhs_size];
                result.push(f(l, r));
            }
        },
        (true, false) => {
            let lhs_slice = &lhs_storage[lhs_offset..lhs_offset + lhs_size];
            let dims = rhs_layout.get_shape();
            let rhs_strides = rhs_layout.get_strides();

            for i in 0..output_size {
                let l = lhs_slice[i % lhs_size];

                let mut rhs_idx = rhs_offset;
                let mut tmp_i = i % rhs_size;

                for d in (0..dims.len()).rev() {
                    let i_dim = tmp_i % dims[d];
                    rhs_idx += if rhs_strides[d] != 0 { i_dim * rhs_strides[d] } else { 0 };
                    tmp_i /= dims[d];
                }

                let r = rhs_storage[rhs_idx];
                result.push(f(l, r));
            }
        },
        (false, true) => {
            let rhs_slice = &rhs_storage[rhs_offset..rhs_offset + rhs_size];
            let dims = lhs_layout.get_shape();
            let lhs_strides = lhs_layout.get_strides();

            for i in 0..output_size {
                let r = rhs_slice[i % rhs_size];

                let mut lhs_idx = lhs_offset;
                let mut tmp_i = i % lhs_size;

                for d in (0..dims.len()).rev() {
                    let i_dim = tmp_i % dims[d];
                    lhs_idx += if lhs_strides[d] != 0 { i_dim * lhs_strides[d] } else { 0 };
                    tmp_i /= dims[d];
                }

                let l = lhs_storage[lhs_idx];
                result.push(f(l, r));
            }
        },
        (false, false) => {
            let lhs_dims = lhs_layout.get_shape();
            let lhs_strides = lhs_layout.get_strides();
            let rhs_dims = rhs_layout.get_shape();
            let rhs_strides = rhs_layout.get_strides();

            for i in 0..output_size {
                let mut lhs_idx = lhs_offset;
                let mut tmp_i = i % lhs_size;

                for d in (0..lhs_dims.len()).rev() {
                    let i_dim = tmp_i % lhs_dims[d];
                    lhs_idx += if lhs_strides[d] != 0 { i_dim * lhs_strides[d] } else { 0 };
                    tmp_i /= lhs_dims[d];
                }

                let mut rhs_idx = rhs_offset;
                tmp_i = i % rhs_size;

                for d in (0..rhs_dims.len()).rev() {
                    let i_dim = tmp_i % rhs_dims[d];
                    rhs_idx += if rhs_strides[d] != 0 { i_dim * rhs_strides[d] } else { 0 };
                    tmp_i /= rhs_dims[d];
                }

                let l = lhs_storage[lhs_idx];
                let r = rhs_storage[rhs_idx];
                result.push(f(l, r));
            }
        },
    }

    result
}

pub fn binary_logical_map<T: Copy, F: FnMut(T, T) -> bool>(
    lhs_storage: &[T],
    rhs_storage: &[T],
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    mut f: F,
) -> Vec<bool> {
    let lhs_offset = lhs_layout.get_offset();
    let rhs_offset = rhs_layout.get_offset();
    let lhs_size = lhs_layout.get_size();
    let rhs_size = rhs_layout.get_size();

    let output_size = lhs_size.max(rhs_size);
    let mut result = Vec::with_capacity(output_size);

    match (lhs_layout.is_contiguous(), rhs_layout.is_contiguous()) {
        (true, true) => {
            let lhs_slice = &lhs_storage[lhs_offset..lhs_offset + lhs_size];
            let rhs_slice = &rhs_storage[rhs_offset..rhs_offset + rhs_size];

            for i in 0..output_size {
                let l = lhs_slice[i % lhs_size];
                let r = rhs_slice[i % rhs_size];
                result.push(f(l, r));
            }
        },
        (true, false) => {
            let lhs_slice = &lhs_storage[lhs_offset..lhs_offset + lhs_size];
            let dims = rhs_layout.get_shape();
            let rhs_strides = rhs_layout.get_strides();

            for i in 0..output_size {
                let l = lhs_slice[i % lhs_size];

                let mut rhs_idx = rhs_offset;
                let mut tmp_i = i % rhs_size;

                for d in (0..dims.len()).rev() {
                    let i_dim = tmp_i % dims[d];
                    rhs_idx += if rhs_strides[d] != 0 { i_dim * rhs_strides[d] } else { 0 };
                    tmp_i /= dims[d];
                }

                let r = rhs_storage[rhs_idx];
                result.push(f(l, r));
            }
        },
        (false, true) => {
            let rhs_slice = &rhs_storage[rhs_offset..rhs_offset + rhs_size];
            let dims = lhs_layout.get_shape();
            let lhs_strides = lhs_layout.get_strides();

            for i in 0..output_size {
                let r = rhs_slice[i % rhs_size];

                let mut lhs_idx = lhs_offset;
                let mut tmp_i = i % lhs_size;

                for d in (0..dims.len()).rev() {
                    let i_dim = tmp_i % dims[d];
                    lhs_idx += if lhs_strides[d] != 0 { i_dim * lhs_strides[d] } else { 0 };
                    tmp_i /= dims[d];
                }

                let l = lhs_storage[lhs_idx];
                result.push(f(l, r));
            }
        },
        (false, false) => {
            let lhs_dims = lhs_layout.get_shape();
            let lhs_strides = lhs_layout.get_strides();
            let rhs_dims = rhs_layout.get_shape();
            let rhs_strides = rhs_layout.get_strides();

            for i in 0..output_size {
                let mut lhs_idx = lhs_offset;
                let mut tmp_i = i % lhs_size;

                for d in (0..lhs_dims.len()).rev() {
                    let i_dim = tmp_i % lhs_dims[d];
                    lhs_idx += if lhs_strides[d] != 0 { i_dim * lhs_strides[d] } else { 0 };
                    tmp_i /= lhs_dims[d];
                }

                let mut rhs_idx = rhs_offset;
                tmp_i = i % rhs_size;

                for d in (0..rhs_dims.len()).rev() {
                    let i_dim = tmp_i % rhs_dims[d];
                    rhs_idx += if rhs_strides[d] != 0 { i_dim * rhs_strides[d] } else { 0 };
                    tmp_i /= rhs_dims[d];
                }

                let l = lhs_storage[lhs_idx];
                let r = rhs_storage[rhs_idx];
                result.push(f(l, r));
            }
        },
    }

    result
}

pub fn unary_map<T: Copy, U: Copy, F: FnMut(T) -> U>(storage: &[T], layout: &Layout, mut f: F) -> Vec<U> {
    let offset = layout.get_offset();
    let size = layout.get_size();

    if layout.is_contiguous() {
        storage[offset..offset + size].iter().map(|&v| f(v)).collect()
    } else {
        let mut result = Vec::with_capacity(size);
        let dims = layout.get_shape();
        let strides = layout.get_strides();

        for i in 0..size {
            let mut idx = offset;
            let mut tmp_i = i;

            for d in (0..dims.len()).rev() {
                let i_dim = tmp_i % dims[d];
                idx += if strides[d] != 0 { i_dim * strides[d] } else { 0 };
                tmp_i /= dims[d];
            }

            let v = unsafe { storage.get_unchecked(idx) };
            result.push(f(*v));
        }
        result
    }
}

pub fn unary_logical_map<T: Copy, F: FnMut(T) -> bool>(storage: &[T], layout: &Layout, mut f: F) -> Vec<bool> {
    let offset = layout.get_offset();
    let size = layout.get_size();

    if layout.is_contiguous() {
        storage[offset..offset + size].iter().map(|&v| f(v)).collect()
    } else {
        let mut result = Vec::with_capacity(size);
        let dims = layout.get_shape();
        let strides = layout.get_strides();

        for i in 0..size {
            let mut idx = offset;
            let mut tmp_i = i;

            for d in (0..dims.len()).rev() {
                let i_dim = tmp_i % dims[d];
                idx += if strides[d] != 0 { i_dim * strides[d] } else { 0 };
                tmp_i /= dims[d];
            }

            let v = unsafe { storage.get_unchecked(idx) };
            result.push(f(*v));
        }
        result
    }
}

pub fn unary_scalar_map<T: Copy, F: FnMut(T, Scalar) -> T>(
    storage: &[T],
    layout: &Layout,
    scalar: Scalar,
    mut f: F,
) -> Vec<T> {
    let offset = layout.get_offset();
    let size = layout.get_size();

    if layout.is_contiguous() {
        storage[offset..offset + size].iter().map(|&v| f(v, scalar)).collect()
    } else {
        let mut result = Vec::with_capacity(size);
        let dims = layout.get_shape();
        let strides = layout.get_strides();

        for i in 0..size {
            let mut idx = offset;
            let mut tmp_i = i;

            for d in (0..dims.len()).rev() {
                let i_dim = tmp_i % dims[d];
                idx += if strides[d] != 0 { i_dim * strides[d] } else { 0 };
                tmp_i /= dims[d];
            }

            let v = unsafe { storage.get_unchecked(idx) };
            result.push(f(*v, scalar));
        }
        result
    }
}

pub fn cmp_map<T: Copy, F: FnMut(T, T) -> bool>(
    lhs_storage: &[T],
    rhs_storage: &[T],
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    mut f: F,
) -> Vec<bool> {
    let lhs_offset = lhs_layout.get_offset();
    let rhs_offset = rhs_layout.get_offset();
    let lhs_size = lhs_layout.get_size();
    let rhs_size = rhs_layout.get_size();

    let output_size = lhs_size.max(rhs_size);
    let mut result = Vec::with_capacity(output_size);

    match (lhs_layout.is_contiguous(), rhs_layout.is_contiguous()) {
        (true, true) => {
            let lhs_slice = &lhs_storage[lhs_offset..lhs_offset + lhs_size];
            let rhs_slice = &rhs_storage[rhs_offset..rhs_offset + rhs_size];

            for i in 0..output_size {
                let l = lhs_slice[i % lhs_size];
                let r = rhs_slice[i % rhs_size];
                result.push(f(l, r));
            }
        },
        (true, false) => {
            let lhs_slice = &lhs_storage[lhs_offset..lhs_offset + lhs_size];
            let dims = rhs_layout.get_shape();
            let rhs_strides = rhs_layout.get_strides();

            for i in 0..output_size {
                let l = lhs_slice[i % lhs_size];

                let mut rhs_idx = rhs_offset;
                let mut tmp_i = i % rhs_size;

                for d in (0..dims.len()).rev() {
                    let i_dim = tmp_i % dims[d];
                    rhs_idx += if rhs_strides[d] != 0 { i_dim * rhs_strides[d] } else { 0 };
                    tmp_i /= dims[d];
                }

                let r = rhs_storage[rhs_idx];
                result.push(f(l, r));
            }
        },
        (false, true) => {
            let rhs_slice = &rhs_storage[rhs_offset..rhs_offset + rhs_size];
            let dims = lhs_layout.get_shape();
            let lhs_strides = lhs_layout.get_strides();

            for i in 0..output_size {
                let r = rhs_slice[i % rhs_size];

                let mut lhs_idx = lhs_offset;
                let mut tmp_i = i % lhs_size;

                for d in (0..dims.len()).rev() {
                    let i_dim = tmp_i % dims[d];
                    lhs_idx += if lhs_strides[d] != 0 { i_dim * lhs_strides[d] } else { 0 };
                    tmp_i /= dims[d];
                }

                let l = lhs_storage[lhs_idx];
                result.push(f(l, r));
            }
        },
        (false, false) => {
            let lhs_dims = lhs_layout.get_shape();
            let lhs_strides = lhs_layout.get_strides();
            let rhs_dims = rhs_layout.get_shape();
            let rhs_strides = rhs_layout.get_strides();

            for i in 0..output_size {
                let mut lhs_idx = lhs_offset;
                let mut tmp_i = i % lhs_size;

                for d in (0..lhs_dims.len()).rev() {
                    let i_dim = tmp_i % lhs_dims[d];
                    lhs_idx += if lhs_strides[d] != 0 { i_dim * lhs_strides[d] } else { 0 };
                    tmp_i /= lhs_dims[d];
                }

                let mut rhs_idx = rhs_offset;
                tmp_i = i % rhs_size;

                for d in (0..rhs_dims.len()).rev() {
                    let i_dim = tmp_i % rhs_dims[d];
                    rhs_idx += if rhs_strides[d] != 0 { i_dim * rhs_strides[d] } else { 0 };
                    tmp_i /= rhs_dims[d];
                }

                let l = lhs_storage[lhs_idx];
                let r = rhs_storage[rhs_idx];
                result.push(f(l, r));
            }
        },
    }

    result
}

pub fn cmp_scalar_map<T: Copy, F: FnMut(T, Scalar) -> bool>(
    storage: &[T],
    layout: &Layout,
    scalar: Scalar,
    mut f: F,
) -> Vec<bool> {
    let offset = layout.get_offset();
    let size = layout.get_size();

    if layout.is_contiguous() {
        storage[offset..offset + size].iter().map(|&v| f(v, scalar)).collect()
    } else {
        let mut result = Vec::with_capacity(size);
        let dims = layout.get_shape();
        let strides = layout.get_strides();

        for i in 0..size {
            let mut idx = offset;
            let mut tmp_i = i;

            for d in (0..dims.len()).rev() {
                let i_dim = tmp_i % dims[d];
                idx += if strides[d] != 0 { i_dim * strides[d] } else { 0 };
                tmp_i /= dims[d];
            }

            let v = unsafe { storage.get_unchecked(idx) };
            result.push(f(*v, scalar));
        }
        result
    }
}

pub fn matmul_map<T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default>(
    lhs_storage: &[T],
    rhs_storage: &[T],
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> HoduResult<Vec<T>> {
    // Batched matrix multiplication with broadcasting
    // Supports: [...batch_dims..., M, K] x [...batch_dims..., K, N] -> [...batch_dims..., M, N]

    let lhs_shape = lhs_layout.get_shape();
    let rhs_shape = rhs_layout.get_shape();
    let lhs_strides = lhs_layout.get_strides();
    let rhs_strides = rhs_layout.get_strides();
    let lhs_offset = lhs_layout.get_offset();
    let rhs_offset = rhs_layout.get_offset();
    let lhs_ndim = lhs_shape.len();
    let rhs_ndim = rhs_shape.len();

    if lhs_ndim < 2 || rhs_ndim < 2 {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "matmul - both tensors must be at least 2D".to_string(),
        });
    }

    // Extract matrix dimensions (last 2 dims)
    let lhs_m = lhs_shape[lhs_ndim - 2];
    let lhs_k = lhs_shape[lhs_ndim - 1];
    let rhs_k = rhs_shape[rhs_ndim - 2];
    let rhs_n = rhs_shape[rhs_ndim - 1];

    if lhs_k != rhs_k {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "matmul - inner dimensions must match".to_string(),
        });
    }

    // Compute batch dimensions
    let lhs_batch_ndim = lhs_ndim - 2;
    let rhs_batch_ndim = rhs_ndim - 2;
    let batch_ndim = lhs_batch_ndim.max(rhs_batch_ndim);

    // Compute broadcasted batch shape
    let mut batch_shape = vec![1; batch_ndim];
    for i in 0..batch_ndim {
        let lhs_dim = if i < lhs_batch_ndim {
            lhs_shape[lhs_batch_ndim - 1 - i]
        } else {
            1
        };
        let rhs_dim = if i < rhs_batch_ndim {
            rhs_shape[rhs_batch_ndim - 1 - i]
        } else {
            1
        };

        if lhs_dim != 1 && rhs_dim != 1 && lhs_dim != rhs_dim {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: "matmul - incompatible batch dimensions".to_string(),
            });
        }
        batch_shape[batch_ndim - 1 - i] = lhs_dim.max(rhs_dim);
    }

    // Result shape: batch_shape + [M, N]
    let total_batches: usize = batch_shape.iter().product();
    let result_size = total_batches * lhs_m * rhs_n;
    let mut result = vec![T::default(); result_size];

    // Perform batched matrix multiplication
    for batch_idx in 0..total_batches {
        // Compute batch indices
        let mut batch_indices = vec![0; batch_ndim];
        let mut temp = batch_idx;
        for i in (0..batch_ndim).rev() {
            batch_indices[i] = temp % batch_shape[i];
            temp /= batch_shape[i];
        }

        // Map batch indices to lhs and rhs indices (with broadcasting)
        let mut lhs_batch_indices = vec![0; lhs_batch_ndim];
        for i in 0..lhs_batch_ndim {
            let batch_dim_idx = batch_ndim - lhs_batch_ndim + i;
            lhs_batch_indices[i] = if lhs_shape[i] == 1 {
                0
            } else {
                batch_indices[batch_dim_idx]
            };
        }

        let mut rhs_batch_indices = vec![0; rhs_batch_ndim];
        for i in 0..rhs_batch_ndim {
            let batch_dim_idx = batch_ndim - rhs_batch_ndim + i;
            rhs_batch_indices[i] = if rhs_shape[i] == 1 {
                0
            } else {
                batch_indices[batch_dim_idx]
            };
        }

        // Compute the matmul for this batch
        for i in 0..lhs_m {
            for j in 0..rhs_n {
                let mut sum = T::default();

                for k in 0..lhs_k {
                    // Calculate lhs index
                    let mut lhs_idx = lhs_offset;
                    for (dim, &idx) in lhs_batch_indices.iter().enumerate() {
                        lhs_idx += idx * lhs_strides[dim];
                    }
                    lhs_idx += i * lhs_strides[lhs_ndim - 2];
                    lhs_idx += k * lhs_strides[lhs_ndim - 1];

                    // Calculate rhs index
                    let mut rhs_idx = rhs_offset;
                    for (dim, &idx) in rhs_batch_indices.iter().enumerate() {
                        rhs_idx += idx * rhs_strides[dim];
                    }
                    rhs_idx += k * rhs_strides[rhs_ndim - 2];
                    rhs_idx += j * rhs_strides[rhs_ndim - 1];

                    let lhs_val = unsafe { *lhs_storage.get_unchecked(lhs_idx) };
                    let rhs_val = unsafe { *rhs_storage.get_unchecked(rhs_idx) };

                    sum = sum + lhs_val * rhs_val;
                }

                let result_idx = batch_idx * lhs_m * rhs_n + i * rhs_n + j;
                result[result_idx] = sum;
            }
        }
    }

    Ok(result)
}

pub fn dot_map<T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default>(
    lhs_storage: &[T],
    rhs_storage: &[T],
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> HoduResult<Vec<T>> {
    // Simple 2D matrix multiplication
    // A: (m, k), B: (k, n) -> C: (m, n)

    let lhs_shape = lhs_layout.get_shape();
    let rhs_shape = rhs_layout.get_shape();
    let lhs_strides = lhs_layout.get_strides();
    let rhs_strides = rhs_layout.get_strides();
    let lhs_offset = lhs_layout.get_offset();
    let rhs_offset = rhs_layout.get_offset();

    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "dot - only 2D tensors supported".to_string(),
        });
    }

    let (m, k1) = (lhs_shape[0], lhs_shape[1]);
    let (k2, n) = (rhs_shape[0], rhs_shape[1]);

    if k1 != k2 {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "dot - inner dimensions must match".to_string(),
        });
    }

    let mut result = vec![T::default(); m * n];

    // Compute matrix multiplication with offset/stride handling
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::default();

            for k in 0..k1 {
                // Calculate actual indices considering offset and strides
                let lhs_idx = lhs_offset + i * lhs_strides[0] + k * lhs_strides[1];
                let rhs_idx = rhs_offset + k * rhs_strides[0] + j * rhs_strides[1];

                let lhs_val = unsafe { *lhs_storage.get_unchecked(lhs_idx) };
                let rhs_val = unsafe { *rhs_storage.get_unchecked(rhs_idx) };

                sum = sum + lhs_val * rhs_val;
            }

            let result_idx = i * n + j;
            result[result_idx] = sum;
        }
    }

    Ok(result)
}

pub fn reduce_sum<T: Copy + Default + ops::Add<Output = T>>(
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
    let mut result = vec![T::default(); output_size];

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
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

        result[0] = sum;
        return Ok((result, output_shape));
    }

    // Multi-dimensional reduction - simplified implementation
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

pub fn reduce_mean<T>(
    storage: &[T],
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<(Vec<T>, Vec<usize>)>
where
    T: Copy + Default + ops::Add<Output = T> + ops::Div<Output = T>,
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

    let mean_result = sum_result.into_iter().map(|x| x / count_val).collect();

    Ok((mean_result, output_shape))
}

pub fn reduce_max<T: Copy + PartialOrd>(
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
    let mut result = vec![storage[offset]; output_size]; // Initialize with first element

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
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

        result[0] = max_val;
        return Ok((result, output_shape));
    }

    // Multi-dimensional case - similar to sum but with max operation
    for output_idx in 0..output_size {
        let mut max_val = storage[offset];
        let mut first = true;

        // Similar logic to sum but finding maximum
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

pub fn reduce_min<T: Copy + PartialOrd>(
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
    let mut result = vec![storage[offset]; output_size];

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
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

        result[0] = min_val;
        return Ok((result, output_shape));
    }

    // Multi-dimensional case - similar to max but with min operation
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

pub fn reduce_prod<T: Copy + ops::Mul<Output = T>>(
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
    let mut result = vec![T::one(); output_size];

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
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

        result[0] = prod_val;
        return Ok((result, output_shape));
    }

    // Multi-dimensional case
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
        + num_traits::NumCast,
    T: num_traits::Float, // This provides sqrt
{
    let (var_result, output_shape) = reduce_var(storage, layout, dims, keep_dim, unbiased)?;

    let std_result = var_result.into_iter().map(|x| x.sqrt()).collect();

    Ok((std_result, output_shape))
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
        + num_traits::NumCast,
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
    let mut result = vec![T::default(); output_size];

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

        result[0] = var_sum / denom_val;
        return Ok((result, output_shape));
    }

    // Multi-dimensional case
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

pub fn reduce_norm<T>(
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
    let mut result = vec![T::default(); output_size];

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

        result[0] = sum_squares.sqrt();
        return Ok((result, output_shape));
    }

    // Multi-dimensional case - compute L2 norm along specified dimensions
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
