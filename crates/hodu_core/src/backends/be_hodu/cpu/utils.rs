use crate::{
    backends::op::conv::{
        ParamsConv1D, ParamsConv2D, ParamsConv3D, ParamsConvTranspose1D, ParamsConvTranspose2D, ParamsConvTranspose3D,
    },
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

pub fn reduce_argmax<T: Copy + PartialOrd>(
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
    let mut result = vec![0i32; output_size];

    // Iterate over output positions
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

    Ok((result, output_shape))
}

pub fn reduce_argmin<T: Copy + PartialOrd>(
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
    let mut result = vec![0i32; output_size];

    // Iterate over output positions
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

    Ok((result, output_shape))
}

pub fn reduce_any<T>(
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
    let mut result = vec![false; output_size];

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
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

            // Check if value is truthy (non-zero/non-default)
            if storage[flat_index] != T::default() {
                result[0] = true;
                return Ok((result, output_shape));
            }
        }

        return Ok((result, output_shape));
    }

    // Multi-dimensional reduction
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

pub fn reduce_all<T>(
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
    let mut result = vec![true; output_size]; // Start with true for all

    // Simple case: reduce all dimensions
    if reduce_dims.len() == ndim {
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

            // Check if value is falsy (zero/default)
            if storage[flat_index] == T::default() {
                result[0] = false;
                return Ok((result, output_shape));
            }
        }

        return Ok((result, output_shape));
    }

    // Multi-dimensional reduction
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

pub fn index_select_map<T: Copy>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "index_select - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_offset = indices_layout.get_offset();
    let indices_size = indices_layout.get_size();
    let indices_strides = indices_layout.get_strides();
    let indices_shape = indices_layout.get_shape();

    // Output shape: replace the indexed dimension with indices size
    let mut output_shape = shape.to_vec();
    output_shape[dim] = indices_size;
    let output_size: usize = output_shape.iter().product();

    let mut result = Vec::with_capacity(output_size);
    let mut output_indices = vec![0; ndim];

    for _ in 0..output_size {
        // Calculate which index to use
        let index_pos = output_indices[dim];

        // Get the actual index value from indices tensor
        let mut indices_idx = indices_offset;
        if indices_layout.is_contiguous() {
            indices_idx += index_pos;
        } else {
            let mut tmp_pos = index_pos;
            for d in (0..indices_shape.len()).rev() {
                let i_dim = tmp_pos % indices_shape[d];
                indices_idx += i_dim * indices_strides[d];
                tmp_pos /= indices_shape[d];
            }
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        // Calculate flat index in source tensor
        let mut flat_index = offset;
        for i in 0..ndim {
            let actual_idx = if i == dim { idx as usize } else { output_indices[i] };
            flat_index += actual_idx * strides[i];
        }

        result.push(storage[flat_index]);

        // Increment output indices
        for i in (0..ndim).rev() {
            output_indices[i] += 1;
            if output_indices[i] < output_shape[i] {
                break;
            }
            output_indices[i] = 0;
        }
    }

    Ok(result)
}

pub fn index_put_map<T: Copy>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    values_storage: &[T],
    values_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "index_put - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_offset = indices_layout.get_offset();
    let indices_size = indices_layout.get_size();
    let indices_strides = indices_layout.get_strides();
    let indices_shape = indices_layout.get_shape();

    let values_shape = values_layout.get_shape();
    let values_strides = values_layout.get_strides();
    let values_offset = values_layout.get_offset();

    // Create a copy of the input storage
    let mut result = storage.to_vec();

    // Expected values shape: replace indexed dimension with indices size
    let mut expected_values_shape = shape.to_vec();
    expected_values_shape[dim] = indices_size;

    if values_shape != expected_values_shape {
        return Err(HoduError::InternalError(format!(
            "index_put - values shape {:?} does not match expected shape {:?}",
            values_shape, expected_values_shape
        )));
    }

    let values_size: usize = expected_values_shape.iter().product();
    let mut output_indices = vec![0; ndim];

    for _ in 0..values_size {
        // Calculate which index to use
        let index_pos = output_indices[dim];

        // Get the actual index value from indices tensor
        let mut indices_idx = indices_offset;
        if indices_layout.is_contiguous() {
            indices_idx += index_pos;
        } else {
            let mut tmp_pos = index_pos;
            for d in (0..indices_shape.len()).rev() {
                let i_dim = tmp_pos % indices_shape[d];
                indices_idx += i_dim * indices_strides[d];
                tmp_pos /= indices_shape[d];
            }
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        // Calculate flat index in result tensor
        let mut flat_index = offset;
        for i in 0..ndim {
            let actual_idx = if i == dim { idx as usize } else { output_indices[i] };
            flat_index += actual_idx * strides[i];
        }

        // Calculate flat index in values tensor
        let mut values_flat_index = values_offset;
        for i in 0..ndim {
            values_flat_index += output_indices[i] * values_strides[i];
        }

        // Put the value
        result[flat_index] = values_storage[values_flat_index];

        // Increment output indices
        for i in (0..ndim).rev() {
            output_indices[i] += 1;
            if output_indices[i] < expected_values_shape[i] {
                break;
            }
            output_indices[i] = 0;
        }
    }

    Ok(result)
}

pub fn gather_map<T: Copy>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "gather - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_shape = indices_layout.get_shape();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();
    let indices_ndim = indices_shape.len();

    if indices_ndim != ndim {
        return Err(HoduError::IncompatibleShapes {
            lhs: shape.to_vec(),
            rhs: indices_shape.to_vec(),
            op: "gather - indices must have same number of dimensions as input".to_string(),
        });
    }

    // Output has same shape as indices
    let output_size: usize = indices_shape.iter().product();
    let mut result = Vec::with_capacity(output_size);
    let mut output_indices = vec![0; ndim];

    for _ in 0..output_size {
        // Get index from indices tensor
        let mut indices_idx = indices_offset;
        for i in 0..ndim {
            indices_idx += output_indices[i] * indices_strides[i];
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        // Calculate flat index in source tensor
        let mut flat_index = offset;
        for i in 0..ndim {
            let actual_idx = if i == dim { idx as usize } else { output_indices[i] };
            flat_index += actual_idx * strides[i];
        }

        result.push(storage[flat_index]);

        // Increment output indices
        for i in (0..ndim).rev() {
            output_indices[i] += 1;
            if output_indices[i] < indices_shape[i] {
                break;
            }
            output_indices[i] = 0;
        }
    }

    Ok(result)
}

pub fn scatter_map<T: Copy>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    src_storage: &[T],
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "scatter - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_shape = indices_layout.get_shape();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let src_size = src_layout.get_size();

    // Check that indices and src have the same shape
    if indices_shape != src_shape {
        return Err(HoduError::IncompatibleShapes {
            lhs: indices_shape.to_vec(),
            rhs: src_shape.to_vec(),
            op: "scatter - indices and src must have the same shape".to_string(),
        });
    }

    // Create output by copying input
    let input_size = layout.get_size();
    let mut result = Vec::with_capacity(input_size);

    // Copy input to output
    if layout.is_contiguous() {
        result.extend_from_slice(&storage[offset..offset + input_size]);
    } else {
        let mut input_indices = vec![0; ndim];
        for _ in 0..input_size {
            let mut flat_index = offset;
            for i in 0..ndim {
                flat_index += input_indices[i] * strides[i];
            }
            result.push(storage[flat_index]);

            for i in (0..ndim).rev() {
                input_indices[i] += 1;
                if input_indices[i] < shape[i] {
                    break;
                }
                input_indices[i] = 0;
            }
        }
    }

    // Now scatter src values according to indices
    let mut scatter_indices = vec![0; ndim];
    for _ in 0..src_size {
        // Get index from indices tensor
        let mut indices_idx = indices_offset;
        for i in 0..ndim {
            indices_idx += scatter_indices[i] * indices_strides[i];
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        // Get source value
        let mut src_idx = src_offset;
        for i in 0..ndim {
            src_idx += scatter_indices[i] * src_strides[i];
        }
        let src_val = src_storage[src_idx];

        // Calculate flat index in output (contiguous result)
        let mut output_idx = 0;
        let mut stride = 1;
        for i in (0..ndim).rev() {
            let actual_idx = if i == dim { idx as usize } else { scatter_indices[i] };
            output_idx += actual_idx * stride;
            stride *= shape[i];
        }

        result[output_idx] = src_val;

        // Increment scatter indices
        for i in (0..ndim).rev() {
            scatter_indices[i] += 1;
            if scatter_indices[i] < src_shape[i] {
                break;
            }
            scatter_indices[i] = 0;
        }
    }

    Ok(result)
}

pub fn scatter_add_map<T: Copy + ops::Add<Output = T>>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    src_storage: &[T],
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "scatter_add - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_shape = indices_layout.get_shape();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let src_size = src_layout.get_size();

    if indices_shape != src_shape {
        return Err(HoduError::IncompatibleShapes {
            lhs: indices_shape.to_vec(),
            rhs: src_shape.to_vec(),
            op: "scatter_add - indices and src must have the same shape".to_string(),
        });
    }

    // Create output by copying input
    let input_size = layout.get_size();
    let mut result = Vec::with_capacity(input_size);

    if layout.is_contiguous() {
        result.extend_from_slice(&storage[offset..offset + input_size]);
    } else {
        let mut input_indices = vec![0; ndim];
        for _ in 0..input_size {
            let mut flat_index = offset;
            for i in 0..ndim {
                flat_index += input_indices[i] * strides[i];
            }
            result.push(storage[flat_index]);

            for i in (0..ndim).rev() {
                input_indices[i] += 1;
                if input_indices[i] < shape[i] {
                    break;
                }
                input_indices[i] = 0;
            }
        }
    }

    // Scatter-add src values according to indices
    let mut scatter_indices = vec![0; ndim];
    for _ in 0..src_size {
        let mut indices_idx = indices_offset;
        for i in 0..ndim {
            indices_idx += scatter_indices[i] * indices_strides[i];
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        let mut src_idx = src_offset;
        for i in 0..ndim {
            src_idx += scatter_indices[i] * src_strides[i];
        }
        let src_val = src_storage[src_idx];

        let mut output_idx = 0;
        let mut stride = 1;
        for i in (0..ndim).rev() {
            let actual_idx = if i == dim { idx as usize } else { scatter_indices[i] };
            output_idx += actual_idx * stride;
            stride *= shape[i];
        }

        result[output_idx] = result[output_idx] + src_val;

        for i in (0..ndim).rev() {
            scatter_indices[i] += 1;
            if scatter_indices[i] < src_shape[i] {
                break;
            }
            scatter_indices[i] = 0;
        }
    }

    Ok(result)
}

pub fn scatter_max_map<T: Copy + PartialOrd>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    src_storage: &[T],
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "scatter_max - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_shape = indices_layout.get_shape();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let src_size = src_layout.get_size();

    if indices_shape != src_shape {
        return Err(HoduError::IncompatibleShapes {
            lhs: indices_shape.to_vec(),
            rhs: src_shape.to_vec(),
            op: "scatter_max - indices and src must have the same shape".to_string(),
        });
    }

    // Create output by copying input
    let input_size = layout.get_size();
    let mut result = Vec::with_capacity(input_size);

    if layout.is_contiguous() {
        result.extend_from_slice(&storage[offset..offset + input_size]);
    } else {
        let mut input_indices = vec![0; ndim];
        for _ in 0..input_size {
            let mut flat_index = offset;
            for i in 0..ndim {
                flat_index += input_indices[i] * strides[i];
            }
            result.push(storage[flat_index]);

            for i in (0..ndim).rev() {
                input_indices[i] += 1;
                if input_indices[i] < shape[i] {
                    break;
                }
                input_indices[i] = 0;
            }
        }
    }

    // Scatter-max src values according to indices
    let mut scatter_indices = vec![0; ndim];
    for _ in 0..src_size {
        let mut indices_idx = indices_offset;
        for i in 0..ndim {
            indices_idx += scatter_indices[i] * indices_strides[i];
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        let mut src_idx = src_offset;
        for i in 0..ndim {
            src_idx += scatter_indices[i] * src_strides[i];
        }
        let src_val = src_storage[src_idx];

        let mut output_idx = 0;
        let mut stride = 1;
        for i in (0..ndim).rev() {
            let actual_idx = if i == dim { idx as usize } else { scatter_indices[i] };
            output_idx += actual_idx * stride;
            stride *= shape[i];
        }

        if src_val > result[output_idx] {
            result[output_idx] = src_val;
        }

        for i in (0..ndim).rev() {
            scatter_indices[i] += 1;
            if scatter_indices[i] < src_shape[i] {
                break;
            }
            scatter_indices[i] = 0;
        }
    }

    Ok(result)
}

pub fn scatter_min_map<T: Copy + PartialOrd>(
    storage: &[T],
    layout: &Layout,
    indices_storage: &[i32],
    indices_layout: &Layout,
    src_storage: &[T],
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<Vec<T>> {
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::InternalError(format!(
            "scatter_min - dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    let indices_shape = indices_layout.get_shape();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let src_size = src_layout.get_size();

    if indices_shape != src_shape {
        return Err(HoduError::IncompatibleShapes {
            lhs: indices_shape.to_vec(),
            rhs: src_shape.to_vec(),
            op: "scatter_min - indices and src must have the same shape".to_string(),
        });
    }

    // Create output by copying input
    let input_size = layout.get_size();
    let mut result = Vec::with_capacity(input_size);

    if layout.is_contiguous() {
        result.extend_from_slice(&storage[offset..offset + input_size]);
    } else {
        let mut input_indices = vec![0; ndim];
        for _ in 0..input_size {
            let mut flat_index = offset;
            for i in 0..ndim {
                flat_index += input_indices[i] * strides[i];
            }
            result.push(storage[flat_index]);

            for i in (0..ndim).rev() {
                input_indices[i] += 1;
                if input_indices[i] < shape[i] {
                    break;
                }
                input_indices[i] = 0;
            }
        }
    }

    // Scatter-min src values according to indices
    let mut scatter_indices = vec![0; ndim];
    for _ in 0..src_size {
        let mut indices_idx = indices_offset;
        for i in 0..ndim {
            indices_idx += scatter_indices[i] * indices_strides[i];
        }

        let idx = indices_storage[indices_idx];
        if idx < 0 || idx >= shape[dim] as i32 {
            return Err(HoduError::InternalError(format!(
                "index {} out of bounds for dimension {} with size {}",
                idx, dim, shape[dim]
            )));
        }

        let mut src_idx = src_offset;
        for i in 0..ndim {
            src_idx += scatter_indices[i] * src_strides[i];
        }
        let src_val = src_storage[src_idx];

        let mut output_idx = 0;
        let mut stride = 1;
        for i in (0..ndim).rev() {
            let actual_idx = if i == dim { idx as usize } else { scatter_indices[i] };
            output_idx += actual_idx * stride;
            stride *= shape[i];
        }

        if src_val < result[output_idx] {
            result[output_idx] = src_val;
        }

        for i in (0..ndim).rev() {
            scatter_indices[i] += 1;
            if scatter_indices[i] < src_shape[i] {
                break;
            }
            scatter_indices[i] = 0;
        }
    }

    Ok(result)
}

// Conv1D
pub fn conv1d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConv1D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, length]
    // Weight: [out_channels, in_channels, kernel_size]
    // Output: [batch, out_channels, output_length]

    let output_length =
        (params.length_input + 2 * params.padding - params.dilation * (params.kernel_size - 1) - 1) / params.stride + 1;

    let output_size = params.batch_size * params.channels_output * output_length;
    let mut result = vec![T::default(); output_size];

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    // Iterate over batch
    for b in 0..params.batch_size {
        // Iterate over output channels
        for oc in 0..params.channels_output {
            // Iterate over output positions
            for ol in 0..output_length {
                let mut sum = T::default();

                // Iterate over input channels
                for ic in 0..params.channels_input {
                    // Iterate over kernel
                    for k in 0..params.kernel_size {
                        // Calculate input position with padding and dilation
                        let il = ol * params.stride + k * params.dilation;

                        // Check if position is within padded input
                        if il < params.padding || il >= params.length_input + params.padding {
                            continue; // Zero padding
                        }

                        let il_actual = il - params.padding;

                        // Get input value
                        let input_idx =
                            input_offset + b * input_strides[0] + ic * input_strides[1] + il_actual * input_strides[2];
                        let input_val = input_storage[input_idx];

                        // Get weight value
                        let weight_idx =
                            weight_offset + oc * weight_strides[0] + ic * weight_strides[1] + k * weight_strides[2];
                        let weight_val = weight_storage[weight_idx];

                        sum = sum + input_val * weight_val;
                    }
                }

                // Store result
                let output_idx = b * params.channels_output * output_length + oc * output_length + ol;
                result[output_idx] = sum;
            }
        }
    }

    Ok(result)
}

// Conv2D
pub fn conv2d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConv2D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, height, width]
    // Weight: [out_channels, in_channels, kernel_h, kernel_w]
    // Output: [batch, out_channels, out_height, out_width]

    let output_height = (params.input_height + 2 * params.padding - params.dilation * (params.kernel_height - 1) - 1)
        / params.stride
        + 1;
    let output_width =
        (params.input_width + 2 * params.padding - params.dilation * (params.kernel_width - 1) - 1) / params.stride + 1;

    let output_size = params.batch_size * params.channels_output * output_height * output_width;
    let mut result = vec![T::default(); output_size];

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    // Iterate over batch
    for b in 0..params.batch_size {
        // Iterate over output channels
        for oc in 0..params.channels_output {
            // Iterate over output height
            for oh in 0..output_height {
                // Iterate over output width
                for ow in 0..output_width {
                    let mut sum = T::default();

                    // Iterate over input channels
                    for ic in 0..params.channels_input {
                        // Iterate over kernel height
                        for kh in 0..params.kernel_height {
                            // Iterate over kernel width
                            for kw in 0..params.kernel_width {
                                // Calculate input position
                                let ih = oh * params.stride + kh * params.dilation;
                                let iw = ow * params.stride + kw * params.dilation;

                                // Check bounds with padding
                                if ih < params.padding
                                    || ih >= params.input_height + params.padding
                                    || iw < params.padding
                                    || iw >= params.input_width + params.padding
                                {
                                    continue; // Zero padding
                                }

                                let ih_actual = ih - params.padding;
                                let iw_actual = iw - params.padding;

                                // Get input value
                                let input_idx = input_offset
                                    + b * input_strides[0]
                                    + ic * input_strides[1]
                                    + ih_actual * input_strides[2]
                                    + iw_actual * input_strides[3];
                                let input_val = input_storage[input_idx];

                                // Get weight value
                                let weight_idx = weight_offset
                                    + oc * weight_strides[0]
                                    + ic * weight_strides[1]
                                    + kh * weight_strides[2]
                                    + kw * weight_strides[3];
                                let weight_val = weight_storage[weight_idx];

                                sum = sum + input_val * weight_val;
                            }
                        }
                    }

                    // Store result
                    let output_idx = b * params.channels_output * output_height * output_width
                        + oc * output_height * output_width
                        + oh * output_width
                        + ow;
                    result[output_idx] = sum;
                }
            }
        }
    }

    Ok(result)
}

// Conv3D
pub fn conv3d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConv3D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, depth, height, width]
    // Weight: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
    // Output: [batch, out_channels, out_depth, out_height, out_width]

    let output_depth =
        (params.input_depth + 2 * params.padding - params.dilation * (params.kernel_depth - 1) - 1) / params.stride + 1;
    let output_height = (params.input_height + 2 * params.padding - params.dilation * (params.kernel_height - 1) - 1)
        / params.stride
        + 1;
    let output_width =
        (params.input_width + 2 * params.padding - params.dilation * (params.kernel_width - 1) - 1) / params.stride + 1;

    let output_size = params.batch_size * params.channels_output * output_depth * output_height * output_width;
    let mut result = vec![T::default(); output_size];

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    // Iterate over batch
    for b in 0..params.batch_size {
        // Iterate over output channels
        for oc in 0..params.channels_output {
            // Iterate over output depth
            for od in 0..output_depth {
                // Iterate over output height
                for oh in 0..output_height {
                    // Iterate over output width
                    for ow in 0..output_width {
                        let mut sum = T::default();

                        // Iterate over input channels
                        for ic in 0..params.channels_input {
                            // Iterate over kernel depth
                            for kd in 0..params.kernel_depth {
                                // Iterate over kernel height
                                for kh in 0..params.kernel_height {
                                    // Iterate over kernel width
                                    for kw in 0..params.kernel_width {
                                        // Calculate input position
                                        let id = od * params.stride + kd * params.dilation;
                                        let ih = oh * params.stride + kh * params.dilation;
                                        let iw = ow * params.stride + kw * params.dilation;

                                        // Check bounds with padding
                                        if id < params.padding
                                            || id >= params.input_depth + params.padding
                                            || ih < params.padding
                                            || ih >= params.input_height + params.padding
                                            || iw < params.padding
                                            || iw >= params.input_width + params.padding
                                        {
                                            continue; // Zero padding
                                        }

                                        let id_actual = id - params.padding;
                                        let ih_actual = ih - params.padding;
                                        let iw_actual = iw - params.padding;

                                        // Get input value
                                        let input_idx = input_offset
                                            + b * input_strides[0]
                                            + ic * input_strides[1]
                                            + id_actual * input_strides[2]
                                            + ih_actual * input_strides[3]
                                            + iw_actual * input_strides[4];
                                        let input_val = input_storage[input_idx];

                                        // Get weight value
                                        let weight_idx = weight_offset
                                            + oc * weight_strides[0]
                                            + ic * weight_strides[1]
                                            + kd * weight_strides[2]
                                            + kh * weight_strides[3]
                                            + kw * weight_strides[4];
                                        let weight_val = weight_storage[weight_idx];

                                        sum = sum + input_val * weight_val;
                                    }
                                }
                            }
                        }

                        // Store result
                        let output_idx = b * params.channels_output * output_depth * output_height * output_width
                            + oc * output_depth * output_height * output_width
                            + od * output_height * output_width
                            + oh * output_width
                            + ow;
                        result[output_idx] = sum;
                    }
                }
            }
        }
    }

    Ok(result)
}

// ConvTranspose1D
pub fn conv_transpose1d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConvTranspose1D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, length]
    // Weight: [in_channels, out_channels, kernel_size] (note: different from conv!)
    // Output: [batch, out_channels, output_length]

    let output_length = (params.length_input - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_size - 1)
        + params.output_padding
        + 1;

    let output_size = params.batch_size * params.channels_output * output_length;
    let mut result = vec![T::default(); output_size];

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    // Iterate over batch
    for b in 0..params.batch_size {
        // Iterate over input channels
        for ic in 0..params.channels_input {
            // Iterate over input positions
            for il in 0..params.length_input {
                // Get input value
                let input_idx = input_offset + b * input_strides[0] + ic * input_strides[1] + il * input_strides[2];
                let input_val = input_storage[input_idx];

                // Iterate over output channels
                for oc in 0..params.channels_output {
                    // Get weight value
                    let weight_idx = weight_offset + ic * weight_strides[0] + oc * weight_strides[1];

                    // Iterate over kernel
                    for k in 0..params.kernel_size {
                        // Calculate output position
                        let ol = il * params.stride + k * params.dilation;

                        // Adjust for padding
                        if ol < params.padding {
                            continue;
                        }
                        let ol_actual = ol - params.padding;

                        if ol_actual >= output_length {
                            continue;
                        }

                        // Get weight value for this kernel position
                        let weight_idx_k = weight_idx + k * weight_strides[2];
                        let weight_val = weight_storage[weight_idx_k];

                        // Accumulate to output
                        let output_idx = b * params.channels_output * output_length + oc * output_length + ol_actual;
                        result[output_idx] = result[output_idx] + input_val * weight_val;
                    }
                }
            }
        }
    }

    Ok(result)
}

// ConvTranspose2D
pub fn conv_transpose2d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConvTranspose2D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, height, width]
    // Weight: [in_channels, out_channels, kernel_h, kernel_w]
    // Output: [batch, out_channels, out_height, out_width]

    let output_height = (params.input_height - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_height - 1)
        + params.output_padding
        + 1;
    let output_width = (params.input_width - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_width - 1)
        + params.output_padding
        + 1;

    let output_size = params.batch_size * params.channels_output * output_height * output_width;
    let mut result = vec![T::default(); output_size];

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    // Iterate over batch
    for b in 0..params.batch_size {
        // Iterate over input channels
        for ic in 0..params.channels_input {
            // Iterate over input height
            for ih in 0..params.input_height {
                // Iterate over input width
                for iw in 0..params.input_width {
                    // Get input value
                    let input_idx = input_offset
                        + b * input_strides[0]
                        + ic * input_strides[1]
                        + ih * input_strides[2]
                        + iw * input_strides[3];
                    let input_val = input_storage[input_idx];

                    // Iterate over output channels
                    for oc in 0..params.channels_output {
                        // Get base weight index
                        let weight_base_idx = weight_offset + ic * weight_strides[0] + oc * weight_strides[1];

                        // Iterate over kernel height
                        for kh in 0..params.kernel_height {
                            // Iterate over kernel width
                            for kw in 0..params.kernel_width {
                                // Calculate output position
                                let oh = ih * params.stride + kh * params.dilation;
                                let ow = iw * params.stride + kw * params.dilation;

                                // Adjust for padding
                                if oh < params.padding || ow < params.padding {
                                    continue;
                                }
                                let oh_actual = oh - params.padding;
                                let ow_actual = ow - params.padding;

                                if oh_actual >= output_height || ow_actual >= output_width {
                                    continue;
                                }

                                // Get weight value
                                let weight_idx = weight_base_idx + kh * weight_strides[2] + kw * weight_strides[3];
                                let weight_val = weight_storage[weight_idx];

                                // Accumulate to output
                                let output_idx = b * params.channels_output * output_height * output_width
                                    + oc * output_height * output_width
                                    + oh_actual * output_width
                                    + ow_actual;
                                result[output_idx] = result[output_idx] + input_val * weight_val;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

// ConvTranspose3D
pub fn conv_transpose3d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConvTranspose3D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, depth, height, width]
    // Weight: [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
    // Output: [batch, out_channels, out_depth, out_height, out_width]

    let output_depth = (params.input_depth - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_depth - 1)
        + params.output_padding
        + 1;
    let output_height = (params.input_height - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_height - 1)
        + params.output_padding
        + 1;
    let output_width = (params.input_width - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_width - 1)
        + params.output_padding
        + 1;

    let output_size = params.batch_size * params.channels_output * output_depth * output_height * output_width;
    let mut result = vec![T::default(); output_size];

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    // Iterate over batch
    for b in 0..params.batch_size {
        // Iterate over input channels
        for ic in 0..params.channels_input {
            // Iterate over input depth
            for id in 0..params.input_depth {
                // Iterate over input height
                for ih in 0..params.input_height {
                    // Iterate over input width
                    for iw in 0..params.input_width {
                        // Get input value
                        let input_idx = input_offset
                            + b * input_strides[0]
                            + ic * input_strides[1]
                            + id * input_strides[2]
                            + ih * input_strides[3]
                            + iw * input_strides[4];
                        let input_val = input_storage[input_idx];

                        // Iterate over output channels
                        for oc in 0..params.channels_output {
                            // Get base weight index
                            let weight_base_idx = weight_offset + ic * weight_strides[0] + oc * weight_strides[1];

                            // Iterate over kernel depth
                            for kd in 0..params.kernel_depth {
                                // Iterate over kernel height
                                for kh in 0..params.kernel_height {
                                    // Iterate over kernel width
                                    for kw in 0..params.kernel_width {
                                        // Calculate output position
                                        let od = id * params.stride + kd * params.dilation;
                                        let oh = ih * params.stride + kh * params.dilation;
                                        let ow = iw * params.stride + kw * params.dilation;

                                        // Adjust for padding
                                        if od < params.padding || oh < params.padding || ow < params.padding {
                                            continue;
                                        }
                                        let od_actual = od - params.padding;
                                        let oh_actual = oh - params.padding;
                                        let ow_actual = ow - params.padding;

                                        if od_actual >= output_depth
                                            || oh_actual >= output_height
                                            || ow_actual >= output_width
                                        {
                                            continue;
                                        }

                                        // Get weight value
                                        let weight_idx = weight_base_idx
                                            + kd * weight_strides[2]
                                            + kh * weight_strides[3]
                                            + kw * weight_strides[4];
                                        let weight_val = weight_storage[weight_idx];

                                        // Accumulate to output
                                        let output_idx =
                                            b * params.channels_output * output_depth * output_height * output_width
                                                + oc * output_depth * output_height * output_width
                                                + od_actual * output_height * output_width
                                                + oh_actual * output_width
                                                + ow_actual;
                                        result[output_idx] = result[output_idx] + input_val * weight_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

// Convolution Gradient Weight operations

/// Conv1d weight gradient: computes gradient of weight given input and grad_output
/// Input: [batch, in_channels, length]
/// GradOutput: [batch, out_channels, output_length]
/// Weight gradient: [out_channels, in_channels, kernel_size]
pub fn conv1d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConv1D,
) -> HoduResult<Vec<T>> {
    let output_length =
        (params.length_input + 2 * params.padding - params.dilation * (params.kernel_size - 1) - 1) / params.stride + 1;

    // Weight gradient shape: [out_channels, in_channels, kernel_size]
    let weight_grad_size = params.channels_output * params.channels_input * params.kernel_size;
    let mut weight_grad = vec![T::default(); weight_grad_size];

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    // Iterate over output channels
    for oc in 0..params.channels_output {
        // Iterate over input channels
        for ic in 0..params.channels_input {
            // Iterate over kernel positions
            for k in 0..params.kernel_size {
                let mut sum = T::default();

                // Iterate over batch
                for b in 0..params.batch_size {
                    // Iterate over output positions
                    for ol in 0..output_length {
                        // Calculate input position
                        let il = ol * params.stride + k * params.dilation;

                        // Check if position is within padded input
                        if il < params.padding || il >= params.length_input + params.padding {
                            continue; // Zero padding
                        }

                        let il_actual = il - params.padding;

                        // Get input value
                        let input_idx =
                            input_offset + b * input_strides[0] + ic * input_strides[1] + il_actual * input_strides[2];
                        let input_val = input_storage[input_idx];

                        // Get grad_output value
                        let grad_output_idx = grad_output_offset
                            + b * grad_output_strides[0]
                            + oc * grad_output_strides[1]
                            + ol * grad_output_strides[2];
                        let grad_output_val = grad_output_storage[grad_output_idx];

                        sum = sum + input_val * grad_output_val;
                    }
                }

                // Store weight gradient
                let weight_grad_idx = oc * params.channels_input * params.kernel_size + ic * params.kernel_size + k;
                weight_grad[weight_grad_idx] = sum;
            }
        }
    }

    Ok(weight_grad)
}

/// Conv2d weight gradient
/// Input: [batch, in_channels, height, width]
/// GradOutput: [batch, out_channels, out_height, out_width]
/// Weight gradient: [out_channels, in_channels, kernel_h, kernel_w]
pub fn conv2d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConv2D,
) -> HoduResult<Vec<T>> {
    let output_height = (params.input_height + 2 * params.padding - params.dilation * (params.kernel_height - 1) - 1)
        / params.stride
        + 1;
    let output_width =
        (params.input_width + 2 * params.padding - params.dilation * (params.kernel_width - 1) - 1) / params.stride + 1;

    // Weight gradient shape: [out_channels, in_channels, kernel_h, kernel_w]
    let weight_grad_size = params.channels_output * params.channels_input * params.kernel_height * params.kernel_width;
    let mut weight_grad = vec![T::default(); weight_grad_size];

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    // Iterate over output channels
    for oc in 0..params.channels_output {
        // Iterate over input channels
        for ic in 0..params.channels_input {
            // Iterate over kernel height
            for kh in 0..params.kernel_height {
                // Iterate over kernel width
                for kw in 0..params.kernel_width {
                    let mut sum = T::default();

                    // Iterate over batch
                    for b in 0..params.batch_size {
                        // Iterate over output height
                        for oh in 0..output_height {
                            // Iterate over output width
                            for ow in 0..output_width {
                                // Calculate input position
                                let ih = oh * params.stride + kh * params.dilation;
                                let iw = ow * params.stride + kw * params.dilation;

                                // Check bounds with padding
                                if ih < params.padding
                                    || ih >= params.input_height + params.padding
                                    || iw < params.padding
                                    || iw >= params.input_width + params.padding
                                {
                                    continue; // Zero padding
                                }

                                let ih_actual = ih - params.padding;
                                let iw_actual = iw - params.padding;

                                // Get input value
                                let input_idx = input_offset
                                    + b * input_strides[0]
                                    + ic * input_strides[1]
                                    + ih_actual * input_strides[2]
                                    + iw_actual * input_strides[3];
                                let input_val = input_storage[input_idx];

                                // Get grad_output value
                                let grad_output_idx = grad_output_offset
                                    + b * grad_output_strides[0]
                                    + oc * grad_output_strides[1]
                                    + oh * grad_output_strides[2]
                                    + ow * grad_output_strides[3];
                                let grad_output_val = grad_output_storage[grad_output_idx];

                                sum = sum + input_val * grad_output_val;
                            }
                        }
                    }

                    // Store weight gradient
                    let weight_grad_idx = oc * params.channels_input * params.kernel_height * params.kernel_width
                        + ic * params.kernel_height * params.kernel_width
                        + kh * params.kernel_width
                        + kw;
                    weight_grad[weight_grad_idx] = sum;
                }
            }
        }
    }

    Ok(weight_grad)
}

/// Conv3d weight gradient
/// Input: [batch, in_channels, depth, height, width]
/// GradOutput: [batch, out_channels, out_depth, out_height, out_width]
/// Weight gradient: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
pub fn conv3d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConv3D,
) -> HoduResult<Vec<T>> {
    let output_depth =
        (params.input_depth + 2 * params.padding - params.dilation * (params.kernel_depth - 1) - 1) / params.stride + 1;
    let output_height = (params.input_height + 2 * params.padding - params.dilation * (params.kernel_height - 1) - 1)
        / params.stride
        + 1;
    let output_width =
        (params.input_width + 2 * params.padding - params.dilation * (params.kernel_width - 1) - 1) / params.stride + 1;

    // Weight gradient shape: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
    let weight_grad_size = params.channels_output
        * params.channels_input
        * params.kernel_depth
        * params.kernel_height
        * params.kernel_width;
    let mut weight_grad = vec![T::default(); weight_grad_size];

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    // Iterate over output channels
    for oc in 0..params.channels_output {
        // Iterate over input channels
        for ic in 0..params.channels_input {
            // Iterate over kernel depth
            for kd in 0..params.kernel_depth {
                // Iterate over kernel height
                for kh in 0..params.kernel_height {
                    // Iterate over kernel width
                    for kw in 0..params.kernel_width {
                        let mut sum = T::default();

                        // Iterate over batch
                        for b in 0..params.batch_size {
                            // Iterate over output depth
                            for od in 0..output_depth {
                                // Iterate over output height
                                for oh in 0..output_height {
                                    // Iterate over output width
                                    for ow in 0..output_width {
                                        // Calculate input position
                                        let id = od * params.stride + kd * params.dilation;
                                        let ih = oh * params.stride + kh * params.dilation;
                                        let iw = ow * params.stride + kw * params.dilation;

                                        // Check bounds with padding
                                        if id < params.padding
                                            || id >= params.input_depth + params.padding
                                            || ih < params.padding
                                            || ih >= params.input_height + params.padding
                                            || iw < params.padding
                                            || iw >= params.input_width + params.padding
                                        {
                                            continue; // Zero padding
                                        }

                                        let id_actual = id - params.padding;
                                        let ih_actual = ih - params.padding;
                                        let iw_actual = iw - params.padding;

                                        // Get input value
                                        let input_idx = input_offset
                                            + b * input_strides[0]
                                            + ic * input_strides[1]
                                            + id_actual * input_strides[2]
                                            + ih_actual * input_strides[3]
                                            + iw_actual * input_strides[4];
                                        let input_val = input_storage[input_idx];

                                        // Get grad_output value
                                        let grad_output_idx = grad_output_offset
                                            + b * grad_output_strides[0]
                                            + oc * grad_output_strides[1]
                                            + od * grad_output_strides[2]
                                            + oh * grad_output_strides[3]
                                            + ow * grad_output_strides[4];
                                        let grad_output_val = grad_output_storage[grad_output_idx];

                                        sum = sum + input_val * grad_output_val;
                                    }
                                }
                            }
                        }

                        // Store weight gradient
                        let weight_grad_idx = oc
                            * params.channels_input
                            * params.kernel_depth
                            * params.kernel_height
                            * params.kernel_width
                            + ic * params.kernel_depth * params.kernel_height * params.kernel_width
                            + kd * params.kernel_height * params.kernel_width
                            + kh * params.kernel_width
                            + kw;
                        weight_grad[weight_grad_idx] = sum;
                    }
                }
            }
        }
    }

    Ok(weight_grad)
}

// ConvTranspose Gradient Weight operations

/// ConvTranspose1d weight gradient
/// Input: [batch, in_channels, length_in]
/// GradOutput: [batch, out_channels, length_out]
/// Weight gradient: [in_channels, out_channels, kernel_size]
pub fn conv_transpose1d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConvTranspose1D,
) -> HoduResult<Vec<T>> {
    let output_length = (params.length_input - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_size - 1)
        + params.output_padding
        + 1;

    // Weight gradient shape: [in_channels, out_channels, kernel_size]
    let weight_grad_size = params.channels_input * params.channels_output * params.kernel_size;
    let mut weight_grad = vec![T::default(); weight_grad_size];

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    // Iterate over input channels (first dimension of weight in ConvTranspose)
    for ic in 0..params.channels_input {
        // Iterate over output channels
        for oc in 0..params.channels_output {
            // Iterate over kernel positions
            for k in 0..params.kernel_size {
                let mut sum = T::default();

                // Iterate over batch
                for b in 0..params.batch_size {
                    // Iterate over input positions
                    for il in 0..params.length_input {
                        // For each input position, find corresponding output positions
                        let out_start = il * params.stride + k * params.dilation;

                        // Check if this output position exists (with padding)
                        if out_start < params.padding || out_start >= output_length + params.padding {
                            continue;
                        }

                        let ol = out_start - params.padding;

                        // Get input value
                        let input_idx =
                            input_offset + b * input_strides[0] + ic * input_strides[1] + il * input_strides[2];
                        let input_val = input_storage[input_idx];

                        // Get grad_output value
                        let grad_output_idx = grad_output_offset
                            + b * grad_output_strides[0]
                            + oc * grad_output_strides[1]
                            + ol * grad_output_strides[2];
                        let grad_output_val = grad_output_storage[grad_output_idx];

                        sum = sum + input_val * grad_output_val;
                    }
                }

                // Store weight gradient
                let weight_grad_idx = ic * params.channels_output * params.kernel_size + oc * params.kernel_size + k;
                weight_grad[weight_grad_idx] = sum;
            }
        }
    }

    Ok(weight_grad)
}

/// ConvTranspose2d weight gradient
/// Input: [batch, in_channels, height_in, width_in]
/// GradOutput: [batch, out_channels, height_out, width_out]
/// Weight gradient: [in_channels, out_channels, kernel_h, kernel_w]
pub fn conv_transpose2d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConvTranspose2D,
) -> HoduResult<Vec<T>> {
    let output_height = (params.input_height - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_height - 1)
        + params.output_padding
        + 1;
    let output_width = (params.input_width - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_width - 1)
        + params.output_padding
        + 1;

    // Weight gradient shape: [in_channels, out_channels, kernel_h, kernel_w]
    let weight_grad_size = params.channels_input * params.channels_output * params.kernel_height * params.kernel_width;
    let mut weight_grad = vec![T::default(); weight_grad_size];

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    // Iterate over input channels
    for ic in 0..params.channels_input {
        // Iterate over output channels
        for oc in 0..params.channels_output {
            // Iterate over kernel height
            for kh in 0..params.kernel_height {
                // Iterate over kernel width
                for kw in 0..params.kernel_width {
                    let mut sum = T::default();

                    // Iterate over batch
                    for b in 0..params.batch_size {
                        // Iterate over input height
                        for ih in 0..params.input_height {
                            // Iterate over input width
                            for iw in 0..params.input_width {
                                // Calculate output position
                                let oh_before_pad = ih * params.stride + kh * params.dilation;
                                let ow_before_pad = iw * params.stride + kw * params.dilation;

                                // Check bounds with padding
                                if oh_before_pad < params.padding
                                    || oh_before_pad >= output_height + params.padding
                                    || ow_before_pad < params.padding
                                    || ow_before_pad >= output_width + params.padding
                                {
                                    continue;
                                }

                                let oh = oh_before_pad - params.padding;
                                let ow = ow_before_pad - params.padding;

                                // Get input value
                                let input_idx = input_offset
                                    + b * input_strides[0]
                                    + ic * input_strides[1]
                                    + ih * input_strides[2]
                                    + iw * input_strides[3];
                                let input_val = input_storage[input_idx];

                                // Get grad_output value
                                let grad_output_idx = grad_output_offset
                                    + b * grad_output_strides[0]
                                    + oc * grad_output_strides[1]
                                    + oh * grad_output_strides[2]
                                    + ow * grad_output_strides[3];
                                let grad_output_val = grad_output_storage[grad_output_idx];

                                sum = sum + input_val * grad_output_val;
                            }
                        }
                    }

                    // Store weight gradient
                    let weight_grad_idx = ic * params.channels_output * params.kernel_height * params.kernel_width
                        + oc * params.kernel_height * params.kernel_width
                        + kh * params.kernel_width
                        + kw;
                    weight_grad[weight_grad_idx] = sum;
                }
            }
        }
    }

    Ok(weight_grad)
}

/// ConvTranspose3d weight gradient
/// Input: [batch, in_channels, depth_in, height_in, width_in]
/// GradOutput: [batch, out_channels, depth_out, height_out, width_out]
/// Weight gradient: [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
pub fn conv_transpose3d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConvTranspose3D,
) -> HoduResult<Vec<T>> {
    let output_depth = (params.input_depth - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_depth - 1)
        + params.output_padding
        + 1;
    let output_height = (params.input_height - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_height - 1)
        + params.output_padding
        + 1;
    let output_width = (params.input_width - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_width - 1)
        + params.output_padding
        + 1;

    // Weight gradient shape: [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
    let weight_grad_size = params.channels_input
        * params.channels_output
        * params.kernel_depth
        * params.kernel_height
        * params.kernel_width;
    let mut weight_grad = vec![T::default(); weight_grad_size];

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    // Iterate over input channels
    for ic in 0..params.channels_input {
        // Iterate over output channels
        for oc in 0..params.channels_output {
            // Iterate over kernel depth
            for kd in 0..params.kernel_depth {
                // Iterate over kernel height
                for kh in 0..params.kernel_height {
                    // Iterate over kernel width
                    for kw in 0..params.kernel_width {
                        let mut sum = T::default();

                        // Iterate over batch
                        for b in 0..params.batch_size {
                            // Iterate over input depth
                            for id in 0..params.input_depth {
                                // Iterate over input height
                                for ih in 0..params.input_height {
                                    // Iterate over input width
                                    for iw in 0..params.input_width {
                                        // Calculate output position
                                        let od_before_pad = id * params.stride + kd * params.dilation;
                                        let oh_before_pad = ih * params.stride + kh * params.dilation;
                                        let ow_before_pad = iw * params.stride + kw * params.dilation;

                                        // Check bounds with padding
                                        if od_before_pad < params.padding
                                            || od_before_pad >= output_depth + params.padding
                                            || oh_before_pad < params.padding
                                            || oh_before_pad >= output_height + params.padding
                                            || ow_before_pad < params.padding
                                            || ow_before_pad >= output_width + params.padding
                                        {
                                            continue;
                                        }

                                        let od = od_before_pad - params.padding;
                                        let oh = oh_before_pad - params.padding;
                                        let ow = ow_before_pad - params.padding;

                                        // Get input value
                                        let input_idx = input_offset
                                            + b * input_strides[0]
                                            + ic * input_strides[1]
                                            + id * input_strides[2]
                                            + ih * input_strides[3]
                                            + iw * input_strides[4];
                                        let input_val = input_storage[input_idx];

                                        // Get grad_output value
                                        let grad_output_idx = grad_output_offset
                                            + b * grad_output_strides[0]
                                            + oc * grad_output_strides[1]
                                            + od * grad_output_strides[2]
                                            + oh * grad_output_strides[3]
                                            + ow * grad_output_strides[4];
                                        let grad_output_val = grad_output_storage[grad_output_idx];

                                        sum = sum + input_val * grad_output_val;
                                    }
                                }
                            }
                        }

                        // Store weight gradient
                        let weight_grad_idx = ic
                            * params.channels_output
                            * params.kernel_depth
                            * params.kernel_height
                            * params.kernel_width
                            + oc * params.kernel_depth * params.kernel_height * params.kernel_width
                            + kd * params.kernel_height * params.kernel_width
                            + kh * params.kernel_width
                            + kw;
                        weight_grad[weight_grad_idx] = sum;
                    }
                }
            }
        }
    }

    Ok(weight_grad)
}
