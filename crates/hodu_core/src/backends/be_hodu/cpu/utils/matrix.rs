use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    types::layout::Layout,
};

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
