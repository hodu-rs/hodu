use crate::{
    backends::be_hodu::cpu::simd,
    compat::*,
    error::{HoduError, HoduResult},
    types::layout::Layout,
};

pub fn matmul_map<
    T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default + 'static + Send + Sync,
>(
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

    // Try SIMD fast path for simple contiguous case
    let is_contiguous = lhs_layout.is_contiguous() && rhs_layout.is_contiguous();
    let is_row_major = lhs_strides[lhs_ndim - 1] == 1 && rhs_strides[rhs_ndim - 1] == 1;
    let is_simple = total_batches == 1 && lhs_offset == 0 && rhs_offset == 0;

    if is_contiguous && is_row_major && is_simple {
        use core::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const f32, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const f32, rhs_storage.len()) };
            let c = simd::ops::matmul::f32(a, b, lhs_m, lhs_k, rhs_n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const f64, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const f64, rhs_storage.len()) };
            let c = simd::ops::matmul::f64(a, b, lhs_m, lhs_k, rhs_n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<u8>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const u8, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const u8, rhs_storage.len()) };
            let c = simd::ops::matmul::u8(a, b, lhs_m, lhs_k, rhs_n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<u16>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const u16, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const u16, rhs_storage.len()) };
            let c = simd::ops::matmul::u16(a, b, lhs_m, lhs_k, rhs_n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<u32>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const u32, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const u32, rhs_storage.len()) };
            let c = simd::ops::matmul::u32(a, b, lhs_m, lhs_k, rhs_n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<i8>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const i8, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const i8, rhs_storage.len()) };
            let c = simd::ops::matmul::i8(a, b, lhs_m, lhs_k, rhs_n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<i16>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const i16, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const i16, rhs_storage.len()) };
            let c = simd::ops::matmul::i16(a, b, lhs_m, lhs_k, rhs_n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<i32>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const i32, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const i32, rhs_storage.len()) };
            let c = simd::ops::matmul::i32(a, b, lhs_m, lhs_k, rhs_n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }
    }

    #[cfg(feature = "rayon")]
    let result = {
        let batch_results: Vec<Vec<T>> = (0..total_batches)
            .into_par_iter()
            .map(|batch_idx| {
                let mut batch_result = vec![T::default(); lhs_m * rhs_n];

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

                        batch_result[i * rhs_n + j] = sum;
                    }
                }

                batch_result
            })
            .collect();

        batch_results.into_iter().flatten().collect()
    };

    #[cfg(not(feature = "rayon"))]
    let result = {
        let result_size = total_batches * lhs_m * rhs_n;
        let mut result = vec![T::default(); result_size];
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
        result
    };

    Ok(result)
}

pub fn dot_map<T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default + 'static + Send + Sync>(
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

    // Try SIMD fast path for simple contiguous case
    let is_contiguous = lhs_layout.is_contiguous() && rhs_layout.is_contiguous();
    let is_row_major = lhs_strides[0] == k1 && lhs_strides[1] == 1 && rhs_strides[0] == n && rhs_strides[1] == 1;
    let is_simple = lhs_offset == 0 && rhs_offset == 0;

    if is_contiguous && is_row_major && is_simple {
        use core::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const f32, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const f32, rhs_storage.len()) };
            let c = simd::ops::dot::f32(a, b, m, k1, n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const f64, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const f64, rhs_storage.len()) };
            let c = simd::ops::dot::f64(a, b, m, k1, n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<u8>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const u8, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const u8, rhs_storage.len()) };
            let c = simd::ops::dot::u8(a, b, m, k1, n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<u16>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const u16, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const u16, rhs_storage.len()) };
            let c = simd::ops::dot::u16(a, b, m, k1, n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<u32>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const u32, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const u32, rhs_storage.len()) };
            let c = simd::ops::dot::u32(a, b, m, k1, n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<i8>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const i8, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const i8, rhs_storage.len()) };
            let c = simd::ops::dot::i8(a, b, m, k1, n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<i16>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const i16, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const i16, rhs_storage.len()) };
            let c = simd::ops::dot::i16(a, b, m, k1, n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }

        if TypeId::of::<T>() == TypeId::of::<i32>() {
            let a = unsafe { core::slice::from_raw_parts(lhs_storage.as_ptr() as *const i32, lhs_storage.len()) };
            let b = unsafe { core::slice::from_raw_parts(rhs_storage.as_ptr() as *const i32, rhs_storage.len()) };
            let c = simd::ops::dot::i32(a, b, m, k1, n);
            let result_typed = unsafe { core::slice::from_raw_parts(c.as_ptr() as *const T, c.len()) }.to_vec();
            return Ok(result_typed);
        }
    }

    #[cfg(feature = "rayon")]
    let result = {
        let row_results: Vec<Vec<T>> = (0..m)
            .into_par_iter()
            .map(|i| {
                let mut row_result = vec![T::default(); n];

                for j in 0..n {
                    let mut sum = T::default();

                    for k in 0..k1 {
                        let lhs_idx = lhs_offset + i * lhs_strides[0] + k * lhs_strides[1];
                        let rhs_idx = rhs_offset + k * rhs_strides[0] + j * rhs_strides[1];

                        let lhs_val = unsafe { *lhs_storage.get_unchecked(lhs_idx) };
                        let rhs_val = unsafe { *rhs_storage.get_unchecked(rhs_idx) };

                        sum = sum + lhs_val * rhs_val;
                    }

                    row_result[j] = sum;
                }

                row_result
            })
            .collect();

        row_results.into_iter().flatten().collect()
    };

    #[cfg(not(feature = "rayon"))]
    let result = {
        let mut result = vec![T::default(); m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();

                for k in 0..k1 {
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

        result
    };

    Ok(result)
}
