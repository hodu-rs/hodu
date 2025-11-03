#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]

use crate::{compat::*, error::HoduResult, types::layout::Layout};

#[cfg(feature = "rayon")]
const PARALLEL_THRESHOLD: usize = 4096;

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
    use rayon::prelude::*;

    let (sum_result, output_shape) = crate::be_hodu::cpu::utils::reduce_sum(storage, layout, dims, keep_dim)?;

    // Calculate the number of elements that were reduced
    let shape = layout.get_shape();
    let reduce_dims: Vec<usize> = if dims.is_empty() {
        (0..shape.len()).collect()
    } else {
        dims.to_vec()
    };

    let count = reduce_dims.iter().map(|&d| shape[d]).product::<usize>();
    let count_val = T::from(count).unwrap();

    if sum_result.len() >= PARALLEL_THRESHOLD {
        let mean_result = sum_result.par_iter().map(|&x| x / count_val).collect();
        Ok((mean_result, output_shape))
    } else {
        let mean_result = sum_result.into_iter().map(|x| x / count_val).collect();
        Ok((mean_result, output_shape))
    }
}

#[cfg(not(feature = "rayon"))]
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
    let (sum_result, output_shape) = crate::be_hodu::cpu::utils::reduce_sum(storage, layout, dims, keep_dim)?;

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
