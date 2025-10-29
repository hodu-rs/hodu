use crate::{compat::*, scalar::Scalar, types::layout::Layout};

#[cfg(feature = "rayon")]
const PARALLEL_THRESHOLD: usize = 4096;
#[cfg(feature = "rayon")]
const PARALLEL_CHUNK_SIZE: usize = 1024;

pub fn unary_map<T: Copy + Send + Sync, U: Copy + Send + Sync, F: Fn(T) -> U + Send + Sync>(
    storage: &[T],
    layout: &Layout,
    f: F,
) -> Vec<U> {
    let offset = layout.get_offset();
    let size = layout.get_size();

    if layout.is_contiguous() {
        #[cfg(feature = "rayon")]
        {
            if size >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;
                storage[offset..offset + size].par_iter().map(|&v| f(v)).collect()
            } else {
                storage[offset..offset + size].iter().map(|&v| f(v)).collect()
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            storage[offset..offset + size].iter().map(|&v| f(v)).collect()
        }
    } else {
        let dims = layout.get_shape();
        let strides = layout.get_strides();

        #[cfg(feature = "rayon")]
        {
            if size >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;
                let chunks = size.div_ceil(PARALLEL_CHUNK_SIZE);
                (0..chunks)
                    .into_par_iter()
                    .flat_map(|chunk_idx| {
                        let start = chunk_idx * PARALLEL_CHUNK_SIZE;
                        let end = (start + PARALLEL_CHUNK_SIZE).min(size);
                        let mut chunk_result = Vec::with_capacity(end - start);
                        for i in start..end {
                            let mut idx = offset;
                            let mut tmp_i = i;

                            for d in (0..dims.len()).rev() {
                                let i_dim = tmp_i % dims[d];
                                idx += if strides[d] != 0 { i_dim * strides[d] } else { 0 };
                                tmp_i /= dims[d];
                            }

                            let v = unsafe { storage.get_unchecked(idx) };
                            chunk_result.push(f(*v));
                        }
                        chunk_result
                    })
                    .collect()
            } else {
                let mut result = Vec::with_capacity(size);
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

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Vec::with_capacity(size);
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
}

pub fn unary_logical_map<T: Copy + Send + Sync, F: Fn(T) -> bool + Send + Sync>(
    storage: &[T],
    layout: &Layout,
    f: F,
) -> Vec<bool> {
    let offset = layout.get_offset();
    let size = layout.get_size();

    if layout.is_contiguous() {
        #[cfg(feature = "rayon")]
        {
            if size >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;
                storage[offset..offset + size].par_iter().map(|&v| f(v)).collect()
            } else {
                storage[offset..offset + size].iter().map(|&v| f(v)).collect()
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            storage[offset..offset + size].iter().map(|&v| f(v)).collect()
        }
    } else {
        let dims = layout.get_shape();
        let strides = layout.get_strides();

        #[cfg(feature = "rayon")]
        {
            if size >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;
                let chunks = size.div_ceil(PARALLEL_CHUNK_SIZE);
                (0..chunks)
                    .into_par_iter()
                    .flat_map(|chunk_idx| {
                        let start = chunk_idx * PARALLEL_CHUNK_SIZE;
                        let end = (start + PARALLEL_CHUNK_SIZE).min(size);
                        let mut chunk_result = Vec::with_capacity(end - start);
                        for i in start..end {
                            let mut idx = offset;
                            let mut tmp_i = i;

                            for d in (0..dims.len()).rev() {
                                let i_dim = tmp_i % dims[d];
                                idx += if strides[d] != 0 { i_dim * strides[d] } else { 0 };
                                tmp_i /= dims[d];
                            }

                            let v = unsafe { storage.get_unchecked(idx) };
                            chunk_result.push(f(*v));
                        }
                        chunk_result
                    })
                    .collect()
            } else {
                let mut result = Vec::with_capacity(size);
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

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Vec::with_capacity(size);
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
}

pub fn unary_scalar_map<T: Copy + Send + Sync, F: Fn(T, Scalar) -> T + Send + Sync>(
    storage: &[T],
    layout: &Layout,
    scalar: Scalar,
    f: F,
) -> Vec<T> {
    let offset = layout.get_offset();
    let size = layout.get_size();

    if layout.is_contiguous() {
        #[cfg(feature = "rayon")]
        {
            if size >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;
                storage[offset..offset + size]
                    .par_iter()
                    .map(|&v| f(v, scalar))
                    .collect()
            } else {
                storage[offset..offset + size].iter().map(|&v| f(v, scalar)).collect()
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            storage[offset..offset + size].iter().map(|&v| f(v, scalar)).collect()
        }
    } else {
        let dims = layout.get_shape();
        let strides = layout.get_strides();

        #[cfg(feature = "rayon")]
        {
            if size >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;
                let chunks = size.div_ceil(PARALLEL_CHUNK_SIZE);
                (0..chunks)
                    .into_par_iter()
                    .flat_map(|chunk_idx| {
                        let start = chunk_idx * PARALLEL_CHUNK_SIZE;
                        let end = (start + PARALLEL_CHUNK_SIZE).min(size);
                        let mut chunk_result = Vec::with_capacity(end - start);
                        for i in start..end {
                            let mut idx = offset;
                            let mut tmp_i = i;

                            for d in (0..dims.len()).rev() {
                                let i_dim = tmp_i % dims[d];
                                idx += if strides[d] != 0 { i_dim * strides[d] } else { 0 };
                                tmp_i /= dims[d];
                            }

                            let v = unsafe { storage.get_unchecked(idx) };
                            chunk_result.push(f(*v, scalar));
                        }
                        chunk_result
                    })
                    .collect()
            } else {
                let mut result = Vec::with_capacity(size);
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

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Vec::with_capacity(size);
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
}
