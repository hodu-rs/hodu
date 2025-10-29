use crate::{compat::*, scalar::Scalar, types::layout::Layout};

#[cfg(feature = "rayon")]
const PARALLEL_THRESHOLD: usize = 4096;
#[cfg(feature = "rayon")]
const PARALLEL_CHUNK_SIZE: usize = 1024;

pub fn cmp_map<T: Copy + Send + Sync, F: Fn(T, T) -> bool + Send + Sync>(
    lhs_storage: &[T],
    rhs_storage: &[T],
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    f: F,
) -> Vec<bool> {
    let lhs_offset = lhs_layout.get_offset();
    let rhs_offset = rhs_layout.get_offset();
    let lhs_size = lhs_layout.get_size();
    let rhs_size = rhs_layout.get_size();

    let output_size = lhs_size.max(rhs_size);

    match (lhs_layout.is_contiguous(), rhs_layout.is_contiguous()) {
        (true, true) => {
            let lhs_slice = &lhs_storage[lhs_offset..lhs_offset + lhs_size];
            let rhs_slice = &rhs_storage[rhs_offset..rhs_offset + rhs_size];

            #[cfg(feature = "rayon")]
            {
                if output_size >= PARALLEL_THRESHOLD {
                    use rayon::prelude::*;
                    let chunks = output_size.div_ceil(PARALLEL_CHUNK_SIZE);
                    (0..chunks)
                        .into_par_iter()
                        .flat_map(|chunk_idx| {
                            let start = chunk_idx * PARALLEL_CHUNK_SIZE;
                            let end = (start + PARALLEL_CHUNK_SIZE).min(output_size);
                            let mut chunk_result = Vec::with_capacity(end - start);
                            for i in start..end {
                                let l = lhs_slice[i % lhs_size];
                                let r = rhs_slice[i % rhs_size];
                                chunk_result.push(f(l, r));
                            }
                            chunk_result
                        })
                        .collect()
                } else {
                    let mut result = Vec::with_capacity(output_size);
                    for i in 0..output_size {
                        let l = lhs_slice[i % lhs_size];
                        let r = rhs_slice[i % rhs_size];
                        result.push(f(l, r));
                    }
                    result
                }
            }

            #[cfg(not(feature = "rayon"))]
            {
                let mut result = Vec::with_capacity(output_size);
                for i in 0..output_size {
                    let l = lhs_slice[i % lhs_size];
                    let r = rhs_slice[i % rhs_size];
                    result.push(f(l, r));
                }
                result
            }
        },
        (true, false) => {
            let lhs_slice = &lhs_storage[lhs_offset..lhs_offset + lhs_size];
            let dims = rhs_layout.get_shape();
            let rhs_strides = rhs_layout.get_strides();

            #[cfg(feature = "rayon")]
            {
                if output_size >= PARALLEL_THRESHOLD {
                    use rayon::prelude::*;
                    let chunks = output_size.div_ceil(PARALLEL_CHUNK_SIZE);
                    (0..chunks)
                        .into_par_iter()
                        .flat_map(|chunk_idx| {
                            let start = chunk_idx * PARALLEL_CHUNK_SIZE;
                            let end = (start + PARALLEL_CHUNK_SIZE).min(output_size);
                            let mut chunk_result = Vec::with_capacity(end - start);
                            for i in start..end {
                                let l = lhs_slice[i % lhs_size];

                                let mut rhs_idx = rhs_offset;
                                let mut tmp_i = i % rhs_size;

                                for d in (0..dims.len()).rev() {
                                    let i_dim = tmp_i % dims[d];
                                    rhs_idx += if rhs_strides[d] != 0 { i_dim * rhs_strides[d] } else { 0 };
                                    tmp_i /= dims[d];
                                }

                                let r = rhs_storage[rhs_idx];
                                chunk_result.push(f(l, r));
                            }
                            chunk_result
                        })
                        .collect()
                } else {
                    let mut result = Vec::with_capacity(output_size);
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
                    result
                }
            }

            #[cfg(not(feature = "rayon"))]
            {
                let mut result = Vec::with_capacity(output_size);
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
                result
            }
        },
        (false, true) => {
            let rhs_slice = &rhs_storage[rhs_offset..rhs_offset + rhs_size];
            let dims = lhs_layout.get_shape();
            let lhs_strides = lhs_layout.get_strides();

            #[cfg(feature = "rayon")]
            {
                if output_size >= PARALLEL_THRESHOLD {
                    use rayon::prelude::*;
                    let chunks = output_size.div_ceil(PARALLEL_CHUNK_SIZE);
                    (0..chunks)
                        .into_par_iter()
                        .flat_map(|chunk_idx| {
                            let start = chunk_idx * PARALLEL_CHUNK_SIZE;
                            let end = (start + PARALLEL_CHUNK_SIZE).min(output_size);
                            let mut chunk_result = Vec::with_capacity(end - start);
                            for i in start..end {
                                let r = rhs_slice[i % rhs_size];

                                let mut lhs_idx = lhs_offset;
                                let mut tmp_i = i % lhs_size;

                                for d in (0..dims.len()).rev() {
                                    let i_dim = tmp_i % dims[d];
                                    lhs_idx += if lhs_strides[d] != 0 { i_dim * lhs_strides[d] } else { 0 };
                                    tmp_i /= dims[d];
                                }

                                let l = lhs_storage[lhs_idx];
                                chunk_result.push(f(l, r));
                            }
                            chunk_result
                        })
                        .collect()
                } else {
                    let mut result = Vec::with_capacity(output_size);
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
                    result
                }
            }

            #[cfg(not(feature = "rayon"))]
            {
                let mut result = Vec::with_capacity(output_size);
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
                result
            }
        },
        (false, false) => {
            let lhs_dims = lhs_layout.get_shape();
            let lhs_strides = lhs_layout.get_strides();
            let rhs_dims = rhs_layout.get_shape();
            let rhs_strides = rhs_layout.get_strides();

            #[cfg(feature = "rayon")]
            {
                if output_size >= PARALLEL_THRESHOLD {
                    use rayon::prelude::*;
                    let chunks = output_size.div_ceil(PARALLEL_CHUNK_SIZE);
                    (0..chunks)
                        .into_par_iter()
                        .flat_map(|chunk_idx| {
                            let start = chunk_idx * PARALLEL_CHUNK_SIZE;
                            let end = (start + PARALLEL_CHUNK_SIZE).min(output_size);
                            let mut chunk_result = Vec::with_capacity(end - start);
                            for i in start..end {
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
                                chunk_result.push(f(l, r));
                            }
                            chunk_result
                        })
                        .collect()
                } else {
                    let mut result = Vec::with_capacity(output_size);
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
                    result
                }
            }

            #[cfg(not(feature = "rayon"))]
            {
                let mut result = Vec::with_capacity(output_size);
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
                result
            }
        },
    }
}

pub fn cmp_scalar_map<T: Copy + Send + Sync, F: Fn(T, Scalar) -> bool + Send + Sync>(
    storage: &[T],
    layout: &Layout,
    scalar: Scalar,
    f: F,
) -> Vec<bool> {
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
