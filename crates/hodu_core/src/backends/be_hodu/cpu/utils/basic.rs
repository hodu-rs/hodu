use crate::{compat::*, scalar::Scalar, types::layout::Layout};

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
