//! Kernel metadata generation for all operation types
//!
//! This module provides metadata generation functions that are used by all backends:
//! - CPU backend (be_cpu)
//! - CUDA backend (be_cuda)
//! - Metal backend (be_metal)
//! - JIT compilation
//!
//! Each operation type has a specific metadata format that matches the kernel expectations.

use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    types::Layout,
};

// ============================================================================
// Binary Operations
// ============================================================================

/// Generate metadata for binary operations (add, sub, mul, div, pow, maximum, minimum)
///
/// Format:
/// - metadata[0]: num_els (total number of elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: lhs_shape
/// - metadata[2+num_dims..2+2*num_dims]: rhs_shape
/// - metadata[2+2*num_dims..2+3*num_dims]: lhs_strides
/// - metadata[2+3*num_dims..2+4*num_dims]: rhs_strides
/// - metadata[2+4*num_dims]: lhs_offset
/// - metadata[2+4*num_dims+1]: rhs_offset
pub fn binary_metadata(lhs_layout: &Layout, rhs_layout: &Layout, output_layout: &Layout) -> Vec<usize> {
    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = output_layout.size();
    let num_dims = output_layout.ndim();

    let mut metadata = Vec::with_capacity(2 + 4 * num_dims + 2);

    // num_els, num_dims
    metadata.push(num_els);
    metadata.push(num_dims);

    // lhs_shape
    for &dim in lhs_shape.dims() {
        metadata.push(dim);
    }

    // rhs_shape
    for &dim in rhs_shape.dims() {
        metadata.push(dim);
    }

    // lhs_strides
    for &stride in lhs_layout.strides() {
        metadata.push(stride);
    }

    // rhs_strides
    for &stride in rhs_layout.strides() {
        metadata.push(stride);
    }

    // lhs_offset, rhs_offset
    metadata.push(lhs_layout.offset());
    metadata.push(rhs_layout.offset());

    metadata
}

// ============================================================================
// Binary Logical Operations
// ============================================================================

/// Generate metadata for binary logical operations (logical_and, logical_or, logical_xor)
/// Alias for binary_metadata (same format)
pub fn binary_logical_metadata(lhs_layout: &Layout, rhs_layout: &Layout, output_layout: &Layout) -> Vec<usize> {
    binary_metadata(lhs_layout, rhs_layout, output_layout)
}

// ============================================================================
// Cmp Operations
// ============================================================================

/// Generate metadata for comparison operations (eq, ne, lt, le, gt, ge)
/// Alias for binary_metadata (same format)
pub fn cmp_metadata(lhs_layout: &Layout, rhs_layout: &Layout, output_layout: &Layout) -> Vec<usize> {
    binary_metadata(lhs_layout, rhs_layout, output_layout)
}

// ============================================================================
// Cmp Scalar Operations
// ============================================================================

/// Generate metadata for comparison with scalar operations (eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar)
/// Alias for unary_metadata (same format)
pub fn cmp_scalar_metadata(input_layout: &Layout, output_layout: &Layout) -> Vec<usize> {
    unary_metadata(input_layout, output_layout)
}

// ============================================================================
// Unary Operations
// ============================================================================

/// Generate metadata for unary operations
/// - basic: neg, abs, sign, square, sqrt, recip
/// - activation: relu, sigmoid, tanh, gelu, softplus, silu, swish, mish
/// - trigonometric: sin, cos, tan
/// - exp: exp, exp2, exp10, ln, log2, log10
///
/// Format:
/// - metadata[0]: num_els (total number of elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
pub fn unary_metadata(input_layout: &Layout, output_layout: &Layout) -> Vec<usize> {
    let shape = input_layout.shape();
    let num_els = output_layout.size();
    let num_dims = input_layout.ndim();

    let mut metadata = Vec::with_capacity(2 + 2 * num_dims + 1);

    // num_els, num_dims
    metadata.push(num_els);
    metadata.push(num_dims);

    // shape
    for &dim in shape.dims() {
        metadata.push(dim);
    }

    // strides
    for &stride in input_layout.strides() {
        metadata.push(stride);
    }

    // offset
    metadata.push(input_layout.offset());

    metadata
}

// ============================================================================
// Unary Logical Operations
// ============================================================================

/// Generate metadata for unary logical operations (logical_not)
/// Alias for unary_metadata (same format)
pub fn unary_logical_metadata(input_layout: &Layout, output_layout: &Layout) -> Vec<usize> {
    unary_metadata(input_layout, output_layout)
}

// ============================================================================
// Unary Scalar Operations
// ============================================================================

/// Generate metadata for unary with scalar operations
/// - arithmetic: add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar, maximum_scalar, minimum_scalar
/// - activation: leaky_relu, elu, prelu
///
/// Alias for unary_metadata (same format)
pub fn unary_scalar_metadata(input_layout: &Layout, output_layout: &Layout) -> Vec<usize> {
    unary_metadata(input_layout, output_layout)
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Generate metadata for matmul operation
///
/// Format:
/// - metadata[0]: num_els
/// - metadata[1]: lhs_ndim
/// - metadata[2]: rhs_ndim
/// - metadata[3]: batch_ndim
/// - metadata[4..4+lhs_ndim]: lhs_shape
/// - metadata[4+lhs_ndim..4+lhs_ndim+rhs_ndim]: rhs_shape
/// - metadata[4+lhs_ndim+rhs_ndim..4+lhs_ndim+rhs_ndim+batch_ndim]: batch_shape
/// - metadata[...]: lhs_strides
/// - metadata[...]: rhs_strides
/// - metadata[...]: lhs_offset
/// - metadata[...]: rhs_offset
/// - metadata[...]: M
/// - metadata[...]: K
/// - metadata[...]: N
pub fn matmul_metadata(lhs_layout: &Layout, rhs_layout: &Layout, output_layout: &Layout) -> HoduResult<Vec<usize>> {
    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let lhs_ndim = lhs_shape.ndim();
    let rhs_ndim = rhs_shape.ndim();

    if lhs_ndim < 2 || rhs_ndim < 2 {
        return Err(HoduError::InvalidArgument("matmul requires at least 2D tensors".into()));
    }

    // Extract matrix dimensions
    let m = lhs_shape.dims()[lhs_ndim - 2];
    let k_lhs = lhs_shape.dims()[lhs_ndim - 1];
    let k_rhs = rhs_shape.dims()[rhs_ndim - 2];
    let n = rhs_shape.dims()[rhs_ndim - 1];

    if k_lhs != k_rhs {
        return Err(HoduError::InvalidArgument(format!(
            "matmul inner dimensions mismatch: {} vs {}",
            k_lhs, k_rhs
        )));
    }

    // Compute batch dimensions
    let lhs_batch_ndim = lhs_ndim - 2;
    let rhs_batch_ndim = rhs_ndim - 2;
    let batch_ndim = lhs_batch_ndim.max(rhs_batch_ndim);

    // Broadcast batch dimensions
    let mut batch_shape = Vec::with_capacity(batch_ndim);
    for i in 0..batch_ndim {
        let lhs_idx = (lhs_batch_ndim as i32 - batch_ndim as i32 + i as i32) as usize;
        let rhs_idx = (rhs_batch_ndim as i32 - batch_ndim as i32 + i as i32) as usize;

        let lhs_dim = if lhs_idx < lhs_batch_ndim {
            lhs_shape.dims()[lhs_idx]
        } else {
            1
        };
        let rhs_dim = if rhs_idx < rhs_batch_ndim {
            rhs_shape.dims()[rhs_idx]
        } else {
            1
        };

        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Err(HoduError::InvalidArgument(format!(
                "matmul batch dimension mismatch at index {}: {} vs {}",
                i, lhs_dim, rhs_dim
            )));
        }

        batch_shape.push(lhs_dim.max(rhs_dim));
    }

    let num_els = output_layout.size();

    let mut metadata = Vec::with_capacity(4 + lhs_ndim + rhs_ndim + batch_ndim + lhs_ndim + rhs_ndim + 2 + 3);

    metadata.push(num_els);
    metadata.push(lhs_ndim);
    metadata.push(rhs_ndim);
    metadata.push(batch_ndim);

    // lhs_shape
    for &dim in lhs_shape.dims() {
        metadata.push(dim);
    }

    // rhs_shape
    for &dim in rhs_shape.dims() {
        metadata.push(dim);
    }

    // batch_shape
    for &dim in &batch_shape {
        metadata.push(dim);
    }

    // lhs_strides
    for &stride in lhs_layout.strides() {
        metadata.push(stride);
    }

    // rhs_strides
    for &stride in rhs_layout.strides() {
        metadata.push(stride);
    }

    // offsets
    metadata.push(lhs_layout.offset());
    metadata.push(rhs_layout.offset());

    // matrix dimensions
    metadata.push(m);
    metadata.push(k_lhs);
    metadata.push(n);

    Ok(metadata)
}

/// Generate metadata for dot operation (2D matrix multiplication)
///
/// Format:
/// - metadata[0]: M
/// - metadata[1]: K
/// - metadata[2]: N
/// - metadata[3]: lhs_stride_m
/// - metadata[4]: lhs_stride_k
/// - metadata[5]: rhs_stride_k
/// - metadata[6]: rhs_stride_n
/// - metadata[7]: lhs_offset
/// - metadata[8]: rhs_offset
pub fn dot_metadata(lhs_layout: &Layout, rhs_layout: &Layout) -> HoduResult<Vec<usize>> {
    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let lhs_ndim = lhs_shape.ndim();
    let rhs_ndim = rhs_shape.ndim();

    if lhs_ndim != 2 || rhs_ndim != 2 {
        return Err(HoduError::InvalidArgument("dot requires exactly 2D tensors".into()));
    }

    // Extract matrix dimensions
    let m = lhs_shape.dims()[0];
    let k_lhs = lhs_shape.dims()[1];
    let k_rhs = rhs_shape.dims()[0];
    let n = rhs_shape.dims()[1];

    if k_lhs != k_rhs {
        return Err(HoduError::InvalidArgument(format!(
            "dot inner dimensions mismatch: {} vs {}",
            k_lhs, k_rhs
        )));
    }

    let mut metadata = Vec::with_capacity(9);

    metadata.push(m);
    metadata.push(k_lhs);
    metadata.push(n);

    // strides
    let lhs_strides = lhs_layout.strides();
    let rhs_strides = rhs_layout.strides();
    metadata.push(lhs_strides[0]); // lhs_stride_m
    metadata.push(lhs_strides[1]); // lhs_stride_k
    metadata.push(rhs_strides[0]); // rhs_stride_k
    metadata.push(rhs_strides[1]); // rhs_stride_n

    // offsets
    metadata.push(lhs_layout.offset());
    metadata.push(rhs_layout.offset());

    Ok(metadata)
}

// ============================================================================
// Reduce Operations
// ============================================================================

/// Generate metadata for reduce operations (sum, mean, max, min, prod, std, var, l2_norm, argmax, argmin, any, all)
///
/// Format:
/// - metadata[0]: input_ndim
/// - metadata[1..1+input_ndim]: input_shape
/// - metadata[1+input_ndim..1+2*input_ndim]: input_strides
/// - metadata[1+2*input_ndim]: input_offset
/// - metadata[2+2*input_ndim]: output_ndim
/// - metadata[3+2*input_ndim..]: output_shape
/// - metadata[...]: num_reduce_dims
/// - metadata[...]: reduce_dims
/// - metadata[...]: keep_dim (0 or 1)
/// - metadata[...]: reduce_size
pub fn reduce_metadata(layout: &Layout, dims: &[usize], keep_dim: bool) -> Vec<usize> {
    let input_shape = layout.shape();
    let input_ndim = input_shape.ndim();

    // Compute output shape
    let mut output_shape_vec = Vec::new();
    for i in 0..input_ndim {
        if dims.contains(&i) {
            if keep_dim {
                output_shape_vec.push(1);
            }
        } else {
            output_shape_vec.push(input_shape.dims()[i]);
        }
    }

    // Handle empty output shape (reduce all dimensions without keep_dim)
    if output_shape_vec.is_empty() {
        output_shape_vec.push(1);
    }

    // Calculate reduce size
    let mut reduce_size: usize = 1;
    for &dim in dims {
        reduce_size *= input_shape.dims()[dim];
    }

    let mut metadata =
        Vec::with_capacity(1 + input_ndim + input_ndim + 1 + 1 + output_shape_vec.len() + 1 + dims.len() + 1 + 1);

    // input_ndim, input_shape
    metadata.push(input_ndim);
    for &dim in input_shape.dims() {
        metadata.push(dim);
    }

    // input_strides
    for &stride in layout.strides() {
        metadata.push(stride);
    }

    // input_offset
    metadata.push(layout.offset());

    // output_ndim, output_shape
    metadata.push(output_shape_vec.len());
    for &dim in &output_shape_vec {
        metadata.push(dim);
    }

    // num_reduce_dims, reduce_dims
    metadata.push(dims.len());
    for &dim in dims {
        metadata.push(dim);
    }

    // keep_dim, reduce_size
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    metadata
}

// ============================================================================
// Concat Operations
// ============================================================================

/// Generate metadata for concat operations (concat, cat, stack)
///
/// Format:
/// - metadata[0]: num_els
/// - metadata[1]: num_dims
/// - metadata[2..2+num_dims]: output_shape
/// - metadata[...]: concat_dim
/// - metadata[...]: num_inputs
/// - metadata[...]: input_shapes (flattened)
/// - metadata[...]: input_strides (flattened)
/// - metadata[...]: input_offsets
/// - metadata[...]: input_buffer_offsets
pub fn concat_metadata(layouts: &[&Layout], dim: usize, output_shape: &[usize]) -> Vec<usize> {
    let num_inputs = layouts.len();
    let ndim = output_shape.len();
    let num_els: usize = output_shape.iter().product();

    let mut metadata =
        Vec::with_capacity(2 + ndim + 1 + 1 + num_inputs * ndim + num_inputs * ndim + num_inputs + num_inputs);

    metadata.push(num_els);
    metadata.push(ndim);

    // output_shape
    for &d in output_shape {
        metadata.push(d);
    }

    // concat_dim, num_inputs
    metadata.push(dim);
    metadata.push(num_inputs);

    // input_shapes (flattened)
    for layout in layouts {
        for &d in layout.shape().dims() {
            metadata.push(d);
        }
    }

    // input_strides (flattened)
    for layout in layouts {
        for &s in layout.strides() {
            metadata.push(s);
        }
    }

    // input_offsets
    for layout in layouts {
        metadata.push(layout.offset());
    }

    // input_buffer_offsets
    let mut buffer_offset = 0;
    for layout in layouts {
        metadata.push(buffer_offset);
        buffer_offset += layout.shape().size();
    }

    metadata
}

// ============================================================================
// Split Operations
// ============================================================================

/// Generate metadata for split operations (split, chunk)
///
/// Format:
/// - metadata[0]: num_els
/// - metadata[1]: num_dims
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[...]: input_strides
/// - metadata[...]: input_offset
/// - metadata[...]: split_dim
/// - metadata[...]: output_size_on_dim
/// - metadata[...]: split_offset (start position)
pub fn split_metadata(layout: &Layout, dim: usize, size: usize, start: usize, output_num_els: usize) -> Vec<usize> {
    let input_shape = layout.shape();
    let ndim = input_shape.ndim();

    let mut metadata = Vec::with_capacity(2 + ndim + ndim + 1 + 3);

    metadata.push(output_num_els);
    metadata.push(ndim);

    // input_shape
    for &d in input_shape.dims() {
        metadata.push(d);
    }

    // input_strides
    for &s in layout.strides() {
        metadata.push(s);
    }

    // input_offset
    metadata.push(layout.offset());

    // split_dim, output_size_on_dim, split_offset
    metadata.push(dim);
    metadata.push(size);
    metadata.push(start);

    metadata
}

// ============================================================================
// Indexing Operations
// ============================================================================

/// Generate metadata for index_select operations
///
/// Format:
/// - [num_els, num_dims, input_shape, input_strides, input_offset, dim, num_indices]
pub fn index_select_metadata(layout: &Layout, dim: usize, num_indices: usize, output_num_els: usize) -> Vec<usize> {
    let input_shape = layout.shape();
    let ndim = input_shape.ndim();

    let mut metadata = Vec::with_capacity(2 + ndim + ndim + 1 + 1 + 1);

    metadata.push(output_num_els);
    metadata.push(ndim);

    for &d in input_shape.dims() {
        metadata.push(d);
    }

    for &s in layout.strides() {
        metadata.push(s);
    }

    metadata.push(layout.offset());
    metadata.push(dim);
    metadata.push(num_indices);

    metadata
}

/// Generate metadata for index_put operations
///
/// Format:
/// - [num_els, num_dims, input_shape, input_strides, input_offset, values_shape, values_strides, values_offset, dim, num_indices]
pub fn index_put_metadata(
    layout: &Layout,
    values_layout: &Layout,
    dim: usize,
    num_indices: usize,
    output_num_els: usize,
) -> Vec<usize> {
    let input_shape = layout.shape();
    let values_shape = values_layout.shape();
    let ndim = input_shape.ndim();

    let mut metadata = Vec::with_capacity(2 + ndim + ndim + 1 + ndim + ndim + 1 + 1 + 1);

    metadata.push(output_num_els);
    metadata.push(ndim);

    // input shape and strides
    for &d in input_shape.dims() {
        metadata.push(d);
    }
    for &s in layout.strides() {
        metadata.push(s);
    }
    metadata.push(layout.offset());

    // values shape and strides
    for &d in values_shape.dims() {
        metadata.push(d);
    }
    for &s in values_layout.strides() {
        metadata.push(s);
    }
    metadata.push(values_layout.offset());

    metadata.push(dim);
    metadata.push(num_indices);

    metadata
}

/// Generate metadata for gather operations
///
/// Format:
/// - [num_els, num_dims, output_shape, input_shape, input_strides, indices_strides (padded), input_offset, indices_offset, dim]
pub fn gather_metadata(layout: &Layout, indices_layout: &Layout, dim: usize, output_num_els: usize) -> Vec<usize> {
    let input_shape = layout.shape();
    let indices_shape = indices_layout.shape();
    let ndim = input_shape.ndim();

    let mut metadata = Vec::with_capacity(2 + 4 * ndim + 3);

    metadata.push(output_num_els);
    metadata.push(ndim);

    // output shape (same as indices shape)
    for &d in indices_shape.dims() {
        metadata.push(d);
    }

    // input shape
    for &d in input_shape.dims() {
        metadata.push(d);
    }

    // input strides
    for &s in layout.strides() {
        metadata.push(s);
    }

    // indices strides (pad to ndim elements)
    for i in 0..ndim {
        if i < indices_layout.strides().len() {
            metadata.push(indices_layout.strides()[i]);
        } else {
            metadata.push(0);
        }
    }

    metadata.push(layout.offset());
    metadata.push(indices_layout.offset());
    metadata.push(dim);

    metadata
}

/// Generate metadata for scatter operations (scatter, scatter_add, scatter_max, scatter_min)
///
/// Format:
/// - [num_els (src elements), num_dims, input_shape, input_strides, src_shape (padded), src_strides (padded), indices_strides (padded), input_offset, src_offset, indices_offset, dim]
pub fn scatter_metadata(layout: &Layout, indices_layout: &Layout, src_layout: &Layout, dim: usize) -> Vec<usize> {
    let input_shape = layout.shape();
    let src_shape = src_layout.shape();
    let ndim = input_shape.ndim();
    let src_num_els = src_shape.size();

    let mut metadata = Vec::with_capacity(2 + 5 * ndim + 4);

    metadata.push(src_num_els);
    metadata.push(ndim);

    // input shape
    for &d in input_shape.dims() {
        metadata.push(d);
    }

    // input strides
    for &s in layout.strides() {
        metadata.push(s);
    }

    // src shape (pad to ndim elements)
    for i in 0..ndim {
        if i < src_shape.ndim() {
            metadata.push(src_shape.dims()[i]);
        } else {
            metadata.push(1);
        }
    }

    // src strides (pad to ndim elements)
    for i in 0..ndim {
        if i < src_layout.strides().len() {
            metadata.push(src_layout.strides()[i]);
        } else {
            metadata.push(0);
        }
    }

    // indices strides (pad to ndim elements)
    for i in 0..ndim {
        if i < indices_layout.strides().len() {
            metadata.push(indices_layout.strides()[i]);
        } else {
            metadata.push(0);
        }
    }

    metadata.push(layout.offset());
    metadata.push(src_layout.offset());
    metadata.push(indices_layout.offset());
    metadata.push(dim);

    metadata
}

// ============================================================================
// Conv Operations
// ============================================================================

/// Generate metadata for conv1d operations
///
/// Format:
/// - [num_els, batch_size, in_channels, out_channels, in_width, kernel_width, out_width, stride_w, padding_w, dilation_w, input_offset, weight_offset]
pub fn conv1d_metadata(
    input_layout: &Layout,
    weight_layout: &Layout,
    stride: usize,
    padding: usize,
    dilation: usize,
    output_shape: &[usize],
) -> Vec<usize> {
    let input_shape = input_layout.shape();
    let weight_shape = weight_layout.shape();

    let batch_size = output_shape[0];
    let out_channels = output_shape[1];
    let out_width = output_shape[2];
    let num_els: usize = output_shape.iter().product();

    let in_channels = input_shape.dims()[1];
    let in_width = input_shape.dims()[2];
    let kernel_width = weight_shape.dims()[2];

    vec![
        num_els,
        batch_size,
        in_channels,
        out_channels,
        in_width,
        kernel_width,
        out_width,
        stride,
        padding,
        dilation,
        input_layout.offset(),
        weight_layout.offset(),
    ]
}

/// Generate metadata for conv2d operations
///
/// Format:
/// - [num_els, batch_size, in_channels, out_channels, in_height, in_width, kernel_height, kernel_width, out_height, out_width, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, input_offset, weight_offset]
pub fn conv2d_metadata(
    input_layout: &Layout,
    weight_layout: &Layout,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    output_shape: &[usize],
) -> Vec<usize> {
    let input_shape = input_layout.shape();
    let weight_shape = weight_layout.shape();

    let batch_size = output_shape[0];
    let out_channels = output_shape[1];
    let out_height = output_shape[2];
    let out_width = output_shape[3];
    let num_els: usize = output_shape.iter().product();

    let in_channels = input_shape.dims()[1];
    let in_height = input_shape.dims()[2];
    let in_width = input_shape.dims()[3];
    let kernel_height = weight_shape.dims()[2];
    let kernel_width = weight_shape.dims()[3];

    vec![
        num_els,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        input_layout.offset(),
        weight_layout.offset(),
    ]
}

/// Generate metadata for conv3d operations
///
/// Format:
/// - [num_els, batch_size, in_channels, out_channels, in_depth, in_height, in_width, kernel_depth, kernel_height, kernel_width, out_depth, out_height, out_width, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, dilation_d, dilation_h, dilation_w, input_offset, weight_offset]
pub fn conv3d_metadata(
    input_layout: &Layout,
    weight_layout: &Layout,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    output_shape: &[usize],
) -> Vec<usize> {
    let input_shape = input_layout.shape();
    let weight_shape = weight_layout.shape();

    let batch_size = output_shape[0];
    let out_channels = output_shape[1];
    let out_depth = output_shape[2];
    let out_height = output_shape[3];
    let out_width = output_shape[4];
    let num_els: usize = output_shape.iter().product();

    let in_channels = input_shape.dims()[1];
    let in_depth = input_shape.dims()[2];
    let in_height = input_shape.dims()[3];
    let in_width = input_shape.dims()[4];
    let kernel_depth = weight_shape.dims()[2];
    let kernel_height = weight_shape.dims()[3];
    let kernel_width = weight_shape.dims()[4];

    vec![
        num_els,
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        out_depth,
        out_height,
        out_width,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        input_layout.offset(),
        weight_layout.offset(),
    ]
}

/// Generate metadata for conv_grad_weight operations
///
/// Format:
/// - [num_els, input_ndim, spatial_dims, input_shape, grad_output_shape, weight_shape, input_strides, grad_output_strides, input_offset, grad_output_offset, stride, padding, dilation]
pub fn conv_grad_weight_metadata(
    input_layout: &Layout,
    grad_output_layout: &Layout,
    weight_shape: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    spatial_dims: usize,
) -> Vec<usize> {
    let input_shape = input_layout.shape();
    let grad_output_shape = grad_output_layout.shape();
    let input_ndim = input_shape.ndim();
    let num_els: usize = weight_shape.iter().product();

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(input_ndim);
    metadata.push(spatial_dims);

    // shapes
    for &d in input_shape.dims() {
        metadata.push(d);
    }
    for &d in grad_output_shape.dims() {
        metadata.push(d);
    }
    for &d in weight_shape {
        metadata.push(d);
    }

    // strides
    for &s in input_layout.strides() {
        metadata.push(s);
    }
    for &s in grad_output_layout.strides() {
        metadata.push(s);
    }

    // offsets
    metadata.push(input_layout.offset());
    metadata.push(grad_output_layout.offset());

    // conv parameters
    for &s in stride {
        metadata.push(s);
    }
    for &p in padding {
        metadata.push(p);
    }
    for &d in dilation {
        metadata.push(d);
    }

    metadata
}

// ============================================================================
// Windowing Operations
// ============================================================================

/// Generate metadata for reduce_window operations (max, mean, sum, min)
///
/// Format:
/// - [output_size, num_dims, input_shape, input_strides, offset, window_shape, strides, padding, output_shape]
pub fn reduce_window_metadata(
    layout: &Layout,
    window_shape: &[usize],
    strides: &[usize],
    padding: &[usize],
    output_shape: &[usize],
) -> Vec<usize> {
    let input_shape = layout.shape();
    let ndim = input_shape.ndim();
    let output_size: usize = output_shape.iter().product();

    let mut metadata = Vec::with_capacity(3 + ndim * 7);

    metadata.push(output_size);
    metadata.push(ndim);

    // input shape
    for &dim in input_shape.dims() {
        metadata.push(dim);
    }

    // input strides
    for &stride in layout.strides() {
        metadata.push(stride);
    }

    // offset
    metadata.push(layout.offset());

    // window shape
    for &w in window_shape {
        metadata.push(w);
    }

    // strides
    for &s in strides {
        metadata.push(s);
    }

    // padding
    for &p in padding {
        metadata.push(p);
    }

    // output shape
    for &dim in output_shape {
        metadata.push(dim);
    }

    metadata
}

// ============================================================================
// Cast Operations
// ============================================================================

/// Generate metadata for cast (to_dtype) operations
///
/// Format:
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
pub fn cast_metadata(layout: &Layout) -> Vec<usize> {
    let shape = layout.shape();
    let num_els = layout.size();
    let num_dims = shape.ndim();

    let mut metadata = Vec::with_capacity(2 + 2 * num_dims + 1);

    // num_els, num_dims
    metadata.push(num_els);
    metadata.push(num_dims);

    // shape
    for &dim in shape.dims() {
        metadata.push(dim);
    }

    // strides
    for &stride in layout.strides() {
        metadata.push(stride);
    }

    // offset
    metadata.push(layout.offset());

    metadata
}

// ============================================================================
// Memory Operations
// ============================================================================

/// Generate metadata for contiguous operations
///
/// Format:
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
pub fn contiguous_metadata(layout: &Layout) -> Vec<usize> {
    cast_metadata(layout)
}
