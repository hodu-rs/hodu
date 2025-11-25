//! Kernel metadata generation
//!
//! Each kernel operation expects metadata in a specific format.
//! This module provides centralized metadata generation for all backends:

use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    ops::Op,
    script::snapshot::SnapshotNode,
};

/// Generate metadata array for a kernel operation
/// Returns a vector of usize values that will be passed as metadata pointer
pub fn generate_metadata(node: &SnapshotNode) -> HoduResult<Vec<usize>> {
    match &node.op {
        Op::Binary(_) | Op::BinaryLogical(_) | Op::Cmp(_) => generate_binary_metadata(node),
        Op::Unary(_) | Op::UnaryLogical(_) => generate_unary_metadata(node),
        // TODO: Add more operation types
        _ => Err(HoduError::UnsupportedOperation(format!(
            "Metadata generation not yet implemented for op: {}",
            node.op
        ))),
    }
}

/// Generate metadata for binary operations
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
fn generate_binary_metadata(node: &SnapshotNode) -> HoduResult<Vec<usize>> {
    if node.input_layouts.len() != 2 {
        return Err(HoduError::InternalError(format!(
            "Binary op should have 2 inputs, got {}",
            node.input_layouts.len()
        )));
    }

    let lhs = &node.input_layouts[0];
    let rhs = &node.input_layouts[1];
    let mut metadata = Vec::new();

    // num_els
    metadata.push(node.output_layout.size());

    // num_dims
    metadata.push(node.output_layout.ndim());

    // lhs_shape
    for &dim in lhs.shape().dims() {
        metadata.push(dim);
    }

    // rhs_shape
    for &dim in rhs.shape().dims() {
        metadata.push(dim);
    }

    // lhs_strides
    for &stride in lhs.strides() {
        metadata.push(stride);
    }

    // rhs_strides
    for &stride in rhs.strides() {
        metadata.push(stride);
    }

    // lhs_offset
    metadata.push(lhs.offset());

    // rhs_offset
    metadata.push(rhs.offset());

    Ok(metadata)
}

/// Generate metadata for unary operations
///
/// Format:
/// - metadata[0]: num_els (total number of elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
fn generate_unary_metadata(node: &SnapshotNode) -> HoduResult<Vec<usize>> {
    if node.input_layouts.len() != 1 {
        return Err(HoduError::InternalError(format!(
            "Unary op should have 1 input, got {}",
            node.input_layouts.len()
        )));
    }

    let input = &node.input_layouts[0];
    let mut metadata = Vec::new();

    // num_els
    metadata.push(node.output_layout.size());

    // num_dims
    metadata.push(input.ndim());

    // shape
    for &dim in input.shape().dims() {
        metadata.push(dim);
    }

    // strides
    for &stride in input.strides() {
        metadata.push(stride);
    }

    // offset
    metadata.push(input.offset());

    Ok(metadata)
}

// TODO: Add more metadata generators:
// - generate_matrix_metadata (matmul, dot)
// - generate_reduce_metadata (sum, mean, etc.)
// - generate_conv_metadata (conv1d, conv2d, etc.)
// - generate_indexing_metadata (gather, scatter, etc.)
// - generate_concat_split_metadata
