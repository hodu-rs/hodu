use crate::{
    error::MetalKernelError,
    kernel::Kernels,
    kernels::macros::ops,
    metal::{Buffer, ComputeCommandEncoder, Device},
    set_params,
    source::Source,
    utils::{BufferOffset, EncoderProvider},
};
use objc2_metal::{MTLResourceUsage, MTLSize};

ops!(det, inv);

/// Executes a matrix determinant operation using Metal compute pipeline.
///
/// Computes the determinant of square matrices with optional batch dimensions.
/// Uses direct formulas for small matrices (1x1, 2x2, 3x3) and LU decomposition
/// with partial pivoting for larger matrices (up to 16x16).
///
/// # Arguments
/// * `kernel` - Det kernel (e.g., det::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer containing square matrices
/// * `output` - Output buffer for determinant values
/// * `batch_size` - Number of matrices in the batch
/// * `metadata` - Metadata describing matrix dimensions and layout
///
/// # Metadata Layout
/// - `metadata[0]`: batch_size (product of batch dimensions)
/// - `metadata[1]`: n (matrix size, N×N)
/// - `metadata[2]`: ndim (total number of dimensions)
/// - `metadata[3..3+ndim]`: shape
/// - `metadata[3+ndim..3+2*ndim]`: strides
/// - `metadata[3+2*ndim]`: offset
#[allow(clippy::too_many_arguments)]
pub fn call_ops_det(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    batch_size: usize,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Linalg, kernel.0)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(encoder, (&input, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    // One thread per batch element
    let threads_per_threadgroup = 256.min(batch_size);
    let threadgroup_count = (batch_size + threads_per_threadgroup - 1) / threads_per_threadgroup;

    let threadgroup_size = MTLSize {
        width: threads_per_threadgroup,
        height: 1,
        depth: 1,
    };

    let grid_size = MTLSize {
        width: threadgroup_count,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);

    Ok(())
}

/// Executes a matrix inverse operation using Metal compute pipeline.
///
/// Computes the inverse of square matrices with optional batch dimensions.
/// Uses direct formulas for small matrices (1x1, 2x2, 3x3) and Gauss-Jordan
/// elimination with partial pivoting for larger matrices (up to 16x16).
///
/// # Arguments
/// * `kernel` - Inv kernel (e.g., inv::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer containing square matrices
/// * `output` - Output buffer for inverse matrices
/// * `batch_size` - Number of matrices in the batch
/// * `metadata` - Metadata describing matrix dimensions and layout
///
/// # Metadata Layout (same as det)
/// - `metadata[0]`: batch_size (product of batch dimensions)
/// - `metadata[1]`: n (matrix size, N×N)
/// - `metadata[2]`: ndim (total number of dimensions)
/// - `metadata[3..3+ndim]`: shape
/// - `metadata[3+ndim..3+2*ndim]`: strides
/// - `metadata[3+2*ndim]`: offset
#[allow(clippy::too_many_arguments)]
pub fn call_ops_inv(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    batch_size: usize,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Linalg, kernel.0)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(encoder, (&input, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    // One thread per batch element
    let threads_per_threadgroup = 256.min(batch_size);
    let threadgroup_count = (batch_size + threads_per_threadgroup - 1) / threads_per_threadgroup;

    let threadgroup_size = MTLSize {
        width: threads_per_threadgroup,
        height: 1,
        depth: 1,
    };

    let grid_size = MTLSize {
        width: threadgroup_count,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);

    Ok(())
}
