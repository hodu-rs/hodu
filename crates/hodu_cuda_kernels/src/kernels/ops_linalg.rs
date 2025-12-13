use crate::{
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(det, inv);

/// Execute a matrix determinant operation
///
/// Computes the determinant of square matrices with optional batch dimensions.
/// Uses direct formulas for small matrices (1x1, 2x2, 3x3) and LU decomposition
/// with partial pivoting for larger matrices (up to 16x16).
///
/// # Arguments
/// * `kernel` - The det kernel (e.g., "det::F32")
/// * `kernels` - Kernel cache
/// * `context` - CUDA context to execute on
/// * `input` - Input tensor device slice containing square matrices
/// * `output` - Output device slice for determinant values
/// * `batch_size` - Number of matrices in the batch
/// * `metadata` - Host slice containing metadata describing matrix layout
///
/// # Metadata layout
/// - metadata[0]: batch_size (product of batch dimensions)
/// - metadata[1]: n (matrix size, N×N)
/// - metadata[2]: ndim (total number of dimensions)
/// - metadata[3..3+ndim]: shape
/// - metadata[3+ndim..3+2*ndim]: strides
/// - metadata[3+2*ndim]: offset
pub fn call_ops_det<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    batch_size: usize,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsLinalg, kernel.0)?;

    // One thread per batch element
    const THREADS_PER_BLOCK: u32 = 256;
    let grid_size = ((batch_size as u32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    let grid_size = grid_size.max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a matrix inverse operation
///
/// Computes the inverse of square matrices with optional batch dimensions.
/// Uses direct formulas for small matrices (1x1, 2x2, 3x3) and Gauss-Jordan
/// elimination with partial pivoting for larger matrices (up to 16x16).
///
/// # Arguments
/// * `kernel` - The inv kernel (e.g., "inv::F32")
/// * `kernels` - Kernel cache
/// * `context` - CUDA context to execute on
/// * `input` - Input tensor device slice containing square matrices
/// * `output` - Output device slice for inverse matrices
/// * `batch_size` - Number of matrices in the batch
/// * `metadata` - Host slice containing metadata describing matrix layout
///
/// # Metadata layout (same as det)
/// - metadata[0]: batch_size (product of batch dimensions)
/// - metadata[1]: n (matrix size, N×N)
/// - metadata[2]: ndim (total number of dimensions)
/// - metadata[3..3+ndim]: shape
/// - metadata[3+ndim..3+2*ndim]: strides
/// - metadata[3+2*ndim]: offset
pub fn call_ops_inv<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    batch_size: usize,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsLinalg, kernel.0)?;

    // One thread per batch element
    const THREADS_PER_BLOCK: u32 = 256;
    let grid_size = ((batch_size as u32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    let grid_size = grid_size.max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
