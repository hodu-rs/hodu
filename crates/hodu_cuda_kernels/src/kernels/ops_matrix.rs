use crate::{
    compat::*,
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::get_global_kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(matmul, dot);

/// Execute a batched matrix multiplication with broadcasting support
///
/// # Arguments
/// * `device` - CUDA device to execute on
/// * `kernels` - Kernel cache
/// * `kernel_name` - The matmul kernel (e.g., "matmul_f32")
/// * `lhs` - Left-hand side matrix device slice with shape [..., M, K]
/// * `rhs` - Right-hand side matrix device slice with shape [..., K, N]
/// * `output` - Output matrix device slice with shape [..., M, N]
/// * `metadata` - Device slice containing metadata describing matrix dimensions
///
/// # Metadata layout
/// - metadata[0]: num_els (total output elements)
/// - metadata[1]: lhs_ndim (number of dimensions in lhs)
/// - metadata[2]: rhs_ndim (number of dimensions in rhs)
/// - metadata[3]: batch_ndim (number of batch dimensions)
/// - metadata[4..4+lhs_ndim]: lhs_shape
/// - metadata[4+lhs_ndim..4+lhs_ndim+rhs_ndim]: rhs_shape
/// - metadata[4+lhs_ndim+rhs_ndim..4+lhs_ndim+rhs_ndim+batch_ndim]: batch_shape
/// - metadata[4+lhs_ndim+rhs_ndim+batch_ndim..4+2*lhs_ndim+rhs_ndim+batch_ndim]: lhs_strides
/// - metadata[4+2*lhs_ndim+rhs_ndim+batch_ndim..4+2*lhs_ndim+2*rhs_ndim+batch_ndim]: rhs_strides
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim]: lhs_offset
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+1]: rhs_offset
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+2]: M (rows of A)
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+3]: K (cols of A / rows of B)
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+4]: N (cols of B)
pub fn call_ops_matmul<T>(
    kernel: crate::kernels::macros::Kernel,
    device: &Arc<CudaDevice>,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let kernels = get_global_kernels();
    let func = kernels.load_function(device, Source::OpsMatrix, kernel.0)?;

    // Extract M, N, and batch info from metadata
    let lhs_ndim = metadata[1];
    let rhs_ndim = metadata[2];
    let batch_ndim = metadata[3];

    let metadata_base = 4 + lhs_ndim + rhs_ndim + batch_ndim + lhs_ndim + rhs_ndim;
    let m = metadata[metadata_base + 2];
    let n = metadata[metadata_base + 4];

    // Calculate total number of batches
    let num_batches = if batch_ndim == 0 {
        1
    } else {
        let batch_shape = &metadata[4 + lhs_ndim + rhs_ndim..4 + lhs_ndim + rhs_ndim + batch_ndim];
        batch_shape.iter().product()
    };

    // For matrix multiplication, we use 2D thread blocks with tiling
    const TILE_SIZE: u32 = 16;

    let grid_width = ((n as u32 + TILE_SIZE - 1) / TILE_SIZE).max(1);
    let grid_height = ((m as u32 + TILE_SIZE - 1) / TILE_SIZE).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_width, grid_height, num_batches as u32),
        block_dim: (TILE_SIZE, TILE_SIZE, 1),
        shared_mem_bytes: 0,
    };

    let stream = device.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(lhs).arg(rhs).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a tiled 2D matrix multiplication optimized with shared memory
///
/// # Arguments
/// * `device` - CUDA device to execute on
/// * `kernels` - Kernel cache
/// * `kernel_name` - The dot kernel (e.g., "dot_f32")
/// * `lhs` - Left input matrix device slice with shape [M, K]
/// * `rhs` - Right input matrix device slice with shape [K, N]
/// * `output` - Output matrix device slice with shape [M, N]
/// * `m` - Number of rows in A (and output)
/// * `n` - Number of columns in B (and output)
/// * `metadata` - Device slice containing metadata describing matrix layout
///
/// # Metadata layout
/// - metadata[0]: M (rows of A)
/// - metadata[1]: K (cols of A / rows of B)
/// - metadata[2]: N (cols of B)
/// - metadata[3]: lhs_stride_m (stride for row dimension of A)
/// - metadata[4]: lhs_stride_k (stride for col dimension of A)
/// - metadata[5]: rhs_stride_k (stride for row dimension of B)
/// - metadata[6]: rhs_stride_n (stride for col dimension of B)
/// - metadata[7]: lhs_offset (starting offset in lhs buffer)
/// - metadata[8]: rhs_offset (starting offset in rhs buffer)
pub fn call_ops_dot<T>(
    kernel: crate::kernels::macros::Kernel,
    device: &Arc<CudaDevice>,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let kernels = get_global_kernels();
    let func = kernels.load_function(device, Source::OpsMatrix, kernel.0)?;

    // Extract matrix dimensions from metadata
    let m = metadata[0];
    let n = metadata[2];

    // Optimized dot product with register blocking (4x4 per thread)
    // Tile size is 32x32, with 8x8 threadgroups
    const DOT_TILE_SIZE: u32 = 32;
    const BLOCK_SIZE: u32 = 4;
    const THREADS_PER_DIM: u32 = DOT_TILE_SIZE / BLOCK_SIZE; // 8

    let grid_width = ((n as u32 + DOT_TILE_SIZE - 1) / DOT_TILE_SIZE).max(1);
    let grid_height = ((m as u32 + DOT_TILE_SIZE - 1) / DOT_TILE_SIZE).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_width, grid_height, 1),
        block_dim: (THREADS_PER_DIM, THREADS_PER_DIM, 1),
        shared_mem_bytes: 0,
    };

    let stream = device.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(lhs).arg(rhs).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
