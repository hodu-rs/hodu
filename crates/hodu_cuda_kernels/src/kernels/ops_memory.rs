use crate::{
    compat::*,
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(contiguous, copy);

/// Execute a contiguous operation to convert a strided tensor to contiguous layout
///
/// # Arguments
/// * `kernel` - The contiguous kernel (e.g., contiguous::F32)
/// * `device` - CUDA device to execute on
/// * `input` - Input tensor device slice (may have non-contiguous strides)
/// * `output` - Output tensor device slice (will be contiguous)
/// * `metadata` - Device slice containing metadata describing tensor layout
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape (dimensions of the tensor)
/// - metadata[2+num_dims..2+2*num_dims]: strides (stride for each dimension)
/// - metadata[2+2*num_dims]: offset (starting offset in input buffer)
pub fn call_ops_contiguous<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsMemory, kernel.0)?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = ((num_els as u32 + block_size - 1) / block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
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

/// Execute a copy operation with strided tensor support
///
/// # Arguments
/// * `kernel` - The copy kernel (e.g., copy::F32)
/// * `device` - CUDA device to execute on
/// * `input` - Input tensor device slice (source, may have non-contiguous strides)
/// * `output` - Output tensor device slice (destination)
/// * `metadata` - Metadata describing the tensor layout
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape (dimensions of the tensor)
/// - metadata[2+num_dims..2+2*num_dims]: strides (stride for each dimension)
/// - metadata[2+2*num_dims]: offset (starting offset in input buffer)
pub fn call_ops_copy<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsMemory, kernel.0)?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = ((num_els as u32 + block_size - 1) / block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
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
