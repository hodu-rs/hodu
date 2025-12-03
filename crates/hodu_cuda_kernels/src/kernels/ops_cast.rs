use crate::{
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(cast);

/// Execute a type cast operation on a tensor
///
/// # Arguments
/// * `kernel` - The cast kernel (e.g., cast::F32_I32 to cast from f32 to i32)
/// * `kernels` - Kernel cache for managing compiled kernels
/// * `context` - CUDA context to execute on
/// * `input` - Input tensor device slice
/// * `output` - Output tensor device slice (different type than input)
/// * `metadata` - Device slice containing metadata describing tensor layout
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape (dimensions of the tensor)
/// - metadata[2+num_dims..2+2*num_dims]: strides (stride for each dimension)
/// - metadata[2+2*num_dims]: offset (starting offset in input buffer)
pub fn call_ops_cast<I, O>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<I>,
    output: &mut CudaSlice<O>,
    metadata: &[usize],
) -> Result<()>
where
    I: cudarc::driver::DeviceRepr,
    O: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsCast, kernel.0)?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

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
