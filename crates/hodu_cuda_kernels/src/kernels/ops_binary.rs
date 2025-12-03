use crate::{
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(
    add,
    sub,
    mul,
    div,
    pow,
    maximum,
    minimum,
    logical_and,
    logical_or,
    logical_xor,
    eq,
    ne,
    lt,
    le,
    gt,
    ge
);

/// Execute a binary operation on two tensors
///
/// Performs element-wise binary operations on tensors with arbitrary shapes and strides.
/// The function automatically handles contiguous and strided memory layouts for optimal
/// performance. Broadcasting is not handled here - shapes must be compatible.
///
/// # Arguments
/// * `kernel` - The binary operation to perform (e.g., "add::F32", "mul::I32")
/// * `kernels` - Kernel cache for managing compiled kernels
/// * `context` - CUDA context to execute on
/// * `lhs` - Left-hand side tensor device slice
/// * `rhs` - Right-hand side tensor device slice
/// * `output` - Output tensor device slice
/// * `metadata` - Device slice containing metadata describing tensor layout
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: lhs_shape (shape of left tensor)
/// - metadata[2+num_dims..2+2*num_dims]: rhs_shape (shape of right tensor)
/// - metadata[2+2*num_dims..2+3*num_dims]: lhs_strides (stride of left tensor)
/// - metadata[2+3*num_dims..2+4*num_dims]: rhs_strides (stride of right tensor)
/// - metadata[2+4*num_dims]: lhs_offset (starting offset in left tensor)
/// - metadata[2+4*num_dims+1]: rhs_offset (starting offset in right tensor)
///
/// Total metadata length: `2 + num_dims * 4 + 2`
pub fn call_ops_binary<I, O>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    lhs: &CudaSlice<I>,
    rhs: &CudaSlice<I>,
    output: &mut CudaSlice<O>,
    metadata: &[usize],
) -> Result<()>
where
    I: cudarc::driver::DeviceRepr,
    O: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsBinary, kernel.0)?;

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
            args.arg(lhs).arg(rhs).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
