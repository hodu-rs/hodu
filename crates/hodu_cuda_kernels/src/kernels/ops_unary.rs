use crate::{
    compat::*,
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::get_global_kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(
    neg,
    abs,
    sign,
    square,
    sqrt,
    recip,
    relu,
    sigmoid,
    tanh,
    gelu,
    softplus,
    silu,
    mish,
    sin,
    cos,
    tan,
    exp,
    exp2,
    exp10,
    ln,
    log2,
    log10,
    logical_not,
    add_scalar,
    sub_scalar,
    mul_scalar,
    div_scalar,
    pow_scalar,
    maximum_scalar,
    minimum_scalar,
    leaky_relu,
    elu,
    prelu,
    eq_scalar,
    ne_scalar,
    lt_scalar,
    le_scalar,
    gt_scalar,
    ge_scalar
);

/// Execute a unary operation on a tensor
///
/// # Arguments
/// * `kernel` - The unary operation to perform (e.g., "abs::F32", "exp::F32")
/// * `device` - CUDA device to execute on
/// * `input` - Input tensor device slice
/// * `output` - Output tensor device slice
/// * `metadata` - Device slice containing metadata describing tensor layout
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape (dimensions of the tensor)
/// - metadata[2+num_dims..2+2*num_dims]: strides (stride for each dimension)
/// - metadata[2+2*num_dims]: offset (starting offset in input buffer)
pub fn call_ops_unary<I, O>(
    kernel: crate::kernels::macros::Kernel,
    device: &Arc<CudaDevice>,
    input: &CudaSlice<I>,
    output: &mut CudaSlice<O>,
    metadata: &[usize],
) -> Result<()>
where
    I: cudarc::driver::DeviceRepr,
    O: cudarc::driver::DeviceRepr,
{
    let kernels = get_global_kernels();
    let func = kernels.load_function(device, Source::OpsUnary, kernel.0)?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = ((num_els as u32 + block_size - 1) / block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let stream = device.default_stream();
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

/// Execute a unary operation on a tensor
///
/// # Arguments
/// * `kernel` - The unary operation to perform (e.g., "abs::F32", "exp::F32")
/// * `device` - CUDA device to execute on
/// * `input` - Input tensor device slice
/// * `output` - Output tensor device slice
/// * `metadata` - Device slice containing metadata describing tensor layout
/// * `scalar_val` - Scalar value to use in the operation (must be of the same type as the tensor elements)
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape (dimensions of the tensor)
/// - metadata[2+num_dims..2+2*num_dims]: strides (stride for each dimension)
/// - metadata[2+2*num_dims]: offset (starting offset in input buffer)
pub fn call_ops_unary_scalar<I, O>(
    kernel: crate::kernels::macros::Kernel,
    device: &Arc<CudaDevice>,
    input: &CudaSlice<I>,
    output: &mut CudaSlice<O>,
    metadata: &[usize],
    scalar_val: I,
) -> Result<()>
where
    I: cudarc::driver::DeviceRepr + Clone,
    O: cudarc::driver::DeviceRepr,
{
    let kernels = get_global_kernels();
    let func = kernels.load_function(device, Source::OpsUnary, kernel.0)?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = ((num_els as u32 + block_size - 1) / block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let stream = device.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    let scalar_dev = stream
        .memcpy_stod(&[scalar_val])
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy scalar: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(output).arg(&metadata_dev).arg(&scalar_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
