use crate::{
    compat::*,
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::get_global_kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(sum, max, min, prod, mean, norm, argmax, argmin, all, any);

/// Execute a reduce operation on a tensor
///
/// Performs reductions like sum, max, min, mean, etc. along specified dimensions.
///
/// # Metadata layout
/// - metadata[0]: num_dims (number of dimensions in input)
/// - metadata[1..1+num_dims]: dims (shape of input)
/// - metadata[1+num_dims..1+2*num_dims]: strides (strides of input)
/// - metadata[1+2*num_dims]: offset (starting offset in input)
/// - metadata[2+2*num_dims]: output_shape_len (number of dimensions in output)
/// - metadata[3+2*num_dims..3+2*num_dims+output_shape_len]: output_shape
/// - metadata[3+2*num_dims+output_shape_len]: num_reduce_dims (number of dims to reduce)
/// - metadata[4+2*num_dims+output_shape_len..]: reduce_dims (dimension indices to reduce)
/// - metadata[...+num_reduce_dims]: keep_dim (1 to keep, 0 to squeeze)
/// - metadata[...+1]: reduce_size (total elements to reduce per output)
///
/// # Kernel signature
/// `(input, output, metadata)`
///
/// # Keep_dim behavior
/// - If keep_dim=true: output shape matches input but reduced dims have size 1
/// - If keep_dim=false: reduced dimensions are squeezed out of output
pub fn call_ops_reduce<T, O>(
    kernel: crate::kernels::macros::Kernel,
    device: &Arc<CudaDevice>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<O>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
    O: cudarc::driver::DeviceRepr,
{
    let kernels = get_global_kernels();
    let func = kernels.load_function(device, Source::OpsReduce, kernel.0)?;

    // Calculate num_els from output_shape in metadata
    let num_dims = metadata[0];
    let output_shape_len = metadata[2 + 2 * num_dims];
    let output_shape_start = 3 + 2 * num_dims;
    let num_els: usize = metadata[output_shape_start..output_shape_start + output_shape_len]
        .iter()
        .product();

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

    // Kernel signature: (input, output, metadata)
    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
