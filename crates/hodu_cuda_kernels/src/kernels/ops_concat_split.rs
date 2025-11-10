use crate::{
    compat::*,
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(concat, split, chunk);

/// Executes a concatenation operation on multiple input tensors along a specified dimension.
///
/// # Arguments
/// * `kernel` - Concat operation kernel (e.g., concat::F32)
/// * `kernels` - Kernel cache for managing compiled kernels
/// * `context` - CUDA context to execute on
/// * `input` - Combined input buffer containing all input tensors
/// * `output` - Output buffer for concatenated result
/// * `metadata` - Metadata describing tensor shapes, strides, and offsets
///
/// # Metadata Layout
/// The metadata slice must contain the following elements in order:
/// - `metadata[0]`: num_els (total number of output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: output_shape (shape of output tensor)
/// - `metadata[2+num_dims]`: concat_dim (dimension along which to concatenate)
/// - `metadata[2+num_dims+1]`: num_inputs (number of input tensors)
/// - `metadata[2+num_dims+2..2+num_dims+2+num_inputs*num_dims]`: input_shapes (flattened)
/// - `metadata[2+num_dims+2+num_inputs*num_dims..2+num_dims+2+2*num_inputs*num_dims]`: input_strides (flattened)
/// - `metadata[2+num_dims+2+2*num_inputs*num_dims..2+num_dims+2+2*num_inputs*num_dims+num_inputs]`: input_offsets
/// - `metadata[2+num_dims+2+2*num_inputs*num_dims+num_inputs..]`: input_buffer_offsets
///
/// Total metadata length: `2 + num_dims + 2 + num_inputs * (2 * num_dims + 2)`
pub fn call_ops_concat<T>(
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
    let func = kernels.load_function(context, Source::OpsConcatSplit, kernel.0)?;

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

/// Executes a split operation to extract a portion of a tensor along a specified dimension.
///
/// # Arguments
/// * `kernel` - Split operation kernel (e.g., split::F32)
/// * `kernels` - Kernel cache for managing compiled kernels
/// * `context` - CUDA context to execute on
/// * `input` - Input tensor buffer
/// * `output` - Output buffer for extracted portion
/// * `metadata` - Metadata describing tensor shape, strides, and split parameters
///
/// # Metadata Layout
/// The metadata slice must contain the following elements in order:
/// - `metadata[0]`: num_els (total number of output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: input_shape (shape of input tensor)
/// - `metadata[2+num_dims..2+2*num_dims]`: strides (stride for each dimension)
/// - `metadata[2+2*num_dims]`: offset (starting offset in input buffer)
/// - `metadata[2+2*num_dims+1]`: split_dim (dimension along which to split)
/// - `metadata[2+2*num_dims+2]`: output_size_on_dim (size of output on split dimension)
/// - `metadata[2+2*num_dims+3]`: split_offset (offset on split dimension)
///
/// Total metadata length: `2 + num_dims * 2 + 4`
pub fn call_ops_split<T>(
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
    let func = kernels.load_function(context, Source::OpsConcatSplit, kernel.0)?;

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
