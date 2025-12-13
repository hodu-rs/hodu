use crate::{
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(
    index_select,
    index_put,
    gather,
    scatter,
    scatter_add,
    scatter_max,
    scatter_min,
    onehot,
    nonzero_count,
    nonzero_fill,
    unique_sort,
    unique_bitonic_step,
    unique_count,
    unique_mark,
    unique_build
);

/// Execute an index_select operation
///
/// Selects elements along a dimension using a 1D indices array.
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: input_strides
/// - metadata[2+2*num_dims]: input_offset
/// - metadata[2+2*num_dims+1]: dim (dimension along which to select)
/// - metadata[2+2*num_dims+2]: num_indices (number of indices)
///
/// # Kernel signature
/// `(input, indices, output, metadata)`
#[allow(clippy::too_many_arguments)]
pub fn call_ops_index_select<T, I>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    indices: &CudaSlice<I>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
    I: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsIndexing, kernel.0)?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Kernel signature: (input, indices, output, metadata)
    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(indices).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute an index_put operation
///
/// Puts values into tensor at specified indices along a dimension.
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: input_strides
/// - metadata[2+2*num_dims..2+3*num_dims]: values_strides
/// - metadata[2+3*num_dims]: input_offset
/// - metadata[2+3*num_dims+1]: values_offset
/// - metadata[2+3*num_dims+2]: dim (dimension along which to put)
/// - metadata[2+3*num_dims+3]: num_indices (number of indices)
///
/// # Kernel signature
/// `(input, indices, values, output, metadata)`
#[allow(clippy::too_many_arguments)]
pub fn call_ops_index_put<T, I>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    indices: &CudaSlice<I>,
    values: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
    I: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsIndexing, kernel.0)?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Kernel signature: (input, indices, values, output, metadata)
    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(indices).arg(values).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a gather operation
///
/// Gathers values from input tensor using indices tensor.
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: input_strides
/// - metadata[2+2*num_dims..2+3*num_dims]: indices_strides
/// - metadata[2+3*num_dims]: input_offset
/// - metadata[2+3*num_dims+1]: indices_offset
/// - metadata[2+3*num_dims+2]: dim (gather dimension)
/// - metadata[2+3*num_dims+3]: num_indices
///
/// # Kernel signature
/// `(input, indices, output, metadata)`
#[allow(clippy::too_many_arguments)]
pub fn call_ops_gather<T, I>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    indices: &CudaSlice<I>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
    I: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsIndexing, kernel.0)?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Kernel signature: (input, indices, output, metadata)
    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(indices).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a scatter operation (scatter, scatter_add, scatter_max, scatter_min)
///
/// Scatters src values into output at positions specified by indices.
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of src elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: input_strides
/// - metadata[2+2*num_dims..2+3*num_dims]: src_shape
/// - metadata[2+3*num_dims..2+4*num_dims]: src_strides
/// - metadata[2+4*num_dims..2+5*num_dims]: indices_strides
/// - metadata[2+5*num_dims]: input_offset
/// - metadata[2+5*num_dims+1]: src_offset
/// - metadata[2+5*num_dims+2]: indices_offset
/// - metadata[2+5*num_dims+3]: dim (scatter dimension)
///
/// # Kernel signature
/// `(input, indices, src, output, metadata)`
#[allow(clippy::too_many_arguments)]
pub fn call_ops_scatter<T, I>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    indices: &CudaSlice<I>,
    src: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
    I: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsIndexing, kernel.0)?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Kernel signature: (input, indices, src, output, metadata)
    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(indices).arg(src).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a onehot operation
///
/// Converts integer indices to one-hot encoded vectors.
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_input_els (total number of input indices)
/// - metadata[2]: num_classes (depth of one-hot dimension)
/// - metadata[3]: axis (dimension for one-hot encoding)
/// - metadata[4]: num_dims_out (number of output dimensions)
/// - metadata[5..5+num_dims_out]: output_shape
///
/// # Kernel signature
/// `(indices, output, metadata)`
#[allow(clippy::too_many_arguments)]
pub fn call_ops_onehot<T, I>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    indices: &CudaSlice<I>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
    I: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsIndexing, kernel.0)?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Kernel signature: (indices, output, metadata)
    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(indices).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Count non-zero elements in a tensor.
///
/// # Arguments
/// * `kernel` - The nonzero_count kernel to execute
/// * `kernels` - Kernel cache
/// * `context` - CUDA context
/// * `input` - Input data slice
/// * `count` - Output count (single u32)
/// * `metadata` - Metadata describing tensor layout
pub fn call_nonzero_count<T>(
    kernel: Kernel,
    kernels: &crate::kernel::Kernels,
    context: &crate::cuda::CudaContext,
    input: &CudaSlice<T>,
    count: &mut CudaSlice<u32>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsIndexing, kernel.0)?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Kernel signature: (input, count, metadata)
    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(count).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Fill indices of non-zero elements.
///
/// # Arguments
/// * `kernel` - The nonzero_fill kernel to execute
/// * `kernels` - Kernel cache
/// * `context` - CUDA context
/// * `input` - Input data slice
/// * `output` - Output indices (i32, shape [N, ndim])
/// * `counter` - Atomic counter (single u32, should be initialized to 0)
/// * `metadata` - Metadata describing tensor layout
pub fn call_nonzero_fill<T>(
    kernel: Kernel,
    kernels: &crate::kernel::Kernels,
    context: &crate::cuda::CudaContext,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<i32>,
    counter: &mut CudaSlice<u32>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsIndexing, kernel.0)?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Kernel signature: (input, output, counter, metadata)
    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(output).arg(counter).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Copy input to sorted buffer and initialize indices for unique operation.
/// metadata layout: [num_els, offset, padded_size]
pub fn call_unique_sort<T>(
    kernel: Kernel,
    kernels: &crate::kernel::Kernels,
    context: &crate::cuda::CudaContext,
    input: &CudaSlice<T>,
    sorted_values: &mut CudaSlice<T>,
    sorted_indices: &mut CudaSlice<i32>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsIndexing, kernel.0)?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    // Use padded_size (metadata[2]) for thread count
    let padded_size = metadata[2];
    let block_size = 256u32;
    let grid_size = (padded_size as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input)
                .arg(sorted_values)
                .arg(sorted_indices)
                .arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute bitonic sort step for unique operation.
/// metadata layout: [num_els, offset, padded_size, k, j]
pub fn call_unique_bitonic_step<T>(
    kernel: Kernel,
    kernels: &crate::kernel::Kernels,
    context: &crate::cuda::CudaContext,
    values: &mut CudaSlice<T>,
    indices: &mut CudaSlice<i32>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsIndexing, kernel.0)?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    // Use padded_size (metadata[2]) for thread count
    let padded_size = metadata[2];
    let block_size = 256u32;
    let grid_size = (padded_size as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(values).arg(indices).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Count unique elements.
pub fn call_unique_count<T>(
    kernel: Kernel,
    kernels: &crate::kernel::Kernels,
    context: &crate::cuda::CudaContext,
    sorted_values: &CudaSlice<T>,
    count: &mut CudaSlice<u32>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsIndexing, kernel.0)?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(sorted_values).arg(count).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Mark unique boundaries.
pub fn call_unique_mark<T>(
    kernel: Kernel,
    kernels: &crate::kernel::Kernels,
    context: &crate::cuda::CudaContext,
    sorted_values: &CudaSlice<T>,
    marks: &mut CudaSlice<u32>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsIndexing, kernel.0)?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(sorted_values).arg(marks).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Compute prefix sum for unique indices (single thread).
pub fn call_unique_prefix_sum(
    kernels: &crate::kernel::Kernels,
    context: &crate::cuda::CudaContext,
    marks: &CudaSlice<u32>,
    unique_idx: &mut CudaSlice<i32>,
    metadata: &[usize],
) -> Result<()> {
    let func = kernels.load_function(context, Source::OpsIndexing, "hodu_cuda_unique_prefix_sum")?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(marks).arg(unique_idx).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Build output arrays for unique operation.
#[allow(clippy::too_many_arguments)]
pub fn call_unique_build<T>(
    kernel: Kernel,
    kernels: &crate::kernel::Kernels,
    context: &crate::cuda::CudaContext,
    sorted_values: &CudaSlice<T>,
    sorted_indices: &CudaSlice<i32>,
    marks: &CudaSlice<u32>,
    unique_idx: &CudaSlice<i32>,
    values: &mut CudaSlice<T>,
    inverse: &mut CudaSlice<i32>,
    counts: &mut CudaSlice<i32>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsIndexing, kernel.0)?;

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(sorted_values)
                .arg(sorted_indices)
                .arg(marks)
                .arg(unique_idx)
                .arg(values)
                .arg(inverse)
                .arg(counts)
                .arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
