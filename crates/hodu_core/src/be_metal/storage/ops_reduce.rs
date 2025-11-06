use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_reduce(
    storage: &MetalStorage,
    layout: &Layout,
    dims: &[u32],
    keep_dim: bool,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract reduce op
    let reduce_op = match op {
        Op::Reduce(reduce_op) => reduce_op,
        _ => return Err(HoduError::BackendError("Lcall_reduceE expects LreduceE op".to_string())),
    };

    let input_shape = layout.shape();
    let input_ndim = input_shape.ndim();

    // Validate reduce dimensions
    for &dim in dims {
        if dim >= input_ndim {
            return Err(HoduError::InvalidAxis {
                axis: dim as i32,
                ndim: input_ndim,
            });
        }
    }

    // Compute output shape
    let mut output_shape_vec = Vec::new();
    for i in 0..input_ndim {
        if dims.contains(&i) {
            if keep_dim {
                output_shape_vec.push(1);
            }
        } else {
            output_shape_vec.push(input_shape.dims()[i as usize]);
        }
    }

    // Handle empty output shape (reduce all dimensions without keep_dim)
    if output_shape_vec.is_empty() {
        output_shape_vec.push(1);
    }

    let output_shape = Shape::new(&output_shape_vec);
    let output_size = output_shape.size();

    // Calculate reduce size
    let mut reduce_size: u64 = 1;
    for &dim in dims {
        reduce_size *= input_shape.dims()[dim as usize] as u64;
    }

    let dtype = storage.dtype();
    let device = storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(output_size as usize, dtype, "reduce_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", reduce_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Convert to usize for kernel call
    let shape_usize: Vec<usize> = input_shape.dims().iter().map(|&d| d as usize).collect();
    let strides_usize: Vec<usize> = layout.strides().iter().map(|&s| s as usize).collect();
    let reduce_dims_usize: Vec<usize> = dims.iter().map(|&d| d as usize).collect();

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_reduce(
        device.device(),
        &command_buffer,
        device.kernels(),
        kernel,
        &shape_usize,
        input_offset,
        &strides_usize,
        layout.offset() as usize,
        &reduce_dims_usize,
        reduce_size as usize,
        keep_dim,
        &output_buffer,
    )?;

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        output_size as usize,
        dtype,
    ))
}
