use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_reduce(
    storage: &MetalStorage,
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract reduce op
    let reduce_op = match op {
        Op::Reduce(reduce_op) => reduce_op,
        _ => {
            return Err(HoduError::BackendError(
                "Lcall_ops_reduceE expects LreduceE op".to_string(),
            ))
        },
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
            output_shape_vec.push(input_shape.dims()[i]);
        }
    }

    // Handle empty output shape (reduce all dimensions without keep_dim)
    if output_shape_vec.is_empty() {
        output_shape_vec.push(1);
    }

    let output_shape = Shape::new(&output_shape_vec);
    let output_size = output_shape.size();

    // Calculate reduce size
    let mut reduce_size: usize = 1;
    for &dim in dims {
        reduce_size *= input_shape.dims()[dim];
    }

    let dtype = storage.dtype();
    let device = storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(output_size, dtype, "reduce_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", reduce_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Build metadata
    // Layout: [num_dims, shape..., strides..., offset, output_shape_len, output_shape..., num_reduce_dims, reduce_dims..., keep_dim, reduce_size]
    let num_dims = input_ndim;
    let output_shape_len = output_shape_vec.len();
    let num_reduce_dims = dims.len();

    let mut metadata = Vec::with_capacity(1 + num_dims * 2 + 1 + 1 + output_shape_len + 1 + num_reduce_dims + 2);
    metadata.push(num_dims);
    metadata.extend(input_shape.dims().iter().copied());
    metadata.extend(layout.strides().iter().copied());
    metadata.push(layout.offset());
    metadata.push(output_shape_len);
    metadata.extend(output_shape_vec.iter().copied());
    metadata.push(num_reduce_dims);
    metadata.extend(dims.iter().copied());
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_reduce(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), output_size, dtype))
}
