use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    scalar::Scalar,
    types::{DType, Device, Layout},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_cmp_scalar(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    scalar: Scalar,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract cmp scalar op
    let cmp_op = match op {
        Op::CmpScalar(cmp_op) => cmp_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_cmp_scalar expects cmp scalar op".to_string(),
            ))
        },
    };

    let shape = input_layout.shape();
    let num_els = shape.size();
    let num_dims = shape.ndim();

    // Build metadata array for Metal kernel
    let mut metadata = Vec::with_capacity(2 + 2 * num_dims as usize + 1);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

    // Add shape
    for &dim in shape.dims() {
        metadata.push(dim as usize);
    }

    // Add strides
    for &stride in input_layout.strides() {
        metadata.push(stride as usize);
    }

    // Add offset
    metadata.push(input_layout.offset() as usize);

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer (cmp ops return BOOL)
    let output_dtype = DType::BOOL;
    let output_buffer = device.new_buffer(num_els as usize, output_dtype, "cmp_scalar_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", cmp_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(input_storage.buffer());

    // Get command buffer and call kernel with scalar
    let command_buffer = device.command_buffer()?;

    macro_rules! call_ops_kernel_scalar {
        ($scalar_val:expr) => {{
            kernels::call_ops_unary_scalar(
                kernel,
                device.kernels(),
                device.device(),
                &command_buffer,
                input_offset,
                &output_buffer,
                &metadata,
                $scalar_val,
            )?;
        }};
    }

    match scalar {
        Scalar::BOOL(s) => call_ops_kernel_scalar!(s),
        Scalar::BF16(s) => call_ops_kernel_scalar!(s),
        Scalar::F16(s) => call_ops_kernel_scalar!(s),
        Scalar::F32(s) => call_ops_kernel_scalar!(s),
        Scalar::U8(s) => call_ops_kernel_scalar!(s),
        #[cfg(feature = "u16")]
        Scalar::U16(s) => call_ops_kernel_scalar!(s),
        Scalar::U32(s) => call_ops_kernel_scalar!(s),
        #[cfg(feature = "u64")]
        Scalar::U64(s) => call_ops_kernel_scalar!(s),
        Scalar::I8(s) => call_ops_kernel_scalar!(s),
        #[cfg(feature = "i16")]
        Scalar::I16(s) => call_ops_kernel_scalar!(s),
        Scalar::I32(s) => call_ops_kernel_scalar!(s),
        #[cfg(feature = "i64")]
        Scalar::I64(s) => call_ops_kernel_scalar!(s),
        _ => {
            return Err(HoduError::UnsupportedDTypeForDevice {
                dtype,
                device: Device::Metal,
            })
        },
    }

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        num_els as usize,
        output_dtype,
    ))
}

pub fn call_ops_unary(input_storage: &MetalStorage, input_layout: &Layout, op: Op) -> HoduResult<MetalStorage> {
    // Extract unary op
    let unary_op = match op {
        Op::Unary(unary_op) => unary_op,
        _ => {
            return Err(HoduError::BackendError(
                "Lcall_ops_unaryE expects LunaryE op".to_string(),
            ))
        },
    };

    let shape = input_layout.shape();
    let num_els = shape.size();
    let num_dims = shape.ndim();

    // Build metadata array for Metal kernel
    let mut metadata = Vec::with_capacity(2 + 2 * num_dims as usize + 1);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

    // Add shape
    for &dim in shape.dims() {
        metadata.push(dim as usize);
    }

    // Add strides
    for &stride in input_layout.strides() {
        metadata.push(stride as usize);
    }

    // Add offset
    metadata.push(input_layout.offset() as usize);

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "unary_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", unary_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(input_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_unary(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        num_els as usize,
        dtype,
    ))
}

pub fn call_ops_unary_logical(input_storage: &MetalStorage, input_layout: &Layout, op: Op) -> HoduResult<MetalStorage> {
    // Extract unary logical op
    let unary_op = match op {
        Op::UnaryLogical(unary_op) => unary_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_unary_logical expects unary logical op".to_string(),
            ))
        },
    };

    let shape = input_layout.shape();
    let num_els = shape.size();
    let num_dims = shape.ndim();

    // Build metadata array for Metal kernel
    let mut metadata = Vec::with_capacity(2 + 2 * num_dims as usize + 1);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

    // Add shape
    for &dim in shape.dims() {
        metadata.push(dim as usize);
    }

    // Add strides
    for &stride in input_layout.strides() {
        metadata.push(stride as usize);
    }

    // Add offset
    metadata.push(input_layout.offset() as usize);

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer (logical ops return BOOL)
    let output_dtype = DType::BOOL;
    let output_buffer = device.new_buffer(num_els as usize, output_dtype, "unary_logical_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", unary_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(input_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_unary(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        num_els as usize,
        output_dtype,
    ))
}

pub fn call_ops_unary_scalar(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    scalar: Scalar,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract unary scalar op
    let unary_op = match op {
        Op::UnaryScalar(unary_op) => unary_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_unary_scalar expects unary scalar op".to_string(),
            ))
        },
    };

    let shape = input_layout.shape();
    let num_els = shape.size();
    let num_dims = shape.ndim();

    // Build metadata array for Metal kernel
    let mut metadata = Vec::with_capacity(2 + 2 * num_dims as usize + 1);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

    // Add shape
    for &dim in shape.dims() {
        metadata.push(dim as usize);
    }

    // Add strides
    for &stride in input_layout.strides() {
        metadata.push(stride as usize);
    }

    // Add offset
    metadata.push(input_layout.offset() as usize);

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "unary_scalar_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", unary_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(input_storage.buffer());

    // Get command buffer and call kernel with scalar
    let command_buffer = device.command_buffer()?;

    macro_rules! call_ops_kernel_scalar {
        ($scalar_val:expr) => {{
            kernels::call_ops_unary_scalar(
                kernel,
                device.kernels(),
                device.device(),
                &command_buffer,
                input_offset,
                &output_buffer,
                &metadata,
                $scalar_val,
            )?;
        }};
    }

    match scalar {
        Scalar::BOOL(s) => call_ops_kernel_scalar!(s),
        Scalar::BF16(s) => call_ops_kernel_scalar!(s),
        Scalar::F16(s) => call_ops_kernel_scalar!(s),
        Scalar::F32(s) => call_ops_kernel_scalar!(s),
        Scalar::U8(s) => call_ops_kernel_scalar!(s),
        #[cfg(feature = "u16")]
        Scalar::U16(s) => call_ops_kernel_scalar!(s),
        Scalar::U32(s) => call_ops_kernel_scalar!(s),
        #[cfg(feature = "u64")]
        Scalar::U64(s) => call_ops_kernel_scalar!(s),
        Scalar::I8(s) => call_ops_kernel_scalar!(s),
        #[cfg(feature = "i16")]
        Scalar::I16(s) => call_ops_kernel_scalar!(s),
        Scalar::I32(s) => call_ops_kernel_scalar!(s),
        #[cfg(feature = "i64")]
        Scalar::I64(s) => call_ops_kernel_scalar!(s),
        _ => {
            return Err(HoduError::UnsupportedDTypeForDevice {
                dtype,
                device: Device::Metal,
            })
        },
    }

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        num_els as usize,
        dtype,
    ))
}
