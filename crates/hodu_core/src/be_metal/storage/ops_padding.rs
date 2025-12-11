use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::{Op, PaddingOp},
    scalar::Scalar,
    types::{DType, Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_pad(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    pad_before: &[usize],
    pad_after: &[usize],
    pad_value: Scalar,
    op: Op,
) -> HoduResult<MetalStorage> {
    let padding_op = match op {
        Op::Padding(padding_op) => padding_op,
        _ => return Err(HoduError::BackendError("call_ops_pad expects Padding op".to_string())),
    };

    let input_shape = input_layout.shape();
    let ndim = input_shape.ndim();

    if pad_before.len() != ndim {
        return Err(HoduError::BackendError(format!(
            "pad_before length {} does not match tensor ndim {}",
            pad_before.len(),
            ndim
        )));
    }

    if pad_after.len() != ndim {
        return Err(HoduError::BackendError(format!(
            "pad_after length {} does not match tensor ndim {}",
            pad_after.len(),
            ndim
        )));
    }

    let mut output_shape_vec = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let out_size = input_shape.dims()[i] + pad_before[i] + pad_after[i];
        output_shape_vec.push(out_size);
    }

    let output_shape = Shape::new(&output_shape_vec);
    let output_size = output_shape.size();
    let metadata = crate::op_metadatas::padding_metadata(input_layout, pad_before, &output_shape_vec);

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    let output_buffer = device.new_buffer(output_size, dtype, "pad_output")?;
    let kernel_name = format!("hodu_metal_{}_{}", padding_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let input_offset = BufferOffset::zero_offset(input_storage.buffer());
    let command_buffer = device.command_buffer()?;

    match padding_op {
        PaddingOp::PadConstant => {
            let pad_value_buffer = create_pad_value_buffer(device, dtype, pad_value)?;
            let pad_value_offset = BufferOffset::zero_offset(&pad_value_buffer);

            kernels::call_ops_pad_constant(
                kernel,
                device.kernels(),
                device.device(),
                &command_buffer,
                input_offset,
                &output_buffer,
                pad_value_offset,
                &metadata,
            )?;
        },
        PaddingOp::PadReflect => {
            kernels::call_ops_pad_reflect(
                kernel,
                device.kernels(),
                device.device(),
                &command_buffer,
                input_offset,
                &output_buffer,
                &metadata,
            )?;
        },
        PaddingOp::PadReplicate => {
            kernels::call_ops_pad_replicate(
                kernel,
                device.kernels(),
                device.device(),
                &command_buffer,
                input_offset,
                &output_buffer,
                &metadata,
            )?;
        },
        PaddingOp::PadCircular => {
            kernels::call_ops_pad_circular(
                kernel,
                device.kernels(),
                device.device(),
                &command_buffer,
                input_offset,
                &output_buffer,
                &metadata,
            )?;
        },
    }

    Ok(MetalStorage::new(output_buffer, device.clone(), output_size, dtype))
}

fn create_pad_value_buffer(
    device: &crate::be_metal::device::MetalDevice,
    dtype: DType,
    pad_value: Scalar,
) -> HoduResult<std::sync::Arc<hodu_metal_kernels::metal::Buffer>> {
    let buffer = device.new_buffer(1, dtype, "pad_value")?;
    let ptr = buffer.contents();

    match dtype {
        DType::BOOL => {
            let v = pad_value.to_bool();
            unsafe { *ptr = v as u8 };
        },
        DType::BF16 => {
            let v = pad_value.to_bf16();
            unsafe { *(ptr as *mut half::bf16) = v };
        },
        DType::F16 => {
            let v = pad_value.to_f16();
            unsafe { *(ptr as *mut half::f16) = v };
        },
        DType::F32 => {
            let v = pad_value.to_f32();
            unsafe { *(ptr as *mut f32) = v };
        },
        DType::U8 => {
            let v = pad_value.to_u8();
            unsafe { *ptr = v };
        },
        #[cfg(feature = "u16")]
        DType::U16 => {
            let v = pad_value.to_u16();
            unsafe { *(ptr as *mut u16) = v };
        },
        DType::U32 => {
            let v = pad_value.to_u32();
            unsafe { *(ptr as *mut u32) = v };
        },
        #[cfg(feature = "u64")]
        DType::U64 => {
            let v = pad_value.to_u64();
            unsafe { *(ptr as *mut u64) = v };
        },
        DType::I8 => {
            let v = pad_value.to_i8();
            unsafe { *(ptr as *mut i8) = v };
        },
        #[cfg(feature = "i16")]
        DType::I16 => {
            let v = pad_value.to_i16();
            unsafe { *(ptr as *mut i16) = v };
        },
        DType::I32 => {
            let v = pad_value.to_i32();
            unsafe { *(ptr as *mut i32) = v };
        },
        #[cfg(feature = "i64")]
        DType::I64 => {
            let v = pad_value.to_i64();
            unsafe { *(ptr as *mut i64) = v };
        },
        _ => {
            return Err(HoduError::UnsupportedDTypeForDevice {
                dtype,
                device: crate::types::Device::Metal,
            })
        },
    }

    Ok(buffer)
}
