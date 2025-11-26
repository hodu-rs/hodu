use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    compat::*,
    error::{HoduError, HoduResult},
    ops::Op,
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};

#[allow(clippy::too_many_arguments)]
pub fn call_ops_conv(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    weight_storage: &CudaStorage,
    weight_layout: &Layout,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    op: Op,
) -> HoduResult<CudaStorage> {
    let conv_op = match op {
        Op::Conv(conv_op) => conv_op,
        _ => return Err(HoduError::BackendError("call_ops_conv expects conv op".to_string())),
    };

    let input_shape = input_layout.shape();
    let weight_shape = weight_layout.shape();

    let batch_size = input_shape[0];
    let out_channels = weight_shape[0];

    let spatial_dims = input_shape.ndim() - 2;
    let mut output_shape_vec = Vec::with_capacity(spatial_dims + 2);
    output_shape_vec.push(batch_size);
    output_shape_vec.push(out_channels);

    for i in 0..spatial_dims {
        let input_size = input_shape[2 + i];
        let kernel_size = weight_shape[2 + i];
        let output_size = (input_size + 2 * padding[i] - dilation[i] * (kernel_size - 1) - 1) / stride[i] + 1;
        output_shape_vec.push(output_size);
    }

    let output_size: usize = output_shape_vec.iter().product();

    let metadata = match spatial_dims {
        1 => crate::op_metadatas::conv1d_metadata(
            input_layout,
            weight_layout,
            stride[0],
            padding[0],
            dilation[0],
            &output_shape_vec,
        ),
        2 => crate::op_metadatas::conv2d_metadata(
            input_layout,
            weight_layout,
            stride,
            padding,
            dilation,
            &output_shape_vec,
        ),
        3 => crate::op_metadatas::conv3d_metadata(
            input_layout,
            weight_layout,
            stride,
            padding,
            dilation,
            &output_shape_vec,
        ),
        _ => {
            return Err(HoduError::BackendError(format!(
                "unsupported spatial dimensions: {}",
                spatial_dims
            )))
        },
    };

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();
    let device_id = input_storage.device_id();
    let device_arc = Arc::clone(&input_storage.device);

    let kernel_name = format!("{}_{}", conv_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_conv {
        ($input:expr, $weight:expr, $ty:ty, $variant:ident) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(output_size as usize)?;
            kernels::call_ops_conv(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                $weight,
                &mut output,
                &metadata,
            )?;
            Ok(CudaStorage::new(
                device_id,
                Arc::clone(&device_arc),
                CudaStorageData::$variant(output),
            ))
        }};
    }

    match (&input_storage.data, &weight_storage.data) {
        (CudaStorageData::BOOL(input), CudaStorageData::BOOL(weight)) => call_conv!(input, weight, bool, BOOL),
        (CudaStorageData::F8E4M3(input), CudaStorageData::F8E4M3(weight)) => {
            call_conv!(input, weight, float8::F8E4M3, F8E4M3)
        },
        #[cfg(feature = "f8e5m2")]
        (CudaStorageData::F8E5M2(input), CudaStorageData::F8E5M2(weight)) => {
            call_conv!(input, weight, float8::F8E5M2, F8E5M2)
        },
        (CudaStorageData::BF16(input), CudaStorageData::BF16(weight)) => call_conv!(input, weight, half::bf16, BF16),
        (CudaStorageData::F16(input), CudaStorageData::F16(weight)) => call_conv!(input, weight, half::f16, F16),
        (CudaStorageData::F32(input), CudaStorageData::F32(weight)) => call_conv!(input, weight, f32, F32),
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(input), CudaStorageData::F64(weight)) => call_conv!(input, weight, f64, F64),
        (CudaStorageData::U8(input), CudaStorageData::U8(weight)) => call_conv!(input, weight, u8, U8),
        #[cfg(feature = "u16")]
        (CudaStorageData::U16(input), CudaStorageData::U16(weight)) => call_conv!(input, weight, u16, U16),
        (CudaStorageData::U32(input), CudaStorageData::U32(weight)) => call_conv!(input, weight, u32, U32),
        #[cfg(feature = "u64")]
        (CudaStorageData::U64(input), CudaStorageData::U64(weight)) => call_conv!(input, weight, u64, U64),
        (CudaStorageData::I8(input), CudaStorageData::I8(weight)) => call_conv!(input, weight, i8, I8),
        #[cfg(feature = "i16")]
        (CudaStorageData::I16(input), CudaStorageData::I16(weight)) => call_conv!(input, weight, i16, I16),
        (CudaStorageData::I32(input), CudaStorageData::I32(weight)) => call_conv!(input, weight, i32, I32),
        #[cfg(feature = "i64")]
        (CudaStorageData::I64(input), CudaStorageData::I64(weight)) => call_conv!(input, weight, i64, I64),
        _ => Err(HoduError::BackendError("mismatched storage types in conv".to_string())),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_ops_conv_grad_weight(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    grad_output_storage: &CudaStorage,
    grad_output_layout: &Layout,
    weight_shape: &crate::types::Shape,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    op: Op,
) -> HoduResult<CudaStorage> {
    // Validate op
    let conv_op = match op {
        Op::Conv(conv_op) => conv_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_conv_grad_weight expects conv op".to_string(),
            ))
        },
    };

    let input_shape = input_layout.shape();
    let spatial_dims = input_shape.ndim() - 2;
    let num_els = weight_shape.size();

    let metadata = crate::op_metadatas::conv_grad_weight_metadata(
        input_layout,
        grad_output_layout,
        weight_shape.dims(),
        stride,
        padding,
        dilation,
        spatial_dims,
    );

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();
    let device_id = input_storage.device_id();
    let device_arc = Arc::clone(&input_storage.device);

    let kernel_name = format!("{}_{}", conv_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_conv_grad_weight {
        ($input:expr, $grad_output:expr, $ty:ty, $variant:ident) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(num_els as usize)?;
            kernels::call_ops_conv_grad_weight(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                $grad_output,
                &mut output,
                &metadata,
            )?;
            Ok(CudaStorage::new(
                device_id,
                Arc::clone(&device_arc),
                CudaStorageData::$variant(output),
            ))
        }};
    }

    match (&input_storage.data, &grad_output_storage.data) {
        (CudaStorageData::BOOL(input), CudaStorageData::BOOL(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, bool, BOOL)
        },
        (CudaStorageData::F8E4M3(input), CudaStorageData::F8E4M3(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, float8::F8E4M3, F8E4M3)
        },
        #[cfg(feature = "f8e5m2")]
        (CudaStorageData::F8E5M2(input), CudaStorageData::F8E5M2(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, float8::F8E5M2, F8E5M2)
        },
        (CudaStorageData::BF16(input), CudaStorageData::BF16(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, half::bf16, BF16)
        },
        (CudaStorageData::F16(input), CudaStorageData::F16(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, half::f16, F16)
        },
        (CudaStorageData::F32(input), CudaStorageData::F32(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, f32, F32)
        },
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(input), CudaStorageData::F64(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, f64, F64)
        },
        (CudaStorageData::U8(input), CudaStorageData::U8(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, u8, U8)
        },
        #[cfg(feature = "u16")]
        (CudaStorageData::U16(input), CudaStorageData::U16(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, u16, U16)
        },
        (CudaStorageData::U32(input), CudaStorageData::U32(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, u32, U32)
        },
        #[cfg(feature = "u64")]
        (CudaStorageData::U64(input), CudaStorageData::U64(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, u64, U64)
        },
        (CudaStorageData::I8(input), CudaStorageData::I8(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, i8, I8)
        },
        #[cfg(feature = "i16")]
        (CudaStorageData::I16(input), CudaStorageData::I16(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, i16, I16)
        },
        (CudaStorageData::I32(input), CudaStorageData::I32(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, i32, I32)
        },
        #[cfg(feature = "i64")]
        (CudaStorageData::I64(input), CudaStorageData::I64(grad_output)) => {
            call_conv_grad_weight!(input, grad_output, i64, I64)
        },
        _ => Err(HoduError::BackendError(
            "mismatched storage types in conv_grad_weight".to_string(),
        )),
    }
}
