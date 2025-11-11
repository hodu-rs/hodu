use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{ConvOp, Op},
    types::{Layout, Shape},
};
use core::ffi::c_void;
use smallvec::{smallvec, SmallVec};

/// Execute convolution operation
///
/// # Arguments
/// * `input_storage` - Input tensor storage
/// * `input_layout` - Input tensor layout
/// * `weight_storage` - Weight tensor storage
/// * `weight_layout` - Weight tensor layout
/// * `stride` - Stride values
/// * `padding` - Padding values
/// * `dilation` - Dilation values
/// * `op` - The convolution operation
///
/// # Returns
/// Output storage containing the convolution result
#[allow(clippy::too_many_arguments)]
pub fn call_ops_conv(
    input_storage: &CpuStorage,
    input_layout: &Layout,
    weight_storage: &CpuStorage,
    weight_layout: &Layout,
    stride: &[u32],
    padding: &[u32],
    dilation: &[u32],
    op: Op,
) -> HoduResult<CpuStorage> {
    // Validate op
    let conv_op = match op {
        Op::Conv(conv_op) => conv_op,
        _ => return Err(HoduError::BackendError("Lcall_convE expects LConvE op".to_string())),
    };

    // Validate dtypes match
    if input_storage.dtype() != weight_storage.dtype() {
        return Err(HoduError::DTypeMismatch {
            expected: input_storage.dtype(),
            got: weight_storage.dtype(),
        });
    }

    let input_shape = input_layout.shape();
    let weight_shape = weight_layout.shape();
    let dtype = input_storage.dtype();

    // Determine spatial dimensions based on op
    let spatial_dims = match conv_op {
        ConvOp::Conv1d | ConvOp::ConvTranspose1d => 1,
        ConvOp::Conv2d | ConvOp::ConvTranspose2d => 2,
        ConvOp::Conv3d | ConvOp::ConvTranspose3d => 3,
        _ => return Err(HoduError::BackendError("call_conv expects forward conv op".to_string())),
    };

    // Validate input shape: [batch, in_channels, spatial...]
    let input_ndim = input_shape.ndim();
    if input_ndim != (2 + spatial_dims) {
        return Err(HoduError::BackendError(format!(
            "input must have {} dimensions, got {}",
            2 + spatial_dims,
            input_ndim
        )));
    }

    // Validate weight shape: [out_channels, in_channels, kernel...]
    let weight_ndim = weight_shape.ndim();
    if weight_ndim != (2 + spatial_dims) {
        return Err(HoduError::BackendError(format!(
            "weight must have {} dimensions, got {}",
            2 + spatial_dims,
            weight_ndim
        )));
    }

    let batch_size = input_shape.dims()[0];
    let in_channels = input_shape.dims()[1];
    let out_channels = weight_shape.dims()[0];

    // Validate in_channels match
    if weight_shape.dims()[1] != in_channels {
        return Err(HoduError::BackendError(format!(
            "weight in_channels {} does not match input in_channels {}",
            weight_shape.dims()[1],
            in_channels
        )));
    }

    // Build metadata and compute output shape based on spatial dimensions
    let (metadata, output_shape) = match spatial_dims {
        1 => {
            let in_width = input_shape.dims()[2];
            let kernel_width = weight_shape.dims()[2];
            let stride_w = stride[0];
            let padding_w = padding[0];
            let dilation_w = dilation[0];

            let out_width = if matches!(conv_op, ConvOp::ConvTranspose1d) {
                // Transposed convolution
                (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + 1
            } else {
                // Standard convolution
                (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1
            };

            let output_shape = Shape::new(&[batch_size, out_channels, out_width]);
            let num_els = output_shape.size();

            let metadata: SmallVec<[usize; 24]> = smallvec![
                num_els as usize,
                in_channels as usize,
                out_channels as usize,
                in_width as usize,
                kernel_width as usize,
                out_width as usize,
                stride_w as usize,
                padding_w as usize,
                dilation_w as usize,
                input_layout.offset() as usize,
                weight_layout.offset() as usize,
            ];

            (metadata, output_shape)
        },
        2 => {
            let in_height = input_shape.dims()[2];
            let in_width = input_shape.dims()[3];
            let kernel_height = weight_shape.dims()[2];
            let kernel_width = weight_shape.dims()[3];
            let stride_h = stride[0];
            let stride_w = stride[1];
            let padding_h = padding[0];
            let padding_w = padding[1];
            let dilation_h = dilation[0];
            let dilation_w = dilation[1];

            let (out_height, out_width) = if matches!(conv_op, ConvOp::ConvTranspose2d) {
                // Transposed convolution
                let out_h = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_height - 1) + 1;
                let out_w = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + 1;
                (out_h, out_w)
            } else {
                // Standard convolution
                let out_h = (in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
                let out_w = (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;
                (out_h, out_w)
            };

            let output_shape = Shape::new(&[batch_size, out_channels, out_height, out_width]);
            let num_els = output_shape.size();

            let metadata: SmallVec<[usize; 24]> = smallvec![
                num_els as usize,
                in_channels as usize,
                out_channels as usize,
                in_height as usize,
                in_width as usize,
                kernel_height as usize,
                kernel_width as usize,
                out_height as usize,
                out_width as usize,
                stride_h as usize,
                stride_w as usize,
                padding_h as usize,
                padding_w as usize,
                dilation_h as usize,
                dilation_w as usize,
                input_layout.offset() as usize,
                weight_layout.offset() as usize,
            ];

            (metadata, output_shape)
        },
        3 => {
            let in_depth = input_shape.dims()[2];
            let in_height = input_shape.dims()[3];
            let in_width = input_shape.dims()[4];
            let kernel_depth = weight_shape.dims()[2];
            let kernel_height = weight_shape.dims()[3];
            let kernel_width = weight_shape.dims()[4];
            let stride_d = stride[0];
            let stride_h = stride[1];
            let stride_w = stride[2];
            let padding_d = padding[0];
            let padding_h = padding[1];
            let padding_w = padding[2];
            let dilation_d = dilation[0];
            let dilation_h = dilation[1];
            let dilation_w = dilation[2];

            let (out_depth, out_height, out_width) = if matches!(conv_op, ConvOp::ConvTranspose3d) {
                // Transposed convolution
                let out_d = (in_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_depth - 1) + 1;
                let out_h = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_height - 1) + 1;
                let out_w = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + 1;
                (out_d, out_h, out_w)
            } else {
                // Standard convolution
                let out_d = (in_depth + 2 * padding_d - dilation_d * (kernel_depth - 1) - 1) / stride_d + 1;
                let out_h = (in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
                let out_w = (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;
                (out_d, out_h, out_w)
            };

            let output_shape = Shape::new(&[batch_size, out_channels, out_depth, out_height, out_width]);
            let num_els = output_shape.size();

            let metadata: SmallVec<[usize; 24]> = smallvec![
                num_els as usize,
                in_channels as usize,
                out_channels as usize,
                in_depth as usize,
                in_height as usize,
                in_width as usize,
                kernel_depth as usize,
                kernel_height as usize,
                kernel_width as usize,
                out_depth as usize,
                out_height as usize,
                out_width as usize,
                stride_d as usize,
                stride_h as usize,
                stride_w as usize,
                padding_d as usize,
                padding_h as usize,
                padding_w as usize,
                dilation_d as usize,
                dilation_h as usize,
                dilation_w as usize,
                input_layout.offset() as usize,
                weight_layout.offset() as usize,
            ];

            (metadata, output_shape)
        },
        _ => unreachable!(),
    };

    // Generate kernel name
    let kernel_name = format!("{}_{}", conv_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let mut output = CpuDevice::zeros(&output_shape, dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($input_data:expr, $weight_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let weight_ptr = $weight_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_conv(kernel, input_ptr, weight_ptr, out_ptr, &metadata)?;
        }};
    }

    match (input_storage, weight_storage, &mut output) {
        (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(weight), CpuStorage::F8E4M3(out)) => {
            call_kernel!(input, weight, out)
        },
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(weight), CpuStorage::F8E5M2(out)) => {
            call_kernel!(input, weight, out)
        },
        (CpuStorage::BF16(input), CpuStorage::BF16(weight), CpuStorage::BF16(out)) => {
            call_kernel!(input, weight, out)
        },
        (CpuStorage::F16(input), CpuStorage::F16(weight), CpuStorage::F16(out)) => {
            call_kernel!(input, weight, out)
        },
        (CpuStorage::F32(input), CpuStorage::F32(weight), CpuStorage::F32(out)) => {
            call_kernel!(input, weight, out)
        },
        #[cfg(feature = "f64")]
        (CpuStorage::F64(input), CpuStorage::F64(weight), CpuStorage::F64(out)) => {
            call_kernel!(input, weight, out)
        },
        _ => {
            return Err(HoduError::BackendError(
                "mismatched or unsupported storage types in call_conv".to_string(),
            ))
        },
    }

    Ok(output)
}

/// Execute convolution gradient weight operation
///
/// # Arguments
/// * `input_storage` - Input tensor storage (from forward pass)
/// * `input_layout` - Input tensor layout
/// * `grad_output_storage` - Gradient output tensor storage
/// * `grad_output_layout` - Gradient output tensor layout
/// * `weight_shape` - Shape of the weight tensor
/// * `stride` - Stride values
/// * `padding` - Padding values
/// * `dilation` - Dilation values
/// * `op` - The convolution gradient weight operation
///
/// # Returns
/// Output storage containing the gradient weights
#[allow(clippy::too_many_arguments)]
pub fn call_ops_conv_grad_weight(
    input_storage: &CpuStorage,
    input_layout: &Layout,
    grad_output_storage: &CpuStorage,
    grad_output_layout: &Layout,
    weight_shape: &Shape,
    stride: &[u32],
    padding: &[u32],
    dilation: &[u32],
    op: Op,
) -> HoduResult<CpuStorage> {
    // Validate op
    let conv_op = match op {
        Op::Conv(conv_op) => conv_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_conv_grad_weight expects Conv op".to_string(),
            ))
        },
    };

    // Validate dtypes match
    if input_storage.dtype() != grad_output_storage.dtype() {
        return Err(HoduError::DTypeMismatch {
            expected: input_storage.dtype(),
            got: grad_output_storage.dtype(),
        });
    }

    let input_shape = input_layout.shape();
    let grad_output_shape = grad_output_layout.shape();
    let dtype = input_storage.dtype();

    // Determine spatial dimensions based on op
    let spatial_dims = match conv_op {
        ConvOp::Conv1dGradWeight | ConvOp::ConvTranspose1dGradWeight => 1,
        ConvOp::Conv2dGradWeight | ConvOp::ConvTranspose2dGradWeight => 2,
        ConvOp::Conv3dGradWeight | ConvOp::ConvTranspose3dGradWeight => 3,
        _ => {
            return Err(HoduError::BackendError(
                "call_conv_grad_weight expects grad weight op".to_string(),
            ))
        },
    };

    let batch_size = input_shape.dims()[0];
    let _in_channels = input_shape.dims()[1];
    let out_channels = grad_output_shape.dims()[1];

    // Build metadata based on spatial dimensions
    // Note: Same layout as forward pass, but metadata[1] contains batch_size instead of in_channels
    let metadata: SmallVec<[usize; 24]> = match spatial_dims {
        1 => {
            let in_width = input_shape.dims()[2];
            let kernel_width = weight_shape.dims()[2];
            let out_width = grad_output_shape.dims()[2];
            let stride_w = stride[0];
            let padding_w = padding[0];
            let dilation_w = dilation[0];

            let num_els = weight_shape.size();

            smallvec![
                num_els as usize,
                batch_size as usize, // metadata[1] = batch_size (instead of in_channels in forward pass)
                out_channels as usize,
                in_width as usize,
                kernel_width as usize,
                out_width as usize,
                stride_w as usize,
                padding_w as usize,
                dilation_w as usize,
                input_layout.offset() as usize,
                grad_output_layout.offset() as usize,
            ]
        },
        2 => {
            let in_height = input_shape.dims()[2];
            let in_width = input_shape.dims()[3];
            let kernel_height = weight_shape.dims()[2];
            let kernel_width = weight_shape.dims()[3];
            let out_height = grad_output_shape.dims()[2];
            let out_width = grad_output_shape.dims()[3];
            let stride_h = stride[0];
            let stride_w = stride[1];
            let padding_h = padding[0];
            let padding_w = padding[1];
            let dilation_h = dilation[0];
            let dilation_w = dilation[1];

            let num_els = weight_shape.size();

            smallvec![
                num_els as usize,
                batch_size as usize, // metadata[1] = batch_size (instead of in_channels in forward pass)
                out_channels as usize,
                in_height as usize,
                in_width as usize,
                kernel_height as usize,
                kernel_width as usize,
                out_height as usize,
                out_width as usize,
                stride_h as usize,
                stride_w as usize,
                padding_h as usize,
                padding_w as usize,
                dilation_h as usize,
                dilation_w as usize,
                input_layout.offset() as usize,
                grad_output_layout.offset() as usize,
            ]
        },
        3 => {
            let in_depth = input_shape.dims()[2];
            let in_height = input_shape.dims()[3];
            let in_width = input_shape.dims()[4];
            let kernel_depth = weight_shape.dims()[2];
            let kernel_height = weight_shape.dims()[3];
            let kernel_width = weight_shape.dims()[4];
            let out_depth = grad_output_shape.dims()[2];
            let out_height = grad_output_shape.dims()[3];
            let out_width = grad_output_shape.dims()[4];
            let stride_d = stride[0];
            let stride_h = stride[1];
            let stride_w = stride[2];
            let padding_d = padding[0];
            let padding_h = padding[1];
            let padding_w = padding[2];
            let dilation_d = dilation[0];
            let dilation_h = dilation[1];
            let dilation_w = dilation[2];

            let num_els = weight_shape.size();

            smallvec![
                num_els as usize,
                batch_size as usize, // metadata[1] = batch_size (instead of in_channels in forward pass)
                out_channels as usize,
                in_depth as usize,
                in_height as usize,
                in_width as usize,
                kernel_depth as usize,
                kernel_height as usize,
                kernel_width as usize,
                out_depth as usize,
                out_height as usize,
                out_width as usize,
                stride_d as usize,
                stride_h as usize,
                stride_w as usize,
                padding_d as usize,
                padding_h as usize,
                padding_w as usize,
                dilation_d as usize,
                dilation_h as usize,
                dilation_w as usize,
                input_layout.offset() as usize,
                grad_output_layout.offset() as usize,
            ]
        },
        _ => unreachable!(),
    };

    // Generate kernel name
    let kernel_name = format!("{}_{}", conv_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage (gradient weights)
    let mut grad_weight = CpuDevice::zeros(weight_shape, dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($input_data:expr, $grad_output_data:expr, $grad_weight_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let grad_output_ptr = $grad_output_data.as_ptr() as *const c_void;
            let grad_weight_ptr = $grad_weight_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_conv_grad_weight(
                kernel,
                input_ptr,
                grad_output_ptr,
                grad_weight_ptr,
                &metadata,
            )?;
        }};
    }

    match (input_storage, grad_output_storage, &mut grad_weight) {
        (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(grad_out), CpuStorage::F8E4M3(grad_w)) => {
            call_kernel!(input, grad_out, grad_w)
        },
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(grad_out), CpuStorage::F8E5M2(grad_w)) => {
            call_kernel!(input, grad_out, grad_w)
        },
        (CpuStorage::BF16(input), CpuStorage::BF16(grad_out), CpuStorage::BF16(grad_w)) => {
            call_kernel!(input, grad_out, grad_w)
        },
        (CpuStorage::F16(input), CpuStorage::F16(grad_out), CpuStorage::F16(grad_w)) => {
            call_kernel!(input, grad_out, grad_w)
        },
        (CpuStorage::F32(input), CpuStorage::F32(grad_out), CpuStorage::F32(grad_w)) => {
            call_kernel!(input, grad_out, grad_w)
        },
        #[cfg(feature = "f64")]
        (CpuStorage::F64(input), CpuStorage::F64(grad_out), CpuStorage::F64(grad_w)) => {
            call_kernel!(input, grad_out, grad_w)
        },
        _ => {
            return Err(HoduError::BackendError(
                "mismatched or unsupported storage types in call_conv_grad_weight".to_string(),
            ))
        },
    }

    Ok(grad_weight)
}
