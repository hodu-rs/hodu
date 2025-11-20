use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    compat::*,
    error::{HoduError, HoduResult},
    ops::Op,
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};

pub fn call_ops_reduce_window(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    window_shape: &[usize],
    strides: &[usize],
    padding: &[usize],
    op: Op,
) -> HoduResult<CudaStorage> {
    let windowing_op = match op {
        Op::Windowing(windowing_op) => windowing_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_reduce_window expects windowing op".to_string(),
            ))
        },
    };

    let input_shape = input_layout.shape();
    let ndim = input_shape.ndim();

    // Validate window_shape, strides, padding dimensions
    if window_shape.len() != ndim {
        return Err(HoduError::BackendError(format!(
            "window_shape length {} does not match tensor ndim {}",
            window_shape.len(),
            ndim
        )));
    }

    if strides.len() != ndim {
        return Err(HoduError::BackendError(format!(
            "strides length {} does not match tensor ndim {}",
            strides.len(),
            ndim
        )));
    }

    if padding.len() != ndim * 2 {
        return Err(HoduError::BackendError(format!(
            "padding length {} does not match tensor ndim * 2 ({})",
            padding.len(),
            ndim * 2
        )));
    }

    // Compute output shape
    let mut output_shape_vec = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let in_size = input_shape.dims()[i];
        let window_size = window_shape[i];
        let stride = strides[i];
        let pad_before = padding[i * 2];
        let pad_after = padding[i * 2 + 1];

        // Output size formula: floor((in_size + pad_before + pad_after - window_size) / stride) + 1
        let padded_size = in_size + pad_before + pad_after;
        if padded_size < window_size {
            return Err(HoduError::BackendError(format!(
                "padded size {} is less than window size {} in dimension {}",
                padded_size, window_size, i
            )));
        }

        let out_size = (padded_size - window_size) / stride + 1;
        output_shape_vec.push(out_size);
    }

    let output_size: usize = output_shape_vec.iter().product();

    // Build metadata array
    // Layout: output_size, num_dims, input_shape, input_strides, offset,
    //         window_shape, strides, padding, output_shape
    let mut metadata = Vec::with_capacity(3 + ndim * 7);

    metadata.push(output_size);
    metadata.push(ndim);

    // Add input shape
    for &dim in input_shape.dims() {
        metadata.push(dim);
    }

    // Add input strides
    for &stride in input_layout.strides() {
        metadata.push(stride);
    }

    // Add offset
    metadata.push(input_layout.offset());

    // Add window shape
    for &w in window_shape {
        metadata.push(w);
    }

    // Add strides
    for &s in strides {
        metadata.push(s);
    }

    // Add padding (already flattened)
    for &p in padding {
        metadata.push(p);
    }

    // Add output shape
    for &d in &output_shape_vec {
        metadata.push(d);
    }

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    let kernel_name = format!("{}_{}", windowing_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_reduce_window {
        ($input:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(output_size as usize)?;
            kernels::call_ops_reduce_window(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                &mut output,
                &metadata,
            )?;
            output
        }};
    }

    let device_id = input_storage.device_id;
    let device_arc = Arc::clone(&input_storage.device);

    match &input_storage.data {
        CudaStorageData::F32(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F32(call_reduce_window!(input, f32)),
        )),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F64(call_reduce_window!(input, f64)),
        )),
        CudaStorageData::F16(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F16(call_reduce_window!(input, half::f16)),
        )),
        _ => Err(HoduError::UnsupportedDTypeForOp {
            dtype: input_storage.dtype(),
            op,
        }),
    }
}
