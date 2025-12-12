use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    einsum::ParsedEinsum,
    error::HoduResult,
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};
use std::sync::Arc;

pub fn call_ops_einsum(
    storage: &CudaStorage,
    inputs: &[&CudaStorage],
    input_layouts: &[&Layout],
    parsed: &ParsedEinsum,
) -> HoduResult<CudaStorage> {
    let dtype = storage.dtype();
    let device = storage.get_device();

    let output_shape = parsed.compute_output_shape();
    let num_els = output_shape.size();
    let output_layout = Layout::from_shape(&output_shape);

    let input_layout_refs: Vec<&Layout> = input_layouts.to_vec();
    let metadata = crate::op_metadatas::einsum_metadata(parsed, &input_layout_refs, &output_layout);

    let kernel_name = format!("hodu_cuda_einsum_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let device_id = storage.device_id;
    let device_arc = Arc::clone(&storage.device);

    macro_rules! call_einsum {
        ($variant:ident, $ty:ty) => {{
            let input_slices: Vec<&CudaSlice<$ty>> = std::iter::once(storage)
                .chain(inputs.iter().cloned())
                .map(|s| match &s.data {
                    CudaStorageData::$variant(data) => data,
                    _ => unreachable!(),
                })
                .collect();

            let mut output: CudaSlice<$ty> = device.new_buffer(num_els)?;
            kernels::call_ops_einsum(
                kernel,
                device.kernels(),
                device.context(),
                &input_slices,
                &mut output,
                &metadata,
            )?;
            output
        }};
    }

    match &storage.data {
        CudaStorageData::F8E4M3(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F8E4M3(call_einsum!(F8E4M3, float8::F8E4M3)),
        )),
        #[cfg(feature = "f8e5m2")]
        CudaStorageData::F8E5M2(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F8E5M2(call_einsum!(F8E5M2, float8::F8E5M2)),
        )),
        CudaStorageData::BF16(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::BF16(call_einsum!(BF16, half::bf16)),
        )),
        CudaStorageData::F16(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F16(call_einsum!(F16, half::f16)),
        )),
        CudaStorageData::F32(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F32(call_einsum!(F32, f32)),
        )),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F64(call_einsum!(F64, f64)),
        )),
        CudaStorageData::U8(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::U8(call_einsum!(U8, u8)),
        )),
        #[cfg(feature = "u16")]
        CudaStorageData::U16(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::U16(call_einsum!(U16, u16)),
        )),
        CudaStorageData::U32(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::U32(call_einsum!(U32, u32)),
        )),
        #[cfg(feature = "u64")]
        CudaStorageData::U64(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::U64(call_einsum!(U64, u64)),
        )),
        CudaStorageData::I8(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::I8(call_einsum!(I8, i8)),
        )),
        #[cfg(feature = "i16")]
        CudaStorageData::I16(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::I16(call_einsum!(I16, i16)),
        )),
        CudaStorageData::I32(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::I32(call_einsum!(I32, i32)),
        )),
        #[cfg(feature = "i64")]
        CudaStorageData::I64(_) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::I64(call_einsum!(I64, i64)),
        )),
        _ => unreachable!("einsum does not support bool"),
    }
}
