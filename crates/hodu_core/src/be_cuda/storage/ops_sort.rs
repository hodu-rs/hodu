use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    op_metadatas,
    ops::{Op, SortOp},
    types::{DType, Layout},
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};
use std::sync::Arc;

pub fn call_topk(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    k: usize,
    last_dim_size: usize,
    outer_size: usize,
    largest: bool,
    sorted: bool,
) -> HoduResult<(CudaStorage, CudaStorage)> {
    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    let output_size = k * outer_size;
    let metadata = op_metadatas::topk_metadata(input_layout, k, last_dim_size, outer_size, largest, sorted);

    let kernel_name = format!("hodu_cuda_topk_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let device_id = input_storage.device_id;
    let device_arc = Arc::clone(&input_storage.device);

    macro_rules! call_topk {
        ($input:expr, $ty:ty, $data_variant:ident) => {{
            let mut values: CudaSlice<$ty> = device.new_buffer(output_size)?;
            let mut indices: CudaSlice<i32> = device.new_buffer(output_size)?;
            kernels::call_topk(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                &mut values,
                &mut indices,
                &metadata,
            )?;
            let values_storage = CudaStorage::new(
                device_id,
                Arc::clone(&device_arc),
                CudaStorageData::$data_variant(values),
            );
            let indices_storage = CudaStorage::new(device_id, device_arc, CudaStorageData::I32(indices));
            (values_storage, indices_storage)
        }};
    }

    match &input_storage.data {
        CudaStorageData::F8E4M3(input) => Ok(call_topk!(input, float8::F8E4M3, F8E4M3)),
        #[cfg(feature = "f8e5m2")]
        CudaStorageData::F8E5M2(input) => Ok(call_topk!(input, float8::F8E5M2, F8E5M2)),
        CudaStorageData::BF16(input) => Ok(call_topk!(input, half::bf16, BF16)),
        CudaStorageData::F16(input) => Ok(call_topk!(input, half::f16, F16)),
        CudaStorageData::F32(input) => Ok(call_topk!(input, f32, F32)),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => Ok(call_topk!(input, f64, F64)),
        CudaStorageData::U8(input) => Ok(call_topk!(input, u8, U8)),
        #[cfg(feature = "u16")]
        CudaStorageData::U16(input) => Ok(call_topk!(input, u16, U16)),
        CudaStorageData::U32(input) => Ok(call_topk!(input, u32, U32)),
        #[cfg(feature = "u64")]
        CudaStorageData::U64(input) => Ok(call_topk!(input, u64, U64)),
        CudaStorageData::I8(input) => Ok(call_topk!(input, i8, I8)),
        #[cfg(feature = "i16")]
        CudaStorageData::I16(input) => Ok(call_topk!(input, i16, I16)),
        CudaStorageData::I32(input) => Ok(call_topk!(input, i32, I32)),
        #[cfg(feature = "i64")]
        CudaStorageData::I64(input) => Ok(call_topk!(input, i64, I64)),
        CudaStorageData::BOOL(_) => Err(HoduError::UnsupportedDTypeForOp {
            dtype: DType::BOOL,
            op: Op::Sort(SortOp::TopK),
        }),
    }
}
