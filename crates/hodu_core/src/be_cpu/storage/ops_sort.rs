use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::HoduResult,
    types::{DType, Layout},
};
use core::ffi::c_void;

pub fn call_topk(
    storage: &CpuStorage,
    layout: &Layout,
    k: usize,
    last_dim_size: usize,
    outer_size: usize,
    largest: bool,
    sorted: bool,
) -> HoduResult<(CpuStorage, CpuStorage)> {
    let dtype = storage.dtype();
    let output_size = outer_size * k;

    let kernel_name = format!("hodu_cpu_topk_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    let metadata = crate::op_metadatas::topk_metadata(layout, k, last_dim_size, outer_size, largest, sorted);

    let mut values_output = CpuDevice::allocate(output_size, dtype)?;
    let mut indices_output = CpuDevice::allocate(output_size, DType::I32)?;

    macro_rules! call_topk {
        ($input_data:expr, $values_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let values_ptr = $values_data.as_mut_ptr() as *mut c_void;
            let indices_ptr = match &mut indices_output {
                CpuStorage::I32(out) => out.as_mut_ptr() as *mut c_void,
                _ => unreachable!("indices should always be I32"),
            };
            hodu_cpu_kernels::call_topk(kernel, input_ptr, values_ptr, indices_ptr, &metadata)?;
        }};
    }

    match (storage, &mut values_output) {
        (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(out)) => call_topk!(input, out),
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(out)) => call_topk!(input, out),
        (CpuStorage::BF16(input), CpuStorage::BF16(out)) => call_topk!(input, out),
        (CpuStorage::F16(input), CpuStorage::F16(out)) => call_topk!(input, out),
        (CpuStorage::F32(input), CpuStorage::F32(out)) => call_topk!(input, out),
        #[cfg(feature = "f64")]
        (CpuStorage::F64(input), CpuStorage::F64(out)) => call_topk!(input, out),
        (CpuStorage::U8(input), CpuStorage::U8(out)) => call_topk!(input, out),
        #[cfg(feature = "u16")]
        (CpuStorage::U16(input), CpuStorage::U16(out)) => call_topk!(input, out),
        (CpuStorage::U32(input), CpuStorage::U32(out)) => call_topk!(input, out),
        #[cfg(feature = "u64")]
        (CpuStorage::U64(input), CpuStorage::U64(out)) => call_topk!(input, out),
        (CpuStorage::I8(input), CpuStorage::I8(out)) => call_topk!(input, out),
        #[cfg(feature = "i16")]
        (CpuStorage::I16(input), CpuStorage::I16(out)) => call_topk!(input, out),
        (CpuStorage::I32(input), CpuStorage::I32(out)) => call_topk!(input, out),
        #[cfg(feature = "i64")]
        (CpuStorage::I64(input), CpuStorage::I64(out)) => call_topk!(input, out),
        _ => unreachable!("dtype mismatch or unsupported dtype for topk"),
    }

    Ok((values_output, indices_output))
}
