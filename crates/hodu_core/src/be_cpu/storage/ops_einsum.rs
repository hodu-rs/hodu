use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    einsum::ParsedEinsum,
    error::HoduResult,
    types::Layout,
};
use core::ffi::c_void;

pub fn call_ops_einsum(
    storage: &CpuStorage,
    inputs: &[&CpuStorage],
    input_layouts: &[&Layout],
    parsed: &ParsedEinsum,
) -> HoduResult<CpuStorage> {
    let dtype = storage.dtype();
    let output_shape = parsed.compute_output_shape();
    let num_els = output_shape.size();
    let output_layout = Layout::from_shape(&output_shape);

    let kernel_name = format!("hodu_cpu_einsum_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    let input_layout_refs: Vec<&Layout> = input_layouts.to_vec();
    let metadata = crate::op_metadatas::einsum_metadata(parsed, &input_layout_refs, &output_layout);

    let mut output = CpuDevice::allocate(num_els, dtype)?;

    macro_rules! call_einsum {
        ($variant:ident) => {{
            let input_ptrs: Vec<*const c_void> = std::iter::once(storage)
                .chain(inputs.iter().cloned())
                .map(|s| match s {
                    CpuStorage::$variant(data) => data.as_ptr() as *const c_void,
                    _ => unreachable!(),
                })
                .collect();

            let out_ptr = match &mut output {
                CpuStorage::$variant(data) => data.as_mut_ptr() as *mut c_void,
                _ => unreachable!(),
            };

            hodu_cpu_kernels::call_ops_einsum(kernel, &input_ptrs, out_ptr, &metadata)?;
        }};
    }

    match storage {
        CpuStorage::F8E4M3(_) => call_einsum!(F8E4M3),
        #[cfg(feature = "f8e5m2")]
        CpuStorage::F8E5M2(_) => call_einsum!(F8E5M2),
        CpuStorage::BF16(_) => call_einsum!(BF16),
        CpuStorage::F16(_) => call_einsum!(F16),
        CpuStorage::F32(_) => call_einsum!(F32),
        #[cfg(feature = "f64")]
        CpuStorage::F64(_) => call_einsum!(F64),
        CpuStorage::U8(_) => call_einsum!(U8),
        #[cfg(feature = "u16")]
        CpuStorage::U16(_) => call_einsum!(U16),
        CpuStorage::U32(_) => call_einsum!(U32),
        #[cfg(feature = "u64")]
        CpuStorage::U64(_) => call_einsum!(U64),
        CpuStorage::I8(_) => call_einsum!(I8),
        #[cfg(feature = "i16")]
        CpuStorage::I16(_) => call_einsum!(I16),
        CpuStorage::I32(_) => call_einsum!(I32),
        #[cfg(feature = "i64")]
        CpuStorage::I64(_) => call_einsum!(I64),
        _ => unreachable!("unsupported dtype for einsum"),
    }

    Ok(output)
}
