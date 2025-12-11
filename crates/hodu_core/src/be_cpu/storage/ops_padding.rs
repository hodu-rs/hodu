use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::{HoduError, HoduResult},
    ops::{Op, PaddingOp},
    scalar::Scalar,
    types::{Layout, Shape},
};
use core::ffi::c_void;

pub fn call_ops_pad(
    storage: &CpuStorage,
    layout: &Layout,
    pad_before: &[usize],
    pad_after: &[usize],
    pad_value: Scalar,
    op: Op,
) -> HoduResult<CpuStorage> {
    let padding_op = match op {
        Op::Padding(padding_op) => padding_op,
        _ => return Err(HoduError::BackendError("call_ops_pad expects Padding op".to_string())),
    };

    let kernel_prefix = format!("hodu_cpu_{}", padding_op);

    let input_shape = layout.shape();
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
    let metadata = crate::op_metadatas::padding_metadata(layout, pad_before, &output_shape_vec);

    let dtype = storage.dtype();
    let kernel_name = format!("{}_{}", kernel_prefix, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    let mut output = CpuDevice::allocate(output_shape.size(), dtype)?;

    macro_rules! call_pad_constant {
        ($input_data:expr, $out_data:expr, $pv:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;
            let pv_ptr = &$pv as *const _ as *const c_void;

            hodu_cpu_kernels::call_ops_pad_constant(kernel, input_ptr, out_ptr, pv_ptr, &metadata)?;
        }};
    }

    macro_rules! call_pad_other {
        ($input_data:expr, $out_data:expr, $call_fn:ident) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::$call_fn(kernel, input_ptr, out_ptr, &metadata)?;
        }};
    }

    match padding_op {
        PaddingOp::PadConstant => match (storage, &mut output) {
            (CpuStorage::BOOL(input), CpuStorage::BOOL(out)) => {
                let pv = pad_value.to_bool();
                call_pad_constant!(input, out, pv);
            },
            (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(out)) => {
                let pv = pad_value.to_f8e4m3();
                call_pad_constant!(input, out, pv);
            },
            #[cfg(feature = "f8e5m2")]
            (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(out)) => {
                let pv = pad_value.to_f8e5m2();
                call_pad_constant!(input, out, pv);
            },
            (CpuStorage::BF16(input), CpuStorage::BF16(out)) => {
                let pv = pad_value.to_bf16();
                call_pad_constant!(input, out, pv);
            },
            (CpuStorage::F16(input), CpuStorage::F16(out)) => {
                let pv = pad_value.to_f16();
                call_pad_constant!(input, out, pv);
            },
            (CpuStorage::F32(input), CpuStorage::F32(out)) => {
                let pv = pad_value.to_f32();
                call_pad_constant!(input, out, pv);
            },
            #[cfg(feature = "f64")]
            (CpuStorage::F64(input), CpuStorage::F64(out)) => {
                let pv = pad_value.to_f64();
                call_pad_constant!(input, out, pv);
            },
            (CpuStorage::U8(input), CpuStorage::U8(out)) => {
                let pv = pad_value.to_u8();
                call_pad_constant!(input, out, pv);
            },
            #[cfg(feature = "u16")]
            (CpuStorage::U16(input), CpuStorage::U16(out)) => {
                let pv = pad_value.to_u16();
                call_pad_constant!(input, out, pv);
            },
            (CpuStorage::U32(input), CpuStorage::U32(out)) => {
                let pv = pad_value.to_u32();
                call_pad_constant!(input, out, pv);
            },
            #[cfg(feature = "u64")]
            (CpuStorage::U64(input), CpuStorage::U64(out)) => {
                let pv = pad_value.to_u64();
                call_pad_constant!(input, out, pv);
            },
            (CpuStorage::I8(input), CpuStorage::I8(out)) => {
                let pv = pad_value.to_i8();
                call_pad_constant!(input, out, pv);
            },
            #[cfg(feature = "i16")]
            (CpuStorage::I16(input), CpuStorage::I16(out)) => {
                let pv = pad_value.to_i16();
                call_pad_constant!(input, out, pv);
            },
            (CpuStorage::I32(input), CpuStorage::I32(out)) => {
                let pv = pad_value.to_i32();
                call_pad_constant!(input, out, pv);
            },
            #[cfg(feature = "i64")]
            (CpuStorage::I64(input), CpuStorage::I64(out)) => {
                let pv = pad_value.to_i64();
                call_pad_constant!(input, out, pv);
            },
            _ => {
                return Err(HoduError::BackendError(
                    "mismatched storage types in call_ops_pad".to_string(),
                ))
            },
        },
        PaddingOp::PadReflect => match (storage, &mut output) {
            (CpuStorage::BOOL(input), CpuStorage::BOOL(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(out)) => {
                call_pad_other!(input, out, call_ops_pad_reflect)
            },
            #[cfg(feature = "f8e5m2")]
            (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(out)) => {
                call_pad_other!(input, out, call_ops_pad_reflect)
            },
            (CpuStorage::BF16(input), CpuStorage::BF16(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            (CpuStorage::F16(input), CpuStorage::F16(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            (CpuStorage::F32(input), CpuStorage::F32(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            #[cfg(feature = "f64")]
            (CpuStorage::F64(input), CpuStorage::F64(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            (CpuStorage::U8(input), CpuStorage::U8(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            #[cfg(feature = "u16")]
            (CpuStorage::U16(input), CpuStorage::U16(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            (CpuStorage::U32(input), CpuStorage::U32(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            #[cfg(feature = "u64")]
            (CpuStorage::U64(input), CpuStorage::U64(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            (CpuStorage::I8(input), CpuStorage::I8(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            #[cfg(feature = "i16")]
            (CpuStorage::I16(input), CpuStorage::I16(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            (CpuStorage::I32(input), CpuStorage::I32(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            #[cfg(feature = "i64")]
            (CpuStorage::I64(input), CpuStorage::I64(out)) => call_pad_other!(input, out, call_ops_pad_reflect),
            _ => {
                return Err(HoduError::BackendError(
                    "mismatched storage types in call_ops_pad".to_string(),
                ))
            },
        },
        PaddingOp::PadReplicate => match (storage, &mut output) {
            (CpuStorage::BOOL(input), CpuStorage::BOOL(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(out)) => {
                call_pad_other!(input, out, call_ops_pad_replicate)
            },
            #[cfg(feature = "f8e5m2")]
            (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(out)) => {
                call_pad_other!(input, out, call_ops_pad_replicate)
            },
            (CpuStorage::BF16(input), CpuStorage::BF16(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            (CpuStorage::F16(input), CpuStorage::F16(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            (CpuStorage::F32(input), CpuStorage::F32(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            #[cfg(feature = "f64")]
            (CpuStorage::F64(input), CpuStorage::F64(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            (CpuStorage::U8(input), CpuStorage::U8(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            #[cfg(feature = "u16")]
            (CpuStorage::U16(input), CpuStorage::U16(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            (CpuStorage::U32(input), CpuStorage::U32(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            #[cfg(feature = "u64")]
            (CpuStorage::U64(input), CpuStorage::U64(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            (CpuStorage::I8(input), CpuStorage::I8(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            #[cfg(feature = "i16")]
            (CpuStorage::I16(input), CpuStorage::I16(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            (CpuStorage::I32(input), CpuStorage::I32(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            #[cfg(feature = "i64")]
            (CpuStorage::I64(input), CpuStorage::I64(out)) => call_pad_other!(input, out, call_ops_pad_replicate),
            _ => {
                return Err(HoduError::BackendError(
                    "mismatched storage types in call_ops_pad".to_string(),
                ))
            },
        },
        PaddingOp::PadCircular => match (storage, &mut output) {
            (CpuStorage::BOOL(input), CpuStorage::BOOL(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(out)) => {
                call_pad_other!(input, out, call_ops_pad_circular)
            },
            #[cfg(feature = "f8e5m2")]
            (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(out)) => {
                call_pad_other!(input, out, call_ops_pad_circular)
            },
            (CpuStorage::BF16(input), CpuStorage::BF16(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            (CpuStorage::F16(input), CpuStorage::F16(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            (CpuStorage::F32(input), CpuStorage::F32(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            #[cfg(feature = "f64")]
            (CpuStorage::F64(input), CpuStorage::F64(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            (CpuStorage::U8(input), CpuStorage::U8(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            #[cfg(feature = "u16")]
            (CpuStorage::U16(input), CpuStorage::U16(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            (CpuStorage::U32(input), CpuStorage::U32(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            #[cfg(feature = "u64")]
            (CpuStorage::U64(input), CpuStorage::U64(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            (CpuStorage::I8(input), CpuStorage::I8(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            #[cfg(feature = "i16")]
            (CpuStorage::I16(input), CpuStorage::I16(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            (CpuStorage::I32(input), CpuStorage::I32(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            #[cfg(feature = "i64")]
            (CpuStorage::I64(input), CpuStorage::I64(out)) => call_pad_other!(input, out, call_ops_pad_circular),
            _ => {
                return Err(HoduError::BackendError(
                    "mismatched storage types in call_ops_pad".to_string(),
                ))
            },
        },
    }

    Ok(output)
}
