use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    ops::{Op, PaddingOp},
    scalar::Scalar,
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};
use std::sync::Arc;

pub fn call_ops_pad(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    pad_before: &[usize],
    pad_after: &[usize],
    pad_value: Scalar,
    op: Op,
) -> HoduResult<CudaStorage> {
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

    let output_size: usize = output_shape_vec.iter().product();
    let metadata = crate::op_metadatas::padding_metadata(input_layout, pad_before, &output_shape_vec);

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    let kernel_name = format!("hodu_cuda_{}_{}", padding_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let device_id = input_storage.device_id;
    let device_arc = Arc::clone(&input_storage.device);

    match padding_op {
        PaddingOp::PadConstant => {
            macro_rules! call_pad_constant {
                ($input:expr, $ty:ty, $pv:expr) => {{
                    let mut output: CudaSlice<$ty> = device.new_buffer(output_size)?;
                    let mut pv_buf: CudaSlice<$ty> = device.new_buffer(1)?;
                    device.context().stream().memcpy_stod(&[$pv], &mut pv_buf)?;
                    kernels::call_ops_pad_constant(
                        kernel,
                        device.kernels(),
                        device.context(),
                        $input,
                        &mut output,
                        &pv_buf,
                        &metadata,
                    )?;
                    output
                }};
            }

            match &input_storage.data {
                CudaStorageData::BOOL(input) => {
                    let pv = pad_value.to_bool();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::BOOL(call_pad_constant!(input, bool, pv)),
                    ))
                },
                CudaStorageData::F8E4M3(input) => {
                    let pv = pad_value.to_f8e4m3();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::F8E4M3(call_pad_constant!(input, float8::F8E4M3, pv)),
                    ))
                },
                #[cfg(feature = "f8e5m2")]
                CudaStorageData::F8E5M2(input) => {
                    let pv = pad_value.to_f8e5m2();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::F8E5M2(call_pad_constant!(input, float8::F8E5M2, pv)),
                    ))
                },
                CudaStorageData::BF16(input) => {
                    let pv = pad_value.to_bf16();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::BF16(call_pad_constant!(input, half::bf16, pv)),
                    ))
                },
                CudaStorageData::F16(input) => {
                    let pv = pad_value.to_f16();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::F16(call_pad_constant!(input, half::f16, pv)),
                    ))
                },
                CudaStorageData::F32(input) => {
                    let pv = pad_value.to_f32();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::F32(call_pad_constant!(input, f32, pv)),
                    ))
                },
                #[cfg(feature = "f64")]
                CudaStorageData::F64(input) => {
                    let pv = pad_value.to_f64();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::F64(call_pad_constant!(input, f64, pv)),
                    ))
                },
                CudaStorageData::U8(input) => {
                    let pv = pad_value.to_u8();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::U8(call_pad_constant!(input, u8, pv)),
                    ))
                },
                #[cfg(feature = "u16")]
                CudaStorageData::U16(input) => {
                    let pv = pad_value.to_u16();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::U16(call_pad_constant!(input, u16, pv)),
                    ))
                },
                CudaStorageData::U32(input) => {
                    let pv = pad_value.to_u32();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::U32(call_pad_constant!(input, u32, pv)),
                    ))
                },
                #[cfg(feature = "u64")]
                CudaStorageData::U64(input) => {
                    let pv = pad_value.to_u64();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::U64(call_pad_constant!(input, u64, pv)),
                    ))
                },
                CudaStorageData::I8(input) => {
                    let pv = pad_value.to_i8();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::I8(call_pad_constant!(input, i8, pv)),
                    ))
                },
                #[cfg(feature = "i16")]
                CudaStorageData::I16(input) => {
                    let pv = pad_value.to_i16();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::I16(call_pad_constant!(input, i16, pv)),
                    ))
                },
                CudaStorageData::I32(input) => {
                    let pv = pad_value.to_i32();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::I32(call_pad_constant!(input, i32, pv)),
                    ))
                },
                #[cfg(feature = "i64")]
                CudaStorageData::I64(input) => {
                    let pv = pad_value.to_i64();
                    Ok(CudaStorage::new(
                        device_id,
                        device_arc,
                        CudaStorageData::I64(call_pad_constant!(input, i64, pv)),
                    ))
                },
            }
        },
        PaddingOp::PadReflect | PaddingOp::PadReplicate | PaddingOp::PadCircular => {
            macro_rules! call_pad_other {
                ($input:expr, $ty:ty, $call_fn:ident) => {{
                    let mut output: CudaSlice<$ty> = device.new_buffer(output_size)?;
                    kernels::$call_fn(
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

            macro_rules! dispatch_pad_other {
                ($call_fn:ident) => {
                    match &input_storage.data {
                        CudaStorageData::BOOL(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::BOOL(call_pad_other!(input, bool, $call_fn)),
                        )),
                        CudaStorageData::F8E4M3(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::F8E4M3(call_pad_other!(input, float8::F8E4M3, $call_fn)),
                        )),
                        #[cfg(feature = "f8e5m2")]
                        CudaStorageData::F8E5M2(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::F8E5M2(call_pad_other!(input, float8::F8E5M2, $call_fn)),
                        )),
                        CudaStorageData::BF16(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::BF16(call_pad_other!(input, half::bf16, $call_fn)),
                        )),
                        CudaStorageData::F16(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::F16(call_pad_other!(input, half::f16, $call_fn)),
                        )),
                        CudaStorageData::F32(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::F32(call_pad_other!(input, f32, $call_fn)),
                        )),
                        #[cfg(feature = "f64")]
                        CudaStorageData::F64(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::F64(call_pad_other!(input, f64, $call_fn)),
                        )),
                        CudaStorageData::U8(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::U8(call_pad_other!(input, u8, $call_fn)),
                        )),
                        #[cfg(feature = "u16")]
                        CudaStorageData::U16(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::U16(call_pad_other!(input, u16, $call_fn)),
                        )),
                        CudaStorageData::U32(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::U32(call_pad_other!(input, u32, $call_fn)),
                        )),
                        #[cfg(feature = "u64")]
                        CudaStorageData::U64(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::U64(call_pad_other!(input, u64, $call_fn)),
                        )),
                        CudaStorageData::I8(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::I8(call_pad_other!(input, i8, $call_fn)),
                        )),
                        #[cfg(feature = "i16")]
                        CudaStorageData::I16(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::I16(call_pad_other!(input, i16, $call_fn)),
                        )),
                        CudaStorageData::I32(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::I32(call_pad_other!(input, i32, $call_fn)),
                        )),
                        #[cfg(feature = "i64")]
                        CudaStorageData::I64(input) => Ok(CudaStorage::new(
                            device_id,
                            device_arc,
                            CudaStorageData::I64(call_pad_other!(input, i64, $call_fn)),
                        )),
                    }
                };
            }

            match padding_op {
                PaddingOp::PadReflect => dispatch_pad_other!(call_ops_pad_reflect),
                PaddingOp::PadReplicate => dispatch_pad_other!(call_ops_pad_replicate),
                PaddingOp::PadCircular => dispatch_pad_other!(call_ops_pad_circular),
                _ => unreachable!(),
            }
        },
    }
}
