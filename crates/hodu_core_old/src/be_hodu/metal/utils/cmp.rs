use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    scalar::Scalar,
    types::{dtype::DType, layout::Layout},
};

pub fn cmp_map(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    kernel_name: &str,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::utils::BufferOffset;

    let dtype = lhs_storage.get_dtype();
    let device = lhs_storage.get_hodu_device();
    let lhs_shape = lhs_layout.get_shape();
    let lhs_strides = lhs_layout.get_strides();
    let lhs_offset = lhs_layout.get_offset();
    let rhs_shape = rhs_layout.get_shape();
    let rhs_strides = rhs_layout.get_strides();
    let rhs_offset = rhs_layout.get_offset();
    let lhs_ndim = lhs_shape.len();
    let rhs_ndim = rhs_shape.len();

    // Calculate output shape with broadcasting
    let ndim = lhs_ndim.max(rhs_ndim);
    let mut output_shape = vec![1; ndim];

    for i in 0..ndim {
        let lhs_dim = if i < lhs_ndim { lhs_shape[lhs_ndim - 1 - i] } else { 1 };
        let rhs_dim = if i < rhs_ndim { rhs_shape[rhs_ndim - 1 - i] } else { 1 };

        if lhs_dim != 1 && rhs_dim != 1 && lhs_dim != rhs_dim {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: format!("comparison '{}' - incompatible dimensions", kernel_name),
            });
        }
        output_shape[ndim - 1 - i] = lhs_dim.max(rhs_dim);
    }

    let output_size: usize = output_shape.iter().product();
    let output_dtype = DType::BOOL;
    let output = device.new_buffer(output_size, output_dtype, kernel_name)?;
    let command_buffer = device.command_buffer()?;

    let lhs = BufferOffset {
        buffer: lhs_storage.buffer(),
        offset_in_bytes: lhs_offset * dtype.get_size_in_bytes(),
    };
    let rhs = BufferOffset {
        buffer: rhs_storage.buffer(),
        offset_in_bytes: rhs_offset * dtype.get_size_in_bytes(),
    };

    macro_rules! dispatch_cmp {
        ($op:ident, $dtype:ident) => {{
            hodu_metal_kernels::kernels::call_binary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::$op::$dtype,
                &output_shape,
                lhs,
                lhs_strides,
                lhs_offset,
                rhs,
                rhs_strides,
                rhs_offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?
        }};
    }

    match kernel_name {
        "eq" => match dtype {
            DType::BF16 => dispatch_cmp!(eq, BF16),
            DType::F16 => dispatch_cmp!(eq, F16),
            DType::F32 => dispatch_cmp!(eq, F32),
            #[cfg(feature = "u8")]
            DType::U8 => dispatch_cmp!(eq, U8),
            DType::U16 => dispatch_cmp!(eq, U16),
            #[cfg(feature = "u32")]
            DType::U32 => dispatch_cmp!(eq, U32),
            #[cfg(feature = "u64")]
            DType::U64 => dispatch_cmp!(eq, U64),
            DType::I8 => dispatch_cmp!(eq, I8),
            #[cfg(feature = "i16")]
            DType::I16 => dispatch_cmp!(eq, I16),
            DType::I32 => dispatch_cmp!(eq, I32),
            #[cfg(feature = "i64")]
            DType::I64 => dispatch_cmp!(eq, I64),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "eq".to_string(),
                })
            },
        },
        "ne" => match dtype {
            DType::BF16 => dispatch_cmp!(ne, BF16),
            DType::F16 => dispatch_cmp!(ne, F16),
            DType::F32 => dispatch_cmp!(ne, F32),
            #[cfg(feature = "u8")]
            DType::U8 => dispatch_cmp!(ne, U8),
            DType::U16 => dispatch_cmp!(ne, U16),
            #[cfg(feature = "u32")]
            DType::U32 => dispatch_cmp!(ne, U32),
            #[cfg(feature = "u64")]
            DType::U64 => dispatch_cmp!(ne, U64),
            DType::I8 => dispatch_cmp!(ne, I8),
            #[cfg(feature = "i16")]
            DType::I16 => dispatch_cmp!(ne, I16),
            DType::I32 => dispatch_cmp!(ne, I32),
            #[cfg(feature = "i64")]
            DType::I64 => dispatch_cmp!(ne, I64),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "ne".to_string(),
                })
            },
        },
        "lt" => match dtype {
            DType::BF16 => dispatch_cmp!(lt, BF16),
            DType::F16 => dispatch_cmp!(lt, F16),
            DType::F32 => dispatch_cmp!(lt, F32),
            #[cfg(feature = "u8")]
            DType::U8 => dispatch_cmp!(lt, U8),
            DType::U16 => dispatch_cmp!(lt, U16),
            #[cfg(feature = "u32")]
            DType::U32 => dispatch_cmp!(lt, U32),
            #[cfg(feature = "u64")]
            DType::U64 => dispatch_cmp!(lt, U64),
            DType::I8 => dispatch_cmp!(lt, I8),
            #[cfg(feature = "i16")]
            DType::I16 => dispatch_cmp!(lt, I16),
            DType::I32 => dispatch_cmp!(lt, I32),
            #[cfg(feature = "i64")]
            DType::I64 => dispatch_cmp!(lt, I64),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "lt".to_string(),
                })
            },
        },
        "le" => match dtype {
            DType::BF16 => dispatch_cmp!(le, BF16),
            DType::F16 => dispatch_cmp!(le, F16),
            DType::F32 => dispatch_cmp!(le, F32),
            #[cfg(feature = "u8")]
            DType::U8 => dispatch_cmp!(le, U8),
            DType::U16 => dispatch_cmp!(le, U16),
            #[cfg(feature = "u32")]
            DType::U32 => dispatch_cmp!(le, U32),
            #[cfg(feature = "u64")]
            DType::U64 => dispatch_cmp!(le, U64),
            DType::I8 => dispatch_cmp!(le, I8),
            #[cfg(feature = "i16")]
            DType::I16 => dispatch_cmp!(le, I16),
            DType::I32 => dispatch_cmp!(le, I32),
            #[cfg(feature = "i64")]
            DType::I64 => dispatch_cmp!(le, I64),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "le".to_string(),
                })
            },
        },
        "gt" => match dtype {
            DType::BF16 => dispatch_cmp!(gt, BF16),
            DType::F16 => dispatch_cmp!(gt, F16),
            DType::F32 => dispatch_cmp!(gt, F32),
            #[cfg(feature = "u8")]
            DType::U8 => dispatch_cmp!(gt, U8),
            DType::U16 => dispatch_cmp!(gt, U16),
            #[cfg(feature = "u32")]
            DType::U32 => dispatch_cmp!(gt, U32),
            #[cfg(feature = "u64")]
            DType::U64 => dispatch_cmp!(gt, U64),
            DType::I8 => dispatch_cmp!(gt, I8),
            #[cfg(feature = "i16")]
            DType::I16 => dispatch_cmp!(gt, I16),
            DType::I32 => dispatch_cmp!(gt, I32),
            #[cfg(feature = "i64")]
            DType::I64 => dispatch_cmp!(gt, I64),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "gt".to_string(),
                })
            },
        },
        "ge" => match dtype {
            DType::BF16 => dispatch_cmp!(ge, BF16),
            DType::F16 => dispatch_cmp!(ge, F16),
            DType::F32 => dispatch_cmp!(ge, F32),
            #[cfg(feature = "u8")]
            DType::U8 => dispatch_cmp!(ge, U8),
            DType::U16 => dispatch_cmp!(ge, U16),
            #[cfg(feature = "u32")]
            DType::U32 => dispatch_cmp!(ge, U32),
            #[cfg(feature = "u64")]
            DType::U64 => dispatch_cmp!(ge, U64),
            DType::I8 => dispatch_cmp!(ge, I8),
            #[cfg(feature = "i16")]
            DType::I16 => dispatch_cmp!(ge, I16),
            DType::I32 => dispatch_cmp!(ge, I32),
            #[cfg(feature = "i64")]
            DType::I64 => dispatch_cmp!(ge, I64),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "ge".to_string(),
                })
            },
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown comparison '{}'", kernel_name),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), output_size, output_dtype))
}

pub fn cmp_scalar_map(
    storage: &MetalStorage,
    layout: &Layout,
    scalar: Scalar,
    kernel_name: &str,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::utils::BufferOffset;

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    let num_els: usize = shape.iter().product();
    // Output is always BOOL for comparison operations
    let output_dtype = DType::BOOL;
    let output = device.new_buffer(num_els, output_dtype, kernel_name)?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    macro_rules! dispatch_cmp_scalar {
        ($op:ident,$dtype:ident,$ty:ty, $conv:expr) => {{
            let val: $ty = $conv;
            hodu_metal_kernels::kernels::call_unary_scalar(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::$op::$dtype,
                shape,
                input,
                strides,
                offset,
                val,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?
        }};
    }

    match kernel_name {
        "eq_scalar" => match dtype {
            DType::BOOL => dispatch_cmp_scalar!(eq_scalar, BOOL, bool, scalar.to_bool()),
            DType::BF16 => dispatch_cmp_scalar!(eq_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_cmp_scalar!(eq_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_cmp_scalar!(eq_scalar, F32, f32, scalar.to_f32()),
            #[cfg(feature = "u8")]
            DType::U8 => dispatch_cmp_scalar!(eq_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_cmp_scalar!(eq_scalar, U16, u16, scalar.to_u16()),
            #[cfg(feature = "u32")]
            DType::U32 => dispatch_cmp_scalar!(eq_scalar, U32, u32, scalar.to_u32()),
            #[cfg(feature = "u64")]
            DType::U64 => dispatch_cmp_scalar!(eq_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_cmp_scalar!(eq_scalar, I8, i8, scalar.to_i8()),
            #[cfg(feature = "i16")]
            DType::I16 => dispatch_cmp_scalar!(eq_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_cmp_scalar!(eq_scalar, I32, i32, scalar.to_i32()),
            #[cfg(feature = "i64")]
            DType::I64 => dispatch_cmp_scalar!(eq_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "eq_scalar".to_string(),
                })
            },
        },
        "ne_scalar" => match dtype {
            DType::BOOL => dispatch_cmp_scalar!(ne_scalar, BOOL, bool, scalar.to_bool()),
            DType::BF16 => dispatch_cmp_scalar!(ne_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_cmp_scalar!(ne_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_cmp_scalar!(ne_scalar, F32, f32, scalar.to_f32()),
            #[cfg(feature = "u8")]
            DType::U8 => dispatch_cmp_scalar!(ne_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_cmp_scalar!(ne_scalar, U16, u16, scalar.to_u16()),
            #[cfg(feature = "u32")]
            DType::U32 => dispatch_cmp_scalar!(ne_scalar, U32, u32, scalar.to_u32()),
            #[cfg(feature = "u64")]
            DType::U64 => dispatch_cmp_scalar!(ne_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_cmp_scalar!(ne_scalar, I8, i8, scalar.to_i8()),
            #[cfg(feature = "i16")]
            DType::I16 => dispatch_cmp_scalar!(ne_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_cmp_scalar!(ne_scalar, I32, i32, scalar.to_i32()),
            #[cfg(feature = "i64")]
            DType::I64 => dispatch_cmp_scalar!(ne_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "ne_scalar".to_string(),
                })
            },
        },
        "lt_scalar" => match dtype {
            DType::BOOL => dispatch_cmp_scalar!(lt_scalar, BOOL, bool, scalar.to_bool()),
            DType::BF16 => dispatch_cmp_scalar!(lt_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_cmp_scalar!(lt_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_cmp_scalar!(lt_scalar, F32, f32, scalar.to_f32()),
            #[cfg(feature = "u8")]
            DType::U8 => dispatch_cmp_scalar!(lt_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_cmp_scalar!(lt_scalar, U16, u16, scalar.to_u16()),
            #[cfg(feature = "u32")]
            DType::U32 => dispatch_cmp_scalar!(lt_scalar, U32, u32, scalar.to_u32()),
            #[cfg(feature = "u64")]
            DType::U64 => dispatch_cmp_scalar!(lt_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_cmp_scalar!(lt_scalar, I8, i8, scalar.to_i8()),
            #[cfg(feature = "i16")]
            DType::I16 => dispatch_cmp_scalar!(lt_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_cmp_scalar!(lt_scalar, I32, i32, scalar.to_i32()),
            #[cfg(feature = "i64")]
            DType::I64 => dispatch_cmp_scalar!(lt_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "lt_scalar".to_string(),
                })
            },
        },
        "le_scalar" => match dtype {
            DType::BOOL => dispatch_cmp_scalar!(le_scalar, BOOL, bool, scalar.to_bool()),
            DType::BF16 => dispatch_cmp_scalar!(le_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_cmp_scalar!(le_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_cmp_scalar!(le_scalar, F32, f32, scalar.to_f32()),
            #[cfg(feature = "u8")]
            DType::U8 => dispatch_cmp_scalar!(le_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_cmp_scalar!(le_scalar, U16, u16, scalar.to_u16()),
            #[cfg(feature = "u32")]
            DType::U32 => dispatch_cmp_scalar!(le_scalar, U32, u32, scalar.to_u32()),
            #[cfg(feature = "u64")]
            DType::U64 => dispatch_cmp_scalar!(le_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_cmp_scalar!(le_scalar, I8, i8, scalar.to_i8()),
            #[cfg(feature = "i16")]
            DType::I16 => dispatch_cmp_scalar!(le_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_cmp_scalar!(le_scalar, I32, i32, scalar.to_i32()),
            #[cfg(feature = "i64")]
            DType::I64 => dispatch_cmp_scalar!(le_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "le_scalar".to_string(),
                })
            },
        },
        "gt_scalar" => match dtype {
            DType::BOOL => dispatch_cmp_scalar!(gt_scalar, BOOL, bool, scalar.to_bool()),
            DType::BF16 => dispatch_cmp_scalar!(gt_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_cmp_scalar!(gt_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_cmp_scalar!(gt_scalar, F32, f32, scalar.to_f32()),
            #[cfg(feature = "u8")]
            DType::U8 => dispatch_cmp_scalar!(gt_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_cmp_scalar!(gt_scalar, U16, u16, scalar.to_u16()),
            #[cfg(feature = "u32")]
            DType::U32 => dispatch_cmp_scalar!(gt_scalar, U32, u32, scalar.to_u32()),
            #[cfg(feature = "u64")]
            DType::U64 => dispatch_cmp_scalar!(gt_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_cmp_scalar!(gt_scalar, I8, i8, scalar.to_i8()),
            #[cfg(feature = "i16")]
            DType::I16 => dispatch_cmp_scalar!(gt_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_cmp_scalar!(gt_scalar, I32, i32, scalar.to_i32()),
            #[cfg(feature = "i64")]
            DType::I64 => dispatch_cmp_scalar!(gt_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "gt_scalar".to_string(),
                })
            },
        },
        "ge_scalar" => match dtype {
            DType::BOOL => dispatch_cmp_scalar!(ge_scalar, BOOL, bool, scalar.to_bool()),
            DType::BF16 => dispatch_cmp_scalar!(ge_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_cmp_scalar!(ge_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_cmp_scalar!(ge_scalar, F32, f32, scalar.to_f32()),
            #[cfg(feature = "u8")]
            DType::U8 => dispatch_cmp_scalar!(ge_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_cmp_scalar!(ge_scalar, U16, u16, scalar.to_u16()),
            #[cfg(feature = "u32")]
            DType::U32 => dispatch_cmp_scalar!(ge_scalar, U32, u32, scalar.to_u32()),
            #[cfg(feature = "u64")]
            DType::U64 => dispatch_cmp_scalar!(ge_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_cmp_scalar!(ge_scalar, I8, i8, scalar.to_i8()),
            #[cfg(feature = "i16")]
            DType::I16 => dispatch_cmp_scalar!(ge_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_cmp_scalar!(ge_scalar, I32, i32, scalar.to_i32()),
            #[cfg(feature = "i64")]
            DType::I64 => dispatch_cmp_scalar!(ge_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "ge_scalar".to_string(),
                })
            },
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown cmp_scalar operation '{}'", kernel_name),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, output_dtype))
}
