use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    scalar::Scalar,
    types::{dtype::DType, layout::Layout},
};

pub fn unary_map(storage: &MetalStorage, layout: &Layout, kernel_name: &str) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::utils::BufferOffset;

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    let num_els: usize = shape.iter().product();
    let output = device.new_buffer(num_els, dtype, kernel_name)?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    macro_rules! dispatch_dtype {
        ($op:ident) => {
            match dtype {
                DType::BF16 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("unary operation '{}'", kernel_name),
                    });
                },
            }
        };
    }

    match kernel_name {
        "neg" => dispatch_dtype!(neg),
        "abs" => dispatch_dtype!(abs),
        "sign" => dispatch_dtype!(sign),
        "square" => dispatch_dtype!(square),
        "sqrt" => dispatch_dtype!(sqrt),
        "recip" => dispatch_dtype!(recip),
        "relu" => dispatch_dtype!(relu),
        "sigmoid" => dispatch_dtype!(sigmoid),
        "tanh" => dispatch_dtype!(tanh),
        "gelu" => dispatch_dtype!(gelu),
        "softplus" => dispatch_dtype!(softplus),
        "silu" => dispatch_dtype!(silu),
        "mish" => dispatch_dtype!(mish),
        "sin" => dispatch_dtype!(sin),
        "cos" => dispatch_dtype!(cos),
        "tan" => dispatch_dtype!(tan),
        "exp" => dispatch_dtype!(exp),
        "exp2" => dispatch_dtype!(exp2),
        "exp10" => dispatch_dtype!(exp10),
        "ln" => dispatch_dtype!(ln),
        "log2" => dispatch_dtype!(log2),
        "log10" => dispatch_dtype!(log10),
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown unary operation '{}'", kernel_name),
            });
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn unary_logical_map(storage: &MetalStorage, layout: &Layout, kernel_name: &str) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::utils::BufferOffset;

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    let num_els: usize = shape.iter().product();
    // Output is always BOOL for logical operations
    let output_dtype = DType::BOOL;
    let output = device.new_buffer(num_els, output_dtype, kernel_name)?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    // Only logical_not is supported
    match kernel_name {
        "logical_not" => match dtype {
            DType::BF16 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::BF16,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::F16 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::F16,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::F32 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::F32,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::U8 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::U8,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::U16 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::U16,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::U32 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::U32,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::U64 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::U64,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::I8 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::I8,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::I16 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::I16,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::I32 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::I32,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::I64 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::I64,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "logical_not".to_string(),
                })
            },
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown unary logical operation '{}'", kernel_name),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, output_dtype))
}

pub fn unary_scalar_map(
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
    let output = device.new_buffer(num_els, dtype, kernel_name)?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    macro_rules! dispatch_unary_scalar {
        ($op:ident, $dtype:ident, $ty:ty, $conv:expr) => {{
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
        "add_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(add_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(add_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(add_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(add_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(add_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(add_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(add_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(add_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(add_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(add_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(add_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "add_scalar".to_string(),
                })
            },
        },
        "sub_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(sub_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(sub_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(sub_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(sub_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(sub_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(sub_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(sub_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(sub_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(sub_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(sub_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(sub_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "sub_scalar".to_string(),
                })
            },
        },
        "mul_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(mul_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(mul_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(mul_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(mul_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(mul_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(mul_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(mul_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(mul_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(mul_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(mul_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(mul_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "mul_scalar".to_string(),
                })
            },
        },
        "div_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(div_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(div_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(div_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(div_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(div_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(div_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(div_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(div_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(div_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(div_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(div_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "div_scalar".to_string(),
                })
            },
        },
        "pow_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(pow_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(pow_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(pow_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(pow_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(pow_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(pow_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(pow_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(pow_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(pow_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(pow_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(pow_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "pow_scalar".to_string(),
                })
            },
        },
        "maximum_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(maximum_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(maximum_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(maximum_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(maximum_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(maximum_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(maximum_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(maximum_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(maximum_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(maximum_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(maximum_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(maximum_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "maximum_scalar".to_string(),
                })
            },
        },
        "minimum_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(minimum_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(minimum_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(minimum_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(minimum_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(minimum_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(minimum_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(minimum_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(minimum_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(minimum_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(minimum_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(minimum_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "minimum_scalar".to_string(),
                })
            },
        },
        "leaky_relu" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(leaky_relu, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(leaky_relu, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(leaky_relu, F32, f32, scalar.to_f32()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "leaky_relu".to_string(),
                })
            },
        },
        "elu" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(elu, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(elu, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(elu, F32, f32, scalar.to_f32()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "elu".to_string(),
                })
            },
        },
        "prelu" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(prelu, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(prelu, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(prelu, F32, f32, scalar.to_f32()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "prelu".to_string(),
                })
            },
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown unary_scalar operation '{}'", kernel_name),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}
