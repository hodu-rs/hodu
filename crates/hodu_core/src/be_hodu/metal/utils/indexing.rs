use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    types::{dtype::DType, layout::Layout},
};

pub fn index_select_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_index_select, utils::BufferOffset};

    // Convert indices to I32 if needed
    let indices_i32 = match indices_storage.get_dtype() {
        DType::I32 => indices_storage.clone(),
        DType::I64 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            // Need to convert to I32
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I32(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I64(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "index_select - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "index_select - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::IncompatibleShapes {
            lhs: shape.to_vec(),
            rhs: vec![],
            op: format!(
                "index_select - dimension {} out of range for {}-dimensional tensor",
                dim, ndim
            ),
        });
    }

    let num_indices: usize = indices_layout.get_shape().iter().product();

    // Calculate output shape
    let mut output_shape = shape.to_vec();
    output_shape[dim] = num_indices;
    let num_els: usize = output_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "index_select")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i32.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I32.get_size_in_bytes(),
    };

    macro_rules! dispatch_index_select {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BOOL => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::BF16 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "index_select".to_string(),
                    })
                },
            }
        };
    }

    dispatch_index_select!(index_select);

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn index_put_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    values_storage: &MetalStorage,
    values_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_index_put, utils::BufferOffset};

    // Convert indices to I32 if needed
    let indices_i32 = match indices_storage.get_dtype() {
        DType::I32 => indices_storage.clone(),
        DType::I64 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I32(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I64(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "index_put - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "index_put - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dtype != values_storage.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: values_storage.get_dtype(),
            op: "index_put".to_string(),
        });
    }

    if dim >= ndim {
        return Err(HoduError::IncompatibleShapes {
            lhs: shape.to_vec(),
            rhs: vec![],
            op: format!(
                "index_put - dimension {} out of range for {}-dimensional tensor",
                dim, ndim
            ),
        });
    }

    let num_indices: usize = indices_layout.get_shape().iter().product();
    let num_els: usize = shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "index_put")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i32.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I32.get_size_in_bytes(),
    };

    let values = BufferOffset {
        buffer: values_storage.buffer(),
        offset_in_bytes: values_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let values_strides = values_layout.get_strides();
    let values_offset = values_layout.get_offset();

    macro_rules! dispatch_index_put {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BOOL => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::BF16 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "index_put".to_string(),
                    })
                },
            }
        };
    }

    dispatch_index_put!(index_put);

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn gather_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_gather, utils::BufferOffset};

    // Convert indices to I64 if needed
    let indices_i64 = match indices_storage.get_dtype() {
        DType::I64 => indices_storage.clone(),
        DType::I32 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I64(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I32(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "gather - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "gather - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::IncompatibleShapes {
            lhs: shape.to_vec(),
            rhs: vec![],
            op: format!(
                "gather - dimension {} out of range for {}-dimensional tensor",
                dim, ndim
            ),
        });
    }

    let indices_shape = indices_layout.get_shape();
    let num_els: usize = indices_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "gather")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i64.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I64.get_size_in_bytes(),
    };

    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    macro_rules! dispatch_gather {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BOOL => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::BF16 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "gather".to_string(),
                    })
                },
            }
        };
    }

    dispatch_gather!(gather);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn scatter_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    src_storage: &MetalStorage,
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_scatter, utils::BufferOffset};

    let indices_i64 = match indices_storage.get_dtype() {
        DType::I64 => indices_storage.clone(),
        DType::I32 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I64(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "scatter - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "scatter - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    if dtype != src_storage.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: src_storage.get_dtype(),
            op: "scatter".to_string(),
        });
    }

    let num_els: usize = shape.iter().product();
    let output = device.new_buffer(num_els, dtype, "scatter")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i64.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I64.get_size_in_bytes(),
    };

    let src = BufferOffset {
        buffer: src_storage.buffer(),
        offset_in_bytes: src_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    macro_rules! dispatch_scatter {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BOOL => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::BF16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "scatter".to_string(),
                    })
                },
            }
        };
    }

    dispatch_scatter!(scatter);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn scatter_add_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    src_storage: &MetalStorage,
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_scatter, utils::BufferOffset};

    let indices_i64 = match indices_storage.get_dtype() {
        DType::I64 => indices_storage.clone(),
        DType::I32 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I64(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "scatter_add - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "scatter_add - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    if dtype != src_storage.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: src_storage.get_dtype(),
            op: "scatter_add".to_string(),
        });
    }

    let num_els: usize = shape.iter().product();
    let output = device.new_buffer(num_els, dtype, "scatter_add")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i64.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I64.get_size_in_bytes(),
    };

    let src = BufferOffset {
        buffer: src_storage.buffer(),
        offset_in_bytes: src_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    macro_rules! dispatch_scatter_add {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "scatter_add".to_string(),
                    })
                },
            }
        };
    }

    dispatch_scatter_add!(scatter_add);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn scatter_max_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    src_storage: &MetalStorage,
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_scatter, utils::BufferOffset};

    let indices_i64 = match indices_storage.get_dtype() {
        DType::I64 => indices_storage.clone(),
        DType::I32 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I64(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "scatter_max - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "scatter_max - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    if dtype != src_storage.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: src_storage.get_dtype(),
            op: "scatter_max".to_string(),
        });
    }

    let num_els: usize = shape.iter().product();
    let output = device.new_buffer(num_els, dtype, "scatter_max")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i64.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I64.get_size_in_bytes(),
    };

    let src = BufferOffset {
        buffer: src_storage.buffer(),
        offset_in_bytes: src_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    macro_rules! dispatch_scatter_max {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "scatter_max".to_string(),
                    })
                },
            }
        };
    }

    dispatch_scatter_max!(scatter_max);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn scatter_min_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    src_storage: &MetalStorage,
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_scatter, utils::BufferOffset};

    let indices_i64 = match indices_storage.get_dtype() {
        DType::I64 => indices_storage.clone(),
        DType::I32 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I64(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "scatter_min - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "scatter_min - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    if dtype != src_storage.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: src_storage.get_dtype(),
            op: "scatter_min".to_string(),
        });
    }

    let num_els: usize = shape.iter().product();
    let output = device.new_buffer(num_els, dtype, "scatter_min")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i64.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I64.get_size_in_bytes(),
    };

    let src = BufferOffset {
        buffer: src_storage.buffer(),
        offset_in_bytes: src_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    macro_rules! dispatch_scatter_min {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "scatter_min".to_string(),
                    })
                },
            }
        };
    }

    dispatch_scatter_min!(scatter_min);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}
