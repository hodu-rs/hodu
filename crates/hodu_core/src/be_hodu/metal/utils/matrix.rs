use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    types::{dtype::DType, layout::Layout},
};

pub fn matmul_map(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{
        kernels::{call_matmul, matmul},
        utils::BufferOffset,
    };

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

    if lhs_ndim < 2 || rhs_ndim < 2 {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "matmul - both tensors must be at least 2D".to_string(),
        });
    }

    let m = lhs_shape[lhs_ndim - 2];
    let k = lhs_shape[lhs_ndim - 1];
    let rhs_k = rhs_shape[rhs_ndim - 2];
    let n = rhs_shape[rhs_ndim - 1];

    if k != rhs_k {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "matmul - inner dimensions must match".to_string(),
        });
    }

    let lhs_batch_ndim = lhs_ndim - 2;
    let rhs_batch_ndim = rhs_ndim - 2;
    let batch_ndim = lhs_batch_ndim.max(rhs_batch_ndim);

    let mut batch_shape = vec![1; batch_ndim];
    for i in 0..batch_ndim {
        let lhs_dim = if i < lhs_batch_ndim {
            lhs_shape[lhs_batch_ndim - 1 - i]
        } else {
            1
        };
        let rhs_dim = if i < rhs_batch_ndim {
            rhs_shape[rhs_batch_ndim - 1 - i]
        } else {
            1
        };

        if lhs_dim != 1 && rhs_dim != 1 && lhs_dim != rhs_dim {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: "matmul - incompatible batch dimensions".to_string(),
            });
        }
        batch_shape[batch_ndim - 1 - i] = lhs_dim.max(rhs_dim);
    }

    let total_batches: usize = batch_shape.iter().product();
    let num_els = total_batches * m * n;

    let output = device.new_buffer(num_els, dtype, "matmul")?;
    let command_buffer = device.command_buffer()?;

    let lhs = BufferOffset {
        buffer: lhs_storage.buffer(),
        offset_in_bytes: lhs_offset * dtype.get_size_in_bytes(),
    };
    let rhs = BufferOffset {
        buffer: rhs_storage.buffer(),
        offset_in_bytes: rhs_offset * dtype.get_size_in_bytes(),
    };

    // Prepare metadata for Metal kernel
    // call_matmul expects: metadata[0] = num_els, metadata[1..] = actual kernel metadata
    // Kernel expects: lhs_ndim, rhs_ndim, batch_ndim, lhs_shape, rhs_shape, batch_shape, lhs_strides, rhs_strides, lhs_offset, rhs_offset, M, K, N
    let mut metadata = Vec::new();
    metadata.push(num_els); // For call_matmul
    metadata.push(lhs_ndim);
    metadata.push(rhs_ndim);
    metadata.push(batch_ndim);
    metadata.extend_from_slice(lhs_shape);
    metadata.extend_from_slice(rhs_shape);
    metadata.extend_from_slice(&batch_shape);
    metadata.extend_from_slice(lhs_strides);
    metadata.extend_from_slice(rhs_strides);
    metadata.push(lhs_offset);
    metadata.push(rhs_offset);
    metadata.push(m);
    metadata.push(k);
    metadata.push(n);

    match dtype {
        DType::BF16 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::BF16,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::F16 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::F16,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::F32 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::F32,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        #[cfg(feature = "u8")]
        DType::U8 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::U8,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::U16 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::U16,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        #[cfg(feature = "u32")]
        DType::U32 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::U32,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        #[cfg(feature = "u64")]
        DType::U64 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::U64,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::I8 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::I8,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        #[cfg(feature = "i16")]
        DType::I16 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::I16,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::I32 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::I32,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        #[cfg(feature = "i64")]
        DType::I64 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::I64,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: "matmul".to_string(),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn dot_map(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{
        kernels::{call_dot, dot},
        utils::BufferOffset,
    };

    let dtype = lhs_storage.get_dtype();
    let device = lhs_storage.get_hodu_device();
    let lhs_shape = lhs_layout.get_shape();
    let lhs_strides = lhs_layout.get_strides();
    let lhs_offset = lhs_layout.get_offset();
    let rhs_shape = rhs_layout.get_shape();
    let rhs_strides = rhs_layout.get_strides();
    let rhs_offset = rhs_layout.get_offset();

    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "dot - only 2D tensors supported".to_string(),
        });
    }

    let (m, k1) = (lhs_shape[0], lhs_shape[1]);
    let (k2, n) = (rhs_shape[0], rhs_shape[1]);

    if k1 != k2 {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "dot - inner dimensions must match".to_string(),
        });
    }

    let num_els = m * n;

    let output = device.new_buffer(num_els, dtype, "dot")?;
    let command_buffer = device.command_buffer()?;

    let lhs = BufferOffset {
        buffer: lhs_storage.buffer(),
        offset_in_bytes: lhs_offset * dtype.get_size_in_bytes(),
    };
    let rhs = BufferOffset {
        buffer: rhs_storage.buffer(),
        offset_in_bytes: rhs_offset * dtype.get_size_in_bytes(),
    };

    // Prepare metadata for Metal kernel
    // Layout: [M, K, unused, N, lhs_stride_m, lhs_stride_k, rhs_stride_k, rhs_stride_n, lhs_offset, rhs_offset]
    // Note: metadata[2] is unused by the kernel, but metadata[3] is N
    let metadata = vec![
        m,
        k1,
        0, // unused (metadata[2])
        n, // metadata[3]
        lhs_strides[0],
        lhs_strides[1],
        rhs_strides[0],
        rhs_strides[1],
        lhs_offset,
        rhs_offset,
    ];

    match dtype {
        DType::BF16 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::BF16,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::F16 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::F16,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::F32 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::F32,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        #[cfg(feature = "u8")]
        DType::U8 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::U8,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::U16 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::U16,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        #[cfg(feature = "u32")]
        DType::U32 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::U32,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        #[cfg(feature = "u64")]
        DType::U64 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::U64,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::I8 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::I8,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        #[cfg(feature = "i16")]
        DType::I16 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::I16,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::I32 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::I32,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        #[cfg(feature = "i64")]
        DType::I64 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::I64,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: "dot".to_string(),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}
