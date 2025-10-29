use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    types::{dtype::DType, layout::Layout},
};

pub fn concat_map(
    first: &MetalStorage,
    others: &[&MetalStorage],
    layouts: &[&Layout],
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_concat, utils::BufferOffset};

    let dtype = first.get_dtype();
    let device = first.get_hodu_device();

    // Validate all storages have same dtype
    for other in others {
        if other.get_dtype() != dtype {
            return Err(HoduError::DTypeConflictInOp {
                left: dtype,
                right: other.get_dtype(),
                op: "concat".to_string(),
            });
        }
    }

    let first_shape = layouts[0].get_shape();
    let ndim = first_shape.len();

    if dim >= ndim {
        return Err(HoduError::IncompatibleShapes {
            lhs: first_shape.to_vec(),
            rhs: vec![],
            op: format!(
                "concat - dimension {} out of range for {}-dimensional tensor",
                dim, ndim
            ),
        });
    }

    // Verify all tensors have same shape except at concat dimension
    for layout in layouts.iter().skip(1) {
        let shape = layout.get_shape();
        if shape.len() != ndim {
            return Err(HoduError::IncompatibleShapes {
                lhs: first_shape.to_vec(),
                rhs: shape.to_vec(),
                op: "concat - all tensors must have the same number of dimensions".to_string(),
            });
        }
        for (j, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
            if j != dim && s1 != s2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: first_shape.to_vec(),
                    rhs: shape.to_vec(),
                    op: format!("concat - dimension {} must match (got {} vs {})", j, s1, s2),
                });
            }
        }
    }

    // Calculate output shape
    let mut output_shape = first_shape.to_vec();
    output_shape[dim] = layouts.iter().map(|l| l.get_shape()[dim]).sum();

    let num_els: usize = output_shape.iter().product();
    let output = device.new_buffer(num_els, dtype, "concat")?;
    let command_buffer = device.command_buffer()?;

    // Prepare input shapes, strides, offsets
    let mut input_shapes = Vec::with_capacity(layouts.len() * ndim);
    let mut input_strides = Vec::with_capacity(layouts.len() * ndim);
    let mut input_offsets = Vec::with_capacity(layouts.len());
    let mut input_buffer_offsets = Vec::with_capacity(layouts.len());

    // Collect all inputs (first + others)
    let all_storages: Vec<&MetalStorage> = std::iter::once(first).chain(others.iter().copied()).collect();

    // Calculate total size needed for combined buffer
    let mut total_elements = 0;
    for (_storage, layout) in all_storages.iter().zip(layouts.iter()) {
        let shape = layout.get_shape();
        let num_els_in_storage: usize = shape.iter().product();
        total_elements += num_els_in_storage;
    }

    // Create a combined buffer that holds all input tensors
    let combined_buffer = device.new_buffer(total_elements, dtype, "concat_combined_input")?;

    // Copy all input tensors into the combined buffer and track offsets
    let mut cumulative_offset = 0;
    let encoder = command_buffer.blit_command_encoder();

    for (storage, layout) in all_storages.iter().zip(layouts.iter()) {
        let shape = layout.get_shape();
        let strides = layout.get_strides();
        let offset = layout.get_offset();
        let num_els_in_storage: usize = shape.iter().product();

        input_shapes.extend_from_slice(shape);
        input_strides.extend_from_slice(strides);
        input_offsets.push(offset);
        input_buffer_offsets.push(cumulative_offset);

        // Copy this storage's buffer to the combined buffer
        let source_offset = offset * dtype.get_size_in_bytes();
        let dest_offset = cumulative_offset * dtype.get_size_in_bytes();
        let size = num_els_in_storage * dtype.get_size_in_bytes();

        encoder.copy_from_buffer(storage.buffer(), source_offset, &combined_buffer, dest_offset, size);

        cumulative_offset += num_els_in_storage;
    }
    encoder.end_encoding();

    let input = BufferOffset {
        buffer: &combined_buffer,
        offset_in_bytes: 0,
    };

    macro_rules! dispatch_concat {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BOOL => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::BF16 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "concat".to_string(),
                    })
                },
            }
        };
    }

    dispatch_concat!(concat);

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn split_map(
    storage: &MetalStorage,
    layout: &Layout,
    dim: usize,
    sizes: &[usize],
) -> HoduResult<Vec<MetalStorage>> {
    use hodu_metal_kernels::{kernels::call_split, utils::BufferOffset};

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
            op: format!("split - dimension {} out of range for {}-dimensional tensor", dim, ndim),
        });
    }

    // Verify sizes sum to dimension size
    let total_size: usize = sizes.iter().sum();
    if total_size != shape[dim] {
        return Err(HoduError::IncompatibleShapes {
            lhs: vec![shape[dim]],
            rhs: vec![total_size],
            op: format!(
                "split - sizes must sum to dimension size (got {} vs {})",
                total_size, shape[dim]
            ),
        });
    }

    let command_buffer = device.command_buffer()?;
    let input_buffer = storage.buffer();
    let input_offset_bytes = offset * dtype.get_size_in_bytes();

    let mut results = Vec::with_capacity(sizes.len());
    let mut split_offset = 0;

    for &size in sizes {
        let mut output_shape = shape.to_vec();
        output_shape[dim] = size;
        let num_els: usize = output_shape.iter().product();

        let output = device.new_buffer(num_els, dtype, "split")?;

        let input = BufferOffset {
            buffer: input_buffer,
            offset_in_bytes: input_offset_bytes,
        };

        macro_rules! dispatch_split {
            ($kernel_mod:ident) => {
                match dtype {
                    DType::BOOL => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::BF16 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::BF16,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::F16 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::F16,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::F32 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::F32,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::U8 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::U8,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::U16 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::U16,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::U32 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::U32,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::U64 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::U64,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::I8 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::I8,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::I16 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::I16,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::I32 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::I32,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::I64 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::I64,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    _ => {
                        return Err(HoduError::UnsupportedDType {
                            dtype,
                            op: "split".to_string(),
                        })
                    },
                }
            };
        }

        dispatch_split!(split);

        results.push(MetalStorage::new(output, device.clone(), num_els, dtype));
        split_offset += size;
    }

    Ok(results)
}
