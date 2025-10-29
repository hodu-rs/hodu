use super::XlaExecutor;
use crate::{
    compat::*,
    error::{HoduError, HoduResult},
};
use hodu_xla::{ElementType, PrimitiveType, XlaBuilder, XlaOp};
use std::f32;

// Helper function to convert XLA errors to HoduError
pub(super) fn xla_error_to_hodu_error(err: hodu_xla::Error) -> HoduError {
    HoduError::InternalError(format!("XLA error: {:?}", err))
}

impl XlaExecutor {
    // Helper method to create multiply computation (cached per executor instance)
    pub(super) fn create_multiply_computation() -> HoduResult<hodu_xla::XlaComputation> {
        let builder = XlaBuilder::new("multiply_computation");
        let lhs = builder
            .parameter(0, ElementType::F32, &[], "lhs")
            .map_err(xla_error_to_hodu_error)?;
        let rhs = builder
            .parameter(1, ElementType::F32, &[], "rhs")
            .map_err(xla_error_to_hodu_error)?;
        let result = lhs.mul_(&rhs).map_err(xla_error_to_hodu_error)?;
        result.build().map_err(xla_error_to_hodu_error)
    }

    // Helper method to convert PrimitiveType to ElementType
    pub(super) fn element_type_to_element_type(element_type: PrimitiveType) -> HoduResult<ElementType> {
        match element_type {
            PrimitiveType::Pred => Ok(ElementType::Pred),
            PrimitiveType::S8 => Ok(ElementType::S8),
            PrimitiveType::S16 => Ok(ElementType::S16),
            PrimitiveType::S32 => Ok(ElementType::S32),
            PrimitiveType::S64 => Ok(ElementType::S64),
            PrimitiveType::U8 => Ok(ElementType::U8),
            PrimitiveType::U16 => Ok(ElementType::U16),
            PrimitiveType::U32 => Ok(ElementType::U32),
            PrimitiveType::U64 => Ok(ElementType::U64),
            PrimitiveType::F16 => Ok(ElementType::F16),
            PrimitiveType::F32 => Ok(ElementType::F32),
            PrimitiveType::Bf16 => Ok(ElementType::Bf16),
            PrimitiveType::F64 => Ok(ElementType::F64),
            PrimitiveType::C64 => Ok(ElementType::C64),
            PrimitiveType::C128 => Ok(ElementType::C128),
            _ => Err(HoduError::InternalError(format!(
                "Cannot convert PrimitiveType {:?} to ElementType",
                element_type
            ))),
        }
    }
}

impl XlaExecutor {
    pub(super) fn convert_constant_to_xla_op(
        &self,
        builder: &XlaBuilder,
        constant: &crate::script::ir::ConstantNode,
    ) -> HoduResult<XlaOp> {
        use crate::types::dtype::DType;

        // Decompress data if needed
        let data = match &constant.compression {
            #[cfg(all(feature = "serde", feature = "std"))]
            Some(crate::script::ir::CompressionType::Gzip) => {
                let mut decoder = flate2::read::GzDecoder::new(&constant.data[..]);
                let mut decompressed = Vec::new();
                std::io::Read::read_to_end(&mut decoder, &mut decompressed)
                    .map_err(|e| HoduError::DecompressionError(e.to_string()))?;
                decompressed
            },
            #[cfg(not(all(feature = "serde", feature = "std")))]
            Some(crate::script::ir::CompressionType::Gzip) => {
                return Err(HoduError::InternalError(
                    "Gzip decompression requires both 'serde' and 'std' features to be enabled".to_string(),
                ));
            },
            Some(crate::script::ir::CompressionType::None) => constant.data.clone(),
            Some(crate::script::ir::CompressionType::Zstd) => {
                return Err(HoduError::InternalError(
                    "Zstd decompression not implemented for XLA".to_string(),
                ));
            },
            None => constant.data.clone(),
        };

        let dims: Vec<i64> = constant.shape.iter().map(|&d| d as i64).collect();

        // Convert DType to XLA ElementType
        let element_type = Self::dtype_to_element_type(constant.dtype)?;

        // Create XLA constant based on dtype
        let xla_op = match constant.dtype {
            DType::F32 => {
                let values: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::F64 => {
                let values: Vec<f64> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                        ])
                    })
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::I32 => {
                let values: Vec<i32> = data
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::I64 => {
                let values: Vec<i64> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                        ])
                    })
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::I16 => {
                let values: Vec<i16> = data
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::I8 => {
                let values: Vec<i8> = data.iter().map(|&b| b as i8).collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::U32 => {
                let values: Vec<u32> = data
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::U64 => {
                let values: Vec<u64> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                        ])
                    })
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::U16 => {
                let values: Vec<u16> = data
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                self.create_xla_constant(builder, &values, &dims, element_type)
            },
            DType::U8 => self.create_xla_constant(builder, &data, &dims, element_type),
            DType::BOOL => {
                let literal = hodu_xla::Literal::create_from_shape_and_untyped_data(
                    element_type,
                    &dims.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                    &data,
                )
                .map_err(xla_error_to_hodu_error)?;
                Ok(builder.constant_literal(&literal).map_err(xla_error_to_hodu_error)?)
            },
            // For F16 and BF16, we'll create them using literal directly since they may not implement NativeType
            DType::F16 => {
                let literal = hodu_xla::Literal::create_from_shape_and_untyped_data(
                    element_type,
                    &dims.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                    &data,
                )
                .map_err(xla_error_to_hodu_error)?;
                Ok(builder.constant_literal(&literal).map_err(xla_error_to_hodu_error)?)
            },
            DType::BF16 => {
                let literal = hodu_xla::Literal::create_from_shape_and_untyped_data(
                    element_type,
                    &dims.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                    &data,
                )
                .map_err(xla_error_to_hodu_error)?;
                Ok(builder.constant_literal(&literal).map_err(xla_error_to_hodu_error)?)
            },
            DType::F8E4M3 | DType::F8E5M2 => {
                return Err(HoduError::InternalError(format!(
                    "XLA does not support {:?} dtype",
                    constant.dtype
                )));
            },
        }?;

        Ok(xla_op)
    }

    pub(super) fn create_xla_constant<T: hodu_xla::NativeType + Copy>(
        &self,
        builder: &XlaBuilder,
        values: &[T],
        dims: &[i64],
        element_type: ElementType,
    ) -> HoduResult<XlaOp> {
        if dims.is_empty() {
            // Scalar constant
            builder.constant_r0(values[0]).map_err(xla_error_to_hodu_error)
        } else if dims.len() == 1 {
            // 1D constant
            builder.constant_r1(values).map_err(xla_error_to_hodu_error)
        } else {
            // Multi-dimensional constant - use literal creation
            let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
            let literal = self.create_literal_from_values(values, &shape, element_type)?;
            builder.constant_literal(&literal).map_err(xla_error_to_hodu_error)
        }
    }

    pub(super) fn create_literal_from_values<T>(
        &self,
        values: &[T],
        shape: &[usize],
        element_type: ElementType,
    ) -> HoduResult<hodu_xla::Literal>
    where
        T: Copy,
    {
        use std::mem;

        // Convert values to bytes based on type
        let data_bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, mem::size_of_val(values)) };

        hodu_xla::Literal::create_from_shape_and_untyped_data(element_type, shape, data_bytes)
            .map_err(xla_error_to_hodu_error)
    }
}
