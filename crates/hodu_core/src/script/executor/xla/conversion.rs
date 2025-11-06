use crate::{
    be::storage::BackendStorage,
    be_cpu::storage::CpuStorage,
    error::{HoduError, HoduResult},
    script::builder::ir::ConstantData,
    tensor::from_storage,
    types::{DType, Layout},
};
use hodu_xla::{ElementType, Literal, XlaBuilder, XlaOp};

/// Convert DType to XLA ElementType
pub fn dtype_to_element_type(dtype: DType) -> HoduResult<ElementType> {
    match dtype {
        DType::BOOL => Ok(ElementType::Pred),
        DType::F8E4M3 => Err(HoduError::XlaError("f8e4m3 not supported".to_string())),
        #[cfg(feature = "f8e5m2")]
        DType::F8E5M2 => Err(HoduError::XlaError("f8e5m2 not supported".to_string())),
        DType::BF16 => Ok(ElementType::Bf16),
        DType::F16 => Ok(ElementType::F16),
        DType::F32 => Ok(ElementType::F32),
        #[cfg(feature = "f64")]
        DType::F64 => Ok(ElementType::F64),
        DType::U8 => Ok(ElementType::U8),
        #[cfg(feature = "u16")]
        DType::U16 => Ok(ElementType::U16),
        DType::U32 => Ok(ElementType::U32),
        #[cfg(feature = "u64")]
        DType::U64 => Ok(ElementType::U64),
        DType::I8 => Ok(ElementType::S8),
        #[cfg(feature = "i16")]
        DType::I16 => Ok(ElementType::S16),
        DType::I32 => Ok(ElementType::S32),
        #[cfg(feature = "i64")]
        DType::I64 => Ok(ElementType::S64),
    }
}

/// Convert Tensor to XLA Literal
pub fn tensor_to_literal(tensor: &crate::tensor::Tensor) -> HoduResult<Literal> {
    tensor.with_storage(|storage| match storage {
        BackendStorage::CPU(cpu_storage) => cpu_storage_to_literal(cpu_storage),
        #[cfg(any(feature = "cuda", feature = "metal"))]
        _ => Err(HoduError::InternalError(
            "Only CPU storage supported for XLA conversion".to_string(),
        )),
    })
}

/// Convert CPU storage to XLA Literal
pub fn cpu_storage_to_literal(storage: &CpuStorage) -> HoduResult<Literal> {
    match storage {
        CpuStorage::BOOL(data) => Ok(Literal::vec1(data)),
        CpuStorage::F8E4M3(_) => Err(HoduError::InternalError("F8E4M3 not supported by XLA".to_string())),
        #[cfg(feature = "f8e5m2")]
        CpuStorage::F8E5M2(_) => Err(HoduError::InternalError("F8E5M2 not supported by XLA".to_string())),
        CpuStorage::BF16(data) => Ok(Literal::vec1(data)),
        CpuStorage::F16(data) => Ok(Literal::vec1(data)),
        CpuStorage::F32(data) => Ok(Literal::vec1(data)),
        #[cfg(feature = "f64")]
        CpuStorage::F64(data) => Ok(Literal::vec1(data)),
        CpuStorage::U8(data) => Ok(Literal::vec1(data)),
        #[cfg(feature = "u16")]
        CpuStorage::U16(data) => Ok(Literal::vec1(data)),
        CpuStorage::U32(data) => Ok(Literal::vec1(data)),
        #[cfg(feature = "u64")]
        CpuStorage::U64(data) => Ok(Literal::vec1(data)),
        CpuStorage::I8(data) => Ok(Literal::vec1(data)),
        #[cfg(feature = "i16")]
        CpuStorage::I16(data) => Ok(Literal::vec1(data)),
        CpuStorage::I32(data) => Ok(Literal::vec1(data)),
        #[cfg(feature = "i64")]
        CpuStorage::I64(data) => Ok(Literal::vec1(data)),
    }
}

/// Convert XLA Literal to Tensor
pub fn literal_to_tensor(literal: &Literal, dtype: DType) -> HoduResult<crate::tensor::Tensor> {
    // Get shape from literal
    let shape_result = literal
        .shape()
        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
    let shape: Vec<usize> = match &shape_result {
        hodu_xla::Shape::Array(array_shape) => array_shape.dims().iter().map(|&d| d as usize).collect(),
        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
    };

    // Convert to CPU storage based on dtype
    let cpu_storage = match dtype {
        DType::BOOL => {
            let data = literal
                .to_vec::<bool>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract BOOL data: {:?}", e)))?;
            CpuStorage::BOOL(data)
        },
        DType::F8E4M3 => return Err(HoduError::InternalError("F8E4M3 not supported by XLA".to_string())),
        #[cfg(feature = "f8e5m2")]
        DType::F8E5M2 => return Err(HoduError::InternalError("F8E5M2 not supported by XLA".to_string())),
        DType::BF16 => {
            let data = literal
                .to_vec::<half::bf16>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract BF16 data: {:?}", e)))?;
            CpuStorage::BF16(data)
        },
        DType::F16 => {
            let data = literal
                .to_vec::<half::f16>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract F16 data: {:?}", e)))?;
            CpuStorage::F16(data)
        },
        DType::F32 => {
            let data = literal
                .to_vec::<f32>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract F32 data: {:?}", e)))?;
            CpuStorage::F32(data)
        },
        #[cfg(feature = "f64")]
        DType::F64 => {
            let data = literal
                .to_vec::<f64>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract F64 data: {:?}", e)))?;
            CpuStorage::F64(data)
        },
        DType::U8 => {
            let data = literal
                .to_vec::<u8>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract U8 data: {:?}", e)))?;
            CpuStorage::U8(data)
        },
        #[cfg(feature = "u16")]
        DType::U16 => {
            let data = literal
                .to_vec::<u16>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract U16 data: {:?}", e)))?;
            CpuStorage::U16(data)
        },
        DType::U32 => {
            let data = literal
                .to_vec::<u32>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract U32 data: {:?}", e)))?;
            CpuStorage::U32(data)
        },
        #[cfg(feature = "u64")]
        DType::U64 => {
            let data = literal
                .to_vec::<u64>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract U64 data: {:?}", e)))?;
            CpuStorage::U64(data)
        },
        DType::I8 => {
            let data = literal
                .to_vec::<i8>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract I8 data: {:?}", e)))?;
            CpuStorage::I8(data)
        },
        #[cfg(feature = "i16")]
        DType::I16 => {
            let data = literal
                .to_vec::<i16>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract I16 data: {:?}", e)))?;
            CpuStorage::I16(data)
        },
        DType::I32 => {
            let data = literal
                .to_vec::<i32>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract I32 data: {:?}", e)))?;
            CpuStorage::I32(data)
        },
        #[cfg(feature = "i64")]
        DType::I64 => {
            let data = literal
                .to_vec::<i64>()
                .map_err(|e| HoduError::InternalError(format!("Failed to extract I64 data: {:?}", e)))?;
            CpuStorage::I64(data)
        },
    };

    let shape_u32: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
    let shape_obj = crate::types::Shape::new(&shape_u32);
    let layout = Layout::from_shape(&shape_obj);
    Ok(from_storage(BackendStorage::CPU(cpu_storage), layout, true, false))
}

/// Create a constant XLA operation from ConstantData
pub fn create_constant_op(builder: &XlaBuilder, constant_data: &ConstantData) -> HoduResult<XlaOp> {
    // Parse the data bytes into appropriate CPU storage
    let cpu_storage = CpuStorage::from_bytes(&constant_data.data, constant_data.dtype)
        .map_err(|e| HoduError::InternalError(format!("Failed to parse constant data: {:?}", e)))?;

    // Convert to XLA literal
    let literal = cpu_storage_to_literal(&cpu_storage)?;

    // Reshape the literal to match the constant's shape
    let dims: Vec<i64> = constant_data.shape.dims().iter().map(|&d| d as i64).collect();
    let reshaped = literal
        .reshape(&dims)
        .map_err(|e| HoduError::InternalError(format!("Failed to reshape constant: {:?}", e)))?;

    // Convert literal to constant op
    builder
        .constant_literal(&reshaped)
        .map_err(|e| HoduError::InternalError(format!("Failed to create constant op: {:?}", e)))
}
