//! Tensor data types for cross-plugin communication
//!
//! Re-exports from hodu_plugin with additional hodu_core integration.

// Re-export base types from hodu_plugin
pub use hodu_plugin::tensor::{PluginDType, TensorData};

// Keep SdkDType as alias for backwards compatibility
pub type SdkDType = PluginDType;

/// Convert hodu_core::DType to PluginDType
pub fn core_dtype_to_plugin(dtype: hodu_core::types::DType) -> PluginDType {
    use hodu_core::types::DType;
    match dtype {
        DType::BOOL => PluginDType::BOOL,
        DType::F8E4M3 => PluginDType::F8E4M3,
        DType::F8E5M2 => PluginDType::F8E5M2,
        DType::BF16 => PluginDType::BF16,
        DType::F16 => PluginDType::F16,
        DType::F32 => PluginDType::F32,
        DType::F64 => PluginDType::F64,
        DType::U8 => PluginDType::U8,
        DType::U16 => PluginDType::U16,
        DType::U32 => PluginDType::U32,
        DType::U64 => PluginDType::U64,
        DType::I8 => PluginDType::I8,
        DType::I16 => PluginDType::I16,
        DType::I32 => PluginDType::I32,
        DType::I64 => PluginDType::I64,
    }
}

/// Convert PluginDType to hodu_core::DType
pub fn plugin_dtype_to_core(dtype: PluginDType) -> hodu_core::types::DType {
    use hodu_core::types::DType;
    match dtype {
        PluginDType::BOOL => DType::BOOL,
        PluginDType::F8E4M3 => DType::F8E4M3,
        PluginDType::F8E5M2 => DType::F8E5M2,
        PluginDType::BF16 => DType::BF16,
        PluginDType::F16 => DType::F16,
        PluginDType::F32 => DType::F32,
        PluginDType::F64 => DType::F64,
        PluginDType::U8 => DType::U8,
        PluginDType::U16 => DType::U16,
        PluginDType::U32 => DType::U32,
        PluginDType::U64 => DType::U64,
        PluginDType::I8 => DType::I8,
        PluginDType::I16 => DType::I16,
        PluginDType::I32 => DType::I32,
        PluginDType::I64 => DType::I64,
        _ => DType::F32, // fallback for future dtypes
    }
}

/// Extension trait for TensorData with hodu_core integration
pub trait TensorDataExt {
    /// Create new tensor data from hodu_core::DType
    fn from_core_dtype(data: Vec<u8>, shape: Vec<usize>, dtype: hodu_core::types::DType) -> TensorData;

    /// Get dtype as hodu_core::DType
    fn core_dtype(&self) -> hodu_core::types::DType;

    /// Load tensor data from an HDT file
    fn load(path: impl AsRef<std::path::Path>) -> Result<TensorData, crate::PluginError>;

    /// Save tensor data to an HDT file
    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), crate::PluginError>;
}

impl TensorDataExt for TensorData {
    fn from_core_dtype(data: Vec<u8>, shape: Vec<usize>, dtype: hodu_core::types::DType) -> TensorData {
        TensorData::new(data, shape, core_dtype_to_plugin(dtype))
    }

    fn core_dtype(&self) -> hodu_core::types::DType {
        plugin_dtype_to_core(self.dtype)
    }

    fn load(path: impl AsRef<std::path::Path>) -> Result<TensorData, crate::PluginError> {
        use crate::hdt;
        let tensor = hdt::load(path).map_err(|e| crate::PluginError::Load(e.to_string()))?;
        let shape: Vec<usize> = tensor.shape().dims().to_vec();
        let dtype: PluginDType = core_dtype_to_plugin(tensor.dtype());
        let data = tensor.to_bytes().map_err(|e| crate::PluginError::Load(e.to_string()))?;
        Ok(TensorData::new(data, shape, dtype))
    }

    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), crate::PluginError> {
        use crate::{hdt, CoreDevice, Shape, Tensor};
        let shape = Shape::new(&self.shape);
        let dtype = self.core_dtype();
        let tensor = Tensor::from_bytes(&self.data, shape, dtype, CoreDevice::CPU)
            .map_err(|e| crate::PluginError::Save(e.to_string()))?;
        hdt::save(&tensor, path).map_err(|e| crate::PluginError::Save(e.to_string()))
    }
}
