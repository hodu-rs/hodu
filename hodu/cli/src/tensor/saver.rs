//! Tensor saving utilities

use hodu_core::format::{hdt, json};
use hodu_core::tensor::Tensor;
use hodu_core::types::{DType, Device as CoreDevice, Shape};
use hodu_plugin::{PluginDType, TensorData};
use std::collections::HashMap;
use std::path::Path;

pub fn save_outputs(
    outputs: &HashMap<String, TensorData>,
    save_dir: &Path,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(save_dir)?;

    for (name, data) in outputs {
        let file_path = save_dir.join(format!("{}.{}", name, format.to_lowercase()));

        let dtype: DType = plugin_dtype_to_core(data.dtype);
        let shape = Shape::new(&data.shape);
        let tensor = Tensor::from_bytes(&data.data, shape, dtype, CoreDevice::CPU)
            .map_err(|e| format!("Failed to create tensor: {}", e))?;

        match format.to_lowercase().as_str() {
            "hdt" => hdt::save(&tensor, &file_path)?,
            "json" => json::save(&tensor, &file_path)?,
            _ => return Err(format!("Unsupported save format: {}", format).into()),
        }
    }

    Ok(())
}

fn plugin_dtype_to_core(dtype: PluginDType) -> DType {
    match dtype {
        PluginDType::Bool => DType::BOOL,
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
