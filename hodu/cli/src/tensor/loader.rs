//! Tensor loading utilities

use hodu_core::format::hdt;
use hodu_plugin::{PluginDType, TensorData};
use std::path::PathBuf;

pub fn load_tensor_file(
    path: &PathBuf,
    expected_shape: &[usize],
    expected_dtype: PluginDType,
) -> Result<TensorData, Box<dyn std::error::Error>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "hdt" => load_tensor_hdt(path, expected_shape, expected_dtype),
        "json" => load_tensor_json(path, expected_shape, expected_dtype),
        _ => Err(format!("Unsupported tensor format: .{}\nSupported: .hdt, .json", ext).into()),
    }
}

fn load_tensor_hdt(
    path: &PathBuf,
    expected_shape: &[usize],
    expected_dtype: PluginDType,
) -> Result<TensorData, Box<dyn std::error::Error>> {
    let tensor = hdt::load(path).map_err(|e| format!("Failed to load HDT file: {}", e))?;

    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let dtype: PluginDType = core_dtype_to_plugin(tensor.dtype());

    if shape != expected_shape {
        return Err(format!(
            "Shape mismatch: file has {:?}, model expects {:?}",
            shape, expected_shape
        )
        .into());
    }

    if dtype != expected_dtype {
        return Err(format!(
            "DType mismatch: file has {}, model expects {}",
            dtype.name(),
            expected_dtype.name()
        )
        .into());
    }

    let data = tensor
        .to_bytes()
        .map_err(|e| format!("Failed to get tensor bytes: {}", e))?;

    Ok(TensorData::new(data, shape, dtype))
}

fn load_tensor_json(
    path: &PathBuf,
    expected_shape: &[usize],
    expected_dtype: PluginDType,
) -> Result<TensorData, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let obj = json
        .as_object()
        .ok_or("JSON must be an object with shape, dtype, data")?;

    let shape: Vec<usize> = obj
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or("Missing 'shape' field")?
        .iter()
        .map(|v| v.as_u64().map(|n| n as usize).ok_or("Invalid shape dimension"))
        .collect::<Result<Vec<_>, _>>()?;

    if shape != expected_shape {
        return Err(format!(
            "Shape mismatch: file has {:?}, model expects {:?}",
            shape, expected_shape
        )
        .into());
    }

    let dtype_str = obj
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'dtype' field")?;
    let dtype = str_to_plugin_dtype(dtype_str)?;

    if dtype != expected_dtype {
        return Err(format!(
            "DType mismatch: file has {}, model expects {}",
            dtype.name(),
            expected_dtype.name()
        )
        .into());
    }

    let data_arr = obj
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or("Missing 'data' field")?;

    let data = json_array_to_bytes(data_arr, dtype)?;

    Ok(TensorData::new(data, shape, dtype))
}

pub fn str_to_plugin_dtype(s: &str) -> Result<PluginDType, Box<dyn std::error::Error>> {
    s.parse::<PluginDType>().map_err(|e| e.into())
}

fn core_dtype_to_plugin(dtype: hodu_core::types::DType) -> PluginDType {
    use hodu_core::types::DType;
    match dtype {
        DType::BOOL => PluginDType::Bool,
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

fn json_array_to_bytes(arr: &[serde_json::Value], dtype: PluginDType) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(arr.len() * dtype.size_in_bytes());

    match dtype {
        PluginDType::F32 => {
            for v in arr {
                let f = v.as_f64().ok_or("Expected number")? as f32;
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        },
        PluginDType::F64 => {
            for v in arr {
                let f = v.as_f64().ok_or("Expected number")?;
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        },
        PluginDType::I32 => {
            for v in arr {
                let n = v.as_i64().ok_or("Expected integer")? as i32;
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        },
        PluginDType::I64 => {
            for v in arr {
                let n = v.as_i64().ok_or("Expected integer")?;
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        },
        _ => return Err(format!("Unsupported dtype for JSON input: {}", dtype.name()).into()),
    }

    Ok(bytes)
}
