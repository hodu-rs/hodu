//! Tensor loading utilities

use hodu_plugin_sdk::{SdkDType, TensorData};
use std::path::PathBuf;

pub fn load_tensor_file(
    path: &PathBuf,
    expected_shape: &[usize],
    expected_dtype: SdkDType,
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
    expected_dtype: SdkDType,
) -> Result<TensorData, Box<dyn std::error::Error>> {
    use hodu_plugin_sdk::hdt;

    let tensor = hdt::load(path).map_err(|e| format!("Failed to load HDT file: {}", e))?;

    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let dtype: SdkDType = tensor.dtype().into();

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
    expected_dtype: SdkDType,
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
    let dtype = str_to_sdk_dtype(dtype_str)?;

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

pub fn str_to_sdk_dtype(s: &str) -> Result<SdkDType, Box<dyn std::error::Error>> {
    match s.to_lowercase().as_str() {
        "bool" => Ok(SdkDType::Bool),
        "f8e4m3" => Ok(SdkDType::F8E4M3),
        "f8e5m2" => Ok(SdkDType::F8E5M2),
        "bf16" => Ok(SdkDType::BF16),
        "f16" => Ok(SdkDType::F16),
        "f32" => Ok(SdkDType::F32),
        "f64" => Ok(SdkDType::F64),
        "u8" => Ok(SdkDType::U8),
        "u16" => Ok(SdkDType::U16),
        "u32" => Ok(SdkDType::U32),
        "u64" => Ok(SdkDType::U64),
        "i8" => Ok(SdkDType::I8),
        "i16" => Ok(SdkDType::I16),
        "i32" => Ok(SdkDType::I32),
        "i64" => Ok(SdkDType::I64),
        _ => Err(format!("Unknown dtype: {}", s).into()),
    }
}

fn json_array_to_bytes(arr: &[serde_json::Value], dtype: SdkDType) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(arr.len() * dtype.size_in_bytes());

    match dtype {
        SdkDType::F32 => {
            for v in arr {
                let f = v.as_f64().ok_or("Expected number")? as f32;
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        },
        SdkDType::F64 => {
            for v in arr {
                let f = v.as_f64().ok_or("Expected number")?;
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        },
        SdkDType::I32 => {
            for v in arr {
                let n = v.as_i64().ok_or("Expected integer")? as i32;
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        },
        SdkDType::I64 => {
            for v in arr {
                let n = v.as_i64().ok_or("Expected integer")?;
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        },
        _ => return Err(format!("Unsupported dtype for JSON input: {}", dtype.name()).into()),
    }

    Ok(bytes)
}
