//! JSON tensor format support
//!
//! A human-readable format for storing tensors, useful for debugging.
//! Supports single tensor or named tensor collections.
//!
//! Format:
//! ```json
//! {
//!   "shape": [2, 3],
//!   "dtype": "f32",
//!   "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
//! }
//! ```
//!
//! For multiple tensors:
//! ```json
//! {
//!   "input": { "shape": [2, 3], "dtype": "f32", "data": [...] },
//!   "weight": { "shape": [3, 4], "dtype": "f32", "data": [...] }
//! }
//! ```

use crate::compat::*;
use crate::error::{HoduError, HoduResult};
use crate::tensor::Tensor;
use crate::types::{DType, Shape};
use float8::F8E4M3;
#[cfg(feature = "f8e5m2")]
use float8::F8E5M2;
use half::{bf16, f16};
use serde_json::{Map, Value};

/// Load a single tensor from JSON file
#[cfg(feature = "std")]
pub fn load(path: impl AsRef<std::path::Path>) -> HoduResult<Tensor> {
    let data = std::fs::read_to_string(path.as_ref())
        .map_err(|e| HoduError::IoError(format!("Failed to read json file: {}", e)))?;
    deserialize(&data)
}

/// Save a single tensor to JSON file
#[cfg(feature = "std")]
pub fn save(tensor: &Tensor, path: impl AsRef<std::path::Path>) -> HoduResult<()> {
    let data = serialize(tensor)?;
    std::fs::write(path.as_ref(), data).map_err(|e| HoduError::IoError(format!("Failed to write json file: {}", e)))?;
    Ok(())
}

/// Load multiple named tensors from JSON file
#[cfg(feature = "std")]
pub fn load_many(path: impl AsRef<std::path::Path>) -> HoduResult<HashMap<String, Tensor>> {
    let data = std::fs::read_to_string(path.as_ref())
        .map_err(|e| HoduError::IoError(format!("Failed to read json file: {}", e)))?;
    deserialize_many(&data)
}

/// Save multiple named tensors to JSON file
#[cfg(feature = "std")]
pub fn save_many(tensors: &HashMap<String, Tensor>, path: impl AsRef<std::path::Path>) -> HoduResult<()> {
    let data = serialize_many(tensors)?;
    std::fs::write(path.as_ref(), data).map_err(|e| HoduError::IoError(format!("Failed to write json file: {}", e)))?;
    Ok(())
}

/// Serialize a single tensor to JSON string
pub fn serialize(tensor: &Tensor) -> HoduResult<String> {
    let value = tensor_to_json(tensor)?;
    serde_json::to_string_pretty(&value)
        .map_err(|e| HoduError::SerializationFailed(format!("Failed to serialize tensor: {}", e)))
}

/// Deserialize a single tensor from JSON string
pub fn deserialize(data: &str) -> HoduResult<Tensor> {
    let value: Value = serde_json::from_str(data)
        .map_err(|e| HoduError::DeserializationFailed(format!("Failed to parse JSON: {}", e)))?;
    json_to_tensor(&value)
}

/// Serialize multiple named tensors to JSON string
pub fn serialize_many(tensors: &HashMap<String, Tensor>) -> HoduResult<String> {
    let mut map = Map::new();
    for (name, tensor) in tensors {
        map.insert(name.clone(), tensor_to_json(tensor)?);
    }
    serde_json::to_string_pretty(&Value::Object(map))
        .map_err(|e| HoduError::SerializationFailed(format!("Failed to serialize tensors: {}", e)))
}

/// Deserialize multiple named tensors from JSON string
pub fn deserialize_many(data: &str) -> HoduResult<HashMap<String, Tensor>> {
    let value: Value = serde_json::from_str(data)
        .map_err(|e| HoduError::DeserializationFailed(format!("Failed to parse JSON: {}", e)))?;

    let obj = value
        .as_object()
        .ok_or_else(|| HoduError::DeserializationFailed("Expected JSON object".into()))?;

    obj.iter()
        .map(|(name, v)| {
            let tensor = json_to_tensor(v)?;
            Ok((name.clone(), tensor))
        })
        .collect()
}

fn tensor_to_json(tensor: &Tensor) -> HoduResult<Value> {
    let shape: Vec<Value> = tensor.shape().dims().iter().map(|&d| Value::Number(d.into())).collect();
    let dtype_str = dtype_to_str(tensor.dtype());
    let data = tensor_data_to_json(tensor)?;

    let mut map = Map::new();
    map.insert("shape".into(), Value::Array(shape));
    map.insert("dtype".into(), Value::String(dtype_str.into()));
    map.insert("data".into(), data);

    Ok(Value::Object(map))
}

fn json_to_tensor(value: &Value) -> HoduResult<Tensor> {
    let obj = value
        .as_object()
        .ok_or_else(|| HoduError::DeserializationFailed("Expected JSON object for tensor".into()))?;

    let shape_arr = obj
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| HoduError::DeserializationFailed("Missing or invalid 'shape' field".into()))?;

    let shape: Vec<usize> = shape_arr
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| HoduError::DeserializationFailed("Invalid shape dimension".into()))
        })
        .collect::<HoduResult<Vec<_>>>()?;

    let dtype_str = obj
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| HoduError::DeserializationFailed("Missing or invalid 'dtype' field".into()))?;

    let dtype = str_to_dtype(dtype_str)?;

    let data_arr = obj
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| HoduError::DeserializationFailed("Missing or invalid 'data' field".into()))?;

    json_data_to_tensor(data_arr, Shape::new(&shape), dtype)
}

fn dtype_to_str(dtype: DType) -> &'static str {
    match dtype {
        DType::BOOL => "bool",
        DType::F8E4M3 => "f8e4m3",
        #[cfg(feature = "f8e5m2")]
        DType::F8E5M2 => "f8e5m2",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        DType::F32 => "f32",
        #[cfg(feature = "f64")]
        DType::F64 => "f64",
        DType::U8 => "u8",
        #[cfg(feature = "u16")]
        DType::U16 => "u16",
        DType::U32 => "u32",
        #[cfg(feature = "u64")]
        DType::U64 => "u64",
        DType::I8 => "i8",
        #[cfg(feature = "i16")]
        DType::I16 => "i16",
        DType::I32 => "i32",
        #[cfg(feature = "i64")]
        DType::I64 => "i64",
        #[allow(unreachable_patterns)]
        _ => "unknown",
    }
}

fn str_to_dtype(s: &str) -> HoduResult<DType> {
    match s.to_lowercase().as_str() {
        "bool" => Ok(DType::BOOL),
        "f8e4m3" => Ok(DType::F8E4M3),
        #[cfg(feature = "f8e5m2")]
        "f8e5m2" => Ok(DType::F8E5M2),
        "bf16" => Ok(DType::BF16),
        "f16" => Ok(DType::F16),
        "f32" => Ok(DType::F32),
        #[cfg(feature = "f64")]
        "f64" => Ok(DType::F64),
        "u8" => Ok(DType::U8),
        #[cfg(feature = "u16")]
        "u16" => Ok(DType::U16),
        "u32" => Ok(DType::U32),
        #[cfg(feature = "u64")]
        "u64" => Ok(DType::U64),
        "i8" => Ok(DType::I8),
        #[cfg(feature = "i16")]
        "i16" => Ok(DType::I16),
        "i32" => Ok(DType::I32),
        #[cfg(feature = "i64")]
        "i64" => Ok(DType::I64),
        _ => Err(HoduError::DeserializationFailed(format!("Unknown dtype: {}", s))),
    }
}

fn tensor_data_to_json(tensor: &Tensor) -> HoduResult<Value> {
    match tensor.dtype() {
        DType::BOOL => {
            let data: Vec<bool> = tensor.to_flatten_vec()?;
            Ok(Value::Array(data.into_iter().map(Value::Bool).collect()))
        },
        DType::F8E4M3 => {
            let data: Vec<F8E4M3> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter()
                    .map(|v| {
                        serde_json::Number::from_f64(f32::from(v) as f64)
                            .map(Value::Number)
                            .unwrap_or(Value::Null)
                    })
                    .collect(),
            ))
        },
        #[cfg(feature = "f8e5m2")]
        DType::F8E5M2 => {
            let data: Vec<F8E5M2> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter()
                    .map(|v| {
                        serde_json::Number::from_f64(f32::from(v) as f64)
                            .map(Value::Number)
                            .unwrap_or(Value::Null)
                    })
                    .collect(),
            ))
        },
        DType::BF16 => {
            let data: Vec<bf16> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter()
                    .map(|v| {
                        serde_json::Number::from_f64(v.to_f64())
                            .map(Value::Number)
                            .unwrap_or(Value::Null)
                    })
                    .collect(),
            ))
        },
        DType::F16 => {
            let data: Vec<f16> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter()
                    .map(|v| {
                        serde_json::Number::from_f64(v.to_f64())
                            .map(Value::Number)
                            .unwrap_or(Value::Null)
                    })
                    .collect(),
            ))
        },
        DType::F32 => {
            let data: Vec<f32> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter()
                    .map(|v| {
                        serde_json::Number::from_f64(v as f64)
                            .map(Value::Number)
                            .unwrap_or(Value::Null)
                    })
                    .collect(),
            ))
        },
        #[cfg(feature = "f64")]
        DType::F64 => {
            let data: Vec<f64> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter()
                    .map(|v| {
                        serde_json::Number::from_f64(v)
                            .map(Value::Number)
                            .unwrap_or(Value::Null)
                    })
                    .collect(),
            ))
        },
        DType::U8 => {
            let data: Vec<u8> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter().map(|v| Value::Number(v.into())).collect(),
            ))
        },
        #[cfg(feature = "u16")]
        DType::U16 => {
            let data: Vec<u16> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter().map(|v| Value::Number(v.into())).collect(),
            ))
        },
        DType::U32 => {
            let data: Vec<u32> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter().map(|v| Value::Number(v.into())).collect(),
            ))
        },
        #[cfg(feature = "u64")]
        DType::U64 => {
            let data: Vec<u64> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter().map(|v| Value::Number(v.into())).collect(),
            ))
        },
        DType::I8 => {
            let data: Vec<i8> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter().map(|v| Value::Number(v.into())).collect(),
            ))
        },
        #[cfg(feature = "i16")]
        DType::I16 => {
            let data: Vec<i16> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter().map(|v| Value::Number(v.into())).collect(),
            ))
        },
        DType::I32 => {
            let data: Vec<i32> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter().map(|v| Value::Number(v.into())).collect(),
            ))
        },
        #[cfg(feature = "i64")]
        DType::I64 => {
            let data: Vec<i64> = tensor.to_flatten_vec()?;
            Ok(Value::Array(
                data.into_iter().map(|v| Value::Number(v.into())).collect(),
            ))
        },
        #[allow(unreachable_patterns)]
        _ => Err(HoduError::UnsupportedDType {
            dtype: tensor.dtype(),
            reason: "JSON serialization not supported for this dtype".into(),
        }),
    }
}

fn json_data_to_tensor(data: &[Value], shape: Shape, dtype: DType) -> HoduResult<Tensor> {
    match dtype {
        DType::BOOL => {
            let values: Vec<bool> = data
                .iter()
                .map(|v| {
                    v.as_bool()
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected bool".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        DType::F8E4M3 => {
            let values: Vec<F8E4M3> = data
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| F8E4M3::from_f32(f as f32))
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected number".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        #[cfg(feature = "f8e5m2")]
        DType::F8E5M2 => {
            let values: Vec<F8E5M2> = data
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| F8E5M2::from_f32(f as f32))
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected number".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        DType::BF16 => {
            let values: Vec<bf16> = data
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| bf16::from_f64(f))
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected number".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        DType::F16 => {
            let values: Vec<f16> = data
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| f16::from_f64(f))
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected number".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        DType::F32 => {
            let values: Vec<f32> = data
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| f as f32)
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected number".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        #[cfg(feature = "f64")]
        DType::F64 => {
            let values: Vec<f64> = data
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected number".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        DType::U8 => {
            let values: Vec<u8> = data
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|n| n as u8)
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected integer".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        #[cfg(feature = "u16")]
        DType::U16 => {
            let values: Vec<u16> = data
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|n| n as u16)
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected integer".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        DType::U32 => {
            let values: Vec<u32> = data
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|n| n as u32)
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected integer".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        #[cfg(feature = "u64")]
        DType::U64 => {
            let values: Vec<u64> = data
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected integer".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        DType::I8 => {
            let values: Vec<i8> = data
                .iter()
                .map(|v| {
                    v.as_i64()
                        .map(|n| n as i8)
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected integer".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        #[cfg(feature = "i16")]
        DType::I16 => {
            let values: Vec<i16> = data
                .iter()
                .map(|v| {
                    v.as_i64()
                        .map(|n| n as i16)
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected integer".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        DType::I32 => {
            let values: Vec<i32> = data
                .iter()
                .map(|v| {
                    v.as_i64()
                        .map(|n| n as i32)
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected integer".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        #[cfg(feature = "i64")]
        DType::I64 => {
            let values: Vec<i64> = data
                .iter()
                .map(|v| {
                    v.as_i64()
                        .ok_or_else(|| HoduError::DeserializationFailed("Expected integer".into()))
                })
                .collect::<HoduResult<Vec<_>>>()?;
            Tensor::from_slice(values, shape)
        },
        #[allow(unreachable_patterns)]
        _ => Err(HoduError::UnsupportedDType {
            dtype,
            reason: "JSON deserialization not supported for this dtype".into(),
        }),
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_f32() {
        let tensor = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let json = serialize(&tensor).unwrap();
        let restored = deserialize(&json).unwrap();

        assert_eq!(tensor.shape(), restored.shape());
        assert_eq!(tensor.dtype(), restored.dtype());
    }

    #[test]
    fn test_serialize_deserialize_i32() {
        let tensor = Tensor::from_slice(vec![1i32, 2, 3, 4], [2, 2]).unwrap();
        let json = serialize(&tensor).unwrap();
        let restored = deserialize(&json).unwrap();

        assert_eq!(tensor.shape(), restored.shape());
        assert_eq!(tensor.dtype(), restored.dtype());
    }

    #[test]
    fn test_serialize_deserialize_many() {
        let mut tensors = HashMap::new();
        tensors.insert("a".to_string(), Tensor::from_slice(vec![1.0f32, 2.0], [2]).unwrap());
        tensors.insert(
            "b".to_string(),
            Tensor::from_slice(vec![3.0f32, 4.0, 5.0], [3]).unwrap(),
        );

        let json = serialize_many(&tensors).unwrap();
        let restored = deserialize_many(&json).unwrap();

        assert_eq!(tensors.len(), restored.len());
        assert!(restored.contains_key("a"));
        assert!(restored.contains_key("b"));
    }
}
