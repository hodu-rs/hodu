//! Hodu Tensor (.hdt) format support
//!
//! A simple binary format for storing tensors using postcard serialization.
//! Supports single tensor or named tensor collections.

use crate::error::{HoduError, HoduResult};
use crate::tensor::Tensor;
use crate::types::{DType, Device, Shape};
use std::collections::HashMap;

/// Single tensor data for serialization
#[derive(serde::Serialize, serde::Deserialize)]
struct TensorData {
    shape: Vec<usize>,
    dtype: DType,
    data: Vec<u8>,
}

/// Multiple named tensors for serialization
#[derive(serde::Serialize, serde::Deserialize)]
struct TensorCollection {
    tensors: Vec<(String, TensorData)>,
}

/// Load a single tensor from .hdt file
pub fn load(path: impl AsRef<std::path::Path>) -> HoduResult<Tensor> {
    let data =
        std::fs::read(path.as_ref()).map_err(|e| HoduError::IoError(format!("Failed to read hdt file: {}", e)))?;
    deserialize(&data)
}

/// Save a single tensor to .hdt file
pub fn save(tensor: &Tensor, path: impl AsRef<std::path::Path>) -> HoduResult<()> {
    let data = serialize(tensor)?;
    std::fs::write(path.as_ref(), data).map_err(|e| HoduError::IoError(format!("Failed to write hdt file: {}", e)))?;
    Ok(())
}

/// Load multiple named tensors from .hdt file
pub fn load_many(path: impl AsRef<std::path::Path>) -> HoduResult<HashMap<String, Tensor>> {
    let data =
        std::fs::read(path.as_ref()).map_err(|e| HoduError::IoError(format!("Failed to read hdt file: {}", e)))?;
    deserialize_many(&data)
}

/// Save multiple named tensors to .hdt file
pub fn save_many(tensors: &HashMap<String, Tensor>, path: impl AsRef<std::path::Path>) -> HoduResult<()> {
    let data = serialize_many(tensors)?;
    std::fs::write(path.as_ref(), data).map_err(|e| HoduError::IoError(format!("Failed to write hdt file: {}", e)))?;
    Ok(())
}

/// Serialize a single tensor to bytes
pub fn serialize(tensor: &Tensor) -> HoduResult<Vec<u8>> {
    let tensor_data = TensorData {
        shape: tensor.shape().dims().to_vec(),
        dtype: tensor.dtype(),
        data: tensor.to_bytes()?,
    };
    postcard::to_allocvec(&tensor_data)
        .map_err(|e| HoduError::SerializationFailed(format!("Failed to serialize tensor: {}", e)))
}

/// Deserialize a single tensor from bytes
pub fn deserialize(data: &[u8]) -> HoduResult<Tensor> {
    let tensor_data: TensorData = postcard::from_bytes(data)
        .map_err(|e| HoduError::DeserializationFailed(format!("Failed to deserialize tensor: {}", e)))?;

    let shape = Shape::new(&tensor_data.shape);
    Tensor::from_bytes(&tensor_data.data, shape, tensor_data.dtype, Device::CPU)
}

/// Serialize multiple named tensors to bytes
pub fn serialize_many(tensors: &HashMap<String, Tensor>) -> HoduResult<Vec<u8>> {
    let collection = TensorCollection {
        tensors: tensors
            .iter()
            .map(|(name, tensor)| {
                Ok((
                    name.clone(),
                    TensorData {
                        shape: tensor.shape().dims().to_vec(),
                        dtype: tensor.dtype(),
                        data: tensor.to_bytes()?,
                    },
                ))
            })
            .collect::<HoduResult<Vec<_>>>()?,
    };
    postcard::to_allocvec(&collection)
        .map_err(|e| HoduError::SerializationFailed(format!("Failed to serialize tensors: {}", e)))
}

/// Deserialize multiple named tensors from bytes
pub fn deserialize_many(data: &[u8]) -> HoduResult<HashMap<String, Tensor>> {
    let collection: TensorCollection = postcard::from_bytes(data)
        .map_err(|e| HoduError::DeserializationFailed(format!("Failed to deserialize tensors: {}", e)))?;

    collection
        .tensors
        .into_iter()
        .map(|(name, tensor_data)| {
            let shape = Shape::new(&tensor_data.shape);
            let tensor = Tensor::from_bytes(&tensor_data.data, shape, tensor_data.dtype, Device::CPU)?;
            Ok((name, tensor))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_single() {
        let tensor = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let data = serialize(&tensor).unwrap();
        let restored = deserialize(&data).unwrap();

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

        let data = serialize_many(&tensors).unwrap();
        let restored = deserialize_many(&data).unwrap();

        assert_eq!(tensors.len(), restored.len());
        assert!(restored.contains_key("a"));
        assert!(restored.contains_key("b"));
    }
}
