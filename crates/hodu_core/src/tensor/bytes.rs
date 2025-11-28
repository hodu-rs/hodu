use crate::{
    be::storage::BackendStorage,
    be_cpu::storage::CpuStorage,
    compat::*,
    error::{HoduError, HoduResult},
    tensor::{internal::from_storage_with_context, Tensor},
    types::{DType, Device, Layout, Shape},
};

#[cfg(feature = "std")]
use std::path::Path;

impl Tensor {
    /// Convert tensor to raw bytes (little-endian)
    ///
    /// Works with tensors on any device (CPU, CUDA, Metal).
    /// GPU tensors are automatically transferred to CPU for serialization.
    pub fn to_bytes(&self) -> HoduResult<Vec<u8>> {
        let cpu_storage = self.with_storage(|storage| storage.to_cpu_storage())?;
        let layout = self.layout();

        // Make contiguous if needed
        let cpu_storage = if !layout.is_contiguous() {
            let contiguous = self.contiguous()?;
            contiguous.with_storage(|storage| storage.to_cpu_storage())?
        } else {
            cpu_storage
        };

        Ok(storage_to_bytes(&cpu_storage))
    }

    /// Create tensor from raw bytes on specified device
    ///
    /// Data is first loaded to CPU, then transferred to the target device if needed.
    pub fn from_bytes(data: &[u8], shape: impl Into<Shape>, dtype: DType, device: Device) -> HoduResult<Self> {
        let shape = shape.into();
        let expected_size = shape.size() * dtype.get_size_in_bytes();

        if data.len() != expected_size {
            return Err(HoduError::InvalidArgument(
                format!(
                    "Data size mismatch: expected {} bytes for shape {:?} with dtype {:?}, got {} bytes",
                    expected_size,
                    shape.dims(),
                    dtype,
                    data.len()
                )
                .into(),
            ));
        }

        let cpu_storage = bytes_to_storage(data, dtype)?;
        let layout = Layout::from_shape(&shape);
        let storage = BackendStorage::CPU(cpu_storage);
        let tensor = from_storage_with_context(storage, layout, true, false);

        // Transfer to target device if not CPU
        match device {
            Device::CPU => Ok(tensor),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => tensor.to_device(device),
            #[cfg(feature = "metal")]
            Device::Metal => tensor.to_device(device),
            #[allow(unreachable_patterns)]
            _ => Ok(tensor),
        }
    }

    /// Save tensor to .hdt file
    #[cfg(feature = "std")]
    pub fn save(&self, path: impl AsRef<Path>) -> HoduResult<()> {
        use postcard;

        #[derive(serde::Serialize)]
        struct TensorData {
            shape: Vec<usize>,
            dtype: DType,
            data: Vec<u8>,
        }

        let tensor_data = TensorData {
            shape: self.shape().dims().to_vec(),
            dtype: self.dtype(),
            data: self.to_bytes()?,
        };

        let data = postcard::to_allocvec(&tensor_data)
            .map_err(|e| HoduError::SerializationFailed(format!("Failed to serialize tensor: {}", e)))?;

        std::fs::write(path.as_ref(), data)
            .map_err(|e| HoduError::IoError(format!("Failed to write hdt file: {}", e)))?;

        Ok(())
    }

    /// Load tensor from .hdt file
    #[cfg(feature = "std")]
    pub fn load(path: impl AsRef<Path>) -> HoduResult<Self> {
        use postcard;

        #[derive(serde::Deserialize)]
        struct TensorData {
            shape: Vec<usize>,
            dtype: DType,
            data: Vec<u8>,
        }

        let data =
            std::fs::read(path.as_ref()).map_err(|e| HoduError::IoError(format!("Failed to read hdt file: {}", e)))?;

        let tensor_data: TensorData = postcard::from_bytes(&data)
            .map_err(|e| HoduError::DeserializationFailed(format!("Failed to deserialize tensor: {}", e)))?;

        let shape = Shape::new(&tensor_data.shape);
        Self::from_bytes(&tensor_data.data, shape, tensor_data.dtype, Device::CPU)
    }
}

fn storage_to_bytes(storage: &CpuStorage) -> Vec<u8> {
    match storage {
        CpuStorage::BOOL(data) => data.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect(),
        CpuStorage::F8E4M3(data) => data.iter().map(|v| v.to_bits()).collect(),
        #[cfg(feature = "f8e5m2")]
        CpuStorage::F8E5M2(data) => data.iter().map(|v| v.to_bits()).collect(),
        CpuStorage::BF16(data) => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        CpuStorage::F16(data) => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        CpuStorage::F32(data) => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        #[cfg(feature = "f64")]
        CpuStorage::F64(data) => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        CpuStorage::U8(data) => data.to_vec(),
        #[cfg(feature = "u16")]
        CpuStorage::U16(data) => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        CpuStorage::U32(data) => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        #[cfg(feature = "u64")]
        CpuStorage::U64(data) => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        CpuStorage::I8(data) => data.iter().map(|&v| v as u8).collect(),
        #[cfg(feature = "i16")]
        CpuStorage::I16(data) => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        CpuStorage::I32(data) => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        #[cfg(feature = "i64")]
        CpuStorage::I64(data) => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn bytes_to_storage(data: &[u8], dtype: DType) -> HoduResult<CpuStorage> {
    use float8::F8E4M3;
    #[cfg(feature = "f8e5m2")]
    use float8::F8E5M2;
    use half::{bf16, f16};

    match dtype {
        DType::BOOL => Ok(CpuStorage::BOOL(data.iter().map(|&b| b != 0).collect())),
        DType::F8E4M3 => Ok(CpuStorage::F8E4M3(data.iter().map(|&b| F8E4M3::from_bits(b)).collect())),
        #[cfg(feature = "f8e5m2")]
        DType::F8E5M2 => Ok(CpuStorage::F8E5M2(data.iter().map(|&b| F8E5M2::from_bits(b)).collect())),
        DType::BF16 => Ok(CpuStorage::BF16(
            data.chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]))
                .collect(),
        )),
        DType::F16 => Ok(CpuStorage::F16(
            data.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]])).collect(),
        )),
        DType::F32 => Ok(CpuStorage::F32(
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )),
        #[cfg(feature = "f64")]
        DType::F64 => Ok(CpuStorage::F64(
            data.chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
        )),
        DType::U8 => Ok(CpuStorage::U8(data.to_vec())),
        #[cfg(feature = "u16")]
        DType::U16 => Ok(CpuStorage::U16(
            data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
        )),
        DType::U32 => Ok(CpuStorage::U32(
            data.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )),
        #[cfg(feature = "u64")]
        DType::U64 => Ok(CpuStorage::U64(
            data.chunks_exact(8)
                .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
        )),
        DType::I8 => Ok(CpuStorage::I8(data.iter().map(|&b| b as i8).collect())),
        #[cfg(feature = "i16")]
        DType::I16 => Ok(CpuStorage::I16(
            data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect(),
        )),
        DType::I32 => Ok(CpuStorage::I32(
            data.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )),
        #[cfg(feature = "i64")]
        DType::I64 => Ok(CpuStorage::I64(
            data.chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
        )),
        #[allow(unreachable_patterns)]
        _ => Err(HoduError::UnsupportedDType {
            dtype,
            reason: "Byte conversion not supported".into(),
        }),
    }
}
