//! Tensor data types for cross-plugin communication

use hodu_core::types::DType;

/// Raw tensor data for cross-plugin communication
///
/// This struct is used to pass tensor data between the CLI and plugins
/// without depending on the full Tensor type and registry.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Raw bytes of tensor data
    pub data: Vec<u8>,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DType,
}

impl TensorData {
    /// Create new tensor data
    pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Self {
        Self { data, shape, dtype }
    }

    /// Number of elements in the tensor
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size of data in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor data is valid (size matches shape * dtype)
    pub fn is_valid(&self) -> bool {
        let expected_size = self.numel() * self.dtype.get_size_in_bytes();
        self.data.len() == expected_size
    }

    /// Get tensor rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Check if tensor is scalar (rank 0)
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }
}
