//! Dispatch manifest parsing (shared with compiler)

use serde::{Deserialize, Serialize};

/// Dispatch manifest for executing a compiled graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchManifest {
    pub name: Option<String>,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    pub constants: Vec<ConstantData>,
    pub dispatches: Vec<KernelDispatch>,
    pub num_buffers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub buffer_id: usize,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantData {
    pub buffer_id: usize,
    pub shape: Vec<usize>,
    pub dtype: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelDispatch {
    pub kernel_name: String,
    pub input_buffers: Vec<usize>,
    pub output_buffer: usize,
    pub metadata: Vec<usize>,
    pub grid_size: usize,
    pub scalar: Option<ScalarValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalarValue {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
    Bool(bool),
}

impl DispatchManifest {
    pub fn from_json(data: &[u8]) -> Option<Self> {
        serde_json::from_slice(data).ok()
    }
}

/// Get byte size for dtype
pub fn dtype_size(dtype: &str) -> usize {
    match dtype.to_lowercase().as_str() {
        "bool" => 1,
        "u8" | "i8" => 1,
        "bf16" | "f16" | "u16" | "i16" => 2,
        "f32" | "u32" | "i32" => 4,
        "f64" | "u64" | "i64" => 8,
        _ => 4, // default to f32
    }
}
