use crate::{
    be::storage::BackendStorage,
    compat::*,
    ops::Op,
    script::builder::ir::*,
    tensor::TensorId,
    types::{Compiler, DType, Device, Layout},
};

/// Compilation options
#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub optimization_level: OptimizationLevel,
    pub device: Device,
    pub memory_limit: Option<usize>,
    pub enable_profiling: bool,
    pub debug_mode: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Basic,
            device: Device::CPU,
            memory_limit: None,
            enable_profiling: false,
            debug_mode: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
}

/// Compiled module ready for execution
#[derive(Debug, Clone)]
pub struct CompiledModule {
    pub module: Module,
    pub execution_plan: Vec<CompiledInstruction>,
    pub input_mapping: HashMap<String, ValueId>,
    pub output_mapping: HashMap<String, ValueId>,
    pub constant_data: HashMap<TensorId, ConstantData>,
    /// Pre-converted constants on target device (cached for performance)
    pub constant_storages: HashMap<TensorId, Arc<BackendStorage>>,
    pub value_layouts: HashMap<ValueId, Layout>,
    pub value_dtypes: HashMap<ValueId, DType>,
    pub value_to_tensor: HashMap<ValueId, TensorId>,
    pub compiler: Compiler,
    pub device: Device,
    /// Maximum ValueId used in this module (cached for efficient storage allocation)
    pub max_value_id: usize,
}

#[derive(Debug, Clone)]
pub struct CompiledInstruction {
    pub op: Op,
    pub inputs: Vec<ValueId>,
    pub result: ValueId,
    pub attributes: HashMap<String, Attribute>,
}
