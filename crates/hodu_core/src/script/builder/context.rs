use super::ir::*;
use crate::{
    error::HoduResult,
    layer::compat::*,
    tensor::{Tensor, TensorId},
};

mod builder;
mod operations;
mod storage;

pub use builder::Builder;
pub use storage::{get_active_builder, is_builder_active, with_active_builder};

/// Unique identifier for a builder instance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BuilderId(u32);

impl BuilderId {
    pub(crate) fn new() -> Self {
        static BUILDER_ID_COUNTER: AtomicU32 = AtomicU32::new(0);
        Self(BUILDER_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Builder state containing the IR module and tracking information
pub struct BuilderState {
    pub name: String,
    pub module: Module,
    pub current_function: Option<String>,
    pub current_block: Option<BlockId>,
    pub value_counter: u32,
    pub block_counter: u32,
    pub tensor_to_value: HashMap<TensorId, ValueId>,
    pub graph_inputs: Vec<(&'static str, Tensor)>,
    pub graph_outputs: Vec<(&'static str, Tensor)>,
    pub intermediate_tensors: Vec<Tensor>,
    pub is_ended: bool,
}

impl BuilderState {
    /// Create a new builder state with the given name
    pub(crate) fn new(name: String) -> Self {
        let module = Module::new(name.clone());
        Self {
            name,
            module,
            current_function: None,
            current_block: None,
            value_counter: 0,
            block_counter: 0,
            tensor_to_value: HashMap::new(),
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
            intermediate_tensors: Vec::new(),
            is_ended: false,
        }
    }

    /// Ensure a default function and block exist, creating them if necessary
    pub(crate) fn ensure_function_and_block(&mut self) -> HoduResult<()> {
        if self.current_function.is_none() {
            let fn_name = format!("{}_main", self.name);
            let entry_block_id = BlockId(self.block_counter);
            self.block_counter += 1;

            let signature = FunctionSignature::new(Vec::new(), Vec::new());
            let function = Function::new(fn_name.clone(), signature, entry_block_id);

            self.module.add_function(function);
            self.current_function = Some(fn_name.clone());

            // Create entry block
            let block = BasicBlock::new(entry_block_id);

            if let Some(function) = self.module.get_function_mut(&fn_name) {
                function.add_block(block);
            }
            self.current_block = Some(entry_block_id);
        }
        Ok(())
    }
}
