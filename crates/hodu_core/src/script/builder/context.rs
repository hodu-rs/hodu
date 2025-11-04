use super::ir::*;
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    tensor::{Tensor, TensorId},
    types::Layout,
};

#[cfg(feature = "std")]
use dashmap::DashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BuilderId(u32);

impl BuilderId {
    pub(crate) fn new() -> Self {
        static BUILDER_ID_COUNTER: AtomicU32 = AtomicU32::new(0);
        Self(BUILDER_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Builder state
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
    pub is_ended: bool,
}

#[cfg(feature = "std")]
static BUILDERS: LazyLock<DashMap<BuilderId, BuilderState>> =
    LazyLock::new(|| DashMap::with_capacity_and_shard_amount(1 << 8, 16));

#[cfg(not(feature = "std"))]
static BUILDERS: LazyLock<Mutex<HashMap<BuilderId, BuilderState>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

static ACTIVE_BUILDER_ID: Mutex<Option<BuilderId>> = Mutex::new(None);

/// Builder handle
pub struct Builder(BuilderId);

impl Builder {
    pub fn new(name: String) -> Self {
        let module = Module::new(name.clone());
        let state = BuilderState {
            name,
            module,
            current_function: None,
            current_block: None,
            value_counter: 0,
            block_counter: 0,
            tensor_to_value: HashMap::new(),
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
            is_ended: false,
        };

        let builder_id = BuilderId::new();
        insert(builder_id, state);
        Builder(builder_id)
    }

    #[cfg(feature = "std")]
    pub fn with_state<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&BuilderState) -> R,
    {
        let state = BUILDERS.get(&self.0)?;
        Some(f(&state))
    }

    #[cfg(not(feature = "std"))]
    pub fn with_state<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&BuilderState) -> R,
    {
        let builders = BUILDERS.lock();
        let state = builders.get(&self.0)?;
        Some(f(state))
    }

    #[cfg(feature = "std")]
    pub fn with_state_mut<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut BuilderState) -> R,
    {
        let mut state = BUILDERS.get_mut(&self.0)?;
        Some(f(&mut state))
    }

    #[cfg(not(feature = "std"))]
    pub fn with_state_mut<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut BuilderState) -> R,
    {
        let mut builders = BUILDERS.lock();
        let state = builders.get_mut(&self.0)?;
        Some(f(state))
    }

    pub fn get_name(&self) -> String {
        self.with_state(|s| s.name.clone())
            .unwrap_or_else(|| format!("Builder({})", self.0 .0))
    }

    pub fn set_name(&self, name: String) -> HoduResult<()> {
        self.with_state_mut(|s| {
            s.name = name.clone();
            s.module.name = name;
        })
        .ok_or_else(|| HoduError::InternalError(format!("Builder {} not found", self.0 .0)))
    }

    pub fn add_input(&self, name: &'static str, tensor: Tensor) -> HoduResult<()> {
        self.with_state_mut(|s| s.graph_inputs.push((name, tensor)))
            .ok_or_else(|| HoduError::InternalError(format!("Builder {} not found", self.0 .0)))
    }

    pub fn add_output(&self, name: &'static str, tensor: Tensor) -> HoduResult<()> {
        self.with_state_mut(|s| s.graph_outputs.push((name, tensor)))
            .ok_or_else(|| HoduError::InternalError(format!("Builder {} not found", self.0 .0)))
    }

    pub fn set_outputs(&self, names: &[&'static str], tensors: &[Tensor]) -> HoduResult<()> {
        if names.len() != tensors.len() {
            return Err(HoduError::InternalError(format!(
                "Names length ({}) must match tensors length ({})",
                names.len(),
                tensors.len()
            )));
        }
        let outputs: Vec<(&'static str, Tensor)> = names
            .iter()
            .zip(tensors.iter())
            .map(|(&name, tensor)| (name, *tensor))
            .collect();
        self.with_state_mut(|s| s.graph_outputs = outputs)
            .ok_or_else(|| HoduError::InternalError(format!("Builder {} not found", self.0 .0)))
    }

    pub fn add_operation(
        &self,
        op: Op,
        op_params: Option<crate::ops::OpParams>,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
        _input_layouts: Vec<Layout>,
        _output_layouts: Vec<Layout>,
    ) -> HoduResult<()> {
        self.with_state_mut(|s| {
            // Get or create default function and block
            if s.current_function.is_none() {
                let fn_name = format!("{}_main", s.name);
                let entry_block_id = BlockId(s.block_counter);
                s.block_counter += 1;

                let signature = FunctionSignature::new(Vec::new(), Vec::new());
                let function = Function::new(fn_name.clone(), signature, entry_block_id);

                s.module.add_function(function);
                s.current_function = Some(fn_name.clone());

                // Create entry block
                let block = BasicBlock::new(entry_block_id);

                if let Some(function) = s.module.get_function_mut(&fn_name) {
                    function.add_block(block);
                }
                s.current_block = Some(entry_block_id);
            }

            // Map input TensorIds to ValueIds
            // If input is a constant, load it first
            let input_values: Vec<ValueId> = inputs
                .iter()
                .map(|&tensor_id| {
                    // Check if already mapped
                    if let Some(&value_id) = s.tensor_to_value.get(&tensor_id) {
                        return value_id;
                    }

                    // Check if this is a constant that needs to be loaded
                    if let Some(_) = crate::tensor::get(tensor_id) {
                        let tensor = crate::tensor::tensor_from_id(tensor_id);
                        if tensor.has_storage() && !tensor.is_runtime() {
                            // This is a constant - add LoadConstant instruction
                            let value_id = ValueId(s.value_counter);
                            s.value_counter += 1;
                            s.tensor_to_value.insert(tensor_id, value_id);

                            // Add constant data to module if not already present
                            if !s.module.constants.contains_key(&tensor_id) {
                                let layout = tensor.layout();
                                let cpu_storage = tensor.with_storage(|storage| storage.to_cpu_storage()).ok();

                                if let Some(cpu_storage) = cpu_storage {
                                    let constant = ConstantData {
                                        tensor_id,
                                        shape: layout.shape().clone(),
                                        dtype: tensor.dtype(),
                                        data: cpu_storage.to_bytes(),
                                        compression: None,
                                    };
                                    s.module.add_constant(tensor_id, constant);
                                }
                            }

                            // Add LoadConstant instruction
                            if let (Some(fn_name), Some(block_id)) = (&s.current_function, &s.current_block) {
                                if let Some(function) = s.module.get_function_mut(fn_name) {
                                    if let Some(block) = function.get_block_mut(*block_id) {
                                        block.add_instruction(Instruction::LoadConstant {
                                            result: value_id,
                                            tensor_id,
                                        });
                                    }
                                }
                            }

                            return value_id;
                        }
                    }

                    // Not a constant, just allocate value_id
                    let value_id = ValueId(s.value_counter);
                    s.value_counter += 1;
                    s.tensor_to_value.insert(tensor_id, value_id);
                    value_id
                })
                .collect();

            // Create ValueId for result (single output)
            let result_value = ValueId(s.value_counter);
            s.value_counter += 1;

            // Store output tensor to value mapping (for first output)
            if let Some(&first_output) = outputs.first() {
                s.tensor_to_value.insert(first_output, result_value);
            }

            // Convert OpParams to attributes
            let mut attributes = HashMap::new();
            if let Some(params) = op_params {
                if let Some(scalar) = params.scalar {
                    attributes.insert("scalar".to_string(), Attribute::Scalar(scalar));
                }
                if !params.scalars.is_empty() {
                    attributes.insert("scalars".to_string(), Attribute::Scalars(params.scalars.clone()));
                }
                if !params.dims.is_empty() {
                    attributes.insert("dims".to_string(), Attribute::Scalars(params.dims.clone()));
                }
                if let Some(keep_dim) = params.keep_dim {
                    attributes.insert("keep_dim".to_string(), Attribute::Bool(keep_dim));
                }
                if let Some(output_index) = params.output_index {
                    attributes.insert("output_index".to_string(), Attribute::Int(output_index as i32));
                }
                if let Some(dtype) = params.dtype {
                    attributes.insert("dtype".to_string(), Attribute::DType(dtype));
                }
            }

            // Create instruction
            let instruction = Instruction::Compute {
                result: result_value,
                op,
                inputs: input_values,
                attributes,
            };

            // Add instruction to current block
            if let (Some(fn_name), Some(block_id)) = (&s.current_function, &s.current_block) {
                if let Some(function) = s.module.get_function_mut(fn_name) {
                    if let Some(block) = function.get_block_mut(*block_id) {
                        block.add_instruction(instruction);
                    }
                }
            }
        })
        .ok_or_else(|| HoduError::InternalError(format!("Builder {} not found", self.0 .0)))
    }

    pub fn start(&self) -> HoduResult<()> {
        let is_ended = self
            .with_state(|s| s.is_ended)
            .ok_or_else(|| HoduError::InternalError(format!("Builder {} not found", self.0 .0)))?;
        let name = self.get_name();

        if is_ended {
            return Err(HoduError::InternalError(format!("Builder '{}' already ended", name)));
        }

        let mut active_id = {
            #[cfg(feature = "std")]
            {
                ACTIVE_BUILDER_ID.lock().unwrap()
            }
            #[cfg(not(feature = "std"))]
            {
                ACTIVE_BUILDER_ID.lock()
            }
        };

        if let Some(active_builder_id) = *active_id {
            let active_name = get(active_builder_id).get_name();
            return Err(HoduError::InternalError(format!(
                "Builder context '{}' already active",
                active_name
            )));
        }

        *active_id = Some(self.0);
        Ok(())
    }

    pub fn end(&self) -> HoduResult<()> {
        let mut active_id = {
            #[cfg(feature = "std")]
            {
                ACTIVE_BUILDER_ID.lock().unwrap()
            }
            #[cfg(not(feature = "std"))]
            {
                ACTIVE_BUILDER_ID.lock()
            }
        };
        let name = self.get_name();

        match active_id.as_ref() {
            None => Err(HoduError::InternalError("No active builder context".to_string())),
            Some(active_builder_id) if active_builder_id != &self.0 => {
                let active_name = get(*active_builder_id).get_name();
                Err(HoduError::InternalError(format!(
                    "Context mismatch: trying to end '{}' but active context is '{}'",
                    name, active_name
                )))
            },
            Some(_) => {
                self.with_state_mut(|s| s.is_ended = true)
                    .ok_or_else(|| HoduError::InternalError(format!("Builder {} not found", self.0 .0)))?;
                *active_id = None;
                Ok(())
            },
        }
    }

    pub fn build(&self) -> HoduResult<Module> {
        self.with_state_mut(|s| super::codegen::build_module(s))
            .ok_or_else(|| HoduError::InternalError(format!("Builder {} not found", self.0 .0)))?
    }
}

#[cfg(feature = "std")]
fn insert(builder_id: BuilderId, state: BuilderState) {
    BUILDERS.insert(builder_id, state);
}

#[cfg(not(feature = "std"))]
fn insert(builder_id: BuilderId, state: BuilderState) {
    let mut builders = BUILDERS.lock();
    builders.insert(builder_id, state);
}

pub fn get(builder_id: BuilderId) -> Builder {
    Builder(builder_id)
}

pub fn get_active_builder() -> HoduResult<Builder> {
    let active_id = {
        #[cfg(feature = "std")]
        {
            ACTIVE_BUILDER_ID.lock().unwrap()
        }
        #[cfg(not(feature = "std"))]
        {
            ACTIVE_BUILDER_ID.lock()
        }
    };
    match active_id.as_ref() {
        Some(builder_id) => Ok(get(*builder_id)),
        None => Err(HoduError::InternalError("No active builder context".to_string())),
    }
}

pub fn with_active_builder<F, R>(f: F) -> HoduResult<R>
where
    F: FnOnce(&mut BuilderState) -> R,
{
    let active_id = {
        #[cfg(feature = "std")]
        {
            ACTIVE_BUILDER_ID.lock().unwrap()
        }
        #[cfg(not(feature = "std"))]
        {
            ACTIVE_BUILDER_ID.lock()
        }
    };
    match active_id.as_ref() {
        Some(builder_id) => {
            #[cfg(feature = "std")]
            {
                match BUILDERS.get_mut(builder_id) {
                    Some(mut state) => Ok(f(&mut state)),
                    None => Err(HoduError::InternalError(format!(
                        "Active builder '{:?}' not found in storage",
                        builder_id
                    ))),
                }
            }
            #[cfg(not(feature = "std"))]
            {
                let mut builders = BUILDERS.lock();
                match builders.get_mut(builder_id) {
                    Some(state) => Ok(f(state)),
                    None => Err(HoduError::InternalError(format!(
                        "Active builder '{:?}' not found in storage",
                        builder_id
                    ))),
                }
            }
        },
        None => Err(HoduError::InternalError("No active builder context".to_string())),
    }
}

pub fn is_builder_active() -> bool {
    let active_id = {
        #[cfg(feature = "std")]
        {
            ACTIVE_BUILDER_ID.lock().unwrap()
        }
        #[cfg(not(feature = "std"))]
        {
            ACTIVE_BUILDER_ID.lock()
        }
    };
    active_id.is_some()
}
