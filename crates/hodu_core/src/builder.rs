pub use crate::types::layout::Layout;
use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    tensor::{Tensor, TensorId},
    {op::Op, script::Script},
};
#[cfg(feature = "std")]
use dashmap::DashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BuilderId(usize);

impl BuilderId {
    pub(crate) fn new() -> Self {
        static BUILDER_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(BUILDER_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Clone)]
pub struct Builder_ {
    pub name: String,
    pub graph_inputs: Vec<(&'static str, Tensor)>,
    pub graph_outputs: Vec<(&'static str, Tensor)>,
    pub operations: Vec<Op>,
    pub operation_outputs: Vec<Vec<TensorId>>,
    pub operation_layouts: Vec<(Vec<Layout>, Vec<Layout>)>,
    pub is_ended: bool,
}

#[cfg(feature = "std")]
static BUILDERS: LazyLock<DashMap<BuilderId, Builder_>> =
    LazyLock::new(|| DashMap::with_capacity_and_shard_amount(1 << 8, 16));

#[cfg(not(feature = "std"))]
static BUILDERS: LazyLock<Mutex<HashMap<BuilderId, Builder_>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static ACTIVE_BUILDER_ID: Mutex<Option<BuilderId>> = Mutex::new(None);

pub struct Builder(BuilderId);

#[cfg(feature = "std")]
pub fn insert(builder_id: BuilderId, builder_: Builder_) {
    BUILDERS.insert(builder_id, builder_);
}

#[cfg(not(feature = "std"))]
pub fn insert(builder_id: BuilderId, builder_: Builder_) {
    let mut builders = BUILDERS.lock();
    builders.insert(builder_id, builder_);
}

pub fn get(builder_id: BuilderId) -> Builder {
    Builder(builder_id)
}

impl Builder {
    pub fn new(name: String) -> Self {
        let builder_ = Builder_ {
            name,
            graph_inputs: vec![],
            graph_outputs: vec![],
            operations: vec![],
            operation_outputs: vec![],
            operation_layouts: vec![],
            is_ended: false,
        };

        let builder_id = BuilderId::new();
        insert(builder_id, builder_);
        Builder(builder_id)
    }

    #[cfg(feature = "std")]
    pub fn with_builder<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&Builder_) -> R,
    {
        let builder_ = BUILDERS.get(&self.0)?;
        Some(f(&builder_))
    }

    #[cfg(not(feature = "std"))]
    pub fn with_builder<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&Builder_) -> R,
    {
        let builders = BUILDERS.lock();
        let builder_ = builders.get(&self.0)?;
        Some(f(builder_))
    }

    #[cfg(feature = "std")]
    pub fn with_builder_mut<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut Builder_) -> R,
    {
        let mut builder_ = BUILDERS.get_mut(&self.0)?;
        Some(f(&mut builder_))
    }

    #[cfg(not(feature = "std"))]
    pub fn with_builder_mut<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut Builder_) -> R,
    {
        let mut builders = BUILDERS.lock();
        let builder_ = builders.get_mut(&self.0)?;
        Some(f(builder_))
    }

    pub fn get_name(&self) -> String {
        self.with_builder(|b| b.name.clone())
            .unwrap_or_else(|| format!("Builder({})", self.0 .0))
    }

    pub fn set_name(&self, name: String) -> HoduResult<()> {
        self.with_builder_mut(|b| b.name = name)
            .ok_or(HoduError::BuilderValidationFailed(format!(
                "Builder {} not found",
                self.0 .0
            )))
    }

    pub fn set_outputs(&self, names: &[&'static str], tensors: &[Tensor]) -> HoduResult<()> {
        if names.len() != tensors.len() {
            return Err(HoduError::BuilderValidationFailed(format!(
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
        self.with_builder_mut(|b| b.graph_outputs = outputs)
            .ok_or(HoduError::BuilderValidationFailed(format!(
                "Builder {} not found",
                self.0 .0
            )))
    }

    pub fn add_input(&self, name: &'static str, tensor: Tensor) -> HoduResult<()> {
        self.with_builder_mut(|b| b.graph_inputs.push((name, tensor)))
            .ok_or(HoduError::BuilderValidationFailed(format!(
                "Builder {} not found",
                self.0 .0
            )))
    }

    pub fn add_output(&self, name: &'static str, tensor: Tensor) -> HoduResult<()> {
        self.with_builder_mut(|b| b.graph_outputs.push((name, tensor)))
            .ok_or(HoduError::BuilderValidationFailed(format!(
                "Builder {} not found",
                self.0 .0
            )))
    }

    pub fn add_operation(
        &self,
        op: Op,
        outputs: Vec<TensorId>,
        input_layouts: Vec<Layout>,
        output_layouts: Vec<Layout>,
    ) -> HoduResult<()> {
        self.with_builder_mut(|b| {
            b.operations.push(op);
            b.operation_outputs.push(outputs);
            b.operation_layouts.push((input_layouts, output_layouts));
        })
        .ok_or(HoduError::BuilderValidationFailed(format!(
            "Builder {} not found",
            self.0 .0
        )))
    }

    pub fn start(&self) -> HoduResult<()> {
        let is_ended = self
            .with_builder(|b| b.is_ended)
            .ok_or(HoduError::BuilderValidationFailed(format!(
                "Builder {} not found",
                self.0 .0
            )))?;
        let name = self.get_name();

        if is_ended {
            return Err(HoduError::BuilderAlreadyEnded(name));
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
            return Err(HoduError::BuilderContextAlreadyActive(active_name));
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
            None => Err(HoduError::BuilderContextNotActive),
            Some(active_builder_id) if active_builder_id != &self.0 => {
                let active_name = get(*active_builder_id).get_name();
                Err(HoduError::BuilderValidationFailed(format!(
                    "Context mismatch: trying to end '{name}' but active context is '{active_name}'"
                )))
            },
            Some(_) => {
                self.with_builder_mut(|b| b.is_ended = true)
                    .ok_or(HoduError::BuilderValidationFailed(format!(
                        "Builder {} not found",
                        self.0 .0
                    )))?;
                *active_id = None;
                Ok(())
            },
        }
    }

    pub fn build(&self) -> HoduResult<Script> {
        use crate::script::ir::{ComputationGraph, GraphNode, InputNode, NodeId, NodeType, OutputNode, ScriptIR};

        let script_ir = self
            .with_builder(|b| -> HoduResult<ScriptIR> {
                let mut script_ir = ScriptIR::new(format!("script_from_{}", b.name));
                let mut graph = ComputationGraph::new();

                use crate::compat::HashSet;
                use crate::script::ir::ConstantNode;

                let mut processed_constants = HashSet::new();

                // Collect all unique constant tensor IDs first
                let mut constant_tensor_ids = Vec::new();
                for op in &b.operations {
                    let input_tensors = op.get_input_tensor_ids();
                    for &tensor_id in &input_tensors {
                        if !processed_constants.contains(&tensor_id)
                            && crate::tensor::get(tensor_id).is_some()
                            && crate::tensor::tensor_from_id(tensor_id).has_storage()
                        {
                            constant_tensor_ids.push(tensor_id);
                            processed_constants.insert(tensor_id);
                        }
                    }
                }

                // Convert constants in parallel
                #[cfg(all(feature = "std", feature = "rayon"))]
                let constant_nodes: Vec<_> = {
                    constant_tensor_ids
                        .par_iter()
                        .map(|&tensor_id| {
                            let tensor = crate::tensor::tensor_from_id(tensor_id);
                            let cpu_storage = tensor.with_storage(|storage| Ok(storage.to_cpu_storage()))?;

                            let constant_node = ConstantNode {
                                tensor_id,
                                shape: tensor.get_layout().get_shape().to_vec(),
                                dtype: tensor.get_dtype(),
                                data: cpu_storage?.to_bytes(),
                                compression: None,
                            };
                            Ok((tensor_id, constant_node))
                        })
                        .collect::<HoduResult<Vec<_>>>()?
                };

                #[cfg(not(all(feature = "std", feature = "rayon")))]
                let constant_nodes: Vec<_> = {
                    constant_tensor_ids
                        .iter()
                        .map(|&tensor_id| {
                            let tensor = crate::tensor::tensor_from_id(tensor_id);
                            let cpu_storage = tensor.with_storage(|storage| Ok(storage.to_cpu_storage()))?;

                            let constant_node = ConstantNode {
                                tensor_id,
                                shape: tensor.get_layout().get_shape().to_vec(),
                                dtype: tensor.get_dtype(),
                                data: cpu_storage?.to_bytes(),
                                compression: None,
                            };
                            Ok((tensor_id, constant_node))
                        })
                        .collect::<HoduResult<Vec<_>>>()?
                };

                // Insert constant nodes
                for (tensor_id, constant_node) in constant_nodes {
                    graph.metadata.constants.insert(tensor_id, constant_node);
                }

                use crate::script::ir::TensorInfo;

                // Process graph_inputs in parallel
                #[cfg(all(feature = "std", feature = "rayon"))]
                let input_data: Vec<_> = {
                    b.graph_inputs
                        .par_iter()
                        .map(|(name, tensor)| {
                            let input_node = InputNode {
                                name: name.to_string(),
                                tensor_id: tensor.id(),
                                optional: false,
                                default_value: None,
                            };
                            let tensor_info = TensorInfo {
                                id: tensor.id(),
                                shape: Some(tensor.get_layout().get_shape().iter().map(|&s| Some(s)).collect()),
                                dtype: Some(tensor.get_dtype()),
                                layout: None,
                                memory_layout: None,
                            };
                            (input_node, tensor_info)
                        })
                        .collect()
                };

                #[cfg(not(all(feature = "std", feature = "rayon")))]
                let input_data: Vec<_> = {
                    b.graph_inputs
                        .iter()
                        .map(|(name, tensor)| {
                            let input_node = InputNode {
                                name: name.to_string(),
                                tensor_id: tensor.id(),
                                optional: false,
                                default_value: None,
                            };
                            let tensor_info = TensorInfo {
                                id: tensor.id(),
                                shape: Some(tensor.get_layout().get_shape().iter().map(|&s| Some(s)).collect()),
                                dtype: Some(tensor.get_dtype()),
                                layout: None,
                                memory_layout: None,
                            };
                            (input_node, tensor_info)
                        })
                        .collect()
                };

                for (input_node, tensor_info) in input_data {
                    graph.metadata.inputs.push(input_node);
                    graph.metadata.tensor_info.insert(tensor_info.id, tensor_info);
                }

                // Process graph_outputs in parallel
                #[cfg(all(feature = "std", feature = "rayon"))]
                let output_data: Vec<_> = {
                    b.graph_outputs
                        .par_iter()
                        .map(|(name, tensor)| {
                            let output_node = OutputNode {
                                name: name.to_string(),
                                tensor_id: tensor.id(),
                                is_intermediate: false,
                            };
                            let tensor_info = TensorInfo {
                                id: tensor.id(),
                                shape: Some(tensor.get_layout().get_shape().iter().map(|&s| Some(s)).collect()),
                                dtype: Some(tensor.get_dtype()),
                                layout: None,
                                memory_layout: None,
                            };
                            (output_node, tensor_info)
                        })
                        .collect()
                };

                #[cfg(not(all(feature = "std", feature = "rayon")))]
                let output_data: Vec<_> = {
                    b.graph_outputs
                        .iter()
                        .map(|(name, tensor)| {
                            let output_node = OutputNode {
                                name: name.to_string(),
                                tensor_id: tensor.id(),
                                is_intermediate: false,
                            };
                            let tensor_info = TensorInfo {
                                id: tensor.id(),
                                shape: Some(tensor.get_layout().get_shape().iter().map(|&s| Some(s)).collect()),
                                dtype: Some(tensor.get_dtype()),
                                layout: None,
                                memory_layout: None,
                            };
                            (output_node, tensor_info)
                        })
                        .collect()
                };

                for (output_node, tensor_info) in output_data {
                    graph.metadata.outputs.push(output_node);
                    graph.metadata.tensor_info.insert(tensor_info.id, tensor_info);
                }

                for (node_counter, op) in b.operations.iter().enumerate() {
                    let node_id = NodeId(node_counter);

                    let input_tensors = op.get_input_tensor_ids();

                    let output_tensors = b.operation_outputs.get(node_counter).cloned().unwrap_or_default();

                    let (input_layouts, output_layouts) =
                        b.operation_layouts.get(node_counter).cloned().unwrap_or_default();

                    use crate::script::ir::TensorInfo;

                    for (tensor_id, layout) in input_tensors.iter().zip(input_layouts.iter()) {
                        if !graph.metadata.tensor_info.contains_key(tensor_id)
                            && crate::tensor::get(*tensor_id).is_some()
                        {
                            let tensor = crate::tensor::tensor_from_id(*tensor_id);
                            let tensor_info = TensorInfo {
                                id: *tensor_id,
                                shape: Some(layout.get_shape().iter().map(|&s| Some(s)).collect()),
                                dtype: Some(tensor.get_dtype()),
                                layout: None,
                                memory_layout: None,
                            };
                            graph.metadata.tensor_info.insert(*tensor_id, tensor_info);
                        }
                    }

                    for (tensor_id, layout) in output_tensors.iter().zip(output_layouts.iter()) {
                        if !graph.metadata.tensor_info.contains_key(tensor_id)
                            && crate::tensor::get(*tensor_id).is_some()
                        {
                            let tensor = crate::tensor::tensor_from_id(*tensor_id);
                            let tensor_info = TensorInfo {
                                id: *tensor_id,
                                shape: Some(layout.get_shape().iter().map(|&s| Some(s)).collect()),
                                dtype: Some(tensor.get_dtype()),
                                layout: None,
                                memory_layout: None,
                            };
                            graph.metadata.tensor_info.insert(*tensor_id, tensor_info);
                        }
                    }

                    let graph_node = GraphNode {
                        id: node_id,
                        node_type: NodeType::Compute,
                        operation: op.clone(),
                        attributes: HashMap::new(),
                        input_tensors,
                        output_tensors,
                        input_layouts,
                        output_layouts,
                    };
                    graph.topology.nodes.push(graph_node);

                    graph.execution_plan.execution_order.push(node_id);
                }

                script_ir.graph = graph;
                Ok(script_ir)
            })
            .ok_or(HoduError::BuilderValidationFailed(format!(
                "Builder {} not found",
                self.0 .0
            )))?;

        let final_script_ir = script_ir?;
        Ok(Script::with_ir(final_script_ir))
    }
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
        None => Err(HoduError::BuilderContextNotActive),
    }
}

pub fn with_active_builder<F, R>(f: F) -> HoduResult<R>
where
    F: FnOnce(&mut Builder_) -> R,
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
                    Some(mut builder_) => Ok(f(&mut builder_)),
                    None => Err(HoduError::BuilderValidationFailed(format!(
                        "Active builder '{builder_id:?}' not found in storage"
                    ))),
                }
            }
            #[cfg(not(feature = "std"))]
            {
                let mut builders = BUILDERS.lock();
                match builders.get_mut(builder_id) {
                    Some(builder_) => Ok(f(builder_)),
                    None => Err(HoduError::BuilderValidationFailed(format!(
                        "Active builder '{builder_id:?}' not found in storage"
                    ))),
                }
            }
        },
        None => Err(HoduError::BuilderContextNotActive),
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

pub fn get_builder_by_name(name: &str) -> Option<Builder> {
    #[cfg(feature = "std")]
    {
        BUILDERS.iter().find_map(|entry| {
            let (id, builder_) = entry.pair();
            if builder_.name == name {
                Some(get(*id))
            } else {
                None
            }
        })
    }
    #[cfg(not(feature = "std"))]
    {
        let builders = BUILDERS.lock();
        for (id, builder_) in builders.iter() {
            if builder_.name == name {
                return Some(get(*id));
            }
        }
        None
    }
}
