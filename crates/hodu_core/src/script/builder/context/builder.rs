use super::{storage, BuilderId, BuilderState};
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
};

/// Builder handle that provides access to builder state
pub struct Builder(BuilderId);

impl Builder {
    /// Create a new builder with the given name
    pub fn new(name: String) -> Self {
        let state = BuilderState::new(name);
        let builder_id = BuilderId::new();
        storage::insert(builder_id, state);
        Builder(builder_id)
    }

    /// Create a builder from an existing ID (internal use)
    pub(super) fn from_id(builder_id: BuilderId) -> Self {
        Builder(builder_id)
    }

    /// Execute a function with immutable access to the builder state
    pub fn with_state<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&BuilderState) -> R,
    {
        storage::with_state(self.0, f)
    }

    /// Execute a function with mutable access to the builder state
    pub fn with_state_mut<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut BuilderState) -> R,
    {
        storage::with_state_mut(self.0, f)
    }

    /// Get the builder's name
    pub fn get_name(&self) -> String {
        self.with_state(|s| s.name.clone())
            .unwrap_or_else(|| format!("Builder({})", self.0 .0))
    }

    /// Set the builder's name
    pub fn set_name(&self, name: String) -> HoduResult<()> {
        self.with_state_mut(|s| {
            s.name = name.clone();
            s.module.name = name;
        })
        .ok_or_else(|| HoduError::BuilderNotFound(format!("{}", self.0 .0)))
    }

    /// Get the builder's state (for inspection)
    pub fn get_state(&self) -> Option<BuilderState> {
        // Note: This clones the entire state, which may be expensive
        // Consider if this is necessary or if we should provide a different API
        self.with_state(|s| BuilderState {
            name: s.name.clone(),
            module: s.module.clone(),
            current_function: s.current_function.clone(),
            current_block: s.current_block,
            value_counter: s.value_counter,
            block_counter: s.block_counter,
            tensor_to_value: s.tensor_to_value.clone(),
            graph_inputs: s.graph_inputs.clone(),
            graph_outputs: s.graph_outputs.clone(),
            is_ended: s.is_ended,
        })
    }

    /// Start building - marks this builder as the active one
    pub fn start(&self) -> HoduResult<()> {
        let is_ended = self
            .with_state(|s| s.is_ended)
            .ok_or_else(|| HoduError::BuilderNotFound(format!("{}", self.0 .0)))?;
        let name = self.get_name();

        if is_ended {
            return Err(HoduError::InternalError(format!("Builder '{}' already ended", name)));
        }

        let mut active_id = storage::lock_active_builder_id();

        if let Some(active_builder_id) = *active_id {
            let active_name = storage::get(active_builder_id).get_name();
            return Err(HoduError::InternalError(format!(
                "Builder context '{}' already active",
                active_name
            )));
        }

        *active_id = Some(self.0);
        Ok(())
    }

    /// End building - clears this builder as the active one
    pub fn end(&self) -> HoduResult<()> {
        let mut active_id = storage::lock_active_builder_id();
        let name = self.get_name();

        match active_id.as_ref() {
            None => Err(HoduError::InternalError("No active builder context".to_string())),
            Some(active_builder_id) if active_builder_id != &self.0 => {
                let active_name = storage::get(*active_builder_id).get_name();
                Err(HoduError::InternalError(format!(
                    "Context mismatch: trying to end '{}' but active context is '{}'",
                    name, active_name
                )))
            },
            Some(_) => {
                self.with_state_mut(|s| s.is_ended = true)
                    .ok_or_else(|| HoduError::BuilderNotFound(format!("{}", self.0 .0)))?;
                *active_id = None;
                Ok(())
            },
        }
    }

    /// Build the final script from the builder state
    pub fn build(&self) -> HoduResult<crate::script::Script> {
        let module = self
            .with_state_mut(super::super::codegen::build_module)
            .ok_or_else(|| HoduError::BuilderNotFound(format!("{}", self.0 .0)))??;
        Ok(crate::script::Script::new(module))
    }

    /// Build the final module from the builder state (if you need the module directly)
    pub fn build_module(&self) -> HoduResult<super::Module> {
        self.with_state_mut(super::super::codegen::build_module)
            .ok_or_else(|| HoduError::BuilderNotFound(format!("{}", self.0 .0)))?
    }
}
