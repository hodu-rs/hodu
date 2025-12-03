mod bytes;
mod core;
mod creation;
mod creation_from_ops;
mod creation_static;
mod display;
pub mod gradient;
mod internal;
mod ops;
mod registry;
pub(crate) mod utils;
mod vec;

// Re-export core types
pub(crate) use core::Tensor_;
pub use core::{Tensor, TensorId};
pub use creation::{get_runtime_device, set_runtime_device};
pub use gradient::{is_computing_gradients, is_in_optimizer_step, set_optimizer_step_flag, ContextId, GradientContext};

// Re-export registry functions
pub use registry::{exists, get, get_dtype, shrink_tensor_storage, tensor_count, with_tensor, with_tensor_mut};

// Re-export internal functions for crate use
pub(crate) use internal::{
    create_builder_tensor, from_shared_storage_with, from_storage, from_storage_with_context, set_grad_tensor_id,
    tensor_from_id,
};

// Re-export for submodules
pub(crate) use registry::{get_all_tensor_ids, insert, remove};
