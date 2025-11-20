use super::{
    core::{Tensor, TensorId, Tensor_},
    gradient, registry,
};
use crate::{
    be::storage::BackendStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    types::Layout,
};

pub(crate) fn from_storage(
    storage: BackendStorage,
    layout: Layout,
    is_runtime: bool,
    requires_grad: bool,
    owner_context: Option<gradient::ContextId>,
) -> Tensor {
    let tensor_ = Tensor_ {
        storage: Some(Arc::new(storage)),
        layout,
        requires_grad,
        grad_tensor_id: None,
        is_runtime,
        is_gradient: false,
        owner_context,
        ref_count: AtomicUsize::new(1),
    };
    let tensor_id = TensorId::new();
    registry::insert(tensor_id, tensor_);
    Tensor::from_id(tensor_id)
}

pub(crate) fn from_storage_with_context(
    storage: BackendStorage,
    layout: Layout,
    is_runtime: bool,
    requires_grad: bool,
) -> Tensor {
    let owner_context = if gradient::is_computing_gradients() || gradient::is_in_optimizer_step() {
        None
    } else {
        Some(gradient::get_active_context())
    };
    from_storage(storage, layout, is_runtime, requires_grad, owner_context)
}

pub(crate) fn from_shared_storage_with(source_tensor: &Tensor, layout: Layout, requires_grad: bool) -> Tensor {
    let storage_arc = registry::with_tensor(source_tensor.id(), |tensor_ref| tensor_ref.storage.clone())
        .expect("Source tensor not found");

    let owner_context = if gradient::is_computing_gradients() || gradient::is_in_optimizer_step() {
        None
    } else {
        Some(gradient::get_active_context())
    };

    let tensor_ = Tensor_ {
        storage: storage_arc,
        layout,
        requires_grad,
        grad_tensor_id: None,
        is_runtime: true,
        is_gradient: false,
        owner_context,
        ref_count: AtomicUsize::new(1),
    };

    let tensor_id = TensorId::new();
    registry::insert(tensor_id, tensor_);
    Tensor::from_id(tensor_id)
}

pub(crate) fn create_builder_tensor(layout: Layout, requires_grad: bool) -> (TensorId, Tensor) {
    let tensor_ = Tensor_ {
        storage: None,
        layout,
        requires_grad,
        grad_tensor_id: None,
        is_runtime: false,
        is_gradient: false,
        owner_context: None,
        ref_count: AtomicUsize::new(1),
    };
    let tensor_id = TensorId::new();
    registry::insert(tensor_id, tensor_);
    let tensor = Tensor::from_id(tensor_id);
    (tensor_id, tensor)
}

pub(crate) fn register_operation_in_builder(
    op: crate::ops::Op,
    op_params: Option<crate::ops::OpParams>,
    inputs: Vec<TensorId>,
    outputs: Vec<TensorId>,
    input_layouts: Vec<Layout>,
    output_layouts: Vec<Layout>,
) -> HoduResult<()> {
    use crate::script::builder;
    if let Ok(active_builder) = builder::get_active_builder() {
        active_builder.add_operation(op, op_params, inputs, outputs, input_layouts, output_layouts)?;
    }
    Ok(())
}

pub(crate) fn tensor_from_id(tensor_id: TensorId) -> Tensor {
    registry::with_tensor(tensor_id, |t| {
        t.ref_count.fetch_add(1, Ordering::Relaxed);
    });
    Tensor::from_id(tensor_id)
}

pub(crate) fn set_grad_tensor_id(tensor_id: TensorId, grad_tensor_id: TensorId) -> HoduResult<()> {
    if let Some(()) = registry::with_tensor_mut(grad_tensor_id, |t| {
        t.ref_count.fetch_add(1, Ordering::Relaxed);
    }) {
        // Successfully incremented ref_count
    } else {
        return Err(HoduError::TensorNotFound(grad_tensor_id));
    }

    let old_grad_id = registry::with_tensor(tensor_id, |tensor_ref| tensor_ref.grad_tensor_id);

    if let Some(Some(old_grad_id)) = old_grad_id {
        let (should_remove, should_cleanup_tape) = registry::with_tensor(old_grad_id, |t| {
            let prev_count = t.ref_count.fetch_sub(1, Ordering::Relaxed);
            let remove = prev_count == 1 && t.is_runtime && t.owner_context.is_none();
            let cleanup = prev_count == 1 && t.requires_grad && t.owner_context.is_none();
            (remove, cleanup)
        })
        .unwrap_or((false, false));

        if should_cleanup_tape {
            gradient::cleanup_tape_for_dropped_tensor(old_grad_id);
        }

        if should_remove {
            registry::remove(old_grad_id);
        }
    }

    if let Some(result) = registry::with_tensor_mut(tensor_id, |tensor_ref| {
        tensor_ref.grad_tensor_id = Some(grad_tensor_id);
        Ok(())
    }) {
        result
    } else {
        Err(HoduError::TensorNotFound(tensor_id))
    }
}

impl Tensor {
    pub(super) fn from_id(tensor_id: TensorId) -> Self {
        Tensor(tensor_id)
    }
}
