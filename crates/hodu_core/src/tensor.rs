mod creation;
mod creation_from_ops;
mod creation_static;
mod display;
pub mod gradient;
mod ops;
pub(crate) mod utils;
mod vec;

use crate::{
    be::storage::BackendStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    types::{DType, Device, Layout, Shape},
};
pub use creation::{get_runtime_device, set_runtime_device};
#[cfg(feature = "std")]
use dashmap::DashMap;
pub use gradient::{is_computing_gradients, is_in_optimizer_step, set_optimizer_step_flag, ContextId, GradientContext};

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct TensorId(usize);

impl TensorId {
    pub(crate) fn new() -> Self {
        static TENSOR_COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(TENSOR_COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    #[cfg(test)]
    pub(crate) fn test_new(id: usize) -> Self {
        Self(id)
    }
}

#[repr(transparent)]
pub struct Tensor(TensorId);

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        // Increment reference count if tensor still exists
        // If it doesn't exist, it's already been cleaned up and we shouldn't clone it
        // But since we're just copying TensorId, it's safe to create the handle
        if let Some(()) = with_tensor(self.0, |t| {
            t.ref_count.fetch_add(1, Ordering::Relaxed);
        }) {
            // Tensor exists, ref_count incremented
        }
        // Note: Even if tensor was removed, we return a new handle
        // The next access will fail with TensorNotFound error
        Tensor(self.0)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        // Always decrement ref_count for accurate tracking
        let (should_remove, should_cleanup_tape, grad_id) = with_tensor(self.0, |t| {
            let prev_count = t.ref_count.load(Ordering::Relaxed);

            // Only decrement if > 0 to avoid underflow
            if prev_count > 0 {
                t.ref_count.fetch_sub(1, Ordering::Relaxed);
            }

            // Don't remove during gradient computation (managed by GradientComputationGuard)
            let remove = if gradient::is_computing_gradients() {
                false
            } else {
                // Remove if:
                // 1. Last or already-zero reference (prev_count == 1 or 0)
                //    prev_count==0 happens when VJP functions drop Tensor after extracting .id()
                // 2. Runtime tensor (is_runtime == true)
                // 3. NOT context-owned (owner_context == None)
                // 4. NOT a gradient tensor (is_gradient == false)
                (prev_count == 1 || prev_count == 0) && t.is_runtime && t.owner_context.is_none() && !t.is_gradient
            };

            // Cleanup tape if:
            // 1. Last reference (prevents cleanup on every clone drop)
            // 2. Tensor requires grad (only these tensors are on the tape)
            // 3. NOT context-owned (forward intermediates need to stay on tape)
            // 4. NOT during gradient computation (tape is being used)
            let cleanup =
                prev_count == 1 && t.requires_grad && t.owner_context.is_none() && !gradient::is_computing_gradients();

            (remove, cleanup, t.grad_tensor_id)
        })
        .unwrap_or((false, false, None));

        if should_cleanup_tape {
            gradient::cleanup_tape_for_dropped_tensor(self.0);
        }

        if should_remove {
            // If this tensor has a gradient, decrement its ref_count and remove if necessary
            if let Some(grad_id) = grad_id {
                if let Some((should_remove_grad, should_cleanup_grad_tape)) = with_tensor(grad_id, |g| {
                    let prev_count = g.ref_count.load(Ordering::Relaxed);
                    if prev_count > 0 {
                        g.ref_count.fetch_sub(1, Ordering::Relaxed);
                    }
                    let remove = (prev_count == 1 || prev_count == 0) && g.is_runtime && g.owner_context.is_none();
                    let cleanup = (prev_count == 1 || prev_count == 0) && g.requires_grad && g.owner_context.is_none();
                    (remove, cleanup)
                }) {
                    if should_cleanup_grad_tape {
                        gradient::cleanup_tape_for_dropped_tensor(grad_id);
                    }
                    if should_remove_grad {
                        remove(grad_id);
                    }
                }
            }

            remove(self.0);
        }
    }
}

pub struct Tensor_ {
    storage: Option<Arc<BackendStorage>>,
    layout: Layout,
    requires_grad: bool,
    grad_tensor_id: Option<TensorId>,
    is_runtime: bool,
    /// Flag indicating this tensor is a gradient attached to a parameter
    /// Gradient tensors should not be cleaned up by context cleanup
    is_gradient: bool,
    /// Owner context for memory management:
    /// - None: User-created tensor (cleaned up when ref_count=0)
    /// - Some(ContextId): Context-owned tensor (cleaned up on context drop)
    owner_context: Option<ContextId>,
    /// Reference count for tracking Tensor handles
    ref_count: AtomicUsize,
}

impl Tensor_ {
    /// Get the owner context of this tensor
    pub(crate) fn owner_context(&self) -> Option<ContextId> {
        self.owner_context
    }
}

#[cfg(feature = "std")]
static TENSORS: LazyLock<DashMap<TensorId, Tensor_>> = LazyLock::new(|| {
    // Use number of available parallelism (CPU cores) for optimal shard count
    // Fallback to 64 if detection fails. Round up to next power of two.
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(64);
    let shard_count = cores.next_power_of_two();
    DashMap::with_capacity_and_shard_amount(1 << 16, shard_count)
});

#[cfg(not(feature = "std"))]
static TENSORS: LazyLock<RwLock<HashMap<TensorId, Tensor_>>> = LazyLock::new(|| {
    // Note: BTreeMap doesn't support reserve(), but provides O(log n) access
    RwLock::new(HashMap::new())
});

pub fn insert(tensor_id: TensorId, tensor_: Tensor_) {
    #[cfg(feature = "std")]
    {
        TENSORS.insert(tensor_id, tensor_);
    }
    #[cfg(not(feature = "std"))]
    {
        let mut tensors = TENSORS.write();
        tensors.insert(tensor_id, tensor_);
    }
}

pub fn exists(tensor_id: TensorId) -> bool {
    #[cfg(feature = "std")]
    {
        TENSORS.contains_key(&tensor_id)
    }
    #[cfg(not(feature = "std"))]
    {
        let tensors = TENSORS.read();
        tensors.contains_key(&tensor_id)
    }
}

pub(crate) fn remove(tensor_id: TensorId) {
    // First, decrement ref_count of grad_tensor if exists
    if let Some(Some(grad_id)) = with_tensor(tensor_id, |t| t.grad_tensor_id) {
        with_tensor(grad_id, |t| {
            t.ref_count.fetch_sub(1, Ordering::Relaxed);
        });
    }

    // Then remove the tensor from TENSORS
    #[cfg(feature = "std")]
    {
        TENSORS.remove(&tensor_id);
    }
    #[cfg(not(feature = "std"))]
    {
        let mut tensors = TENSORS.write();
        tensors.remove(&tensor_id);
    }
}

// Provide a function that mimics DashMap's get() behavior for backward compatibility
// For std: returns a reference guard, for no-std: returns option to indicate existence
#[cfg(feature = "std")]
pub fn get(tensor_id: TensorId) -> Option<dashmap::mapref::one::Ref<'static, TensorId, Tensor_>> {
    TENSORS.get(&tensor_id)
}

#[cfg(not(feature = "std"))]
pub fn get(tensor_id: TensorId) -> Option<()> {
    if exists(tensor_id) {
        Some(())
    } else {
        None
    }
}

// Shrink the TENSORS map to reduce memory fragmentation
// This is useful after heavy tensor operations to reclaim memory
pub fn shrink_tensor_storage() {
    #[cfg(feature = "std")]
    {
        TENSORS.shrink_to_fit();
    }
    #[cfg(not(feature = "std"))]
    {
        let mut tensors = TENSORS.write();
        tensors.shrink_to_fit();
    }
}

// Get the number of tensors currently stored (for debugging)
pub fn tensor_count() -> usize {
    #[cfg(feature = "std")]
    {
        TENSORS.len()
    }
    #[cfg(not(feature = "std"))]
    {
        let tensors = TENSORS.read();
        tensors.len()
    }
}

pub(crate) fn get_all_tensor_ids() -> Vec<TensorId> {
    #[cfg(feature = "std")]
    {
        TENSORS.iter().map(|entry| *entry.key()).collect()
    }
    #[cfg(not(feature = "std"))]
    {
        let tensors = TENSORS.read();
        tensors.keys().copied().collect()
    }
}

// Helper function to access tensor data with a closure
pub fn with_tensor<F, R>(tensor_id: TensorId, f: F) -> Option<R>
where
    F: FnOnce(&Tensor_) -> R,
{
    #[cfg(feature = "std")]
    {
        TENSORS.get(&tensor_id).map(|tensor_ref| f(&tensor_ref))
    }
    #[cfg(not(feature = "std"))]
    {
        let tensors = TENSORS.read();
        tensors.get(&tensor_id).map(f)
    }
}

// Helper function to access tensor data mutably with a closure
pub fn with_tensor_mut<F, R>(tensor_id: TensorId, f: F) -> Option<R>
where
    F: FnOnce(&mut Tensor_) -> R,
{
    #[cfg(feature = "std")]
    {
        TENSORS.get_mut(&tensor_id).map(|mut tensor_ref| f(&mut tensor_ref))
    }
    #[cfg(not(feature = "std"))]
    {
        let mut tensors = TENSORS.write();
        tensors.get_mut(&tensor_id).map(f)
    }
}

impl Tensor {
    /// Returns the unique tensor identifier.
    ///
    /// Every tensor is assigned a unique ID at creation time.
    ///
    /// # Returns
    ///
    /// The unique TensorId of this tensor.
    #[inline]
    pub fn id(&self) -> TensorId {
        self.0
    }

    pub(crate) fn has_storage(&self) -> bool {
        with_tensor(self.0, |t| t.storage.is_some()).unwrap_or(false)
    }

    pub(crate) fn with_storage<R>(&self, f: impl FnOnce(&BackendStorage) -> HoduResult<R>) -> HoduResult<R> {
        with_tensor(self.0, |tensor_ref| {
            let storage = tensor_ref.storage.as_ref().ok_or(HoduError::StorageNotFound(self.0))?;
            f(storage.as_ref())
        })
        .ok_or(HoduError::TensorNotFound(self.0))?
    }

    // #[cfg(feature = "std")]
    // pub(crate) fn with_storage_mut<R>(&self, f: impl FnOnce(&mut BackendStorage) -> HoduResult<R>) -> HoduResult<R> {
    //     with_tensor(self.0, |tensor_ref| {
    //         let storage = tensor_ref.storage.as_ref().ok_or(HoduError::StorageNotFound(self.0))?;
    //         let mut storage_guard = storage.write().map_err(|_| HoduError::StorageCorrupted(self.0))?;
    //         f(&mut storage_guard)
    //     })
    //     .ok_or(HoduError::TensorNotFound(self.0))?
    // }
    //
    // #[cfg(not(feature = "std"))]
    // pub(crate) fn with_storage_mut<R>(&self, f: impl FnOnce(&mut BackendStorage) -> HoduResult<R>) -> HoduResult<R> {
    //     with_tensor(self.0, |tensor_ref| {
    //         let storage = tensor_ref.storage.as_ref().ok_or(HoduError::StorageNotFound(self.0))?;
    //         let mut storage_guard = storage.write();
    //         f(&mut storage_guard)
    //     })
    //     .ok_or(HoduError::TensorNotFound(self.0))?
    // }

    pub fn is_runtime(&self) -> bool {
        with_tensor(self.0, |t| t.is_runtime).unwrap_or(false)
    }

    /// Returns a clone of the tensor's layout.
    /// For performance-sensitive code, prefer `with_layout` to avoid cloning.
    pub fn layout(&self) -> Layout {
        with_tensor(self.0, |t| t.layout.clone()).unwrap_or_else(|| Layout::from_shape(&Shape::scalar()))
    }

    /// Accesses the tensor's layout without cloning via a closure.
    /// This is more efficient than `layout()` when you only need to read layout properties.
    #[inline]
    pub fn with_layout<R>(&self, f: impl FnOnce(&Layout) -> R) -> Option<R> {
        with_tensor(self.0, |t| f(&t.layout))
    }

    /// Returns a clone of the tensor's shape.
    /// For performance-sensitive code, prefer `with_shape` to avoid cloning.
    pub fn shape(&self) -> Shape {
        with_tensor(self.0, |t| t.layout.shape().clone()).unwrap_or_else(Shape::scalar)
    }

    /// Accesses the tensor's shape without cloning via a closure.
    /// This is more efficient than `shape()` when you only need to read shape properties.
    #[inline]
    pub fn with_shape<R>(&self, f: impl FnOnce(&Shape) -> R) -> Option<R> {
        with_tensor(self.0, |t| f(t.layout.shape()))
    }

    pub fn strides(&self) -> Vec<usize> {
        with_tensor(self.0, |t| t.layout.strides().to_vec()).unwrap_or_else(Vec::new)
    }

    pub fn offset(&self) -> usize {
        with_tensor(self.0, |t| t.layout.offset()).unwrap_or(0)
    }

    pub fn ndim(&self) -> usize {
        with_tensor(self.0, |t| t.layout.ndim()).unwrap_or(0)
    }

    pub fn dim_size(&self, index: i32) -> Option<usize> {
        with_tensor(self.0, |t| t.layout.dim_size(index)).unwrap_or(None)
    }

    pub fn size(&self) -> usize {
        with_tensor(self.0, |t| t.layout.size()).unwrap_or(1)
    }

    pub fn device(&self) -> Device {
        self.with_storage(|storage| Ok(storage.device())).unwrap_or(Device::CPU)
    }

    pub fn dtype(&self) -> DType {
        self.with_storage(|storage| Ok(storage.dtype())).unwrap_or(DType::F32)
    }

    pub fn set_requires_grad(&self, requires: bool) -> HoduResult<()> {
        if requires {
            let dtype = self.dtype();
            if !dtype.is_float() {
                return Err(HoduError::RequiresGradNotSet(self.0));
            }
        }

        if let Some(result) = with_tensor_mut(self.0, |tensor_ref| {
            tensor_ref.requires_grad = requires;
            Ok(())
        }) {
            result
        } else {
            Err(HoduError::TensorNotFound(self.0))
        }
    }

    pub fn requires_grad(&self) -> HoduResult<()> {
        self.set_requires_grad(true)
    }

    pub fn is_requires_grad(&self) -> bool {
        with_tensor(self.0, |t| t.requires_grad).unwrap_or(false)
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    pub fn backward(&self) -> HoduResult<()> {
        let dtype = self.dtype();
        if !dtype.is_float() {
            return Err(HoduError::GradientComputationFailed(format!(
                "Cannot compute gradients for non-float tensor with dtype {dtype:?}"
            )));
        }

        if !self.is_requires_grad() {
            return Err(HoduError::RequiresGradNotSet(self.0));
        }

        gradient::compute_gradients(self.id())
    }

    pub fn grad(&self) -> HoduResult<Self> {
        if let Some(result) = with_tensor(self.0, |tensor_ref| {
            if let Some(grad_id) = tensor_ref.grad_tensor_id {
                // Use tensor_from_id to properly increment ref_count
                Ok(tensor_from_id(grad_id))
            } else {
                Err(HoduError::GradientNotComputed(self.0))
            }
        }) {
            result
        } else {
            Err(HoduError::TensorNotFound(self.0))
        }
    }

    pub fn zero_grad(&self) -> HoduResult<()> {
        if !self.is_requires_grad() {
            return Ok(()); // No gradient to zero
        }

        // Get old gradient ID and replace with None temporarily
        let old_grad_id = with_tensor_mut(self.0, |tensor_ref| tensor_ref.grad_tensor_id.take());

        // with_tensor_mut returns Option<Option<TensorId>>, need to unwrap both levels
        if let Some(Some(old_grad_id)) = old_grad_id {
            // Get shape and dtype from old gradient WITHOUT incrementing ref_count
            let (shape, dtype) = with_tensor(old_grad_id, |t| {
                let shape = t.layout.shape().to_vec();
                let dtype = t.storage.as_ref().map(|s| s.dtype()).unwrap_or(DType::F32);
                (shape, dtype)
            })
            .ok_or(HoduError::TensorNotFound(old_grad_id))?;

            // Decrement ref_count of old gradient and check if it should be removed
            let (should_remove, should_cleanup_tape) = with_tensor(old_grad_id, |t| {
                let prev_count = t.ref_count.fetch_sub(1, Ordering::Relaxed);
                let remove = prev_count == 1 && t.is_runtime && t.owner_context.is_none();
                // Cleanup tape if it was the last reference and tensor requires grad
                let cleanup = prev_count == 1 && t.requires_grad && t.owner_context.is_none();
                (remove, cleanup)
            })
            .unwrap_or((false, false));

            if should_cleanup_tape {
                gradient::cleanup_tape_for_dropped_tensor(old_grad_id);
            }

            if should_remove {
                remove(old_grad_id);
            }

            // Create zeros tensor with same shape
            let zeros = Self::zeros(&shape, dtype)?;
            let zeros_id = zeros.id();

            // Mark as gradient tensor and increment ref_count BEFORE drop
            // This ensures is_gradient=true is set atomically with ref_count increment
            if let Some(()) = with_tensor_mut(zeros_id, |t| {
                t.is_gradient = true;
                t.ref_count.fetch_add(1, Ordering::Relaxed); // ref_count=2
            }) {
                // Successfully set is_gradient
            } else {
                return Err(HoduError::TensorNotFound(zeros_id));
            }

            with_tensor_mut(self.0, |tensor_ref| {
                tensor_ref.grad_tensor_id = Some(zeros_id);
            });

            // zeros will drop here, decrementing ref_count back to 1
        }

        Ok(())
    }
}

pub(crate) fn from_storage(
    storage: BackendStorage,
    layout: Layout,
    is_runtime: bool,
    requires_grad: bool,
    owner_context: Option<ContextId>,
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
    insert(tensor_id, tensor_);
    Tensor(tensor_id)
}

pub(crate) fn from_storage_with_context(
    storage: BackendStorage,
    layout: Layout,
    is_runtime: bool,
    requires_grad: bool,
) -> Tensor {
    let owner_context = if gradient::is_computing_gradients() || gradient::is_in_optimizer_step() {
        // Managed by GradientComputationGuard's keep_alive
        // Optimizer intermediate tensors should be cleaned up immediately
        None
    } else {
        Some(gradient::get_active_context())
    };
    from_storage(storage, layout, is_runtime, requires_grad, owner_context)
}

pub(crate) fn from_shared_storage_with(source_tensor: &Tensor, layout: Layout, requires_grad: bool) -> Tensor {
    let storage_arc =
        with_tensor(source_tensor.id(), |tensor_ref| tensor_ref.storage.clone()).expect("Source tensor not found");

    // Runtime tensors (operation results) should be owned by current context
    // This ensures they are not removed until backward completes
    let owner_context = if gradient::is_computing_gradients() || gradient::is_in_optimizer_step() {
        // Managed by GradientComputationGuard's keep_alive
        // Optimizer intermediate tensors should be cleaned up immediately
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
    insert(tensor_id, tensor_);
    Tensor(tensor_id)
}

pub(crate) fn create_builder_tensor(layout: Layout, requires_grad: bool) -> (TensorId, Tensor) {
    let tensor_ = Tensor_ {
        storage: None,
        layout,
        requires_grad,
        grad_tensor_id: None,
        is_runtime: false,
        is_gradient: false,
        owner_context: None, // Builder tensors are user-created
        ref_count: AtomicUsize::new(1),
    };
    let tensor_id = TensorId::new();
    insert(tensor_id, tensor_);
    let tensor = Tensor(tensor_id);
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
    // Increment ref_count for the new handle
    with_tensor(tensor_id, |t| {
        t.ref_count.fetch_add(1, Ordering::Relaxed);
    });
    Tensor(tensor_id)
}

pub(crate) fn set_grad_tensor_id(tensor_id: TensorId, grad_tensor_id: TensorId) -> HoduResult<()> {
    // Increment ref_count for the new gradient tensor (but don't set is_gradient)
    // is_gradient should only be set for parameter gradients in zero_grad()
    if let Some(()) = with_tensor_mut(grad_tensor_id, |t| {
        t.ref_count.fetch_add(1, Ordering::Relaxed);
    }) {
        // Successfully incremented ref_count
    } else {
        return Err(HoduError::TensorNotFound(grad_tensor_id));
    }

    // Get old gradient ID (if exists) - must release lock before modifying ref_count
    let old_grad_id = with_tensor(tensor_id, |tensor_ref| tensor_ref.grad_tensor_id);

    // Decrement ref_count of old gradient and remove if necessary
    if let Some(Some(old_grad_id)) = old_grad_id {
        let (should_remove, should_cleanup_tape) = with_tensor(old_grad_id, |t| {
            let prev_count = t.ref_count.fetch_sub(1, Ordering::Relaxed);
            let remove = prev_count == 1 && t.is_runtime && t.owner_context.is_none();
            // Cleanup tape if it was the last reference and tensor requires grad
            let cleanup = prev_count == 1 && t.requires_grad && t.owner_context.is_none();
            (remove, cleanup)
        })
        .unwrap_or((false, false));

        if should_cleanup_tape {
            gradient::cleanup_tape_for_dropped_tensor(old_grad_id);
        }

        if should_remove {
            remove(old_grad_id);
        }
    }

    // Now set new gradient
    if let Some(result) = with_tensor_mut(tensor_id, |tensor_ref| {
        tensor_ref.grad_tensor_id = Some(grad_tensor_id);
        Ok(())
    }) {
        result
    } else {
        Err(HoduError::TensorNotFound(tensor_id))
    }
}
