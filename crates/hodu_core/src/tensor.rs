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
pub use gradient::{is_computing_gradients, GradientContext};

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct TensorId(u32);

impl TensorId {
    pub(crate) fn new() -> Self {
        static TENSOR_COUNTER: AtomicU32 = AtomicU32::new(0);
        Self(TENSOR_COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    #[cfg(test)]
    pub fn test_new(id: u32) -> Self {
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
        // Increment reference count
        #[cfg(feature = "std")]
        {
            if let Some(tensor_ref) = TENSORS.get(&self.0) {
                tensor_ref.ref_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if let Some(tensors) = TENSORS.try_read() {
                if let Some(tensor_) = tensors.get(&self.0) {
                    tensor_.ref_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        Tensor(self.0)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        #[cfg(feature = "std")]
        {
            if let Some(tensor_ref) = TENSORS.get(&self.0) {
                let prev_count = tensor_ref.ref_count.fetch_sub(1, Ordering::Relaxed);
                if prev_count == 1 {
                    // Last reference - drop the guard first
                    drop(tensor_ref);
                    // Now remove from global HashMap
                    TENSORS.remove(&self.0);
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            let should_remove = if let Some(tensors) = TENSORS.try_read() {
                if let Some(tensor_) = tensors.get(&self.0) {
                    let prev_count = tensor_.ref_count.fetch_sub(1, Ordering::Relaxed);
                    prev_count == 1
                } else {
                    false
                }
            } else {
                false
            };

            if should_remove {
                if let Some(mut tensors) = TENSORS.try_write() {
                    tensors.remove(&self.0);
                }
            }
        }
    }
}

pub struct Tensor_ {
    storage: Option<Arc<BackendStorage>>,
    is_runtime: bool,
    layout: Layout,
    requires_grad: bool,
    grad_tensor_id: Option<TensorId>,
    ref_count: AtomicU32,
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

#[cfg(feature = "std")]
pub fn insert(tensor_id: TensorId, tensor_: Tensor_) {
    TENSORS.insert(tensor_id, tensor_);
}

#[cfg(not(feature = "std"))]
pub fn insert(tensor_id: TensorId, tensor_: Tensor_) {
    let mut tensors = TENSORS.write();
    tensors.insert(tensor_id, tensor_);
}

// Check if a tensor exists in the global store
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

    pub fn strides(&self) -> Vec<u32> {
        with_tensor(self.0, |t| t.layout.strides().to_vec()).unwrap_or_else(Vec::new)
    }

    pub fn offset(&self) -> u32 {
        with_tensor(self.0, |t| t.layout.offset()).unwrap_or(0)
    }

    pub fn ndim(&self) -> u32 {
        with_tensor(self.0, |t| t.layout.shape().ndim()).unwrap_or(0)
    }

    pub fn dim(&self, index: u32) -> Option<u32> {
        with_tensor(self.0, |t| t.layout.shape().dim(index)).unwrap_or(None)
    }

    pub fn size(&self) -> u32 {
        with_tensor(self.0, |t| t.layout.shape().size()).unwrap_or(1)
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
                Ok(Tensor(grad_id))
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

        if let Ok(grad_tensor) = self.grad() {
            let zeros = Self::zeros(grad_tensor.shape(), grad_tensor.dtype())?;

            with_tensor_mut(self.0, |tensor_ref| {
                tensor_ref.grad_tensor_id = Some(zeros.0);
            });
        }

        Ok(())
    }
}

pub(crate) fn from_storage(storage: BackendStorage, layout: Layout, is_runtime: bool, requires_grad: bool) -> Tensor {
    let tensor_ = Tensor_ {
        storage: Some(Arc::new(storage)),
        is_runtime,
        layout,
        requires_grad,
        grad_tensor_id: None,
        ref_count: AtomicU32::new(1),
    };
    let tensor_id = TensorId::new();
    insert(tensor_id, tensor_);
    Tensor(tensor_id)
}

pub(crate) fn from_shared_storage_with(source_tensor: &Tensor, layout: Layout, requires_grad: bool) -> Tensor {
    let storage_arc =
        with_tensor(source_tensor.id(), |tensor_ref| tensor_ref.storage.clone()).expect("Source tensor not found");

    let tensor_ = Tensor_ {
        storage: storage_arc,
        is_runtime: true,
        layout,
        requires_grad,
        grad_tensor_id: None,
        ref_count: AtomicU32::new(1),
    };

    let tensor_id = TensorId::new();
    insert(tensor_id, tensor_);
    Tensor(tensor_id)
}

pub(crate) fn create_builder_tensor(layout: Layout, requires_grad: bool) -> (TensorId, Tensor) {
    let tensor_ = Tensor_ {
        storage: None,
        is_runtime: false,
        layout,
        requires_grad,
        grad_tensor_id: None,
        ref_count: AtomicU32::new(1),
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
    Tensor(tensor_id)
}

pub(crate) fn set_grad_tensor_id(tensor_id: TensorId, grad_tensor_id: TensorId) -> HoduResult<()> {
    if let Some(result) = with_tensor_mut(tensor_id, |tensor_ref| {
        tensor_ref.grad_tensor_id = Some(grad_tensor_id);
        Ok(())
    }) {
        result
    } else {
        Err(HoduError::TensorNotFound(tensor_id))
    }
}
