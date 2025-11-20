use super::{gradient, registry};
use crate::{
    be::storage::BackendStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    types::{DType, Device, Layout, Shape},
};
pub use gradient::ContextId;

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct TensorId(pub(super) usize);

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
pub struct Tensor(pub(super) TensorId);

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        if let Some(()) = registry::with_tensor(self.0, |t| {
            t.ref_count.fetch_add(1, Ordering::Relaxed);
        }) {
            // Tensor exists, ref_count incremented
        }
        Tensor(self.0)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        let (should_remove, should_cleanup_tape, grad_id) = registry::with_tensor(self.0, |t| {
            let prev_count = t.ref_count.load(Ordering::Relaxed);

            if prev_count > 0 {
                t.ref_count.fetch_sub(1, Ordering::Relaxed);
            }

            let remove = if gradient::is_computing_gradients() {
                false
            } else {
                (prev_count == 1 || prev_count == 0) && t.is_runtime && t.owner_context.is_none() && !t.is_gradient
            };

            let cleanup =
                prev_count == 1 && t.requires_grad && t.owner_context.is_none() && !gradient::is_computing_gradients();

            (remove, cleanup, t.grad_tensor_id)
        })
        .unwrap_or((false, false, None));

        if should_cleanup_tape {
            gradient::cleanup_tape_for_dropped_tensor(self.0);
        }

        if should_remove {
            if let Some(grad_id) = grad_id {
                if let Some((should_remove_grad, should_cleanup_grad_tape)) = registry::with_tensor(grad_id, |g| {
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
                        registry::remove(grad_id);
                    }
                }
            }

            registry::remove(self.0);
        }
    }
}

pub struct Tensor_ {
    pub(super) storage: Option<Arc<BackendStorage>>,
    pub(super) layout: Layout,
    pub(super) dtype: Option<DType>, // For builder tensors without storage
    pub(super) requires_grad: bool,
    pub(super) grad_tensor_id: Option<TensorId>,
    pub(super) is_runtime: bool,
    pub(super) is_gradient: bool,
    pub(super) owner_context: Option<ContextId>,
    pub(super) ref_count: AtomicUsize,
}

impl Tensor_ {
    pub(crate) fn owner_context(&self) -> Option<ContextId> {
        self.owner_context
    }
}

impl Tensor {
    #[inline]
    pub fn id(&self) -> TensorId {
        self.0
    }

    pub(crate) fn has_storage(&self) -> bool {
        registry::with_tensor(self.0, |t| t.storage.is_some()).unwrap_or(false)
    }

    pub(crate) fn with_storage<R>(&self, f: impl FnOnce(&BackendStorage) -> HoduResult<R>) -> HoduResult<R> {
        registry::with_tensor(self.0, |tensor_ref| {
            let storage = tensor_ref.storage.as_ref().ok_or(HoduError::StorageNotFound(self.0))?;
            f(storage.as_ref())
        })
        .ok_or(HoduError::TensorNotFound(self.0))?
    }

    pub fn is_runtime(&self) -> bool {
        registry::with_tensor(self.0, |t| t.is_runtime).unwrap_or(false)
    }

    pub fn layout(&self) -> Layout {
        registry::with_tensor(self.0, |t| t.layout.clone()).unwrap_or_else(|| Layout::from_shape(&Shape::scalar()))
    }

    #[inline]
    pub fn with_layout<R>(&self, f: impl FnOnce(&Layout) -> R) -> Option<R> {
        registry::with_tensor(self.0, |t| f(&t.layout))
    }

    pub fn shape(&self) -> Shape {
        registry::with_tensor(self.0, |t| t.layout.shape().clone()).unwrap_or_else(Shape::scalar)
    }

    #[inline]
    pub fn with_shape<R>(&self, f: impl FnOnce(&Shape) -> R) -> Option<R> {
        registry::with_tensor(self.0, |t| f(t.layout.shape()))
    }

    pub fn strides(&self) -> Vec<usize> {
        registry::with_tensor(self.0, |t| t.layout.strides().to_vec()).unwrap_or_else(Vec::new)
    }

    pub fn offset(&self) -> usize {
        registry::with_tensor(self.0, |t| t.layout.offset()).unwrap_or(0)
    }

    pub fn ndim(&self) -> usize {
        registry::with_tensor(self.0, |t| t.layout.ndim()).unwrap_or(0)
    }

    pub fn dim_size(&self, index: i32) -> Option<usize> {
        registry::with_tensor(self.0, |t| t.layout.dim_size(index)).unwrap_or(None)
    }

    pub fn size(&self) -> usize {
        registry::with_tensor(self.0, |t| t.layout.size()).unwrap_or(1)
    }

    pub fn device(&self) -> Device {
        self.with_storage(|storage| Ok(storage.device())).unwrap_or(Device::CPU)
    }

    pub fn dtype(&self) -> DType {
        registry::with_tensor(self.0, |t| {
            // For builder tensors with explicit dtype
            if let Some(dtype) = t.dtype {
                return dtype;
            }
            // For runtime tensors with storage
            if let Some(ref storage) = t.storage {
                return storage.dtype();
            }
            // Default
            DType::F32
        })
        .unwrap_or(DType::F32)
    }

    pub fn set_requires_grad(&self, requires: bool) -> HoduResult<()> {
        if requires {
            let dtype = self.dtype();
            if !dtype.is_float() {
                return Err(HoduError::RequiresGradNotSet(self.0));
            }
        }

        if let Some(result) = registry::with_tensor_mut(self.0, |tensor_ref| {
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
        registry::with_tensor(self.0, |t| t.requires_grad).unwrap_or(false)
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
        if let Some(result) = registry::with_tensor(self.0, |tensor_ref| {
            if let Some(grad_id) = tensor_ref.grad_tensor_id {
                Ok(super::internal::tensor_from_id(grad_id))
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
            return Ok(());
        }

        let old_grad_id = registry::with_tensor_mut(self.0, |tensor_ref| tensor_ref.grad_tensor_id.take());

        if let Some(Some(old_grad_id)) = old_grad_id {
            let (shape, dtype) = registry::with_tensor(old_grad_id, |t| {
                let shape = t.layout.shape().to_vec();
                let dtype = t.storage.as_ref().map(|s| s.dtype()).unwrap_or(DType::F32);
                (shape, dtype)
            })
            .ok_or(HoduError::TensorNotFound(old_grad_id))?;

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

            let zeros = Self::zeros(&shape, dtype)?;
            let zeros_id = zeros.id();

            if let Some(()) = registry::with_tensor_mut(zeros_id, |t| {
                t.is_gradient = true;
                t.ref_count.fetch_add(1, Ordering::Relaxed);
            }) {
                // Successfully set is_gradient
            } else {
                return Err(HoduError::TensorNotFound(zeros_id));
            }

            registry::with_tensor_mut(self.0, |tensor_ref| {
                tensor_ref.grad_tensor_id = Some(zeros_id);
            });
        }

        Ok(())
    }
}
