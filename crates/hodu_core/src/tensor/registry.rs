use super::core::{TensorId, Tensor_};
use dashmap::DashMap;
use std::sync::atomic::Ordering;
use std::sync::LazyLock;

static TENSORS: LazyLock<DashMap<TensorId, Tensor_>> = LazyLock::new(|| {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(64);
    let shard_count = cores.next_power_of_two();
    DashMap::with_capacity_and_shard_amount(1 << 16, shard_count)
});

pub fn insert(tensor_id: TensorId, tensor_: Tensor_) {
    TENSORS.insert(tensor_id, tensor_);
}

pub fn exists(tensor_id: TensorId) -> bool {
    TENSORS.contains_key(&tensor_id)
}

pub(crate) fn remove(tensor_id: TensorId) {
    if let Some(Some(grad_id)) = with_tensor(tensor_id, |t| t.grad_tensor_id) {
        with_tensor(grad_id, |t| {
            t.ref_count.fetch_sub(1, Ordering::Relaxed);
        });
    }
    TENSORS.remove(&tensor_id);
}

pub fn get(tensor_id: TensorId) -> Option<dashmap::mapref::one::Ref<'static, TensorId, Tensor_>> {
    TENSORS.get(&tensor_id)
}

pub fn shrink_tensor_storage() {
    TENSORS.shrink_to_fit();
}

pub fn tensor_count() -> usize {
    TENSORS.len()
}

pub(crate) fn get_all_tensor_ids() -> Vec<TensorId> {
    TENSORS.iter().map(|entry| *entry.key()).collect()
}

pub fn with_tensor<F, R>(tensor_id: TensorId, f: F) -> Option<R>
where
    F: FnOnce(&Tensor_) -> R,
{
    TENSORS.get(&tensor_id).map(|tensor_ref| f(&tensor_ref))
}

pub fn with_tensor_mut<F, R>(tensor_id: TensorId, f: F) -> Option<R>
where
    F: FnOnce(&mut Tensor_) -> R,
{
    TENSORS.get_mut(&tensor_id).map(|mut tensor_ref| f(&mut tensor_ref))
}

/// Get the dtype of a tensor from the registry
pub fn get_dtype(tensor_id: TensorId) -> Option<crate::types::DType> {
    with_tensor(tensor_id, |t| t.dtype).flatten()
}
