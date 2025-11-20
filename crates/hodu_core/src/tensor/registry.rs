use super::core::{TensorId, Tensor_};
use crate::layer::compat::*;

#[cfg(feature = "std")]
use dashmap::DashMap;

#[cfg(feature = "std")]
static TENSORS: LazyLock<DashMap<TensorId, Tensor_>> = LazyLock::new(|| {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(64);
    let shard_count = cores.next_power_of_two();
    DashMap::with_capacity_and_shard_amount(1 << 16, shard_count)
});

#[cfg(not(feature = "std"))]
static TENSORS: LazyLock<RwLock<HashMap<TensorId, Tensor_>>> = LazyLock::new(|| RwLock::new(HashMap::new()));

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
    if let Some(Some(grad_id)) = with_tensor(tensor_id, |t| t.grad_tensor_id) {
        with_tensor(grad_id, |t| {
            t.ref_count.fetch_sub(1, Ordering::Relaxed);
        });
    }

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
