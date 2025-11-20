use super::{BuilderId, BuilderState};
use crate::{compat::*, error::HoduError, error::HoduResult};

#[cfg(feature = "std")]
use dashmap::DashMap;

/// Global storage for all builder states (std version uses DashMap for concurrent access)
#[cfg(feature = "std")]
static BUILDERS: LazyLock<DashMap<BuilderId, BuilderState>> =
    LazyLock::new(|| DashMap::with_capacity_and_shard_amount(1 << 8, 16));

/// Global storage for all builder states (no_std version uses Mutex + HashMap)
#[cfg(not(feature = "std"))]
static BUILDERS: LazyLock<Mutex<HashMap<BuilderId, BuilderState>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Tracks the currently active builder ID
static ACTIVE_BUILDER_ID: Mutex<Option<BuilderId>> = Mutex::new(None);

/// Insert a builder state into global storage (std version)
#[cfg(feature = "std")]
pub(super) fn insert(builder_id: BuilderId, state: BuilderState) {
    BUILDERS.insert(builder_id, state);
}

/// Insert a builder state into global storage (no_std version)
#[cfg(not(feature = "std"))]
pub(super) fn insert(builder_id: BuilderId, state: BuilderState) {
    let mut builders = BUILDERS.lock();
    builders.insert(builder_id, state);
}

/// Get a builder handle from an ID
pub(super) fn get(builder_id: BuilderId) -> super::Builder {
    super::Builder::from_id(builder_id)
}

/// Get immutable access to a builder state (std version)
#[cfg(feature = "std")]
pub(super) fn with_state<F, R>(builder_id: BuilderId, f: F) -> Option<R>
where
    F: FnOnce(&BuilderState) -> R,
{
    let state = BUILDERS.get(&builder_id)?;
    Some(f(&state))
}

/// Get immutable access to a builder state (no_std version)
#[cfg(not(feature = "std"))]
pub(super) fn with_state<F, R>(builder_id: BuilderId, f: F) -> Option<R>
where
    F: FnOnce(&BuilderState) -> R,
{
    let builders = BUILDERS.lock();
    let state = builders.get(&builder_id)?;
    Some(f(state))
}

/// Get mutable access to a builder state (std version)
#[cfg(feature = "std")]
pub(super) fn with_state_mut<F, R>(builder_id: BuilderId, f: F) -> Option<R>
where
    F: FnOnce(&mut BuilderState) -> R,
{
    let mut state = BUILDERS.get_mut(&builder_id)?;
    Some(f(&mut state))
}

/// Get mutable access to a builder state (no_std version)
#[cfg(not(feature = "std"))]
pub(super) fn with_state_mut<F, R>(builder_id: BuilderId, f: F) -> Option<R>
where
    F: FnOnce(&mut BuilderState) -> R,
{
    let mut builders = BUILDERS.lock();
    let state = builders.get_mut(&builder_id)?;
    Some(f(state))
}

/// Get the currently active builder
pub fn get_active_builder() -> HoduResult<super::Builder> {
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

/// Execute a function with the active builder's mutable state
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

/// Check if there is an active builder
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

/// Lock the active builder ID for modification (std version)
#[cfg(feature = "std")]
pub(super) fn lock_active_builder_id() -> std::sync::MutexGuard<'static, Option<BuilderId>> {
    ACTIVE_BUILDER_ID.lock().unwrap()
}

/// Lock the active builder ID for modification (no_std version)
#[cfg(not(feature = "std"))]
pub(super) fn lock_active_builder_id() -> spin::MutexGuard<'static, Option<BuilderId>> {
    ACTIVE_BUILDER_ID.lock()
}
