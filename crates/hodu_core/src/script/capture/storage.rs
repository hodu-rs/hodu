use super::board::{CaptureBoard, CaptureBoardId, CapturedOp};
use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    ops::{Op, OpParams},
    tensor::{Tensor, TensorId},
    types::Layout,
};
#[cfg(feature = "std")]
use dashmap::DashMap;

// Global storage (std version with DashMap)
#[cfg(feature = "std")]
static BOARDS: LazyLock<DashMap<CaptureBoardId, CaptureBoard>> =
    LazyLock::new(|| DashMap::with_capacity_and_shard_amount(1 << 4, 4));

// Global storage (no_std version with Mutex + HashMap)
#[cfg(not(feature = "std"))]
static BOARDS: LazyLock<Mutex<HashMap<CaptureBoardId, CaptureBoard>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

// Active board tracker
static ACTIVE_BOARD: Mutex<Option<CaptureBoardId>> = Mutex::new(None);

/// Check if any capture board is active (std version)
#[cfg(feature = "std")]
pub fn is_active() -> bool {
    ACTIVE_BOARD.lock().unwrap().is_some()
}

/// Check if any capture board is active (no_std version)
#[cfg(not(feature = "std"))]
pub fn is_active() -> bool {
    ACTIVE_BOARD.lock().is_some()
}

/// Get the active board ID (std version)
#[cfg(feature = "std")]
pub fn active_board_id() -> Option<CaptureBoardId> {
    *ACTIVE_BOARD.lock().unwrap()
}

/// Get the active board ID (no_std version)
#[cfg(not(feature = "std"))]
pub fn active_board_id() -> Option<CaptureBoardId> {
    *ACTIVE_BOARD.lock()
}

/// Set active board (std version)
#[cfg(feature = "std")]
pub(super) fn set_active(id: Option<CaptureBoardId>) {
    let mut guard = ACTIVE_BOARD.lock().unwrap();
    *guard = id;
}

/// Set active board (no_std version)
#[cfg(not(feature = "std"))]
pub(super) fn set_active(id: Option<CaptureBoardId>) {
    let mut guard = ACTIVE_BOARD.lock();
    *guard = id;
}

/// Register a board for capturing (std version)
#[cfg(feature = "std")]
pub fn register_board(board: CaptureBoard) -> CaptureBoardId {
    let id = board.id;
    BOARDS.insert(id, board);
    id
}

/// Register a board for capturing (no_std version)
#[cfg(not(feature = "std"))]
pub fn register_board(board: CaptureBoard) -> CaptureBoardId {
    let id = board.id;
    BOARDS.lock().insert(id, board);
    id
}

/// Capture an operation to the active board (std version)
#[cfg(feature = "std")]
pub fn capture_operation(
    op: Op,
    params: Option<OpParams>,
    input_ids: Vec<TensorId>,
    output_id: TensorId,
    input_layouts: Vec<Layout>,
    output_layout: Layout,
) -> HoduResult<()> {
    let board_id = match active_board_id() {
        Some(id) => id,
        None => return Ok(()),
    };

    let captured = CapturedOp {
        op,
        params,
        input_ids,
        output_id,
        input_layouts,
        output_layout,
    };

    if let Some(mut board) = BOARDS.get_mut(&board_id) {
        board.add_op(captured);
    }

    Ok(())
}

/// Capture an operation to the active board (no_std version)
#[cfg(not(feature = "std"))]
pub fn capture_operation(
    op: Op,
    params: Option<OpParams>,
    input_ids: Vec<TensorId>,
    output_id: TensorId,
    input_layouts: Vec<Layout>,
    output_layout: Layout,
) -> HoduResult<()> {
    let board_id = match active_board_id() {
        Some(id) => id,
        None => return Ok(()),
    };

    let captured = CapturedOp {
        op,
        params,
        input_ids,
        output_id,
        input_layouts,
        output_layout,
    };

    if let Some(board) = BOARDS.lock().get_mut(&board_id) {
        board.add_op(captured);
    }

    Ok(())
}

/// Take a board out of the registry (std version)
#[cfg(feature = "std")]
pub fn take_board(id: CaptureBoardId) -> Option<CaptureBoard> {
    BOARDS.remove(&id).map(|(_, b)| b)
}

/// Take a board out of the registry (no_std version)
#[cfg(not(feature = "std"))]
pub fn take_board(id: CaptureBoardId) -> Option<CaptureBoard> {
    BOARDS.lock().remove(&id)
}

/// Wrapper for mutable access to active board
pub struct ActiveBoardGuard;

impl ActiveBoardGuard {
    /// Add input to active board (std version)
    #[cfg(feature = "std")]
    pub fn add_input(&self, name: &str, tensor: Tensor) -> HoduResult<()> {
        let board_id = active_board_id().ok_or(HoduError::BuilderNotActive)?;
        if let Some(mut board) = BOARDS.get_mut(&board_id) {
            board.add_input(name, tensor)
        } else {
            Err(HoduError::BuilderNotActive)
        }
    }

    /// Add input to active board (no_std version)
    #[cfg(not(feature = "std"))]
    pub fn add_input(&self, name: &str, tensor: Tensor) -> HoduResult<()> {
        let board_id = active_board_id().ok_or(HoduError::BuilderNotActive)?;
        if let Some(board) = BOARDS.lock().get_mut(&board_id) {
            board.add_input(name, tensor)
        } else {
            Err(HoduError::BuilderNotActive)
        }
    }
}

/// Get active board for modification
pub fn get_active_board() -> HoduResult<ActiveBoardGuard> {
    if is_active() {
        Ok(ActiveBoardGuard)
    } else {
        Err(HoduError::BuilderNotActive)
    }
}
