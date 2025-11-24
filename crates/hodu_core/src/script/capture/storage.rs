use super::board::{CaptureBoardId, CaptureBoard_, CapturedOp};
use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    ops::{Op, OpParams},
    tensor::{Tensor, TensorId},
    types::Layout,
};

#[cfg(feature = "std")]
use dashmap::DashMap;

// ============================================================================
// Global Storage
// ============================================================================

#[cfg(feature = "std")]
static BOARDS: LazyLock<DashMap<CaptureBoardId, CaptureBoard_>> =
    LazyLock::new(|| DashMap::with_capacity_and_shard_amount(1 << 4, 4));

#[cfg(not(feature = "std"))]
static BOARDS: LazyLock<Mutex<HashMap<CaptureBoardId, CaptureBoard_>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

static ACTIVE_BOARD: Mutex<Option<CaptureBoardId>> = Mutex::new(None);

// ============================================================================
// Active Board Management
// ============================================================================

/// Check if any capture board is currently active
pub fn is_active() -> bool {
    #[cfg(feature = "std")]
    {
        ACTIVE_BOARD.lock().unwrap().is_some()
    }
    #[cfg(not(feature = "std"))]
    {
        ACTIVE_BOARD.lock().is_some()
    }
}

/// Get the currently active board ID
pub fn active_board_id() -> Option<CaptureBoardId> {
    #[cfg(feature = "std")]
    {
        *ACTIVE_BOARD.lock().unwrap()
    }
    #[cfg(not(feature = "std"))]
    {
        *ACTIVE_BOARD.lock()
    }
}

/// Set the active board ID
pub(super) fn set_active(id: Option<CaptureBoardId>) {
    #[cfg(feature = "std")]
    {
        *ACTIVE_BOARD.lock().unwrap() = id;
    }
    #[cfg(not(feature = "std"))]
    {
        *ACTIVE_BOARD.lock() = id;
    }
}

// ============================================================================
// Board Registration and Retrieval
// ============================================================================

/// Register a new board in global storage
pub(super) fn register_board(board: CaptureBoard_) {
    let id = board.id;
    #[cfg(feature = "std")]
    {
        BOARDS.insert(id, board);
    }
    #[cfg(not(feature = "std"))]
    {
        BOARDS.lock().insert(id, board);
    }
}

/// Take a board out of storage (removes it)
pub(super) fn take_board(id: CaptureBoardId) -> Option<CaptureBoard_> {
    #[cfg(feature = "std")]
    {
        BOARDS.remove(&id).map(|(_, b)| b)
    }
    #[cfg(not(feature = "std"))]
    {
        BOARDS.lock().remove(&id)
    }
}

// ============================================================================
// Board Mutation Helpers
// ============================================================================

/// Add a target to a specific board
pub(super) fn add_target_to_board(id: CaptureBoardId, name: String, tensor: Tensor) {
    #[cfg(feature = "std")]
    {
        if let Some(mut board) = BOARDS.get_mut(&id) {
            board.add_target(&name, tensor);
        }
    }
    #[cfg(not(feature = "std"))]
    {
        if let Some(board) = BOARDS.lock().get_mut(&id) {
            board.add_target(&name, tensor);
        }
    }
}

/// Add an input to the currently active board
pub fn add_input_to_active(name: &str, tensor: Tensor) -> HoduResult<()> {
    let board_id = active_board_id().ok_or(HoduError::BuilderNotActive)?;

    #[cfg(feature = "std")]
    {
        if let Some(mut board) = BOARDS.get_mut(&board_id) {
            board.add_input(name, tensor)
        } else {
            Err(HoduError::BuilderNotActive)
        }
    }
    #[cfg(not(feature = "std"))]
    {
        if let Some(board) = BOARDS.lock().get_mut(&board_id) {
            board.add_input(name, tensor)
        } else {
            Err(HoduError::BuilderNotActive)
        }
    }
}

/// Capture an operation to the currently active board
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
        None => return Ok(()), // No active board, silently skip
    };

    let captured = CapturedOp {
        op,
        params,
        input_ids,
        output_id,
        input_layouts,
        output_layout,
    };

    #[cfg(feature = "std")]
    {
        if let Some(mut board) = BOARDS.get_mut(&board_id) {
            board.add_op(captured);
        }
    }
    #[cfg(not(feature = "std"))]
    {
        if let Some(board) = BOARDS.lock().get_mut(&board_id) {
            board.add_op(captured);
        }
    }

    Ok(())
}
