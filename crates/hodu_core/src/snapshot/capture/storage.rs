use super::board::{CaptureBoardId, CaptureBoard_, CapturedOp};
use crate::{
    error::{HoduError, HoduResult},
    ops::{Op, OpParams},
    tensor::{Tensor, TensorId},
    types::Layout,
};
use dashmap::DashMap;
use std::sync::LazyLock;

// ============================================================================
// Global Storage
// ============================================================================

static BOARDS: LazyLock<DashMap<CaptureBoardId, CaptureBoard_>> =
    LazyLock::new(|| DashMap::with_capacity_and_shard_amount(1 << 4, 4));

thread_local! {
    static ACTIVE_BOARD: std::cell::RefCell<Option<CaptureBoardId>> = const { std::cell::RefCell::new(None) };
}

// ============================================================================
// Active Board Management
// ============================================================================

/// Check if any capture board is currently active
pub fn is_active() -> bool {
    ACTIVE_BOARD.with(|cell| cell.borrow().is_some())
}

/// Get the currently active board ID
pub fn active_board_id() -> Option<CaptureBoardId> {
    ACTIVE_BOARD.with(|cell| *cell.borrow())
}

/// Set the active board ID
pub(super) fn set_active(id: Option<CaptureBoardId>) {
    ACTIVE_BOARD.with(|cell| *cell.borrow_mut() = id);
}

// ============================================================================
// Board Registration and Retrieval
// ============================================================================

/// Register a new board in global storage
pub(super) fn register_board(board: CaptureBoard_) {
    let id = board.id;
    BOARDS.insert(id, board);
}

/// Take a board out of storage (removes it)
pub(super) fn take_board(id: CaptureBoardId) -> Option<CaptureBoard_> {
    BOARDS.remove(&id).map(|(_, b)| b)
}

// ============================================================================
// Board Mutation Helpers
// ============================================================================

/// Add a target to a specific board
pub(super) fn add_target_to_board(id: CaptureBoardId, name: String, tensor: Tensor) {
    if let Some(mut board) = BOARDS.get_mut(&id) {
        board.add_target(&name, tensor);
    }
}

/// Add an input to the currently active board
pub fn add_input_to_active(name: &str, tensor: Tensor) -> HoduResult<()> {
    let board_id = active_board_id().ok_or(HoduError::CaptureNotActive)?;

    if let Some(mut board) = BOARDS.get_mut(&board_id) {
        board.add_input(name, tensor)
    } else {
        Err(HoduError::CaptureNotActive)
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

    if let Some(mut board) = BOARDS.get_mut(&board_id) {
        board.add_op(captured);
    }

    Ok(())
}
