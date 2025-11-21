mod board;
mod storage;

pub use board::{CaptureBoard, CaptureBoardId, CapturedInput, CapturedOp, CapturedOutput};
use storage::set_active;
pub use storage::{
    active_board_id, capture_operation, get_active_board, is_active, register_board, take_board, ActiveBoardGuard,
};

impl CaptureBoard {
    /// Start capturing operations on this board
    pub fn start(&self) {
        set_active(Some(self.id));
    }

    /// Stop capturing operations
    pub fn stop(&self) {
        set_active(None);
    }
}
