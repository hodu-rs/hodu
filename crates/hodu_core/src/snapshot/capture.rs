mod board;
mod storage;

pub use board::{CaptureBoard, CaptureBoardId, CapturedInput, CapturedOp, CapturedTarget};
pub use storage::{
    active_board_id, add_input_to_active, capture_operation, capture_operation_with_symbolic, is_active,
};
