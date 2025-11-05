use super::{super::context::BuilderState, super::ir::*};
use crate::{error::HoduResult, tensor::TensorId};

use super::tracing::allocate_value_id;

/// Load constant tensor
pub fn load_constant(state: &mut BuilderState, block: &mut BasicBlock, tensor_id: TensorId) -> HoduResult<ValueId> {
    // Add constant data to module if not already present
    if !state.module.constants.contains_key(&tensor_id) {
        let tensor = crate::tensor::tensor_from_id(tensor_id);
        let layout = tensor.layout();

        // Get CPU storage data
        let cpu_storage = tensor.with_storage(|storage| storage.to_cpu_storage())?;

        let constant = ConstantData {
            tensor_id,
            shape: layout.shape().clone(),
            dtype: tensor.dtype(),
            data: cpu_storage.to_bytes(),
            compression: None,
        };

        state.module.add_constant(tensor_id, constant);
    }

    // Allocate value and emit load instruction
    let value_id = allocate_value_id(state, tensor_id);
    block.add_instruction(Instruction::LoadConstant {
        result: value_id,
        tensor_id,
    });

    Ok(value_id)
}
