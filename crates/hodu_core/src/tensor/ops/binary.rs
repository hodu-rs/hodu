use crate::{
    error::HoduResult,
    ops::{BinaryLogicalOp, BinaryLogicalParams, BinaryOp, BinaryParams, Op, OpParams},
    tensor::{create_builder_tensor, from_storage_with_context, gradient, utils::broadcast_tensors2, Tensor},
    types::DType,
    utils::valid::{
        validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op, validate_same_device,
        validate_same_dtype,
    },
};

macro_rules! binary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> HoduResult<Self> {
            validate_same_device(&[self, rhs], Op::Binary(BinaryOp::$op_name))?;
            validate_same_dtype(&[self, rhs], Op::Binary(BinaryOp::$op_name))?;
            validate_dtype_for_device(self.dtype(), self.device())?;
            validate_dtype_for_op(self.dtype(), Op::Binary(BinaryOp::$op_name))?;
            let validate_requires_grad = validate_requires_grad_for_op(Op::Binary(BinaryOp::$op_name));

            let (lhs, rhs) = broadcast_tensors2(self, rhs)?;

            if crate::snapshot::capture::is_active() {
                let lhs_layout = lhs.layout();
                let rhs_layout = rhs.layout();
                let requires_grad = (lhs.is_requires_grad() || rhs.is_requires_grad()) && validate_requires_grad;
                let result_layout = lhs_layout.clone();
                let (result_id, result_tensor) =
                    create_builder_tensor(result_layout.clone(), lhs.dtype(), requires_grad);

                crate::snapshot::capture::capture_operation(
                    Op::Binary(BinaryOp::$op_name),
                    Some(OpParams::Binary(BinaryParams)),
                    vec![lhs.id(), rhs.id()],
                    result_id,
                    vec![lhs_layout, rhs_layout],
                    result_layout,
                )?;

                if requires_grad {
                    gradient::record_operation(
                        vec![lhs.id(), rhs.id()],
                        result_id,
                        Op::Binary(BinaryOp::$op_name),
                        OpParams::Binary(BinaryParams),
                    )?;
                }

                Ok(result_tensor)
            } else {
                let lhs_layout = lhs.layout();
                let rhs_layout = rhs.layout();

                let storage = lhs.with_storage(|lhs_storage| {
                    rhs.with_storage(|rhs_storage| {
                        lhs_storage.call_ops_binary(
                            rhs_storage,
                            &lhs_layout,
                            &rhs_layout,
                            Op::Binary(BinaryOp::$op_name),
                        )
                    })
                })?;

                let requires_grad = lhs.is_requires_grad() || rhs.is_requires_grad();
                let requires_grad = requires_grad && validate_requires_grad;

                let result = from_storage_with_context(storage, lhs_layout, true, requires_grad);

                if !gradient::is_computing_gradients() && requires_grad {
                    gradient::record_operation(
                        vec![lhs.id(), rhs.id()],
                        result.id(),
                        Op::Binary(BinaryOp::$op_name),
                        OpParams::Binary(BinaryParams),
                    )?;
                }

                Ok(result)
            }
        }
    };
}

macro_rules! binary_logical_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> HoduResult<Self> {
            validate_same_device(&[self, rhs], Op::BinaryLogical(BinaryLogicalOp::$op_name))?;
            validate_same_dtype(&[self, rhs], Op::BinaryLogical(BinaryLogicalOp::$op_name))?;
            validate_dtype_for_device(self.dtype(), self.device())?;
            validate_dtype_for_op(self.dtype(), Op::BinaryLogical(BinaryLogicalOp::$op_name))?;

            let (lhs, rhs) = broadcast_tensors2(self, rhs)?;

            if crate::snapshot::capture::is_active() {
                let lhs_layout = lhs.layout();
                let rhs_layout = rhs.layout();
                let result_layout = lhs_layout.clone();
                let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), DType::BOOL, false);

                crate::snapshot::capture::capture_operation(
                    Op::BinaryLogical(BinaryLogicalOp::$op_name),
                    Some(OpParams::BinaryLogical(BinaryLogicalParams)),
                    vec![lhs.id(), rhs.id()],
                    result_id,
                    vec![lhs_layout, rhs_layout],
                    result_layout,
                )?;

                Ok(result_tensor)
            } else {
                let lhs_layout = lhs.layout();
                let rhs_layout = rhs.layout();

                let storage = lhs.with_storage(|lhs_storage| {
                    rhs.with_storage(|rhs_storage| {
                        lhs_storage.call_ops_binary_logical(
                            rhs_storage,
                            &lhs_layout,
                            &rhs_layout,
                            Op::BinaryLogical(BinaryLogicalOp::$op_name),
                        )
                    })
                })?;

                let result = from_storage_with_context(storage, lhs_layout, true, false);

                Ok(result)
            }
        }
    };
}

impl Tensor {
    // binary operations
    binary_op!(add, Add);
    binary_op!(sub, Sub);
    binary_op!(mul, Mul);
    binary_op!(div, Div);
    binary_op!(rem, Rem);
    binary_op!(pow, Pow);
    binary_op!(maximum, Maximum);
    binary_op!(minimum, Minimum);

    // binary logical operations
    binary_logical_op!(logical_and, LogicalAnd);
    binary_logical_op!(logical_or, LogicalOr);
    binary_logical_op!(logical_xor, LogicalXor);
}
