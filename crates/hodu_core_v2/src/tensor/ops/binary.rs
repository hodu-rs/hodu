use crate::{
    error::HoduResult,
    layer::compat::*,
    ops::{BinaryLogicalOp, BinaryOp, Op},
    tensor::{from_storage, gradient, utils::broadcast_tensors2, Tensor},
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

            let storage = lhs.with_storage(|lhs_storage| {
                rhs.with_storage(|rhs_storage| {
                    lhs_storage.call_binary(
                        rhs_storage,
                        &lhs.layout(),
                        &rhs.layout(),
                        Op::Binary(BinaryOp::$op_name),
                    )
                })
            })?;

            let requires_grad = lhs.is_requires_grad() || rhs.is_requires_grad();
            let requires_grad = requires_grad && validate_requires_grad;

            let result = from_storage(storage, lhs.layout(), true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Binary(BinaryOp::$op_name);
                gradient::record_operation(result.id(), op, vec![self.id(), rhs.id()])?;
            }

            Ok(result)
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

            let storage = lhs.with_storage(|lhs_storage| {
                rhs.with_storage(|rhs_storage| {
                    lhs_storage.call_binary_logical(
                        rhs_storage,
                        &lhs.layout(),
                        &rhs.layout(),
                        Op::BinaryLogical(BinaryLogicalOp::$op_name),
                    )
                })
            })?;

            let result = from_storage(storage, lhs.layout(), true, false);

            Ok(result)
        }
    };
}

impl Tensor {
    // binary operations
    binary_op!(add, Add);
    binary_op!(sub, Sub);
    binary_op!(mul, Mul);
    binary_op!(div, Div);
    binary_op!(pow, Pow);
    binary_op!(maximum, Maximum);
    binary_op!(minimum, Minimum);

    // binary logical operations
    binary_logical_op!(logical_and, LogicalAnd);
    binary_logical_op!(logical_or, LogicalOr);
    binary_logical_op!(logical_xor, LogicalXor);
}
