use crate::{
    builder,
    compat::*,
    error::HoduResult,
    op::{
        self,
        utils::{validate_dtype_for_device, validate_dtype_for_op, validate_same_device, validate_same_dtype},
        Op,
    },
    tensor::{
        create_builder_tensor_with_grad, from_storage_with_grad, gradient, register_operation_in_builder,
        utils::broadcast_tensors2, Tensor,
    },
    types::dtype::DType,
};

macro_rules! binary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> HoduResult<Self> {
            let (lhs_broadcasted, rhs_broadcasted) = broadcast_tensors2(self, rhs)?;

            let (lhs_broadcasted, rhs_broadcasted) =
                if lhs_broadcasted.get_dtype() == DType::BOOL && rhs_broadcasted.get_dtype() != DType::BOOL {
                    let lhs_broadcasted = lhs_broadcasted.to_dtype(rhs_broadcasted.get_dtype())?;
                    (lhs_broadcasted, rhs_broadcasted)
                } else if lhs_broadcasted.get_dtype() != DType::BOOL && rhs_broadcasted.get_dtype() == DType::BOOL {
                    let rhs_broadcasted = rhs_broadcasted.to_dtype(lhs_broadcasted.get_dtype())?;
                    (lhs_broadcasted, rhs_broadcasted)
                } else {
                    (lhs_broadcasted, rhs_broadcasted)
                };

            // Validate device, dtype for device, and dtype for operation
            validate_same_device(&[&lhs_broadcasted, &rhs_broadcasted], stringify!($op_name))?;
            validate_dtype_for_device(
                lhs_broadcasted.get_dtype(),
                &lhs_broadcasted.get_device(),
                stringify!($op_name),
            )?;
            let op = Op::Binary(op::BinaryOp::$op_name, lhs_broadcasted.id(), rhs_broadcasted.id());
            validate_dtype_for_op(lhs_broadcasted.get_dtype(), &op)?;
            validate_same_dtype(&[&lhs_broadcasted, &rhs_broadcasted], stringify!($op_name))?;

            if builder::is_builder_active() {
                let result_layout = lhs_broadcasted.get_layout().clone();
                let requires_grad = self.is_requires_grad() || rhs.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                register_operation_in_builder(
                    op.clone(),
                    vec![result_id],
                    vec![
                        lhs_broadcasted.get_layout().clone(),
                        rhs_broadcasted.get_layout().clone(),
                    ],
                    vec![result_layout],
                );

                if self.is_requires_grad() || rhs.is_requires_grad() {
                    gradient::record_operation(result_id, op, vec![self.id(), rhs.id()])?;
                }

                Ok(result_tensor)
            } else {
                let storage = lhs_broadcasted.with_storage(|lhs_storage| {
                    rhs_broadcasted.with_storage(|rhs_storage| {
                        lhs_storage.binary_impl::<op::$op_name>(
                            rhs_storage,
                            &lhs_broadcasted.get_layout(),
                            &rhs_broadcasted.get_layout(),
                        )
                    })
                })?;
                let layout = lhs_broadcasted.get_layout().clone();
                let requires_grad = self.is_requires_grad() || rhs.is_requires_grad();
                let result = from_storage_with_grad(storage, layout, true, requires_grad);

                if !gradient::is_computing_gradients() && (self.is_requires_grad() || rhs.is_requires_grad()) {
                    let op = Op::Binary(op::BinaryOp::$op_name, self.id(), rhs.id());
                    gradient::record_operation(result.id(), op, vec![self.id(), rhs.id()])?;
                }

                Ok(result)
            }
        }
    };
}

macro_rules! binary_logical_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> HoduResult<Self> {
            let (lhs_broadcasted, rhs_broadcasted) = broadcast_tensors2(self, rhs)?;

            // Validate device, dtype for device, and dtype for operation
            validate_same_device(&[&lhs_broadcasted, &rhs_broadcasted], stringify!($op_name))?;
            validate_dtype_for_device(
                lhs_broadcasted.get_dtype(),
                &lhs_broadcasted.get_device(),
                stringify!($op_name),
            )?;
            let op = Op::BinaryLogical(
                op::BinaryLogicalOp::$op_name,
                lhs_broadcasted.id(),
                rhs_broadcasted.id(),
            );
            validate_dtype_for_op(lhs_broadcasted.get_dtype(), &op)?;
            validate_same_dtype(&[&lhs_broadcasted, &rhs_broadcasted], stringify!($op_name))?;

            if builder::is_builder_active() {
                let result_layout = lhs_broadcasted.get_layout().clone();
                let requires_grad = self.is_requires_grad() || rhs.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                register_operation_in_builder(
                    op.clone(),
                    vec![result_id],
                    vec![
                        lhs_broadcasted.get_layout().clone(),
                        rhs_broadcasted.get_layout().clone(),
                    ],
                    vec![result_layout],
                );

                if self.is_requires_grad() || rhs.is_requires_grad() {
                    gradient::record_operation(result_id, op, vec![self.id(), rhs.id()])?;
                }

                Ok(result_tensor)
            } else {
                let storage = lhs_broadcasted.with_storage(|lhs_storage| {
                    rhs_broadcasted.with_storage(|rhs_storage| {
                        lhs_storage.binary_logical_impl::<op::$op_name>(
                            rhs_storage,
                            &lhs_broadcasted.get_layout(),
                            &rhs_broadcasted.get_layout(),
                        )
                    })
                })?;
                let layout = lhs_broadcasted.get_layout().clone();
                let requires_grad = self.is_requires_grad() || rhs.is_requires_grad();
                let result = from_storage_with_grad(storage, layout, true, requires_grad);

                if !gradient::is_computing_gradients() && (self.is_requires_grad() || rhs.is_requires_grad()) {
                    let op = Op::BinaryLogical(op::BinaryLogicalOp::$op_name, self.id(), rhs.id());
                    gradient::record_operation(result.id(), op, vec![self.id(), rhs.id()])?;
                }

                Ok(result)
            }
        }
    };
}

impl Tensor {
    // Binary operations
    binary_op!(add, Add);
    binary_op!(sub, Sub);
    binary_op!(mul, Mul);
    binary_op!(div, Div);
    binary_op!(pow, Pow);
    binary_op!(maximum, Maximum);
    binary_op!(minimum, Minimum);

    // Binary logical operations
    binary_logical_op!(logical_and, LogicalAnd);
    binary_logical_op!(logical_or, LogicalOr);
    binary_logical_op!(logical_xor, LogicalXor);
}
