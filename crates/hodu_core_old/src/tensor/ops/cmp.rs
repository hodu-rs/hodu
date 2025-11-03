use crate::{
    builder,
    compat::*,
    error::HoduResult,
    op::{
        self,
        utils::{validate_dtype_for_device, validate_dtype_for_op, validate_same_device, validate_same_dtype},
        Op,
    },
    scalar::Scalar,
    tensor::{
        create_builder_tensor_with_grad, from_storage_with_grad, gradient, register_operation_in_builder,
        utils::broadcast_tensors2, Tensor,
    },
};

macro_rules! cmp_op {
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
            let op = Op::Cmp(op::CmpOp::$op_name, lhs_broadcasted.id(), rhs_broadcasted.id());
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
                        lhs_storage.cmp_impl::<op::$op_name>(
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
                    let op = Op::Cmp(op::CmpOp::$op_name, self.id(), rhs.id());
                    gradient::record_operation(result.id(), op, vec![self.id(), rhs.id()])?;
                }

                Ok(result)
            }
        }
    };
}

macro_rules! cmp_scalar_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name<T: Into<Scalar>>(&self, scalar: T) -> HoduResult<Self> {
            let scalar = scalar.into();

            // Validate dtype for device and operation
            validate_dtype_for_device(self.get_dtype(), &self.get_device(), stringify!($op_name))?;
            let op = Op::CmpScalar(op::CmpScalarOp::$op_name, self.id(), scalar);
            validate_dtype_for_op(self.get_dtype(), &op)?;

            if builder::is_builder_active() {
                let result_layout = self.get_layout().clone();
                let requires_grad = self.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                register_operation_in_builder(
                    op.clone(),
                    vec![result_id],
                    vec![self.get_layout().clone()],
                    vec![result_layout],
                );

                if self.is_requires_grad() {
                    gradient::record_operation(result_id, op, vec![self.id()])?;
                }

                Ok(result_tensor)
            } else {
                let storage =
                    self.with_storage(|storage| storage.cmp_scalar_impl::<op::$op_name>(&self.get_layout(), scalar))?;
                let layout = self.get_layout().clone();
                let requires_grad = self.is_requires_grad();
                let result = from_storage_with_grad(storage, layout, true, requires_grad);

                if !gradient::is_computing_gradients() && self.is_requires_grad() {
                    let op = Op::CmpScalar(op::CmpScalarOp::$op_name, self.id(), scalar);
                    gradient::record_operation(result.id(), op, vec![self.id()])?;
                }

                Ok(result)
            }
        }
    };
}

impl Tensor {
    // Comparison operations
    cmp_op!(eq, Eq);
    cmp_op!(ne, Ne);
    cmp_op!(lt, Lt);
    cmp_op!(le, Le);
    cmp_op!(gt, Gt);
    cmp_op!(ge, Ge);

    // Comparison scalar operations
    cmp_scalar_op!(eq_scalar, EqScalar);
    cmp_scalar_op!(ne_scalar, NeScalar);
    cmp_scalar_op!(lt_scalar, LtScalar);
    cmp_scalar_op!(le_scalar, LeScalar);
    cmp_scalar_op!(gt_scalar, GtScalar);
    cmp_scalar_op!(ge_scalar, GeScalar);
}
