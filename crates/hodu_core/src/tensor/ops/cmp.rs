use crate::{
    error::HoduResult,
    ops::{CmpOp, CmpScalarOp, Op},
    scalar::Scalar,
    script::builder,
    tensor::{create_builder_tensor, from_storage, register_operation_in_builder, utils::broadcast_tensors2, Tensor},
    types::Layout,
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_same_device, validate_same_dtype},
};

macro_rules! cmp_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> HoduResult<Self> {
            validate_same_device(&[self, rhs], Op::Cmp(CmpOp::$op_name))?;
            validate_same_dtype(&[self, rhs], Op::Cmp(CmpOp::$op_name))?;
            validate_dtype_for_device(self.dtype(), self.device())?;
            validate_dtype_for_op(self.dtype(), Op::Cmp(CmpOp::$op_name))?;

            let (lhs, rhs) = broadcast_tensors2(self, rhs)?;

            if builder::is_builder_active() {
                let result_layout = lhs.layout();
                let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), false);

                register_operation_in_builder(
                    Op::Cmp(CmpOp::$op_name),
                    None,
                    vec![lhs.id(), rhs.id()],
                    vec![result_id],
                    vec![lhs.layout(), rhs.layout()],
                    vec![result_layout],
                )?;

                Ok(result_tensor)
            } else {
                let storage = lhs.with_storage(|lhs_storage| {
                    rhs.with_storage(|rhs_storage| {
                        lhs_storage.call_cmp(
                            rhs_storage,
                            &lhs.layout(),
                            &rhs.layout(),
                            Op::Cmp(CmpOp::$op_name),
                        )
                    })
                })?;

                let result = from_storage(storage, lhs.layout(), true, false);

                Ok(result)
            }
        }
    };
}

macro_rules! cmp_scalar_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name<T: Into<Scalar>>(&self, scalar: T) -> HoduResult<Self> {
            validate_dtype_for_device(self.dtype(), self.device())?;
            validate_dtype_for_op(self.dtype(), Op::CmpScalar(CmpScalarOp::$op_name))?;

            let scalar = scalar.into();

            if builder::is_builder_active() {
                let result_layout = Layout::from_shape(&self.shape());
                let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), false);

                let mut op_params = crate::ops::OpParams::default();
                op_params.scalar = Some(scalar);

                register_operation_in_builder(
                    Op::CmpScalar(CmpScalarOp::$op_name),
                    Some(op_params),
                    vec![self.id()],
                    vec![result_id],
                    vec![self.layout()],
                    vec![result_layout],
                )?;

                Ok(result_tensor)
            } else {
                let storage = self.with_storage(|storage| {
                    storage.call_cmp_scalar(&self.layout(), scalar, Op::CmpScalar(CmpScalarOp::$op_name))
                })?;

                let layout = Layout::from_shape(&self.shape());

                let result = from_storage(storage, layout, true, false);

                Ok(result)
            }
        }
    };
}

impl Tensor {
    // comparison operations
    cmp_op!(eq, Eq);
    cmp_op!(ne, Ne);
    cmp_op!(lt, Lt);
    cmp_op!(le, Le);
    cmp_op!(gt, Gt);
    cmp_op!(ge, Ge);

    // comparison scalar operations
    cmp_scalar_op!(eq_scalar, EqScalar);
    cmp_scalar_op!(ne_scalar, NeScalar);
    cmp_scalar_op!(lt_scalar, LtScalar);
    cmp_scalar_op!(le_scalar, LeScalar);
    cmp_scalar_op!(gt_scalar, GtScalar);
    cmp_scalar_op!(ge_scalar, GeScalar);
}
