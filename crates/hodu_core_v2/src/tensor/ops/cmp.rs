use crate::{
    error::HoduResult,
    ops::{CmpOp, CmpScalarOp, Op},
    scalar::Scalar,
    tensor::{from_storage, utils::broadcast_tensors2, Tensor},
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
    };
}

macro_rules! cmp_scalar_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name<T: Into<Scalar>>(&self, scalar: T) -> HoduResult<Self> {
            validate_dtype_for_device(self.dtype(), self.device())?;
            validate_dtype_for_op(self.dtype(), Op::CmpScalar(CmpScalarOp::$op_name))?;

            let storage = self.with_storage(|storage| {
                storage.call_cmp_scalar(
                    &self.layout(),
                    scalar.into(),
                    Op::CmpScalar(CmpScalarOp::$op_name),
                )
            })?;

            let layout = Layout::from_shape(&self.shape());

            let result = from_storage(storage, layout, true, false);

            Ok(result)
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
