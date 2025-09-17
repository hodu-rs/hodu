use crate::{
    backends::{builder, op, op::Op},
    compat::*,
    error::{HoduError, HoduResult},
    scalar::Scalar,
    tensor::{
        create_builder_tensor_with_grad, from_storage_with_grad, gradient, register_operation_in_builder, Tensor,
    },
    types::dtype::DType,
};

macro_rules! binary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> HoduResult<Self> {
            if builder::is_builder_active() {
                let result_layout = self.get_layout().clone();
                let requires_grad = self.is_requires_grad() || rhs.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                let op = Op::Binary(op::BinaryOp::$op_name, self.id(), rhs.id());
                register_operation_in_builder(
                    op.clone(),
                    result_id,
                    vec![self.get_layout().clone(), rhs.get_layout().clone()],
                    vec![result_layout],
                );

                if self.is_requires_grad() || rhs.is_requires_grad() {
                    gradient::record_operation(result_id, op, vec![self.id(), rhs.id()])?;
                }

                Ok(result_tensor)
            } else {
                let storage = self.with_storage(|lhs_storage| {
                    rhs.with_storage(|rhs_storage| {
                        lhs_storage.binary_impl::<op::$op_name>(rhs_storage, &self.get_layout(), &rhs.get_layout())
                    })
                })?;
                let layout = self.get_layout().clone();
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
            if builder::is_builder_active() {
                let result_layout = self.get_layout().clone();
                let requires_grad = self.is_requires_grad() || rhs.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                let op = Op::BinaryLogical(op::BinaryLogicalOp::$op_name, self.id(), rhs.id());
                register_operation_in_builder(
                    op.clone(),
                    result_id,
                    vec![self.get_layout().clone(), rhs.get_layout().clone()],
                    vec![result_layout],
                );

                if self.is_requires_grad() || rhs.is_requires_grad() {
                    gradient::record_operation(result_id, op, vec![self.id(), rhs.id()])?;
                }

                Ok(result_tensor)
            } else {
                let storage = self.with_storage(|lhs_storage| {
                    rhs.with_storage(|rhs_storage| {
                        lhs_storage.binary_logical_impl::<op::$op_name>(
                            rhs_storage,
                            &self.get_layout(),
                            &rhs.get_layout(),
                        )
                    })
                })?;
                let layout = self.get_layout().clone();
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

macro_rules! cmp_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> HoduResult<Self> {
            if builder::is_builder_active() {
                let result_layout = self.get_layout().clone();
                let requires_grad = self.is_requires_grad() || rhs.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                let op = Op::Cmp(op::CmpOp::$op_name, self.id(), rhs.id());
                register_operation_in_builder(
                    op.clone(),
                    result_id,
                    vec![self.get_layout().clone(), rhs.get_layout().clone()],
                    vec![result_layout],
                );

                if self.is_requires_grad() || rhs.is_requires_grad() {
                    gradient::record_operation(result_id, op, vec![self.id(), rhs.id()])?;
                }

                Ok(result_tensor)
            } else {
                let storage = self.with_storage(|lhs_storage| {
                    rhs.with_storage(|rhs_storage| {
                        lhs_storage.cmp_impl::<op::$op_name>(rhs_storage, &self.get_layout(), &rhs.get_layout())
                    })
                })?;
                let layout = self.get_layout().clone();
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
            if builder::is_builder_active() {
                let result_layout = self.get_layout().clone();
                let requires_grad = self.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                let op = Op::CmpScalar(op::CmpScalarOp::$op_name, self.id(), scalar);
                register_operation_in_builder(
                    op.clone(),
                    result_id,
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

macro_rules! unary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> HoduResult<Self> {
            if builder::is_builder_active() {
                let result_layout = self.get_layout().clone();
                let requires_grad = self.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                let op = Op::Unary(op::UnaryOp::$op_name, self.id());
                register_operation_in_builder(
                    op.clone(),
                    result_id,
                    vec![self.get_layout().clone()],
                    vec![result_layout],
                );

                if self.is_requires_grad() {
                    gradient::record_operation(result_id, op, vec![self.id()])?;
                }

                Ok(result_tensor)
            } else {
                let storage = self.with_storage(|storage| storage.unary_impl::<op::$op_name>(&self.get_layout()))?;
                let layout = self.get_layout().clone();
                let requires_grad = self.is_requires_grad();
                let result = from_storage_with_grad(storage, layout, true, requires_grad);

                if !gradient::is_computing_gradients() && self.is_requires_grad() {
                    let op = Op::Unary(op::UnaryOp::$op_name, self.id());
                    gradient::record_operation(result.id(), op, vec![self.id()])?;
                }

                Ok(result)
            }
        }
    };
}

macro_rules! unary_logical_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> HoduResult<Self> {
            if builder::is_builder_active() {
                let result_layout = self.get_layout().clone();
                let requires_grad = self.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                let op = Op::UnaryLogical(op::UnaryLogicalOp::$op_name, self.id());
                register_operation_in_builder(
                    op.clone(),
                    result_id,
                    vec![self.get_layout().clone()],
                    vec![result_layout],
                );

                if self.is_requires_grad() {
                    gradient::record_operation(result_id, op, vec![self.id()])?;
                }

                Ok(result_tensor)
            } else {
                let storage =
                    self.with_storage(|storage| storage.unary_logical_impl::<op::$op_name>(&self.get_layout()))?;
                let layout = self.get_layout().clone();
                let requires_grad = self.is_requires_grad();
                let result = from_storage_with_grad(storage, layout, true, requires_grad);

                if !gradient::is_computing_gradients() && self.is_requires_grad() {
                    let op = Op::UnaryLogical(op::UnaryLogicalOp::$op_name, self.id());
                    gradient::record_operation(result.id(), op, vec![self.id()])?;
                }

                Ok(result)
            }
        }
    };
}

macro_rules! unary_scalar_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name<T: Into<Scalar>>(&self, scalar: T) -> HoduResult<Self> {
            let scalar = scalar.into();
            if builder::is_builder_active() {
                let result_layout = self.get_layout().clone();
                let requires_grad = self.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                let op = Op::UnaryScalar(op::UnaryScalarOp::$op_name, self.id(), scalar);
                register_operation_in_builder(
                    op.clone(),
                    result_id,
                    vec![self.get_layout().clone()],
                    vec![result_layout],
                );

                if self.is_requires_grad() {
                    gradient::record_operation(result_id, op, vec![self.id()])?;
                }

                Ok(result_tensor)
            } else {
                let storage =
                    self.with_storage(|storage| storage.unary_scalar_impl::<op::$op_name>(&self.get_layout(), scalar))?;
                let layout = self.get_layout().clone();
                let requires_grad = self.is_requires_grad();
                let result = from_storage_with_grad(storage, layout, true, requires_grad);

                if !gradient::is_computing_gradients() && self.is_requires_grad() {
                    let op = Op::UnaryScalar(op::UnaryScalarOp::$op_name, self.id(), scalar);
                    gradient::record_operation(result.id(), op, vec![self.id()])?;
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

    // Unary operations
    unary_op!(neg, Neg);
    unary_op!(abs, Abs);
    unary_op!(sign, Sign);
    unary_op!(square, Square);
    unary_op!(relu, Relu);
    unary_op!(sigmoid, Sigmoid);
    unary_op!(tanh, Tanh);
    unary_op!(gelu, Gelu);
    unary_op!(sin, Sin);
    unary_op!(cos, Cos);
    unary_op!(tan, Tan);
    unary_op!(ln, Ln);
    unary_op!(log10, Log10);
    unary_op!(log2, Log2);
    unary_op!(exp, Exp);
    unary_op!(exp10, Exp10);
    unary_op!(exp2, Exp2);
    unary_op!(softplus, Softplus);
    unary_op!(recip, Recip);
    unary_op!(sqrt, Sqrt);

    // Unary logical operations
    unary_logical_op!(logical_not, LogicalNot);

    // Unary scalar operations
    unary_scalar_op!(add_scalar, AddScalar);
    unary_scalar_op!(sub_scalar, SubScalar);
    unary_scalar_op!(mul_scalar, MulScalar);
    unary_scalar_op!(div_scalar, DivScalar);
    unary_scalar_op!(pow_scalar, PowScalar);
    unary_scalar_op!(maximum_scalar, MaximumScalar);
    unary_scalar_op!(minimum_scalar, MinimumScalar);
    unary_scalar_op!(leaky_relu, LeakyRelu);
    unary_scalar_op!(elu, Elu);

    // Matrix operations
    pub fn matmul(&self, other: &Self) -> HoduResult<Self> {
        if builder::is_builder_active() {
            let lhs_layout = self.get_layout();
            let rhs_layout = other.get_layout();
            let lhs_shape = lhs_layout.get_shape();
            let rhs_shape = rhs_layout.get_shape();

            if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "matmul - only 2D tensors supported".to_string(),
                });
            }

            let (m, k1) = (lhs_shape[0], lhs_shape[1]);
            let (k2, n) = (rhs_shape[0], rhs_shape[1]);

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "matmul - inner dimensions must match".to_string(),
                });
            }

            let result_shape = vec![m, n];
            let result_layout = crate::types::layout::Layout::from_shape(&result_shape);
            let requires_grad = self.is_requires_grad() || other.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let op = Op::Matrix(op::MatrixOp::Matmul, self.id(), other.id());
            register_operation_in_builder(
                op.clone(),
                result_id,
                vec![self.get_layout().clone(), other.get_layout().clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() || other.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), other.id()])?;
            }

            Ok(result_tensor)
        } else {
            let lhs_layout = self.get_layout();
            let rhs_layout = other.get_layout();
            let lhs_shape = lhs_layout.get_shape();
            let rhs_shape = rhs_layout.get_shape();

            if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
                return Err(crate::error::HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "matmul - only 2D tensors supported".to_string(),
                });
            }

            let (m, _k1) = (lhs_shape[0], lhs_shape[1]);
            let (_k2, n) = (rhs_shape[0], rhs_shape[1]);
            let result_shape = vec![m, n];
            let result_layout = crate::types::layout::Layout::from_shape(&result_shape);

            let storage = self.with_storage(|lhs_storage| {
                other.with_storage(|rhs_storage| {
                    lhs_storage.matmul(rhs_storage, &self.get_layout(), &other.get_layout())
                })
            })?;

            let requires_grad = self.is_requires_grad() || other.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || other.is_requires_grad()) {
                let op = Op::Matrix(op::MatrixOp::Matmul, self.id(), other.id());
                gradient::record_operation(result.id(), op, vec![self.id(), other.id()])?;
            }

            Ok(result)
        }
    }

    pub fn to_dtype(&self, dtype: DType) -> HoduResult<Self> {
        if builder::is_builder_active() {
            let result_layout = self.get_layout().clone();
            let requires_grad = self.is_requires_grad() && dtype.is_float();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let op = Op::Cast(op::CastOp::ToDType, self.id());
            register_operation_in_builder(
                op.clone(),
                result_id,
                vec![self.get_layout().clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.to_dtype(dtype))?;
            let layout = self.get_layout().clone();
            let requires_grad = self.is_requires_grad() && dtype.is_float();
            let result = from_storage_with_grad(storage, layout, true, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Cast(op::CastOp::ToDType, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }
}
