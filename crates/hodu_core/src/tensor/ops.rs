use super::utils::{broadcast_tensors2, broadcast_tensors3};
use crate::{
    builder,
    compat::*,
    error::{HoduError, HoduResult},
    op::{
        self,
        utils::{
            validate_dtype_for_device, validate_dtype_for_op, validate_indices_dtype, validate_same_device,
            validate_same_dtype,
        },
        Op,
    },
    scalar::Scalar,
    tensor::{
        create_builder_tensor_with_grad, from_shared_storage_with_grad, from_storage_with_grad, gradient,
        register_operation_in_builder, Tensor, TensorId,
    },
    types::{dtype::DType, layout::Layout},
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

macro_rules! unary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> HoduResult<Self> {
            // Validate dtype for device and operation
            validate_dtype_for_device(self.get_dtype(), &self.get_device(), stringify!($op_name))?;
            let op = Op::Unary(op::UnaryOp::$op_name, self.id());
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
            // Validate dtype for device and operation
            validate_dtype_for_device(self.get_dtype(), &self.get_device(), stringify!($op_name))?;
            let op = Op::UnaryLogical(op::UnaryLogicalOp::$op_name, self.id());
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

            // Validate dtype for device and operation
            validate_dtype_for_device(self.get_dtype(), &self.get_device(), stringify!($op_name))?;
            let op = Op::UnaryScalar(op::UnaryScalarOp::$op_name, self.id(), scalar);
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
    unary_op!(sqrt, Sqrt);
    unary_op!(recip, Recip);

    unary_op!(relu, Relu);
    unary_op!(sigmoid, Sigmoid);
    unary_op!(tanh, Tanh);
    unary_op!(gelu, Gelu);
    unary_op!(softplus, Softplus);
    unary_op!(silu, Silu);
    pub fn swish(&self) -> HoduResult<Self> {
        self.silu()
    }
    unary_op!(mish, Mish);

    unary_op!(sin, Sin);
    unary_op!(cos, Cos);
    unary_op!(tan, Tan);

    unary_op!(exp, Exp);
    unary_op!(exp2, Exp2);
    unary_op!(exp10, Exp10);
    unary_op!(ln, Ln);
    unary_op!(log2, Log2);
    unary_op!(log10, Log10);

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
    unary_scalar_op!(prelu, Prelu);

    // Matrix operations
    pub fn matmul(&self, other: &Self) -> HoduResult<Self> {
        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, other], "matmul")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "matmul")?;
        let op = Op::Matrix(op::MatrixOp::Matmul, self.id(), other.id());
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, other], "matmul")?;

        // Supports ND batched matrix multiplication with broadcasting (like XLA matmul)
        let lhs_layout = self.get_layout();
        let rhs_layout = other.get_layout();
        let lhs_shape = lhs_layout.get_shape();
        let rhs_shape = rhs_layout.get_shape();
        let lhs_ndim = lhs_shape.len();
        let rhs_ndim = rhs_shape.len();

        // Both tensors must be at least 1D
        if lhs_ndim < 1 || rhs_ndim < 1 {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: "matmul - both tensors must be at least 1D".to_string(),
            });
        }

        // Handle 1D x 1D case - vector dot product
        if lhs_ndim == 1 && rhs_ndim == 1 {
            if lhs_shape[0] != rhs_shape[0] {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "matmul - 1D vectors must have same length".to_string(),
                });
            }
            return self.mul(other)?.sum_all();
        }

        // Handle 2D x 1D case - matrix-vector product
        if lhs_ndim == 2 && rhs_ndim == 1 {
            let k1 = lhs_shape[1];
            let k2 = rhs_shape[0];

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "matmul - incompatible dimensions for matrix-vector product".to_string(),
                });
            }

            let rhs_reshaped = other.reshape([k2, 1])?;
            let result = self.dot_2d(&rhs_reshaped)?;
            return result.squeeze(Some(-1));
        }

        // Handle 1D x 2D case - vector-matrix product
        if lhs_ndim == 1 && rhs_ndim == 2 {
            let k1 = lhs_shape[0];
            let k2 = rhs_shape[0];

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "matmul - incompatible dimensions for vector-matrix product".to_string(),
                });
            }

            let lhs_reshaped = self.reshape([1, k1])?;
            let result = lhs_reshaped.dot_2d(other)?;
            return result.squeeze(Some(0));
        }

        // Handle 2D x 2D case - standard matrix multiplication
        if lhs_ndim == 2 && rhs_ndim == 2 {
            return self.dot_2d(other);
        }

        // Handle ND x ND case - batched matrix multiplication with broadcasting
        self.matmul_batched(other)
    }

    // Helper: ND batched matrix multiplication
    fn matmul_batched(&self, other: &Self) -> HoduResult<Self> {
        let lhs_layout = self.get_layout();
        let rhs_layout = other.get_layout();
        let lhs_shape = lhs_layout.get_shape();
        let rhs_shape = rhs_layout.get_shape();
        let lhs_ndim = lhs_shape.len();
        let rhs_ndim = rhs_shape.len();

        if lhs_ndim < 2 || rhs_ndim < 2 {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: "matmul_batched - both tensors must be at least 2D".to_string(),
            });
        }

        // Check that last two dimensions are compatible for matmul
        let lhs_inner = lhs_shape[lhs_ndim - 1];
        let rhs_outer = rhs_shape[rhs_ndim - 2];

        if lhs_inner != rhs_outer {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: "matmul_batched - inner dimensions must match".to_string(),
            });
        }

        // Compute broadcast shape for batch dimensions
        let lhs_batch_shape = &lhs_shape[..lhs_ndim - 2];
        let rhs_batch_shape = &rhs_shape[..rhs_ndim - 2];

        let max_batch_ndim = lhs_batch_shape.len().max(rhs_batch_shape.len());
        let mut batch_shape = vec![0; max_batch_ndim];

        for i in 0..max_batch_ndim {
            let lhs_dim = if i < lhs_batch_shape.len() {
                lhs_batch_shape[lhs_batch_shape.len() - 1 - i]
            } else {
                1
            };
            let rhs_dim = if i < rhs_batch_shape.len() {
                rhs_batch_shape[rhs_batch_shape.len() - 1 - i]
            } else {
                1
            };

            if lhs_dim != 1 && rhs_dim != 1 && lhs_dim != rhs_dim {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "matmul_batched - incompatible batch dimensions".to_string(),
                });
            }
            batch_shape[max_batch_ndim - 1 - i] = lhs_dim.max(rhs_dim);
        }

        // Broadcast both tensors to have the same batch dimensions
        let mut lhs_broadcast_shape = batch_shape.clone();
        lhs_broadcast_shape.push(lhs_shape[lhs_ndim - 2]);
        lhs_broadcast_shape.push(lhs_shape[lhs_ndim - 1]);

        let mut rhs_broadcast_shape = batch_shape.clone();
        rhs_broadcast_shape.push(rhs_shape[rhs_ndim - 2]);
        rhs_broadcast_shape.push(rhs_shape[rhs_ndim - 1]);

        let lhs_broadcasted = self.broadcast(&lhs_broadcast_shape)?;
        let rhs_broadcasted = other.broadcast(&rhs_broadcast_shape)?;

        // Result shape: batch_shape + [M, N]
        let mut result_shape = batch_shape;
        result_shape.push(lhs_shape[lhs_ndim - 2]); // M
        result_shape.push(rhs_shape[rhs_ndim - 1]); // N

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&result_shape);
            let requires_grad = self.is_requires_grad() || other.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let op = Op::Matrix(op::MatrixOp::Matmul, lhs_broadcasted.id(), rhs_broadcasted.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![
                    lhs_broadcasted.get_layout().clone(),
                    rhs_broadcasted.get_layout().clone(),
                ],
                vec![result_layout],
            );

            if self.is_requires_grad() || other.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), other.id()])?;
            }

            Ok(result_tensor)
        } else {
            let result_layout = Layout::from_shape(&result_shape);

            let storage = lhs_broadcasted.with_storage(|lhs_storage| {
                rhs_broadcasted.with_storage(|rhs_storage| {
                    lhs_storage.matmul(
                        rhs_storage,
                        &lhs_broadcasted.get_layout(),
                        &rhs_broadcasted.get_layout(),
                    )
                })
            })?;

            let requires_grad = self.is_requires_grad() || other.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || other.is_requires_grad()) {
                let op = Op::Matrix(op::MatrixOp::Matmul, lhs_broadcasted.id(), rhs_broadcasted.id());
                gradient::record_operation(result.id(), op, vec![self.id(), other.id()])?;
            }

            Ok(result)
        }
    }

    pub fn dot(&self, other: &Self) -> HoduResult<Self> {
        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, other], "dot")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "dot")?;
        let op = Op::Matrix(op::MatrixOp::Dot, self.id(), other.id());
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, other], "dot")?;

        // Simple dot operation - supports 1D and 2D combinations (like XLA dot)
        let lhs_layout = self.get_layout();
        let rhs_layout = other.get_layout();
        let lhs_shape = lhs_layout.get_shape();
        let rhs_shape = rhs_layout.get_shape();
        let lhs_ndim = lhs_shape.len();
        let rhs_ndim = rhs_shape.len();

        // Case 1: 1D x 1D - dot product (inner product)
        if lhs_ndim == 1 && rhs_ndim == 1 {
            if lhs_shape[0] != rhs_shape[0] {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "dot - 1D vectors must have same length".to_string(),
                });
            }
            return self.mul(other)?.sum_all();
        }

        // Case 2: 2D x 1D - matrix-vector product
        if lhs_ndim == 2 && rhs_ndim == 1 {
            let k1 = lhs_shape[1];
            let k2 = rhs_shape[0];

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "dot - incompatible dimensions for matrix-vector product".to_string(),
                });
            }

            let rhs_reshaped = other.reshape([k2, 1])?;
            let result = self.dot_2d(&rhs_reshaped)?;
            return result.squeeze(Some(-1));
        }

        // Case 3: 1D x 2D - vector-matrix product
        if lhs_ndim == 1 && rhs_ndim == 2 {
            let k1 = lhs_shape[0];
            let k2 = rhs_shape[0];

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "dot - incompatible dimensions for vector-matrix product".to_string(),
                });
            }

            let lhs_reshaped = self.reshape([1, k1])?;
            let result = lhs_reshaped.dot_2d(other)?;
            return result.squeeze(Some(0));
        }

        // Case 4: 2D x 2D - matrix multiplication
        if lhs_ndim == 2 && rhs_ndim == 2 {
            return self.dot_2d(other);
        }

        // Dot operation only supports 1D and 2D tensors
        // For higher dimensional (batched) operations, use matmul instead
        Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "dot - only supports 1D and 2D tensors. Use matmul() for batched operations".to_string(),
        })
    }

    // Helper: 2D matrix multiplication (assumes both are 2D)
    fn dot_2d(&self, other: &Self) -> HoduResult<Self> {
        if builder::is_builder_active() {
            let lhs_layout = self.get_layout();
            let rhs_layout = other.get_layout();
            let lhs_shape = lhs_layout.get_shape();
            let rhs_shape = rhs_layout.get_shape();

            let (m, k1) = (lhs_shape[0], lhs_shape[1]);
            let (k2, n) = (rhs_shape[0], rhs_shape[1]);

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "dot_2d - inner dimensions must match".to_string(),
                });
            }

            let result_shape = vec![m, n];
            let result_layout = Layout::from_shape(&result_shape);
            let requires_grad = self.is_requires_grad() || other.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let op = Op::Matrix(op::MatrixOp::Dot, self.id(), other.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
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

            let (m, _k1) = (lhs_shape[0], lhs_shape[1]);
            let (_k2, n) = (rhs_shape[0], rhs_shape[1]);
            let result_shape = vec![m, n];
            let result_layout = Layout::from_shape(&result_shape);

            let storage = self.with_storage(|lhs_storage| {
                other.with_storage(|rhs_storage| lhs_storage.dot(rhs_storage, &self.get_layout(), &other.get_layout()))
            })?;

            let requires_grad = self.is_requires_grad() || other.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || other.is_requires_grad()) {
                let op = Op::Matrix(op::MatrixOp::Dot, self.id(), other.id());
                gradient::record_operation(result.id(), op, vec![self.id(), other.id()])?;
            }

            Ok(result)
        }
    }

    // Reduction Operations
    pub fn sum(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Sum, dims, keep_dim)
    }

    pub fn sum_all(&self) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Sum, &[], false)
    }

    pub fn sum_to_shape(&self, target_shape: &[usize]) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let current_shape = current_layout.get_shape();

        if current_shape == target_shape {
            return Ok(*self);
        }

        let mut result = *self;

        if current_shape.len() > target_shape.len() {
            let dims_to_sum: Vec<usize> = (0..(current_shape.len() - target_shape.len())).collect();
            for &dim in dims_to_sum.iter().rev() {
                result = result.sum(&[dim], false)?;
            }
        }

        let result_layout = result.get_layout();
        let result_shape = result_layout.get_shape();
        for (i, (&target_dim, &current_dim)) in target_shape.iter().zip(result_shape.iter()).enumerate() {
            if target_dim == 1 && current_dim > 1 {
                result = result.sum(&[i], true)?;
            }
        }

        Ok(result)
    }

    pub fn mean(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Mean, dims, keep_dim)
    }

    pub fn mean_all(&self) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Mean, &[], false)
    }

    pub fn max(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Max, dims, keep_dim)
    }

    pub fn min(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Min, dims, keep_dim)
    }

    pub fn prod(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Prod, dims, keep_dim)
    }

    pub fn std(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Std, dims, keep_dim)
    }

    pub fn std_all(&self) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Std, &[], false)
    }

    pub fn var(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Var, dims, keep_dim)
    }

    pub fn var_all(&self) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Var, &[], false)
    }

    pub fn norm(&self, p: impl Into<Scalar>, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        let p_scalar = p.into();
        match p_scalar.to_u32() {
            1 => self.l1_norm(dims, keep_dim),
            2 => self.l2_norm(dims, keep_dim),
            _ => {
                let p_dtype = p_scalar.to_dtype(self.get_dtype());
                self.abs()?
                    .pow_scalar(p_dtype)?
                    .sum(dims, keep_dim)?
                    .pow_scalar(Scalar::one(self.get_dtype()) / p_dtype)
            },
        }
    }

    pub fn l2_norm(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Norm, dims, keep_dim)
    }

    pub fn l1_norm(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.abs()?.sum(dims, keep_dim)
    }

    pub fn argmax(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::ArgMax, dims, keep_dim)
    }

    pub fn argmin(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::ArgMin, dims, keep_dim)
    }

    pub fn any(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Any, dims, keep_dim)
    }

    pub fn all(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::All, dims, keep_dim)
    }

    fn reduce_operation(&self, reduce_op: op::ReduceOp, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        // Validate dtype for device and operation
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), &reduce_op.to_string())?;
        let dims_scalars: Vec<Scalar> = dims.iter().map(|&d| Scalar::U64(d as u64)).collect();
        let op = Op::Reduce(reduce_op, self.id(), keep_dim, dims_scalars.clone());
        validate_dtype_for_op(self.get_dtype(), &op)?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let shape = layout.get_shape();
            let ndim = shape.len();

            // Calculate output shape
            let reduce_dims: Vec<usize> = if dims.is_empty() {
                (0..ndim).collect()
            } else {
                dims.to_vec()
            };

            let mut output_shape = shape.to_vec();
            for &dim in &reduce_dims {
                if keep_dim {
                    output_shape[dim] = 1;
                } else {
                    output_shape[dim] = 0;
                }
            }
            if !keep_dim {
                output_shape.retain(|&size| size != 0);
            }

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(op.clone(), vec![result_id], vec![layout.clone()], vec![result_layout]);

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.reduce(reduce_op, &self.get_layout(), dims, keep_dim))?;
            let layout = self.get_layout();
            let shape = layout.get_shape();
            let ndim = shape.len();

            // Calculate output shape
            let reduce_dims: Vec<usize> = if dims.is_empty() {
                (0..ndim).collect()
            } else {
                dims.to_vec()
            };

            let mut output_shape = shape.to_vec();
            for &dim in &reduce_dims {
                if keep_dim {
                    output_shape[dim] = 1;
                } else {
                    output_shape[dim] = 0;
                }
            }
            if !keep_dim {
                output_shape.retain(|&size| size != 0);
            }

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let dims_scalars: Vec<Scalar> = dims.iter().map(|&d| Scalar::U64(d as u64)).collect();
                let op = Op::Reduce(reduce_op, self.id(), keep_dim, dims_scalars);
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    // Concat Operations
    pub fn concat<D: Into<Scalar>>(tensors: &[&Self], dim: D) -> HoduResult<Self> {
        if tensors.is_empty() {
            return Err(HoduError::InternalError(
                "concat requires at least one tensor".to_string(),
            ));
        }
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        let first = tensors[0];

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(tensors, "concat")?;
        validate_dtype_for_device(first.get_dtype(), &first.get_device(), "concat")?;
        let tensor_ids: Vec<TensorId> = tensors.iter().map(|t| t.id()).collect();
        let op = Op::Concat(op::ConcatOp::Concat, tensor_ids.clone(), vec![dim_scalar]);
        validate_dtype_for_op(first.get_dtype(), &op)?;
        validate_same_dtype(tensors, "concat")?;

        let mut output_shape = first.get_layout().get_shape().to_vec();
        for tensor in &tensors[1..] {
            let layout = tensor.get_layout();
            let shape = layout.get_shape();
            if shape.len() != output_shape.len() {
                return Err(HoduError::IncompatibleShapes {
                    lhs: output_shape,
                    rhs: shape.to_vec(),
                    op: "concat - all tensors must have same number of dimensions".to_string(),
                });
            }
            output_shape[dim_usize] += shape[dim_usize];
        }

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = tensors.iter().any(|t| t.is_requires_grad());
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let input_layouts: Vec<Layout> = tensors.iter().map(|t| t.get_layout().clone()).collect();
            register_operation_in_builder(op.clone(), vec![result_id], input_layouts, vec![result_layout]);

            if requires_grad {
                gradient::record_operation(result_id, op, tensor_ids)?;
            }

            Ok(result_tensor)
        } else {
            let layouts: Vec<_> = tensors.iter().map(|t| t.get_layout()).collect();
            let layout_refs: Vec<_> = layouts.iter().collect();

            // Clone storages to avoid lifetime issues
            let mut all_storages: Vec<_> = Vec::new();
            for tensor in tensors.iter() {
                let storage = tensor.with_storage(|s| Ok(s.clone()))?;
                all_storages.push(storage);
            }

            let first_storage = &all_storages[0];
            let other_refs: Vec<_> = all_storages[1..].iter().collect();
            let storage = first_storage.concat(&other_refs, &layout_refs, dim_usize)?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = tensors.iter().any(|t| t.is_requires_grad());
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let tensor_ids: Vec<TensorId> = tensors.iter().map(|t| t.id()).collect();
                let op = Op::Concat(op::ConcatOp::Concat, tensor_ids.clone(), vec![dim_scalar]);
                gradient::record_operation(result.id(), op, tensor_ids)?;
            }

            Ok(result)
        }
    }

    pub fn cat<D: Into<Scalar>>(tensors: &[&Self], dim: D) -> HoduResult<Self> {
        Self::concat(tensors, dim)
    }

    pub fn stack<D: Into<Scalar>>(tensors: &[&Self], dim: D) -> HoduResult<Self> {
        if tensors.is_empty() {
            return Err(HoduError::InternalError(
                "stack requires at least one tensor".to_string(),
            ));
        }
        let dim_scalar = dim.into();
        let dim_isize = dim_scalar.to_i64() as isize;

        let unsqueezed: Vec<Self> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim_isize))
            .collect::<HoduResult<_>>()?;
        let unsqueezed_refs: Vec<&Self> = unsqueezed.iter().collect();
        Self::concat(&unsqueezed_refs, dim_scalar)
    }

    // Split Operations
    pub fn split<D: Into<Scalar>>(&self, sizes: &[usize], dim: D) -> HoduResult<Vec<Self>> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate dtype for device and operation
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "split")?;
        let mut params = vec![dim_scalar];
        params.extend(sizes.iter().map(|&s| Scalar::U64(s as u64)));
        let op = Op::Split(op::SplitOp::Split, self.id(), params.clone(), 0);
        validate_dtype_for_op(self.get_dtype(), &op)?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let shape = layout.get_shape();

            let requires_grad = self.is_requires_grad();
            let mut result_tensors = Vec::new();
            let mut result_layouts = Vec::new();

            for &size in sizes {
                let mut result_shape = shape.to_vec();
                result_shape[dim_usize] = size;
                let result_layout = Layout::from_shape(&result_shape);
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);
                result_tensors.push((result_id, result_tensor));
                result_layouts.push(result_layout);
            }

            // Register separate operation for each split output with its output_index
            for (output_index, ((result_id, _), result_layout)) in
                result_tensors.iter().zip(&result_layouts).enumerate()
            {
                let op = Op::Split(op::SplitOp::Split, self.id(), params.clone(), output_index);
                register_operation_in_builder(
                    op.clone(),
                    vec![*result_id],
                    vec![layout.clone()],
                    vec![result_layout.clone()],
                );

                if requires_grad {
                    gradient::record_operation(*result_id, op, vec![self.id()])?;
                }
            }

            Ok(result_tensors.into_iter().map(|(_, t)| t).collect())
        } else {
            let storages = self.with_storage(|storage| storage.split(&self.get_layout(), dim_usize, sizes))?;
            let layout = self.get_layout();
            let shape = layout.get_shape();
            let requires_grad = self.is_requires_grad();

            let results: Vec<Self> = storages
                .into_iter()
                .zip(sizes.iter())
                .map(|(storage, &size)| {
                    let mut result_shape = shape.to_vec();
                    result_shape[dim_usize] = size;
                    let result_layout = Layout::from_shape(&result_shape);
                    from_storage_with_grad(storage, result_layout, true, requires_grad)
                })
                .collect();

            if !gradient::is_computing_gradients() && requires_grad {
                let mut params = vec![dim_scalar];
                params.extend(sizes.iter().map(|&s| Scalar::U64(s as u64)));
                // Record operation for each split result with its output_index
                for (output_index, result) in results.iter().enumerate() {
                    let op = Op::Split(op::SplitOp::Split, self.id(), params.clone(), output_index);
                    gradient::record_operation(result.id(), op, vec![self.id()])?;
                }
            }

            Ok(results)
        }
    }

    pub fn chunk<D: Into<Scalar>>(&self, chunks: usize, dim: D) -> HoduResult<Vec<Self>> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;
        let layout = self.get_layout();
        let shape = layout.get_shape();
        let dim_size = shape[dim_usize];

        let chunk_size = dim_size.div_ceil(chunks);
        let sizes: Vec<usize> = (0..chunks)
            .map(|i| {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(dim_size);
                end - start
            })
            .filter(|&s| s > 0)
            .collect();

        self.split(&sizes, dim_scalar)
    }

    // Normalization Operations
    pub fn softmax<D: Into<Scalar>>(&self, dim: D) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Numerical stability: subtract max
        // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        let max_val = self.max(&[dim_usize], true)?;
        let shifted = self.sub(&max_val)?;
        let exp_vals = shifted.exp()?;
        let sum_exp = exp_vals.sum(&[dim_usize], true)?;
        exp_vals.div(&sum_exp)
    }

    pub fn log_softmax<D: Into<Scalar>>(&self, dim: D) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Numerical stability: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        let max_val = self.max(&[dim_usize], true)?;
        let shifted = self.sub(&max_val)?;
        let exp_vals = shifted.exp()?;
        let sum_exp = exp_vals.sum(&[dim_usize], true)?;
        let log_sum_exp = sum_exp.ln()?;
        shifted.sub(&log_sum_exp)
    }

    // Indexing Operations
    pub fn index_select<D: Into<Scalar>>(&self, dim: D, indices: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices], "index_select")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "index_select")?;
        let op = Op::Indexing(
            op::IndexingOp::IndexSelect,
            vec![self.id(), indices.id()],
            vec![dim_scalar],
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_indices_dtype(indices, "index_select")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let shape = layout.get_shape();

            // Output shape: replace indexed dimension with indices size
            let mut output_shape = shape.to_vec();
            output_shape[dim_usize] = indices_layout.get_size();

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    storage.index_select(&self.get_layout(), indices_storage, &indices.get_layout(), dim_usize)
                })
            })?;

            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let shape = layout.get_shape();
            let mut output_shape = shape.to_vec();
            output_shape[dim_usize] = indices_layout.get_size();

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Indexing(
                    op::IndexingOp::IndexSelect,
                    vec![self.id(), indices.id()],
                    vec![dim_scalar],
                );
                gradient::record_operation(result.id(), op, vec![self.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    pub fn index_put<D: Into<Scalar>>(&self, dim: D, indices: &Self, values: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, values], "index_put")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "index_put")?;
        let op = Op::Indexing(
            op::IndexingOp::IndexPut,
            vec![self.id(), indices.id(), values.id()],
            vec![dim_scalar],
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, values], "index_put")?;
        validate_indices_dtype(indices, "index_put")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let values_layout = values.get_layout();

            // Output has same shape as input
            let result_layout = layout.clone();
            let requires_grad = self.is_requires_grad() || values.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone(), values_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), values.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    values.with_storage(|values_storage| {
                        storage.index_put(
                            &self.get_layout(),
                            indices_storage,
                            &indices.get_layout(),
                            values_storage,
                            &values.get_layout(),
                            dim_usize,
                        )
                    })
                })
            })?;

            let result_layout = self.get_layout();
            let requires_grad = self.is_requires_grad() || values.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Indexing(
                    op::IndexingOp::IndexPut,
                    vec![self.id(), indices.id(), values.id()],
                    vec![dim_scalar],
                );
                gradient::record_operation(result.id(), op, vec![self.id(), values.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    pub fn gather<D: Into<Scalar>>(&self, dim: D, indices: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices], "gather")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "gather")?;
        let op = Op::Indexing(op::IndexingOp::Gather, vec![self.id(), indices.id()], vec![dim_scalar]);
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_indices_dtype(indices, "gather")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();

            // Output has same shape as indices
            let result_layout = indices_layout.clone();
            let requires_grad = self.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    storage.gather(&self.get_layout(), indices_storage, &indices.get_layout(), dim_usize)
                })
            })?;

            let result_layout = indices.get_layout().clone();
            let requires_grad = self.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Indexing(op::IndexingOp::Gather, vec![self.id(), indices.id()], vec![dim_scalar]);
                gradient::record_operation(result.id(), op, vec![self.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    pub fn scatter<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], "scatter")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "scatter")?;
        let op = Op::Indexing(
            op::IndexingOp::Scatter,
            vec![self.id(), indices.id(), src.id()],
            vec![dim_scalar],
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, src], "scatter")?;
        validate_indices_dtype(indices, "scatter")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let src_layout = src.get_layout();

            // Output has same shape as self
            let result_layout = layout.clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone(), src_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() || src.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.scatter(
                            &self.get_layout(),
                            indices_storage,
                            &indices.get_layout(),
                            src_storage,
                            &src.get_layout(),
                            dim_usize,
                        )
                    })
                })
            })?;

            let result_layout = self.get_layout().clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || src.is_requires_grad()) {
                let op = Op::Indexing(
                    op::IndexingOp::Scatter,
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                );
                gradient::record_operation(result.id(), op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    pub fn scatter_add<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], "scatter_add")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "scatter_add")?;
        let op = Op::Indexing(
            op::IndexingOp::ScatterAdd,
            vec![self.id(), indices.id(), src.id()],
            vec![dim_scalar],
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, src], "scatter_add")?;
        validate_indices_dtype(indices, "scatter_add")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let src_layout = src.get_layout();

            let result_layout = layout.clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone(), src_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() || src.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.scatter_add(
                            &self.get_layout(),
                            indices_storage,
                            &indices.get_layout(),
                            src_storage,
                            &src.get_layout(),
                            dim_usize,
                        )
                    })
                })
            })?;

            let result_layout = self.get_layout().clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || src.is_requires_grad()) {
                let op = Op::Indexing(
                    op::IndexingOp::ScatterAdd,
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                );
                gradient::record_operation(result.id(), op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    pub fn scatter_max<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], "scatter_max")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "scatter_max")?;
        let op = Op::Indexing(
            op::IndexingOp::ScatterMax,
            vec![self.id(), indices.id(), src.id()],
            vec![dim_scalar],
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, src], "scatter_max")?;
        validate_indices_dtype(indices, "scatter_max")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let src_layout = src.get_layout();

            let result_layout = layout.clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone(), src_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() || src.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.scatter_max(
                            &self.get_layout(),
                            indices_storage,
                            &indices.get_layout(),
                            src_storage,
                            &src.get_layout(),
                            dim_usize,
                        )
                    })
                })
            })?;

            let result_layout = self.get_layout().clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || src.is_requires_grad()) {
                let op = Op::Indexing(
                    op::IndexingOp::ScatterMax,
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                );
                gradient::record_operation(result.id(), op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    pub fn scatter_min<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], "scatter_min")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "scatter_min")?;
        let op = Op::Indexing(
            op::IndexingOp::ScatterMin,
            vec![self.id(), indices.id(), src.id()],
            vec![dim_scalar],
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, src], "scatter_min")?;
        validate_indices_dtype(indices, "scatter_min")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let src_layout = src.get_layout();

            let result_layout = layout.clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone(), src_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() || src.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.scatter_min(
                            &self.get_layout(),
                            indices_storage,
                            &indices.get_layout(),
                            src_storage,
                            &src.get_layout(),
                            dim_usize,
                        )
                    })
                })
            })?;

            let result_layout = self.get_layout().clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || src.is_requires_grad()) {
                let op = Op::Indexing(
                    op::IndexingOp::ScatterMin,
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                );
                gradient::record_operation(result.id(), op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    // Selection Operations
    pub fn where3(&self, condition: &Tensor, other: &Tensor) -> HoduResult<Self> {
        let (condition, x, y) = broadcast_tensors3(condition, self, other)?;

        let mask = condition.to_dtype(x.get_dtype())?;
        let one = Self::ones_like(&mask)?;
        let inv_mask = one.sub(&mask)?;

        let x_part = mask.mul(&x)?;
        let y_part = inv_mask.mul(&y)?;
        x_part.add(&y_part)
    }

    pub fn masked_fill<T: Into<Scalar>>(&self, mask: &Tensor, value: T) -> HoduResult<Self> {
        let value_scalar = value.into();
        let fill_tensor = Self::full_like(self, value_scalar)?;
        self.where3(mask, &fill_tensor)
    }

    pub fn clamp<T: Into<Scalar>>(&self, min: T, max: T) -> HoduResult<Self> {
        let min_scalar = min.into();
        let max_scalar = max.into();

        let min_tensor = Self::full_like(self, min_scalar)?;
        let clamped_min = self.where3(&self.lt_scalar(min_scalar)?, &min_tensor)?;

        let max_tensor = Self::full_like(&clamped_min, max_scalar)?;
        clamped_min.where3(&clamped_min.gt_scalar(max_scalar)?, &max_tensor)
    }

    pub fn clamp_min<T: Into<Scalar>>(&self, min: T) -> HoduResult<Self> {
        let min_scalar = min.into();
        let min_tensor = Self::full_like(self, min_scalar)?;
        self.where3(&self.lt_scalar(min_scalar)?, &min_tensor)
    }

    pub fn clamp_max<T: Into<Scalar>>(&self, max: T) -> HoduResult<Self> {
        let max_scalar = max.into();
        let max_tensor = Self::full_like(self, max_scalar)?;
        self.where3(&self.gt_scalar(max_scalar)?, &max_tensor)
    }

    // Convolution Operations
    pub fn conv1d(&self, weight: &Self, stride: usize, padding: usize, dilation: usize) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let weight_layout = weight.get_layout();
        let input_shape = input_layout.get_shape();
        let weight_shape = weight_layout.get_shape();

        // Input: [batch, in_channels, length]
        // Weight: [out_channels, in_channels, kernel_size]
        if input_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv1d - input must be 3D [batch, in_channels, length]".to_string(),
            });
        }
        if weight_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv1d - weight must be 3D [out_channels, in_channels, kernel_size]".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let length_input = input_shape[2];
        let channels_output = weight_shape[0];
        let kernel_size = weight_shape[2];

        if input_shape[1] != weight_shape[1] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: weight_shape.to_vec(),
                op: "conv1d - input and weight channel mismatch".to_string(),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], "conv1d")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "conv1d")?;
        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(length_input as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_size as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];
        let op = Op::Conv(op::ConvOp::Conv1d, self.id(), weight.id(), params_scalars.clone());
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, weight], "conv1d")?;

        let params = op::conv::ParamsConv1D {
            batch_size,
            length_input,
            channels_output,
            channels_input,
            kernel_size,
            padding,
            stride,
            dilation,
        };

        let output_length = (length_input + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        let output_shape = vec![batch_size, channels_output, output_length];

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone(), weight_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), weight.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.conv1d(weight_storage, &input_layout, &weight_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let params_scalars = vec![
                    Scalar::U32(batch_size as u32),
                    Scalar::U32(length_input as u32),
                    Scalar::U32(channels_output as u32),
                    Scalar::U32(channels_input as u32),
                    Scalar::U32(kernel_size as u32),
                    Scalar::U32(padding as u32),
                    Scalar::U32(stride as u32),
                    Scalar::U32(dilation as u32),
                ];
                let op = Op::Conv(op::ConvOp::Conv1d, self.id(), weight.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id(), weight.id()])?;
            }

            Ok(result)
        }
    }

    pub fn conv2d(&self, weight: &Self, stride: usize, padding: usize, dilation: usize) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let weight_layout = weight.get_layout();
        let input_shape = input_layout.get_shape();
        let weight_shape = weight_layout.get_shape();

        // Input: [batch, in_channels, height, width]
        // Weight: [out_channels, in_channels, kernel_h, kernel_w]
        if input_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv2d - input must be 4D [batch, in_channels, height, width]".to_string(),
            });
        }
        if weight_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv2d - weight must be 4D [out_channels, in_channels, kernel_h, kernel_w]".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];
        let channels_output = weight_shape[0];
        let kernel_height = weight_shape[2];
        let kernel_width = weight_shape[3];

        if input_shape[1] != weight_shape[1] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: weight_shape.to_vec(),
                op: "conv2d - input and weight channel mismatch".to_string(),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], "conv2d")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "conv2d")?;
        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(input_height as u32),
            Scalar::U32(input_width as u32),
            Scalar::U32(kernel_height as u32),
            Scalar::U32(kernel_width as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];
        let op = Op::Conv(op::ConvOp::Conv2d, self.id(), weight.id(), params_scalars.clone());
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, weight], "conv2d")?;

        let params = op::conv::ParamsConv2D {
            batch_size,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            channels_output,
            channels_input,
            padding,
            stride,
            dilation,
        };

        let output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
        let output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;
        let output_shape = vec![batch_size, channels_output, output_height, output_width];

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone(), weight_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), weight.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.conv2d(weight_storage, &input_layout, &weight_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let params_scalars = vec![
                    Scalar::U32(batch_size as u32),
                    Scalar::U32(input_height as u32),
                    Scalar::U32(input_width as u32),
                    Scalar::U32(kernel_height as u32),
                    Scalar::U32(kernel_width as u32),
                    Scalar::U32(channels_output as u32),
                    Scalar::U32(channels_input as u32),
                    Scalar::U32(padding as u32),
                    Scalar::U32(stride as u32),
                    Scalar::U32(dilation as u32),
                ];
                let op = Op::Conv(op::ConvOp::Conv2d, self.id(), weight.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id(), weight.id()])?;
            }

            Ok(result)
        }
    }

    pub fn conv3d(&self, weight: &Self, stride: usize, padding: usize, dilation: usize) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let weight_layout = weight.get_layout();
        let input_shape = input_layout.get_shape();
        let weight_shape = weight_layout.get_shape();

        // Input: [batch, in_channels, depth, height, width]
        // Weight: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
        if input_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv3d - input must be 5D [batch, in_channels, depth, height, width]".to_string(),
            });
        }
        if weight_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv3d - weight must be 5D [out_channels, in_channels, kernel_d, kernel_h, kernel_w]".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let input_depth = input_shape[2];
        let input_height = input_shape[3];
        let input_width = input_shape[4];
        let channels_output = weight_shape[0];
        let kernel_depth = weight_shape[2];
        let kernel_height = weight_shape[3];
        let kernel_width = weight_shape[4];

        if input_shape[1] != weight_shape[1] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: weight_shape.to_vec(),
                op: "conv3d - input and weight channel mismatch".to_string(),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], "conv3d")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "conv3d")?;
        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(input_depth as u32),
            Scalar::U32(input_height as u32),
            Scalar::U32(input_width as u32),
            Scalar::U32(kernel_depth as u32),
            Scalar::U32(kernel_height as u32),
            Scalar::U32(kernel_width as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];
        let op = Op::Conv(op::ConvOp::Conv3d, self.id(), weight.id(), params_scalars.clone());
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, weight], "conv3d")?;

        let params = op::conv::ParamsConv3D {
            batch_size,
            input_depth,
            input_height,
            input_width,
            kernel_depth,
            kernel_height,
            kernel_width,
            channels_output,
            channels_input,
            padding,
            stride,
            dilation,
        };

        let output_depth = (input_depth + 2 * padding - dilation * (kernel_depth - 1) - 1) / stride + 1;
        let output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
        let output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;
        let output_shape = vec![batch_size, channels_output, output_depth, output_height, output_width];

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone(), weight_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), weight.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.conv3d(weight_storage, &input_layout, &weight_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let params_scalars = vec![
                    Scalar::U32(batch_size as u32),
                    Scalar::U32(input_depth as u32),
                    Scalar::U32(input_height as u32),
                    Scalar::U32(input_width as u32),
                    Scalar::U32(kernel_depth as u32),
                    Scalar::U32(kernel_height as u32),
                    Scalar::U32(kernel_width as u32),
                    Scalar::U32(channels_output as u32),
                    Scalar::U32(channels_input as u32),
                    Scalar::U32(padding as u32),
                    Scalar::U32(stride as u32),
                    Scalar::U32(dilation as u32),
                ];
                let op = Op::Conv(op::ConvOp::Conv3d, self.id(), weight.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id(), weight.id()])?;
            }

            Ok(result)
        }
    }

    pub fn conv_transpose1d(
        &self,
        weight: &Self,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let weight_layout = weight.get_layout();
        let input_shape = input_layout.get_shape();
        let weight_shape = weight_layout.get_shape();

        // Input: [batch, in_channels, length]
        // Weight: [in_channels, out_channels, kernel_size] (note: different from conv!)
        if input_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose1d - input must be 3D [batch, in_channels, length]".to_string(),
            });
        }
        if weight_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose1d - weight must be 3D [in_channels, out_channels, kernel_size]".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let length_input = input_shape[2];
        let channels_output = weight_shape[1];
        let kernel_size = weight_shape[2];

        if input_shape[1] != weight_shape[0] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: weight_shape.to_vec(),
                op: "conv_transpose1d - input and weight channel mismatch".to_string(),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], "conv_transpose1d")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "conv_transpose1d")?;
        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(length_input as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_size as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(output_padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];
        let op = Op::Conv(
            op::ConvOp::ConvTranspose1d,
            self.id(),
            weight.id(),
            params_scalars.clone(),
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, weight], "conv_transpose1d")?;

        let params = op::conv::ParamsConvTranspose1D {
            batch_size,
            length_input,
            channels_output,
            channels_input,
            kernel_size,
            padding,
            output_padding,
            stride,
            dilation,
        };

        let output_length =
            (length_input - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
        let output_shape = vec![batch_size, channels_output, output_length];

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone(), weight_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), weight.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.conv_transpose1d(weight_storage, &input_layout, &weight_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let params_scalars = vec![
                    Scalar::U32(batch_size as u32),
                    Scalar::U32(length_input as u32),
                    Scalar::U32(channels_output as u32),
                    Scalar::U32(channels_input as u32),
                    Scalar::U32(kernel_size as u32),
                    Scalar::U32(padding as u32),
                    Scalar::U32(output_padding as u32),
                    Scalar::U32(stride as u32),
                    Scalar::U32(dilation as u32),
                ];
                let op = Op::Conv(op::ConvOp::ConvTranspose1d, self.id(), weight.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id(), weight.id()])?;
            }

            Ok(result)
        }
    }

    pub fn conv_transpose2d(
        &self,
        weight: &Self,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let weight_layout = weight.get_layout();
        let input_shape = input_layout.get_shape();
        let weight_shape = weight_layout.get_shape();

        // Input: [batch, in_channels, height, width]
        // Weight: [in_channels, out_channels, kernel_h, kernel_w]
        if input_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose2d - input must be 4D [batch, in_channels, height, width]".to_string(),
            });
        }
        if weight_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose2d - weight must be 4D [in_channels, out_channels, kernel_h, kernel_w]".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];
        let channels_output = weight_shape[1];
        let kernel_height = weight_shape[2];
        let kernel_width = weight_shape[3];

        if input_shape[1] != weight_shape[0] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: weight_shape.to_vec(),
                op: "conv_transpose2d - input and weight channel mismatch".to_string(),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], "conv_transpose2d")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "conv_transpose2d")?;
        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(input_height as u32),
            Scalar::U32(input_width as u32),
            Scalar::U32(kernel_height as u32),
            Scalar::U32(kernel_width as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(output_padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];
        let op = Op::Conv(
            op::ConvOp::ConvTranspose2d,
            self.id(),
            weight.id(),
            params_scalars.clone(),
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, weight], "conv_transpose2d")?;

        let params = op::conv::ParamsConvTranspose2D {
            batch_size,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            channels_output,
            channels_input,
            padding,
            output_padding,
            stride,
            dilation,
        };

        let output_height =
            (input_height - 1) * stride - 2 * padding + dilation * (kernel_height - 1) + output_padding + 1;
        let output_width =
            (input_width - 1) * stride - 2 * padding + dilation * (kernel_width - 1) + output_padding + 1;
        let output_shape = vec![batch_size, channels_output, output_height, output_width];

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone(), weight_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), weight.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.conv_transpose2d(weight_storage, &input_layout, &weight_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let params_scalars = vec![
                    Scalar::U32(batch_size as u32),
                    Scalar::U32(input_height as u32),
                    Scalar::U32(input_width as u32),
                    Scalar::U32(kernel_height as u32),
                    Scalar::U32(kernel_width as u32),
                    Scalar::U32(channels_output as u32),
                    Scalar::U32(channels_input as u32),
                    Scalar::U32(padding as u32),
                    Scalar::U32(output_padding as u32),
                    Scalar::U32(stride as u32),
                    Scalar::U32(dilation as u32),
                ];
                let op = Op::Conv(op::ConvOp::ConvTranspose2d, self.id(), weight.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id(), weight.id()])?;
            }

            Ok(result)
        }
    }

    pub fn conv_transpose3d(
        &self,
        weight: &Self,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let weight_layout = weight.get_layout();
        let input_shape = input_layout.get_shape();
        let weight_shape = weight_layout.get_shape();

        // Input: [batch, in_channels, depth, height, width]
        // Weight: [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
        if input_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose3d - input must be 5D [batch, in_channels, depth, height, width]".to_string(),
            });
        }
        if weight_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose3d - weight must be 5D [in_channels, out_channels, kernel_d, kernel_h, kernel_w]"
                    .to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let input_depth = input_shape[2];
        let input_height = input_shape[3];
        let input_width = input_shape[4];
        let channels_output = weight_shape[1];
        let kernel_depth = weight_shape[2];
        let kernel_height = weight_shape[3];
        let kernel_width = weight_shape[4];

        if input_shape[1] != weight_shape[0] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: weight_shape.to_vec(),
                op: "conv_transpose3d - input and weight channel mismatch".to_string(),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], "conv_transpose3d")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "conv_transpose3d")?;
        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(input_depth as u32),
            Scalar::U32(input_height as u32),
            Scalar::U32(input_width as u32),
            Scalar::U32(kernel_depth as u32),
            Scalar::U32(kernel_height as u32),
            Scalar::U32(kernel_width as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(output_padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];
        let op = Op::Conv(
            op::ConvOp::ConvTranspose3d,
            self.id(),
            weight.id(),
            params_scalars.clone(),
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, weight], "conv_transpose3d")?;

        let params = op::conv::ParamsConvTranspose3D {
            batch_size,
            input_depth,
            input_height,
            input_width,
            kernel_depth,
            kernel_height,
            kernel_width,
            channels_output,
            channels_input,
            padding,
            output_padding,
            stride,
            dilation,
        };

        let output_depth =
            (input_depth - 1) * stride - 2 * padding + dilation * (kernel_depth - 1) + output_padding + 1;
        let output_height =
            (input_height - 1) * stride - 2 * padding + dilation * (kernel_height - 1) + output_padding + 1;
        let output_width =
            (input_width - 1) * stride - 2 * padding + dilation * (kernel_width - 1) + output_padding + 1;
        let output_shape = vec![batch_size, channels_output, output_depth, output_height, output_width];

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone(), weight_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), weight.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.conv_transpose3d(weight_storage, &input_layout, &weight_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let params_scalars = vec![
                    Scalar::U32(batch_size as u32),
                    Scalar::U32(input_depth as u32),
                    Scalar::U32(input_height as u32),
                    Scalar::U32(input_width as u32),
                    Scalar::U32(kernel_depth as u32),
                    Scalar::U32(kernel_height as u32),
                    Scalar::U32(kernel_width as u32),
                    Scalar::U32(channels_output as u32),
                    Scalar::U32(channels_input as u32),
                    Scalar::U32(padding as u32),
                    Scalar::U32(output_padding as u32),
                    Scalar::U32(stride as u32),
                    Scalar::U32(dilation as u32),
                ];
                let op = Op::Conv(op::ConvOp::ConvTranspose3d, self.id(), weight.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id(), weight.id()])?;
            }

            Ok(result)
        }
    }

    // Convolution gradient operations (internal use for backpropagation)
    pub(crate) fn conv1d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let grad_output_layout = grad_output.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_output_shape = grad_output_layout.get_shape();

        // Input: [batch, in_channels, length]
        // GradOutput: [batch, out_channels, length_out]
        // Weight: [out_channels, in_channels, kernel_size]
        if input_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv1d_grad_weight - input must be 3D".to_string(),
            });
        }
        if grad_output_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
                op: "conv1d_grad_weight - grad_output must be 3D".to_string(),
            });
        }
        if weight_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv1d_grad_weight - weight_shape must be 3D".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let length_input = input_shape[2];
        let channels_output = grad_output_shape[1];
        let kernel_size = weight_shape[2];

        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(length_input as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_size as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];

        let op = Op::Conv(
            op::ConvOp::Conv1dGradWeight,
            self.id(),
            grad_output.id(),
            params_scalars,
        );

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(weight_shape);
            let requires_grad = false; // Gradient of gradient not tracked
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op,
                vec![result_id],
                vec![input_layout.clone(), grad_output_layout.clone()],
                vec![result_layout],
            );

            Ok(result_tensor)
        } else {
            let params = op::conv::ParamsConv1D {
                batch_size,
                length_input,
                channels_output,
                channels_input,
                kernel_size,
                padding,
                stride,
                dilation,
            };

            let storage = self.with_storage(|input_storage| {
                grad_output.with_storage(|grad_output_storage| {
                    input_storage.conv1d_grad_weight(grad_output_storage, &input_layout, &grad_output_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(weight_shape);
            let result = from_storage_with_grad(storage, result_layout, true, false);

            Ok(result)
        }
    }

    pub(crate) fn conv2d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let grad_output_layout = grad_output.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_output_shape = grad_output_layout.get_shape();

        // Input: [batch, in_channels, height, width]
        // GradOutput: [batch, out_channels, height_out, width_out]
        // Weight: [out_channels, in_channels, kernel_h, kernel_w]
        if input_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv2d_grad_weight - input must be 4D".to_string(),
            });
        }
        if grad_output_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
                op: "conv2d_grad_weight - grad_output must be 4D".to_string(),
            });
        }
        if weight_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv2d_grad_weight - weight_shape must be 4D".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let height_input = input_shape[2];
        let width_input = input_shape[3];
        let channels_output = grad_output_shape[1];
        let kernel_h = weight_shape[2];
        let kernel_w = weight_shape[3];

        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(height_input as u32),
            Scalar::U32(width_input as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_h as u32),
            Scalar::U32(kernel_w as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];

        let op = Op::Conv(
            op::ConvOp::Conv2dGradWeight,
            self.id(),
            grad_output.id(),
            params_scalars,
        );

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(weight_shape);
            let requires_grad = false;
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op,
                vec![result_id],
                vec![input_layout.clone(), grad_output_layout.clone()],
                vec![result_layout],
            );

            Ok(result_tensor)
        } else {
            let params = op::conv::ParamsConv2D {
                batch_size,
                input_height: height_input,
                input_width: width_input,
                kernel_height: kernel_h,
                kernel_width: kernel_w,
                channels_output,
                channels_input,
                padding,
                stride,
                dilation,
            };

            let storage = self.with_storage(|input_storage| {
                grad_output.with_storage(|grad_output_storage| {
                    input_storage.conv2d_grad_weight(grad_output_storage, &input_layout, &grad_output_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(weight_shape);
            let result = from_storage_with_grad(storage, result_layout, true, false);

            Ok(result)
        }
    }

    pub(crate) fn conv3d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let grad_output_layout = grad_output.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_output_shape = grad_output_layout.get_shape();

        // Input: [batch, in_channels, depth, height, width]
        // GradOutput: [batch, out_channels, depth_out, height_out, width_out]
        // Weight: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
        if input_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv3d_grad_weight - input must be 5D".to_string(),
            });
        }
        if grad_output_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
                op: "conv3d_grad_weight - grad_output must be 5D".to_string(),
            });
        }
        if weight_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv3d_grad_weight - weight_shape must be 5D".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let depth_input = input_shape[2];
        let height_input = input_shape[3];
        let width_input = input_shape[4];
        let channels_output = grad_output_shape[1];
        let kernel_d = weight_shape[2];
        let kernel_h = weight_shape[3];
        let kernel_w = weight_shape[4];

        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(depth_input as u32),
            Scalar::U32(height_input as u32),
            Scalar::U32(width_input as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_d as u32),
            Scalar::U32(kernel_h as u32),
            Scalar::U32(kernel_w as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];

        let op = Op::Conv(
            op::ConvOp::Conv3dGradWeight,
            self.id(),
            grad_output.id(),
            params_scalars,
        );

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(weight_shape);
            let requires_grad = false;
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op,
                vec![result_id],
                vec![input_layout.clone(), grad_output_layout.clone()],
                vec![result_layout],
            );

            Ok(result_tensor)
        } else {
            let params = op::conv::ParamsConv3D {
                batch_size,
                input_depth: depth_input,
                input_height: height_input,
                input_width: width_input,
                kernel_depth: kernel_d,
                kernel_height: kernel_h,
                kernel_width: kernel_w,
                channels_output,
                channels_input,
                padding,
                stride,
                dilation,
            };

            let storage = self.with_storage(|input_storage| {
                grad_output.with_storage(|grad_output_storage| {
                    input_storage.conv3d_grad_weight(grad_output_storage, &input_layout, &grad_output_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(weight_shape);
            let result = from_storage_with_grad(storage, result_layout, true, false);

            Ok(result)
        }
    }

    pub(crate) fn conv_transpose1d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let grad_output_layout = grad_output.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_output_shape = grad_output_layout.get_shape();

        // Input: [batch, in_channels, length_in]
        // GradOutput: [batch, out_channels, length_out]
        // Weight: [in_channels, out_channels, kernel_size]
        if input_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose1d_grad_weight - input must be 3D".to_string(),
            });
        }
        if grad_output_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose1d_grad_weight - grad_output must be 3D".to_string(),
            });
        }
        if weight_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose1d_grad_weight - weight_shape must be 3D".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let length_input = input_shape[2];
        let channels_output = grad_output_shape[1];
        let kernel_size = weight_shape[2];

        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(length_input as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_size as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(output_padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];

        let op = Op::Conv(
            op::ConvOp::ConvTranspose1dGradWeight,
            self.id(),
            grad_output.id(),
            params_scalars,
        );

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(weight_shape);
            let requires_grad = false;
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);
            register_operation_in_builder(
                op,
                vec![result_id],
                vec![input_layout.clone(), grad_output_layout.clone()],
                vec![result_layout],
            );
            Ok(result_tensor)
        } else {
            let params = op::conv::ParamsConvTranspose1D {
                batch_size,
                length_input,
                channels_output,
                channels_input,
                kernel_size,
                padding,
                stride,
                output_padding,
                dilation,
            };
            let storage = self.with_storage(|input_storage| {
                grad_output.with_storage(|grad_output_storage| {
                    input_storage.conv_transpose1d_grad_weight(
                        grad_output_storage,
                        &input_layout,
                        &grad_output_layout,
                        &params,
                    )
                })
            })?;
            let result_layout = Layout::from_shape(weight_shape);
            let result = from_storage_with_grad(storage, result_layout, true, false);
            Ok(result)
        }
    }

    pub(crate) fn conv_transpose2d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let grad_output_layout = grad_output.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_output_shape = grad_output_layout.get_shape();

        // Input: [batch, in_channels, height_in, width_in]
        // GradOutput: [batch, out_channels, height_out, width_out]
        // Weight: [in_channels, out_channels, kernel_height, kernel_width]
        if input_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose2d_grad_weight - input must be 4D".to_string(),
            });
        }
        if grad_output_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose2d_grad_weight - grad_output must be 4D".to_string(),
            });
        }
        if weight_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose2d_grad_weight - weight_shape must be 4D".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];
        let channels_output = grad_output_shape[1];
        let kernel_height = weight_shape[2];
        let kernel_width = weight_shape[3];

        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(input_height as u32),
            Scalar::U32(input_width as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_height as u32),
            Scalar::U32(kernel_width as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(output_padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];

        let op = Op::Conv(
            op::ConvOp::ConvTranspose2dGradWeight,
            self.id(),
            grad_output.id(),
            params_scalars,
        );

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(weight_shape);
            let requires_grad = false;
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);
            register_operation_in_builder(
                op,
                vec![result_id],
                vec![input_layout.clone(), grad_output_layout.clone()],
                vec![result_layout],
            );
            Ok(result_tensor)
        } else {
            let params = op::conv::ParamsConvTranspose2D {
                batch_size,
                input_height,
                input_width,
                kernel_height,
                kernel_width,
                channels_output,
                channels_input,
                padding,
                stride,
                output_padding,
                dilation,
            };
            let storage = self.with_storage(|input_storage| {
                grad_output.with_storage(|grad_output_storage| {
                    input_storage.conv_transpose2d_grad_weight(
                        grad_output_storage,
                        &input_layout,
                        &grad_output_layout,
                        &params,
                    )
                })
            })?;
            let result_layout = Layout::from_shape(weight_shape);
            let result = from_storage_with_grad(storage, result_layout, true, false);
            Ok(result)
        }
    }

    pub(crate) fn conv_transpose3d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let grad_output_layout = grad_output.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_output_shape = grad_output_layout.get_shape();

        // Input: [batch, in_channels, depth_in, height_in, width_in]
        // GradOutput: [batch, out_channels, depth_out, height_out, width_out]
        // Weight: [in_channels, out_channels, kernel_depth, kernel_height, kernel_width]
        if input_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose3d_grad_weight - input must be 5D".to_string(),
            });
        }
        if grad_output_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose3d_grad_weight - grad_output must be 5D".to_string(),
            });
        }
        if weight_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose3d_grad_weight - weight_shape must be 5D".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let input_depth = input_shape[2];
        let input_height = input_shape[3];
        let input_width = input_shape[4];
        let channels_output = grad_output_shape[1];
        let kernel_depth = weight_shape[2];
        let kernel_height = weight_shape[3];
        let kernel_width = weight_shape[4];

        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(input_depth as u32),
            Scalar::U32(input_height as u32),
            Scalar::U32(input_width as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_depth as u32),
            Scalar::U32(kernel_height as u32),
            Scalar::U32(kernel_width as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(output_padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];

        let op = Op::Conv(
            op::ConvOp::ConvTranspose3dGradWeight,
            self.id(),
            grad_output.id(),
            params_scalars,
        );

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(weight_shape);
            let requires_grad = false;
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);
            register_operation_in_builder(
                op,
                vec![result_id],
                vec![input_layout.clone(), grad_output_layout.clone()],
                vec![result_layout],
            );
            Ok(result_tensor)
        } else {
            let params = op::conv::ParamsConvTranspose3D {
                batch_size,
                input_depth,
                input_height,
                input_width,
                kernel_depth,
                kernel_height,
                kernel_width,
                channels_output,
                channels_input,
                padding,
                stride,
                output_padding,
                dilation,
            };
            let storage = self.with_storage(|input_storage| {
                grad_output.with_storage(|grad_output_storage| {
                    input_storage.conv_transpose3d_grad_weight(
                        grad_output_storage,
                        &input_layout,
                        &grad_output_layout,
                        &params,
                    )
                })
            })?;
            let result_layout = Layout::from_shape(weight_shape);
            let result = from_storage_with_grad(storage, result_layout, true, false);
            Ok(result)
        }
    }

    // Windowing Operations
    pub fn reduce_window(
        &self,
        window_shape: &[usize],
        strides: &[usize],
        padding: &[(usize, usize)],
        reduction: &str,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let input_shape = input_layout.get_shape();
        let rank = input_shape.len();

        // Parse reduction type
        let reduction_type = match reduction.to_lowercase().as_str() {
            "max" => op::window_reduction::WindowReduction::Max,
            "mean" => op::window_reduction::WindowReduction::Mean,
            "sum" => op::window_reduction::WindowReduction::Sum,
            "min" => op::window_reduction::WindowReduction::Min,
            _ => {
                return Err(HoduError::InternalError(format!(
                    "Invalid reduction type '{}'. Must be one of: 'max', 'mean', 'sum', 'min'",
                    reduction
                )))
            },
        };

        // Validate inputs
        if window_shape.len() != rank {
            return Err(HoduError::InternalError(format!(
                "window_shape length {} must match tensor rank {}",
                window_shape.len(),
                rank
            )));
        }
        if strides.len() != rank {
            return Err(HoduError::InternalError(format!(
                "strides length {} must match tensor rank {}",
                strides.len(),
                rank
            )));
        }
        if padding.len() != rank {
            return Err(HoduError::InternalError(format!(
                "padding length {} must match tensor rank {}",
                padding.len(),
                rank
            )));
        }

        // Pack parameters: rank, window_shape, strides, padding, reduction_type
        let mut params_scalars = vec![Scalar::U32(rank as u32)];
        for &ws in window_shape {
            params_scalars.push(Scalar::U32(ws as u32));
        }
        for &s in strides {
            params_scalars.push(Scalar::U32(s as u32));
        }
        for &(pad_lo, pad_hi) in padding {
            params_scalars.push(Scalar::U32(pad_lo as u32));
            params_scalars.push(Scalar::U32(pad_hi as u32));
        }
        params_scalars.push(Scalar::U32(match reduction_type {
            op::window_reduction::WindowReduction::Max => 0,
            op::window_reduction::WindowReduction::Mean => 1,
            op::window_reduction::WindowReduction::Sum => 2,
            op::window_reduction::WindowReduction::Min => 3,
        }));

        // Validate dtype for device and operation
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "reduce_window")?;
        let op = Op::Windowing(op::WindowingOp::ReduceWindow, self.id(), params_scalars.clone());
        validate_dtype_for_op(self.get_dtype(), &op)?;

        // Calculate output shape
        let mut output_shape = Vec::with_capacity(rank);
        for i in 0..rank {
            let padded_size = input_shape[i] + padding[i].0 + padding[i].1;
            let out_size = (padded_size - window_shape[i]) / strides[i] + 1;
            output_shape.push(out_size);
        }

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                input_storage.reduce_window(&input_layout, window_shape, strides, padding, reduction_type)
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let mut params_scalars = vec![Scalar::U32(rank as u32)];
                for &ws in window_shape {
                    params_scalars.push(Scalar::U32(ws as u32));
                }
                for &s in strides {
                    params_scalars.push(Scalar::U32(s as u32));
                }
                for &(pad_lo, pad_hi) in padding {
                    params_scalars.push(Scalar::U32(pad_lo as u32));
                    params_scalars.push(Scalar::U32(pad_hi as u32));
                }
                params_scalars.push(Scalar::U32(match reduction_type {
                    op::window_reduction::WindowReduction::Max => 0,
                    op::window_reduction::WindowReduction::Mean => 1,
                    op::window_reduction::WindowReduction::Sum => 2,
                    op::window_reduction::WindowReduction::Min => 3,
                }));

                let op = Op::Windowing(op::WindowingOp::ReduceWindow, self.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    // Shape Operations
    pub fn reshape<S: AsRef<[usize]>>(&self, shape: S) -> HoduResult<Self> {
        let new_shape = shape.as_ref();
        let current_layout = self.get_layout();
        let current_size = current_layout.get_size();
        let new_size = new_shape.iter().product::<usize>();

        // Check that total size remains the same
        if current_size != new_size {
            return Err(HoduError::IncompatibleShapes {
                lhs: current_layout.get_shape().to_vec(),
                rhs: new_shape.to_vec(),
                op: "reshape - total size must remain the same".to_string(),
            });
        }

        // First check if tensor is contiguous
        if !current_layout.is_contiguous() {
            // If not contiguous, make it contiguous first then reshape
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.reshape(new_shape);
        }

        let new_layout = Layout::from_shape(new_shape);
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);

            let op = Op::Shape(op::ShapeOp::Reshape, self.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![current_layout.clone()],
                vec![new_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Reshape, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn view<S: AsRef<[usize]>>(&self, shape: S) -> HoduResult<Self> {
        // view is an alias for reshape
        self.reshape(shape)
    }

    pub fn flatten(&self) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let total_size = current_layout.get_size();
        let new_shape = vec![total_size];
        let new_layout = Layout::from_shape(&new_shape);
        let requires_grad = self.is_requires_grad();

        // First check if tensor is contiguous
        if !current_layout.is_contiguous() {
            // If not contiguous, make it contiguous first then flatten
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.flatten();
        }

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);

            let op = Op::Shape(op::ShapeOp::Flatten, self.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![current_layout.clone()],
                vec![new_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Flatten, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn squeeze<D: Into<Scalar> + Clone>(&self, dim: Option<D>) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let current_shape = current_layout.get_shape();
        let ndim = current_shape.len();

        let new_shape = if let Some(ref dim) = dim {
            // Squeeze specific dimension
            let dim_scalar = dim.clone().into();
            let dim_i32 = dim_scalar.to_i64() as i32;
            let actual_dim = if dim_i32 < 0 {
                (ndim as i32 + dim_i32) as usize
            } else {
                dim_i32 as usize
            };

            if actual_dim >= ndim {
                return Err(HoduError::IncompatibleShapes {
                    lhs: current_shape.to_vec(),
                    rhs: vec![],
                    op: format!(
                        "squeeze - dimension {} out of range for {}-dimensional tensor",
                        dim_i32, ndim
                    ),
                });
            }

            if current_shape[actual_dim] != 1 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: current_shape.to_vec(),
                    rhs: vec![],
                    op: format!(
                        "squeeze - cannot squeeze dimension {} with size {}",
                        dim_i32, current_shape[actual_dim]
                    ),
                });
            }

            let mut new_shape = current_shape.to_vec();
            new_shape.remove(actual_dim);
            new_shape
        } else {
            // Squeeze all dimensions of size 1
            current_shape.iter().filter(|&&size| size != 1).copied().collect()
        };

        // First check if tensor is contiguous
        if !current_layout.is_contiguous() {
            // If not contiguous, make it contiguous first then squeeze
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.squeeze(dim.clone());
        }

        let new_layout = Layout::from_shape(&new_shape);
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);

            let op = Op::Shape(op::ShapeOp::Squeeze, self.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![current_layout.clone()],
                vec![new_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Squeeze, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn unsqueeze<D: Into<Scalar>>(&self, dim: D) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let current_shape = current_layout.get_shape();
        let ndim = current_shape.len();

        // Convert negative dimension to positive
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i64() as i32;
        let actual_dim = if dim_i32 < 0 {
            (ndim as i32 + dim_i32 + 1) as usize
        } else {
            dim_i32 as usize
        };

        // Check bounds (can insert at position 0 to ndim inclusive)
        if actual_dim > ndim {
            return Err(HoduError::IncompatibleShapes {
                lhs: current_shape.to_vec(),
                rhs: vec![],
                op: format!(
                    "unsqueeze - dimension {} out of range for {}-dimensional tensor",
                    dim_i32, ndim
                ),
            });
        }

        // Create new shape with dimension of size 1 inserted
        let mut new_shape = current_shape.to_vec();
        new_shape.insert(actual_dim, 1);

        // First check if tensor is contiguous
        if !current_layout.is_contiguous() {
            // If not contiguous, make it contiguous first then unsqueeze
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.unsqueeze(dim_scalar);
        }

        let new_layout = Layout::from_shape(&new_shape);
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);

            let op = Op::Shape(op::ShapeOp::Unsqueeze, self.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![current_layout.clone()],
                vec![new_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Unsqueeze, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn broadcast(&self, shape: &[usize]) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let target_layout = current_layout.broadcast_to(shape)?;

        // First check if tensor is contiguous
        if !current_layout.is_contiguous() {
            // If not contiguous, make it contiguous first then broadcast
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.broadcast(shape);
        }

        if builder::is_builder_active() {
            let requires_grad = self.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(target_layout.clone(), requires_grad);

            let op = Op::Shape(op::ShapeOp::Broadcast, self.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![current_layout.clone()],
                vec![target_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, target_layout, self.is_requires_grad());

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Broadcast, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn broadcast_like(&self, other: &Self) -> HoduResult<Self> {
        let other_layout = other.get_layout();
        let other_shape = other_layout.get_shape();
        self.broadcast(other_shape)
    }

    pub fn broadcast_left(&self, left_shape: &[usize]) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let current_shape = current_layout.get_shape();

        let mut target_shape = left_shape.to_vec();
        target_shape.extend_from_slice(current_shape);

        self.broadcast(&target_shape)
    }

    pub fn transpose<D1: Into<Scalar>, D2: Into<Scalar>>(&self, dim1: D1, dim2: D2) -> HoduResult<Self> {
        let layout = self.get_layout();

        // Convert scalars to i32 for layout.transpose
        let dim1_scalar = dim1.into();
        let dim2_scalar = dim2.into();
        let dim1_i32 = dim1_scalar.to_i64() as i32;
        let dim2_i32 = dim2_scalar.to_i64() as i32;

        let new_layout = layout.transpose(dim1_i32, dim2_i32)?;
        let requires_grad = self.is_requires_grad();

        // First check if tensor is contiguous
        if !layout.is_contiguous() {
            // If not contiguous, make it contiguous first then transpose
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.transpose(dim1_scalar, dim2_scalar);
        }

        if builder::is_builder_active() {
            let (tensor_id, result) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);
            let op = Op::Shape(op::ShapeOp::Transpose, self.id());
            register_operation_in_builder(op.clone(), vec![tensor_id], vec![self.get_layout()], vec![new_layout]);

            if self.is_requires_grad() {
                gradient::record_operation(tensor_id, op, vec![self.id()])?;
            }

            Ok(result)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Transpose, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn t(&self) -> HoduResult<Self> {
        self.transpose(-2, -1)
    }

    pub fn permute<A: Into<Scalar> + Copy>(&self, axes: &[A]) -> HoduResult<Self> {
        let layout = self.get_layout();
        let shape = layout.get_shape();
        let ndim = shape.len();

        // Validate axes length
        if axes.len() != ndim {
            return Err(HoduError::IncompatibleShapes {
                lhs: shape.to_vec(),
                rhs: vec![],
                op: format!(
                    "permute - axes length {} must match tensor dimensions {}",
                    axes.len(),
                    ndim
                ),
            });
        }

        // Convert Scalar axes to usize, handling negative indices
        let mut axes_usize = Vec::with_capacity(ndim);
        for &axis in axes {
            let axis_scalar = axis.into();
            let axis_i32 = axis_scalar.to_i64() as i32;
            let actual_axis = if axis_i32 < 0 {
                (ndim as i32 + axis_i32) as usize
            } else {
                axis_i32 as usize
            };

            if actual_axis >= ndim {
                return Err(HoduError::IncompatibleShapes {
                    lhs: shape.to_vec(),
                    rhs: vec![],
                    op: format!(
                        "permute - axis {} out of range for {}-dimensional tensor",
                        axis_i32, ndim
                    ),
                });
            }

            axes_usize.push(actual_axis);
        }

        // Check that axes contains each dimension exactly once
        let mut seen = vec![false; ndim];
        for &axis in &axes_usize {
            if seen[axis] {
                return Err(HoduError::IncompatibleShapes {
                    lhs: shape.to_vec(),
                    rhs: axes_usize.clone(),
                    op: format!("permute - duplicate axis {} in permutation", axis),
                });
            }
            seen[axis] = true;
        }

        let new_layout = layout.permute(&axes_usize)?;
        let requires_grad = self.is_requires_grad();

        // First check if tensor is contiguous
        if !layout.is_contiguous() {
            // If not contiguous, make it contiguous first then permute
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.permute(axes);
        }

        if builder::is_builder_active() {
            let (tensor_id, result) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);
            let op = Op::Shape(op::ShapeOp::Permute, self.id());
            register_operation_in_builder(op.clone(), vec![tensor_id], vec![self.get_layout()], vec![new_layout]);

            if self.is_requires_grad() {
                gradient::record_operation(tensor_id, op, vec![self.id()])?;
            }

            Ok(result)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Permute, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn slice<S: Into<Scalar> + Copy>(&self, dim: usize, start: S, end: Option<S>, step: S) -> HoduResult<Self> {
        let layout = self.get_layout();

        // Convert Scalar to isize
        let start_scalar = start.into();
        let start_isize = start_scalar.to_i64() as isize;

        let end_isize = end.map(|e| {
            let end_scalar = e.into();
            end_scalar.to_i64() as isize
        });

        let step_scalar = step.into();
        let step_isize = step_scalar.to_i64() as isize;

        let new_layout = layout.slice(dim, start_isize, end_isize, step_isize)?;
        let requires_grad = self.is_requires_grad();

        // First check if tensor is contiguous
        if !layout.is_contiguous() {
            // If not contiguous, make it contiguous first then slice
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.slice(dim, start, end, step);
        }

        // Store slice parameters in scalars: [dim, start, end_or_max, step]
        // Use i32::MAX to represent None for end
        let end_value = end_isize.unwrap_or(i32::MAX as isize);
        let scalars = vec![
            Scalar::I32(dim as i32),
            Scalar::I32(start_isize as i32),
            Scalar::I32(end_value as i32),
            Scalar::I32(step_isize as i32),
        ];

        if builder::is_builder_active() {
            let (tensor_id, result) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);
            let op = Op::ShapeScalars(op::ShapeScalarsOp::Slice, self.id(), scalars.clone());
            register_operation_in_builder(op.clone(), vec![tensor_id], vec![self.get_layout()], vec![new_layout]);

            if self.is_requires_grad() {
                gradient::record_operation(tensor_id, op, vec![self.id()])?;
            }

            Ok(result)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::ShapeScalars(op::ShapeScalarsOp::Slice, self.id(), scalars);
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    // Cast Operations
    pub fn to_dtype(&self, dtype: DType) -> HoduResult<Self> {
        // Validate that target dtype is supported on current device
        validate_dtype_for_device(dtype, &self.get_device(), "to_dtype")?;

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(self.get_layout().get_shape());
            let requires_grad = self.is_requires_grad() && dtype.is_float();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let op = Op::Cast(op::CastOp::ToDType, self.id());
            register_operation_in_builder(
                op,
                vec![result_id],
                vec![self.get_layout().clone()],
                vec![result_layout],
            );

            Ok(result_tensor)
        } else {
            let layout = Layout::from_shape(self.get_layout().get_shape());
            let storage = self.with_storage(|storage| storage.to_dtype(dtype, &self.get_layout()))?;
            let requires_grad = self.is_requires_grad() && dtype.is_float();
            let result = from_storage_with_grad(storage, layout, true, requires_grad);

            Ok(result)
        }
    }

    // Memory Operations
    pub fn contiguous(&self) -> HoduResult<Self> {
        let layout = self.get_layout();

        // If already contiguous, return self
        if layout.is_contiguous() {
            return Ok(*self);
        }

        if builder::is_builder_active() {
            let contiguous_layout = Layout::from_shape(layout.get_shape());
            let requires_grad = self.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(contiguous_layout.clone(), requires_grad);

            let op = Op::Memory(op::MemoryOp::Contiguous, self.id());
            register_operation_in_builder(op, vec![result_id], vec![layout], vec![contiguous_layout]);

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.contiguous(&layout))?;
            let contiguous_layout = Layout::from_shape(layout.get_shape());
            let requires_grad = self.is_requires_grad();
            let result = from_storage_with_grad(storage, contiguous_layout, true, requires_grad);

            Ok(result)
        }
    }

    pub fn set_(&mut self, src: &Tensor) -> HoduResult<()> {
        self.0 = src.id();
        Ok(())
    }
}
