use crate::{
    backends::{
        builder,
        op::{self, Op},
    },
    compat::*,
    error::{HoduError, HoduResult},
    scalar::Scalar,
    tensor::{
        create_builder_tensor_with_grad, from_shared_storage_with_grad, from_storage_with_grad, gradient,
        register_operation_in_builder, Tensor,
    },
    types::{dtype::DType, layout::Layout},
};

// Utility function to broadcast two tensors to the same shape
fn broadcast_tensors(a: &Tensor, b: &Tensor) -> HoduResult<(Tensor, Tensor)> {
    let a_layout = a.get_layout();
    let b_layout = b.get_layout();
    let a_shape = a_layout.get_shape();
    let b_shape = b_layout.get_shape();
    let a_ndim = a_layout.get_ndim();
    let b_ndim = b_layout.get_ndim();

    let output_ndim = a_ndim.max(b_ndim);
    let mut output_shape = vec![0; output_ndim];

    // Compute output shape from right to left (broadcasting rules)
    for i in 0..output_ndim {
        let a_dim = if i < a_ndim { a_shape[a_ndim - 1 - i] } else { 1 };
        let b_dim = if i < b_ndim { b_shape[b_ndim - 1 - i] } else { 1 };

        if a_dim == 1 || b_dim == 1 || a_dim == b_dim {
            output_shape[output_ndim - 1 - i] = a_dim.max(b_dim);
        } else {
            return Err(HoduError::IncompatibleShapes {
                lhs: a_shape.to_vec(),
                rhs: b_shape.to_vec(),
                op: "broadcast_tensors".to_string(),
            });
        }
    }

    let a_broadcasted = a.broadcast(&output_shape)?;
    let b_broadcasted = b.broadcast(&output_shape)?;

    Ok((a_broadcasted, b_broadcasted))
}

macro_rules! binary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> HoduResult<Self> {
            let (lhs_broadcasted, rhs_broadcasted) = broadcast_tensors(self, rhs)?;

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

            if builder::is_builder_active() {
                let result_layout = lhs_broadcasted.get_layout().clone();
                let requires_grad = self.is_requires_grad() || rhs.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                let op = Op::Binary(op::BinaryOp::$op_name, lhs_broadcasted.id(), rhs_broadcasted.id());
                register_operation_in_builder(
                    op.clone(),
                    result_id,
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
            let (lhs_broadcasted, rhs_broadcasted) = broadcast_tensors(self, rhs)?;

            if builder::is_builder_active() {
                let result_layout = lhs_broadcasted.get_layout().clone();
                let requires_grad = self.is_requires_grad() || rhs.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                let op = Op::BinaryLogical(
                    op::BinaryLogicalOp::$op_name,
                    lhs_broadcasted.id(),
                    rhs_broadcasted.id(),
                );
                register_operation_in_builder(
                    op.clone(),
                    result_id,
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
            let (lhs_broadcasted, rhs_broadcasted) = broadcast_tensors(self, rhs)?;

            if builder::is_builder_active() {
                let result_layout = lhs_broadcasted.get_layout().clone();
                let requires_grad = self.is_requires_grad() || rhs.is_requires_grad();
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

                let op = Op::Cmp(op::CmpOp::$op_name, lhs_broadcasted.id(), rhs_broadcasted.id());
                register_operation_in_builder(
                    op.clone(),
                    result_id,
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

            let rhs_reshaped = other.reshape(&[k2, 1])?;
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

            let lhs_reshaped = self.reshape(&[1, k1])?;
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
                result_id,
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

            let rhs_reshaped = other.reshape(&[k2, 1])?;
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

            let lhs_reshaped = self.reshape(&[1, k1])?;
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

            let (m, _k1) = (lhs_shape[0], lhs_shape[1]);
            let (_k2, n) = (rhs_shape[0], rhs_shape[1]);
            let result_shape = vec![m, n];
            let result_layout = Layout::from_shape(&result_shape);

            let storage = self.with_storage(|lhs_storage| {
                other.with_storage(|rhs_storage| {
                    lhs_storage.dot(rhs_storage, &self.get_layout(), &other.get_layout())
                })
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

    pub fn norm(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Norm, dims, keep_dim)
    }

    pub fn l2_norm(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Norm, dims, keep_dim)
    }

    pub fn l1_norm(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.abs()?.sum(dims, keep_dim)
    }

    fn reduce_operation(&self, reduce_op: op::ReduceOp, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
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
                if output_shape.is_empty() {
                    output_shape = vec![1];
                }
            }

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let dims_scalars: Vec<Scalar> = dims.iter().map(|&d| Scalar::U64(d as u64)).collect();
            let op = Op::Reduce(reduce_op, self.id(), dims_scalars);
            register_operation_in_builder(op.clone(), result_id, vec![layout.clone()], vec![result_layout]);

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
                if output_shape.is_empty() {
                    output_shape = vec![1];
                }
            }

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let dims_scalars: Vec<Scalar> = dims.iter().map(|&d| Scalar::U64(d as u64)).collect();
                let op = Op::Reduce(reduce_op, self.id(), dims_scalars);
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
            register_operation_in_builder(op.clone(), result_id, vec![current_layout.clone()], vec![new_layout]);

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
            register_operation_in_builder(op.clone(), result_id, vec![current_layout.clone()], vec![new_layout]);

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

    pub fn squeeze(&self, dim: Option<isize>) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let current_shape = current_layout.get_shape();
        let ndim = current_shape.len();

        let new_shape = if let Some(dim) = dim {
            // Squeeze specific dimension
            let actual_dim = if dim < 0 {
                (ndim as isize + dim) as usize
            } else {
                dim as usize
            };

            if actual_dim >= ndim {
                return Err(HoduError::IncompatibleShapes {
                    lhs: current_shape.to_vec(),
                    rhs: vec![],
                    op: format!(
                        "squeeze - dimension {} out of range for {}-dimensional tensor",
                        dim, ndim
                    ),
                });
            }

            if current_shape[actual_dim] != 1 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: current_shape.to_vec(),
                    rhs: vec![],
                    op: format!(
                        "squeeze - cannot squeeze dimension {} with size {}",
                        dim, current_shape[actual_dim]
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
            return contiguous_tensor.squeeze(dim);
        }

        let new_layout = Layout::from_shape(&new_shape);
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);

            let op = Op::Shape(op::ShapeOp::Squeeze, self.id());
            register_operation_in_builder(op.clone(), result_id, vec![current_layout.clone()], vec![new_layout]);

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

    pub fn unsqueeze(&self, dim: isize) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let current_shape = current_layout.get_shape();
        let ndim = current_shape.len();

        // Convert negative dimension to positive
        let actual_dim = if dim < 0 {
            (ndim as isize + dim + 1) as usize
        } else {
            dim as usize
        };

        // Check bounds (can insert at position 0 to ndim inclusive)
        if actual_dim > ndim {
            return Err(HoduError::IncompatibleShapes {
                lhs: current_shape.to_vec(),
                rhs: vec![],
                op: format!(
                    "unsqueeze - dimension {} out of range for {}-dimensional tensor",
                    dim, ndim
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
            return contiguous_tensor.unsqueeze(dim);
        }

        let new_layout = Layout::from_shape(&new_shape);
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);

            let op = Op::Shape(op::ShapeOp::Unsqueeze, self.id());
            register_operation_in_builder(op.clone(), result_id, vec![current_layout.clone()], vec![new_layout]);

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
            register_operation_in_builder(op.clone(), result_id, vec![current_layout.clone()], vec![target_layout]);

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

    pub fn transpose(&self, dim1: i32, dim2: i32) -> HoduResult<Self> {
        let layout = self.get_layout();
        let new_layout = layout.transpose(dim1, dim2)?;
        let requires_grad = self.is_requires_grad();

        // First check if tensor is contiguous
        if !layout.is_contiguous() {
            // If not contiguous, make it contiguous first then transpose
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.transpose(dim1, dim2);
        }

        if builder::is_builder_active() {
            let (tensor_id, result) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);
            let op = Op::Shape(op::ShapeOp::Transpose, self.id());
            register_operation_in_builder(op.clone(), tensor_id, vec![self.get_layout()], vec![new_layout]);

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

    // Cast Operations
    pub fn to_dtype(&self, dtype: DType) -> HoduResult<Self> {
        if builder::is_builder_active() {
            let result_layout = self.get_layout().clone();
            let requires_grad = self.is_requires_grad() && dtype.is_float();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let op = Op::Cast(op::CastOp::ToDType, self.id());
            register_operation_in_builder(op, result_id, vec![self.get_layout().clone()], vec![result_layout]);

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.to_dtype(dtype))?;
            let layout = self.get_layout().clone();
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
            register_operation_in_builder(op, result_id, vec![layout], vec![contiguous_layout]);

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
