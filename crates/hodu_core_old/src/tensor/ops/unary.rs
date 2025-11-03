use crate::{
    builder,
    compat::*,
    error::HoduResult,
    op::{
        self,
        utils::{validate_dtype_for_device, validate_dtype_for_op},
        Op,
    },
    scalar::Scalar,
    tensor::{
        create_builder_tensor_with_grad, from_storage_with_grad, gradient, register_operation_in_builder, Tensor,
    },
};

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
}
