use crate::{
    error::HoduResult,
    layer::compat::*,
    ops::{Op, OpParams, UnaryLogicalOp, UnaryOp, UnaryScalarOp},
    scalar::Scalar,
    script::builder,
    tensor::{create_builder_tensor, from_storage_with_context, gradient, register_operation_in_builder, Tensor},
    types::Layout,
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op},
};

macro_rules! unary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> HoduResult<Self> {
            validate_dtype_for_device(self.dtype(), self.device())?;
            validate_dtype_for_op(self.dtype(), Op::Unary(UnaryOp::$op_name))?;
            let validate_requires_grad = validate_requires_grad_for_op(Op::Unary(UnaryOp::$op_name));

            let input_layout = self.layout();

            if builder::is_builder_active() {
                let requires_grad = self.is_requires_grad() && validate_requires_grad;
                let (result_id, result_tensor) = create_builder_tensor(input_layout.clone(), requires_grad);

                register_operation_in_builder(
                    Op::Unary(UnaryOp::$op_name),
                    None,
                    vec![self.id()],
                    vec![result_id],
                    vec![input_layout.clone()],
                    vec![input_layout],
                )?;

                if requires_grad {
                    gradient::record_operation(result_id, Op::Unary(UnaryOp::$op_name), vec![self.id()])?;
                }

                Ok(result_tensor)
            } else {
                let storage =
                    self.with_storage(|storage| storage.call_ops_unary(&input_layout, Op::Unary(UnaryOp::$op_name)))?;

                let requires_grad = self.is_requires_grad() && validate_requires_grad;
                let layout = Layout::from_shape(&self.shape());

                let result = from_storage_with_context(storage, layout, true, requires_grad);

                if !gradient::is_computing_gradients() && requires_grad {
                    let op = Op::Unary(UnaryOp::$op_name);
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
            validate_dtype_for_device(self.dtype(), self.device())?;
            validate_dtype_for_op(self.dtype(), Op::UnaryLogical(UnaryLogicalOp::$op_name))?;

            let input_layout = self.layout();

            if builder::is_builder_active() {
                let (result_id, result_tensor) = create_builder_tensor(input_layout.clone(), false);

                register_operation_in_builder(
                    Op::UnaryLogical(UnaryLogicalOp::$op_name),
                    None,
                    vec![self.id()],
                    vec![result_id],
                    vec![input_layout.clone()],
                    vec![input_layout],
                )?;

                Ok(result_tensor)
            } else {
                let storage = self.with_storage(|storage| {
                    storage.call_ops_unary_logical(&input_layout, Op::UnaryLogical(UnaryLogicalOp::$op_name))
                })?;

                let layout = Layout::from_shape(&self.shape());

                let result = from_storage_with_context(storage, layout, true, false);

                Ok(result)
            }
        }
    };
}

macro_rules! unary_scalar_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name<T: Into<Scalar>>(&self, scalar: T) -> HoduResult<Self> {
            validate_dtype_for_device(self.dtype(), self.device())?;
            validate_dtype_for_op(self.dtype(), Op::UnaryScalar(UnaryScalarOp::$op_name))?;
            let validate_requires_grad = validate_requires_grad_for_op(Op::UnaryScalar(UnaryScalarOp::$op_name));

            let scalar_value = scalar.into();
            let input_layout = self.layout();

            if builder::is_builder_active() {
                let requires_grad = self.is_requires_grad() && validate_requires_grad;
                let (result_id, result_tensor) = create_builder_tensor(input_layout.clone(), requires_grad);

                register_operation_in_builder(
                    Op::UnaryScalar(UnaryScalarOp::$op_name),
                    Some(OpParams {
                        scalar: Some(scalar_value),
                        ..Default::default()
                    }),
                    vec![self.id()],
                    vec![result_id],
                    vec![input_layout.clone()],
                    vec![input_layout],
                )?;

                if requires_grad {
                    gradient::record_operation_with_scalar(
                        result_id,
                        Op::UnaryScalar(UnaryScalarOp::$op_name),
                        vec![self.id()],
                        scalar_value,
                    )?;
                }

                Ok(result_tensor)
            } else {
                let storage = self.with_storage(|storage| {
                    storage.call_ops_unary_scalar(
                        &input_layout,
                        scalar_value,
                        Op::UnaryScalar(UnaryScalarOp::$op_name),
                    )
                })?;

                let requires_grad = self.is_requires_grad() && validate_requires_grad;
                let layout = Layout::from_shape(&self.shape());

                let result = from_storage_with_context(storage, layout, true, requires_grad);

                if !gradient::is_computing_gradients() && requires_grad {
                    let op = Op::UnaryScalar(UnaryScalarOp::$op_name);
                    gradient::record_operation_with_scalar(result.id(), op, vec![self.id()], scalar_value)?;
                }

                Ok(result)
            }
        }
    };
}

impl Tensor {
    // unary operations
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

    // unary logical operations
    unary_logical_op!(logical_not, LogicalNot);

    // unary scalar operations
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
