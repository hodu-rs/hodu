use crate::{
    error::HoduResult,
    ops::{Op, OpParams, UnaryLogicalOp, UnaryLogicalParams, UnaryOp, UnaryParams, UnaryScalarOp, UnaryScalarParams},
    scalar::Scalar,
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::{DType, Layout},
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op},
};

macro_rules! unary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> HoduResult<Self> {
            validate_dtype_for_device(self.dtype(), self.device())?;
            validate_dtype_for_op(self.dtype(), Op::Unary(UnaryOp::$op_name))?;
            let validate_requires_grad = validate_requires_grad_for_op(Op::Unary(UnaryOp::$op_name));

            let input_layout = self.layout();

            if crate::snapshot::capture::is_active() {
                let requires_grad = self.is_requires_grad() && validate_requires_grad;
                let (result_id, result_tensor) =
                    create_builder_tensor(input_layout.clone(), self.dtype(), requires_grad);

                crate::snapshot::capture::capture_operation(
                    Op::Unary(UnaryOp::$op_name),
                    Some(OpParams::Unary(UnaryParams)),
                    vec![self.id()],
                    result_id,
                    vec![input_layout.clone()],
                    input_layout,
                )?;

                if requires_grad {
                    gradient::record_operation(
                        vec![self.id()],
                        result_id,
                        Op::Unary(UnaryOp::$op_name),
                        OpParams::Unary(UnaryParams),
                    )?;
                }

                Ok(result_tensor)
            } else {
                let storage =
                    self.with_storage(|storage| storage.call_ops_unary(&input_layout, Op::Unary(UnaryOp::$op_name)))?;

                let requires_grad = self.is_requires_grad() && validate_requires_grad;
                let layout = Layout::from_shape(&self.shape());

                let result = from_storage_with_context(storage, layout, true, requires_grad);

                if !gradient::is_computing_gradients() && requires_grad {
                    gradient::record_operation(
                        vec![self.id()],
                        result.id(),
                        Op::Unary(UnaryOp::$op_name),
                        OpParams::Unary(UnaryParams),
                    )?;
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

            if crate::snapshot::capture::is_active() {
                let (result_id, result_tensor) = create_builder_tensor(input_layout.clone(), DType::BOOL, false);

                crate::snapshot::capture::capture_operation(
                    Op::UnaryLogical(UnaryLogicalOp::$op_name),
                    Some(OpParams::UnaryLogical(UnaryLogicalParams)),
                    vec![self.id()],
                    result_id,
                    vec![input_layout.clone()],
                    input_layout,
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

            if crate::snapshot::capture::is_active() {
                let requires_grad = self.is_requires_grad() && validate_requires_grad;
                let (result_id, result_tensor) =
                    create_builder_tensor(input_layout.clone(), self.dtype(), requires_grad);

                crate::snapshot::capture::capture_operation(
                    Op::UnaryScalar(UnaryScalarOp::$op_name),
                    Some(OpParams::UnaryScalar(UnaryScalarParams { scalar: scalar_value })),
                    vec![self.id()],
                    result_id,
                    vec![input_layout.clone()],
                    input_layout,
                )?;

                if requires_grad {
                    gradient::record_operation(
                        vec![self.id()],
                        result_id,
                        Op::UnaryScalar(UnaryScalarOp::$op_name),
                        OpParams::UnaryScalar(UnaryScalarParams { scalar: scalar_value }),
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
                    gradient::record_operation(
                        vec![self.id()],
                        result.id(),
                        Op::UnaryScalar(UnaryScalarOp::$op_name),
                        OpParams::UnaryScalar(UnaryScalarParams { scalar: scalar_value }),
                    )?;
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
    unary_op!(hardsigmoid, HardSigmoid);
    unary_op!(gelu, Gelu);
    unary_op!(softplus, Softplus);
    unary_op!(silu, Silu);
    pub fn swish(&self) -> HoduResult<Self> {
        self.silu()
    }
    unary_op!(hardsilu, HardSilu);
    pub fn hardswish(&self) -> HoduResult<Self> {
        self.hardsilu()
    }
    unary_op!(mish, Mish);

    unary_op!(sin, Sin);
    unary_op!(cos, Cos);
    unary_op!(tan, Tan);
    unary_op!(asin, Asin);
    unary_op!(acos, Acos);
    unary_op!(atan, Atan);

    unary_op!(sinh, Sinh);
    unary_op!(cosh, Cosh);
    unary_op!(tanh, Tanh);
    unary_op!(asinh, Asinh);
    unary_op!(acosh, Acosh);
    unary_op!(atanh, Atanh);

    unary_op!(exp, Exp);
    unary_op!(exp2, Exp2);
    unary_op!(exp10, Exp10);
    unary_op!(ln, Ln);
    unary_op!(log2, Log2);
    unary_op!(log10, Log10);

    unary_op!(ceil, Ceil);
    unary_op!(floor, Floor);
    unary_op!(round, Round);

    unary_op!(erf, Erf);

    // unary logical operations
    unary_logical_op!(logical_not, LogicalNot);

    // unary scalar operations
    unary_scalar_op!(add_scalar, AddScalar);
    unary_scalar_op!(sub_scalar, SubScalar);
    unary_scalar_op!(mul_scalar, MulScalar);
    unary_scalar_op!(div_scalar, DivScalar);
    unary_scalar_op!(rem_scalar, RemScalar);
    unary_scalar_op!(pow_scalar, PowScalar);
    unary_scalar_op!(maximum_scalar, MaximumScalar);
    unary_scalar_op!(minimum_scalar, MinimumScalar);

    unary_scalar_op!(leaky_relu, LeakyRelu);
    unary_scalar_op!(elu, Elu);
    unary_scalar_op!(prelu, Prelu);
}
