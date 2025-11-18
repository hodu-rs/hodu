use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{Op, OpParams, WindowingOp},
    scalar::Scalar,
    script::builder,
    tensor::{create_builder_tensor, from_storage, gradient, register_operation_in_builder, Tensor},
    types::{Layout, Shape},
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op},
};

impl Tensor {
    pub fn reduce_window(
        &self,
        window_shape: impl Into<Shape>,
        strides: impl Into<Shape>,
        padding: &[(usize, usize)],
        reduction: &str,
    ) -> HoduResult<Self> {
        let window_shape = window_shape.into();
        let strides = strides.into();

        let input_shape = self.shape();
        let rank = input_shape.ndim();

        // Parse reduction type
        let windowing_op = match reduction.to_lowercase().as_str() {
            "max" => WindowingOp::ReduceWindowMax,
            "mean" => WindowingOp::ReduceWindowMean,
            "sum" => WindowingOp::ReduceWindowSum,
            "min" => WindowingOp::ReduceWindowMin,
            _ => {
                return Err(HoduError::InvalidLayout {
                    reason: format!(
                        "invalid reduction type '{}'. Must be one of: 'max', 'mean', 'sum', 'min'",
                        reduction
                    ),
                })
            },
        };

        // Validate inputs
        if window_shape.ndim() != rank {
            return Err(HoduError::InvalidLayout {
                reason: format!(
                    "window_shape length {} must match tensor rank {}",
                    window_shape.ndim(),
                    rank
                ),
            });
        }
        if strides.ndim() != rank {
            return Err(HoduError::InvalidLayout {
                reason: format!("strides length {} must match tensor rank {}", strides.ndim(), rank),
            });
        }
        if padding.len() != rank {
            return Err(HoduError::InvalidLayout {
                reason: format!("padding length {} must match tensor rank {}", padding.len(), rank),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Windowing(windowing_op))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Windowing(windowing_op));

        // Calculate output shape
        let input_dims = input_shape.dims();
        let window_dims = window_shape.dims();
        let stride_dims = strides.dims();
        let mut output_dims = Vec::with_capacity(rank);
        for i in 0..rank {
            let padded_size = input_dims[i] + padding[i].0 + padding[i].1;
            let out_size = (padded_size - window_dims[i]) / stride_dims[i] + 1;
            output_dims.push(out_size);
        }

        // Flatten padding to [lo, hi, lo, hi, ...]
        let mut padding_flat = Vec::with_capacity(padding.len() * 2);
        for &(lo, hi) in padding {
            padding_flat.push(lo);
            padding_flat.push(hi);
        }

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&Shape::from(output_dims));
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), requires_grad);

            let mut scalars = Vec::new();
            // Add window_shape
            for &dim in window_dims {
                scalars.push(Scalar::from(dim));
            }
            // Add strides
            for &stride in stride_dims {
                scalars.push(Scalar::from(stride));
            }
            // Add padding
            for &pad in &padding_flat {
                scalars.push(Scalar::from(pad));
            }

            let op_params = OpParams {
                scalars,
                ..Default::default()
            };

            register_operation_in_builder(
                Op::Windowing(windowing_op),
                Some(op_params),
                vec![self.id()],
                vec![result_id],
                vec![self.layout()],
                vec![result_layout],
            )?;

            if requires_grad {
                let mut grad_scalars = Vec::new();
                grad_scalars.push(Scalar::from(rank));
                for &dim in window_dims {
                    grad_scalars.push(Scalar::from(dim));
                }
                for &stride in stride_dims {
                    grad_scalars.push(Scalar::from(stride));
                }
                for &pad in &padding_flat {
                    grad_scalars.push(Scalar::from(pad));
                }

                gradient::record_operation_with_scalars(
                    result_id,
                    Op::Windowing(windowing_op),
                    vec![self.id()],
                    grad_scalars,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                input_storage.call_ops_reduce_window(
                    &self.layout(),
                    window_dims,
                    stride_dims,
                    &padding_flat,
                    Op::Windowing(windowing_op),
                )
            })?;

            let result_layout = Layout::from_shape(&Shape::from(output_dims));
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let mut scalars = Vec::new();
                scalars.push(Scalar::from(rank));
                for &dim in window_dims {
                    scalars.push(Scalar::from(dim));
                }
                for &stride in stride_dims {
                    scalars.push(Scalar::from(stride));
                }
                for &pad in &padding_flat {
                    scalars.push(Scalar::from(pad));
                }

                gradient::record_operation_with_scalars(
                    result.id(),
                    Op::Windowing(windowing_op),
                    vec![self.id()],
                    scalars,
                )?;
            }

            Ok(result)
        }
    }
}
