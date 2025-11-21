use crate::{
    capture,
    compat::*,
    error::{HoduError, HoduResult},
    ops::{Op, OpParams, ReduceWindowParams, WindowingOp},
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
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

        let op_params = OpParams::ReduceWindow(ReduceWindowParams {
            window_shape: window_dims.to_vec(),
            strides: stride_dims.to_vec(),
            padding: padding.to_vec(),
            aux_tensors: vec![],
        });

        if capture::is_active() {
            let result_layout = Layout::from_shape(&Shape::from(output_dims));
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), requires_grad);

            capture::capture_operation(
                Op::Windowing(windowing_op),
                Some(op_params.clone()),
                vec![self.id()],
                result_id,
                vec![self.layout()],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], result_id, Op::Windowing(windowing_op), op_params)?;
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
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(vec![self.id()], result.id(), Op::Windowing(windowing_op), op_params)?;
            }

            Ok(result)
        }
    }
}
