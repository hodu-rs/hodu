use crate::{
    builder,
    compat::*,
    error::{HoduError, HoduResult},
    op::{
        self,
        utils::{validate_dtype_for_device, validate_dtype_for_op},
        Op,
    },
    scalar::Scalar,
    tensor::{
        create_builder_tensor_with_grad, from_storage_with_grad, gradient, register_operation_in_builder, Tensor,
    },
    types::layout::Layout,
};

// Windowing Operations
impl Tensor {
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
        let mut params_scalars = vec![Scalar::I32(rank as i32)];
        for &ws in window_shape {
            params_scalars.push(Scalar::I32(ws as i32));
        }
        for &s in strides {
            params_scalars.push(Scalar::I32(s as i32));
        }
        for &(pad_lo, pad_hi) in padding {
            params_scalars.push(Scalar::I32(pad_lo as i32));
            params_scalars.push(Scalar::I32(pad_hi as i32));
        }
        params_scalars.push(Scalar::I32(match reduction_type {
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
                let mut params_scalars = vec![Scalar::I32(rank as i32)];
                for &ws in window_shape {
                    params_scalars.push(Scalar::I32(ws as i32));
                }
                for &s in strides {
                    params_scalars.push(Scalar::I32(s as i32));
                }
                for &(pad_lo, pad_hi) in padding {
                    params_scalars.push(Scalar::I32(pad_lo as i32));
                    params_scalars.push(Scalar::I32(pad_hi as i32));
                }
                params_scalars.push(Scalar::I32(match reduction_type {
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
}
