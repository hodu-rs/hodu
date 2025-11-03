use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{Op, WindowingOp},
    tensor::{from_storage, Tensor},
    types::{Layout, Shape},
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op},
};

impl Tensor {
    pub fn reduce_window(
        &self,
        window_shape: impl Into<Shape>,
        strides: impl Into<Shape>,
        padding: &[(u32, u32)],
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
                return Err(HoduError::InternalError(format!(
                    "invalid reduction type '{}'. Must be one of: 'max', 'mean', 'sum', 'min'",
                    reduction
                )))
            },
        };

        // Validate inputs
        if window_shape.ndim() != rank {
            return Err(HoduError::InternalError(format!(
                "window_shape length {} must match tensor rank {}",
                window_shape.ndim(),
                rank
            )));
        }
        if strides.ndim() != rank {
            return Err(HoduError::InternalError(format!(
                "strides length {} must match tensor rank {}",
                strides.ndim(),
                rank
            )));
        }
        if padding.len() != rank as usize {
            return Err(HoduError::InternalError(format!(
                "padding length {} must match tensor rank {}",
                padding.len(),
                rank
            )));
        }

        // Validate device, dtype for device, and dtype for operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Windowing(windowing_op))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Windowing(windowing_op));

        // Calculate output shape
        let input_dims = input_shape.dims();
        let window_dims = window_shape.dims();
        let stride_dims = strides.dims();
        let mut output_dims = Vec::with_capacity(rank as usize);
        for i in 0..rank as usize {
            let padded_size = input_dims[i] + padding[i].0 + padding[i].1;
            let out_size = (padded_size - window_dims[i]) / stride_dims[i] + 1;
            output_dims.push(out_size);
        }

        // Get u32 arrays (already u32)
        let window_shape_u32 = window_dims;
        let strides_u32 = stride_dims;
        // Flatten padding to [lo, hi, lo, hi, ...]
        let mut padding_u32 = Vec::with_capacity(padding.len() * 2);
        for &(lo, hi) in padding {
            padding_u32.push(lo);
            padding_u32.push(hi);
        }

        let storage = self.with_storage(|input_storage| {
            input_storage.call_reduce_window(
                &self.layout(),
                window_shape_u32,
                strides_u32,
                &padding_u32,
                Op::Windowing(windowing_op),
            )
        })?;

        let result_layout = Layout::from_shape(&Shape::from(output_dims));
        let requires_grad = self.is_requires_grad() && validate_requires_grad;
        let result = from_storage(storage, result_layout, true, requires_grad);

        Ok(result)
    }
}
