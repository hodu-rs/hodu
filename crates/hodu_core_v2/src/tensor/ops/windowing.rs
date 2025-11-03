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
        window_shape: &[usize],
        strides: &[usize],
        padding: &[(usize, usize)],
        reduction: &str,
    ) -> HoduResult<Self> {
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
        if window_shape.len() != rank as usize {
            return Err(HoduError::InternalError(format!(
                "window_shape length {} must match tensor rank {}",
                window_shape.len(),
                rank
            )));
        }
        if strides.len() != rank as usize {
            return Err(HoduError::InternalError(format!(
                "strides length {} must match tensor rank {}",
                strides.len(),
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
        let mut output_dims = Vec::with_capacity(rank as usize);
        for i in 0..rank as usize {
            let padded_size = input_dims[i] + padding[i].0 as u32 + padding[i].1 as u32;
            let out_size = (padded_size - window_shape[i] as u32) / strides[i] as u32 + 1;
            output_dims.push(out_size);
        }

        // Convert to u32 arrays
        let window_shape_u32: Vec<u32> = window_shape.iter().map(|&x| x as u32).collect();
        let strides_u32: Vec<u32> = strides.iter().map(|&x| x as u32).collect();
        // Flatten padding to [lo, hi, lo, hi, ...]
        let mut padding_u32 = Vec::with_capacity(padding.len() * 2);
        for &(lo, hi) in padding {
            padding_u32.push(lo as u32);
            padding_u32.push(hi as u32);
        }

        let storage = self.with_storage(|input_storage| {
            input_storage.call_reduce_window(
                &self.layout(),
                &window_shape_u32,
                &strides_u32,
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
