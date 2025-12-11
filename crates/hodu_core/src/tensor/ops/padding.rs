use crate::{
    error::{HoduError, HoduResult},
    op_params::{OpParams, PaddingParams},
    ops::{Op, PaddingOp},
    scalar::Scalar,
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::{Layout, Shape},
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op},
};

impl Tensor {
    pub fn pad(&self, padding: &[(usize, usize)], mode: &str, value: impl Into<Scalar>) -> HoduResult<Self> {
        let input_shape = self.shape();
        let rank = input_shape.ndim();

        if padding.len() != rank {
            return Err(HoduError::InvalidLayout {
                reason: format!("padding length {} must match tensor rank {}", padding.len(), rank),
            });
        }

        let padding_op = match mode.to_lowercase().as_str() {
            "constant" => PaddingOp::PadConstant,
            "reflect" => PaddingOp::PadReflect,
            "replicate" | "edge" => PaddingOp::PadReplicate,
            "circular" | "wrap" => PaddingOp::PadCircular,
            _ => {
                return Err(HoduError::InvalidLayout {
                    reason: format!(
                        "invalid padding mode '{}'. Must be one of: 'constant', 'reflect', 'replicate', 'edge', 'circular', 'wrap'",
                        mode
                    ),
                })
            },
        };

        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Padding(padding_op))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Padding(padding_op));

        let input_dims = input_shape.dims();
        let mut output_dims = Vec::with_capacity(rank);
        let mut pad_before = Vec::with_capacity(rank);
        let mut pad_after = Vec::with_capacity(rank);

        for i in 0..rank {
            let (pb, pa) = padding[i];
            pad_before.push(pb);
            pad_after.push(pa);
            output_dims.push(input_dims[i] + pb + pa);
        }

        let pad_value: Scalar = value.into();

        let op_params = OpParams::Padding(PaddingParams {
            padding: padding.to_vec(),
            pad_value,
        });

        let input_layout = self.layout();
        let result_layout = Layout::from_shape(&Shape::from(output_dims));
        let requires_grad = self.is_requires_grad() && validate_requires_grad;

        if crate::snapshot::capture::is_active() {
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                Op::Padding(padding_op),
                Some(op_params.clone()),
                vec![self.id()],
                result_id,
                vec![input_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], result_id, Op::Padding(padding_op), op_params)?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                input_storage.call_ops_pad(
                    &self.layout(),
                    &pad_before,
                    &pad_after,
                    pad_value,
                    Op::Padding(padding_op),
                )
            })?;

            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(vec![self.id()], result.id(), Op::Padding(padding_op), op_params)?;
            }

            Ok(result)
        }
    }

    pub fn pad_constant(&self, padding: &[(usize, usize)], value: impl Into<Scalar>) -> HoduResult<Self> {
        self.pad(padding, "constant", value)
    }

    pub fn pad_reflect(&self, padding: &[(usize, usize)]) -> HoduResult<Self> {
        self.pad(padding, "reflect", 0.0f32)
    }

    pub fn pad_replicate(&self, padding: &[(usize, usize)]) -> HoduResult<Self> {
        self.pad(padding, "replicate", 0.0f32)
    }

    pub fn pad_edge(&self, padding: &[(usize, usize)]) -> HoduResult<Self> {
        self.pad(padding, "replicate", 0.0f32)
    }

    pub fn pad_circular(&self, padding: &[(usize, usize)]) -> HoduResult<Self> {
        self.pad(padding, "circular", 0.0f32)
    }

    pub fn pad_wrap(&self, padding: &[(usize, usize)]) -> HoduResult<Self> {
        self.pad(padding, "circular", 0.0f32)
    }
}
