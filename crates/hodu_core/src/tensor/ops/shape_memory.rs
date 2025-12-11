use crate::{
    error::HoduResult,
    op_params::{FlipParams, OpParams},
    ops::{Op, ShapeMemoryOp},
    scalar::Scalar,
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::Layout,
};

impl Tensor {
    pub fn flip<D: Into<Scalar> + Copy>(&self, dims: &[D]) -> HoduResult<Self> {
        let ndim = self.ndim();
        let mut dims_usize = Vec::with_capacity(dims.len());

        for &d in dims {
            let dim_scalar = d.into();
            let dim_i32 = dim_scalar.to_i32();
            let dim_usize = if dim_i32 < 0 {
                (ndim as i32 + dim_i32) as usize
            } else {
                dim_i32 as usize
            };
            dims_usize.push(dim_usize);
        }

        let op_params = OpParams::Flip(FlipParams {
            dims: dims_usize.clone(),
        });

        let input_layout = self.layout();
        let result_layout = Layout::from_shape(&self.shape());
        let requires_grad = self.is_requires_grad();

        if crate::snapshot::capture::is_active() {
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                Op::ShapeMemory(ShapeMemoryOp::Flip),
                Some(op_params.clone()),
                vec![self.id()],
                result_id,
                vec![input_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result_id,
                    Op::ShapeMemory(ShapeMemoryOp::Flip),
                    op_params,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage =
                self.with_storage(|input_storage| input_storage.call_ops_flip(&self.layout(), &dims_usize))?;

            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result.id(),
                    Op::ShapeMemory(ShapeMemoryOp::Flip),
                    op_params,
                )?;
            }

            Ok(result)
        }
    }
}
