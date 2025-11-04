use crate::{
    error::HoduResult,
    ops::{CastOp, MemoryOp, Op},
    script::builder,
    tensor::{create_builder_tensor, from_storage, register_operation_in_builder, Tensor},
    types::{DType, Layout},
    utils::valid::validate_dtype_for_device,
};

impl Tensor {
    // cast operations
    pub fn to_dtype(&self, dtype: DType) -> HoduResult<Self> {
        validate_dtype_for_device(dtype, self.device())?;

        if builder::is_builder_active() {
            let layout = Layout::from_shape(&self.shape());
            let (result_id, result_tensor) = create_builder_tensor(layout.clone(), false);

            let mut op_params = crate::ops::OpParams::default();
            op_params.dtype = Some(dtype);

            register_operation_in_builder(
                Op::Cast(CastOp::ToDType),
                Some(op_params),
                vec![self.id()],
                vec![result_id],
                vec![self.layout()],
                vec![layout],
            )?;

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.to_dtype(&self.layout(), dtype))?;
            let layout = Layout::from_shape(&self.shape());

            let result = from_storage(storage, layout, true, false);

            Ok(result)
        }
    }

    // memory operations
    pub fn contiguous(&self) -> HoduResult<Self> {
        if self.is_contiguous() {
            return Ok(*self);
        }

        if builder::is_builder_active() {
            let layout = Layout::from_shape(&self.shape());
            let (result_id, result_tensor) = create_builder_tensor(layout.clone(), false);

            register_operation_in_builder(
                Op::Memory(MemoryOp::Contiguous),
                None,
                vec![self.id()],
                vec![result_id],
                vec![self.layout()],
                vec![layout],
            )?;

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.contiguous(&self.layout()))?;
            let layout = Layout::from_shape(&self.shape());

            let result = from_storage(storage, layout, true, false);

            Ok(result)
        }
    }

    pub fn set_(&mut self, src: &Self) -> HoduResult<()> {
        self.0 = src.id();
        Ok(())
    }
}
