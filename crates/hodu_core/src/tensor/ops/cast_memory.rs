use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{CastOp, MemoryOp, Op, OpParams},
    script::builder,
    tensor::{create_builder_tensor, from_storage_with_context, register_operation_in_builder, Tensor},
    types::{DType, Device, Layout},
    utils::valid::validate_dtype_for_device,
};

impl Tensor {
    // cast operations
    pub fn to_dtype(&self, dtype: DType) -> HoduResult<Self> {
        validate_dtype_for_device(dtype, self.device())?;

        if builder::is_builder_active() {
            let input_layout = self.layout();
            let (result_id, result_tensor) = create_builder_tensor(input_layout.clone(), false);

            let op_params = OpParams {
                dtype: Some(dtype),
                ..Default::default()
            };

            register_operation_in_builder(
                Op::Cast(CastOp::ToDType),
                Some(op_params),
                vec![self.id()],
                vec![result_id],
                vec![input_layout.clone()],
                vec![input_layout],
            )?;

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.to_dtype(&self.layout(), dtype))?;
            let layout = Layout::from_shape(&self.shape());

            let result = from_storage_with_context(storage, layout, true, false);

            Ok(result)
        }
    }

    pub fn to_device(&self, device: Device) -> HoduResult<Self> {
        validate_dtype_for_device(self.dtype(), device)?;

        if builder::is_builder_active() {
            Err(HoduError::BuilderNotActive)
        } else {
            let layout = Layout::from_shape(&self.shape());
            let storage = self.with_storage(|storage| storage.to_device(&layout, device))?;

            let result = from_storage_with_context(storage, layout, true, false);

            Ok(result)
        }
    }

    // memory operations
    pub fn contiguous(&self) -> HoduResult<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&self.shape());
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), false);

            register_operation_in_builder(
                Op::Memory(MemoryOp::Contiguous),
                None,
                vec![self.id()],
                vec![result_id],
                vec![self.layout()],
                vec![result_layout],
            )?;

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.contiguous(&self.layout()))?;
            let layout = Layout::from_shape(&self.shape());

            let result = from_storage_with_context(storage, layout, true, false);

            Ok(result)
        }
    }

    pub fn set_(&mut self, src: &Self) -> HoduResult<()> {
        // Replace storage of self with src's storage, preserving TensorId and gradient
        use crate::tensor::{with_tensor, with_tensor_mut};

        // Get src's storage (clone the Arc)
        let src_storage = with_tensor(src.id(), |t| t.storage.clone()).ok_or(HoduError::TensorNotFound(src.id()))?;

        let src_layout = src.layout().clone();

        // Update self's storage and layout, keeping TensorId and gradient
        with_tensor_mut(self.id(), |tensor_ref| {
            tensor_ref.storage = src_storage;
            tensor_ref.layout = src_layout;
        })
        .ok_or(HoduError::TensorNotFound(self.id()))?;

        Ok(())
    }
}
