use crate::{
    builder,
    compat::*,
    error::HoduResult,
    op::{self, utils::validate_dtype_for_device, Op},
    tensor::{create_builder_tensor_with_grad, from_storage_with_grad, register_operation_in_builder, Tensor},
    types::{dtype::DType, layout::Layout},
};

impl Tensor {
    // Cast Operations
    pub fn to_dtype(&self, dtype: DType) -> HoduResult<Self> {
        // Validate that target dtype is supported on current device
        validate_dtype_for_device(dtype, &self.get_device(), "to_dtype")?;

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(self.get_layout().get_shape());
            let requires_grad = self.is_requires_grad() && dtype.is_float();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let op = Op::Cast(op::CastOp::ToDType, self.id());
            register_operation_in_builder(
                op,
                vec![result_id],
                vec![self.get_layout().clone()],
                vec![result_layout],
            );

            Ok(result_tensor)
        } else {
            let layout = Layout::from_shape(self.get_layout().get_shape());
            let storage = self.with_storage(|storage| storage.to_dtype(dtype, &self.get_layout()))?;
            let requires_grad = self.is_requires_grad() && dtype.is_float();
            let result = from_storage_with_grad(storage, layout, true, requires_grad);

            Ok(result)
        }
    }

    // Memory Operations
    pub fn contiguous(&self) -> HoduResult<Self> {
        let layout = self.get_layout();

        // If already contiguous, return self
        if layout.is_contiguous() {
            return Ok(*self);
        }

        if builder::is_builder_active() {
            let contiguous_layout = Layout::from_shape(layout.get_shape());
            let requires_grad = self.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(contiguous_layout.clone(), requires_grad);

            let op = Op::Memory(op::MemoryOp::Contiguous, self.id());
            register_operation_in_builder(op, vec![result_id], vec![layout], vec![contiguous_layout]);

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.contiguous(&layout))?;
            let contiguous_layout = Layout::from_shape(layout.get_shape());
            let requires_grad = self.is_requires_grad();
            let result = from_storage_with_grad(storage, contiguous_layout, true, requires_grad);

            Ok(result)
        }
    }

    pub fn set_(&mut self, src: &Tensor) -> HoduResult<()> {
        self.0 = src.id();
        Ok(())
    }
}
