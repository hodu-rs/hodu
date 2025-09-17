use crate::{
    backends::builder::{get_active_builder, is_builder_active},
    error::{HoduError, HoduResult},
    tensor::{insert, Tensor, TensorId, Tensor_},
    types::layout::Layout,
};

impl Tensor {
    pub fn input(name: &'static str, shape: &[usize]) -> HoduResult<Self> {
        if !is_builder_active() {
            return Err(HoduError::StaticTensorCreationRequiresBuilderContext);
        }

        let layout = Layout::from_shape(shape);
        let tensor_ = Tensor_ {
            storage: None,
            is_runtime: false,
            layout,
            requires_grad: false,
            grad_tensor_id: None,
        };
        let tensor_id = TensorId::new();
        insert(tensor_id, tensor_);

        let tensor = Tensor(tensor_id);

        let active_builder = get_active_builder()?;
        active_builder.add_input(name, tensor)?;

        Ok(tensor)
    }
}
