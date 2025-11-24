use crate::{
    compat::AtomicUsize,
    error::{HoduError, HoduResult},
    tensor::{insert, Tensor, TensorId, Tensor_},
    types::{DType, Layout, Shape},
};

impl Tensor {
    pub fn input(name: &'static str, shape: impl Into<Shape>, dtype: DType) -> HoduResult<Self> {
        let shape = shape.into();
        if !crate::script::capture::is_active() {
            return Err(HoduError::BuilderNotActive);
        }

        let layout = Layout::from_shape(&shape);
        let tensor_ = Tensor_ {
            storage: None,
            layout,
            dtype: Some(dtype),
            requires_grad: false,
            grad_tensor_id: None,
            is_runtime: false,
            is_gradient: false,
            owner_context: None, // Builder input tensors are user-created
            ref_count: AtomicUsize::new(1),
        };
        let tensor_id = TensorId::new();
        insert(tensor_id, tensor_);

        let tensor = Tensor(tensor_id);

        crate::script::capture::add_input_to_active(name, tensor.clone())?;

        Ok(tensor)
    }
}
