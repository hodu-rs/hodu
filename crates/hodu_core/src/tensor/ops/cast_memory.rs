use crate::{
    error::HoduResult,
    tensor::{from_storage, Tensor},
    types::{DType, Layout},
    utils::valid::validate_dtype_for_device,
};

impl Tensor {
    // cast operations
    pub fn to_dtype(&self, dtype: DType) -> HoduResult<Self> {
        validate_dtype_for_device(dtype, self.device())?;

        let storage = self.with_storage(|storage| storage.to_dtype(&self.layout(), dtype))?;
        let layout = Layout::from_shape(&self.shape());

        let result = from_storage(storage, layout, true, false);

        Ok(result)
    }

    // memory operations
    pub fn contiguous(&self) -> HoduResult<Self> {
        if self.is_contiguous() {
            return Ok(*self);
        }

        let storage = self.with_storage(|storage| storage.contiguous(&self.layout()))?;
        let layout = Layout::from_shape(&self.shape());

        let result = from_storage(storage, layout, true, false);

        Ok(result)
    }

    pub fn set_(&mut self, src: &Self) -> HoduResult<()> {
        self.0 = src.id();
        Ok(())
    }
}
