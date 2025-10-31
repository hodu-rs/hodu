use crate::{
    be::storage::BackendStorageT,
    error::HoduResult,
    types::{DType, Shape},
};

pub trait BackendDeviceT: Sized {
    type BackendStorage: BackendStorageT;

    fn zeros(_: &Shape, _: DType) -> HoduResult<Self::BackendStorage>;

    fn randn(_: &Shape, _: DType, _: f32, _: f32) -> HoduResult<Self::BackendStorage>;

    fn rand_uniform(_: &Shape, _: DType, _: f32, _: f32) -> HoduResult<Self::BackendStorage>;
}
