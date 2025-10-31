use crate::{
    be::device::BackendDeviceT,
    be_cpu::storage::CpuStorage,
    error::HoduResult,
    ops::Op,
    scalar::Scalar,
    types::{DType, Device, Layout},
};

pub trait BackendStorageT: Sized {
    type BackendDevice: BackendDeviceT;

    fn dtype(&self) -> DType;

    fn device(&self) -> Device;

    fn backend_device(&self) -> &Self::BackendDevice;

    fn to_cpu_storage(&self) -> HoduResult<CpuStorage>;

    fn const_set(&mut self, _: Scalar, _: &Layout) -> HoduResult<()>;

    fn call_binary(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_binary_logical(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_cmp(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_cmp_scalar(&self, _: &Layout, _: Scalar, _: Op) -> HoduResult<Self>;

    fn call_unary(&self, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_unary_logical(&self, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_unary_scalar(&self, _: &Layout, _: Scalar, _: Op) -> HoduResult<Self>;
}
