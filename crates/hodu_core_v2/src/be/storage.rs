use crate::{
    be::device::BackendDeviceT,
    be_cpu::storage::CpuStorage,
    error::HoduResult,
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
}
