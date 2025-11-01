use crate::{
    be::storage::{BackendStorage, BackendStorageT},
    be_cpu::device::CpuDevice,
    error::HoduResult,
    into::faltten::IntoFlattened,
    types::{DType, Device, Shape},
};

pub trait BackendDeviceT: Sized {
    type BackendStorage: BackendStorageT;

    fn zeros(_: &Shape, _: DType) -> HoduResult<Self::BackendStorage>;

    fn randn(_: &Shape, _: DType, _: f32, _: f32) -> HoduResult<Self::BackendStorage>;

    fn rand_uniform(_: &Shape, _: DType, _: f32, _: f32) -> HoduResult<Self::BackendStorage>;
}

pub enum BackendDevice {
    CPU(CpuDevice),
}

impl BackendDevice {
    pub fn storage_from_flatten<T: IntoFlattened>(data: T, device: Device) -> HoduResult<BackendStorage> {
        match device {
            Device::CPU => Ok(BackendStorage::CPU(data.to_cpu_storage())),
            _ => panic!("Unsupported device: {:?}", device),
        }
    }

    pub(crate) fn zeros(shape: &Shape, device: Device, dtype: DType) -> HoduResult<BackendStorage> {
        match device {
            Device::CPU => Ok(BackendStorage::CPU(CpuDevice::zeros(shape, dtype)?)),
            _ => panic!("Unsupported device: {:?}", device),
        }
    }

    pub(crate) fn randn(
        shape: &Shape,
        device: Device,
        dtype: DType,
        mean: f32,
        std: f32,
    ) -> HoduResult<BackendStorage> {
        match device {
            Device::CPU => Ok(BackendStorage::CPU(CpuDevice::randn(shape, dtype, mean, std)?)),
            _ => panic!("Unsupported device: {:?}", device),
        }
    }

    pub(crate) fn rand_uniform(
        shape: &Shape,
        device: Device,
        dtype: DType,
        low: f32,
        high: f32,
    ) -> HoduResult<BackendStorage> {
        match device {
            Device::CPU => Ok(BackendStorage::CPU(CpuDevice::rand_uniform(shape, dtype, low, high)?)),
            _ => panic!("Unsupported device: {:?}", device),
        }
    }
}
