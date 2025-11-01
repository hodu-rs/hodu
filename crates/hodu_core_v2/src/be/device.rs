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
    #[cfg(feature = "metal")]
    Metal(crate::be_metal::device::MetalDevice),
}

impl BackendDevice {
    pub fn storage_from_flatten<T: IntoFlattened>(data: T, device: Device) -> HoduResult<BackendStorage> {
        match device {
            Device::CPU => Ok(BackendStorage::CPU(data.to_cpu_storage())),
            #[cfg(feature = "metal")]
            Device::Metal => {
                let cpu_storage = data.to_cpu_storage();
                Ok(BackendStorage::Metal(
                    crate::be_metal::storage::MetalStorage::from_cpu_storage(&cpu_storage)?,
                ))
            },
            #[allow(unreachable_patterns)]
            _ => panic!("Unsupported device: {:?}", device),
        }
    }

    pub(crate) fn zeros(shape: &Shape, device: Device, dtype: DType) -> HoduResult<BackendStorage> {
        match device {
            Device::CPU => Ok(BackendStorage::CPU(CpuDevice::zeros(shape, dtype)?)),
            #[cfg(feature = "metal")]
            Device::Metal => Ok(BackendStorage::Metal(crate::be_metal::device::MetalDevice::zeros(
                shape, dtype,
            )?)),
            #[allow(unreachable_patterns)]
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
            #[cfg(feature = "metal")]
            Device::Metal => Ok(BackendStorage::Metal(crate::be_metal::device::MetalDevice::randn(
                shape, dtype, mean, std,
            )?)),
            #[allow(unreachable_patterns)]
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
            #[cfg(feature = "metal")]
            Device::Metal => Ok(BackendStorage::Metal(
                crate::be_metal::device::MetalDevice::rand_uniform(shape, dtype, low, high)?,
            )),
            #[allow(unreachable_patterns)]
            _ => panic!("Unsupported device: {:?}", device),
        }
    }
}
