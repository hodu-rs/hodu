#![allow(clippy::upper_case_acronyms)]

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
    #[allow(dead_code)]
    CPU(CpuDevice),
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    CUDA(crate::be_cuda::device::CudaDevice),
    #[cfg(feature = "metal")]
    #[allow(dead_code)]
    Metal(crate::be_metal::device::MetalDevice),
}

impl BackendDevice {
    pub fn storage_from_flatten<T: IntoFlattened>(data: T, device: Device) -> HoduResult<BackendStorage> {
        match device {
            Device::CPU => Ok(BackendStorage::CPU(data.to_cpu_storage())),
            #[cfg(feature = "cuda")]
            Device::CUDA(device_id) => {
                let cpu_storage = data.to_cpu_storage();
                Ok(BackendStorage::CUDA(
                    crate::be_cuda::storage::CudaStorage::from_cpu_storage(&cpu_storage, device_id)?,
                ))
            },
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
            #[cfg(feature = "cuda")]
            Device::CUDA(device_id) => {
                let cpu_storage = CpuDevice::zeros(shape, dtype)?;
                Ok(BackendStorage::CUDA(
                    crate::be_cuda::storage::CudaStorage::from_cpu_storage(&cpu_storage, device_id)?,
                ))
            },
            #[cfg(feature = "metal")]
            Device::Metal => {
                let cpu_storage = CpuDevice::zeros(shape, dtype)?;
                Ok(BackendStorage::Metal(
                    crate::be_metal::storage::MetalStorage::from_cpu_storage(&cpu_storage)?,
                ))
            },
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
            #[cfg(feature = "cuda")]
            Device::CUDA(device_id) => {
                let cpu_storage = CpuDevice::randn(shape, dtype, mean, std)?;
                Ok(BackendStorage::CUDA(
                    crate::be_cuda::storage::CudaStorage::from_cpu_storage(&cpu_storage, device_id)?,
                ))
            },
            #[cfg(feature = "metal")]
            Device::Metal => {
                let cpu_storage = CpuDevice::randn(shape, dtype, mean, std)?;
                Ok(BackendStorage::Metal(
                    crate::be_metal::storage::MetalStorage::from_cpu_storage(&cpu_storage)?,
                ))
            },
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
            #[cfg(feature = "cuda")]
            Device::CUDA(device_id) => {
                let cpu_storage = CpuDevice::rand_uniform(shape, dtype, low, high)?;
                Ok(BackendStorage::CUDA(
                    crate::be_cuda::storage::CudaStorage::from_cpu_storage(&cpu_storage, device_id)?,
                ))
            },
            #[cfg(feature = "metal")]
            Device::Metal => {
                let cpu_storage = CpuDevice::rand_uniform(shape, dtype, low, high)?;
                Ok(BackendStorage::Metal(
                    crate::be_metal::storage::MetalStorage::from_cpu_storage(&cpu_storage)?,
                ))
            },
            #[allow(unreachable_patterns)]
            _ => panic!("Unsupported device: {:?}", device),
        }
    }
}
