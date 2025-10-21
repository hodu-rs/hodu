use crate::{
    backends::be_hodu::{
        cpu::device::CpuDevice,
        metal::{device::MetalDevice, storage::MetalStorage},
        storage::{HoduStorage, HoduStorageT},
    },
    error::HoduResult,
    flatten::IntoFlattened,
    types::{device::Device, dtype::DType, layout::Layout},
};

pub trait HoduDeviceT: Sized {
    type HoduStorage: HoduStorageT;

    fn zeros(_: &Layout, _: DType) -> HoduResult<Self::HoduStorage>;

    fn randn(_: &Layout, _: DType, _: f64, _: f64) -> HoduResult<Self::HoduStorage>;

    fn rand_uniform(_: &Layout, _: DType, _: f64, _: f64) -> HoduResult<Self::HoduStorage>;
}

pub enum HoduDevice {
    CPU(CpuDevice),
    METAL(MetalDevice),
}

impl HoduDevice {
    pub fn storage_from_flatten<T: IntoFlattened>(data: T, device: Device) -> HoduResult<HoduStorage> {
        match device {
            Device::CPU => Ok(HoduStorage::CPU(data.to_cpu_storage())),
            Device::METAL => {
                let cpu_storage = data.to_cpu_storage();
                let metal_storage = MetalStorage::from_cpu_storage(&cpu_storage)?;
                Ok(HoduStorage::METAL(metal_storage))
            },
            _ => panic!("Unsupported device: {:?}", device),
        }
    }

    pub(crate) fn zeros(layout: &Layout, device: Device, dtype: DType) -> HoduResult<HoduStorage> {
        match device {
            Device::CPU => Ok(HoduStorage::CPU(CpuDevice::zeros(layout, dtype)?)),
            Device::METAL => {
                let cpu_storage = CpuDevice::zeros(layout, dtype)?;
                let metal_storage = MetalStorage::from_cpu_storage(&cpu_storage)?;
                Ok(HoduStorage::METAL(metal_storage))
            },
            _ => panic!("Unsupported device: {:?}", device),
        }
    }

    pub(crate) fn randn(layout: &Layout, device: Device, dtype: DType, mean: f64, std: f64) -> HoduResult<HoduStorage> {
        match device {
            Device::CPU => Ok(HoduStorage::CPU(CpuDevice::randn(layout, dtype, mean, std)?)),
            Device::METAL => {
                let cpu_storage = CpuDevice::randn(layout, dtype, mean, std)?;
                let metal_storage = MetalStorage::from_cpu_storage(&cpu_storage)?;
                Ok(HoduStorage::METAL(metal_storage))
            },
            _ => panic!("Unsupported device: {:?}", device),
        }
    }

    pub(crate) fn rand_uniform(
        layout: &Layout,
        device: Device,
        dtype: DType,
        low: f64,
        high: f64,
    ) -> HoduResult<HoduStorage> {
        match device {
            Device::CPU => Ok(HoduStorage::CPU(CpuDevice::rand_uniform(layout, dtype, low, high)?)),
            Device::METAL => {
                let cpu_storage = CpuDevice::rand_uniform(layout, dtype, low, high)?;
                let metal_storage = MetalStorage::from_cpu_storage(&cpu_storage)?;
                Ok(HoduStorage::METAL(metal_storage))
            },
            _ => panic!("Unsupported device: {:?}", device),
        }
    }
}
