use crate::{
    backends::be_hodu::{
        cpu::device::CpuDevice,
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
}

pub enum HoduDevice {
    CPU(CpuDevice),
}

impl HoduDevice {
    pub fn storage_from_flatten<T: IntoFlattened>(data: T, device: Device) -> HoduResult<HoduStorage> {
        match device {
            Device::CPU => Ok(HoduStorage::CPU(data.to_cpu_storage())),
            _ => panic!(""),
        }
    }

    pub(crate) fn zeros(layout: &Layout, device: Device, dtype: DType) -> HoduResult<HoduStorage> {
        match device {
            Device::CPU => Ok(HoduStorage::CPU(CpuDevice::zeros(layout, dtype)?)),
            _ => panic!(""),
        }
    }

    pub(crate) fn randn(layout: &Layout, device: Device, dtype: DType, mean: f64, std: f64) -> HoduResult<HoduStorage> {
        match device {
            Device::CPU => Ok(HoduStorage::CPU(CpuDevice::randn(layout, dtype, mean, std)?)),
            _ => panic!(""),
        }
    }
}
