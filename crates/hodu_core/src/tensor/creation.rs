use crate::{
    backends::{be_hodu::device::HoduDevice, builder::is_builder_active},
    compat::*,
    error::HoduResult,
    flatten::IntoFlattened,
    scalar::Scalar,
    tensor::{from_storage, Tensor},
    types::{device::Device, dtype::DType, layout::Layout},
};

static RUNTIME_DEVICE: Mutex<Device> = Mutex::new(Device::CPU);

pub fn get_runtime_device() -> Device {
    #[cfg(feature = "std")]
    {
        *RUNTIME_DEVICE.lock().unwrap()
    }
    #[cfg(not(feature = "std"))]
    {
        *RUNTIME_DEVICE.lock()
    }
}

pub fn set_runtime_device(device: Device) {
    #[cfg(feature = "std")]
    {
        *RUNTIME_DEVICE.lock().unwrap() = device;
    }
    #[cfg(not(feature = "std"))]
    {
        *RUNTIME_DEVICE.lock() = device;
    }
}

impl Tensor {
    pub fn new<T>(data: T) -> HoduResult<Self>
    where
        T: IntoFlattened,
    {
        let device = if is_builder_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let layout = Layout::from_shape(&data.get_shape());
        let storage = HoduDevice::storage_from_flatten(data, device)?;
        Ok(from_storage(storage, layout, !is_builder_active()))
    }

    pub fn zeros(shape: &[usize], dtype: DType) -> HoduResult<Self> {
        let device = if is_builder_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let layout = Layout::from_shape(shape);
        let storage = HoduDevice::zeros(&layout, device, dtype)?;
        Ok(from_storage(storage, layout, !is_builder_active()))
    }

    pub fn zeros_like(tensor: &Tensor) -> HoduResult<Self> {
        let layout = tensor.get_layout();
        let shape = layout.get_shape();
        Self::zeros(shape, tensor.get_dtype())
    }

    pub fn ones(shape: &[usize], dtype: DType) -> HoduResult<Self> {
        let device = if is_builder_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let layout = Layout::from_shape(shape);
        let mut storage = HoduDevice::zeros(&layout, device, dtype)?;
        storage.const_set(Scalar::one(dtype), &layout)?;
        Ok(from_storage(storage, layout, !is_builder_active()))
    }

    pub fn ones_like(tensor: &Tensor) -> HoduResult<Self> {
        let layout = tensor.get_layout();
        let shape = layout.get_shape();
        Self::ones(shape, tensor.get_dtype())
    }

    pub fn full<T: Into<Scalar>>(shape: &[usize], value: T) -> HoduResult<Self> {
        let value = value.into();
        let device = if is_builder_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let layout = Layout::from_shape(shape);
        let mut storage = HoduDevice::zeros(&layout, device, value.get_dtype())?;
        storage.const_set(value, &layout)?;
        Ok(from_storage(storage, layout, !is_builder_active()))
    }

    pub fn full_like(tensor: &Tensor, value: Scalar) -> HoduResult<Self> {
        let layout = tensor.get_layout();
        let shape = layout.get_shape();
        Self::full(shape, value)
    }

    pub fn randn<T: Into<Scalar>>(shape: &[usize], mean: T, std: T) -> HoduResult<Self> {
        let mean = mean.into();
        let std = std.into();
        let device = if is_builder_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let dtype = if mean.is_float() {
            mean.get_dtype()
        } else {
            std.get_dtype()
        };
        let layout = Layout::from_shape(shape);
        let storage = HoduDevice::randn(&layout, device, dtype, mean.to_f64(), std.to_f64())?;
        Ok(from_storage(storage, layout, !is_builder_active()))
    }

    pub fn randn_like<T: Into<Scalar>>(tensor: &Tensor, mean: T, std: T) -> HoduResult<Self> {
        let layout = tensor.get_layout();
        let shape = layout.get_shape();
        Self::randn(shape, mean, std)
    }

    pub fn rand_uniform<T: Into<Scalar>>(shape: &[usize], low: T, high: T) -> HoduResult<Self> {
        let low = low.into();
        let high = high.into();
        let device = if is_builder_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let dtype = if low.is_float() {
            low.get_dtype()
        } else {
            high.get_dtype()
        };
        let layout = Layout::from_shape(shape);
        let storage = HoduDevice::rand_uniform(&layout, device, dtype, low.to_f64(), high.to_f64())?;
        Ok(from_storage(storage, layout, !is_builder_active()))
    }

    pub fn rand_uniform_like<T: Into<Scalar>>(tensor: &Tensor, low: T, high: T) -> HoduResult<Self> {
        let layout = tensor.get_layout();
        let shape = layout.get_shape();
        Self::rand_uniform(shape, low, high)
    }
}
