use crate::{
    be::device::BackendDevice,
    error::HoduResult,
    into::faltten::IntoFlattened,
    layer::compat::*,
    scalar::Scalar,
    script::builder::is_builder_active,
    tensor::{from_storage, Tensor},
    types::{DType, Device, Layout, Shape},
};

// Use AtomicU8 for lock-free access to runtime device
// Device encoding: CPU=0, CUDA=1, Metal=2
static RUNTIME_DEVICE: AtomicU8 = AtomicU8::new(0); // Default: CPU

#[inline]
pub fn get_runtime_device() -> Device {
    let device_id = RUNTIME_DEVICE.load(Ordering::Relaxed);
    match device_id {
        0 => Device::CPU,
        #[cfg(feature = "cuda")]
        1 => Device::CUDA(0), // Default CUDA device 0
        #[cfg(feature = "metal")]
        2 => Device::Metal,
        _ => Device::CPU, // Fallback
    }
}

#[inline]
pub fn set_runtime_device(device: Device) {
    let device_id = match device {
        Device::CPU => 0,
        #[cfg(feature = "cuda")]
        Device::CUDA(_) => 1, // Store as CUDA (device index not preserved for runtime default)
        #[cfg(feature = "metal")]
        Device::Metal => 2,
    };
    RUNTIME_DEVICE.store(device_id, Ordering::Relaxed);
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
        let shape = Shape::from(&data.get_shape());
        let layout = Layout::from_shape(&shape);
        let storage = BackendDevice::storage_from_flatten(data, device)?;
        Ok(from_storage(storage, layout, !is_builder_active(), false))
    }

    pub fn zeros(shape: impl Into<Shape>, dtype: DType) -> HoduResult<Self> {
        let shape = shape.into();
        let device = if is_builder_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let storage = BackendDevice::zeros(&shape, device, dtype)?;
        let layout = Layout::from_shape(&shape);
        Ok(from_storage(storage, layout, !is_builder_active(), false))
    }

    pub fn zeros_like(tensor: &Self) -> HoduResult<Self> {
        let shape = tensor.shape();
        Self::zeros(&shape, tensor.dtype())
    }

    pub fn ones(shape: impl Into<Shape>, dtype: DType) -> HoduResult<Self> {
        let shape = shape.into();
        let device = if is_builder_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let layout = Layout::from_shape(&shape);
        let mut storage = BackendDevice::zeros(&shape, device, dtype)?;
        storage.const_set(Scalar::one(dtype), &layout)?;
        Ok(from_storage(storage, layout, !is_builder_active(), false))
    }

    pub fn ones_like(tensor: &Self) -> HoduResult<Self> {
        let shape = tensor.shape();
        Self::ones(&shape, tensor.dtype())
    }

    pub fn full<T: Into<Scalar>>(shape: impl Into<Shape>, value: T) -> HoduResult<Self> {
        let shape = shape.into();
        let value = value.into();
        let device = if is_builder_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let mut storage = BackendDevice::zeros(&shape, device, value.dtype())?;
        let layout = Layout::from_shape(&shape);
        storage.const_set(value, &layout)?;
        Ok(from_storage(storage, layout, !is_builder_active(), false))
    }

    pub fn full_like<T: Into<Scalar>>(tensor: &Self, value: T) -> HoduResult<Self> {
        let shape = tensor.shape();
        Self::full(&shape, value)
    }

    pub fn randn<T: Into<Scalar>>(shape: impl Into<Shape>, mean: T, std: T) -> HoduResult<Self> {
        let shape = shape.into();
        let mean = mean.into();
        let std = std.into();
        let device = if is_builder_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let dtype = if mean.is_float() {
            mean.dtype()
        } else if std.is_float() {
            std.dtype()
        } else {
            DType::F32
        };
        let storage = BackendDevice::randn(&shape, device, dtype, mean.to_f32(), std.to_f32())?;
        let layout = Layout::from_shape(&shape);
        Ok(from_storage(storage, layout, !is_builder_active(), false))
    }

    pub fn randn_like<T: Into<Scalar>>(tensor: &Self, mean: T, std: T) -> HoduResult<Self> {
        let shape = tensor.shape();
        Self::randn(&shape, mean, std)
    }

    pub fn rand_uniform<T: Into<Scalar>>(shape: impl Into<Shape>, low: T, high: T) -> HoduResult<Self> {
        let shape = shape.into();
        let low = low.into();
        let high = high.into();
        let device = if is_builder_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let dtype = if low.is_float() {
            low.dtype()
        } else if high.is_float() {
            high.dtype()
        } else {
            DType::F32
        };
        let storage = BackendDevice::rand_uniform(&shape, device, dtype, low.to_f32(), high.to_f32())?;
        let layout = Layout::from_shape(&shape);
        Ok(from_storage(storage, layout, !is_builder_active(), false))
    }

    pub fn rand_uniform_like<T: Into<Scalar>>(tensor: &Self, low: T, high: T) -> HoduResult<Self> {
        let shape = tensor.shape();
        Self::rand_uniform(&shape, low, high)
    }
}
