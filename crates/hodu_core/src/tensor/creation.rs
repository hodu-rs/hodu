use crate::{
    be::device::BackendDevice,
    compat::*,
    error::{HoduError, HoduResult},
    into::flatten::IntoFlattened,
    scalar::Scalar,
    tensor::{from_storage, Tensor},
    types::{DType, Device, Layout, Shape},
};

// Use AtomicUsize for lock-free access to runtime device with CUDA device ID
// Device encoding: 0 = CPU, 1-16 = CUDA(0-15), 17 = Metal
static RUNTIME_DEVICE: AtomicUsize = AtomicUsize::new(0); // Default: CPU

#[inline]
pub fn get_runtime_device() -> Device {
    let encoded = RUNTIME_DEVICE.load(Ordering::Relaxed);
    match encoded {
        0 => Device::CPU,
        #[cfg(feature = "cuda")]
        1..=16 => Device::CUDA(encoded - 1),
        #[cfg(any(feature = "metal", feature = "metal-device"))]
        17 => Device::Metal,
        _ => Device::CPU, // Fallback
    }
}

#[inline]
pub fn set_runtime_device(device: Device) {
    let encoded = match device {
        Device::CPU => 0,
        #[cfg(feature = "cuda")]
        Device::CUDA(device_id) => {
            if device_id > 15 {
                panic!("CUDA device ID must be <= 15, got {}", device_id);
            }
            device_id + 1
        },
        #[cfg(any(feature = "metal", feature = "metal-device"))]
        Device::Metal => 17,
    };
    RUNTIME_DEVICE.store(encoded, Ordering::Relaxed);
}

impl Tensor {
    pub fn new<T>(data: T) -> HoduResult<Self>
    where
        T: IntoFlattened,
    {
        let device = if crate::snapshot::capture::is_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let shape = Shape::from(&data.get_shape_vec());
        let layout = Layout::from_shape(&shape);
        let storage = BackendDevice::storage_from_flatten(data, device)?;
        Ok(from_storage(
            storage,
            layout,
            !crate::snapshot::capture::is_active(),
            false,
            None,
        ))
    }

    pub fn from_slice<T>(data: T, shape: impl Into<Shape>) -> HoduResult<Self>
    where
        T: IntoFlattened,
    {
        let device = if crate::snapshot::capture::is_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let data_shape_vec = data.get_shape_vec();
        let shape = shape.into();
        let data_shape = Shape::from(&data_shape_vec);

        if shape.size() != data_shape.size() {
            return Err(HoduError::SizeMismatch {
                expected: shape.size(),
                got: data_shape.size(),
            });
        }

        let layout = Layout::from_shape(&shape);
        let storage = BackendDevice::storage_from_flatten(data, device)?;
        Ok(from_storage(
            storage,
            layout,
            !crate::snapshot::capture::is_active(),
            false,
            None,
        ))
    }

    pub fn empty(shape: impl Into<Shape>, dtype: DType) -> HoduResult<Self> {
        let shape = shape.into();
        let device = if crate::snapshot::capture::is_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let storage = BackendDevice::allocate(shape.size(), device, dtype)?;
        let layout = Layout::from_shape(&shape);
        Ok(from_storage(
            storage,
            layout,
            !crate::snapshot::capture::is_active(),
            false,
            None,
        ))
    }

    pub fn empty_like(tensor: &Self) -> HoduResult<Self> {
        let shape = tensor.shape();
        Self::empty(&shape, tensor.dtype())
    }

    pub fn zeros(shape: impl Into<Shape>, dtype: DType) -> HoduResult<Self> {
        let shape = shape.into();
        let device = if crate::snapshot::capture::is_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let storage = BackendDevice::zeros(shape.size(), device, dtype)?;
        let layout = Layout::from_shape(&shape);
        Ok(from_storage(
            storage,
            layout,
            !crate::snapshot::capture::is_active(),
            false,
            None,
        ))
    }

    pub fn zeros_like(tensor: &Self) -> HoduResult<Self> {
        let shape = tensor.shape();
        Self::zeros(&shape, tensor.dtype())
    }

    pub fn ones(shape: impl Into<Shape>, dtype: DType) -> HoduResult<Self> {
        let shape = shape.into();
        let device = if crate::snapshot::capture::is_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let layout = Layout::from_shape(&shape);
        let mut storage = BackendDevice::zeros(shape.size(), device, dtype)?;
        storage.const_set(Scalar::one(dtype), &layout)?;
        Ok(from_storage(
            storage,
            layout,
            !crate::snapshot::capture::is_active(),
            false,
            None,
        ))
    }

    pub fn ones_like(tensor: &Self) -> HoduResult<Self> {
        let shape = tensor.shape();
        Self::ones(&shape, tensor.dtype())
    }

    pub fn full<T: Into<Scalar>>(shape: impl Into<Shape>, value: T) -> HoduResult<Self> {
        let shape = shape.into();
        let value = value.into();
        let device = if crate::snapshot::capture::is_active() {
            Device::CPU
        } else {
            get_runtime_device()
        };
        let mut storage = BackendDevice::zeros(shape.size(), device, value.dtype())?;
        let layout = Layout::from_shape(&shape);
        storage.const_set(value, &layout)?;
        Ok(from_storage(
            storage,
            layout,
            !crate::snapshot::capture::is_active(),
            false,
            None,
        ))
    }

    pub fn full_like<T: Into<Scalar>>(tensor: &Self, value: T) -> HoduResult<Self> {
        let shape = tensor.shape();
        Self::full(&shape, value)
    }

    pub fn scalar<T: Into<Scalar>>(value: T) -> HoduResult<Self> {
        Self::full(&[] as &[usize], value)
    }

    pub fn randn<T: Into<Scalar>>(shape: impl Into<Shape>, mean: T, std: T) -> HoduResult<Self> {
        let shape = shape.into();
        let mean = mean.into();
        let std = std.into();
        let device = if crate::snapshot::capture::is_active() {
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
        let storage = BackendDevice::randn(shape.size(), device, dtype, mean.to_f32(), std.to_f32())?;
        let layout = Layout::from_shape(&shape);
        Ok(from_storage(
            storage,
            layout,
            !crate::snapshot::capture::is_active(),
            false,
            None,
        ))
    }

    pub fn randn_like<T: Into<Scalar>>(tensor: &Self, mean: T, std: T) -> HoduResult<Self> {
        let shape = tensor.shape();
        Self::randn(&shape, mean, std)
    }

    pub fn rand_uniform<T: Into<Scalar>>(shape: impl Into<Shape>, low: T, high: T) -> HoduResult<Self> {
        let shape = shape.into();
        let low = low.into();
        let high = high.into();
        let device = if crate::snapshot::capture::is_active() {
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
        let storage = BackendDevice::rand_uniform(shape.size(), device, dtype, low.to_f32(), high.to_f32())?;
        let layout = Layout::from_shape(&shape);
        Ok(from_storage(
            storage,
            layout,
            !crate::snapshot::capture::is_active(),
            false,
            None,
        ))
    }

    pub fn rand_uniform_like<T: Into<Scalar>>(tensor: &Self, low: T, high: T) -> HoduResult<Self> {
        let shape = tensor.shape();
        Self::rand_uniform(&shape, low, high)
    }
}
