use crate::{
    be::device::BackendDeviceT,
    be_cuda::storage::CudaStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    types::{DType, Shape},
};
use hodu_cuda_kernels::{
    cuda::{CudaContext, CudaSlice},
    kernel::Kernels,
};

#[derive(Clone)]
pub struct CudaDevice {
    pub(crate) cuda_device_id: usize,
    pub(crate) context: Arc<CudaContext>,
    pub(crate) kernels: Arc<Kernels>,
}

// Global device pool: maps CUDA device ID -> CudaDevice
static CUDA_DEVICES: LazyLock<RwLock<HashMap<usize, Arc<CudaDevice>>>> = LazyLock::new(|| RwLock::new(HashMap::new()));

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({})", self.cuda_device_id)
    }
}

impl CudaDevice {
    /// Get or create a CUDA device for the given device ID
    pub fn get(cuda_device_id: usize) -> HoduResult<Arc<CudaDevice>> {
        // Try to get existing device first (read lock)
        {
            let devices = CUDA_DEVICES.read()?;
            if let Some(device) = devices.get(&cuda_device_id) {
                return Ok(device.clone());
            }
        }

        // Need to create new device (write lock)
        let mut devices = CUDA_DEVICES.write()?;

        // Double-check in case another thread created it
        if let Some(device) = devices.get(&cuda_device_id) {
            return Ok(device.clone());
        }

        // Create new device
        let context = CudaContext::new(cuda_device_id).map_err(|e| {
            HoduError::BackendError(format!(
                "Failed to create CUDA context for device {}: {:?}",
                cuda_device_id, e
            ))
        })?;
        let kernels = Kernels::new();

        let device = Arc::new(CudaDevice {
            cuda_device_id,
            context,
            kernels: Arc::new(kernels),
        });

        devices.insert(cuda_device_id, device.clone());
        Ok(device)
    }

    /// Returns a reference to the default CUDA device (device 0)
    pub fn global() -> HoduResult<Arc<CudaDevice>> {
        Self::get(0)
    }

    pub fn device_id(&self) -> usize {
        self.cuda_device_id
    }

    pub fn synchronize(&self) -> HoduResult<()> {
        self.context
            .synchronize()
            .map_err(|e| HoduError::BackendError(format!("CUDA synchronize failed: {:?}", e)))?;
        Ok(())
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub fn new_buffer<T>(&self, element_count: usize) -> HoduResult<CudaSlice<T>>
    where
        T: hodu_cuda_kernels::cuda::DeviceRepr,
    {
        let stream = self.context.default_stream();
        unsafe {
            stream
                .alloc(element_count)
                .map_err(|e| HoduError::BackendError(format!("CUDA alloc failed: {:?}", e)))
        }
    }

    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> HoduResult<CudaSlice<T>>
    where
        T: hodu_cuda_kernels::cuda::DeviceRepr + Clone,
    {
        let stream = self.context.default_stream();
        stream
            .memcpy_stod(data)
            .map_err(|e| HoduError::BackendError(format!("CUDA memcpy_stod failed: {:?}", e)))
    }
}

impl BackendDeviceT for CudaDevice {
    type BackendStorage = CudaStorage;

    fn zeros(_: &Shape, _: DType) -> HoduResult<CudaStorage> {
        Err(HoduError::NotImplemented("zeros on CUDA device".to_string()))
    }

    fn randn(_: &Shape, _: DType, _: f32, _: f32) -> HoduResult<Self::BackendStorage> {
        Err(HoduError::NotImplemented("randn on CUDA device".to_string()))
    }

    fn rand_uniform(_: &Shape, _: DType, _: f32, _: f32) -> HoduResult<Self::BackendStorage> {
        Err(HoduError::NotImplemented("rand_uniform on CUDA device".to_string()))
    }
}
