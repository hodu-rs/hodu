use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::storage::CpuStorage,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    types::DType,
};
use hodu_metal_kernels::{
    kernel::Kernels,
    metal::{Buffer, BufferMap, CommandBuffer, Commands, Device},
    RESOURCE_OPTIONS,
};
use std::sync::{Arc, LazyLock, RwLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub(crate) fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone)]
pub struct MetalDevice {
    pub(crate) id: DeviceId,
    pub(crate) device: Device,
    pub(crate) commands: Arc<RwLock<Commands>>,
    pub(crate) buffers: Arc<RwLock<BufferMap>>,
    pub(crate) kernels: Arc<Kernels>,
}

// Global singleton Metal device
static METAL_DEVICE: LazyLock<MetalDevice> = LazyLock::new(|| {
    let device = Device::system_default().expect("No Metal device available");
    let command_queue = device.new_command_queue().expect("Failed to create command queue");
    let commands = Commands::new(command_queue).expect("Failed to create Commands");
    let kernels = Kernels::new();

    MetalDevice {
        id: DeviceId::new(),
        device,
        commands: Arc::new(RwLock::new(commands)),
        buffers: Arc::new(RwLock::new(BufferMap::default())),
        kernels: Arc::new(kernels),
    }
});

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.id)
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl MetalDevice {
    /// Returns a reference to the global Metal device singleton
    pub fn global() -> &'static MetalDevice {
        &METAL_DEVICE
    }

    fn drop_unused_buffers(&self) -> HoduResult<()> {
        let mut buffers = self.buffers.write()?;
        for subbuffers in buffers.values_mut() {
            let newbuffers = subbuffers
                .iter()
                .filter(|s| Arc::strong_count(*s) > 1)
                .map(Arc::clone)
                .collect();
            *subbuffers = newbuffers;
        }
        Ok(())
    }

    pub fn command_buffer(&self) -> HoduResult<CommandBuffer> {
        let mut commands = self.commands.write()?;
        let (flushed, command_buffer) = commands.command_buffer()?;
        if flushed {
            self.drop_unused_buffers()?
        }
        Ok(command_buffer.clone())
    }

    pub fn wait_until_completed(&self) -> HoduResult<()> {
        let mut commands = self.commands.write()?;
        commands.wait_until_completed()?;
        Ok(())
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn new_buffer(&self, element_count: usize, dtype: DType, _name: &str) -> HoduResult<Arc<Buffer>> {
        let size = element_count * dtype.size_in_bytes();
        self.allocate_buffer(size)
    }

    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> HoduResult<Arc<Buffer>> {
        let size = core::mem::size_of_val(data);
        let new_buffer = self
            .device
            .new_buffer_with_data(data.as_ptr().cast(), size, RESOURCE_OPTIONS)?;
        let mut buffers = self.buffers.write()?;

        let subbuffers = buffers.entry(size).or_insert(vec![]);

        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }

    pub fn new_buffer_with_cpu_storage(&self, cpu_storage: &CpuStorage) -> HoduResult<Arc<Buffer>> {
        match cpu_storage {
            CpuStorage::BOOL(data) => self.new_buffer_with_data(data),
            CpuStorage::BF16(data) => self.new_buffer_with_data(data),
            CpuStorage::F16(data) => self.new_buffer_with_data(data),
            CpuStorage::F32(data) => self.new_buffer_with_data(data),
            CpuStorage::U8(data) => self.new_buffer_with_data(data),
            #[cfg(feature = "u16")]
            CpuStorage::U16(data) => self.new_buffer_with_data(data),
            CpuStorage::U32(data) => self.new_buffer_with_data(data),
            #[cfg(feature = "u64")]
            CpuStorage::U64(data) => self.new_buffer_with_data(data),
            CpuStorage::I8(data) => self.new_buffer_with_data(data),
            #[cfg(feature = "i16")]
            CpuStorage::I16(data) => self.new_buffer_with_data(data),
            CpuStorage::I32(data) => self.new_buffer_with_data(data),
            #[cfg(feature = "i64")]
            CpuStorage::I64(data) => self.new_buffer_with_data(data),
            _ => Err(HoduError::UnsupportedDTypeForDevice {
                dtype: cpu_storage.dtype(),
                device: crate::types::Device::Metal,
            }),
        }
    }

    pub fn allocate_buffer(&self, size: usize) -> HoduResult<Arc<Buffer>> {
        let mut buffers = self.buffers.write()?;
        if let Some(b) = find_available_buffer(size, &buffers) {
            // Cloning also ensures we increment the strong count
            return Ok(b.clone());
        }
        let size = buf_size(size);
        let subbuffers = buffers.entry(size).or_insert(vec![]);

        let new_buffer = self.device.new_buffer(size, RESOURCE_OPTIONS)?;
        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }
}

fn buf_size(size: usize) -> usize {
    size.saturating_sub(1).next_power_of_two()
}

fn find_available_buffer(size: usize, buffers: &BufferMap) -> Option<Arc<Buffer>> {
    let mut best_buffer: Option<&Arc<Buffer>> = None;
    let mut best_buffer_size = usize::MAX;
    for (buffer_size, subbuffers) in buffers.iter() {
        if buffer_size >= &size && buffer_size < &best_buffer_size {
            for sub in subbuffers {
                if Arc::strong_count(sub) == 1 {
                    best_buffer = Some(sub);
                    best_buffer_size = *buffer_size;
                }
            }
        }
    }
    best_buffer.cloned()
}

impl BackendDeviceT for MetalDevice {
    type BackendStorage = MetalStorage;

    fn allocate(size: usize, dtype: DType) -> HoduResult<Self::BackendStorage> {
        let device = MetalDevice::global().clone();
        let buffer = device.new_buffer(size, dtype, "allocate")?;
        Ok(MetalStorage::new(buffer, device, size, dtype))
    }

    fn zeros(_: usize, _: DType) -> HoduResult<MetalStorage> {
        Err(HoduError::NotImplemented("zeros on Metal device".to_string()))
    }

    fn randn(_: usize, _: DType, _: f32, _: f32) -> HoduResult<Self::BackendStorage> {
        Err(HoduError::NotImplemented("randn on Metal device".to_string()))
    }

    fn rand_uniform(_: usize, _: DType, _: f32, _: f32) -> HoduResult<Self::BackendStorage> {
        Err(HoduError::NotImplemented("rand_uniform on Metal device".to_string()))
    }
}
