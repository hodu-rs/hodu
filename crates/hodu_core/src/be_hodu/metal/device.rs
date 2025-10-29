use crate::{
    be_hodu::{
        cpu::storage::CpuStorage,
        device::HoduDeviceT,
        metal::{error::MetalError, storage::MetalStorage},
        storage::HoduStorageT,
    },
    error::{HoduError, HoduResult},
    types::{dtype::DType, layout::Layout},
};
use hodu_metal_kernels::{
    kernel::Kernels,
    metal::{Buffer, BufferMap, CommandBuffer, Commands, ComputePipeline, Device, MTLResourceOptions},
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

pub const RESOURCE_OPTIONS: MTLResourceOptions =
    objc2_metal::MTLResourceOptions(MTLResourceOptions::StorageModeShared.bits());

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

    pub fn compile(&self, func_name: &'static str, kernel: ug::lang::ssa::Kernel) -> HoduResult<ComputePipeline> {
        let mut buf = vec![];
        ug_metal::code_gen::gen(&mut buf, func_name, &kernel)?;
        let metal_code = String::from_utf8(buf)?;
        let lib = self
            .device
            .new_library_with_source(&metal_code, None)
            .map_err(MetalError::from)?;
        let func = lib.get_function(func_name, None).map_err(MetalError::from)?;
        let pl = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(MetalError::from)?;
        Ok(pl)
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn metal_device(&self) -> &Device {
        &self.device
    }

    fn drop_unused_buffers(&self) -> HoduResult<()> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
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
        let mut commands = self.commands.write().map_err(MetalError::from)?;
        let (flushed, command_buffer) = commands.command_buffer().map_err(MetalError::from)?;
        if flushed {
            self.drop_unused_buffers()?
        }
        Ok(command_buffer.clone())
    }

    pub fn wait_until_completed(&self) -> HoduResult<()> {
        let mut commands = self.commands.write().map_err(MetalError::from)?;
        commands.wait_until_completed().map_err(MetalError::from)?;
        Ok(())
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn new_buffer(&self, element_count: usize, dtype: DType, _name: &str) -> HoduResult<Arc<Buffer>> {
        let size = element_count * dtype.get_size_in_bytes();
        self.allocate_buffer(size)
    }

    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> HoduResult<Arc<Buffer>> {
        let size = core::mem::size_of_val(data);
        let new_buffer = self
            .device
            .new_buffer_with_data(data.as_ptr().cast(), size, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;

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
            CpuStorage::U16(data) => self.new_buffer_with_data(data),
            CpuStorage::U32(data) => self.new_buffer_with_data(data),
            CpuStorage::U64(data) => self.new_buffer_with_data(data),
            CpuStorage::I8(data) => self.new_buffer_with_data(data),
            CpuStorage::I16(data) => self.new_buffer_with_data(data),
            CpuStorage::I32(data) => self.new_buffer_with_data(data),
            CpuStorage::I64(data) => self.new_buffer_with_data(data),
            _ => Err(HoduError::InternalError(format!(
                "Unsupported dtype for Metal buffer: {:?}",
                cpu_storage.get_dtype()
            ))),
        }
    }

    pub fn allocate_buffer(&self, size: usize) -> HoduResult<Arc<Buffer>> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        if let Some(b) = find_available_buffer(size, &buffers) {
            // Cloning also ensures we increment the strong count
            return Ok(b.clone());
        }
        let size = buf_size(size);
        let subbuffers = buffers.entry(size).or_insert(vec![]);

        let new_buffer = self
            .device
            .new_buffer(size, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
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

impl HoduDeviceT for MetalDevice {
    type HoduStorage = MetalStorage;

    fn zeros(_: &Layout, _: DType) -> HoduResult<MetalStorage> {
        Err(HoduError::InternalError(
            "zeros not supported on Metal Device".to_string(),
        ))
    }

    fn randn(_: &Layout, _: DType, _: f64, _: f64) -> HoduResult<Self::HoduStorage> {
        Err(HoduError::InternalError(
            "randn not supported on Metal Device".to_string(),
        ))
    }

    fn rand_uniform(_: &Layout, _: DType, _: f64, _: f64) -> HoduResult<Self::HoduStorage> {
        Err(HoduError::InternalError(
            "randn uniform not supported on Metal Device".to_string(),
        ))
    }
}
