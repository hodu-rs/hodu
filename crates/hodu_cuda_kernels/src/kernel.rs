use crate::{cuda::*, error::CudaKernelError, source::Source};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

/// RwLock compatibility wrapper
struct RwLockGuard<T>(T);

impl<T> core::ops::Deref for RwLockGuard<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> core::ops::DerefMut for RwLockGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Extension trait for RwLock to provide Result-returning API
trait RwLockExt<T> {
    fn read_compat(&self) -> Result<RwLockGuard<impl core::ops::Deref<Target = T> + '_>, String>;
    fn write_compat(&self) -> Result<RwLockGuard<impl core::ops::DerefMut<Target = T> + '_>, String>;
}

impl<T> RwLockExt<T> for RwLock<T> {
    fn read_compat(&self) -> Result<RwLockGuard<impl core::ops::Deref<Target = T> + '_>, String> {
        self.read().map(RwLockGuard).map_err(|e| format!("{:?}", e))
    }

    fn write_compat(&self) -> Result<RwLockGuard<impl core::ops::DerefMut<Target = T> + '_>, String> {
        self.write().map(RwLockGuard).map_err(|e| format!("{:?}", e))
    }
}

#[derive(Debug, Clone)]
pub enum KernelName {
    Ref(&'static str),
    Value(String),
}

impl AsRef<str> for KernelName {
    fn as_ref(&self) -> &str {
        match self {
            Self::Ref(r) => r,
            Self::Value(v) => v.as_str(),
        }
    }
}

impl Hash for KernelName {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Ref(r) => r.hash(state),
            Self::Value(v) => v.hash(state),
        }
    }
}

impl PartialEq for KernelName {
    fn eq(&self, other: &Self) -> bool {
        let v1: &str = self.as_ref();
        let v2: &str = other.as_ref();
        v1 == v2
    }
}

impl Eq for KernelName {}

impl PartialOrd for KernelName {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KernelName {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        let v1: &str = self.as_ref();
        let v2: &str = other.as_ref();
        v1.cmp(v2)
    }
}

impl From<&'static str> for KernelName {
    fn from(value: &'static str) -> Self {
        Self::Ref(value)
    }
}

impl From<String> for KernelName {
    fn from(value: String) -> Self {
        Self::Value(value)
    }
}

type Ptxs = HashMap<Source, Ptx>;
type Modules = HashMap<Source, Arc<CudaModule>>;
type Functions = HashMap<(Source, KernelName), CudaFunction>;

#[derive(Debug)]
pub struct Kernels {
    ptxs: RwLock<Ptxs>,
    modules: RwLock<Modules>,
    functions: RwLock<Functions>,
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels {
    /// Create a new Kernels instance for managing CUDA kernel compilation and caching
    ///
    /// Each instance maintains its own cache of compiled PTX, modules, and functions.
    /// For multi-device scenarios, create separate Kernels instances per device.
    pub fn new() -> Self {
        let ptxs = RwLock::new(Ptxs::new());
        let modules = RwLock::new(Modules::new());
        let functions = RwLock::new(Functions::new());
        Self {
            ptxs,
            modules,
            functions,
        }
    }

    fn get_source_code(&self, source: Source) -> &'static str {
        match source {
            Source::OpsBinary => crate::source::get_ops_binary(),
            Source::OpsCast => crate::source::get_ops_cast(),
            Source::OpsConcatSplit => crate::source::get_ops_concat_split(),
            Source::OpsConv => crate::source::get_ops_conv(),
            Source::OpsEinsum => crate::source::get_ops_einsum(),
            Source::OpsIndexing => crate::source::get_ops_indexing(),
            Source::OpsMatrix => crate::source::get_ops_matrix(),
            Source::OpsMemory => crate::source::get_ops_memory(),
            Source::OpsPadding => crate::source::get_ops_padding(),
            Source::OpsReduce => crate::source::get_ops_reduce(),
            Source::OpsResize => crate::source::get_ops_resize(),
            Source::OpsScan => crate::source::get_ops_scan(),
            Source::OpsShapeMemory => crate::source::get_ops_shape_memory(),
            Source::OpsUnary => crate::source::get_ops_unary(),
            Source::OpsWindowing => crate::source::get_ops_windowing(),
            Source::Storage => crate::source::get_storage(),
        }
    }

    pub fn load_ptx(&self, source: Source) -> Result<Ptx, CudaKernelError> {
        let mut ptxs = self.ptxs.write_compat().map_err(CudaKernelError::Message)?;

        if let Some(ptx) = ptxs.get(&source) {
            return Ok(ptx.clone());
        }

        // Get pre-compiled PTX from source
        let ptx_str = self.get_source_code(source);
        let ptx = Ptx::from_src(ptx_str);

        ptxs.insert(source, ptx.clone());
        Ok(ptx)
    }

    pub fn load_function(
        &self,
        context: &Arc<CudaContext>,
        source: Source,
        name: impl Into<KernelName>,
    ) -> Result<CudaFunction, CudaKernelError> {
        let name = name.into();
        let key = (source, name.clone());

        {
            let functions = self.functions.read_compat().map_err(CudaKernelError::Message)?;
            if let Some(func) = functions.get(&key) {
                return Ok(func.clone());
            }
        }

        // Load or get module
        let module = {
            let modules = self.modules.read_compat().map_err(CudaKernelError::Message)?;
            if let Some(module) = modules.get(&source) {
                module.clone()
            } else {
                drop(modules);
                let ptx = self.load_ptx(source)?;
                let module = context
                    .load_module(ptx)
                    .map_err(|e| CudaKernelError::LaunchError(format!("Failed to load module: {:?}", e)))?;

                let mut modules = self.modules.write_compat().map_err(CudaKernelError::Message)?;
                modules.insert(source, module.clone());
                module
            }
        };

        // Load function from module
        let func = module
            .load_function(name.as_ref())
            .map_err(|e| CudaKernelError::InvalidKernel(format!("Failed to load function: {:?}", e)))?;

        let mut functions = self.functions.write_compat().map_err(CudaKernelError::Message)?;
        functions.insert(key, func.clone());

        Ok(func)
    }
}
