use crate::compat::*;

// Re-export cudarc types
pub use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};
pub use cudarc::nvrtc::Ptx;

// Extension trait for launching kernels with backwards compatibility
pub trait CudaFunctionExt {
    unsafe fn launch<'a, Args>(
        &'a self,
        stream: &'a Arc<CudaStream>,
        cfg: LaunchConfig,
        args: Args,
    ) -> Result<(), cudarc::driver::DriverError>
    where
        Args: FnOnce(&mut cudarc::driver::LaunchArgs<'a>);
}

impl CudaFunctionExt for CudaFunction {
    unsafe fn launch<'a, Args>(
        &'a self,
        stream: &'a Arc<CudaStream>,
        cfg: LaunchConfig,
        args: Args,
    ) -> Result<(), cudarc::driver::DriverError>
    where
        Args: FnOnce(&mut cudarc::driver::LaunchArgs<'a>),
    {
        let mut launch_args = stream.launch_builder(self);
        args(&mut launch_args);
        launch_args.launch(cfg).map(|_| ())
    }
}
