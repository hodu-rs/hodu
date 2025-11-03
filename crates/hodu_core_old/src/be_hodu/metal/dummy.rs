use crate::{
    be_hodu::{cpu::storage::CpuStorage, device::HoduDeviceT, storage::HoduStorageT},
    compat::*,
    error::HoduResult,
    scalar::Scalar,
    types::{device::Device, dtype::DType, layout::Layout},
};

pub mod device {
    use super::{storage::MetalStorage, *};

    #[derive(Clone)]
    pub struct MetalDevice;

    impl HoduDeviceT for MetalDevice {
        type HoduStorage = MetalStorage;

        fn zeros(_: &Layout, _: DType) -> HoduResult<MetalStorage> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn randn(_: &Layout, _: DType, _: f64, _: f64) -> HoduResult<Self::HoduStorage> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn rand_uniform(_: &Layout, _: DType, _: f64, _: f64) -> HoduResult<Self::HoduStorage> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }
    }
}

pub mod storage {
    use super::{device::MetalDevice, *};

    #[derive(Debug, Clone)]
    pub struct MetalStorage;

    impl MetalStorage {
        pub fn from_cpu_storage(_: &CpuStorage) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }
    }

    impl HoduStorageT for MetalStorage {
        type HoduDevice = MetalDevice;

        fn get_dtype(&self) -> DType {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn get_device(&self) -> Device {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn get_hodu_device(&self) -> &Self::HoduDevice {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn to_cpu_storage(&self) -> HoduResult<CpuStorage> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn const_set(&mut self, _: Scalar, _: &Layout) -> HoduResult<()> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn binary_impl<B: crate::op::BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn binary_logical_impl<B: crate::op::BinaryLogicalOpT>(
            &self,
            _: &Self,
            _: &Layout,
            _: &Layout,
        ) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn cmp_impl<C: crate::op::CmpOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn cmp_scalar_impl<C: crate::op::CmpScalarOpT>(&self, _: &Layout, _: Scalar) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn unary_impl<U: crate::op::UnaryOpT>(&self, _: &Layout) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn unary_logical_impl<U: crate::op::UnaryLogicalOpT>(&self, _: &Layout) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn unary_scalar_impl<U: crate::op::UnaryScalarOpT>(&self, _: &Layout, _: Scalar) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn matmul(&self, _: &Self, _: &Layout, _: &Layout) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn dot(&self, _: &Self, _: &Layout, _: &Layout) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn reduce(&self, _: crate::op::ReduceOp, _: &Layout, _: &[usize], _: bool) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn concat(&self, _: &[&Self], _: &[&Layout], _: usize) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn split(&self, _: &Layout, _: usize, _: &[usize]) -> HoduResult<Vec<Self>> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn index_select(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn index_put(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn scatter(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn scatter_add(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn scatter_max(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn scatter_min(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn conv1d(&self, _: &Self, _: &Layout, _: &Layout, _: &crate::op::conv::ParamsConv1D) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn conv2d(&self, _: &Self, _: &Layout, _: &Layout, _: &crate::op::conv::ParamsConv2D) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn conv3d(&self, _: &Self, _: &Layout, _: &Layout, _: &crate::op::conv::ParamsConv3D) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn conv_transpose1d(
            &self,
            _: &Self,
            _: &Layout,
            _: &Layout,
            _: &crate::op::conv::ParamsConvTranspose1D,
        ) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn conv_transpose2d(
            &self,
            _: &Self,
            _: &Layout,
            _: &Layout,
            _: &crate::op::conv::ParamsConvTranspose2D,
        ) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn conv_transpose3d(
            &self,
            _: &Self,
            _: &Layout,
            _: &Layout,
            _: &crate::op::conv::ParamsConvTranspose3D,
        ) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn conv1d_grad_weight(
            &self,
            _: &Self,
            _: &Layout,
            _: &Layout,
            _: &crate::op::conv::ParamsConv1D,
        ) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn conv2d_grad_weight(
            &self,
            _: &Self,
            _: &Layout,
            _: &Layout,
            _: &crate::op::conv::ParamsConv2D,
        ) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn conv3d_grad_weight(
            &self,
            _: &Self,
            _: &Layout,
            _: &Layout,
            _: &crate::op::conv::ParamsConv3D,
        ) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn conv_transpose1d_grad_weight(
            &self,
            _: &Self,
            _: &Layout,
            _: &Layout,
            _: &crate::op::conv::ParamsConvTranspose1D,
        ) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn conv_transpose2d_grad_weight(
            &self,
            _: &Self,
            _: &Layout,
            _: &Layout,
            _: &crate::op::conv::ParamsConvTranspose2D,
        ) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn conv_transpose3d_grad_weight(
            &self,
            _: &Self,
            _: &Layout,
            _: &Layout,
            _: &crate::op::conv::ParamsConvTranspose3D,
        ) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn reduce_window(
            &self,
            _: &Layout,
            _: &[usize],
            _: &[usize],
            _: &[(usize, usize)],
            _: crate::op::window_reduction::WindowReduction,
        ) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn to_dtype(&self, _: DType, _: &Layout) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }

        fn contiguous(&self, _: &Layout) -> HoduResult<Self> {
            panic!("Metal feature is not enabled. Please enable the 'metal' feature to use Metal backend.")
        }
    }
}

pub mod error {
    use crate::compat::*;

    #[derive(Debug)]
    pub enum LockError {}

    impl fmt::Display for LockError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "")
        }
    }

    #[cfg(feature = "std")]
    impl std::error::Error for LockError {}

    #[derive(Debug)]
    pub enum MetalError {}

    impl fmt::Display for MetalError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "")
        }
    }

    #[cfg(feature = "std")]
    impl std::error::Error for MetalError {}
}
