use crate::{
    backends::{
        be_hodu::{
            cpu::storage::CpuStorage,
            device::{HoduDevice, HoduDeviceT},
        },
        op::{
            conv::{
                ParamsConv1D, ParamsConv2D, ParamsConv3D, ParamsConvTranspose1D, ParamsConvTranspose2D,
                ParamsConvTranspose3D,
            },
            window_reduction::WindowReduction,
            BinaryLogicalOpT, BinaryOpT, CmpOpT, CmpScalarOpT, ReduceOp, UnaryLogicalOpT, UnaryOpT, UnaryScalarOpT,
        },
    },
    compat::*,
    error::HoduResult,
    scalar::Scalar,
    types::{device::Device, dtype::DType, layout::Layout},
};

pub trait HoduStorageT: Sized {
    type HoduDevice: HoduDeviceT;

    fn get_dtype(&self) -> DType;

    fn get_device(&self) -> Device;

    fn get_hodu_device(&self) -> Self::HoduDevice;

    fn to_cpu_storage(&self) -> HoduResult<CpuStorage>;

    fn const_set(&mut self, _: Scalar, _: &Layout) -> HoduResult<()>;

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> HoduResult<Self>;

    fn binary_logical_impl<B: BinaryLogicalOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> HoduResult<Self>;

    fn cmp_impl<C: CmpOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> HoduResult<Self>;

    fn cmp_scalar_impl<C: CmpScalarOpT>(&self, _: &Layout, _: Scalar) -> HoduResult<Self>;

    fn unary_impl<U: UnaryOpT>(&self, _: &Layout) -> HoduResult<Self>;

    fn unary_logical_impl<U: UnaryLogicalOpT>(&self, _: &Layout) -> HoduResult<Self>;

    fn unary_scalar_impl<U: UnaryScalarOpT>(&self, _: &Layout, _: Scalar) -> HoduResult<Self>;

    fn matmul(&self, _: &Self, _: &Layout, _: &Layout) -> HoduResult<Self>;

    fn dot(&self, _: &Self, _: &Layout, _: &Layout) -> HoduResult<Self>;

    fn reduce(&self, _: ReduceOp, _: &Layout, _: &[usize], _: bool) -> HoduResult<Self>;

    fn concat(&self, _: &[&Self], _: &[&Layout], _: usize) -> HoduResult<Self>;

    fn split(&self, _: &Layout, _: usize, _: &[usize]) -> HoduResult<Vec<Self>>;

    fn index_select(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self>;

    fn index_put(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self>;

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self>;

    fn scatter(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self>;

    fn scatter_add(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self>;

    fn scatter_max(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self>;

    fn scatter_min(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize) -> HoduResult<Self>;

    fn conv1d(&self, _: &Self, _: &Layout, _: &Layout, _: &ParamsConv1D) -> HoduResult<Self>;

    fn conv2d(&self, _: &Self, _: &Layout, _: &Layout, _: &ParamsConv2D) -> HoduResult<Self>;

    fn conv3d(&self, _: &Self, _: &Layout, _: &Layout, _: &ParamsConv3D) -> HoduResult<Self>;

    fn conv_transpose1d(&self, _: &Self, _: &Layout, _: &Layout, _: &ParamsConvTranspose1D) -> HoduResult<Self>;

    fn conv_transpose2d(&self, _: &Self, _: &Layout, _: &Layout, _: &ParamsConvTranspose2D) -> HoduResult<Self>;

    fn conv_transpose3d(&self, _: &Self, _: &Layout, _: &Layout, _: &ParamsConvTranspose3D) -> HoduResult<Self>;

    fn conv1d_grad_weight(&self, _: &Self, _: &Layout, _: &Layout, _: &ParamsConv1D) -> HoduResult<Self>;

    fn conv2d_grad_weight(&self, _: &Self, _: &Layout, _: &Layout, _: &ParamsConv2D) -> HoduResult<Self>;

    fn conv3d_grad_weight(&self, _: &Self, _: &Layout, _: &Layout, _: &ParamsConv3D) -> HoduResult<Self>;

    fn conv_transpose1d_grad_weight(
        &self,
        _: &Self,
        _: &Layout,
        _: &Layout,
        _: &ParamsConvTranspose1D,
    ) -> HoduResult<Self>;

    fn conv_transpose2d_grad_weight(
        &self,
        _: &Self,
        _: &Layout,
        _: &Layout,
        _: &ParamsConvTranspose2D,
    ) -> HoduResult<Self>;

    fn conv_transpose3d_grad_weight(
        &self,
        _: &Self,
        _: &Layout,
        _: &Layout,
        _: &ParamsConvTranspose3D,
    ) -> HoduResult<Self>;

    fn reduce_window(
        &self,
        _: &Layout,
        _: &[usize],
        _: &[usize],
        _: &[(usize, usize)],
        _: WindowReduction,
    ) -> HoduResult<Self>;

    fn to_dtype(&self, _: DType) -> HoduResult<Self>;

    fn contiguous(&self, _: &Layout) -> HoduResult<Self>;
}

#[derive(Debug, Clone)]
pub enum HoduStorage {
    CPU(CpuStorage),
}

impl HoduStorage {
    pub fn get_dtype(&self) -> DType {
        match self {
            Self::CPU(storage) => storage.get_dtype(),
        }
    }

    pub fn get_device(&self) -> Device {
        match self {
            Self::CPU(storage) => storage.get_device(),
        }
    }

    pub fn get_hodu_device(&self) -> HoduDevice {
        match self {
            Self::CPU(storage) => HoduDevice::CPU(storage.get_hodu_device()),
        }
    }

    pub fn to_cpu_storage(&self) -> HoduResult<CpuStorage> {
        match self {
            Self::CPU(storage) => storage.to_cpu_storage(),
        }
    }

    pub(crate) fn const_set(&mut self, scalar: Scalar, layout: &Layout) -> HoduResult<()> {
        match self {
            Self::CPU(storage) => storage.const_set(scalar, layout),
        }
    }

    pub(crate) fn binary_impl<B: BinaryOpT>(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> HoduResult<Self> {
        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => {
                let storage = lhs_storage.binary_impl::<B>(rhs_storage, lhs_layout, rhs_layout)?;
                Ok(Self::CPU(storage))
            },
            // (lhs_storage, rhs_storage) => Err(HoduError::DeviceConflictInOp {
            //     left: lhs_storage.get_device(),
            //     right: rhs_storage.get_device(),
            //     op: B::NAME.to_string(),
            // }),
        }
    }

    pub(crate) fn binary_logical_impl<B: BinaryLogicalOpT>(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> HoduResult<Self> {
        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => {
                let storage = lhs_storage.binary_logical_impl::<B>(rhs_storage, lhs_layout, rhs_layout)?;
                Ok(Self::CPU(storage))
            },
        }
    }

    pub(crate) fn cmp_impl<C: CmpOpT>(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> HoduResult<Self> {
        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => {
                let storage = lhs_storage.cmp_impl::<C>(rhs_storage, lhs_layout, rhs_layout)?;
                Ok(Self::CPU(storage))
            },
        }
    }

    pub(crate) fn cmp_scalar_impl<C: CmpScalarOpT>(&self, layout: &Layout, scalar: Scalar) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => {
                let storage = storage.cmp_scalar_impl::<C>(layout, scalar)?;
                Ok(Self::CPU(storage))
            },
        }
    }

    pub(crate) fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => {
                let storage = storage.unary_impl::<U>(layout)?;
                Ok(Self::CPU(storage))
            },
        }
    }

    pub(crate) fn unary_logical_impl<U: UnaryLogicalOpT>(&self, layout: &Layout) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => {
                let storage = storage.unary_logical_impl::<U>(layout)?;
                Ok(Self::CPU(storage))
            },
        }
    }

    pub(crate) fn unary_scalar_impl<U: UnaryScalarOpT>(&self, layout: &Layout, scalar: Scalar) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => {
                let storage = storage.unary_scalar_impl::<U>(layout, scalar)?;
                Ok(Self::CPU(storage))
            },
        }
    }

    pub(crate) fn matmul(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> HoduResult<Self> {
        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => {
                let storage = lhs_storage.matmul(rhs_storage, lhs_layout, rhs_layout)?;
                Ok(Self::CPU(storage))
            },
        }
    }

    pub(crate) fn dot(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> HoduResult<Self> {
        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => {
                let storage = lhs_storage.dot(rhs_storage, lhs_layout, rhs_layout)?;
                Ok(Self::CPU(storage))
            },
        }
    }

    pub(crate) fn reduce(
        &self,
        reduce_op: crate::backends::op::ReduceOp,
        layout: &Layout,
        dims: &[usize],
        keep_dim: bool,
    ) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => {
                let reduced_storage = storage.reduce(reduce_op, layout, dims, keep_dim)?;
                Ok(Self::CPU(reduced_storage))
            },
        }
    }

    pub(crate) fn concat(&self, others: &[&Self], layouts: &[&Layout], dim: usize) -> HoduResult<Self> {
        match self {
            Self::CPU(first_storage) => {
                let other_cpu_storages: Vec<&CpuStorage> = others
                    .iter()
                    .map(|s| match s {
                        Self::CPU(storage) => storage,
                    })
                    .collect();
                let result = first_storage.concat(&other_cpu_storages, layouts, dim)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn split(&self, layout: &Layout, dim: usize, sizes: &[usize]) -> HoduResult<Vec<Self>> {
        match self {
            Self::CPU(storage) => {
                let results = storage.split(layout, dim, sizes)?;
                Ok(results.into_iter().map(Self::CPU).collect())
            },
        }
    }

    pub(crate) fn index_select(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        dim: usize,
    ) -> HoduResult<Self> {
        match (self, indices_storage) {
            (Self::CPU(storage), Self::CPU(indices)) => {
                let result = storage.index_select(layout, indices, indices_layout, dim)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn index_put(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        values_storage: &Self,
        values_layout: &Layout,
        dim: usize,
    ) -> HoduResult<Self> {
        match (self, indices_storage, values_storage) {
            (Self::CPU(storage), Self::CPU(indices), Self::CPU(values)) => {
                let result = storage.index_put(layout, indices, indices_layout, values, values_layout, dim)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn gather(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        dim: usize,
    ) -> HoduResult<Self> {
        match (self, indices_storage) {
            (Self::CPU(storage), Self::CPU(indices)) => {
                let result = storage.gather(layout, indices, indices_layout, dim)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn scatter(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        src_storage: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> HoduResult<Self> {
        match (self, indices_storage, src_storage) {
            (Self::CPU(storage), Self::CPU(indices), Self::CPU(src)) => {
                let result = storage.scatter(layout, indices, indices_layout, src, src_layout, dim)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn scatter_add(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        src_storage: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> HoduResult<Self> {
        match (self, indices_storage, src_storage) {
            (Self::CPU(storage), Self::CPU(indices), Self::CPU(src)) => {
                let result = storage.scatter_add(layout, indices, indices_layout, src, src_layout, dim)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn scatter_max(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        src_storage: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> HoduResult<Self> {
        match (self, indices_storage, src_storage) {
            (Self::CPU(storage), Self::CPU(indices), Self::CPU(src)) => {
                let result = storage.scatter_max(layout, indices, indices_layout, src, src_layout, dim)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn scatter_min(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        src_storage: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> HoduResult<Self> {
        match (self, indices_storage, src_storage) {
            (Self::CPU(storage), Self::CPU(indices), Self::CPU(src)) => {
                let result = storage.scatter_min(layout, indices, indices_layout, src, src_layout, dim)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn conv1d(
        &self,
        weight_storage: &Self,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &ParamsConv1D,
    ) -> HoduResult<Self> {
        match (self, weight_storage) {
            (Self::CPU(input), Self::CPU(weight)) => {
                let result = input.conv1d(weight, input_layout, weight_layout, params)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn conv2d(
        &self,
        weight_storage: &Self,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &ParamsConv2D,
    ) -> HoduResult<Self> {
        match (self, weight_storage) {
            (Self::CPU(input), Self::CPU(weight)) => {
                let result = input.conv2d(weight, input_layout, weight_layout, params)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn conv3d(
        &self,
        weight_storage: &Self,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &ParamsConv3D,
    ) -> HoduResult<Self> {
        match (self, weight_storage) {
            (Self::CPU(input), Self::CPU(weight)) => {
                let result = input.conv3d(weight, input_layout, weight_layout, params)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn conv_transpose1d(
        &self,
        weight_storage: &Self,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> HoduResult<Self> {
        match (self, weight_storage) {
            (Self::CPU(input), Self::CPU(weight)) => {
                let result = input.conv_transpose1d(weight, input_layout, weight_layout, params)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn conv_transpose2d(
        &self,
        weight_storage: &Self,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> HoduResult<Self> {
        match (self, weight_storage) {
            (Self::CPU(input), Self::CPU(weight)) => {
                let result = input.conv_transpose2d(weight, input_layout, weight_layout, params)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn conv_transpose3d(
        &self,
        weight_storage: &Self,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &ParamsConvTranspose3D,
    ) -> HoduResult<Self> {
        match (self, weight_storage) {
            (Self::CPU(input), Self::CPU(weight)) => {
                let result = input.conv_transpose3d(weight, input_layout, weight_layout, params)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn conv1d_grad_weight(
        &self,
        grad_output_storage: &Self,
        input_layout: &Layout,
        grad_output_layout: &Layout,
        params: &ParamsConv1D,
    ) -> HoduResult<Self> {
        match (self, grad_output_storage) {
            (Self::CPU(input), Self::CPU(grad_output)) => {
                let result = input.conv1d_grad_weight(grad_output, input_layout, grad_output_layout, params)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn conv2d_grad_weight(
        &self,
        grad_output_storage: &Self,
        input_layout: &Layout,
        grad_output_layout: &Layout,
        params: &ParamsConv2D,
    ) -> HoduResult<Self> {
        match (self, grad_output_storage) {
            (Self::CPU(input), Self::CPU(grad_output)) => {
                let result = input.conv2d_grad_weight(grad_output, input_layout, grad_output_layout, params)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn conv3d_grad_weight(
        &self,
        grad_output_storage: &Self,
        input_layout: &Layout,
        grad_output_layout: &Layout,
        params: &ParamsConv3D,
    ) -> HoduResult<Self> {
        match (self, grad_output_storage) {
            (Self::CPU(input), Self::CPU(grad_output)) => {
                let result = input.conv3d_grad_weight(grad_output, input_layout, grad_output_layout, params)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn conv_transpose1d_grad_weight(
        &self,
        grad_output_storage: &Self,
        input_layout: &Layout,
        grad_output_layout: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> HoduResult<Self> {
        match (self, grad_output_storage) {
            (Self::CPU(input), Self::CPU(grad_output)) => {
                let result =
                    input.conv_transpose1d_grad_weight(grad_output, input_layout, grad_output_layout, params)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn conv_transpose2d_grad_weight(
        &self,
        grad_output_storage: &Self,
        input_layout: &Layout,
        grad_output_layout: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> HoduResult<Self> {
        match (self, grad_output_storage) {
            (Self::CPU(input), Self::CPU(grad_output)) => {
                let result =
                    input.conv_transpose2d_grad_weight(grad_output, input_layout, grad_output_layout, params)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn conv_transpose3d_grad_weight(
        &self,
        grad_output_storage: &Self,
        input_layout: &Layout,
        grad_output_layout: &Layout,
        params: &ParamsConvTranspose3D,
    ) -> HoduResult<Self> {
        match (self, grad_output_storage) {
            (Self::CPU(input), Self::CPU(grad_output)) => {
                let result =
                    input.conv_transpose3d_grad_weight(grad_output, input_layout, grad_output_layout, params)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn reduce_window(
        &self,
        input_layout: &Layout,
        window_shape: &[usize],
        strides: &[usize],
        padding: &[(usize, usize)],
        reduction: WindowReduction,
    ) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => {
                let result = storage.reduce_window(input_layout, window_shape, strides, padding, reduction)?;
                Ok(Self::CPU(result))
            },
        }
    }

    pub(crate) fn to_dtype(&self, dtype: DType) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => {
                let converted_storage = storage.to_dtype(dtype)?;
                Ok(Self::CPU(converted_storage))
            },
        }
    }

    pub(crate) fn contiguous(&self, layout: &Layout) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => {
                let contiguous_storage = storage.contiguous(layout)?;
                Ok(Self::CPU(contiguous_storage))
            },
        }
    }
}
