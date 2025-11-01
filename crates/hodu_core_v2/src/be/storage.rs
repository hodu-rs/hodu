#![allow(clippy::too_many_arguments)]

use crate::{
    be::device::{BackendDevice, BackendDeviceT},
    be_cpu::storage::CpuStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{Op, ReduceOp},
    scalar::Scalar,
    types::{DType, Device, Layout, Shape},
};

pub trait BackendStorageT: Sized {
    type BackendDevice: BackendDeviceT;

    fn dtype(&self) -> DType;

    fn device(&self) -> Device;

    fn backend_device(&self) -> &Self::BackendDevice;

    fn to_cpu_storage(&self) -> HoduResult<CpuStorage>;

    fn const_set(&mut self, _: Scalar, _: &Layout) -> HoduResult<()>;

    fn call_binary(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_binary_logical(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_cmp(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_cmp_scalar(&self, _: &Layout, _: Scalar, _: Op) -> HoduResult<Self>;

    fn call_unary(&self, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_unary_logical(&self, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_unary_scalar(&self, _: &Layout, _: Scalar, _: Op) -> HoduResult<Self>;

    fn call_matmul(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_dot(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_reduce(&self, _: &Layout, _: &[u32], _: bool, _: Op) -> HoduResult<Self>;

    fn call_concat(&self, _: &[&Self], _: &[&Layout], _: u32, _: Op) -> HoduResult<Self>;

    fn call_split(&self, _: &Layout, _: u32, _: u32, _: u32, _: Op) -> HoduResult<Self>;

    fn call_index_select(&self, _: &Layout, _: &Self, _: &Layout, _: u32, _: Op) -> HoduResult<Self>;

    fn call_put(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: u32, _: Op) -> HoduResult<Self>;

    fn call_gather(&self, _: &Layout, _: &Self, _: &Layout, _: u32, _: Op) -> HoduResult<Self>;

    fn call_scatter(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: u32, _: Op) -> HoduResult<Self>;

    fn call_conv(&self, _: &Layout, _: &Self, _: &Layout, _: &[u32], _: &[u32], _: &[u32], _: Op) -> HoduResult<Self>;

    fn call_conv_grad_weight(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Shape,
        _: &[u32],
        _: &[u32],
        _: &[u32],
        _: Op,
    ) -> HoduResult<Self>;

    fn call_reduce_window(&self, _: &Layout, _: &[u32], _: &[u32], _: &[u32], _: ReduceOp, _: Op) -> HoduResult<Self>;

    fn to_dtype(&self, _: &Layout, _: DType) -> HoduResult<Self>;

    fn contiguous(&self, _: &Layout) -> HoduResult<Self>;
}

#[derive(Debug, Clone)]
pub enum BackendStorage {
    CPU(CpuStorage),
}

impl BackendStorage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::CPU(storage) => storage.dtype(),
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::CPU(storage) => storage.device(),
        }
    }

    pub fn backend_device(&self) -> BackendDevice {
        match self {
            Self::CPU(storage) => BackendDevice::CPU(storage.backend_device().clone()),
        }
    }

    pub fn to_cpu_storage(&self) -> HoduResult<CpuStorage> {
        match self {
            Self::CPU(storage) => Ok(storage.clone()),
        }
    }

    pub(crate) fn const_set(&mut self, scalar: Scalar, layout: &Layout) -> HoduResult<()> {
        match self {
            Self::CPU(storage) => storage.const_set(scalar, layout),
        }
    }

    pub(crate) fn call_binary(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        let lhs_device = self.device();
        let rhs_device = rhs_storage.device();
        if lhs_device != rhs_device {
            return Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            });
        }

        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => Ok(Self::CPU(lhs_storage.call_binary(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
        }
    }

    pub(crate) fn call_binary_logical(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        let lhs_device = self.device();
        let rhs_device = rhs_storage.device();
        if lhs_device != rhs_device {
            return Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            });
        }

        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => Ok(Self::CPU(lhs_storage.call_binary_logical(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
        }
    }

    pub(crate) fn call_cmp(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        let lhs_device = self.device();
        let rhs_device = rhs_storage.device();
        if lhs_device != rhs_device {
            return Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            });
        }

        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => Ok(Self::CPU(lhs_storage.call_cmp(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
        }
    }

    pub(crate) fn call_cmp_scalar(&self, layout: &Layout, scalar: Scalar, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_cmp_scalar(layout, scalar, op)?)),
        }
    }

    pub(crate) fn call_unary(&self, layout: &Layout, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_unary(layout, op)?)),
        }
    }

    pub(crate) fn call_unary_logical(&self, layout: &Layout, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_unary_logical(layout, op)?)),
        }
    }

    pub(crate) fn call_unary_scalar(&self, layout: &Layout, scalar: Scalar, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_unary_scalar(layout, scalar, op)?)),
        }
    }

    pub(crate) fn call_matmul(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        let lhs_device = self.device();
        let rhs_device = rhs_storage.device();
        if lhs_device != rhs_device {
            return Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            });
        }

        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => Ok(Self::CPU(lhs_storage.call_matmul(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
        }
    }

    pub(crate) fn call_dot(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        let lhs_device = self.device();
        let rhs_device = rhs_storage.device();
        if lhs_device != rhs_device {
            return Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            });
        }

        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => Ok(Self::CPU(lhs_storage.call_dot(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
        }
    }

    pub(crate) fn call_reduce(&self, layout: &Layout, dims: &[u32], keep_dim: bool, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_reduce(layout, dims, keep_dim, op)?)),
        }
    }

    pub(crate) fn call_concat(&self, others: &[&Self], layouts: &[&Layout], dim: u32, op: Op) -> HoduResult<Self> {
        // Check all storages are on the same device
        let device = self.device();
        for other in others {
            let other_device = other.device();
            if device != other_device {
                return Err(HoduError::DeviceMismatch {
                    expected: device,
                    got: other_device,
                });
            }
        }

        match self {
            Self::CPU(storage) => {
                let others_cpu: Vec<&CpuStorage> = others
                    .iter()
                    .map(|s| match s {
                        Self::CPU(cpu) => cpu,
                    })
                    .collect();
                Ok(Self::CPU(storage.call_concat(&others_cpu, layouts, dim, op)?))
            },
        }
    }

    pub(crate) fn call_split(&self, layout: &Layout, dim: u32, start: u32, size: u32, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_split(layout, dim, start, size, op)?)),
        }
    }

    pub(crate) fn call_index_select(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        dim: u32,
        op: Op,
    ) -> HoduResult<Self> {
        // Check devices match
        let device = self.device();
        let indices_device = indices_storage.device();
        if device != indices_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: indices_device,
            });
        }

        match (self, indices_storage) {
            (Self::CPU(storage), Self::CPU(indices)) => Ok(Self::CPU(storage.call_index_select(
                layout,
                indices,
                indices_layout,
                dim,
                op,
            )?)),
        }
    }

    pub(crate) fn call_index_put(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        values_storage: &Self,
        values_layout: &Layout,
        dim: u32,
        op: Op,
    ) -> HoduResult<Self> {
        // Check all devices match
        let device = self.device();
        let indices_device = indices_storage.device();
        let values_device = values_storage.device();

        if device != indices_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: indices_device,
            });
        }
        if device != values_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: values_device,
            });
        }

        match (self, indices_storage, values_storage) {
            (Self::CPU(storage), Self::CPU(indices), Self::CPU(values)) => Ok(Self::CPU(storage.call_put(
                layout,
                indices,
                indices_layout,
                values,
                values_layout,
                dim,
                op,
            )?)),
        }
    }

    pub(crate) fn call_gather(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        dim: u32,
        op: Op,
    ) -> HoduResult<Self> {
        // Check devices match
        let device = self.device();
        let indices_device = indices_storage.device();
        if device != indices_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: indices_device,
            });
        }

        match (self, indices_storage) {
            (Self::CPU(storage), Self::CPU(indices)) => Ok(Self::CPU(storage.call_gather(
                layout,
                indices,
                indices_layout,
                dim,
                op,
            )?)),
        }
    }

    pub(crate) fn call_scatter(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        src_storage: &Self,
        src_layout: &Layout,
        dim: u32,
        op: Op,
    ) -> HoduResult<Self> {
        // Check all devices match
        let device = self.device();
        let indices_device = indices_storage.device();
        let src_device = src_storage.device();

        if device != indices_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: indices_device,
            });
        }
        if device != src_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: src_device,
            });
        }

        match (self, indices_storage, src_storage) {
            (Self::CPU(storage), Self::CPU(indices), Self::CPU(src)) => Ok(Self::CPU(storage.call_scatter(
                layout,
                indices,
                indices_layout,
                src,
                src_layout,
                dim,
                op,
            )?)),
        }
    }

    pub(crate) fn call_conv(
        &self,
        layout: &Layout,
        weight_storage: &Self,
        weight_layout: &Layout,
        stride: &[u32],
        padding: &[u32],
        dilation: &[u32],
        op: Op,
    ) -> HoduResult<Self> {
        // Check devices match
        let device = self.device();
        let weight_device = weight_storage.device();
        if device != weight_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: weight_device,
            });
        }

        match (self, weight_storage) {
            (Self::CPU(storage), Self::CPU(weight)) => Ok(Self::CPU(storage.call_conv(
                layout,
                weight,
                weight_layout,
                stride,
                padding,
                dilation,
                op,
            )?)),
        }
    }

    pub(crate) fn call_conv_grad_weight(
        &self,
        layout: &Layout,
        grad_output_storage: &Self,
        grad_output_layout: &Layout,
        weight_shape: &Shape,
        stride: &[u32],
        padding: &[u32],
        dilation: &[u32],
        op: Op,
    ) -> HoduResult<Self> {
        // Check devices match
        let device = self.device();
        let grad_output_device = grad_output_storage.device();
        if device != grad_output_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: grad_output_device,
            });
        }

        match (self, grad_output_storage) {
            (Self::CPU(storage), Self::CPU(grad_output)) => Ok(Self::CPU(storage.call_conv_grad_weight(
                layout,
                grad_output,
                grad_output_layout,
                weight_shape,
                stride,
                padding,
                dilation,
                op,
            )?)),
        }
    }

    pub(crate) fn call_reduce_window(
        &self,
        layout: &Layout,
        window_shape: &[u32],
        strides: &[u32],
        padding: &[u32],
        reduce_op: ReduceOp,
        op: Op,
    ) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_reduce_window(
                layout,
                window_shape,
                strides,
                padding,
                reduce_op,
                op,
            )?)),
        }
    }

    pub(crate) fn to_dtype(&self, layout: &Layout, target_dtype: DType) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => {
                let converted_storage = storage.to_dtype(layout, target_dtype)?;
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
