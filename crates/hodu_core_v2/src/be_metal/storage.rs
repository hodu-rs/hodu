mod ops_binary;
mod ops_concat_split;
mod ops_conv;
mod ops_indexing;
mod ops_matrix;
mod ops_reduce;
mod ops_unary;
mod ops_windowing;

use crate::{
    be::storage::BackendStorageT,
    be_cpu::storage::CpuStorage,
    be_metal::device::MetalDevice,
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    scalar::Scalar,
    types::{DType, Device, Layout, Shape},
};
use hodu_metal_kernels::{
    kernels::{call_cast, call_const_set, call_contiguous, const_set, contiguous},
    metal::Buffer,
    utils::BufferOffset,
};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MetalStorage {
    buffer: Arc<Buffer>,
    device: MetalDevice,
    count: usize,
    dtype: DType,
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

impl MetalStorage {
    pub fn new(buffer: Arc<Buffer>, device: MetalDevice, count: usize, dtype: DType) -> Self {
        Self {
            buffer,
            device,
            count,
            dtype,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn to_cpu<T: Clone>(&self) -> HoduResult<Vec<T>> {
        let size = self.count * self.dtype.get_size_in_bytes();
        let buffer = self.device.allocate_buffer(size)?;
        {
            let command_buffer = self.device.command_buffer()?;
            command_buffer.set_label("to_cpu");
            let blit = command_buffer.blit_command_encoder();
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, size);
            blit.end_encoding();
        }
        self.device.wait_until_completed()?;
        Ok(read_to_vec(&buffer, self.count))
    }

    pub fn from_cpu_storage(cpu_storage: &CpuStorage) -> HoduResult<Self> {
        let device = MetalDevice::global().clone();
        let dtype = cpu_storage.dtype();
        let count = match cpu_storage {
            CpuStorage::BOOL(v) => v.len(),
            CpuStorage::BF16(v) => v.len(),
            CpuStorage::F16(v) => v.len(),
            CpuStorage::F32(v) => v.len(),
            CpuStorage::U8(v) => v.len(),
            #[cfg(feature = "u16")]
            CpuStorage::U16(v) => v.len(),
            CpuStorage::U32(v) => v.len(),
            #[cfg(feature = "u64")]
            CpuStorage::U64(v) => v.len(),
            CpuStorage::I8(v) => v.len(),
            #[cfg(feature = "i16")]
            CpuStorage::I16(v) => v.len(),
            CpuStorage::I32(v) => v.len(),
            #[cfg(feature = "i64")]
            CpuStorage::I64(v) => v.len(),
            _ => {
                return Err(HoduError::InternalError(format!(
                    "Unsupported dtype for Metal: {:?}",
                    dtype
                )))
            },
        };
        let buffer = device.new_buffer_with_cpu_storage(cpu_storage)?;
        Ok(Self::new(buffer, device, count, dtype))
    }
}

impl BackendStorageT for MetalStorage {
    type BackendDevice = MetalDevice;

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> Device {
        Device::Metal
    }

    fn backend_device(&self) -> &MetalDevice {
        &self.device
    }

    fn to_cpu_storage(&self) -> HoduResult<CpuStorage> {
        match self.dtype {
            DType::BOOL => Ok(CpuStorage::BOOL(self.to_cpu()?)),
            DType::BF16 => Ok(CpuStorage::BF16(self.to_cpu()?)),
            DType::F16 => Ok(CpuStorage::F16(self.to_cpu()?)),
            DType::F32 => Ok(CpuStorage::F32(self.to_cpu()?)),
            DType::I8 => Ok(CpuStorage::I8(self.to_cpu()?)),
            #[cfg(feature = "i16")]
            DType::I16 => Ok(CpuStorage::I16(self.to_cpu()?)),
            DType::I32 => Ok(CpuStorage::I32(self.to_cpu()?)),
            #[cfg(feature = "i64")]
            DType::I64 => Ok(CpuStorage::I64(self.to_cpu()?)),
            DType::U8 => Ok(CpuStorage::U8(self.to_cpu()?)),
            #[cfg(feature = "u16")]
            DType::U16 => Ok(CpuStorage::U16(self.to_cpu()?)),
            DType::U32 => Ok(CpuStorage::U32(self.to_cpu()?)),
            #[cfg(feature = "u64")]
            DType::U64 => Ok(CpuStorage::U64(self.to_cpu()?)),

            // not supported
            DType::F8E4M3 => Ok(CpuStorage::F8E4M3(self.to_cpu()?)),
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => Ok(CpuStorage::F8E5M2(self.to_cpu()?)),
            #[cfg(feature = "f64")]
            DType::F64 => Ok(CpuStorage::F64(self.to_cpu()?)),
        }
    }

    fn const_set(&mut self, scalar: Scalar, layout: &Layout) -> HoduResult<()> {
        let shape = layout.shape();
        let strides = layout.strides();
        let offset = layout.offset();
        let num_els = layout.size();

        let command_buffer = self.device.command_buffer()?;

        // Build metadata: [num_els, num_dims, shape..., strides..., offset]
        let mut metadata = Vec::with_capacity(2 + shape.ndim() as usize * 2 + 1);
        metadata.push(num_els as usize);
        metadata.push(shape.ndim() as usize);
        for i in 0..shape.ndim() {
            metadata.push(shape[i] as usize);
        }
        for i in 0..shape.ndim() {
            metadata.push(strides[i as usize] as usize);
        }
        metadata.push(offset as usize);

        // Match scalar type and call appropriate kernel
        macro_rules! call_kernel {
            ($scalar_variant:ident, $kernel_variant:ident, $val:expr) => {
                call_const_set(
                    self.device.device(),
                    &command_buffer,
                    self.device.kernels(),
                    const_set::$kernel_variant,
                    &self.buffer,
                    &metadata,
                    $val,
                )?
            };
        }

        match (self.dtype, scalar) {
            (DType::BOOL, Scalar::BOOL(v)) => call_kernel!(BOOL, BOOL, v),
            (DType::BF16, Scalar::BF16(v)) => call_kernel!(BF16, BF16, v),
            (DType::F16, Scalar::F16(v)) => call_kernel!(F16, F16, v),
            (DType::F32, Scalar::F32(v)) => call_kernel!(F32, F32, v),
            (DType::I8, Scalar::I8(v)) => call_kernel!(I8, I8, v),
            #[cfg(feature = "i16")]
            (DType::I16, Scalar::I16(v)) => call_kernel!(I16, I16, v),
            (DType::I32, Scalar::I32(v)) => call_kernel!(I32, I32, v),
            #[cfg(feature = "i64")]
            (DType::I64, Scalar::I64(v)) => call_kernel!(I64, I64, v),
            (DType::U8, Scalar::U8(v)) => call_kernel!(U8, U8, v),
            #[cfg(feature = "u16")]
            (DType::U16, Scalar::U16(v)) => call_kernel!(U16, U16, v),
            (DType::U32, Scalar::U32(v)) => call_kernel!(U32, U32, v),
            #[cfg(feature = "u64")]
            (DType::U64, Scalar::U64(v)) => call_kernel!(U64, U64, v),
            _ => {
                return Err(HoduError::DTypeMismatch {
                    expected: self.dtype,
                    got: scalar.dtype(),
                })
            },
        }

        Ok(())
    }

    fn call_binary(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_binary::call_binary(self, rhs_storage, lhs_layout, rhs_layout, op)
    }

    fn call_binary_logical(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        ops_binary::call_binary_logical(self, rhs_storage, lhs_layout, rhs_layout, op)
    }

    fn call_cmp(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_binary::call_cmp(self, rhs_storage, lhs_layout, rhs_layout, op)
    }

    fn call_cmp_scalar(&self, layout: &Layout, scalar: Scalar, op: Op) -> HoduResult<Self> {
        ops_unary::call_cmp_scalar(self, layout, scalar, op)
    }

    fn call_unary(&self, layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_unary::call_unary(self, layout, op)
    }

    fn call_unary_logical(&self, layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_unary::call_unary_logical(self, layout, op)
    }

    fn call_unary_scalar(&self, layout: &Layout, scalar: Scalar, op: Op) -> HoduResult<Self> {
        ops_unary::call_unary_scalar(self, layout, scalar, op)
    }

    fn call_matmul(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_matrix::call_matmul(self, rhs_storage, lhs_layout, rhs_layout, op)
    }

    fn call_dot(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_matrix::call_dot(self, rhs_storage, lhs_layout, rhs_layout, op)
    }

    fn call_reduce(&self, layout: &Layout, dims: &[u32], keep_dim: bool, op: Op) -> HoduResult<Self> {
        ops_reduce::call_reduce(self, layout, dims, keep_dim, op)
    }

    fn call_concat(&self, others: &[&Self], layouts: &[&Layout], dim: u32, op: Op) -> HoduResult<Self> {
        ops_concat_split::call_concat(self, others, layouts, dim, op)
    }

    fn call_split(&self, layout: &Layout, dim: u32, start: u32, size: u32, op: Op) -> HoduResult<Self> {
        ops_concat_split::call_split(self, layout, dim, start, size, op)
    }

    fn call_index_select(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        dim: u32,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_index_select(self, layout, indices_storage, indices_layout, dim, op)
    }

    fn call_put(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        values_storage: &Self,
        values_layout: &Layout,
        dim: u32,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_index_put(
            self,
            layout,
            indices_storage,
            indices_layout,
            values_storage,
            values_layout,
            dim,
            op,
        )
    }

    fn call_gather(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        dim: u32,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_gather(self, layout, indices_storage, indices_layout, dim, op)
    }

    fn call_scatter(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        src_storage: &Self,
        src_layout: &Layout,
        dim: u32,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_scatter(
            self,
            layout,
            indices_storage,
            indices_layout,
            src_storage,
            src_layout,
            dim,
            op,
        )
    }

    fn call_conv(
        &self,
        layout: &Layout,
        weight_storage: &Self,
        weight_layout: &Layout,
        stride: &[u32],
        padding: &[u32],
        dilation: &[u32],
        op: Op,
    ) -> HoduResult<Self> {
        ops_conv::call_conv(
            self,
            layout,
            weight_storage,
            weight_layout,
            stride,
            padding,
            dilation,
            op,
        )
    }

    fn call_conv_grad_weight(
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
        ops_conv::call_conv_grad_weight(
            self,
            layout,
            grad_output_storage,
            grad_output_layout,
            weight_shape,
            stride,
            padding,
            dilation,
            op,
        )
    }

    fn call_reduce_window(
        &self,
        layout: &Layout,
        window_shape: &[u32],
        strides: &[u32],
        padding: &[u32],
        op: Op,
    ) -> HoduResult<Self> {
        ops_windowing::call_reduce_window(self, layout, window_shape, strides, padding, op)
    }

    fn to_dtype(&self, layout: &Layout, target_dtype: DType) -> HoduResult<Self> {
        if self.dtype == target_dtype {
            // Still need to make it contiguous according to layout
            return self.contiguous(layout);
        }

        let shape = layout.shape();
        let strides = layout.strides();
        let offset = layout.offset();
        let num_els = layout.size();

        // Create output buffer with target dtype
        let output_buffer = self.device.new_buffer(num_els as usize, target_dtype, "to_dtype")?;

        let command_buffer = self.device.command_buffer()?;

        // Build metadata: [num_els, num_dims, shape..., strides..., offset]
        let mut metadata = Vec::with_capacity(2 + shape.ndim() as usize * 2 + 1);
        metadata.push(num_els as usize);
        metadata.push(shape.ndim() as usize);
        for i in 0..shape.ndim() {
            metadata.push(shape[i] as usize);
        }
        for i in 0..shape.ndim() {
            metadata.push(strides[i as usize] as usize);
        }
        metadata.push(offset as usize);

        let input = BufferOffset {
            buffer: &self.buffer,
            offset_in_bytes: 0,
        };

        // Build kernel name: cast_<src>_to_<dst>
        let kernel_name = format!("cast_{}_to_{}", self.dtype, target_dtype);
        let kernel_name_str = Box::leak(kernel_name.into_boxed_str());

        call_cast(
            self.device.device(),
            &command_buffer,
            self.device.kernels(),
            kernel_name_str,
            input,
            &output_buffer,
            &metadata,
        )?;

        Ok(Self::new(
            output_buffer,
            self.device.clone(),
            num_els as usize,
            target_dtype,
        ))
    }

    fn contiguous(&self, layout: &Layout) -> HoduResult<Self> {
        // If already contiguous, return clone
        if layout.is_contiguous() {
            return Ok(self.clone());
        }

        let shape = layout.shape();
        let strides = layout.strides();
        let offset = layout.offset();
        let num_els = layout.size();

        // Create output buffer
        let output_buffer = self.device.new_buffer(num_els as usize, self.dtype, "contiguous")?;

        let command_buffer = self.device.command_buffer()?;

        // Build metadata: [num_els, num_dims, shape..., strides..., offset]
        let mut metadata = Vec::with_capacity(2 + shape.ndim() as usize * 2 + 1);
        metadata.push(num_els as usize);
        metadata.push(shape.ndim() as usize);
        for i in 0..shape.ndim() {
            metadata.push(shape[i] as usize);
        }
        for i in 0..shape.ndim() {
            metadata.push(strides[i as usize] as usize);
        }
        metadata.push(offset as usize);

        let input = BufferOffset {
            buffer: &self.buffer,
            offset_in_bytes: 0,
        };

        // Call appropriate contiguous kernel based on dtype
        macro_rules! call_kernel {
            ($kernel_variant:ident) => {
                call_contiguous(
                    self.device.device(),
                    &command_buffer,
                    self.device.kernels(),
                    contiguous::$kernel_variant,
                    input,
                    &output_buffer,
                    &metadata,
                )?
            };
        }

        match self.dtype {
            DType::BOOL => call_kernel!(BOOL),
            DType::BF16 => call_kernel!(BF16),
            DType::F16 => call_kernel!(F16),
            DType::F32 => call_kernel!(F32),
            DType::I8 => call_kernel!(I8),
            #[cfg(feature = "i16")]
            DType::I16 => call_kernel!(I16),
            DType::I32 => call_kernel!(I32),
            #[cfg(feature = "i64")]
            DType::I64 => call_kernel!(I64),
            DType::U8 => call_kernel!(U8),
            #[cfg(feature = "u16")]
            DType::U16 => call_kernel!(U16),
            DType::U32 => call_kernel!(U32),
            #[cfg(feature = "u64")]
            DType::U64 => call_kernel!(U64),
            _ => {
                return Err(HoduError::UnsupportedDTypeForDevice {
                    dtype: self.dtype,
                    device: Device::Metal,
                })
            },
        }

        Ok(Self::new(
            output_buffer,
            self.device.clone(),
            num_els as usize,
            self.dtype,
        ))
    }
}
