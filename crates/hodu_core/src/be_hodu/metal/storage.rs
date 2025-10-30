use crate::{
    be_hodu::{
        cpu::storage::CpuStorage,
        metal::{device::MetalDevice, utils::*},
        storage::HoduStorageT,
    },
    error::{HoduError, HoduResult},
    op::{
        conv::{
            ParamsConv1D, ParamsConv2D, ParamsConv3D, ParamsConvTranspose1D, ParamsConvTranspose2D,
            ParamsConvTranspose3D,
        },
        window_reduction::WindowReduction,
        BinaryLogicalOpT, BinaryOpT, CmpOpT, CmpScalarOpT, ReduceOp, UnaryLogicalOpT, UnaryOpT, UnaryScalarOpT,
    },
    scalar::Scalar,
    types::{dtype::DType, layout::Layout},
};
use hodu_metal_kernels::metal::Buffer;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MetalStorage {
    buffer: Arc<Buffer>,
    device: MetalDevice,
    count: usize,
    dtype: DType,
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
        let dtype = cpu_storage.get_dtype();
        let count = match cpu_storage {
            CpuStorage::BOOL(v) => v.len(),
            CpuStorage::BF16(v) => v.len(),
            CpuStorage::F16(v) => v.len(),
            CpuStorage::F32(v) => v.len(),
            #[cfg(feature = "u8")]
            CpuStorage::U8(v) => v.len(),
            CpuStorage::U16(v) => v.len(),
            #[cfg(feature = "u32")]
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

impl HoduStorageT for MetalStorage {
    type HoduDevice = MetalDevice;

    fn get_dtype(&self) -> DType {
        self.dtype
    }

    fn get_device(&self) -> crate::types::device::Device {
        crate::types::device::Device::Metal
    }

    fn get_hodu_device(&self) -> &MetalDevice {
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
            #[cfg(feature = "u8")]
            DType::U8 => Ok(CpuStorage::U8(self.to_cpu()?)),
            DType::U16 => Ok(CpuStorage::U16(self.to_cpu()?)),
            #[cfg(feature = "u32")]
            DType::U32 => Ok(CpuStorage::U32(self.to_cpu()?)),
            #[cfg(feature = "u64")]
            DType::U64 => Ok(CpuStorage::U64(self.to_cpu()?)),

            // not supported
            DType::F8E4M3 => Ok(CpuStorage::F8E4M3(self.to_cpu()?)),
            DType::F8E5M2 => Ok(CpuStorage::F8E5M2(self.to_cpu()?)),
            DType::F64 => Ok(CpuStorage::F64(self.to_cpu()?)),
        }
    }

    fn const_set(&mut self, scalar: Scalar, layout: &Layout) -> HoduResult<()> {
        use hodu_metal_kernels::kernels::{call_const_set, const_set};

        let dtype = self.get_dtype();
        let device = self.get_hodu_device();
        let shape = layout.get_shape();
        let strides = layout.get_strides();
        let offset = layout.get_offset();
        let command_buffer = self.device.command_buffer()?;

        macro_rules! const_set_impl {
            ($ty:ty, $kernel:expr, $conv:expr) => {{
                let val: $ty = $conv;
                call_const_set(
                    device.device(),
                    &command_buffer,
                    device.kernels(),
                    $kernel,
                    shape,
                    strides,
                    offset,
                    val,
                    &self.buffer,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            }};
        }

        match (dtype, scalar) {
            (DType::BOOL, s) => const_set_impl!(bool, const_set::BOOL, s.to_bool()),
            (DType::BF16, s) => const_set_impl!(half::bf16, const_set::BF16, s.to_bf16()),
            (DType::F16, s) => const_set_impl!(half::f16, const_set::F16, s.to_f16()),
            (DType::F32, s) => const_set_impl!(f32, const_set::F32, s.to_f32()),
            #[cfg(feature = "u8")]
            (DType::U8, s) => const_set_impl!(u8, const_set::U8, s.to_u8()),
            (DType::U16, s) => const_set_impl!(u16, const_set::U16, s.to_u16()),
            #[cfg(feature = "u32")]
            (DType::U32, s) => const_set_impl!(u32, const_set::U32, s.to_u32()),
            #[cfg(feature = "u64")]
            (DType::U64, s) => const_set_impl!(u64, const_set::U64, s.to_u64()),
            (DType::I8, s) => const_set_impl!(i8, const_set::I8, s.to_i8()),
            #[cfg(feature = "i16")]
            (DType::I16, s) => const_set_impl!(i16, const_set::I16, s.to_i16()),
            (DType::I32, s) => const_set_impl!(i32, const_set::I32, s.to_i32()),
            #[cfg(feature = "i64")]
            (DType::I64, s) => const_set_impl!(i64, const_set::I64, s.to_i64()),

            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "const_set".to_string(),
                })
            },
        }

        Ok(())
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> HoduResult<Self> {
        binary_map(self, rhs_storage, lhs_layout, rhs_layout, B::NAME)
    }

    fn binary_logical_impl<B: BinaryLogicalOpT>(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> HoduResult<Self> {
        binary_logical_map(self, rhs_storage, lhs_layout, rhs_layout, B::NAME)
    }

    fn cmp_impl<C: CmpOpT>(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> HoduResult<Self> {
        cmp_map(self, rhs_storage, lhs_layout, rhs_layout, C::NAME)
    }

    fn cmp_scalar_impl<C: CmpScalarOpT>(&self, layout: &Layout, scalar: Scalar) -> HoduResult<Self> {
        cmp_scalar_map(self, layout, scalar, C::NAME)
    }

    fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> HoduResult<Self> {
        unary_map(self, layout, U::NAME)
    }

    fn unary_logical_impl<U: UnaryLogicalOpT>(&self, layout: &Layout) -> HoduResult<Self> {
        unary_logical_map(self, layout, U::NAME)
    }

    fn unary_scalar_impl<U: UnaryScalarOpT>(&self, layout: &Layout, scalar: Scalar) -> HoduResult<Self> {
        unary_scalar_map(self, layout, scalar, U::NAME)
    }

    fn matmul(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> HoduResult<Self> {
        match (self.get_dtype(), rhs_storage.get_dtype()) {
            (DType::BF16, DType::BF16) => {
                let result_storage = matmul_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            (DType::F16, DType::F16) => {
                let result_storage = matmul_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            (DType::F32, DType::F32) => {
                let result_storage = matmul_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            #[cfg(feature = "u8")]
            (DType::U8, DType::U8) => {
                let result_storage = matmul_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            (DType::U16, DType::U16) => {
                let result_storage = matmul_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            #[cfg(feature = "u32")]
            (DType::U32, DType::U32) => {
                let result_storage = matmul_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            #[cfg(feature = "u64")]
            (DType::U64, DType::U64) => {
                let result_storage = matmul_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            (DType::I8, DType::I8) => {
                let result_storage = matmul_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            #[cfg(feature = "i16")]
            (DType::I16, DType::I16) => {
                let result_storage = matmul_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            (DType::I32, DType::I32) => {
                let result_storage = matmul_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            #[cfg(feature = "i64")]
            (DType::I64, DType::I64) => {
                let result_storage = matmul_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            _ => Err(HoduError::DTypeConflictInOp {
                left: self.get_dtype(),
                right: rhs_storage.get_dtype(),
                op: "matmul".to_string(),
            }),
        }
    }

    fn dot(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> HoduResult<Self> {
        match (self.get_dtype(), rhs_storage.get_dtype()) {
            (DType::BF16, DType::BF16) => {
                let result_storage = dot_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            (DType::F16, DType::F16) => {
                let result_storage = dot_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            (DType::F32, DType::F32) => {
                let result_storage = dot_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            #[cfg(feature = "u8")]
            (DType::U8, DType::U8) => {
                let result_storage = dot_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            (DType::U16, DType::U16) => {
                let result_storage = dot_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            #[cfg(feature = "u32")]
            (DType::U32, DType::U32) => {
                let result_storage = dot_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            #[cfg(feature = "u64")]
            (DType::U64, DType::U64) => {
                let result_storage = dot_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            (DType::I8, DType::I8) => {
                let result_storage = dot_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            #[cfg(feature = "i16")]
            (DType::I16, DType::I16) => {
                let result_storage = dot_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            (DType::I32, DType::I32) => {
                let result_storage = dot_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            #[cfg(feature = "i64")]
            (DType::I64, DType::I64) => {
                let result_storage = dot_map(self, rhs_storage, lhs_layout, rhs_layout)?;
                Ok(result_storage)
            },
            _ => Err(HoduError::DTypeConflictInOp {
                left: self.get_dtype(),
                right: rhs_storage.get_dtype(),
                op: "dot".to_string(),
            }),
        }
    }

    fn reduce(&self, reduce_op: ReduceOp, layout: &Layout, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        reduce_map(self, layout, reduce_op, dims, keep_dim)
    }

    fn concat(&self, others: &[&Self], layouts: &[&Layout], dim: usize) -> HoduResult<Self> {
        concat_map(self, others, layouts, dim)
    }

    fn split(&self, layout: &Layout, dim: usize, sizes: &[usize]) -> HoduResult<Vec<Self>> {
        split_map(self, layout, dim, sizes)
    }

    fn index_select(&self, layout: &Layout, indices: &Self, indices_layout: &Layout, dim: usize) -> HoduResult<Self> {
        index_select_map(self, layout, indices, indices_layout, dim)
    }

    fn index_put(
        &self,
        layout: &Layout,
        indices: &Self,
        indices_layout: &Layout,
        values: &Self,
        values_layout: &Layout,
        dim: usize,
    ) -> HoduResult<Self> {
        index_put_map(self, layout, indices, indices_layout, values, values_layout, dim)
    }

    fn gather(&self, layout: &Layout, indices: &Self, indices_layout: &Layout, dim: usize) -> HoduResult<Self> {
        gather_map(self, layout, indices, indices_layout, dim)
    }

    fn scatter(
        &self,
        layout: &Layout,
        indices: &Self,
        indices_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> HoduResult<Self> {
        scatter_map(self, layout, indices, indices_layout, src, src_layout, dim)
    }

    fn scatter_add(
        &self,
        layout: &Layout,
        indices: &Self,
        indices_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> HoduResult<Self> {
        scatter_add_map(self, layout, indices, indices_layout, src, src_layout, dim)
    }

    fn scatter_max(
        &self,
        layout: &Layout,
        indices: &Self,
        indices_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> HoduResult<Self> {
        scatter_max_map(self, layout, indices, indices_layout, src, src_layout, dim)
    }

    fn scatter_min(
        &self,
        layout: &Layout,
        indices: &Self,
        indices_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> HoduResult<Self> {
        scatter_min_map(self, layout, indices, indices_layout, src, src_layout, dim)
    }

    fn conv1d(
        &self,
        weight: &Self,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &ParamsConv1D,
    ) -> HoduResult<Self> {
        conv1d_map(self, weight, input_layout, weight_layout, params)
    }

    fn conv2d(
        &self,
        weight: &Self,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &ParamsConv2D,
    ) -> HoduResult<Self> {
        conv2d_map(self, weight, input_layout, weight_layout, params)
    }

    fn conv3d(
        &self,
        weight: &Self,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &ParamsConv3D,
    ) -> HoduResult<Self> {
        conv3d_map(self, weight, input_layout, weight_layout, params)
    }

    fn conv_transpose1d(
        &self,
        weight_storage: &Self,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> HoduResult<Self> {
        conv_transpose1d_map(self, input_layout, weight_storage, weight_layout, params)
    }

    fn conv_transpose2d(
        &self,
        weight_storage: &Self,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> HoduResult<Self> {
        conv_transpose2d_map(self, input_layout, weight_storage, weight_layout, params)
    }

    fn conv_transpose3d(
        &self,
        weight_storage: &Self,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &ParamsConvTranspose3D,
    ) -> HoduResult<Self> {
        conv_transpose3d_map(self, input_layout, weight_storage, weight_layout, params)
    }

    fn conv1d_grad_weight(
        &self,
        grad_output: &Self,
        input_layout: &Layout,
        grad_output_layout: &Layout,
        params: &ParamsConv1D,
    ) -> HoduResult<Self> {
        conv1d_grad_weight_map(self, grad_output, input_layout, grad_output_layout, params)
    }

    fn conv2d_grad_weight(
        &self,
        grad_output: &Self,
        input_layout: &Layout,
        grad_output_layout: &Layout,
        params: &ParamsConv2D,
    ) -> HoduResult<Self> {
        conv2d_grad_weight_map(self, grad_output, input_layout, grad_output_layout, params)
    }

    fn conv3d_grad_weight(
        &self,
        grad_output: &Self,
        input_layout: &Layout,
        grad_output_layout: &Layout,
        params: &ParamsConv3D,
    ) -> HoduResult<Self> {
        conv3d_grad_weight_map(self, grad_output, input_layout, grad_output_layout, params)
    }

    fn conv_transpose1d_grad_weight(
        &self,
        grad_output: &Self,
        input_layout: &Layout,
        grad_output_layout: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> HoduResult<Self> {
        conv_transpose1d_grad_weight_map(self, grad_output, input_layout, grad_output_layout, params)
    }

    fn conv_transpose2d_grad_weight(
        &self,
        grad_output: &Self,
        input_layout: &Layout,
        grad_output_layout: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> HoduResult<Self> {
        conv_transpose2d_grad_weight_map(self, grad_output, input_layout, grad_output_layout, params)
    }

    fn conv_transpose3d_grad_weight(
        &self,
        grad_output: &Self,
        input_layout: &Layout,
        grad_output_layout: &Layout,
        params: &ParamsConvTranspose3D,
    ) -> HoduResult<Self> {
        conv_transpose3d_grad_weight_map(self, grad_output, input_layout, grad_output_layout, params)
    }

    fn reduce_window(
        &self,
        input_layout: &Layout,
        window_shape: &[usize],
        strides: &[usize],
        padding: &[(usize, usize)],
        reduction: WindowReduction,
    ) -> HoduResult<Self> {
        reduce_window_map(self, input_layout, window_shape, strides, padding, reduction)
    }

    fn to_dtype(&self, target_dtype: DType, input_layout: &Layout) -> HoduResult<Self> {
        to_dtype_map(self, input_layout, target_dtype)
    }

    fn contiguous(&self, layout: &Layout) -> HoduResult<Self> {
        contiguous_map(self, layout)
    }
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}
