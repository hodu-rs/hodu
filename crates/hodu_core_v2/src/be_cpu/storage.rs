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
    be_cpu::device::CpuDevice,
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    scalar::Scalar,
    types::{DType, Device, Layout},
};
use float8::F8E4M3;
#[cfg(feature = "f8e5m2")]
use float8::F8E5M2;
use half::{bf16, f16};

#[derive(Debug, Clone)]
pub enum CpuStorage {
    BOOL(Vec<bool>),
    F8E4M3(Vec<F8E4M3>),
    #[cfg(feature = "f8e5m2")]
    F8E5M2(Vec<F8E5M2>),
    BF16(Vec<bf16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    #[cfg(feature = "f64")]
    F64(Vec<f64>),
    U8(Vec<u8>),
    #[cfg(feature = "u16")]
    U16(Vec<u16>),
    U32(Vec<u32>),
    #[cfg(feature = "u64")]
    U64(Vec<u64>),
    I8(Vec<i8>),
    #[cfg(feature = "i16")]
    I16(Vec<i16>),
    I32(Vec<i32>),
    #[cfg(feature = "i64")]
    I64(Vec<i64>),
}

#[derive(Debug, Clone)]
pub enum CpuStorageRef<'a> {
    BOOL(&'a [bool]),
    F8E4M3(&'a [F8E4M3]),
    #[cfg(feature = "f8e5m2")]
    F8E5M2(&'a [F8E5M2]),
    BF16(&'a [bf16]),
    F16(&'a [f16]),
    F32(&'a [f32]),
    #[cfg(feature = "f64")]
    F64(&'a [f64]),
    U8(&'a [u8]),
    #[cfg(feature = "u16")]
    U16(&'a [u16]),
    U32(&'a [u32]),
    #[cfg(feature = "u64")]
    U64(&'a [u64]),
    I8(&'a [i8]),
    #[cfg(feature = "i16")]
    I16(&'a [i16]),
    I32(&'a [i32]),
    #[cfg(feature = "i64")]
    I64(&'a [i64]),
}

impl CpuStorage {
    pub fn from_vec<T: 'static>(vec: Vec<T>) -> Self {
        let any_vec = &vec as &dyn core::any::Any;

        macro_rules! try_downcast {
            ($type:ty, $variant:ident) => {
                if let Some(v) = any_vec.downcast_ref::<Vec<$type>>() {
                    return Self::$variant(v.clone());
                }
            };
            ($type:ty, $variant:ident, $cfg:meta) => {
                #[cfg($cfg)]
                if let Some(v) = any_vec.downcast_ref::<Vec<$type>>() {
                    return Self::$variant(v.clone());
                }
            };
        }

        try_downcast!(bool, BOOL);
        try_downcast!(F8E4M3, F8E4M3);
        try_downcast!(F8E5M2, F8E5M2, feature = "f8e5m2");
        try_downcast!(bf16, BF16);
        try_downcast!(f16, F16);
        try_downcast!(f32, F32);
        try_downcast!(f64, F64, feature = "f64");
        try_downcast!(u8, U8);
        try_downcast!(u16, U16, feature = "u16");
        try_downcast!(u32, U32);
        try_downcast!(u64, U64, feature = "u64");
        try_downcast!(i8, I8);
        try_downcast!(i16, I16, feature = "i16");
        try_downcast!(i32, I32);
        try_downcast!(i64, I64, feature = "i64");

        panic!("Unsupported vector type for CpuStorage");
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        macro_rules! to_bytes_float_convert {
            ($data:expr, $elem_size:expr) => {{
                let mut bytes = Vec::with_capacity($data.len() * $elem_size);
                for &f in $data {
                    bytes.extend_from_slice(&f32::from(f).to_le_bytes());
                }
                bytes
            }};
        }

        macro_rules! to_bytes_direct {
            ($data:expr, $elem_size:expr) => {{
                let mut bytes = Vec::with_capacity($data.len() * $elem_size);
                for &n in $data {
                    bytes.extend_from_slice(&n.to_le_bytes());
                }
                bytes
            }};
        }

        match self {
            Self::BOOL(data) => data.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect(),
            Self::F8E4M3(data) => to_bytes_float_convert!(data, 4),
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(data) => to_bytes_float_convert!(data, 4),
            Self::BF16(data) => to_bytes_float_convert!(data, 4),
            Self::F16(data) => to_bytes_float_convert!(data, 4),
            Self::F32(data) => to_bytes_direct!(data, 4),
            #[cfg(feature = "f64")]
            Self::F64(data) => to_bytes_direct!(data, 8),
            Self::U8(data) => data.clone(),
            #[cfg(feature = "u16")]
            Self::U16(data) => to_bytes_direct!(data, 2),
            Self::U32(data) => to_bytes_direct!(data, 4),
            #[cfg(feature = "u64")]
            Self::U64(data) => to_bytes_direct!(data, 8),
            Self::I8(data) => data.iter().map(|&n| n as u8).collect(),
            #[cfg(feature = "i16")]
            Self::I16(data) => to_bytes_direct!(data, 2),
            Self::I32(data) => to_bytes_direct!(data, 4),
            #[cfg(feature = "i64")]
            Self::I64(data) => to_bytes_direct!(data, 8),
        }
    }
}

impl BackendStorageT for CpuStorage {
    type BackendDevice = CpuDevice;

    fn dtype(&self) -> DType {
        match self {
            Self::BOOL(_) => DType::BOOL,
            Self::F8E4M3(_) => DType::F8E4M3,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(_) => DType::F8E5M2,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            #[cfg(feature = "f64")]
            Self::F64(_) => DType::F64,
            Self::U8(_) => DType::U8,
            #[cfg(feature = "u16")]
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            #[cfg(feature = "u64")]
            Self::U64(_) => DType::U64,
            Self::I8(_) => DType::I8,
            #[cfg(feature = "i16")]
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            #[cfg(feature = "i64")]
            Self::I64(_) => DType::I64,
        }
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn backend_device(&self) -> &CpuDevice {
        &CpuDevice
    }

    fn to_cpu_storage(&self) -> HoduResult<CpuStorage> {
        Ok(self.clone())
    }

    fn const_set(&mut self, scalar: Scalar, layout: &Layout) -> HoduResult<()> {
        #[inline]
        fn set_values<T: Copy>(data: &mut [T], layout: &Layout, value: T) {
            let shape = layout.shape();
            let strides = layout.strides();
            let offset = layout.offset();

            if strides == Layout::compute_strides(shape) && offset == 0 {
                data.fill(value);
            } else {
                let ndim = shape.ndim() as usize;
                let mut indices = vec![0u32; ndim];

                loop {
                    let flat_index = (offset
                        + indices
                            .iter()
                            .enumerate()
                            .map(|(i, &idx)| idx * strides[i])
                            .sum::<u32>()) as usize;

                    if flat_index < data.len() {
                        data[flat_index] = value;
                    }

                    // Increment multi-dimensional index
                    let mut carry = true;
                    for i in (0..ndim).rev() {
                        if carry {
                            indices[i] += 1;
                            if indices[i] < shape[i as u32] {
                                carry = false;
                            } else {
                                indices[i] = 0;
                            }
                        }
                    }

                    if carry {
                        break;
                    }
                }
            }
        }

        let expected_dtype = self.dtype();
        let got_dtype = scalar.dtype();

        macro_rules! match_and_set {
            ($($storage_variant:ident, $scalar_variant:ident $(, $cfg:meta)?);* $(;)?) => {
                match (self, scalar) {
                    $(
                        $(#[cfg($cfg)])?
                        (Self::$storage_variant(data), Scalar::$scalar_variant(v)) => {
                            set_values(data, layout, v);
                        }
                    )*
                    _ => {
                        return Err(HoduError::DTypeMismatch {
                            expected: expected_dtype,
                            got: got_dtype,
                        })
                    }
                }
            };
        }

        match_and_set! {
            BOOL, BOOL;
            F8E4M3, F8E4M3;
            F8E5M2, F8E5M2, feature = "f8e5m2";
            BF16, BF16;
            F16, F16;
            F32, F32;
            F64, F64, feature = "f64";
            U8, U8;
            U16, U16, feature = "u16";
            U32, U32;
            U64, U64, feature = "u64";
            I8, I8;
            I16, I16, feature = "i16";
            I32, I32;
            I64, I64, feature = "i64";
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
        weight_shape: &crate::types::Shape,
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
        reduce_op: crate::ops::ReduceOp,
        op: Op,
    ) -> HoduResult<Self> {
        ops_windowing::call_reduce_window(self, layout, window_shape, strides, padding, reduce_op, op)
    }

    fn to_dtype(&self, layout: &Layout, target_dtype: DType) -> HoduResult<Self> {
        if self.dtype() == target_dtype {
            return Ok(self.clone());
        }

        let result = match (self, target_dtype) {
            // BOOL conversions
            (Self::BOOL(storage), DType::BOOL) => Self::BOOL(storage.clone()),
            (Self::BOOL(storage), DType::F8E4M3) => Self::F8E4M3(
                storage
                    .iter()
                    .map(|&v| F8E4M3::from(if v { 1.0f32 } else { 0.0f32 }))
                    .collect(),
            ),
            #[cfg(feature = "f8e5m2")]
            (Self::BOOL(storage), DType::F8E5M2) => Self::F8E5M2(
                storage
                    .iter()
                    .map(|&v| F8E5M2::from(if v { 1.0f32 } else { 0.0f32 }))
                    .collect(),
            ),
            (Self::BOOL(storage), DType::BF16) => Self::BF16(
                storage
                    .iter()
                    .map(|&v| bf16::from_f32(if v { 1.0f32 } else { 0.0f32 }))
                    .collect(),
            ),
            (Self::BOOL(storage), DType::F16) => Self::F16(
                storage
                    .iter()
                    .map(|&v| f16::from_f32(if v { 1.0f32 } else { 0.0f32 }))
                    .collect(),
            ),
            (Self::BOOL(storage), DType::F32) => {
                Self::F32(storage.iter().map(|&v| if v { 1.0f32 } else { 0.0f32 }).collect())
            },
            #[cfg(feature = "f64")]
            (Self::BOOL(storage), DType::F64) => {
                Self::F64(storage.iter().map(|&v| if v { 1.0f64 } else { 0.0f64 }).collect())
            },
            (Self::BOOL(storage), DType::U8) => Self::U8(storage.iter().map(|&v| if v { 1u8 } else { 0u8 }).collect()),
            #[cfg(feature = "u16")]
            (Self::BOOL(storage), DType::U16) => {
                Self::U16(storage.iter().map(|&v| if v { 1u16 } else { 0u16 }).collect())
            },
            (Self::BOOL(storage), DType::U32) => {
                Self::U32(storage.iter().map(|&v| if v { 1u32 } else { 0u32 }).collect())
            },
            #[cfg(feature = "u64")]
            (Self::BOOL(storage), DType::U64) => {
                Self::U64(storage.iter().map(|&v| if v { 1u64 } else { 0u64 }).collect())
            },
            (Self::BOOL(storage), DType::I8) => Self::I8(storage.iter().map(|&v| if v { 1i8 } else { 0i8 }).collect()),
            #[cfg(feature = "i16")]
            (Self::BOOL(storage), DType::I16) => {
                Self::I16(storage.iter().map(|&v| if v { 1i16 } else { 0i16 }).collect())
            },
            (Self::BOOL(storage), DType::I32) => {
                Self::I32(storage.iter().map(|&v| if v { 1i32 } else { 0i32 }).collect())
            },
            #[cfg(feature = "i64")]
            (Self::BOOL(storage), DType::I64) => {
                Self::I64(storage.iter().map(|&v| if v { 1i64 } else { 0i64 }).collect())
            },

            // F8E4M3 conversions
            (Self::F8E4M3(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v.to_f32() != 0.0).collect()),
            (Self::F8E4M3(storage), DType::F8E4M3) => Self::F8E4M3(storage.clone()),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E4M3(storage), DType::F8E5M2) => {
                Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v.to_f32())).collect())
            },
            (Self::F8E4M3(storage), DType::BF16) => {
                Self::BF16(storage.iter().map(|&v| bf16::from_f32(v.to_f32())).collect())
            },
            (Self::F8E4M3(storage), DType::F16) => {
                Self::F16(storage.iter().map(|&v| f16::from_f32(v.to_f32())).collect())
            },
            (Self::F8E4M3(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v.to_f32()).collect()),
            #[cfg(feature = "f64")]
            (Self::F8E4M3(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v.to_f64()).collect()),
            (Self::F8E4M3(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v.to_f32() as u8).collect()),
            #[cfg(feature = "u16")]
            (Self::F8E4M3(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v.to_f32() as u16).collect()),
            (Self::F8E4M3(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v.to_f32() as u32).collect()),
            #[cfg(feature = "u64")]
            (Self::F8E4M3(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v.to_f32() as u64).collect()),
            (Self::F8E4M3(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v.to_f32() as i8).collect()),
            #[cfg(feature = "i16")]
            (Self::F8E4M3(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v.to_f32() as i16).collect()),
            (Self::F8E4M3(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v.to_f32() as i32).collect()),
            #[cfg(feature = "i64")]
            (Self::F8E4M3(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v.to_f32() as i64).collect()),

            // F8E5M2 conversions
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v.to_f32() != 0.0).collect()),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(storage), DType::F8E4M3) => {
                Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v.to_f32())).collect())
            },
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(storage), DType::F8E5M2) => Self::F8E5M2(storage.clone()),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(storage), DType::BF16) => {
                Self::BF16(storage.iter().map(|&v| bf16::from_f32(v.to_f32())).collect())
            },
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(storage), DType::F16) => {
                Self::F16(storage.iter().map(|&v| f16::from_f32(v.to_f32())).collect())
            },
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v.to_f32()).collect()),
            #[cfg(feature = "f8e5m2")]
            #[cfg(feature = "f64")]
            (Self::F8E5M2(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v.to_f64()).collect()),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v.to_f32() as u8).collect()),
            #[cfg(feature = "f8e5m2")]
            #[cfg(feature = "u16")]
            (Self::F8E5M2(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v.to_f32() as u16).collect()),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v.to_f32() as u32).collect()),
            #[cfg(feature = "f8e5m2")]
            #[cfg(feature = "u64")]
            (Self::F8E5M2(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v.to_f32() as u64).collect()),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v.to_f32() as i8).collect()),
            #[cfg(feature = "f8e5m2")]
            #[cfg(feature = "i16")]
            (Self::F8E5M2(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v.to_f32() as i16).collect()),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v.to_f32() as i32).collect()),
            #[cfg(feature = "f8e5m2")]
            #[cfg(feature = "i64")]
            (Self::F8E5M2(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v.to_f32() as i64).collect()),

            // BF16 conversions
            (Self::BF16(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v.to_f32() != 0.0).collect()),
            (Self::BF16(storage), DType::F8E4M3) => {
                Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v.to_f32())).collect())
            },
            #[cfg(feature = "f8e5m2")]
            (Self::BF16(storage), DType::F8E5M2) => {
                Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v.to_f32())).collect())
            },
            (Self::BF16(storage), DType::BF16) => Self::BF16(storage.clone()),
            (Self::BF16(storage), DType::F16) => {
                Self::F16(storage.iter().map(|&v| f16::from_f32(v.to_f32())).collect())
            },
            (Self::BF16(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v.to_f32()).collect()),
            #[cfg(feature = "f64")]
            (Self::BF16(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v.to_f64()).collect()),
            (Self::BF16(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v.to_f32() as u8).collect()),
            #[cfg(feature = "u16")]
            (Self::BF16(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v.to_f32() as u16).collect()),
            (Self::BF16(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v.to_f32() as u32).collect()),
            #[cfg(feature = "u64")]
            (Self::BF16(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v.to_f32() as u64).collect()),
            (Self::BF16(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v.to_f32() as i8).collect()),
            #[cfg(feature = "i16")]
            (Self::BF16(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v.to_f32() as i16).collect()),
            (Self::BF16(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v.to_f32() as i32).collect()),
            #[cfg(feature = "i64")]
            (Self::BF16(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v.to_f32() as i64).collect()),

            // F16 conversions
            (Self::F16(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v.to_f32() != 0.0).collect()),
            (Self::F16(storage), DType::F8E4M3) => {
                Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v.to_f32())).collect())
            },
            #[cfg(feature = "f8e5m2")]
            (Self::F16(storage), DType::F8E5M2) => {
                Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v.to_f32())).collect())
            },
            (Self::F16(storage), DType::BF16) => {
                Self::BF16(storage.iter().map(|&v| bf16::from_f32(v.to_f32())).collect())
            },
            (Self::F16(storage), DType::F16) => Self::F16(storage.clone()),
            (Self::F16(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v.to_f32()).collect()),
            #[cfg(feature = "f64")]
            (Self::F16(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v.to_f64()).collect()),
            (Self::F16(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v.to_f32() as u8).collect()),
            #[cfg(feature = "u16")]
            (Self::F16(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v.to_f32() as u16).collect()),
            (Self::F16(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v.to_f32() as u32).collect()),
            #[cfg(feature = "u64")]
            (Self::F16(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v.to_f32() as u64).collect()),
            (Self::F16(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v.to_f32() as i8).collect()),
            #[cfg(feature = "i16")]
            (Self::F16(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v.to_f32() as i16).collect()),
            (Self::F16(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v.to_f32() as i32).collect()),
            #[cfg(feature = "i64")]
            (Self::F16(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v.to_f32() as i64).collect()),

            // F32 conversions
            (Self::F32(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v != 0.0).collect()),
            (Self::F32(storage), DType::F8E4M3) => Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v)).collect()),
            #[cfg(feature = "f8e5m2")]
            (Self::F32(storage), DType::F8E5M2) => Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v)).collect()),
            (Self::F32(storage), DType::BF16) => Self::BF16(storage.iter().map(|&v| bf16::from_f32(v)).collect()),
            (Self::F32(storage), DType::F16) => Self::F16(storage.iter().map(|&v| f16::from_f32(v)).collect()),
            (Self::F32(storage), DType::F32) => Self::F32(storage.clone()),
            #[cfg(feature = "f64")]
            (Self::F32(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v as f64).collect()),
            (Self::F32(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v as u8).collect()),
            #[cfg(feature = "u16")]
            (Self::F32(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v as u16).collect()),
            (Self::F32(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v as u32).collect()),
            #[cfg(feature = "u64")]
            (Self::F32(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v as u64).collect()),
            (Self::F32(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v as i8).collect()),
            #[cfg(feature = "i16")]
            (Self::F32(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v as i16).collect()),
            (Self::F32(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v as i32).collect()),
            #[cfg(feature = "i64")]
            (Self::F32(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v as i64).collect()),

            // F64 conversions
            #[cfg(feature = "f64")]
            (Self::F64(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v != 0.0).collect()),
            #[cfg(feature = "f64")]
            (Self::F64(storage), DType::F8E4M3) => {
                Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v as f32)).collect())
            },
            #[cfg(feature = "f64")]
            #[cfg(feature = "f8e5m2")]
            (Self::F64(storage), DType::F8E5M2) => {
                Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v as f32)).collect())
            },
            #[cfg(feature = "f64")]
            (Self::F64(storage), DType::BF16) => Self::BF16(storage.iter().map(|&v| bf16::from_f64(v)).collect()),
            #[cfg(feature = "f64")]
            (Self::F64(storage), DType::F16) => Self::F16(storage.iter().map(|&v| f16::from_f64(v)).collect()),
            #[cfg(feature = "f64")]
            (Self::F64(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v as f32).collect()),
            #[cfg(feature = "f64")]
            (Self::F64(storage), DType::F64) => Self::F64(storage.clone()),
            #[cfg(feature = "f64")]
            (Self::F64(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v as u8).collect()),
            #[cfg(feature = "f64")]
            #[cfg(feature = "u16")]
            (Self::F64(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v as u16).collect()),
            #[cfg(feature = "f64")]
            (Self::F64(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v as u32).collect()),
            #[cfg(feature = "f64")]
            #[cfg(feature = "u64")]
            (Self::F64(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v as u64).collect()),
            #[cfg(feature = "f64")]
            (Self::F64(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v as i8).collect()),
            #[cfg(feature = "f64")]
            #[cfg(feature = "i16")]
            (Self::F64(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v as i16).collect()),
            #[cfg(feature = "f64")]
            (Self::F64(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v as i32).collect()),
            #[cfg(feature = "f64")]
            #[cfg(feature = "i64")]
            (Self::F64(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v as i64).collect()),

            // U8 conversions
            (Self::U8(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v != 0).collect()),
            (Self::U8(storage), DType::F8E4M3) => {
                Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v as f32)).collect())
            },
            #[cfg(feature = "f8e5m2")]
            (Self::U8(storage), DType::F8E5M2) => {
                Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v as f32)).collect())
            },
            (Self::U8(storage), DType::BF16) => Self::BF16(storage.iter().map(|&v| bf16::from_f32(v as f32)).collect()),
            (Self::U8(storage), DType::F16) => Self::F16(storage.iter().map(|&v| f16::from_f32(v as f32)).collect()),
            (Self::U8(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v as f32).collect()),
            #[cfg(feature = "f64")]
            (Self::U8(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v as f64).collect()),
            (Self::U8(storage), DType::U8) => Self::U8(storage.clone()),
            #[cfg(feature = "u16")]
            (Self::U8(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v as u16).collect()),
            (Self::U8(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v as u32).collect()),
            #[cfg(feature = "u64")]
            (Self::U8(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v as u64).collect()),
            (Self::U8(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v as i8).collect()),
            #[cfg(feature = "i16")]
            (Self::U8(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v as i16).collect()),
            (Self::U8(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v as i32).collect()),
            #[cfg(feature = "i64")]
            (Self::U8(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v as i64).collect()),

            // U16 conversions
            #[cfg(feature = "u16")]
            (Self::U16(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v != 0).collect()),
            #[cfg(feature = "u16")]
            (Self::U16(storage), DType::F8E4M3) => {
                Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v as f32)).collect())
            },
            #[cfg(feature = "u16")]
            #[cfg(feature = "f8e5m2")]
            (Self::U16(storage), DType::F8E5M2) => {
                Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v as f32)).collect())
            },
            #[cfg(feature = "u16")]
            (Self::U16(storage), DType::BF16) => {
                Self::BF16(storage.iter().map(|&v| bf16::from_f32(v as f32)).collect())
            },
            #[cfg(feature = "u16")]
            (Self::U16(storage), DType::F16) => Self::F16(storage.iter().map(|&v| f16::from_f32(v as f32)).collect()),
            #[cfg(feature = "u16")]
            (Self::U16(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v as f32).collect()),
            #[cfg(feature = "u16")]
            #[cfg(feature = "f64")]
            (Self::U16(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v as f64).collect()),
            #[cfg(feature = "u16")]
            (Self::U16(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v as u8).collect()),
            #[cfg(feature = "u16")]
            (Self::U16(storage), DType::U16) => Self::U16(storage.clone()),
            #[cfg(feature = "u16")]
            (Self::U16(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v as u32).collect()),
            #[cfg(feature = "u16")]
            #[cfg(feature = "u64")]
            (Self::U16(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v as u64).collect()),
            #[cfg(feature = "u16")]
            (Self::U16(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v as i8).collect()),
            #[cfg(feature = "u16")]
            #[cfg(feature = "i16")]
            (Self::U16(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v as i16).collect()),
            #[cfg(feature = "u16")]
            (Self::U16(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v as i32).collect()),
            #[cfg(feature = "u16")]
            #[cfg(feature = "i64")]
            (Self::U16(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v as i64).collect()),

            // U32 conversions
            (Self::U32(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v != 0).collect()),
            (Self::U32(storage), DType::F8E4M3) => {
                Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v as f32)).collect())
            },
            #[cfg(feature = "f8e5m2")]
            (Self::U32(storage), DType::F8E5M2) => {
                Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v as f32)).collect())
            },
            (Self::U32(storage), DType::BF16) => {
                Self::BF16(storage.iter().map(|&v| bf16::from_f32(v as f32)).collect())
            },
            (Self::U32(storage), DType::F16) => Self::F16(storage.iter().map(|&v| f16::from_f32(v as f32)).collect()),
            (Self::U32(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v as f32).collect()),
            #[cfg(feature = "f64")]
            (Self::U32(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v as f64).collect()),
            (Self::U32(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v as u8).collect()),
            #[cfg(feature = "u16")]
            (Self::U32(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v as u16).collect()),
            (Self::U32(storage), DType::U32) => Self::U32(storage.clone()),
            #[cfg(feature = "u64")]
            (Self::U32(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v as u64).collect()),
            (Self::U32(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v as i8).collect()),
            #[cfg(feature = "i16")]
            (Self::U32(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v as i16).collect()),
            (Self::U32(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v as i32).collect()),
            #[cfg(feature = "i64")]
            (Self::U32(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v as i64).collect()),

            // U64 conversions
            #[cfg(feature = "u64")]
            (Self::U64(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v != 0).collect()),
            #[cfg(feature = "u64")]
            (Self::U64(storage), DType::F8E4M3) => {
                Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v as f32)).collect())
            },
            #[cfg(feature = "u64")]
            #[cfg(feature = "f8e5m2")]
            (Self::U64(storage), DType::F8E5M2) => {
                Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v as f32)).collect())
            },
            #[cfg(feature = "u64")]
            (Self::U64(storage), DType::BF16) => {
                Self::BF16(storage.iter().map(|&v| bf16::from_f32(v as f32)).collect())
            },
            #[cfg(feature = "u64")]
            (Self::U64(storage), DType::F16) => Self::F16(storage.iter().map(|&v| f16::from_f32(v as f32)).collect()),
            #[cfg(feature = "u64")]
            (Self::U64(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v as f32).collect()),
            #[cfg(feature = "u64")]
            #[cfg(feature = "f64")]
            (Self::U64(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v as f64).collect()),
            #[cfg(feature = "u64")]
            (Self::U64(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v as u8).collect()),
            #[cfg(feature = "u64")]
            #[cfg(feature = "u16")]
            (Self::U64(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v as u16).collect()),
            #[cfg(feature = "u64")]
            (Self::U64(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v as u32).collect()),
            #[cfg(feature = "u64")]
            (Self::U64(storage), DType::U64) => Self::U64(storage.clone()),
            #[cfg(feature = "u64")]
            (Self::U64(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v as i8).collect()),
            #[cfg(feature = "u64")]
            #[cfg(feature = "i16")]
            (Self::U64(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v as i16).collect()),
            #[cfg(feature = "u64")]
            (Self::U64(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v as i32).collect()),
            #[cfg(feature = "u64")]
            #[cfg(feature = "i64")]
            (Self::U64(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v as i64).collect()),

            // I8 conversions
            (Self::I8(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v != 0).collect()),
            (Self::I8(storage), DType::F8E4M3) => {
                Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v as f32)).collect())
            },
            #[cfg(feature = "f8e5m2")]
            (Self::I8(storage), DType::F8E5M2) => {
                Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v as f32)).collect())
            },
            (Self::I8(storage), DType::BF16) => Self::BF16(storage.iter().map(|&v| bf16::from_f32(v as f32)).collect()),
            (Self::I8(storage), DType::F16) => Self::F16(storage.iter().map(|&v| f16::from_f32(v as f32)).collect()),
            (Self::I8(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v as f32).collect()),
            #[cfg(feature = "f64")]
            (Self::I8(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v as f64).collect()),
            (Self::I8(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v as u8).collect()),
            #[cfg(feature = "u16")]
            (Self::I8(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v as u16).collect()),
            (Self::I8(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v as u32).collect()),
            #[cfg(feature = "u64")]
            (Self::I8(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v as u64).collect()),
            (Self::I8(storage), DType::I8) => Self::I8(storage.clone()),
            #[cfg(feature = "i16")]
            (Self::I8(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v as i16).collect()),
            (Self::I8(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v as i32).collect()),
            #[cfg(feature = "i64")]
            (Self::I8(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v as i64).collect()),

            // I16 conversions
            #[cfg(feature = "i16")]
            (Self::I16(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v != 0).collect()),
            #[cfg(feature = "i16")]
            (Self::I16(storage), DType::F8E4M3) => {
                Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v as f32)).collect())
            },
            #[cfg(feature = "i16")]
            #[cfg(feature = "f8e5m2")]
            (Self::I16(storage), DType::F8E5M2) => {
                Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v as f32)).collect())
            },
            #[cfg(feature = "i16")]
            (Self::I16(storage), DType::BF16) => {
                Self::BF16(storage.iter().map(|&v| bf16::from_f32(v as f32)).collect())
            },
            #[cfg(feature = "i16")]
            (Self::I16(storage), DType::F16) => Self::F16(storage.iter().map(|&v| f16::from_f32(v as f32)).collect()),
            #[cfg(feature = "i16")]
            (Self::I16(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v as f32).collect()),
            #[cfg(feature = "i16")]
            #[cfg(feature = "f64")]
            (Self::I16(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v as f64).collect()),
            #[cfg(feature = "i16")]
            (Self::I16(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v as u8).collect()),
            #[cfg(feature = "i16")]
            #[cfg(feature = "u16")]
            (Self::I16(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v as u16).collect()),
            #[cfg(feature = "i16")]
            (Self::I16(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v as u32).collect()),
            #[cfg(feature = "i16")]
            #[cfg(feature = "u64")]
            (Self::I16(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v as u64).collect()),
            #[cfg(feature = "i16")]
            (Self::I16(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v as i8).collect()),
            #[cfg(feature = "i16")]
            (Self::I16(storage), DType::I16) => Self::I16(storage.clone()),
            #[cfg(feature = "i16")]
            (Self::I16(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v as i32).collect()),
            #[cfg(feature = "i16")]
            #[cfg(feature = "i64")]
            (Self::I16(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v as i64).collect()),

            // I32 conversions
            (Self::I32(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v != 0).collect()),
            (Self::I32(storage), DType::F8E4M3) => {
                Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v as f32)).collect())
            },
            #[cfg(feature = "f8e5m2")]
            (Self::I32(storage), DType::F8E5M2) => {
                Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v as f32)).collect())
            },
            (Self::I32(storage), DType::BF16) => {
                Self::BF16(storage.iter().map(|&v| bf16::from_f32(v as f32)).collect())
            },
            (Self::I32(storage), DType::F16) => Self::F16(storage.iter().map(|&v| f16::from_f32(v as f32)).collect()),
            (Self::I32(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v as f32).collect()),
            #[cfg(feature = "f64")]
            (Self::I32(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v as f64).collect()),
            (Self::I32(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v as u8).collect()),
            #[cfg(feature = "u16")]
            (Self::I32(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v as u16).collect()),
            (Self::I32(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v as u32).collect()),
            #[cfg(feature = "u64")]
            (Self::I32(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v as u64).collect()),
            (Self::I32(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v as i8).collect()),
            #[cfg(feature = "i16")]
            (Self::I32(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v as i16).collect()),
            (Self::I32(storage), DType::I32) => Self::I32(storage.clone()),
            #[cfg(feature = "i64")]
            (Self::I32(storage), DType::I64) => Self::I64(storage.iter().map(|&v| v as i64).collect()),

            // I64 conversions
            #[cfg(feature = "i64")]
            (Self::I64(storage), DType::BOOL) => Self::BOOL(storage.iter().map(|&v| v != 0).collect()),
            #[cfg(feature = "i64")]
            (Self::I64(storage), DType::F8E4M3) => {
                Self::F8E4M3(storage.iter().map(|&v| F8E4M3::from(v as f32)).collect())
            },
            #[cfg(feature = "i64")]
            #[cfg(feature = "f8e5m2")]
            (Self::I64(storage), DType::F8E5M2) => {
                Self::F8E5M2(storage.iter().map(|&v| F8E5M2::from(v as f32)).collect())
            },
            #[cfg(feature = "i64")]
            (Self::I64(storage), DType::BF16) => {
                Self::BF16(storage.iter().map(|&v| bf16::from_f32(v as f32)).collect())
            },
            #[cfg(feature = "i64")]
            (Self::I64(storage), DType::F16) => Self::F16(storage.iter().map(|&v| f16::from_f32(v as f32)).collect()),
            #[cfg(feature = "i64")]
            (Self::I64(storage), DType::F32) => Self::F32(storage.iter().map(|&v| v as f32).collect()),
            #[cfg(feature = "i64")]
            #[cfg(feature = "f64")]
            (Self::I64(storage), DType::F64) => Self::F64(storage.iter().map(|&v| v as f64).collect()),
            #[cfg(feature = "i64")]
            (Self::I64(storage), DType::U8) => Self::U8(storage.iter().map(|&v| v as u8).collect()),
            #[cfg(feature = "i64")]
            #[cfg(feature = "u16")]
            (Self::I64(storage), DType::U16) => Self::U16(storage.iter().map(|&v| v as u16).collect()),
            #[cfg(feature = "i64")]
            (Self::I64(storage), DType::U32) => Self::U32(storage.iter().map(|&v| v as u32).collect()),
            #[cfg(feature = "i64")]
            #[cfg(feature = "u64")]
            (Self::I64(storage), DType::U64) => Self::U64(storage.iter().map(|&v| v as u64).collect()),
            #[cfg(feature = "i64")]
            (Self::I64(storage), DType::I8) => Self::I8(storage.iter().map(|&v| v as i8).collect()),
            #[cfg(feature = "i64")]
            #[cfg(feature = "i16")]
            (Self::I64(storage), DType::I16) => Self::I16(storage.iter().map(|&v| v as i16).collect()),
            #[cfg(feature = "i64")]
            (Self::I64(storage), DType::I32) => Self::I32(storage.iter().map(|&v| v as i32).collect()),
            #[cfg(feature = "i64")]
            (Self::I64(storage), DType::I64) => Self::I64(storage.clone()),
        };

        Ok(result)
    }

    fn contiguous(&self, layout: &Layout) -> HoduResult<Self> {
        // If already contiguous, return clone
        if layout.is_contiguous() {
            return Ok(self.clone());
        }

        let shape = layout.shape();
        let strides = layout.strides();
        let total_size = layout.size();

        macro_rules! contiguous_impl {
            ($storage:expr, $dtype_variant:ident) => {{
                let mut contiguous_data = Vec::with_capacity(total_size as usize);

                // Multi-dimensional index iteration
                let mut indices = vec![0; shape.ndim() as usize];
                for _ in 0..total_size {
                    // Calculate linear offset from multi-dimensional indices
                    let mut offset = layout.offset();
                    for (_i, (&idx, &stride)) in indices.iter().zip(strides.iter()).enumerate() {
                        offset += idx * stride;
                    }

                    // Copy element at calculated offset
                    if (offset as usize) < $storage.len() {
                        contiguous_data.push($storage[offset as usize]);
                    }

                    // Increment indices (row-major order)
                    let mut carry = 1;
                    for i in (0..shape.ndim()).rev() {
                        let i_usize = i as usize;
                        indices[i_usize] += carry;
                        if indices[i_usize] < shape[i] {
                            carry = 0;
                            break;
                        }
                        indices[i_usize] = 0;
                    }
                    if carry != 0 {
                        break;
                    }
                }

                Self::$dtype_variant(contiguous_data)
            }};
        }

        let result = match self {
            Self::BOOL(storage) => contiguous_impl!(storage, BOOL),
            Self::F8E4M3(storage) => contiguous_impl!(storage, F8E4M3),
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(storage) => contiguous_impl!(storage, F8E5M2),
            Self::BF16(storage) => contiguous_impl!(storage, BF16),
            Self::F16(storage) => contiguous_impl!(storage, F16),
            Self::F32(storage) => contiguous_impl!(storage, F32),
            #[cfg(feature = "f64")]
            Self::F64(storage) => contiguous_impl!(storage, F64),
            Self::U8(storage) => contiguous_impl!(storage, U8),
            #[cfg(feature = "u16")]
            Self::U16(storage) => contiguous_impl!(storage, U16),
            Self::U32(storage) => contiguous_impl!(storage, U32),
            #[cfg(feature = "u64")]
            Self::U64(storage) => contiguous_impl!(storage, U64),
            Self::I8(storage) => contiguous_impl!(storage, I8),
            #[cfg(feature = "i16")]
            Self::I16(storage) => contiguous_impl!(storage, I16),
            Self::I32(storage) => contiguous_impl!(storage, I32),
            #[cfg(feature = "i64")]
            Self::I64(storage) => contiguous_impl!(storage, I64),
        };

        Ok(result)
    }
}
