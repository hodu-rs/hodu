mod ops_binary;
mod ops_unary;

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
}
