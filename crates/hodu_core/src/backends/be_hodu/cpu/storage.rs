use crate::{
    backends::{
        be_hodu::{
            cpu::{
                device::CpuDevice,
                utils::{
                    binary_logical_map, binary_map, cmp_map, cmp_scalar_map, concat_map, dot_map, matmul_map,
                    reduce_max, reduce_mean, reduce_min, reduce_norm, reduce_prod, reduce_std, reduce_sum, reduce_var,
                    split_map, unary_logical_map, unary_map, unary_scalar_map,
                },
            },
            storage::HoduStorageT,
        },
        op::{BinaryLogicalOpT, BinaryOpT, CmpOpT, CmpScalarOpT, ReduceOp, UnaryLogicalOpT, UnaryOpT, UnaryScalarOpT},
    },
    compat::*,
    error::{HoduError, HoduResult},
    scalar::Scalar,
    types::{device::Device, dtype::DType, layout::Layout},
};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};

#[derive(Debug, Clone)]
pub enum CpuStorage {
    BOOL(Vec<bool>),
    F8E4M3(Vec<F8E4M3>),
    F8E5M2(Vec<F8E5M2>),
    BF16(Vec<bf16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

#[derive(Debug, Clone)]
pub enum CpuStorageRef<'a> {
    BOOL(&'a [bool]),
    F8E4M3(&'a [F8E4M3]),
    F8E5M2(&'a [F8E5M2]),
    BF16(&'a [bf16]),
    F16(&'a [f16]),
    F32(&'a [f32]),
    F64(&'a [f64]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
    I8(&'a [i8]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
}

impl CpuStorage {
    pub fn from_vec<T: 'static>(vec: Vec<T>) -> Self {
        let any_vec = &vec as &dyn core::any::Any;

        if let Some(v) = any_vec.downcast_ref::<Vec<bool>>() {
            return Self::BOOL(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<F8E4M3>>() {
            return Self::F8E4M3(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<F8E5M2>>() {
            return Self::F8E5M2(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<bf16>>() {
            return Self::BF16(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<f16>>() {
            return Self::F16(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<f32>>() {
            return Self::F32(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<f64>>() {
            return Self::F64(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<u8>>() {
            return Self::U8(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<u16>>() {
            return Self::U16(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<u32>>() {
            return Self::U32(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<u64>>() {
            return Self::U64(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<i8>>() {
            return Self::I8(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<i16>>() {
            return Self::I16(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<i32>>() {
            return Self::I32(v.clone());
        }
        if let Some(v) = any_vec.downcast_ref::<Vec<i64>>() {
            return Self::I64(v.clone());
        }

        panic!("Unsupported vector type for CpuStorage");
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::BOOL(data) => {
                let mut bytes = Vec::with_capacity(data.len());
                for &b in data {
                    bytes.push(if b { 1u8 } else { 0u8 });
                }
                bytes
            },
            Self::F8E4M3(data) => {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for &f in data {
                    bytes.extend_from_slice(&f32::from(f).to_le_bytes());
                }
                bytes
            },
            Self::F8E5M2(data) => {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for &f in data {
                    bytes.extend_from_slice(&f32::from(f).to_le_bytes());
                }
                bytes
            },
            Self::BF16(data) => {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for &f in data {
                    bytes.extend_from_slice(&f32::from(f).to_le_bytes());
                }
                bytes
            },
            Self::F16(data) => {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for &f in data {
                    bytes.extend_from_slice(&f32::from(f).to_le_bytes());
                }
                bytes
            },
            Self::F32(data) => {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for &f in data {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                bytes
            },
            Self::F64(data) => {
                let mut bytes = Vec::with_capacity(data.len() * 8);
                for &f in data {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                bytes
            },
            Self::U8(data) => data.clone(),
            Self::U16(data) => {
                let mut bytes = Vec::with_capacity(data.len() * 2);
                for &n in data {
                    bytes.extend_from_slice(&n.to_le_bytes());
                }
                bytes
            },
            Self::U32(data) => {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for &n in data {
                    bytes.extend_from_slice(&n.to_le_bytes());
                }
                bytes
            },
            Self::U64(data) => {
                let mut bytes = Vec::with_capacity(data.len() * 8);
                for &n in data {
                    bytes.extend_from_slice(&n.to_le_bytes());
                }
                bytes
            },
            Self::I8(data) => {
                let mut bytes = Vec::with_capacity(data.len());
                for &n in data {
                    bytes.push(n as u8);
                }
                bytes
            },
            Self::I16(data) => {
                let mut bytes = Vec::with_capacity(data.len() * 2);
                for &n in data {
                    bytes.extend_from_slice(&n.to_le_bytes());
                }
                bytes
            },
            Self::I32(data) => {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for &n in data {
                    bytes.extend_from_slice(&n.to_le_bytes());
                }
                bytes
            },
            Self::I64(data) => {
                let mut bytes = Vec::with_capacity(data.len() * 8);
                for &n in data {
                    bytes.extend_from_slice(&n.to_le_bytes());
                }
                bytes
            },
        }
    }
}

impl HoduStorageT for CpuStorage {
    type HoduDevice = CpuDevice;

    fn get_dtype(&self) -> DType {
        match self {
            Self::BOOL(_) => DType::BOOL,
            Self::F8E4M3(_) => DType::F8E4M3,
            Self::F8E5M2(_) => DType::F8E5M2,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::U8(_) => DType::U8,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::U64(_) => DType::U64,
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
        }
    }

    fn get_device(&self) -> Device {
        Device::CPU
    }

    fn get_hodu_device(&self) -> CpuDevice {
        CpuDevice
    }

    fn to_cpu_storage(&self) -> HoduResult<CpuStorage> {
        Ok(self.clone())
    }

    fn const_set(&mut self, scalar: Scalar, layout: &Layout) -> HoduResult<()> {
        fn set_values<T: Copy>(data: &mut [T], layout: &Layout, value: T) {
            let shape = layout.get_shape();
            let strides = layout.get_strides();
            let offset = layout.get_offset();

            if strides == Layout::compute_strides(shape) && offset == 0 {
                data.fill(value);
            } else {
                let mut indices = vec![0; shape.len()];
                loop {
                    let mut flat_index = offset;
                    for (i, &idx) in indices.iter().enumerate() {
                        flat_index += idx * strides[i];
                    }
                    if flat_index < data.len() {
                        data[flat_index] = value;
                    }

                    let mut carry = 1;
                    for i in (0..indices.len()).rev() {
                        indices[i] += carry;
                        if indices[i] < shape[i] {
                            carry = 0;
                            break;
                        }
                        indices[i] = 0;
                    }
                    if carry == 1 {
                        break;
                    }
                }
            }
        }

        let expected_dtype = self.get_dtype();
        match (self, scalar) {
            (Self::BOOL(data), Scalar::BOOL(v)) => set_values(data, layout, v),
            (Self::F8E4M3(data), Scalar::F8E4M3(v)) => set_values(data, layout, v),
            (Self::F8E5M2(data), Scalar::F8E5M2(v)) => set_values(data, layout, v),
            (Self::BF16(data), Scalar::BF16(v)) => set_values(data, layout, v),
            (Self::F16(data), Scalar::F16(v)) => set_values(data, layout, v),
            (Self::F32(data), Scalar::F32(v)) => set_values(data, layout, v),
            (Self::F64(data), Scalar::F64(v)) => set_values(data, layout, v),
            (Self::U8(data), Scalar::U8(v)) => set_values(data, layout, v),
            (Self::U16(data), Scalar::U16(v)) => set_values(data, layout, v),
            (Self::U32(data), Scalar::U32(v)) => set_values(data, layout, v),
            (Self::U64(data), Scalar::U64(v)) => set_values(data, layout, v),
            (Self::I8(data), Scalar::I8(v)) => set_values(data, layout, v),
            (Self::I16(data), Scalar::I16(v)) => set_values(data, layout, v),
            (Self::I32(data), Scalar::I32(v)) => set_values(data, layout, v),
            (Self::I64(data), Scalar::I64(v)) => set_values(data, layout, v),
            _ => {
                return Err(HoduError::DTypeMismatch {
                    expected: expected_dtype,
                    got: scalar.get_dtype(),
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
        match (self, rhs_storage) {
            (Self::BOOL(lhs_storage), Self::BOOL(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::bool);
                Ok(Self::BOOL(data))
            },
            (Self::F8E4M3(lhs_storage), Self::F8E4M3(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::f8e4m3);
                Ok(Self::F8E4M3(data))
            },
            (Self::F8E5M2(lhs_storage), Self::F8E5M2(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::f8e5m2);
                Ok(Self::F8E5M2(data))
            },
            (Self::BF16(lhs_storage), Self::BF16(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::bf16);
                Ok(Self::BF16(data))
            },
            (Self::F16(lhs_storage), Self::F16(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::f16);
                Ok(Self::F16(data))
            },
            (Self::F32(lhs_storage), Self::F32(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::f32);
                Ok(Self::F32(data))
            },
            (Self::F64(lhs_storage), Self::F64(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::f64);
                Ok(Self::F64(data))
            },
            (Self::U8(lhs_storage), Self::U8(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::u8);
                Ok(Self::U8(data))
            },
            (Self::U16(lhs_storage), Self::U16(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::u16);
                Ok(Self::U16(data))
            },
            (Self::U32(lhs_storage), Self::U32(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::u32);
                Ok(Self::U32(data))
            },
            (Self::U64(lhs_storage), Self::U64(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::u64);
                Ok(Self::U64(data))
            },
            (Self::I8(lhs_storage), Self::I8(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::i8);
                Ok(Self::I8(data))
            },
            (Self::I16(lhs_storage), Self::I16(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::i16);
                Ok(Self::I16(data))
            },
            (Self::I32(lhs_storage), Self::I32(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::i32);
                Ok(Self::I32(data))
            },
            (Self::I64(lhs_storage), Self::I64(rhs_storage)) => {
                let data = binary_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::i64);
                Ok(Self::I64(data))
            },
            _ => Err(HoduError::DTypeConflictInOp {
                left: self.get_dtype(),
                right: rhs_storage.get_dtype(),
                op: B::NAME.to_string(),
            }),
        }
    }

    fn binary_logical_impl<B: BinaryLogicalOpT>(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> HoduResult<CpuStorage> {
        match (self, rhs_storage) {
            (Self::BOOL(lhs_storage), Self::BOOL(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::bool);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::F8E4M3(lhs_storage), Self::F8E4M3(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::f8e4m3);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::F8E5M2(lhs_storage), Self::F8E5M2(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::f8e5m2);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::BF16(lhs_storage), Self::BF16(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::bf16);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::F16(lhs_storage), Self::F16(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::f16);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::F32(lhs_storage), Self::F32(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::f32);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::F64(lhs_storage), Self::F64(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::f64);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::U8(lhs_storage), Self::U8(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::u8);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::U16(lhs_storage), Self::U16(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::u16);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::U32(lhs_storage), Self::U32(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::u32);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::U64(lhs_storage), Self::U64(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::u64);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::I8(lhs_storage), Self::I8(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::i8);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::I16(lhs_storage), Self::I16(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::i16);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::I32(lhs_storage), Self::I32(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::i32);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::I64(lhs_storage), Self::I64(rhs_storage)) => {
                let data = binary_logical_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, B::i64);
                Ok(CpuStorage::BOOL(data))
            },
            _ => Err(HoduError::DTypeConflictInOp {
                left: self.get_dtype(),
                right: rhs_storage.get_dtype(),
                op: B::NAME.to_string(),
            }),
        }
    }

    fn cmp_impl<C: CmpOpT>(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> HoduResult<CpuStorage> {
        match (self, rhs_storage) {
            (Self::BOOL(lhs_storage), Self::BOOL(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::bool);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::F8E4M3(lhs_storage), Self::F8E4M3(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::f8e4m3);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::F8E5M2(lhs_storage), Self::F8E5M2(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::f8e5m2);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::BF16(lhs_storage), Self::BF16(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::bf16);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::F16(lhs_storage), Self::F16(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::f16);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::F32(lhs_storage), Self::F32(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::f32);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::F64(lhs_storage), Self::F64(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::f64);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::U8(lhs_storage), Self::U8(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::u8);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::U16(lhs_storage), Self::U16(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::u16);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::U32(lhs_storage), Self::U32(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::u32);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::U64(lhs_storage), Self::U64(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::u64);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::I8(lhs_storage), Self::I8(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::i8);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::I16(lhs_storage), Self::I16(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::i16);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::I32(lhs_storage), Self::I32(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::i32);
                Ok(CpuStorage::BOOL(data))
            },
            (Self::I64(lhs_storage), Self::I64(rhs_storage)) => {
                let data = cmp_map(lhs_storage, rhs_storage, lhs_layout, rhs_layout, C::i64);
                Ok(CpuStorage::BOOL(data))
            },
            _ => Err(HoduError::DTypeConflictInOp {
                left: self.get_dtype(),
                right: rhs_storage.get_dtype(),
                op: C::NAME.to_string(),
            }),
        }
    }

    fn unary_scalar_impl<U: UnaryScalarOpT>(&self, layout: &Layout, scalar: Scalar) -> HoduResult<Self> {
        match self {
            Self::BOOL(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::bool);
                Ok(Self::BOOL(data))
            },
            Self::F8E4M3(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::f8e4m3);
                Ok(Self::F8E4M3(data))
            },
            Self::F8E5M2(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::f8e5m2);
                Ok(Self::F8E5M2(data))
            },
            Self::BF16(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::bf16);
                Ok(Self::BF16(data))
            },
            Self::F16(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::f16);
                Ok(Self::F16(data))
            },
            Self::F32(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::f32);
                Ok(Self::F32(data))
            },
            Self::F64(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::f64);
                Ok(Self::F64(data))
            },
            Self::U8(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::u8);
                Ok(Self::U8(data))
            },
            Self::U16(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::u16);
                Ok(Self::U16(data))
            },
            Self::U32(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::u32);
                Ok(Self::U32(data))
            },
            Self::U64(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::u64);
                Ok(Self::U64(data))
            },
            Self::I8(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::i8);
                Ok(Self::I8(data))
            },
            Self::I16(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::i16);
                Ok(Self::I16(data))
            },
            Self::I32(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::i32);
                Ok(Self::I32(data))
            },
            Self::I64(storage) => {
                let data = unary_scalar_map(storage, layout, scalar, U::i64);
                Ok(Self::I64(data))
            },
        }
    }

    fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> HoduResult<Self> {
        match self {
            Self::BOOL(storage) => {
                let data = unary_map(storage, layout, U::bool);
                Ok(Self::BOOL(data))
            },
            Self::F8E4M3(storage) => {
                let data = unary_map(storage, layout, U::f8e4m3);
                Ok(Self::F8E4M3(data))
            },
            Self::F8E5M2(storage) => {
                let data = unary_map(storage, layout, U::f8e5m2);
                Ok(Self::F8E5M2(data))
            },
            Self::BF16(storage) => {
                let data = unary_map(storage, layout, U::bf16);
                Ok(Self::BF16(data))
            },
            Self::F16(storage) => {
                let data = unary_map(storage, layout, U::f16);
                Ok(Self::F16(data))
            },
            Self::F32(storage) => {
                let data = unary_map(storage, layout, U::f32);
                Ok(Self::F32(data))
            },
            Self::F64(storage) => {
                let data = unary_map(storage, layout, U::f64);
                Ok(Self::F64(data))
            },
            Self::U8(storage) => {
                let data = unary_map(storage, layout, U::u8);
                Ok(Self::U8(data))
            },
            Self::U16(storage) => {
                let data = unary_map(storage, layout, U::u16);
                Ok(Self::U16(data))
            },
            Self::U32(storage) => {
                let data = unary_map(storage, layout, U::u32);
                Ok(Self::U32(data))
            },
            Self::U64(storage) => {
                let data = unary_map(storage, layout, U::u64);
                Ok(Self::U64(data))
            },
            Self::I8(storage) => {
                let data = unary_map(storage, layout, U::i8);
                Ok(Self::I8(data))
            },
            Self::I16(storage) => {
                let data = unary_map(storage, layout, U::i16);
                Ok(Self::I16(data))
            },
            Self::I32(storage) => {
                let data = unary_map(storage, layout, U::i32);
                Ok(Self::I32(data))
            },
            Self::I64(storage) => {
                let data = unary_map(storage, layout, U::i64);
                Ok(Self::I64(data))
            },
        }
    }

    fn unary_logical_impl<U: UnaryLogicalOpT>(&self, layout: &Layout) -> HoduResult<CpuStorage> {
        match self {
            Self::BOOL(storage) => {
                let data = unary_logical_map(storage, layout, U::bool);
                Ok(CpuStorage::BOOL(data))
            },
            Self::F8E4M3(storage) => {
                let data = unary_logical_map(storage, layout, U::f8e4m3);
                Ok(CpuStorage::BOOL(data))
            },
            Self::F8E5M2(storage) => {
                let data = unary_logical_map(storage, layout, U::f8e5m2);
                Ok(CpuStorage::BOOL(data))
            },
            Self::BF16(storage) => {
                let data = unary_logical_map(storage, layout, U::bf16);
                Ok(CpuStorage::BOOL(data))
            },
            Self::F16(storage) => {
                let data = unary_logical_map(storage, layout, U::f16);
                Ok(CpuStorage::BOOL(data))
            },
            Self::F32(storage) => {
                let data = unary_logical_map(storage, layout, U::f32);
                Ok(CpuStorage::BOOL(data))
            },
            Self::F64(storage) => {
                let data = unary_logical_map(storage, layout, U::f64);
                Ok(CpuStorage::BOOL(data))
            },
            Self::U8(storage) => {
                let data = unary_logical_map(storage, layout, U::u8);
                Ok(CpuStorage::BOOL(data))
            },
            Self::U16(storage) => {
                let data = unary_logical_map(storage, layout, U::u16);
                Ok(CpuStorage::BOOL(data))
            },
            Self::U32(storage) => {
                let data = unary_logical_map(storage, layout, U::u32);
                Ok(CpuStorage::BOOL(data))
            },
            Self::U64(storage) => {
                let data = unary_logical_map(storage, layout, U::u64);
                Ok(CpuStorage::BOOL(data))
            },
            Self::I8(storage) => {
                let data = unary_logical_map(storage, layout, U::i8);
                Ok(CpuStorage::BOOL(data))
            },
            Self::I16(storage) => {
                let data = unary_logical_map(storage, layout, U::i16);
                Ok(CpuStorage::BOOL(data))
            },
            Self::I32(storage) => {
                let data = unary_logical_map(storage, layout, U::i32);
                Ok(CpuStorage::BOOL(data))
            },
            Self::I64(storage) => {
                let data = unary_logical_map(storage, layout, U::i64);
                Ok(CpuStorage::BOOL(data))
            },
        }
    }

    fn cmp_scalar_impl<C: CmpScalarOpT>(&self, layout: &Layout, scalar: Scalar) -> HoduResult<CpuStorage> {
        match self {
            Self::BOOL(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::bool);
                Ok(CpuStorage::BOOL(data))
            },
            Self::F8E4M3(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::f8e4m3);
                Ok(CpuStorage::BOOL(data))
            },
            Self::F8E5M2(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::f8e5m2);
                Ok(CpuStorage::BOOL(data))
            },
            Self::BF16(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::bf16);
                Ok(CpuStorage::BOOL(data))
            },
            Self::F16(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::f16);
                Ok(CpuStorage::BOOL(data))
            },
            Self::F32(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::f32);
                Ok(CpuStorage::BOOL(data))
            },
            Self::F64(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::f64);
                Ok(CpuStorage::BOOL(data))
            },
            Self::U8(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::u8);
                Ok(CpuStorage::BOOL(data))
            },
            Self::U16(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::u16);
                Ok(CpuStorage::BOOL(data))
            },
            Self::U32(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::u32);
                Ok(CpuStorage::BOOL(data))
            },
            Self::U64(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::u64);
                Ok(CpuStorage::BOOL(data))
            },
            Self::I8(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::i8);
                Ok(CpuStorage::BOOL(data))
            },
            Self::I16(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::i16);
                Ok(CpuStorage::BOOL(data))
            },
            Self::I32(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::i32);
                Ok(CpuStorage::BOOL(data))
            },
            Self::I64(storage) => {
                let data = cmp_scalar_map(storage, layout, scalar, C::i64);
                Ok(CpuStorage::BOOL(data))
            },
        }
    }

    fn matmul(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> HoduResult<Self> {
        match (self, rhs_storage) {
            (Self::F8E4M3(lhs_data), Self::F8E4M3(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::F8E4M3(result_data))
            },
            (Self::F8E5M2(lhs_data), Self::F8E5M2(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::F8E5M2(result_data))
            },
            (Self::BF16(lhs_data), Self::BF16(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::BF16(result_data))
            },
            (Self::F16(lhs_data), Self::F16(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::F16(result_data))
            },
            (Self::F32(lhs_data), Self::F32(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::F32(result_data))
            },
            (Self::F64(lhs_data), Self::F64(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::F64(result_data))
            },
            (Self::U8(lhs_data), Self::U8(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::U8(result_data))
            },
            (Self::U16(lhs_data), Self::U16(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::U16(result_data))
            },
            (Self::U32(lhs_data), Self::U32(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::U32(result_data))
            },
            (Self::U64(lhs_data), Self::U64(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::U64(result_data))
            },
            (Self::I8(lhs_data), Self::I8(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::I8(result_data))
            },
            (Self::I16(lhs_data), Self::I16(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::I16(result_data))
            },
            (Self::I32(lhs_data), Self::I32(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::I32(result_data))
            },
            (Self::I64(lhs_data), Self::I64(rhs_data)) => {
                let result_data = matmul_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::I64(result_data))
            },
            _ => Err(HoduError::DTypeConflictInOp {
                left: self.get_dtype(),
                right: rhs_storage.get_dtype(),
                op: "matmul".to_string(),
            }),
        }
    }

    fn dot(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> HoduResult<Self> {
        match (self, rhs_storage) {
            (Self::F8E4M3(lhs_data), Self::F8E4M3(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::F8E4M3(result_data))
            },
            (Self::F8E5M2(lhs_data), Self::F8E5M2(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::F8E5M2(result_data))
            },
            (Self::BF16(lhs_data), Self::BF16(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::BF16(result_data))
            },
            (Self::F16(lhs_data), Self::F16(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::F16(result_data))
            },
            (Self::F32(lhs_data), Self::F32(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::F32(result_data))
            },
            (Self::F64(lhs_data), Self::F64(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::F64(result_data))
            },
            (Self::U8(lhs_data), Self::U8(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::U8(result_data))
            },
            (Self::U16(lhs_data), Self::U16(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::U16(result_data))
            },
            (Self::U32(lhs_data), Self::U32(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::U32(result_data))
            },
            (Self::U64(lhs_data), Self::U64(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::U64(result_data))
            },
            (Self::I8(lhs_data), Self::I8(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::I8(result_data))
            },
            (Self::I16(lhs_data), Self::I16(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::I16(result_data))
            },
            (Self::I32(lhs_data), Self::I32(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::I32(result_data))
            },
            (Self::I64(lhs_data), Self::I64(rhs_data)) => {
                let result_data = dot_map(lhs_data, rhs_data, lhs_layout, rhs_layout)?;
                Ok(Self::I64(result_data))
            },
            _ => Err(HoduError::DTypeConflictInOp {
                left: self.get_dtype(),
                right: rhs_storage.get_dtype(),
                op: "dot".to_string(),
            }),
        }
    }

    fn reduce(&self, reduce_op: ReduceOp, layout: &Layout, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        match self {
            Self::F8E4M3(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Mean => reduce_mean(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    ReduceOp::Std => reduce_std(storage, layout, dims, keep_dim, false)?,
                    ReduceOp::Var => reduce_var(storage, layout, dims, keep_dim, false)?,
                    ReduceOp::Norm => reduce_norm(storage, layout, dims, keep_dim)?,
                };
                Ok(Self::F8E4M3(data))
            },
            Self::F8E5M2(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Mean => reduce_mean(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    ReduceOp::Std => reduce_std(storage, layout, dims, keep_dim, false)?,
                    ReduceOp::Var => reduce_var(storage, layout, dims, keep_dim, false)?,
                    ReduceOp::Norm => reduce_norm(storage, layout, dims, keep_dim)?,
                };
                Ok(Self::F8E5M2(data))
            },
            Self::BF16(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Mean => reduce_mean(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    ReduceOp::Std => reduce_std(storage, layout, dims, keep_dim, false)?,
                    ReduceOp::Var => reduce_var(storage, layout, dims, keep_dim, false)?,
                    ReduceOp::Norm => reduce_norm(storage, layout, dims, keep_dim)?,
                };
                Ok(Self::BF16(data))
            },
            Self::F16(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Mean => reduce_mean(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    ReduceOp::Std => reduce_std(storage, layout, dims, keep_dim, false)?,
                    ReduceOp::Var => reduce_var(storage, layout, dims, keep_dim, false)?,
                    ReduceOp::Norm => reduce_norm(storage, layout, dims, keep_dim)?,
                };
                Ok(Self::F16(data))
            },
            Self::F32(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Mean => reduce_mean(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    ReduceOp::Std => reduce_std(storage, layout, dims, keep_dim, false)?,
                    ReduceOp::Var => reduce_var(storage, layout, dims, keep_dim, false)?,
                    ReduceOp::Norm => reduce_norm(storage, layout, dims, keep_dim)?,
                };
                Ok(Self::F32(data))
            },
            Self::F64(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Mean => reduce_mean(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    ReduceOp::Std => reduce_std(storage, layout, dims, keep_dim, false)?,
                    ReduceOp::Var => reduce_var(storage, layout, dims, keep_dim, false)?,
                    ReduceOp::Norm => reduce_norm(storage, layout, dims, keep_dim)?,
                };
                Ok(Self::F64(data))
            },
            Self::U8(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    _ => {
                        return Err(HoduError::UnsupportedDType {
                            dtype: self.get_dtype(),
                            op: reduce_op.to_string(),
                        })
                    },
                };
                Ok(Self::U8(data))
            },
            Self::U16(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    _ => {
                        return Err(HoduError::UnsupportedDType {
                            dtype: self.get_dtype(),
                            op: reduce_op.to_string(),
                        })
                    },
                };
                Ok(Self::U16(data))
            },
            Self::U32(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    _ => {
                        return Err(HoduError::UnsupportedDType {
                            dtype: self.get_dtype(),
                            op: reduce_op.to_string(),
                        })
                    },
                };
                Ok(Self::U32(data))
            },
            Self::U64(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    _ => {
                        return Err(HoduError::UnsupportedDType {
                            dtype: self.get_dtype(),
                            op: reduce_op.to_string(),
                        })
                    },
                };
                Ok(Self::U64(data))
            },
            Self::I8(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    _ => {
                        return Err(HoduError::UnsupportedDType {
                            dtype: self.get_dtype(),
                            op: reduce_op.to_string(),
                        })
                    },
                };
                Ok(Self::I8(data))
            },
            Self::I16(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    _ => {
                        return Err(HoduError::UnsupportedDType {
                            dtype: self.get_dtype(),
                            op: reduce_op.to_string(),
                        })
                    },
                };
                Ok(Self::I16(data))
            },
            Self::I32(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    _ => {
                        return Err(HoduError::UnsupportedDType {
                            dtype: self.get_dtype(),
                            op: reduce_op.to_string(),
                        })
                    },
                };
                Ok(Self::I32(data))
            },
            Self::I64(storage) => {
                let (data, _shape) = match reduce_op {
                    ReduceOp::Sum => reduce_sum(storage, layout, dims, keep_dim)?,
                    ReduceOp::Max => reduce_max(storage, layout, dims, keep_dim)?,
                    ReduceOp::Min => reduce_min(storage, layout, dims, keep_dim)?,
                    ReduceOp::Prod => reduce_prod(storage, layout, dims, keep_dim)?,
                    _ => {
                        return Err(HoduError::UnsupportedDType {
                            dtype: self.get_dtype(),
                            op: reduce_op.to_string(),
                        })
                    },
                };
                Ok(Self::I64(data))
            },
            _ => Err(HoduError::UnsupportedDType {
                dtype: self.get_dtype(),
                op: "reduce".to_string(),
            }),
        }
    }

    fn concat(&self, others: &[&Self], layouts: &[&Layout], dim: usize) -> HoduResult<Self> {
        // Validate all storages have the same dtype
        let first_dtype = self.get_dtype();
        for other in others {
            if other.get_dtype() != first_dtype {
                return Err(HoduError::DTypeConflictInOp {
                    left: first_dtype,
                    right: other.get_dtype(),
                    op: "concat".to_string(),
                });
            }
        }

        // All layouts include self's layout at index 0
        let first_layout = layouts[0];
        let first_shape = first_layout.get_shape();
        let ndim = first_shape.len();

        if dim >= ndim {
            return Err(HoduError::IncompatibleShapes {
                lhs: first_shape.to_vec(),
                rhs: vec![],
                op: format!(
                    "concat - dimension {} out of range for {}-dimensional tensor",
                    dim, ndim
                ),
            });
        }

        // Verify all tensors have the same shape except at concat dimension
        for (_i, layout) in layouts.iter().enumerate().skip(1) {
            let shape = layout.get_shape();
            if shape.len() != ndim {
                return Err(HoduError::IncompatibleShapes {
                    lhs: first_shape.to_vec(),
                    rhs: shape.to_vec(),
                    op: "concat - all tensors must have the same number of dimensions".to_string(),
                });
            }
            for (j, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if j != dim && s1 != s2 {
                    return Err(HoduError::IncompatibleShapes {
                        lhs: first_shape.to_vec(),
                        rhs: shape.to_vec(),
                        op: format!("concat - dimension {} must match (got {} vs {})", j, s1, s2),
                    });
                }
            }
        }

        // Calculate output shape
        let mut output_shape = first_shape.to_vec();
        output_shape[dim] = layouts.iter().map(|l| l.get_shape()[dim]).sum();

        macro_rules! concat_impl {
            ($first_storage:expr, $other_storages:expr, $dtype_variant:ident) => {{
                let other_storage_slices: Vec<&[_]> = $other_storages
                    .iter()
                    .map(|s| match s {
                        Self::$dtype_variant(storage) => storage.as_slice(),
                        _ => unreachable!(),
                    })
                    .collect();

                let result = concat_map($first_storage, &other_storage_slices, layouts, dim, &output_shape);

                Self::$dtype_variant(result)
            }};
        }

        let result = match (self, others) {
            (Self::BOOL(first), _) => concat_impl!(first, others, BOOL),
            (Self::F8E4M3(first), _) => concat_impl!(first, others, F8E4M3),
            (Self::F8E5M2(first), _) => concat_impl!(first, others, F8E5M2),
            (Self::BF16(first), _) => concat_impl!(first, others, BF16),
            (Self::F16(first), _) => concat_impl!(first, others, F16),
            (Self::F32(first), _) => concat_impl!(first, others, F32),
            (Self::F64(first), _) => concat_impl!(first, others, F64),
            (Self::U8(first), _) => concat_impl!(first, others, U8),
            (Self::U16(first), _) => concat_impl!(first, others, U16),
            (Self::U32(first), _) => concat_impl!(first, others, U32),
            (Self::U64(first), _) => concat_impl!(first, others, U64),
            (Self::I8(first), _) => concat_impl!(first, others, I8),
            (Self::I16(first), _) => concat_impl!(first, others, I16),
            (Self::I32(first), _) => concat_impl!(first, others, I32),
            (Self::I64(first), _) => concat_impl!(first, others, I64),
        };

        Ok(result)
    }

    fn split(&self, layout: &Layout, dim: usize, sizes: &[usize]) -> HoduResult<Vec<Self>> {
        let shape = layout.get_shape();
        let ndim = shape.len();

        if dim >= ndim {
            return Err(HoduError::IncompatibleShapes {
                lhs: shape.to_vec(),
                rhs: vec![],
                op: format!("split - dimension {} out of range for {}-dimensional tensor", dim, ndim),
            });
        }

        // Verify sizes sum to dimension size
        let total_size: usize = sizes.iter().sum();
        if total_size != shape[dim] {
            return Err(HoduError::IncompatibleShapes {
                lhs: vec![shape[dim]],
                rhs: vec![total_size],
                op: format!(
                    "split - sizes must sum to dimension size (got {} vs {})",
                    total_size, shape[dim]
                ),
            });
        }

        macro_rules! split_impl {
            ($storage:expr, $dtype_variant:ident) => {{
                let results = split_map($storage, layout, dim, sizes);
                results.into_iter().map(|r| Self::$dtype_variant(r)).collect()
            }};
        }

        let results = match self {
            Self::BOOL(storage) => split_impl!(storage, BOOL),
            Self::F8E4M3(storage) => split_impl!(storage, F8E4M3),
            Self::F8E5M2(storage) => split_impl!(storage, F8E5M2),
            Self::BF16(storage) => split_impl!(storage, BF16),
            Self::F16(storage) => split_impl!(storage, F16),
            Self::F32(storage) => split_impl!(storage, F32),
            Self::F64(storage) => split_impl!(storage, F64),
            Self::U8(storage) => split_impl!(storage, U8),
            Self::U16(storage) => split_impl!(storage, U16),
            Self::U32(storage) => split_impl!(storage, U32),
            Self::U64(storage) => split_impl!(storage, U64),
            Self::I8(storage) => split_impl!(storage, I8),
            Self::I16(storage) => split_impl!(storage, I16),
            Self::I32(storage) => split_impl!(storage, I32),
            Self::I64(storage) => split_impl!(storage, I64),
        };

        Ok(results)
    }

    fn to_dtype(&self, target_dtype: DType) -> HoduResult<Self> {
        if self.get_dtype() == target_dtype {
            return Ok(self.clone());
        }

        macro_rules! convert_storage {
            ($storage:expr, $convert_fn:expr) => {{
                let layout = Layout::from_shape(&[$storage.len()]);
                let data = unary_map($storage, &layout, $convert_fn);
                data
            }};
        }

        let result = match (self, target_dtype) {
            // BOOL conversions
            (Self::BOOL(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v)),
            (Self::BOOL(storage), DType::F8E4M3) => Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(if v {
                1.0f32
            } else {
                0.0f32
            }))),
            (Self::BOOL(storage), DType::F8E5M2) => Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(if v {
                1.0f32
            } else {
                0.0f32
            }))),
            (Self::BOOL(storage), DType::BF16) => Self::BF16(convert_storage!(storage, |v| bf16::from_f32(if v {
                1.0f32
            } else {
                0.0f32
            }))),
            (Self::BOOL(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| f16::from_f32(if v {
                1.0f32
            } else {
                0.0f32
            }))),
            (Self::BOOL(storage), DType::F32) => {
                Self::F32(convert_storage!(storage, |v| if v { 1.0f32 } else { 0.0f32 }))
            },
            (Self::BOOL(storage), DType::F64) => {
                Self::F64(convert_storage!(storage, |v| if v { 1.0f64 } else { 0.0f64 }))
            },
            (Self::BOOL(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| if v { 1u8 } else { 0u8 })),
            (Self::BOOL(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| if v { 1u16 } else { 0u16 })),
            (Self::BOOL(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| if v { 1u32 } else { 0u32 })),
            (Self::BOOL(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| if v { 1u64 } else { 0u64 })),
            (Self::BOOL(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| if v { 1i8 } else { 0i8 })),
            (Self::BOOL(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| if v { 1i16 } else { 0i16 })),
            (Self::BOOL(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| if v { 1i32 } else { 0i32 })),
            (Self::BOOL(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| if v { 1i64 } else { 0i64 })),

            // F8E4M3 conversions
            (Self::F8E4M3(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v.to_f32() != 0.0)),
            (Self::F8E4M3(storage), DType::F8E4M3) => Self::F8E4M3(convert_storage!(storage, |v| v)),
            (Self::F8E4M3(storage), DType::F8E5M2) => {
                Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(v.to_f32())))
            },
            (Self::F8E4M3(storage), DType::BF16) => {
                Self::BF16(convert_storage!(storage, |v| bf16::from_f32(v.to_f32())))
            },
            (Self::F8E4M3(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| f16::from_f32(v.to_f32()))),
            (Self::F8E4M3(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v.to_f32())),
            (Self::F8E4M3(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v.to_f64())),
            (Self::F8E4M3(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v.to_f32() as u8)),
            (Self::F8E4M3(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v.to_f32() as u16)),
            (Self::F8E4M3(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v.to_f32() as u32)),
            (Self::F8E4M3(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v.to_f32() as u64)),
            (Self::F8E4M3(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v.to_f32() as i8)),
            (Self::F8E4M3(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v.to_f32() as i16)),
            (Self::F8E4M3(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v.to_f32() as i32)),
            (Self::F8E4M3(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v.to_f32() as i64)),

            // F8E5M2 conversions
            (Self::F8E5M2(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v.to_f32() != 0.0)),
            (Self::F8E5M2(storage), DType::F8E4M3) => {
                Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(v.to_f32())))
            },
            (Self::F8E5M2(storage), DType::F8E5M2) => Self::F8E5M2(convert_storage!(storage, |v| v)),
            (Self::F8E5M2(storage), DType::BF16) => {
                Self::BF16(convert_storage!(storage, |v| bf16::from_f32(v.to_f32())))
            },
            (Self::F8E5M2(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| f16::from_f32(v.to_f32()))),
            (Self::F8E5M2(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v.to_f32())),
            (Self::F8E5M2(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v.to_f64())),
            (Self::F8E5M2(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v.to_f32() as u8)),
            (Self::F8E5M2(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v.to_f32() as u16)),
            (Self::F8E5M2(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v.to_f32() as u32)),
            (Self::F8E5M2(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v.to_f32() as u64)),
            (Self::F8E5M2(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v.to_f32() as i8)),
            (Self::F8E5M2(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v.to_f32() as i16)),
            (Self::F8E5M2(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v.to_f32() as i32)),
            (Self::F8E5M2(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v.to_f32() as i64)),

            // BF16 conversions
            (Self::BF16(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v.to_f32() != 0.0)),
            (Self::BF16(storage), DType::F8E4M3) => {
                Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(v.to_f32())))
            },
            (Self::BF16(storage), DType::F8E5M2) => {
                Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(v.to_f32())))
            },
            (Self::BF16(storage), DType::BF16) => Self::BF16(convert_storage!(storage, |v| v)),
            (Self::BF16(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| f16::from_f32(v.to_f32()))),
            (Self::BF16(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v.to_f32())),
            (Self::BF16(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v.to_f64())),
            (Self::BF16(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v.to_f32() as u8)),
            (Self::BF16(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v.to_f32() as u16)),
            (Self::BF16(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v.to_f32() as u32)),
            (Self::BF16(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v.to_f32() as u64)),
            (Self::BF16(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v.to_f32() as i8)),
            (Self::BF16(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v.to_f32() as i16)),
            (Self::BF16(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v.to_f32() as i32)),
            (Self::BF16(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v.to_f32() as i64)),

            // F16 conversions
            (Self::F16(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v.to_f32() != 0.0)),
            (Self::F16(storage), DType::F8E4M3) => {
                Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(v.to_f32())))
            },
            (Self::F16(storage), DType::F8E5M2) => {
                Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(v.to_f32())))
            },
            (Self::F16(storage), DType::BF16) => Self::BF16(convert_storage!(storage, |v| bf16::from_f32(v.to_f32()))),
            (Self::F16(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| v)),
            (Self::F16(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v.to_f32())),
            (Self::F16(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v.to_f64())),
            (Self::F16(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v.to_f32() as u8)),
            (Self::F16(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v.to_f32() as u16)),
            (Self::F16(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v.to_f32() as u32)),
            (Self::F16(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v.to_f32() as u64)),
            (Self::F16(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v.to_f32() as i8)),
            (Self::F16(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v.to_f32() as i16)),
            (Self::F16(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v.to_f32() as i32)),
            (Self::F16(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v.to_f32() as i64)),

            // F32 conversions
            (Self::F32(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v != 0.0)),
            (Self::F32(storage), DType::F8E4M3) => Self::F8E4M3(convert_storage!(storage, F8E4M3::from)),
            (Self::F32(storage), DType::F8E5M2) => Self::F8E5M2(convert_storage!(storage, F8E5M2::from)),
            (Self::F32(storage), DType::BF16) => Self::BF16(convert_storage!(storage, bf16::from_f32)),
            (Self::F32(storage), DType::F16) => Self::F16(convert_storage!(storage, f16::from_f32)),
            (Self::F32(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v)),
            (Self::F32(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v as f64)),
            (Self::F32(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v as u8)),
            (Self::F32(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v as u16)),
            (Self::F32(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v as u32)),
            (Self::F32(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v as u64)),
            (Self::F32(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v as i8)),
            (Self::F32(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v as i16)),
            (Self::F32(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v as i32)),
            (Self::F32(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v as i64)),

            // F64 conversions
            (Self::F64(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v != 0.0)),
            (Self::F64(storage), DType::F8E4M3) => Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(v as f32))),
            (Self::F64(storage), DType::F8E5M2) => Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(v as f32))),
            (Self::F64(storage), DType::BF16) => Self::BF16(convert_storage!(storage, bf16::from_f64)),
            (Self::F64(storage), DType::F16) => Self::F16(convert_storage!(storage, f16::from_f64)),
            (Self::F64(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v as f32)),
            (Self::F64(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v)),
            (Self::F64(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v as u8)),
            (Self::F64(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v as u16)),
            (Self::F64(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v as u32)),
            (Self::F64(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v as u64)),
            (Self::F64(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v as i8)),
            (Self::F64(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v as i16)),
            (Self::F64(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v as i32)),
            (Self::F64(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v as i64)),

            // U8 conversions
            (Self::U8(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v != 0)),
            (Self::U8(storage), DType::F8E4M3) => Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(v as f32))),
            (Self::U8(storage), DType::F8E5M2) => Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(v as f32))),
            (Self::U8(storage), DType::BF16) => Self::BF16(convert_storage!(storage, |v| bf16::from_f32(v as f32))),
            (Self::U8(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| f16::from_f32(v as f32))),
            (Self::U8(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v as f32)),
            (Self::U8(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v as f64)),
            (Self::U8(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v)),
            (Self::U8(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v as u16)),
            (Self::U8(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v as u32)),
            (Self::U8(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v as u64)),
            (Self::U8(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v as i8)),
            (Self::U8(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v as i16)),
            (Self::U8(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v as i32)),
            (Self::U8(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v as i64)),

            // U16 conversions
            (Self::U16(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v != 0)),
            (Self::U16(storage), DType::F8E4M3) => Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(v as f32))),
            (Self::U16(storage), DType::F8E5M2) => Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(v as f32))),
            (Self::U16(storage), DType::BF16) => Self::BF16(convert_storage!(storage, |v| bf16::from_f32(v as f32))),
            (Self::U16(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| f16::from_f32(v as f32))),
            (Self::U16(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v as f32)),
            (Self::U16(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v as f64)),
            (Self::U16(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v as u8)),
            (Self::U16(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v)),
            (Self::U16(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v as u32)),
            (Self::U16(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v as u64)),
            (Self::U16(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v as i8)),
            (Self::U16(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v as i16)),
            (Self::U16(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v as i32)),
            (Self::U16(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v as i64)),

            // U32 conversions
            (Self::U32(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v != 0)),
            (Self::U32(storage), DType::F8E4M3) => Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(v as f32))),
            (Self::U32(storage), DType::F8E5M2) => Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(v as f32))),
            (Self::U32(storage), DType::BF16) => Self::BF16(convert_storage!(storage, |v| bf16::from_f32(v as f32))),
            (Self::U32(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| f16::from_f32(v as f32))),
            (Self::U32(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v as f32)),
            (Self::U32(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v as f64)),
            (Self::U32(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v as u8)),
            (Self::U32(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v as u16)),
            (Self::U32(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v)),
            (Self::U32(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v as u64)),
            (Self::U32(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v as i8)),
            (Self::U32(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v as i16)),
            (Self::U32(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v as i32)),
            (Self::U32(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v as i64)),

            // U64 conversions
            (Self::U64(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v != 0)),
            (Self::U64(storage), DType::F8E4M3) => Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(v as f32))),
            (Self::U64(storage), DType::F8E5M2) => Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(v as f32))),
            (Self::U64(storage), DType::BF16) => Self::BF16(convert_storage!(storage, |v| bf16::from_f32(v as f32))),
            (Self::U64(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| f16::from_f32(v as f32))),
            (Self::U64(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v as f32)),
            (Self::U64(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v as f64)),
            (Self::U64(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v as u8)),
            (Self::U64(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v as u16)),
            (Self::U64(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v as u32)),
            (Self::U64(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v)),
            (Self::U64(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v as i8)),
            (Self::U64(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v as i16)),
            (Self::U64(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v as i32)),
            (Self::U64(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v as i64)),

            // I8 conversions
            (Self::I8(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v != 0)),
            (Self::I8(storage), DType::F8E4M3) => Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(v as f32))),
            (Self::I8(storage), DType::F8E5M2) => Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(v as f32))),
            (Self::I8(storage), DType::BF16) => Self::BF16(convert_storage!(storage, |v| bf16::from_f32(v as f32))),
            (Self::I8(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| f16::from_f32(v as f32))),
            (Self::I8(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v as f32)),
            (Self::I8(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v as f64)),
            (Self::I8(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v as u8)),
            (Self::I8(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v as u16)),
            (Self::I8(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v as u32)),
            (Self::I8(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v as u64)),
            (Self::I8(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v)),
            (Self::I8(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v as i16)),
            (Self::I8(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v as i32)),
            (Self::I8(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v as i64)),

            // I16 conversions
            (Self::I16(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v != 0)),
            (Self::I16(storage), DType::F8E4M3) => Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(v as f32))),
            (Self::I16(storage), DType::F8E5M2) => Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(v as f32))),
            (Self::I16(storage), DType::BF16) => Self::BF16(convert_storage!(storage, |v| bf16::from_f32(v as f32))),
            (Self::I16(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| f16::from_f32(v as f32))),
            (Self::I16(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v as f32)),
            (Self::I16(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v as f64)),
            (Self::I16(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v as u8)),
            (Self::I16(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v as u16)),
            (Self::I16(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v as u32)),
            (Self::I16(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v as u64)),
            (Self::I16(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v as i8)),
            (Self::I16(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v)),
            (Self::I16(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v as i32)),
            (Self::I16(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v as i64)),

            // I32 conversions
            (Self::I32(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v != 0)),
            (Self::I32(storage), DType::F8E4M3) => Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(v as f32))),
            (Self::I32(storage), DType::F8E5M2) => Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(v as f32))),
            (Self::I32(storage), DType::BF16) => Self::BF16(convert_storage!(storage, |v| bf16::from_f32(v as f32))),
            (Self::I32(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| f16::from_f32(v as f32))),
            (Self::I32(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v as f32)),
            (Self::I32(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v as f64)),
            (Self::I32(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v as u8)),
            (Self::I32(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v as u16)),
            (Self::I32(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v as u32)),
            (Self::I32(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v as u64)),
            (Self::I32(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v as i8)),
            (Self::I32(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v as i16)),
            (Self::I32(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v)),
            (Self::I32(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v as i64)),

            // I64 conversions
            (Self::I64(storage), DType::BOOL) => Self::BOOL(convert_storage!(storage, |v| v != 0)),
            (Self::I64(storage), DType::F8E4M3) => Self::F8E4M3(convert_storage!(storage, |v| F8E4M3::from(v as f32))),
            (Self::I64(storage), DType::F8E5M2) => Self::F8E5M2(convert_storage!(storage, |v| F8E5M2::from(v as f32))),
            (Self::I64(storage), DType::BF16) => Self::BF16(convert_storage!(storage, |v| bf16::from_f32(v as f32))),
            (Self::I64(storage), DType::F16) => Self::F16(convert_storage!(storage, |v| f16::from_f32(v as f32))),
            (Self::I64(storage), DType::F32) => Self::F32(convert_storage!(storage, |v| v as f32)),
            (Self::I64(storage), DType::F64) => Self::F64(convert_storage!(storage, |v| v as f64)),
            (Self::I64(storage), DType::U8) => Self::U8(convert_storage!(storage, |v| v as u8)),
            (Self::I64(storage), DType::U16) => Self::U16(convert_storage!(storage, |v| v as u16)),
            (Self::I64(storage), DType::U32) => Self::U32(convert_storage!(storage, |v| v as u32)),
            (Self::I64(storage), DType::U64) => Self::U64(convert_storage!(storage, |v| v as u64)),
            (Self::I64(storage), DType::I8) => Self::I8(convert_storage!(storage, |v| v as i8)),
            (Self::I64(storage), DType::I16) => Self::I16(convert_storage!(storage, |v| v as i16)),
            (Self::I64(storage), DType::I32) => Self::I32(convert_storage!(storage, |v| v as i32)),
            (Self::I64(storage), DType::I64) => Self::I64(convert_storage!(storage, |v| v)),
        };

        Ok(result)
    }

    fn contiguous(&self, layout: &Layout) -> HoduResult<Self> {
        // If already contiguous, return clone
        if layout.is_contiguous() {
            return Ok(self.clone());
        }

        let shape = layout.get_shape();
        let strides = layout.get_strides();
        let total_size = layout.get_size();

        macro_rules! contiguous_impl {
            ($storage:expr, $dtype_variant:ident) => {{
                let mut contiguous_data = Vec::with_capacity(total_size);

                // Multi-dimensional index iteration
                let mut indices = vec![0; shape.len()];
                for _ in 0..total_size {
                    // Calculate linear offset from multi-dimensional indices
                    let mut offset = layout.get_offset();
                    for (_i, (&idx, &stride)) in indices.iter().zip(strides.iter()).enumerate() {
                        offset += idx * stride;
                    }

                    // Copy element at calculated offset
                    if offset < $storage.len() {
                        contiguous_data.push($storage[offset]);
                    }

                    // Increment indices (row-major order)
                    let mut carry = 1;
                    for i in (0..shape.len()).rev() {
                        indices[i] += carry;
                        if indices[i] < shape[i] {
                            carry = 0;
                            break;
                        }
                        indices[i] = 0;
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
            Self::F8E5M2(storage) => contiguous_impl!(storage, F8E5M2),
            Self::BF16(storage) => contiguous_impl!(storage, BF16),
            Self::F16(storage) => contiguous_impl!(storage, F16),
            Self::F32(storage) => contiguous_impl!(storage, F32),
            Self::F64(storage) => contiguous_impl!(storage, F64),
            Self::U8(storage) => contiguous_impl!(storage, U8),
            Self::U16(storage) => contiguous_impl!(storage, U16),
            Self::U32(storage) => contiguous_impl!(storage, U32),
            Self::U64(storage) => contiguous_impl!(storage, U64),
            Self::I8(storage) => contiguous_impl!(storage, I8),
            Self::I16(storage) => contiguous_impl!(storage, I16),
            Self::I32(storage) => contiguous_impl!(storage, I32),
            Self::I64(storage) => contiguous_impl!(storage, I64),
        };

        Ok(result)
    }
}
