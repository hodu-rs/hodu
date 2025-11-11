use crate::{
    be::device::BackendDeviceT,
    be_cpu::storage::CpuStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    types::DType,
};
use float8::F8E4M3;
#[cfg(feature = "f8e5m2")]
use float8::F8E5M2;
use half::{bf16, f16};

#[derive(Debug, Clone)]
pub struct CpuDevice;

impl BackendDeviceT for CpuDevice {
    type BackendStorage = CpuStorage;

    #[allow(clippy::uninit_vec)]
    fn allocate(size: usize, dtype: DType) -> HoduResult<Self::BackendStorage> {
        let storage = match dtype {
            DType::BOOL => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::BOOL(v)
            },
            DType::F8E4M3 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::F8E4M3(v)
            },
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::F8E5M2(v)
            },
            DType::BF16 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::BF16(v)
            },
            DType::F16 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::F16(v)
            },
            DType::F32 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::F32(v)
            },
            #[cfg(feature = "f64")]
            DType::F64 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::F64(v)
            },
            DType::U8 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::U8(v)
            },
            #[cfg(feature = "u16")]
            DType::U16 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::U16(v)
            },
            DType::U32 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::U32(v)
            },
            #[cfg(feature = "u64")]
            DType::U64 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::U64(v)
            },
            DType::I8 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::I8(v)
            },
            #[cfg(feature = "i16")]
            DType::I16 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::I16(v)
            },
            DType::I32 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::I32(v)
            },
            #[cfg(feature = "i64")]
            DType::I64 => {
                let mut v = Vec::with_capacity(size);
                unsafe {
                    v.set_len(size);
                }
                CpuStorage::I64(v)
            },
        };
        Ok(storage)
    }

    fn zeros(size: usize, dtype: DType) -> HoduResult<CpuStorage> {
        let storage = match dtype {
            DType::BOOL => CpuStorage::BOOL(vec![false; size]),
            DType::F8E4M3 => CpuStorage::F8E4M3(vec![F8E4M3::ZERO; size]),
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => CpuStorage::F8E5M2(vec![F8E5M2::ZERO; size]),
            DType::BF16 => CpuStorage::BF16(vec![bf16::ZERO; size]),
            DType::F16 => CpuStorage::F16(vec![f16::ZERO; size]),
            DType::F32 => CpuStorage::F32(vec![0f32; size]),
            #[cfg(feature = "f64")]
            DType::F64 => CpuStorage::F64(vec![0f64; size]),
            DType::U8 => CpuStorage::U8(vec![0u8; size]),
            #[cfg(feature = "u16")]
            DType::U16 => CpuStorage::U16(vec![0u16; size]),
            DType::U32 => CpuStorage::U32(vec![0u32; size]),
            #[cfg(feature = "u64")]
            DType::U64 => CpuStorage::U64(vec![0u64; size]),
            DType::I8 => CpuStorage::I8(vec![0i8; size]),
            #[cfg(feature = "i16")]
            DType::I16 => CpuStorage::I16(vec![0i16; size]),
            DType::I32 => CpuStorage::I32(vec![0i32; size]),
            #[cfg(feature = "i64")]
            DType::I64 => CpuStorage::I64(vec![0i64; size]),
        };
        Ok(storage)
    }

    fn randn(size: usize, dtype: DType, mean: f32, std: f32) -> HoduResult<CpuStorage> {
        use rand::prelude::*;

        let mut rng = rand::rng();
        match dtype {
            DType::F8E4M3 => {
                let mut data = Vec::with_capacity(size);
                let normal = rand_distr::Normal::new(F8E4M3::from_f32(mean), F8E4M3::from_f32(std))
                    .map_err(|e| HoduError::BackendError(format!("normal distribution error: {e}")))?;
                for _i in 0..size {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F8E4M3(data))
            },
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => {
                let mut data = Vec::with_capacity(size);
                let normal = rand_distr::Normal::new(F8E5M2::from_f32(mean), F8E5M2::from_f32(std))
                    .map_err(|e| HoduError::BackendError(format!("normal distribution error: {e}")))?;
                for _i in 0..size {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F8E5M2(data))
            },
            DType::BF16 => {
                let mut data = Vec::with_capacity(size);
                let normal = rand_distr::Normal::new(bf16::from_f32(mean), bf16::from_f32(std))
                    .map_err(|e| HoduError::BackendError(format!("normal distribution error: {e}")))?;
                for _i in 0..size {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::BF16(data))
            },
            DType::F16 => {
                let mut data = Vec::with_capacity(size);
                let normal = rand_distr::Normal::new(f16::from_f32(mean), f16::from_f32(std))
                    .map_err(|e| HoduError::BackendError(format!("normal distribution error: {e}")))?;
                for _i in 0..size {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F16(data))
            },
            DType::F32 => {
                let mut data = Vec::with_capacity(size);
                let normal = rand_distr::Normal::new(mean, std)
                    .map_err(|e| HoduError::BackendError(format!("normal distribution error: {e}")))?;
                for _i in 0..size {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F32(data))
            },
            #[cfg(feature = "f64")]
            DType::F64 => {
                let mut data = Vec::with_capacity(size);
                let normal = rand_distr::Normal::new(mean as f64, std as f64)
                    .map_err(|e| HoduError::BackendError(format!("normal distribution error: {e}")))?;
                for _i in 0..size {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F64(data))
            },
            _ => Err(HoduError::UnsupportedDType {
                dtype,
                reason: "randn operation only supports floating-point types (f8e4m3, f8e5m2, bf16, f16, f32, f64)"
                    .to_string(),
            }),
        }
    }

    fn rand_uniform(size: usize, dtype: DType, low: f32, high: f32) -> HoduResult<CpuStorage> {
        use rand::prelude::*;

        let mut rng = rand::rng();
        match dtype {
            DType::F8E4M3 => {
                let mut data = Vec::with_capacity(size);
                let uniform = rand_distr::Uniform::new(F8E4M3::from_f32(low), F8E4M3::from_f32(high))
                    .map_err(|e| HoduError::BackendError(format!("uniform distribution error: {e}")))?;
                for _i in 0..size {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F8E4M3(data))
            },
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => {
                let mut data = Vec::with_capacity(size);
                let uniform = rand_distr::Uniform::new(F8E5M2::from_f32(low), F8E5M2::from_f32(high))
                    .map_err(|e| HoduError::BackendError(format!("uniform distribution error: {e}")))?;
                for _i in 0..size {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F8E5M2(data))
            },
            DType::BF16 => {
                let mut data = Vec::with_capacity(size);
                let uniform = rand_distr::Uniform::new(bf16::from_f32(low), bf16::from_f32(high))
                    .map_err(|e| HoduError::BackendError(format!("uniform distribution error: {e}")))?;
                for _i in 0..size {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::BF16(data))
            },
            DType::F16 => {
                let mut data = Vec::with_capacity(size);
                let uniform = rand_distr::Uniform::new(f16::from_f32(low), f16::from_f32(high))
                    .map_err(|e| HoduError::BackendError(format!("uniform distribution error: {e}")))?;
                for _i in 0..size {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F16(data))
            },
            DType::F32 => {
                let mut data = Vec::with_capacity(size);
                let uniform = rand_distr::Uniform::new(low, high)
                    .map_err(|e| HoduError::BackendError(format!("uniform distribution error: {e}")))?;
                for _i in 0..size {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F32(data))
            },
            #[cfg(feature = "f64")]
            DType::F64 => {
                let mut data = Vec::with_capacity(size);
                let uniform = rand_distr::Uniform::new(low as f64, high as f64)
                    .map_err(|e| HoduError::BackendError(format!("uniform distribution error: {e}")))?;
                for _i in 0..size {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F64(data))
            },
            _ => Err(HoduError::UnsupportedDType {
                dtype,
                reason:
                    "rand_uniform operation only supports floating-point types (f8e4m3, f8e5m2, bf16, f16, f32, f64)"
                        .to_string(),
            }),
        }
    }
}
