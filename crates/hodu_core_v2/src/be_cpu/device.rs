use crate::{
    be::device::BackendDeviceT,
    be_cpu::storage::CpuStorage,
    error::{HoduError, HoduResult},
    types::{DType, Shape},
};
use float8::F8E4M3;
#[cfg(feature = "f8e5m2")]
use float8::F8E5M2;
use half::{bf16, f16};

#[derive(Debug, Clone)]
pub struct CpuDevice;

impl BackendDeviceT for CpuDevice {
    type BackendStorage = CpuStorage;

    fn zeros(shape: &Shape, dtype: DType) -> HoduResult<CpuStorage> {
        let elem_count = shape.size() as usize;
        let storage = match dtype {
            DType::BOOL => CpuStorage::BOOL(vec![false; elem_count]),
            DType::F8E4M3 => CpuStorage::F8E4M3(vec![F8E4M3::ZERO; elem_count]),
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => CpuStorage::F8E5M2(vec![F8E5M2::ZERO; elem_count]),
            DType::BF16 => CpuStorage::BF16(vec![bf16::ZERO; elem_count]),
            DType::F16 => CpuStorage::F16(vec![f16::ZERO; elem_count]),
            DType::F32 => CpuStorage::F32(vec![0f32; elem_count]),
            #[cfg(feature = "f64")]
            DType::F64 => CpuStorage::F64(vec![0f64; elem_count]),
            DType::U8 => CpuStorage::U8(vec![0u8; elem_count]),
            #[cfg(feature = "u16")]
            DType::U16 => CpuStorage::U16(vec![0u16; elem_count]),
            DType::U32 => CpuStorage::U32(vec![0u32; elem_count]),
            #[cfg(feature = "u64")]
            DType::U64 => CpuStorage::U64(vec![0u64; elem_count]),
            DType::I8 => CpuStorage::I8(vec![0i8; elem_count]),
            #[cfg(feature = "i16")]
            DType::I16 => CpuStorage::I16(vec![0i16; elem_count]),
            DType::I32 => CpuStorage::I32(vec![0i32; elem_count]),
            #[cfg(feature = "i64")]
            DType::I64 => CpuStorage::I64(vec![0i64; elem_count]),
        };
        Ok(storage)
    }

    fn randn(shape: &Shape, dtype: DType, mean: f32, std: f32) -> HoduResult<CpuStorage> {
        use rand::prelude::*;

        let elem_count = shape.size() as usize;
        let mut rng = rand::rng();
        match dtype {
            DType::F8E4M3 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(F8E4M3::from_f32(mean), F8E4M3::from_f32(std))
                    .map_err(|e| HoduError::InternalError(format!("normal distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F8E4M3(data))
            },
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(F8E5M2::from_f32(mean), F8E5M2::from_f32(std))
                    .map_err(|e| HoduError::InternalError(format!("normal distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F8E5M2(data))
            },
            DType::BF16 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(bf16::from_f32(mean), bf16::from_f32(std))
                    .map_err(|e| HoduError::InternalError(format!("normal distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::BF16(data))
            },
            DType::F16 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(f16::from_f32(mean), f16::from_f32(std))
                    .map_err(|e| HoduError::InternalError(format!("normal distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F16(data))
            },
            DType::F32 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(mean, std)
                    .map_err(|e| HoduError::InternalError(format!("normal distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F32(data))
            },
            #[cfg(feature = "f64")]
            DType::F64 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(mean as f64, std as f64)
                    .map_err(|e| HoduError::InternalError(format!("normal distribution error: {e}")))?;
                for _i in 0..elem_count {
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

    fn rand_uniform(shape: &Shape, dtype: DType, low: f32, high: f32) -> HoduResult<CpuStorage> {
        use rand::prelude::*;

        let elem_count = shape.size() as usize;
        let mut rng = rand::rng();
        match dtype {
            DType::F8E4M3 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand_distr::Uniform::new(F8E4M3::from_f32(low), F8E4M3::from_f32(high))
                    .map_err(|e| HoduError::InternalError(format!("uniform distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F8E4M3(data))
            },
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand_distr::Uniform::new(F8E5M2::from_f32(low), F8E5M2::from_f32(high))
                    .map_err(|e| HoduError::InternalError(format!("uniform distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F8E5M2(data))
            },
            DType::BF16 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand_distr::Uniform::new(bf16::from_f32(low), bf16::from_f32(high))
                    .map_err(|e| HoduError::InternalError(format!("uniform distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::BF16(data))
            },
            DType::F16 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand_distr::Uniform::new(f16::from_f32(low), f16::from_f32(high))
                    .map_err(|e| HoduError::InternalError(format!("uniform distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F16(data))
            },
            DType::F32 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand_distr::Uniform::new(low, high)
                    .map_err(|e| HoduError::InternalError(format!("uniform distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F32(data))
            },
            #[cfg(feature = "f64")]
            DType::F64 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand_distr::Uniform::new(low as f64, high as f64)
                    .map_err(|e| HoduError::InternalError(format!("uniform distribution error: {e}")))?;
                for _i in 0..elem_count {
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
