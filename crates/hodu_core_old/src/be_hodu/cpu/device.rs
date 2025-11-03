use crate::{
    be_hodu::{cpu::storage::CpuStorage, device::HoduDeviceT},
    compat::*,
    error::{HoduError, HoduResult},
    types::{dtype::DType, layout::Layout},
};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};

#[derive(Debug, Clone)]
pub struct CpuDevice;

impl HoduDeviceT for CpuDevice {
    type HoduStorage = CpuStorage;

    fn zeros(layout: &Layout, dtype: DType) -> HoduResult<CpuStorage> {
        let elem_count = layout.get_size();
        let storage = match dtype {
            DType::BOOL => CpuStorage::BOOL(vec![false; elem_count]),
            DType::F8E4M3 => CpuStorage::F8E4M3(vec![F8E4M3::ZERO; elem_count]),
            DType::F8E5M2 => CpuStorage::F8E5M2(vec![F8E5M2::ZERO; elem_count]),
            DType::BF16 => CpuStorage::BF16(vec![bf16::ZERO; elem_count]),
            DType::F16 => CpuStorage::F16(vec![f16::ZERO; elem_count]),
            DType::F32 => CpuStorage::F32(vec![0f32; elem_count]),
            DType::F64 => CpuStorage::F64(vec![0f64; elem_count]),
            #[cfg(feature = "u8")]
            DType::U8 => CpuStorage::U8(vec![0u8; elem_count]),
            DType::U16 => CpuStorage::U16(vec![0u16; elem_count]),
            #[cfg(feature = "u32")]
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

    fn randn(layout: &Layout, dtype: DType, mean: f64, std: f64) -> HoduResult<CpuStorage> {
        use rand::prelude::*;

        let elem_count = layout.get_size();
        let mut rng = rand::rng();
        match dtype {
            DType::F8E4M3 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(F8E4M3::from_f64(mean), F8E4M3::from_f64(std))
                    .map_err(|e| HoduError::InternalError(format!("Normal distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F8E4M3(data))
            },
            DType::F8E5M2 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(F8E5M2::from_f64(mean), F8E5M2::from_f64(std))
                    .map_err(|e| HoduError::InternalError(format!("Normal distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F8E5M2(data))
            },
            DType::BF16 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(bf16::from_f64(mean), bf16::from_f64(std))
                    .map_err(|e| HoduError::InternalError(format!("Normal distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::BF16(data))
            },
            DType::F16 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(f16::from_f64(mean), f16::from_f64(std))
                    .map_err(|e| HoduError::InternalError(format!("Normal distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F16(data))
            },
            DType::F32 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(mean as f32, std as f32)
                    .map_err(|e| HoduError::InternalError(format!("Normal distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F32(data))
            },
            DType::F64 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(mean, std)
                    .map_err(|e| HoduError::InternalError(format!("Normal distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F64(data))
            },
            _ => Err(HoduError::UnsupportedDType {
                dtype,
                op: "randn".to_string(),
            }),
        }
    }

    fn rand_uniform(layout: &Layout, dtype: DType, low: f64, high: f64) -> HoduResult<CpuStorage> {
        use rand::prelude::*;

        let elem_count = layout.get_size();
        let mut rng = rand::rng();
        match dtype {
            DType::F8E4M3 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand_distr::Uniform::new(F8E4M3::from_f64(low), F8E4M3::from_f64(high))
                    .map_err(|e| HoduError::InternalError(format!("Uniform distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F8E4M3(data))
            },
            DType::F8E5M2 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand_distr::Uniform::new(F8E5M2::from_f64(low), F8E5M2::from_f64(high))
                    .map_err(|e| HoduError::InternalError(format!("Uniform distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F8E5M2(data))
            },
            DType::BF16 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand_distr::Uniform::new(bf16::from_f64(low), bf16::from_f64(high))
                    .map_err(|e| HoduError::InternalError(format!("Uniform distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::BF16(data))
            },
            DType::F16 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand_distr::Uniform::new(f16::from_f64(low), f16::from_f64(high))
                    .map_err(|e| HoduError::InternalError(format!("Uniform distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F16(data))
            },
            DType::F32 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand_distr::Uniform::new(low as f32, high as f32)
                    .map_err(|e| HoduError::InternalError(format!("Uniform distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F32(data))
            },
            DType::F64 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand_distr::Uniform::new(low, high)
                    .map_err(|e| HoduError::InternalError(format!("Uniform distribution error: {e}")))?;
                for _i in 0..elem_count {
                    data.push(uniform.sample(&mut rng))
                }
                Ok(CpuStorage::F64(data))
            },
            _ => Err(HoduError::UnsupportedDType {
                dtype,
                op: "rand_uniform".to_string(),
            }),
        }
    }
}
