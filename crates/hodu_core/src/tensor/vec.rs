#![allow(clippy::type_complexity)]

use crate::{
    be_hodu::{cpu::storage::CpuStorage, storage::HoduStorageT},
    compat::*,
    error::{HoduError, HoduResult},
    tensor::Tensor,
    types::{dtype::DType, layout::Layout},
};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};

impl Tensor {
    pub fn to_flatten_vec<T: Clone + 'static>(&self) -> HoduResult<Vec<T>> {
        let cpu_storage = self.with_storage(|storage| storage.to_cpu_storage())?;
        let layout = self.get_layout();

        extract_vec_from_cpu_storage::<T>(&cpu_storage, &layout)
    }

    pub fn to_vec1d<T: Clone + 'static>(&self) -> HoduResult<Vec<T>> {
        let layout = self.get_layout();
        if layout.get_shape().len() != 1 {
            return Err(HoduError::InternalError(format!(
                "Expected 1D tensor, but got {}D tensor",
                layout.get_shape().len()
            )));
        }
        self.to_flatten_vec()
    }

    pub fn to_vec2d<T: Clone + 'static>(&self) -> HoduResult<Vec<Vec<T>>> {
        let layout = self.get_layout();
        let dims = layout.get_shape();
        if dims.len() != 2 {
            return Err(HoduError::InternalError(format!(
                "Expected 2D tensor, but got {}D tensor",
                dims.len()
            )));
        }

        let flat_data = self.to_flatten_vec::<T>()?;
        let rows = dims[0];
        let cols = dims[1];

        Ok((0..rows)
            .map(|i| {
                let start = i * cols;
                flat_data[start..start + cols].to_vec()
            })
            .collect())
    }

    pub fn to_vec3d<T: Clone + 'static>(&self) -> HoduResult<Vec<Vec<Vec<T>>>> {
        let layout = self.get_layout();
        let dims = layout.get_shape();
        if dims.len() != 3 {
            return Err(HoduError::InternalError(format!(
                "Expected 3D tensor, but got {}D tensor",
                dims.len()
            )));
        }

        let flat_data = self.to_flatten_vec::<T>()?;
        let (d0, d1, d2) = (dims[0], dims[1], dims[2]);
        let stride1 = d1 * d2;

        Ok((0..d0)
            .map(|i| {
                let offset_i = i * stride1;
                (0..d1)
                    .map(|j| {
                        let start = offset_i + j * d2;
                        flat_data[start..start + d2].to_vec()
                    })
                    .collect()
            })
            .collect())
    }

    pub fn to_vec4d<T: Clone + 'static>(&self) -> HoduResult<Vec<Vec<Vec<Vec<T>>>>> {
        let layout = self.get_layout();
        let dims = layout.get_shape();
        if dims.len() != 4 {
            return Err(HoduError::InternalError(format!(
                "Expected 4D tensor, but got {}D tensor",
                dims.len()
            )));
        }

        let flat_data = self.to_flatten_vec::<T>()?;
        let (d0, d1, d2, d3) = (dims[0], dims[1], dims[2], dims[3]);
        let stride1 = d1 * d2 * d3;
        let stride2 = d2 * d3;

        Ok((0..d0)
            .map(|i| {
                let offset_i = i * stride1;
                (0..d1)
                    .map(|j| {
                        let offset_j = offset_i + j * stride2;
                        (0..d2)
                            .map(|k| {
                                let start = offset_j + k * d3;
                                flat_data[start..start + d3].to_vec()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect())
    }

    pub fn to_vec5d<T: Clone + 'static>(&self) -> HoduResult<Vec<Vec<Vec<Vec<Vec<T>>>>>> {
        let layout = self.get_layout();
        let dims = layout.get_shape();
        if dims.len() != 5 {
            return Err(HoduError::InternalError(format!(
                "Expected 5D tensor, but got {}D tensor",
                dims.len()
            )));
        }

        let flat_data = self.to_flatten_vec::<T>()?;
        let (d0, d1, d2, d3, d4) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let stride1 = d1 * d2 * d3 * d4;
        let stride2 = d2 * d3 * d4;
        let stride3 = d3 * d4;

        Ok((0..d0)
            .map(|i| {
                let offset_i = i * stride1;
                (0..d1)
                    .map(|j| {
                        let offset_j = offset_i + j * stride2;
                        (0..d2)
                            .map(|k| {
                                let offset_k = offset_j + k * stride3;
                                (0..d3)
                                    .map(|l| {
                                        let start = offset_k + l * d4;
                                        flat_data[start..start + d4].to_vec()
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect())
    }

    pub fn to_vec6d<T: Clone + 'static>(&self) -> HoduResult<Vec<Vec<Vec<Vec<Vec<Vec<T>>>>>>> {
        let layout = self.get_layout();
        let dims = layout.get_shape();
        if dims.len() != 6 {
            return Err(HoduError::InternalError(format!(
                "Expected 6D tensor, but got {}D tensor",
                dims.len()
            )));
        }

        let flat_data = self.to_flatten_vec::<T>()?;
        let (d0, d1, d2, d3, d4, d5) = (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
        let stride1 = d1 * d2 * d3 * d4 * d5;
        let stride2 = d2 * d3 * d4 * d5;
        let stride3 = d3 * d4 * d5;
        let stride4 = d4 * d5;

        Ok((0..d0)
            .map(|i| {
                let offset_i = i * stride1;
                (0..d1)
                    .map(|j| {
                        let offset_j = offset_i + j * stride2;
                        (0..d2)
                            .map(|k| {
                                let offset_k = offset_j + k * stride3;
                                (0..d3)
                                    .map(|l| {
                                        let offset_l = offset_k + l * stride4;
                                        (0..d4)
                                            .map(|m| {
                                                let start = offset_l + m * d5;
                                                flat_data[start..start + d5].to_vec()
                                            })
                                            .collect()
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect())
    }
}

fn extract_vec_from_cpu_storage<T: Clone + 'static>(cpu_storage: &CpuStorage, layout: &Layout) -> HoduResult<Vec<T>> {
    let target_dtype = get_dtype_for_type::<T>()
        .ok_or_else(|| HoduError::InternalError("Unsupported type for extraction".to_string()))?;

    let cpu_storage = if cpu_storage.get_dtype() != target_dtype {
        cpu_storage.to_dtype(target_dtype, layout)?
    } else {
        cpu_storage.clone()
    };

    let total_elements = layout.get_shape().iter().product::<usize>();
    let mut result = Vec::with_capacity(total_elements);

    macro_rules! extract_storage {
        ($data:expr) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>($data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError(
                    "Type mismatch in extraction".to_string(),
                ));
            }
        };
    }

    match &cpu_storage {
        CpuStorage::BOOL(data) => extract_storage!(data),
        CpuStorage::F8E4M3(data) => extract_storage!(data),
        CpuStorage::F8E5M2(data) => extract_storage!(data),
        CpuStorage::BF16(data) => extract_storage!(data),
        CpuStorage::F16(data) => extract_storage!(data),
        CpuStorage::F32(data) => extract_storage!(data),
        CpuStorage::F64(data) => extract_storage!(data),
        #[cfg(feature = "u8")]
        CpuStorage::U8(data) => extract_storage!(data),
        CpuStorage::U16(data) => extract_storage!(data),
        #[cfg(feature = "u32")]
        CpuStorage::U32(data) => extract_storage!(data),
        #[cfg(feature = "u64")]
        CpuStorage::U64(data) => extract_storage!(data),
        CpuStorage::I8(data) => extract_storage!(data),
        #[cfg(feature = "i16")]
        CpuStorage::I16(data) => extract_storage!(data),
        CpuStorage::I32(data) => extract_storage!(data),
        #[cfg(feature = "i64")]
        CpuStorage::I64(data) => extract_storage!(data),
    }

    Ok(result)
}

fn try_cast_slice<T: 'static, U: 'static>(data: &[U]) -> Result<&[T], ()> {
    if core::any::TypeId::of::<T>() == core::any::TypeId::of::<U>() {
        unsafe { Ok(core::slice::from_raw_parts(data.as_ptr() as *const T, data.len())) }
    } else {
        Err(())
    }
}

fn extract_with_layout<T: Clone>(result: &mut Vec<T>, data: &[T], layout: &Layout) {
    let dims = layout.get_shape();
    let strides = layout.get_strides();

    if layout.is_contiguous() {
        result.extend_from_slice(data);
    } else {
        let total_elements = dims.iter().product::<usize>();
        let mut indices = vec![0; dims.len()];

        for _ in 0..total_elements {
            let flat_idx = indices.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum::<usize>();

            if flat_idx < data.len() {
                result.push(data[flat_idx].clone());
            }

            // Increment indices
            let mut carry = 1;
            for i in (0..dims.len()).rev() {
                indices[i] += carry;
                if indices[i] < dims[i] {
                    carry = 0;
                    break;
                } else {
                    indices[i] = 0;
                }
            }

            if carry == 1 {
                break;
            }
        }
    }
}

fn get_dtype_for_type<T: 'static>() -> Option<DType> {
    let type_id = core::any::TypeId::of::<T>();

    if type_id == core::any::TypeId::of::<bool>() {
        return Some(DType::BOOL);
    }
    if type_id == core::any::TypeId::of::<F8E4M3>() {
        return Some(DType::F8E4M3);
    }
    if type_id == core::any::TypeId::of::<F8E5M2>() {
        return Some(DType::F8E5M2);
    }
    if type_id == core::any::TypeId::of::<bf16>() {
        return Some(DType::BF16);
    }
    if type_id == core::any::TypeId::of::<f16>() {
        return Some(DType::F16);
    }
    if type_id == core::any::TypeId::of::<f32>() {
        return Some(DType::F32);
    }
    if type_id == core::any::TypeId::of::<f64>() {
        return Some(DType::F64);
    }

    #[cfg(feature = "u8")]
    if type_id == core::any::TypeId::of::<u8>() {
        return Some(DType::U8);
    }

    if type_id == core::any::TypeId::of::<u16>() {
        return Some(DType::U16);
    }

    #[cfg(feature = "u32")]
    if type_id == core::any::TypeId::of::<u32>() {
        return Some(DType::U32);
    }

    #[cfg(feature = "u64")]
    if type_id == core::any::TypeId::of::<u64>() {
        return Some(DType::U64);
    }

    if type_id == core::any::TypeId::of::<i8>() {
        return Some(DType::I8);
    }

    #[cfg(feature = "i16")]
    if type_id == core::any::TypeId::of::<i16>() {
        return Some(DType::I16);
    }

    if type_id == core::any::TypeId::of::<i32>() {
        return Some(DType::I32);
    }

    #[cfg(feature = "i64")]
    if type_id == core::any::TypeId::of::<i64>() {
        return Some(DType::I64);
    }

    None
}
