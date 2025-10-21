#![allow(clippy::type_complexity)]

use crate::{
    backends::be_hodu::{cpu::storage::CpuStorage, storage::HoduStorageT},
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

        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            result.push(flat_data[start..end].to_vec());
        }

        Ok(result)
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
        let dim0 = dims[0];
        let dim1 = dims[1];
        let dim2 = dims[2];

        let mut result = Vec::with_capacity(dim0);
        for i in 0..dim0 {
            let mut slice_i = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut slice_j = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    let idx = (i * dim1 * dim2) + (j * dim2) + k;
                    slice_j.push(flat_data[idx].clone());
                }
                slice_i.push(slice_j);
            }
            result.push(slice_i);
        }

        Ok(result)
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
        let dim0 = dims[0];
        let dim1 = dims[1];
        let dim2 = dims[2];
        let dim3 = dims[3];

        let mut result = Vec::with_capacity(dim0);
        for i in 0..dim0 {
            let mut slice_i = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut slice_j = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    let mut slice_k = Vec::with_capacity(dim3);
                    for l in 0..dim3 {
                        let idx = (i * dim1 * dim2 * dim3) + (j * dim2 * dim3) + (k * dim3) + l;
                        slice_k.push(flat_data[idx].clone());
                    }
                    slice_j.push(slice_k);
                }
                slice_i.push(slice_j);
            }
            result.push(slice_i);
        }

        Ok(result)
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
        let dim0 = dims[0];
        let dim1 = dims[1];
        let dim2 = dims[2];
        let dim3 = dims[3];
        let dim4 = dims[4];

        let mut result = Vec::with_capacity(dim0);
        for i in 0..dim0 {
            let mut slice_i = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut slice_j = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    let mut slice_k = Vec::with_capacity(dim3);
                    for l in 0..dim3 {
                        let mut slice_l = Vec::with_capacity(dim4);
                        for m in 0..dim4 {
                            let idx = (i * dim1 * dim2 * dim3 * dim4)
                                + (j * dim2 * dim3 * dim4)
                                + (k * dim3 * dim4)
                                + (l * dim4)
                                + m;
                            slice_l.push(flat_data[idx].clone());
                        }
                        slice_k.push(slice_l);
                    }
                    slice_j.push(slice_k);
                }
                slice_i.push(slice_j);
            }
            result.push(slice_i);
        }

        Ok(result)
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
        let dim0 = dims[0];
        let dim1 = dims[1];
        let dim2 = dims[2];
        let dim3 = dims[3];
        let dim4 = dims[4];
        let dim5 = dims[5];

        let mut result = Vec::with_capacity(dim0);
        for i in 0..dim0 {
            let mut slice_i = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut slice_j = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    let mut slice_k = Vec::with_capacity(dim3);
                    for l in 0..dim3 {
                        let mut slice_l = Vec::with_capacity(dim4);
                        for m in 0..dim4 {
                            let mut slice_m = Vec::with_capacity(dim5);
                            for n in 0..dim5 {
                                let idx = (i * dim1 * dim2 * dim3 * dim4 * dim5)
                                    + (j * dim2 * dim3 * dim4 * dim5)
                                    + (k * dim3 * dim4 * dim5)
                                    + (l * dim4 * dim5)
                                    + (m * dim5)
                                    + n;
                                slice_m.push(flat_data[idx].clone());
                            }
                            slice_l.push(slice_m);
                        }
                        slice_k.push(slice_l);
                    }
                    slice_j.push(slice_k);
                }
                slice_i.push(slice_j);
            }
            result.push(slice_i);
        }

        Ok(result)
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

    match &cpu_storage {
        CpuStorage::BOOL(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::F8E4M3(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::F8E5M2(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::BF16(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::F16(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::F32(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::F64(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::U8(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::U16(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::U32(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::U64(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::I8(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::I16(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::I32(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
        CpuStorage::I64(data) => {
            if let Ok(typed_data) = try_cast_slice::<T, _>(data) {
                extract_with_layout(&mut result, typed_data, layout);
            } else {
                return Err(HoduError::InternalError("Type mismatch in extraction".to_string()));
            }
        },
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
    if core::any::TypeId::of::<T>() == core::any::TypeId::of::<bool>() {
        Some(DType::BOOL)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<F8E4M3>() {
        Some(DType::F8E4M3)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<F8E5M2>() {
        Some(DType::F8E5M2)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<bf16>() {
        Some(DType::BF16)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f16>() {
        Some(DType::F16)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        Some(DType::F32)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
        Some(DType::F64)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<u8>() {
        Some(DType::U8)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<u16>() {
        Some(DType::U16)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<u32>() {
        Some(DType::U32)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<u64>() {
        Some(DType::U64)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<i8>() {
        Some(DType::I8)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<i16>() {
        Some(DType::I16)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<i32>() {
        Some(DType::I32)
    } else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<i64>() {
        Some(DType::I64)
    } else {
        None
    }
}
