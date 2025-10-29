use crate::{
    be_hodu::{cpu::storage::CpuStorage, storage::HoduStorageT},
    compat::*,
    error::{HoduError, HoduResult},
    op::{window_reduction::WindowReduction, DivScalar},
    scalar::Scalar,
    types::layout::Layout,
};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};

#[cfg(feature = "rayon")]
const PARALLEL_THRESHOLD: usize = 4096;

// Helper function to convert flat index to multi-dimensional indices
#[allow(dead_code)]
fn flat_to_indices(mut flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        indices[i] = flat_idx % shape[i];
        flat_idx /= shape[i];
    }
    indices
}

pub fn reduce_window_impl(
    storage: &CpuStorage,
    input_layout: &Layout,
    window_shape: &[usize],
    strides: &[usize],
    padding: &[(usize, usize)],
    reduction: WindowReduction,
) -> HoduResult<CpuStorage> {
    let input_shape = input_layout.get_shape();
    let rank = input_shape.len();

    if window_shape.len() != rank || strides.len() != rank || padding.len() != rank {
        return Err(HoduError::InternalError(
            "window_shape, strides, and padding must have same rank as input".to_string(),
        ));
    }

    // Calculate output shape
    let mut output_shape = Vec::with_capacity(rank);
    for i in 0..rank {
        let padded_size = input_shape[i] + padding[i].0 + padding[i].1;
        let out_size = (padded_size - window_shape[i]) / strides[i] + 1;
        output_shape.push(out_size);
    }

    macro_rules! impl_reduce_window {
        ($storage:expr, $T:ty, $init_val:expr, $reduce_op:expr, $variant:ident) => {{
            let output_size: usize = output_shape.iter().product();

            #[cfg(feature = "rayon")]
            {
                if output_size >= PARALLEL_THRESHOLD {
                    use rayon::prelude::*;
                    let output_data: Vec<$T> = (0..output_size)
                        .into_par_iter()
                        .map(|out_idx| {
                            // Calculate output coordinates
                            let out_coords = flat_to_indices(out_idx, &output_shape);

                            // Initialize accumulator
                            let mut acc = $init_val;

                            // Iterate over window
                            let window_size: usize = window_shape.iter().product();
                            for win_idx in 0..window_size {
                                // Calculate window coordinates
                                let window_coords = flat_to_indices(win_idx, window_shape);

                                // Calculate absolute coordinates in input space (before padding adjustment)
                                // Window starts at (output_coord * stride) and we need to account for padding
                                let mut input_coords = vec![0; rank];
                                let mut in_bounds = true;
                                for i in 0..rank {
                                    // Calculate position in padded space
                                    let padded_pos = out_coords[i] * strides[i] + window_coords[i];
                                    // Check if within padded bounds
                                    if padded_pos < padding[i].0 {
                                        in_bounds = false;
                                        break;
                                    }
                                    // Convert to actual input coordinates
                                    let input_pos = padded_pos - padding[i].0;
                                    if input_pos >= input_shape[i] {
                                        in_bounds = false;
                                        break;
                                    }
                                    input_coords[i] = input_pos;
                                }

                                // Get value from input or use init_val for padding
                                let val = if in_bounds {
                                    // Calculate flat index using layout strides and offset
                                    let mut idx = input_layout.get_offset();
                                    for i in 0..rank {
                                        idx += input_coords[i] * input_layout.get_strides()[i];
                                    }
                                    $storage[idx]
                                } else {
                                    $init_val // Padding region uses init value
                                };

                                acc = $reduce_op(acc, val);
                            }

                            acc
                        })
                        .collect();

                    Ok(CpuStorage::$variant(output_data))
                } else {
                    let mut output_data = vec![$init_val; output_size];

                    for out_idx in 0..output_size {
                        let out_coords = flat_to_indices(out_idx, &output_shape);

                        let mut acc = $init_val;

                        let window_size: usize = window_shape.iter().product();
                        for win_idx in 0..window_size {
                            let window_coords = flat_to_indices(win_idx, window_shape);

                            let mut input_coords = vec![0; rank];
                            let mut in_bounds = true;
                            for i in 0..rank {
                                let padded_pos = out_coords[i] * strides[i] + window_coords[i];
                                if padded_pos < padding[i].0 {
                                    in_bounds = false;
                                    break;
                                }
                                let input_pos = padded_pos - padding[i].0;
                                if input_pos >= input_shape[i] {
                                    in_bounds = false;
                                    break;
                                }
                                input_coords[i] = input_pos;
                            }

                            let val = if in_bounds {
                                let mut idx = input_layout.get_offset();
                                for i in 0..rank {
                                    idx += input_coords[i] * input_layout.get_strides()[i];
                                }
                                $storage[idx]
                            } else {
                                $init_val
                            };

                            acc = $reduce_op(acc, val);
                        }

                        output_data[out_idx] = acc;
                    }

                    Ok(CpuStorage::$variant(output_data))
                }
            }

            #[cfg(not(feature = "rayon"))]
            {
                let mut output_data = vec![$init_val; output_size];

                // Iterate over output positions
                for out_idx in 0..output_size {
                    // Calculate output coordinates
                    let mut out_coords = vec![0; rank];
                    let mut tmp = out_idx;
                    for i in (0..rank).rev() {
                        out_coords[i] = tmp % output_shape[i];
                        tmp /= output_shape[i];
                    }

                    // Initialize accumulator
                    let mut acc = $init_val;
                    let mut window_coords = vec![0; rank];

                    // Iterate over window
                    let window_size: usize = window_shape.iter().product();
                    for win_idx in 0..window_size {
                        // Calculate window coordinates
                        let mut tmp = win_idx;
                        for i in (0..rank).rev() {
                            window_coords[i] = tmp % window_shape[i];
                            tmp /= window_shape[i];
                        }

                        // Calculate absolute coordinates in input space (before padding adjustment)
                        // Window starts at (output_coord * stride) and we need to account for padding
                        let mut input_coords = vec![0; rank];
                        let mut in_bounds = true;
                        for i in 0..rank {
                            // Calculate position in padded space
                            let padded_pos = out_coords[i] * strides[i] + window_coords[i];
                            // Check if within padded bounds
                            if padded_pos < padding[i].0 {
                                in_bounds = false;
                                break;
                            }
                            // Convert to actual input coordinates
                            let input_pos = padded_pos - padding[i].0;
                            if input_pos >= input_shape[i] {
                                in_bounds = false;
                                break;
                            }
                            input_coords[i] = input_pos;
                        }

                        // Get value from input or use init_val for padding
                        let val = if in_bounds {
                            // Calculate flat index using layout strides and offset
                            let mut idx = input_layout.get_offset();
                            for i in 0..rank {
                                idx += input_coords[i] * input_layout.get_strides()[i];
                            }
                            $storage[idx]
                        } else {
                            $init_val // Padding region uses init value
                        };

                        acc = $reduce_op(acc, val);
                    }

                    output_data[out_idx] = acc;
                }

                Ok(CpuStorage::$variant(output_data))
            }
        }};
    }

    match reduction {
        WindowReduction::Max => match storage {
            CpuStorage::F8E4M3(storage) => {
                impl_reduce_window!(
                    storage,
                    F8E4M3,
                    F8E4M3::MIN,
                    |a: F8E4M3, b: F8E4M3| if a > b { a } else { b },
                    F8E4M3
                )
            },
            CpuStorage::F8E5M2(storage) => {
                impl_reduce_window!(
                    storage,
                    F8E5M2,
                    F8E5M2::MIN,
                    |a: F8E5M2, b: F8E5M2| if a > b { a } else { b },
                    F8E5M2
                )
            },
            CpuStorage::F16(storage) => {
                impl_reduce_window!(storage, f16, f16::MIN, |a: f16, b: f16| if a > b { a } else { b }, F16)
            },
            CpuStorage::BF16(storage) => impl_reduce_window!(
                storage,
                bf16,
                bf16::MIN,
                |a: bf16, b: bf16| if a > b { a } else { b },
                BF16
            ),
            CpuStorage::F32(storage) => {
                impl_reduce_window!(storage, f32, f32::MIN, |a: f32, b: f32| if a > b { a } else { b }, F32)
            },
            CpuStorage::F64(storage) => {
                impl_reduce_window!(storage, f64, f64::MIN, |a: f64, b: f64| if a > b { a } else { b }, F64)
            },
            CpuStorage::U8(storage) => {
                impl_reduce_window!(storage, u8, u8::MIN, |a: u8, b: u8| if a > b { a } else { b }, U8)
            },
            CpuStorage::U16(storage) => {
                impl_reduce_window!(storage, u16, u16::MIN, |a: u16, b: u16| if a > b { a } else { b }, U16)
            },
            CpuStorage::U32(storage) => {
                impl_reduce_window!(storage, u32, u32::MIN, |a: u32, b: u32| if a > b { a } else { b }, U32)
            },
            CpuStorage::U64(storage) => {
                impl_reduce_window!(storage, u64, u64::MIN, |a: u64, b: u64| if a > b { a } else { b }, U64)
            },
            CpuStorage::I8(storage) => {
                impl_reduce_window!(storage, i8, i8::MIN, |a: i8, b: i8| if a > b { a } else { b }, I8)
            },
            CpuStorage::I16(storage) => {
                impl_reduce_window!(storage, i16, i16::MIN, |a: i16, b: i16| if a > b { a } else { b }, I16)
            },
            CpuStorage::I32(storage) => {
                impl_reduce_window!(storage, i32, i32::MIN, |a: i32, b: i32| if a > b { a } else { b }, I32)
            },
            CpuStorage::I64(storage) => {
                impl_reduce_window!(storage, i64, i64::MIN, |a: i64, b: i64| if a > b { a } else { b }, I64)
            },
            _ => Err(HoduError::UnsupportedDType {
                dtype: storage.get_dtype(),
                op: "reduce_window with Max reduction".to_string(),
            }),
        },
        WindowReduction::Min => match storage {
            CpuStorage::F8E4M3(storage) => {
                impl_reduce_window!(
                    storage,
                    F8E4M3,
                    F8E4M3::MAX,
                    |a: F8E4M3, b: F8E4M3| if a < b { a } else { b },
                    F8E4M3
                )
            },
            CpuStorage::F8E5M2(storage) => {
                impl_reduce_window!(
                    storage,
                    F8E5M2,
                    F8E5M2::MAX,
                    |a: F8E5M2, b: F8E5M2| if a < b { a } else { b },
                    F8E5M2
                )
            },
            CpuStorage::F16(storage) => {
                impl_reduce_window!(storage, f16, f16::MAX, |a: f16, b: f16| if a < b { a } else { b }, F16)
            },
            CpuStorage::BF16(storage) => {
                impl_reduce_window!(
                    storage,
                    bf16,
                    bf16::MAX,
                    |a: bf16, b: bf16| if a < b { a } else { b },
                    BF16
                )
            },
            CpuStorage::F32(storage) => {
                impl_reduce_window!(storage, f32, f32::MAX, |a: f32, b: f32| if a < b { a } else { b }, F32)
            },
            CpuStorage::F64(storage) => {
                impl_reduce_window!(storage, f64, f64::MAX, |a: f64, b: f64| if a < b { a } else { b }, F64)
            },
            CpuStorage::U8(storage) => {
                impl_reduce_window!(storage, u8, u8::MAX, |a: u8, b: u8| if a < b { a } else { b }, U8)
            },
            CpuStorage::U16(storage) => {
                impl_reduce_window!(storage, u16, u16::MAX, |a: u16, b: u16| if a < b { a } else { b }, U16)
            },
            CpuStorage::U32(storage) => {
                impl_reduce_window!(storage, u32, u32::MAX, |a: u32, b: u32| if a < b { a } else { b }, U32)
            },
            CpuStorage::U64(storage) => {
                impl_reduce_window!(storage, u64, u64::MAX, |a: u64, b: u64| if a < b { a } else { b }, U64)
            },
            CpuStorage::I8(storage) => {
                impl_reduce_window!(storage, i8, i8::MAX, |a: i8, b: i8| if a < b { a } else { b }, I8)
            },
            CpuStorage::I16(storage) => {
                impl_reduce_window!(storage, i16, i16::MAX, |a: i16, b: i16| if a < b { a } else { b }, I16)
            },
            CpuStorage::I32(storage) => {
                impl_reduce_window!(storage, i32, i32::MAX, |a: i32, b: i32| if a < b { a } else { b }, I32)
            },
            CpuStorage::I64(storage) => {
                impl_reduce_window!(storage, i64, i64::MAX, |a: i64, b: i64| if a < b { a } else { b }, I64)
            },
            _ => Err(HoduError::UnsupportedDType {
                dtype: storage.get_dtype(),
                op: "reduce_window with Min reduction".to_string(),
            }),
        },
        WindowReduction::Sum => match storage {
            CpuStorage::F8E4M3(storage) => impl_reduce_window!(storage, F8E4M3, F8E4M3::ZERO, |a, b| a + b, F8E4M3),
            CpuStorage::F8E5M2(storage) => impl_reduce_window!(storage, F8E5M2, F8E5M2::ZERO, |a, b| a + b, F8E5M2),
            CpuStorage::F16(storage) => impl_reduce_window!(storage, f16, f16::ZERO, |a, b| a + b, F16),
            CpuStorage::BF16(storage) => impl_reduce_window!(storage, bf16, bf16::ZERO, |a, b| a + b, BF16),
            CpuStorage::F32(storage) => impl_reduce_window!(storage, f32, 0.0f32, |a, b| a + b, F32),
            CpuStorage::F64(storage) => impl_reduce_window!(storage, f64, 0.0f64, |a, b| a + b, F64),
            CpuStorage::U8(storage) => impl_reduce_window!(storage, u8, 0u8, |a, b| a + b, U8),
            CpuStorage::U16(storage) => impl_reduce_window!(storage, u16, 0u16, |a, b| a + b, U16),
            CpuStorage::U32(storage) => impl_reduce_window!(storage, u32, 0u32, |a, b| a + b, U32),
            CpuStorage::U64(storage) => impl_reduce_window!(storage, u64, 0u64, |a, b| a + b, U64),
            CpuStorage::I8(storage) => impl_reduce_window!(storage, i8, 0i8, |a, b| a + b, I8),
            CpuStorage::I16(storage) => impl_reduce_window!(storage, i16, 0i16, |a, b| a + b, I16),
            CpuStorage::I32(storage) => impl_reduce_window!(storage, i32, 0i32, |a, b| a + b, I32),
            CpuStorage::I64(storage) => impl_reduce_window!(storage, i64, 0i64, |a, b| a + b, I64),
            _ => Err(HoduError::UnsupportedDType {
                dtype: storage.get_dtype(),
                op: "reduce_window with Sum reduction".to_string(),
            }),
        },
        WindowReduction::Mean => {
            // First compute sum
            let sum_result = match storage {
                CpuStorage::F8E4M3(storage) => impl_reduce_window!(storage, F8E4M3, F8E4M3::ZERO, |a, b| a + b, F8E4M3),
                CpuStorage::F8E5M2(storage) => impl_reduce_window!(storage, F8E5M2, F8E5M2::ZERO, |a, b| a + b, F8E5M2),
                CpuStorage::F16(storage) => impl_reduce_window!(storage, f16, f16::ZERO, |a, b| a + b, F16),
                CpuStorage::BF16(storage) => impl_reduce_window!(storage, bf16, bf16::ZERO, |a, b| a + b, BF16),
                CpuStorage::F32(storage) => impl_reduce_window!(storage, f32, 0.0f32, |a, b| a + b, F32),
                CpuStorage::F64(storage) => impl_reduce_window!(storage, f64, 0.0f64, |a, b| a + b, F64),
                _ => Err(HoduError::UnsupportedDType {
                    dtype: storage.get_dtype(),
                    op: "reduce_window with Mean reduction".to_string(),
                }),
            }?;
            // Then divide by window size
            let window_size: usize = window_shape.iter().product();
            let output_layout = Layout::from_shape(&output_shape);
            let window_size_scalar = Scalar::from_f32(window_size as f32, sum_result.get_dtype());
            sum_result.unary_scalar_impl::<DivScalar>(&output_layout, window_size_scalar)
        },
    }
}
