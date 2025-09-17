use crate::{compat::*, tensor::Tensor, types::dtype::DType};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};

const MAX_ELEMENTS_PER_DIM: usize = 4;

fn format_float<T>(value: T) -> String
where
    T: fmt::Display + Copy + Into<f64>,
{
    let val = value.into();
    if val.fract() == 0.0 {
        format!("{val:.0}.")
    } else {
        let formatted = format!("{val:.6}");
        let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
        if trimmed.contains('.') {
            trimmed.to_string()
        } else {
            format!("{trimmed}.")
        }
    }
}

trait FormatValue {
    fn format_value(&self, use_float_format: bool) -> String;
    fn format_value_with_precision(&self, use_float_format: bool, precision: Option<usize>) -> String;
}

impl FormatValue for bool {
    fn format_value(&self, _use_float_format: bool) -> String {
        format!("{self}")
    }

    fn format_value_with_precision(&self, _use_float_format: bool, _precision: Option<usize>) -> String {
        format!("{self}")
    }
}

impl FormatValue for F8E4M3 {
    fn format_value(&self, use_float_format: bool) -> String {
        let val = f32::from(*self);
        if use_float_format {
            format_float(val)
        } else {
            format!("{val}")
        }
    }

    fn format_value_with_precision(&self, use_float_format: bool, precision: Option<usize>) -> String {
        let val = f32::from(*self);
        if let Some(precision) = precision {
            format!("{val:.precision$}")
        } else {
            self.format_value(use_float_format)
        }
    }
}

impl FormatValue for F8E5M2 {
    fn format_value(&self, use_float_format: bool) -> String {
        let val = f32::from(*self);
        if use_float_format {
            format_float(val)
        } else {
            format!("{val}")
        }
    }

    fn format_value_with_precision(&self, use_float_format: bool, precision: Option<usize>) -> String {
        let val = f32::from(*self);
        if let Some(precision) = precision {
            format!("{val:.precision$}")
        } else {
            self.format_value(use_float_format)
        }
    }
}

impl FormatValue for bf16 {
    fn format_value(&self, use_float_format: bool) -> String {
        let val = f32::from(*self);
        if use_float_format {
            format_float(val)
        } else {
            format!("{val}")
        }
    }

    fn format_value_with_precision(&self, use_float_format: bool, precision: Option<usize>) -> String {
        let val = f32::from(*self);
        if let Some(precision) = precision {
            format!("{val:.precision$}")
        } else {
            self.format_value(use_float_format)
        }
    }
}

impl FormatValue for f16 {
    fn format_value(&self, use_float_format: bool) -> String {
        let val = f32::from(*self);
        if use_float_format {
            format_float(val)
        } else {
            format!("{val}")
        }
    }

    fn format_value_with_precision(&self, use_float_format: bool, precision: Option<usize>) -> String {
        let val = f32::from(*self);
        if let Some(precision) = precision {
            format!("{val:.precision$}")
        } else {
            self.format_value(use_float_format)
        }
    }
}

impl FormatValue for f32 {
    fn format_value(&self, use_float_format: bool) -> String {
        if use_float_format {
            format_float(*self)
        } else {
            format!("{self}")
        }
    }

    fn format_value_with_precision(&self, use_float_format: bool, precision: Option<usize>) -> String {
        if let Some(precision) = precision {
            format!("{self:.precision$}")
        } else {
            self.format_value(use_float_format)
        }
    }
}

impl FormatValue for f64 {
    fn format_value(&self, use_float_format: bool) -> String {
        if use_float_format {
            format_float(*self)
        } else {
            format!("{self}")
        }
    }

    fn format_value_with_precision(&self, use_float_format: bool, precision: Option<usize>) -> String {
        if let Some(precision) = precision {
            format!("{self:.precision$}")
        } else {
            self.format_value(use_float_format)
        }
    }
}

macro_rules! impl_format_value_for_int {
    ($($t:ty),*) => {
        $(
            impl FormatValue for $t {
                fn format_value(&self, _use_float_format: bool) -> String {
                    format!("{}", self)
                }

                fn format_value_with_precision(&self, _use_float_format: bool, _precision: Option<usize>) -> String {
                    format!("{}", self)
                }
            }
        )*
    };
}

impl_format_value_for_int!(u8, u16, u32, u64, i8, i16, i32, i64);

fn display_tensor_data<T>(
    f: &mut fmt::Formatter<'_>,
    data: &[T],
    stride: usize,
    shape: &[usize],
    depth: usize,
    use_float_format: bool,
    column_width: Option<usize>,
) -> fmt::Result
where
    T: FormatValue,
{
    let precision = f.precision();
    match shape.len() {
        0 => {
            write!(
                f,
                "{}",
                data[0].format_value_with_precision(use_float_format, precision)
            )
        },
        1 => {
            write!(f, "[")?;
            let len = data.len();
            let (show_start, show_end) = if len <= MAX_ELEMENTS_PER_DIM * 2 {
                (len, 0)
            } else {
                (MAX_ELEMENTS_PER_DIM, MAX_ELEMENTS_PER_DIM)
            };

            let width = column_width.unwrap_or_else(|| {
                data.iter()
                    .map(|val| val.format_value_with_precision(use_float_format, precision).len())
                    .max()
                    .unwrap_or(0)
            });

            for (i, item) in data.iter().enumerate().take(show_start.min(len)) {
                if i > 0 {
                    write!(f, ", ")?;
                }
                let formatted = item.format_value_with_precision(use_float_format, precision);
                write!(f, "{formatted:width$}")?;
            }

            if len > show_start + show_end {
                write!(f, ", ... ")?;
            }

            if show_end > 0 && len > show_start + show_end {
                for item in data.iter().skip(len - show_end) {
                    write!(f, ", ")?;
                    let formatted = item.format_value_with_precision(use_float_format, precision);
                    write!(f, "{formatted:width$}")?;
                }
            }
            write!(f, "]")
        },
        _ => {
            let sub_stride = stride / shape[0];
            write!(f, "[")?;
            let dim_size = shape[0];
            let (show_start, show_end) = if dim_size <= MAX_ELEMENTS_PER_DIM * 2 {
                (dim_size, 0)
            } else {
                (MAX_ELEMENTS_PER_DIM, MAX_ELEMENTS_PER_DIM)
            };

            let width = if shape.len() == 2 {
                let mut max_width = 0;
                for i in 0..shape[0].min(3) {
                    for j in 0..shape[1].min(3) {
                        let idx = i * sub_stride + j;
                        if idx < data.len() {
                            let formatted = data[idx].format_value_with_precision(use_float_format, precision);
                            max_width = max_width.max(formatted.len());
                        }
                    }
                }
                Some(max_width)
            } else {
                None
            };

            for i in 0..show_start.min(dim_size) {
                if i > 0 {
                    if depth < 2 {
                        write!(f, ",\n{}", " ".repeat(depth + 1))?;
                    } else {
                        write!(f, ", ")?;
                    }
                }
                display_tensor_data(
                    f,
                    &data[i * sub_stride..(i + 1) * sub_stride],
                    sub_stride,
                    &shape[1..],
                    depth + 1,
                    use_float_format,
                    width,
                )?;
            }

            if dim_size > show_start + show_end {
                if depth < 2 {
                    write!(f, ",\n{}", " ".repeat(depth + 1))?;
                    write!(f, "...")?;
                } else {
                    write!(f, ", ... ")?;
                }
            }

            if show_end > 0 && dim_size > show_start + show_end {
                for i in (dim_size - show_end)..dim_size {
                    if depth < 2 {
                        write!(f, ",\n{}", " ".repeat(depth + 1))?;
                    } else {
                        write!(f, ", ")?;
                    }
                    display_tensor_data(
                        f,
                        &data[i * sub_stride..(i + 1) * sub_stride],
                        sub_stride,
                        &shape[1..],
                        depth + 1,
                        use_float_format,
                        width,
                    )?;
                }
            }
            write!(f, "]")
        },
    }
}

fn debug_format_data<T>(data: &[T], use_float_format: bool) -> String
where
    T: FormatValue,
{
    let mut result = String::from("[");
    let len = data.len();

    let (show_start, show_end) = if len <= MAX_ELEMENTS_PER_DIM * 2 {
        (len, 0)
    } else {
        (MAX_ELEMENTS_PER_DIM, MAX_ELEMENTS_PER_DIM)
    };

    for (i, item) in data.iter().enumerate().take(show_start.min(len)) {
        if i > 0 {
            result.push_str(", ");
        }
        result.push_str(&item.format_value(use_float_format));
    }

    if len > show_start + show_end {
        result.push_str(", ... ");
    }

    if show_end > 0 && len > show_start + show_end {
        for item in data.iter().skip(len - show_end) {
            result.push_str(", ");
            result.push_str(&item.format_value(use_float_format));
        }
    }

    result.push(']');
    result
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let has_storage =
            crate::tensor::with_tensor(self.0, |tensor_ref| tensor_ref.storage.is_some()).unwrap_or(false);
        if !has_storage {
            write!(f, "<none>")
        } else {
            let layout = self.get_layout();
            match self.get_dtype() {
                DType::BOOL => {
                    if let Ok(data) = self.to_flatten_vec::<bool>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::F8E4M3 => {
                    if let Ok(data) = self.to_flatten_vec::<F8E4M3>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, true, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::F8E5M2 => {
                    if let Ok(data) = self.to_flatten_vec::<F8E5M2>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, true, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::BF16 => {
                    if let Ok(data) = self.to_flatten_vec::<bf16>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, true, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::F16 => {
                    if let Ok(data) = self.to_flatten_vec::<f16>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, true, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::F32 => {
                    if let Ok(data) = self.to_flatten_vec::<f32>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, true, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::F64 => {
                    if let Ok(data) = self.to_flatten_vec::<f64>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, true, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::U8 => {
                    if let Ok(data) = self.to_flatten_vec::<u8>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::U16 => {
                    if let Ok(data) = self.to_flatten_vec::<u16>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::U32 => {
                    if let Ok(data) = self.to_flatten_vec::<u32>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::U64 => {
                    if let Ok(data) = self.to_flatten_vec::<u64>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::I8 => {
                    if let Ok(data) = self.to_flatten_vec::<i8>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::I16 => {
                    if let Ok(data) = self.to_flatten_vec::<i16>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::I32 => {
                    if let Ok(data) = self.to_flatten_vec::<i32>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
                DType::I64 => {
                    if let Ok(data) = self.to_flatten_vec::<i64>() {
                        display_tensor_data(f, &data, layout.get_size(), layout.get_shape(), 0, false, None)
                    } else {
                        write!(f, "<error>")
                    }
                },
            }
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(device={}, dtype={}, shape=[",
            self.get_device(),
            self.get_dtype(),
        )?;
        let layout = self.get_layout();
        let shape = layout.get_shape();
        for (i, dim) in shape.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?
            }
            write!(f, "{dim}")?;
        }
        write!(f, "], data=")?;

        let has_storage =
            crate::tensor::with_tensor(self.0, |tensor_ref| tensor_ref.storage.is_some()).unwrap_or(false);
        if !has_storage {
            write!(f, "<none>")?;
        } else {
            match self.get_dtype() {
                DType::BOOL => {
                    if let Ok(data) = self.to_flatten_vec::<bool>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::F8E4M3 => {
                    if let Ok(data) = self.to_flatten_vec::<F8E4M3>() {
                        write!(f, "{}", debug_format_data(&data, true))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::F8E5M2 => {
                    if let Ok(data) = self.to_flatten_vec::<F8E5M2>() {
                        write!(f, "{}", debug_format_data(&data, true))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::BF16 => {
                    if let Ok(data) = self.to_flatten_vec::<bf16>() {
                        write!(f, "{}", debug_format_data(&data, true))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::F16 => {
                    if let Ok(data) = self.to_flatten_vec::<f16>() {
                        write!(f, "{}", debug_format_data(&data, true))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::F32 => {
                    if let Ok(data) = self.to_flatten_vec::<f32>() {
                        write!(f, "{}", debug_format_data(&data, true))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::F64 => {
                    if let Ok(data) = self.to_flatten_vec::<f64>() {
                        write!(f, "{}", debug_format_data(&data, true))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::U8 => {
                    if let Ok(data) = self.to_flatten_vec::<u8>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::U16 => {
                    if let Ok(data) = self.to_flatten_vec::<u16>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::U32 => {
                    if let Ok(data) = self.to_flatten_vec::<u32>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::U64 => {
                    if let Ok(data) = self.to_flatten_vec::<u64>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::I8 => {
                    if let Ok(data) = self.to_flatten_vec::<i8>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::I16 => {
                    if let Ok(data) = self.to_flatten_vec::<i16>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::I32 => {
                    if let Ok(data) = self.to_flatten_vec::<i32>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
                DType::I64 => {
                    if let Ok(data) = self.to_flatten_vec::<i64>() {
                        write!(f, "{}", debug_format_data(&data, false))?;
                    } else {
                        write!(f, "<error>")?;
                    }
                },
            }
        }

        write!(f, ")")
    }
}
