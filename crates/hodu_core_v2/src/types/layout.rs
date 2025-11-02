use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{Op, ShapeOp},
    types::shape::Shape,
};

/// Layout describes the memory layout of a tensor, including shape, strides, and offset.
///
/// This struct manages how multi-dimensional tensor data maps to linear memory,
/// supporting non-contiguous views through stride manipulation.
#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct Layout {
    shape: Shape,
    strides: Vec<u32>,
    offset: u32,
}

impl Layout {
    /// Creates a new layout with given shape and strides.
    pub fn new(shape: Shape, strides: Vec<u32>) -> Self {
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    /// Creates a contiguous layout from a shape.
    pub fn from_shape(shape: &Shape) -> Self {
        let strides = Self::compute_strides(shape);
        Self {
            shape: shape.clone(),
            strides,
            offset: 0,
        }
    }

    /// Returns a reference to the shape.
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the strides as a slice.
    #[inline]
    pub fn strides(&self) -> &[u32] {
        &self.strides
    }

    /// Returns the offset into the underlying storage.
    #[inline]
    pub fn offset(&self) -> u32 {
        self.offset
    }

    /// Returns the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> u32 {
        self.shape.ndim()
    }

    /// Returns the size of a specific dimension.
    #[inline]
    pub fn dim(&self, index: u32) -> Option<u32> {
        self.shape.dim(index)
    }

    /// Returns the total number of elements.
    #[inline]
    pub fn size(&self) -> u32 {
        self.shape.size()
    }

    /// Returns the required buffer size (offset + size).
    pub fn buffer_size(&self) -> u32 {
        self.offset + self.size()
    }

    /// Sets a new shape (warning: does not update strides).
    pub fn set_shape(&mut self, shape: Shape) {
        self.shape = shape;
    }

    /// Sets new strides.
    pub fn set_strides(&mut self, strides: Vec<u32>) {
        self.strides = strides;
    }

    /// Sets a new offset.
    pub fn set_offset(&mut self, offset: u32) {
        self.offset = offset;
    }

    /// Checks if the layout is contiguous (C-order).
    pub fn is_contiguous(&self) -> bool {
        let ndim = self.ndim();
        if ndim == 0 {
            return true;
        }

        let mut expected_stride: u32 = 1;
        for i in (0..ndim).rev() {
            let idx = i as usize;
            if self.strides[idx] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }

        true
    }

    /// Computes contiguous strides for a given shape.
    pub fn compute_strides(shape: &Shape) -> Vec<u32> {
        let ndim = shape.ndim();
        if ndim == 0 {
            return vec![];
        }

        let ndim_usize = ndim as usize;
        let mut strides = vec![1; ndim_usize];
        for i in (0..ndim - 1).rev() {
            let idx = i as usize;
            let idx_next = (i + 1) as usize;
            strides[idx] = strides[idx_next] * shape[i + 1];
        }
        strides
    }

    /// Broadcasts this layout to a target shape.
    pub fn broadcast_to(&self, target_shape: &Shape) -> HoduResult<Self> {
        let shape = &self.shape;

        if shape.ndim() > target_shape.ndim() {
            return Err(HoduError::IncompatibleShapes {
                lhs: shape.clone(),
                rhs: target_shape.clone(),
                op: Op::Shape(ShapeOp::Broadcast),
            });
        }

        let rank_diff = target_shape.ndim() - shape.ndim();
        let rank_diff_usize = rank_diff as usize;
        let mut padded_shape = vec![1; rank_diff_usize];
        padded_shape.extend_from_slice(shape.dims());

        let target_ndim = target_shape.ndim();
        let target_ndim_usize = target_ndim as usize;
        let mut new_strides = vec![0; target_ndim_usize];

        for i in 0..target_ndim {
            let i_usize = i as usize;
            let src_dim = padded_shape[i_usize];
            let tgt_dim = target_shape[i];

            if src_dim == tgt_dim {
                if i < rank_diff {
                    new_strides[i_usize] = 0;
                } else {
                    new_strides[i_usize] = self.strides[i_usize - rank_diff_usize];
                }
            } else if src_dim == 1 {
                new_strides[i_usize] = 0;
            } else {
                return Err(HoduError::IncompatibleShapes {
                    lhs: shape.clone(),
                    rhs: target_shape.clone(),
                    op: Op::Shape(ShapeOp::Broadcast),
                });
            }
        }

        Ok(Self {
            shape: target_shape.clone(),
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Broadcasts two layouts to a common shape.
    pub fn broadcast_layouts(lhs: &Self, rhs: &Self) -> HoduResult<(Self, Self)> {
        let broadcast_shape =
            Shape::broadcast_shape(&lhs.shape, &rhs.shape).ok_or_else(|| HoduError::IncompatibleShapes {
                lhs: lhs.shape.clone(),
                rhs: rhs.shape.clone(),
                op: Op::Shape(ShapeOp::Broadcast),
            })?;

        let lhs_broadcast = lhs.broadcast_to(&broadcast_shape)?;
        let rhs_broadcast = rhs.broadcast_to(&broadcast_shape)?;

        Ok((lhs_broadcast, rhs_broadcast))
    }

    /// Transposes two dimensions.
    pub fn transpose(&self, dim1: i32, dim2: i32) -> HoduResult<Self> {
        let ndim = self.ndim();
        if ndim < 2 {
            return Err(HoduError::InternalError(
                "transpose requires at least 2 dimensions".to_string(),
            ));
        }

        let dim1 = self
            .shape
            .normalize_axis(dim1)
            .ok_or_else(|| HoduError::InternalError(format!("invalid axis: {}", dim1)))?;
        let dim2 = self
            .shape
            .normalize_axis(dim2)
            .ok_or_else(|| HoduError::InternalError(format!("invalid axis: {}", dim2)))?;

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        let dim1_usize = dim1 as usize;
        let dim2_usize = dim2 as usize;
        new_shape.dims_mut().swap(dim1_usize, dim2_usize);
        new_strides.swap(dim1_usize, dim2_usize);

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Permutes dimensions according to the given axes.
    pub fn permute(&self, axes: &[u32]) -> HoduResult<Self> {
        let ndim = self.ndim();

        if axes.len() as u32 != ndim {
            return Err(HoduError::InternalError(format!(
                "permute axes length {} must match tensor dimensions {}",
                axes.len(),
                ndim
            )));
        }

        // Validate axes
        let ndim_usize = ndim as usize;
        let mut seen = vec![false; ndim_usize];
        for &axis in axes {
            if axis >= ndim {
                return Err(HoduError::InternalError(format!(
                    "permute axis {} out of range for {}-dimensional tensor",
                    axis, ndim
                )));
            }
            let axis_usize = axis as usize;
            if seen[axis_usize] {
                return Err(HoduError::InternalError(format!(
                    "permute axis {} appears more than once",
                    axis
                )));
            }
            seen[axis_usize] = true;
        }

        // Permute shape and strides according to axes
        let mut new_dims = Vec::with_capacity(ndim_usize);
        let mut new_strides = Vec::with_capacity(ndim_usize);

        for &axis in axes {
            new_dims.push(self.shape[axis]);
            let axis_usize = axis as usize;
            new_strides.push(self.strides[axis_usize]);
        }

        Ok(Self {
            shape: Shape::from(new_dims),
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Slices a dimension with start, end, and step.
    pub fn slice(&self, dim: u32, start: i32, end: Option<i32>, step: i32) -> HoduResult<Self> {
        if dim >= self.ndim() {
            return Err(HoduError::InternalError(format!(
                "slice dimension {} out of range for {}-dimensional tensor",
                dim,
                self.ndim()
            )));
        }

        if step == 0 {
            return Err(HoduError::InternalError("slice step cannot be zero".to_string()));
        }

        let dim_size = self.shape[dim] as i32;

        // Normalize negative indices and handle None for end
        let start_idx = if start < 0 { dim_size + start } else { start };
        let end_idx = match end {
            Some(e) => {
                if e < 0 {
                    dim_size + e
                } else {
                    e
                }
            },
            None => {
                if step > 0 {
                    dim_size
                } else {
                    -1
                }
            },
        };

        // Clamp indices to valid range
        let clamped_start = start_idx.clamp(0, dim_size);
        let clamped_end = if step > 0 {
            end_idx.clamp(0, dim_size)
        } else {
            end_idx.clamp(-1, dim_size - 1)
        };

        // Calculate new size for the dimension
        let new_size = if step > 0 {
            if clamped_end > clamped_start {
                (clamped_end - clamped_start + step - 1) / step
            } else {
                0
            }
        } else if clamped_start > clamped_end {
            (clamped_start - clamped_end + (-step) - 1) / (-step)
        } else {
            0
        };

        if new_size <= 0 {
            return Err(HoduError::InternalError("slice results in empty tensor".to_string()));
        }

        // Create a new Layout with adjusted shape, strides and offset
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        let dim_usize = dim as usize;
        new_shape.dims_mut()[dim_usize] = new_size as u32;
        new_strides[dim_usize] = self.strides[dim_usize] * step.unsigned_abs();

        // Calculate new offset
        let new_offset = self.offset + clamped_start as u32 * self.strides[dim_usize];

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
        })
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shape: {}, strides: [", self.shape)?;
        for (i, &stride) in self.strides.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", stride)?;
        }
        write!(f, "], offset: {}", self.offset)
    }
}

impl fmt::Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Layout[shape: {:?}, strides: [", self.shape)?;
        for (i, &stride) in self.strides.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", stride)?;
        }
        write!(f, "], offset: {}]", self.offset)
    }
}
