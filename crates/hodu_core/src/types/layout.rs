use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    ops::{Op, ShapeOp},
    types::shape::Shape,
};
use smallvec::SmallVec;

/// Layout describes the memory layout of a tensor, including shape, strides, and offset.
///
/// This struct manages how multi-dimensional tensor data maps to linear memory,
/// supporting non-contiguous views through stride manipulation.
#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Layout {
    shape: Shape,
    strides: SmallVec<[usize; 8]>,
    offset: usize,
}

impl Layout {
    /// Creates a new layout with given shape and strides.
    pub fn new(shape: Shape, strides: Vec<usize>) -> Self {
        Self {
            shape,
            strides: SmallVec::from_vec(strides),
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
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the offset into the underlying storage.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Returns the size of a specific dimension.
    #[inline]
    pub fn dim_size(&self, index: i32) -> Option<usize> {
        let ndim = self.ndim() as i32;
        let normalized = if index < 0 { ndim + index } else { index };

        if normalized >= 0 && normalized < ndim {
            Some(self.shape[normalized as usize])
        } else {
            None
        }
    }

    /// Returns the total number of elements.
    #[inline]
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Returns the required buffer size (offset + size).
    pub fn buffer_size(&self) -> usize {
        self.offset + self.size()
    }

    /// Sets a new shape (warning: does not update strides).
    pub fn set_shape(&mut self, shape: Shape) {
        self.shape = shape;
    }

    /// Sets new strides.
    pub fn set_strides(&mut self, strides: Vec<usize>) {
        self.strides = SmallVec::from_vec(strides);
    }

    /// Sets a new offset.
    pub fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }

    /// Checks if the layout is contiguous (C-order).
    pub fn is_contiguous(&self) -> bool {
        let ndim = self.ndim();
        if ndim == 0 {
            return true;
        }

        let mut expected_stride = 1;
        for i in (0..ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }

        true
    }

    /// Computes contiguous strides for a given shape.
    pub fn compute_strides(shape: &Shape) -> SmallVec<[usize; 8]> {
        let ndim = shape.ndim();
        if ndim == 0 {
            return SmallVec::new();
        }

        let mut strides = SmallVec::from_elem(1, ndim);
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Reshapes the layout to a new shape (view operation).
    pub fn reshape(&self, new_shape: &Shape) -> HoduResult<Self> {
        let old_size = self.size();
        let new_size = new_shape.size();

        // Check that total number of elements is the same
        if old_size != new_size {
            return Err(HoduError::incompatible_shapes(
                self.shape.clone(),
                new_shape.clone(),
                Op::Shape(ShapeOp::Reshape),
            ));
        }

        // Reshape only works on contiguous tensors
        if !self.is_contiguous() {
            return Err(HoduError::InvalidLayout {
                reason: "Cannot reshape a non-contiguous tensor. Use .contiguous() before .reshape()".to_string(),
            });
        }

        // Create new layout with the new shape and recomputed strides
        let new_strides = Self::compute_strides(new_shape);

        Ok(Self {
            shape: new_shape.clone(),
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Flattens the layout to 1D.
    pub fn flatten(&self) -> HoduResult<Self> {
        let total_size = self.size();
        let new_shape = Shape::from(vec![total_size]);

        // Flatten only works on contiguous tensors
        if !self.is_contiguous() {
            return Err(HoduError::InvalidLayout {
                reason: "Cannot flatten a non-contiguous tensor. Use .contiguous() before .flatten()".to_string(),
            });
        }

        let new_strides = Self::compute_strides(&new_shape);

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Squeezes dimensions of size 1.
    ///
    /// If `dims` is empty, all dimensions of size 1 are removed.
    /// If `dims` is not empty, only the specified dimensions are removed (if they have size 1).
    /// The layout must be contiguous for squeezing to succeed.
    pub fn squeeze(&self, dims_to_squeeze: &[i32]) -> HoduResult<Self> {
        let ndim = self.ndim();
        let shape_dims = self.shape.dims();

        // Squeeze only works on contiguous tensors
        if !self.is_contiguous() {
            return Err(HoduError::InvalidLayout {
                reason: "Cannot squeeze a non-contiguous tensor. Use .contiguous() before .squeeze()".to_string(),
            });
        }

        let new_dims = if dims_to_squeeze.is_empty() {
            // Squeeze all dimensions of size 1
            shape_dims.iter().filter(|&&size| size != 1).copied().collect()
        } else {
            // Normalize and validate dimensions to squeeze
            let mut actual_dims = Vec::with_capacity(dims_to_squeeze.len().min(8));
            for &dim in dims_to_squeeze {
                let actual_dim = if dim < 0 {
                    (ndim as i32 + dim) as usize
                } else {
                    dim as usize
                };

                if actual_dim >= ndim {
                    return Err(HoduError::InvalidAxis { axis: dim, ndim });
                }

                if shape_dims[actual_dim] != 1 {
                    return Err(HoduError::InvalidLayout {
                        reason: format!("Cannot squeeze dimension {} with size {}", dim, shape_dims[actual_dim]),
                    });
                }

                actual_dims.push(actual_dim);
            }

            // Sort dimensions in descending order to remove from back to front
            actual_dims.sort_unstable();
            actual_dims.reverse();

            let mut new_dims = shape_dims.to_vec();
            for &dim_idx in &actual_dims {
                new_dims.remove(dim_idx);
            }
            new_dims
        };

        let new_shape = Shape::from(new_dims);
        let new_strides = Self::compute_strides(&new_shape);

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Unsqueezes (adds) a dimension of size 1 at the specified position.
    pub fn unsqueeze(&self, dim: i32) -> HoduResult<Self> {
        let ndim = self.ndim();
        let dims = self.shape.dims();

        // Convert negative dimension to positive
        let actual_dim = if dim < 0 {
            (ndim as i32 + dim + 1) as usize
        } else {
            dim as usize
        };

        // Check bounds (can insert at position 0 to ndim inclusive)
        if actual_dim > ndim {
            return Err(HoduError::InvalidAxis { axis: dim, ndim });
        }

        // Create new shape with dimension of size 1 inserted
        let mut new_dims = Vec::with_capacity((ndim + 1).min(8));
        new_dims.extend_from_slice(dims);
        new_dims.insert(actual_dim, 1);

        // Unsqueeze only works on contiguous tensors
        if !self.is_contiguous() {
            return Err(HoduError::InvalidLayout {
                reason: "Cannot unsqueeze a non-contiguous tensor. Use .contiguous() before .unsqueeze()".to_string(),
            });
        }

        let new_shape = Shape::from(new_dims);
        let new_strides = Self::compute_strides(&new_shape);

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Broadcasts this layout to a target shape.
    pub fn broadcast_to(&self, target_shape: &Shape) -> HoduResult<Self> {
        let shape = &self.shape;

        if shape.ndim() > target_shape.ndim() {
            return Err(HoduError::incompatible_shapes(
                shape.clone(),
                target_shape.clone(),
                Op::Shape(ShapeOp::Broadcast),
            ));
        }

        let rank_diff = target_shape.ndim() - shape.ndim();
        let target_ndim = target_shape.ndim();
        let mut padded_shape = Vec::with_capacity(target_ndim.min(8));
        padded_shape.resize(rank_diff, 1);
        padded_shape.extend_from_slice(shape.dims());

        let mut new_strides = Vec::with_capacity(target_ndim.min(8));
        new_strides.resize(target_ndim, 0);

        for i in 0..target_ndim {
            let src_dim = padded_shape[i];
            let tgt_dim = target_shape[i];

            if src_dim == tgt_dim {
                if i < rank_diff {
                    new_strides[i] = 0;
                } else {
                    new_strides[i] = self.strides[i - rank_diff];
                }
            } else if src_dim == 1 {
                new_strides[i] = 0;
            } else {
                return Err(HoduError::incompatible_shapes(
                    shape.clone(),
                    target_shape.clone(),
                    Op::Shape(ShapeOp::Broadcast),
                ));
            }
        }

        Ok(Self {
            shape: target_shape.clone(),
            strides: SmallVec::from_vec(new_strides),
            offset: self.offset,
        })
    }

    /// Broadcasts two layouts to a common shape.
    pub fn broadcast_layouts(lhs: &Self, rhs: &Self) -> HoduResult<(Self, Self)> {
        let broadcast_shape = Shape::broadcast_shape(&lhs.shape, &rhs.shape).ok_or_else(|| {
            HoduError::incompatible_shapes(lhs.shape.clone(), rhs.shape.clone(), Op::Shape(ShapeOp::Broadcast))
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

        new_shape.dims_mut().swap(dim1, dim2);
        new_strides.swap(dim1, dim2);

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Permutes dimensions according to the given axes.
    pub fn permute(&self, axes: &[i32]) -> HoduResult<Self> {
        let ndim = self.ndim();

        if axes.len() != ndim {
            return Err(HoduError::InternalError(format!(
                "permute axes length {} must match tensor dimensions {}",
                axes.len(),
                ndim
            )));
        }

        // Normalize negative axes and validate
        let mut actual_axes = Vec::with_capacity(ndim.min(8));
        let mut seen = Vec::with_capacity(ndim.min(8));
        seen.resize(ndim, false);

        for &axis in axes {
            let actual_axis = self
                .shape
                .normalize_axis(axis)
                .ok_or(HoduError::InvalidAxis { axis, ndim })?;

            if seen[actual_axis] {
                return Err(HoduError::InternalError(format!(
                    "permute axis {} appears more than once",
                    axis
                )));
            }
            seen[actual_axis] = true;
            actual_axes.push(actual_axis);
        }

        // Permute shape and strides according to axes
        let mut new_dims = Vec::with_capacity(ndim.min(8));
        let mut new_strides = Vec::with_capacity(ndim.min(8));

        for &axis_idx in &actual_axes {
            new_dims.push(self.shape.dims()[axis_idx]);
            new_strides.push(self.strides[axis_idx]);
        }

        Ok(Self {
            shape: Shape::from(new_dims),
            strides: SmallVec::from_vec(new_strides),
            offset: self.offset,
        })
    }

    /// Slices a dimension with start, end, and step.
    pub fn slice(&self, dim: i32, start: i32, end: Option<i32>, step: i32) -> HoduResult<Self> {
        let ndim = self.ndim();

        // Normalize negative dim
        let actual_dim = if dim < 0 {
            (ndim as i32 + dim) as usize
        } else {
            dim as usize
        };

        if actual_dim >= ndim {
            return Err(HoduError::InvalidAxis { axis: dim, ndim });
        }

        if step == 0 {
            return Err(HoduError::InternalError("slice step cannot be zero".to_string()));
        }

        let dim_size = self.shape.dims()[actual_dim] as i32;

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

        new_shape.dims_mut()[actual_dim] = new_size as usize;
        new_strides[actual_dim] = self.strides[actual_dim] * step.unsigned_abs() as usize;

        // Calculate new offset
        let new_offset = self.offset + clamped_start as usize * self.strides[actual_dim];

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
