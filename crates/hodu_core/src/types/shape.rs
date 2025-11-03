use crate::layer::compat::*;

/// Shape represents the dimensions of a tensor using u32 for memory efficiency.
///
/// This struct provides a type-safe wrapper around dimension vectors
/// and includes utilities for shape manipulation and validation.
#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct Shape {
    dims: Vec<u32>,
}

impl Shape {
    /// Creates a new shape from a slice of dimensions.
    #[inline]
    pub fn new(dims: &[u32]) -> Self {
        Self { dims: dims.to_vec() }
    }

    /// Creates a scalar shape (0 dimensions).
    #[inline]
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    /// Returns the dimensions as a slice.
    #[inline]
    pub fn dims(&self) -> &[u32] {
        &self.dims
    }

    /// Returns a mutable reference to the dimensions.
    #[inline]
    pub fn dims_mut(&mut self) -> &mut Vec<u32> {
        &mut self.dims
    }

    /// Returns the number of dimensions (rank).
    #[inline]
    pub fn ndim(&self) -> u32 {
        self.dims.len() as u32
    }

    /// Returns the size of a specific dimension.
    #[inline]
    pub fn dim(&self, index: u32) -> Option<u32> {
        self.dims.get(index as usize).copied()
    }

    /// Returns the total number of elements.
    #[inline]
    pub fn size(&self) -> u32 {
        self.dims.iter().product()
    }

    /// Returns true if this is a scalar (0 dimensions).
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Checks if this shape can be broadcast to the target shape.
    pub fn can_broadcast_to(&self, target: &Shape) -> bool {
        if self.ndim() > target.ndim() {
            return false;
        }

        let offset = target.ndim() - self.ndim();
        for (i, &dim) in self.dims.iter().enumerate() {
            let target_idx = (i as u32 + offset) as usize;
            let target_dim = target.dims[target_idx];
            if dim != 1 && dim != target_dim {
                return false;
            }
        }
        true
    }

    /// Computes the broadcast shape between two shapes.
    pub fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> Option<Shape> {
        let max_ndim = lhs.ndim().max(rhs.ndim());
        let max_ndim_usize = max_ndim as usize;
        let mut result_dims = Vec::with_capacity(max_ndim_usize);

        for i in 0..max_ndim {
            let lhs_idx = lhs.ndim().saturating_sub(max_ndim - i) as usize;
            let rhs_idx = rhs.ndim().saturating_sub(max_ndim - i) as usize;

            let lhs_dim = lhs.dims.get(lhs_idx).copied().unwrap_or(1);
            let rhs_dim = rhs.dims.get(rhs_idx).copied().unwrap_or(1);

            if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
                return None;
            }
            result_dims.push(lhs_dim.max(rhs_dim));
        }

        Some(Shape { dims: result_dims })
    }

    /// Normalizes a dimension index, handling negative indices.
    pub fn normalize_axis(&self, axis: i32) -> Option<u32> {
        let ndim = self.ndim() as i32;
        let normalized = if axis < 0 { ndim + axis } else { axis };

        if normalized >= 0 && normalized < ndim {
            Some(normalized as u32)
        } else {
            None
        }
    }
}

impl From<Vec<u32>> for Shape {
    fn from(dims: Vec<u32>) -> Self {
        Self { dims }
    }
}

impl From<&[u32]> for Shape {
    fn from(dims: &[u32]) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        let dims: Vec<u32> = dims.iter().map(|&d| d as u32).collect();
        Self { dims }
    }
}

impl From<&Vec<usize>> for Shape {
    fn from(dims: &Vec<usize>) -> Self {
        Self::from(dims.as_slice())
    }
}

impl AsRef<[u32]> for Shape {
    fn as_ref(&self) -> &[u32] {
        &self.dims
    }
}

impl core::ops::Index<u32> for Shape {
    type Output = u32;

    fn index(&self, index: u32) -> &Self::Output {
        &self.dims[index as usize]
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, &dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape[")?;
        fmt::Display::fmt(self, f)?;
        write!(f, "]")
    }
}
