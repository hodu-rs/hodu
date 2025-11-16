use crate::layer::compat::*;
use smallvec::SmallVec;

/// Shape represents the dimensions of a tensor using usize for native platform compatibility.
///
/// This struct provides a type-safe wrapper around dimension vectors
/// and includes utilities for shape manipulation and validation.
#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Shape {
    dims: SmallVec<[usize; 8]>,
}

#[cfg(feature = "serde")]
impl bincode::Encode for Shape {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.dims.as_slice(), encoder)
    }
}

#[cfg(feature = "serde")]
impl<Context> bincode::Decode<Context> for Shape {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let vec: Vec<usize> = bincode::Decode::decode(decoder)?;
        Ok(Shape {
            dims: SmallVec::from_vec(vec),
        })
    }
}

#[cfg(feature = "serde")]
impl<'de, Context> bincode::BorrowDecode<'de, Context> for Shape {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let vec: Vec<usize> = bincode::BorrowDecode::borrow_decode(decoder)?;
        Ok(Shape {
            dims: SmallVec::from_vec(vec),
        })
    }
}

impl Shape {
    /// Creates a new shape from a slice of dimensions.
    #[inline]
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: SmallVec::from_slice(dims),
        }
    }

    /// Creates a scalar shape (0 dimensions).
    #[inline]
    pub fn scalar() -> Self {
        Self { dims: SmallVec::new() }
    }

    /// Returns the dimensions as a slice.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns a mutable reference to the dimensions.
    #[inline]
    pub fn dims_mut(&mut self) -> &mut SmallVec<[usize; 8]> {
        &mut self.dims
    }

    /// Converts the shape to a Vec<usize>.
    #[inline]
    pub fn to_vec(&self) -> Vec<usize> {
        self.dims.to_vec()
    }

    /// Returns the last dimension, or None if the shape is scalar.
    #[inline]
    pub fn last(&self) -> Option<usize> {
        self.dims.last().copied()
    }

    /// Returns the number of dimensions (rank).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the size of a specific dimension.
    #[inline]
    pub fn dim_size(&self, index: i32) -> Option<usize> {
        let index = if index < 0 { self.ndim() as i32 + index } else { index } as usize;
        self.dims.get(index).copied()
    }

    /// Returns the total number of elements.
    #[inline]
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns true if this is a scalar (0 dimensions).
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Checks if this shape can be broadcast to the target shape.
    #[inline]
    pub fn can_broadcast_to(&self, target: &Shape) -> bool {
        if self.ndim() > target.ndim() {
            return false;
        }

        let offset = target.ndim() - self.ndim();
        for (i, &dim) in self.dims.iter().enumerate() {
            let target_idx = i + offset;
            let target_dim = target.dims[target_idx];
            if dim != 1 && dim != target_dim {
                return false;
            }
        }
        true
    }

    /// Computes the broadcast shape between two shapes.
    #[inline]
    pub fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> Option<Shape> {
        let max_ndim = lhs.ndim().max(rhs.ndim());
        let mut result_dims = SmallVec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let lhs_idx = lhs.ndim().saturating_sub(max_ndim - i);
            let rhs_idx = rhs.ndim().saturating_sub(max_ndim - i);

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
    #[inline]
    pub fn normalize_axis(&self, axis: i32) -> Option<usize> {
        let ndim = self.ndim() as i32;
        let normalized = if axis < 0 { ndim + axis } else { axis };

        if normalized >= 0 && normalized < ndim {
            Some(normalized as usize)
        } else {
            None
        }
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self {
            dims: SmallVec::from_vec(dims),
        }
    }
}

impl From<SmallVec<[usize; 8]>> for Shape {
    fn from(dims: SmallVec<[usize; 8]>) -> Self {
        Self { dims }
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims)
    }
}

impl From<&Vec<usize>> for Shape {
    fn from(dims: &Vec<usize>) -> Self {
        Self::from(dims.as_slice())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        shape.clone()
    }
}

// i8
impl<const N: usize> From<[i8; N]> for Shape {
    fn from(dims: [i8; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

impl<const N: usize> From<&[i8; N]> for Shape {
    fn from(dims: &[i8; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

// i16
#[cfg(feature = "i16")]
impl<const N: usize> From<[i16; N]> for Shape {
    fn from(dims: [i16; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

#[cfg(feature = "i16")]
impl<const N: usize> From<&[i16; N]> for Shape {
    fn from(dims: &[i16; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

// i32
impl<const N: usize> From<[i32; N]> for Shape {
    fn from(dims: [i32; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

impl<const N: usize> From<&[i32; N]> for Shape {
    fn from(dims: &[i32; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

// i64
#[cfg(feature = "i64")]
impl<const N: usize> From<[i64; N]> for Shape {
    fn from(dims: [i64; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

#[cfg(feature = "i64")]
impl<const N: usize> From<&[i64; N]> for Shape {
    fn from(dims: &[i64; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

// isize
impl<const N: usize> From<[isize; N]> for Shape {
    fn from(dims: [isize; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

impl<const N: usize> From<&[isize; N]> for Shape {
    fn from(dims: &[isize; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

// u8
impl<const N: usize> From<[u8; N]> for Shape {
    fn from(dims: [u8; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

impl<const N: usize> From<&[u8; N]> for Shape {
    fn from(dims: &[u8; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

// u16
#[cfg(feature = "u16")]
impl<const N: usize> From<[u16; N]> for Shape {
    fn from(dims: [u16; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

#[cfg(feature = "u16")]
impl<const N: usize> From<&[u16; N]> for Shape {
    fn from(dims: &[u16; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

// u32
impl<const N: usize> From<[u32; N]> for Shape {
    fn from(dims: [u32; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

impl<const N: usize> From<&[u32; N]> for Shape {
    fn from(dims: &[u32; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

// u64
#[cfg(feature = "u64")]
impl<const N: usize> From<[u64; N]> for Shape {
    fn from(dims: [u64; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

#[cfg(feature = "u64")]
impl<const N: usize> From<&[u64; N]> for Shape {
    fn from(dims: &[u64; N]) -> Self {
        let dims: SmallVec<[usize; 8]> = dims.iter().map(|&d| d as usize).collect();
        Self { dims }
    }
}

// usize
impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Self::from(dims.as_slice())
    }
}

impl<const N: usize> From<&[usize; N]> for Shape {
    fn from(dims: &[usize; N]) -> Self {
        Self::from(dims.as_slice())
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.dims
    }
}

impl core::ops::Index<usize> for Shape {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
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
