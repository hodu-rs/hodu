use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

/// Global counter for generating unique DynamicDimId
static DIM_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Unique identifier for a dynamic dimension
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DynamicDimId(pub u32);

impl DynamicDimId {
    /// Creates a new unique DynamicDimId
    pub fn new() -> Self {
        Self(DIM_COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Returns the raw ID value
    pub fn id(&self) -> u32 {
        self.0
    }
}

impl Default for DynamicDimId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for DynamicDimId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DynamicDim({})", self.0)
    }
}

impl fmt::Display for DynamicDimId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?{}", self.0)
    }
}

/// A dimension that can be either concrete or dynamic
#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Dim {
    /// Concrete dimension with known size
    Concrete(usize),
    /// Dynamic dimension determined at runtime
    /// Contains optional upper bound for allocation
    Dynamic { id: DynamicDimId, max_bound: Option<usize> },
}

impl Dim {
    /// Creates a concrete dimension
    #[inline]
    pub fn concrete(size: usize) -> Self {
        Self::Concrete(size)
    }

    /// Creates a dynamic dimension with an optional max bound
    #[inline]
    pub fn dynamic(max_bound: Option<usize>) -> Self {
        Self::Dynamic {
            id: DynamicDimId::new(),
            max_bound,
        }
    }

    /// Creates a dynamic dimension with a specific ID
    #[inline]
    pub fn dynamic_with_id(id: DynamicDimId, max_bound: Option<usize>) -> Self {
        Self::Dynamic { id, max_bound }
    }

    /// Returns true if this dimension is dynamic
    #[inline]
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Self::Dynamic { .. })
    }

    /// Returns true if this dimension is concrete
    #[inline]
    pub fn is_concrete(&self) -> bool {
        matches!(self, Self::Concrete(_))
    }

    /// Returns the concrete size if this dimension is concrete
    #[inline]
    pub fn concrete_size(&self) -> Option<usize> {
        match self {
            Self::Concrete(size) => Some(*size),
            Self::Dynamic { .. } => None,
        }
    }

    /// Returns the size for allocation purposes
    /// For concrete dims, returns the size
    /// For dynamic dims, returns the max_bound if available
    #[inline]
    pub fn allocation_size(&self) -> Option<usize> {
        match self {
            Self::Concrete(size) => Some(*size),
            Self::Dynamic { max_bound, .. } => *max_bound,
        }
    }

    /// Returns the dynamic dimension ID if this is a dynamic dimension
    #[inline]
    pub fn dynamic_id(&self) -> Option<DynamicDimId> {
        match self {
            Self::Concrete(_) => None,
            Self::Dynamic { id, .. } => Some(*id),
        }
    }

    /// Returns the max bound for dynamic dimensions
    #[inline]
    pub fn max_bound(&self) -> Option<usize> {
        match self {
            Self::Concrete(_) => None,
            Self::Dynamic { max_bound, .. } => *max_bound,
        }
    }
}

impl From<usize> for Dim {
    fn from(size: usize) -> Self {
        Self::Concrete(size)
    }
}

impl fmt::Debug for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Concrete(size) => write!(f, "{}", size),
            Self::Dynamic { id, max_bound } => {
                write!(f, "{}", id)?;
                if let Some(bound) = max_bound {
                    write!(f, "(max={})", bound)?;
                }
                Ok(())
            },
        }
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Concrete(size) => write!(f, "{}", size),
            Self::Dynamic { id, max_bound } => {
                write!(f, "{}", id)?;
                if let Some(bound) = max_bound {
                    write!(f, "(max={})", bound)?;
                }
                Ok(())
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_dim_id_uniqueness() {
        let id1 = DynamicDimId::new();
        let id2 = DynamicDimId::new();
        let id3 = DynamicDimId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_dim_concrete() {
        let dim = Dim::concrete(10);
        assert!(dim.is_concrete());
        assert!(!dim.is_dynamic());
        assert_eq!(dim.concrete_size(), Some(10));
        assert_eq!(dim.allocation_size(), Some(10));
        assert!(dim.dynamic_id().is_none());
    }

    #[test]
    fn test_dim_dynamic() {
        let dim = Dim::dynamic(Some(100));
        assert!(!dim.is_concrete());
        assert!(dim.is_dynamic());
        assert_eq!(dim.concrete_size(), None);
        assert_eq!(dim.allocation_size(), Some(100));
        assert!(dim.dynamic_id().is_some());
        assert_eq!(dim.max_bound(), Some(100));
    }

    #[test]
    fn test_dim_dynamic_unbounded() {
        let dim = Dim::dynamic(None);
        assert!(dim.is_dynamic());
        assert_eq!(dim.allocation_size(), None);
        assert_eq!(dim.max_bound(), None);
    }
}
