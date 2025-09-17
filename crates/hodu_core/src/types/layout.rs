use crate::{
    compat::*,
    error::{HoduError, HoduResult},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Layout {
    pub fn new(shape: &[usize], strides: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            offset: 0,
        }
    }

    pub fn from_shape(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
            offset: 0,
        }
    }

    pub fn get_shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn get_strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn get_offset(&self) -> usize {
        self.offset
    }

    pub fn get_ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn get_dim_size(&self, dim: usize) -> Option<usize> {
        self.shape.get(dim).copied()
    }

    pub fn get_size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn set_shape(&mut self, shape: &[usize]) {
        self.shape = shape.to_vec();
    }

    pub fn set_strides(&mut self, strides: &[usize]) {
        self.strides = strides.to_vec();
    }

    pub fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }

    pub fn is_contiguous(&self) -> bool {
        if self.get_ndim() == 0 {
            return true;
        }

        let mut expected_stride = 1;
        for i in (0..self.get_ndim()).rev() {
            if self.get_strides()[i] != expected_stride {
                return false;
            }
            expected_stride *= self.get_shape()[i];
        }

        true
    }

    pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() {
            return vec![];
        }

        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn broadcast_to(&self, target_shape: &[usize]) -> HoduResult<Self> {
        let shape = self.get_shape();

        if shape.len() > target_shape.len() {
            return Err(HoduError::IncompatibleShapes {
                lhs: shape.to_vec(),
                rhs: target_shape.to_vec(),
                op: "broadcast".to_string(),
            });
        }

        let rank_diff = target_shape.len() - shape.len();
        let mut padded_shape = vec![1; rank_diff];
        padded_shape.extend_from_slice(shape);

        let mut new_strides = vec![0; target_shape.len()];

        for i in 0..target_shape.len() {
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
                return Err(HoduError::IncompatibleShapes {
                    lhs: shape.to_vec(),
                    rhs: target_shape.to_vec(),
                    op: format!("broadcast at dimension {}", i),
                });
            }
        }

        Ok(Self {
            shape: target_shape.to_vec(),
            strides: new_strides,
            offset: self.offset,
        })
    }

    pub fn broadcast_layouts(lhs: &Self, rhs: &Self) -> HoduResult<(Self, Self)> {
        let lhs_shape = &lhs.shape;
        let rhs_shape = &rhs.shape;

        let max_rank = lhs_shape.len().max(rhs_shape.len());
        let mut broadcast_shape = Vec::with_capacity(max_rank);

        let lhs_padded = {
            let mut padded = vec![1; max_rank - lhs_shape.len()];
            padded.extend_from_slice(lhs_shape);
            padded
        };

        let rhs_padded = {
            let mut padded = vec![1; max_rank - rhs_shape.len()];
            padded.extend_from_slice(rhs_shape);
            padded
        };

        for (i, (&dim1, &dim2)) in lhs_padded.iter().zip(rhs_padded.iter()).enumerate() {
            if dim1 != 1 && dim2 != 1 && dim1 != dim2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: format!("broadcast at dimension {}", i),
                });
            }
            broadcast_shape.push(dim1.max(dim2));
        }

        let lhs_broadcast = lhs.broadcast_to(&broadcast_shape)?;
        let rhs_broadcast = rhs.broadcast_to(&broadcast_shape)?;

        Ok((lhs_broadcast, rhs_broadcast))
    }
}
