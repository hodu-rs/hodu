use crate::error::{HoduError, HoduResult};
use crate::types::Shape;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct ParsedEinsum {
    pub input_subscripts: Vec<Vec<char>>,
    pub output_subscripts: Vec<char>,
    pub contraction_indices: Vec<char>,
    pub all_indices: Vec<char>,
    pub index_sizes: HashMap<char, usize>,
}

impl ParsedEinsum {
    pub fn parse(equation: &str, input_shapes: &[&Shape]) -> HoduResult<Self> {
        let equation = equation.replace(' ', "");

        let (inputs_str, output_str) = if equation.contains("->") {
            let parts: Vec<&str> = equation.split("->").collect();
            if parts.len() != 2 {
                return Err(HoduError::InvalidArgument(
                    "einsum equation must have exactly one '->'".to_string(),
                ));
            }
            (parts[0], Some(parts[1]))
        } else {
            (equation.as_str(), None)
        };

        let input_subscripts: Vec<Vec<char>> = inputs_str.split(',').map(|s| s.chars().collect()).collect();

        if input_subscripts.len() != input_shapes.len() {
            return Err(HoduError::InvalidArgument(format!(
                "einsum equation has {} operands but {} tensors provided",
                input_subscripts.len(),
                input_shapes.len()
            )));
        }

        for (i, (subs, shape)) in input_subscripts.iter().zip(input_shapes.iter()).enumerate() {
            if subs.len() != shape.ndim() {
                return Err(HoduError::InvalidArgument(format!(
                    "einsum operand {} has {} subscripts but tensor has {} dimensions",
                    i,
                    subs.len(),
                    shape.ndim()
                )));
            }
        }

        let mut index_sizes: HashMap<char, usize> = HashMap::new();
        for (subs, shape) in input_subscripts.iter().zip(input_shapes.iter()) {
            for (idx, &c) in subs.iter().enumerate() {
                let dim_size = shape.dims()[idx];
                if let Some(&existing_size) = index_sizes.get(&c) {
                    if existing_size != dim_size {
                        return Err(HoduError::InvalidArgument(format!(
                            "einsum index '{}' has inconsistent sizes: {} vs {}",
                            c, existing_size, dim_size
                        )));
                    }
                } else {
                    index_sizes.insert(c, dim_size);
                }
            }
        }

        let mut all_indices: Vec<char> = index_sizes.keys().cloned().collect();
        all_indices.sort();

        let mut index_counts: HashMap<char, usize> = HashMap::new();
        for subs in &input_subscripts {
            for &c in subs {
                *index_counts.entry(c).or_insert(0) += 1;
            }
        }

        let output_subscripts: Vec<char> = if let Some(out_str) = output_str {
            out_str.chars().collect()
        } else {
            all_indices
                .iter()
                .filter(|&&c| index_counts.get(&c).copied().unwrap_or(0) == 1)
                .cloned()
                .collect()
        };

        for &c in &output_subscripts {
            if !index_sizes.contains_key(&c) {
                return Err(HoduError::InvalidArgument(format!(
                    "einsum output index '{}' not found in any input",
                    c
                )));
            }
        }

        let output_set: HashSet<char> = output_subscripts.iter().cloned().collect();
        let contraction_indices: Vec<char> = all_indices
            .iter()
            .filter(|c| !output_set.contains(c))
            .cloned()
            .collect();

        Ok(Self {
            input_subscripts,
            output_subscripts,
            contraction_indices,
            all_indices,
            index_sizes,
        })
    }

    pub fn compute_output_shape(&self) -> Shape {
        let dims: Vec<usize> = self.output_subscripts.iter().map(|c| self.index_sizes[c]).collect();
        Shape::from(&dims)
    }

    pub fn num_output_elements(&self) -> usize {
        self.output_subscripts.iter().map(|c| self.index_sizes[c]).product()
    }

    pub fn num_contraction_elements(&self) -> usize {
        if self.contraction_indices.is_empty() {
            1
        } else {
            self.contraction_indices.iter().map(|c| self.index_sizes[c]).product()
        }
    }

    pub fn get_index_to_dim_map(&self, input_idx: usize) -> Vec<i32> {
        let subs = &self.input_subscripts[input_idx];
        self.all_indices
            .iter()
            .map(|c| subs.iter().position(|s| s == c).map(|p| p as i32).unwrap_or(-1))
            .collect()
    }

    /// Returns dim_to_index mapping: for each dimension of the input,
    /// which index id does it correspond to.
    /// This properly handles repeated indices like "ii" in trace operations.
    pub fn get_dim_to_index_map(&self, input_idx: usize) -> Vec<usize> {
        let subs = &self.input_subscripts[input_idx];
        subs.iter()
            .map(|c| self.all_indices.iter().position(|a| a == c).unwrap())
            .collect()
    }

    pub fn get_output_index_ids(&self) -> Vec<usize> {
        self.output_subscripts
            .iter()
            .map(|c| self.all_indices.iter().position(|a| a == c).unwrap())
            .collect()
    }

    pub fn get_contraction_index_ids(&self) -> Vec<usize> {
        self.contraction_indices
            .iter()
            .map(|c| self.all_indices.iter().position(|a| a == c).unwrap())
            .collect()
    }

    pub fn get_index_sizes_vec(&self) -> Vec<usize> {
        self.all_indices.iter().map(|c| self.index_sizes[c]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_matmul() {
        let a_shape = Shape::from(&[2, 3]);
        let b_shape = Shape::from(&[3, 4]);
        let parsed = ParsedEinsum::parse("ij,jk->ik", &[&a_shape, &b_shape]).unwrap();

        assert_eq!(parsed.input_subscripts, vec![vec!['i', 'j'], vec!['j', 'k']]);
        assert_eq!(parsed.output_subscripts, vec!['i', 'k']);
        assert_eq!(parsed.contraction_indices, vec!['j']);
        assert_eq!(parsed.compute_output_shape().dims(), &[2, 4]);
    }

    #[test]
    fn test_parse_transpose() {
        let a_shape = Shape::from(&[2, 3]);
        let parsed = ParsedEinsum::parse("ij->ji", &[&a_shape]).unwrap();

        assert_eq!(parsed.input_subscripts, vec![vec!['i', 'j']]);
        assert_eq!(parsed.output_subscripts, vec!['j', 'i']);
        assert!(parsed.contraction_indices.is_empty());
        assert_eq!(parsed.compute_output_shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_parse_trace() {
        let a_shape = Shape::from(&[3, 3]);
        let parsed = ParsedEinsum::parse("ii->", &[&a_shape]).unwrap();

        assert_eq!(parsed.input_subscripts, vec![vec!['i', 'i']]);
        assert!(parsed.output_subscripts.is_empty());
        assert_eq!(parsed.contraction_indices, vec!['i']);
        assert_eq!(parsed.compute_output_shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_parse_outer_product() {
        let a_shape = Shape::from(&[2]);
        let b_shape = Shape::from(&[3]);
        let parsed = ParsedEinsum::parse("i,j->ij", &[&a_shape, &b_shape]).unwrap();

        assert_eq!(parsed.output_subscripts, vec!['i', 'j']);
        assert!(parsed.contraction_indices.is_empty());
        assert_eq!(parsed.compute_output_shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_parse_batch_matmul() {
        let a_shape = Shape::from(&[2, 3, 4]);
        let b_shape = Shape::from(&[2, 4, 5]);
        let parsed = ParsedEinsum::parse("bij,bjk->bik", &[&a_shape, &b_shape]).unwrap();

        assert_eq!(parsed.output_subscripts, vec!['b', 'i', 'k']);
        assert_eq!(parsed.contraction_indices, vec!['j']);
        assert_eq!(parsed.compute_output_shape().dims(), &[2, 3, 5]);
    }

    #[test]
    fn test_implicit_output() {
        let a_shape = Shape::from(&[2, 3]);
        let b_shape = Shape::from(&[3, 4]);
        let parsed = ParsedEinsum::parse("ij,jk", &[&a_shape, &b_shape]).unwrap();

        assert_eq!(parsed.output_subscripts, vec!['i', 'k']);
        assert_eq!(parsed.contraction_indices, vec!['j']);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a_shape = Shape::from(&[2, 3]);
        let b_shape = Shape::from(&[4, 5]); // j dimension mismatch: 3 vs 4
        let result = ParsedEinsum::parse("ij,jk->ik", &[&a_shape, &b_shape]);

        assert!(result.is_err());
    }
}
