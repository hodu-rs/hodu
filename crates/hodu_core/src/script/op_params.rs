#![allow(dead_code)]

use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    scalar::Scalar,
    script::builder::ir::Attribute,
    types::DType,
};

/// Helper for extracting parameters from attributes HashMap
pub struct OpParams<'a> {
    attributes: &'a HashMap<String, Attribute>,
}

impl<'a> OpParams<'a> {
    /// Create a new OpParams from attributes
    pub fn new(attributes: &'a HashMap<String, Attribute>) -> Self {
        Self { attributes }
    }

    /// Get a boolean value by name
    pub fn get_bool(&self, name: &str) -> HoduResult<bool> {
        self.attributes
            .get(name)
            .and_then(|a| if let Attribute::Bool(b) = a { Some(*b) } else { None })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get an optional boolean value by name
    pub fn get_bool_opt(&self, name: &str) -> Option<bool> {
        self.attributes
            .get(name)
            .and_then(|a| if let Attribute::Bool(b) = a { Some(*b) } else { None })
    }

    /// Get an integer value by name
    pub fn get_int(&self, name: &str) -> HoduResult<i32> {
        self.attributes
            .get(name)
            .and_then(|a| if let Attribute::Int(i) = a { Some(*i) } else { None })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get a usize value by name (handles Int, Usize, and Scalar)
    pub fn get_usize(&self, name: &str) -> HoduResult<usize> {
        self.attributes
            .get(name)
            .and_then(|a| match a {
                Attribute::Int(i) => Some(*i as usize),
                Attribute::Usize(u) => Some(*u),
                Attribute::Scalar(s) => Some(s.to_usize()),
                _ => None,
            })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get a float value by name
    pub fn get_float(&self, name: &str) -> HoduResult<f32> {
        self.attributes
            .get(name)
            .and_then(|a| if let Attribute::Float(f) = a { Some(*f) } else { None })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get a string value by name
    pub fn get_string(&self, name: &str) -> HoduResult<&str> {
        self.attributes
            .get(name)
            .and_then(|a| {
                if let Attribute::String(s) = a {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get a scalar value by name
    pub fn get_scalar(&self, name: &str) -> HoduResult<Scalar> {
        self.attributes
            .get(name)
            .and_then(|a| if let Attribute::Scalar(s) = a { Some(*s) } else { None })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get a scalars array by name
    pub fn get_scalars(&self, name: &str) -> HoduResult<&[Scalar]> {
        self.attributes
            .get(name)
            .and_then(|a| {
                if let Attribute::Scalars(s) = a {
                    Some(s.as_slice())
                } else {
                    None
                }
            })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get an optional scalars array by name
    pub fn get_scalars_opt(&self, name: &str) -> Option<&[Scalar]> {
        self.attributes.get(name).and_then(|a| {
            if let Attribute::Scalars(s) = a {
                Some(s.as_slice())
            } else {
                None
            }
        })
    }

    /// Get a usize array by name (converts Scalars to Vec<usize>)
    pub fn get_usize_array(&self, name: &str) -> HoduResult<Vec<usize>> {
        self.attributes
            .get(name)
            .and_then(|a| {
                if let Attribute::Scalars(s) = a {
                    Some(s.iter().map(|sc| sc.to_usize()).collect())
                } else {
                    None
                }
            })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get an integer array by name
    pub fn get_int_array(&self, name: &str) -> HoduResult<&[i32]> {
        self.attributes
            .get(name)
            .and_then(|a| {
                if let Attribute::IntArray(arr) = a {
                    Some(arr.as_slice())
                } else {
                    None
                }
            })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get an optional integer value by name
    pub fn get_int_opt(&self, name: &str) -> Option<i32> {
        self.attributes
            .get(name)
            .and_then(|a| if let Attribute::Int(i) = a { Some(*i) } else { None })
    }

    /// Get an integer value as i64 by name
    pub fn get_int_as_i64(&self, name: &str) -> HoduResult<i64> {
        self.attributes
            .get(name)
            .and_then(|a| {
                if let Attribute::Int(i) = a {
                    Some(*i as i64)
                } else {
                    None
                }
            })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get an optional integer value as i64 by name
    pub fn get_int_as_i64_opt(&self, name: &str) -> Option<i64> {
        self.attributes.get(name).and_then(|a| {
            if let Attribute::Int(i) = a {
                Some(*i as i64)
            } else {
                None
            }
        })
    }

    /// Get an integer array as Vec<i64> by name
    pub fn get_int_array_as_i64(&self, name: &str) -> HoduResult<Vec<i64>> {
        self.attributes
            .get(name)
            .and_then(|a| {
                if let Attribute::IntArray(arr) = a {
                    Some(arr.iter().map(|&d| d as i64).collect())
                } else {
                    None
                }
            })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get an integer array as Vec<usize> by name
    pub fn get_int_array_as_usize(&self, name: &str) -> HoduResult<Vec<usize>> {
        self.attributes
            .get(name)
            .and_then(|a| {
                if let Attribute::IntArray(arr) = a {
                    Some(arr.iter().map(|&d| d as usize).collect())
                } else {
                    None
                }
            })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get an optional integer array as Vec<i64> by name
    pub fn get_int_array_as_i64_opt(&self, name: &str) -> Option<Vec<i64>> {
        self.attributes.get(name).and_then(|a| {
            if let Attribute::IntArray(arr) = a {
                Some(arr.iter().map(|&d| d as i64).collect())
            } else {
                None
            }
        })
    }

    /// Get a float array by name
    pub fn get_float_array(&self, name: &str) -> HoduResult<&[f32]> {
        self.attributes
            .get(name)
            .and_then(|a| {
                if let Attribute::FloatArray(arr) = a {
                    Some(arr.as_slice())
                } else {
                    None
                }
            })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get a dtype by name
    pub fn get_dtype(&self, name: &str) -> HoduResult<DType> {
        self.attributes
            .get(name)
            .and_then(|a| if let Attribute::DType(dt) = a { Some(*dt) } else { None })
            .ok_or_else(|| HoduError::MissingAttribute(name.to_string()))
    }

    /// Get an optional dtype by name
    pub fn get_dtype_opt(&self, name: &str) -> Option<DType> {
        self.attributes
            .get(name)
            .and_then(|a| if let Attribute::DType(dt) = a { Some(*dt) } else { None })
    }

    /// Check if an attribute exists
    pub fn has(&self, name: &str) -> bool {
        self.attributes.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_params() {
        let mut attrs = HashMap::new();
        attrs.insert("keep_dim".to_string(), Attribute::Bool(true));
        attrs.insert("stride".to_string(), Attribute::Int(2));
        attrs.insert("padding".to_string(), Attribute::Usize(1));

        let params = OpParams::new(&attrs);

        assert_eq!(params.get_int("stride").unwrap(), 2);
        assert_eq!(params.get_usize("padding").unwrap(), 1);
        assert!(params.has("keep_dim"));
        assert!(!params.has("missing"));
    }
}
