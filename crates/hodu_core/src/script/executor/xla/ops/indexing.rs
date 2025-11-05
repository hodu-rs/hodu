use crate::{
    error::{HoduError, HoduResult},
    ops::{IndexingOp, Op},
    script::builder::ir::Attribute,
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

/// Execute indexing operations
pub fn execute(
    _builder: &XlaBuilder,
    op: &Op,
    inputs: &[XlaOp],
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<XlaOp> {
    match op {
        Op::Indexing(indexing_op) => {
            match indexing_op {
                IndexingOp::IndexSelect => {
                    if inputs.len() != 2 {
                        return Err(HoduError::InternalError("IndexSelect requires 2 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("IndexSelect requires dim attribute".to_string()))?;
                    inputs[0]
                        .take(&inputs[1], dim)
                        .map_err(|e| HoduError::InternalError(format!("XLA take failed: {:?}", e)))
                },
                IndexingOp::IndexPut => {
                    if inputs.len() != 3 {
                        return Err(HoduError::InternalError("IndexPut requires 3 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("IndexPut requires dim attribute".to_string()))?;

                    // Create update computation
                    let update_builder = hodu_xla::XlaBuilder::new("index_put_update");
                    let shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let element_type = match shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.element_type(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let _old = update_builder
                        .parameter(0, element_type, &[], "old")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let new = update_builder
                        .parameter(1, element_type, &[], "new")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let update_computation = new
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build computation: {:?}", e)))?;

                    let indices_shape = inputs[1]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let indices_rank = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let base_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let base_rank = match &base_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                    let inserted_window_dims = vec![dim];
                    let scatter_dims_to_operand_dims = vec![dim];
                    let index_vector_dim = Some(indices_rank as i64);

                    let indices_dims_vec = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let mut indices_dims_plus_1 = indices_dims_vec;
                    indices_dims_plus_1.push(1);
                    let indices_reshaped = inputs[1]
                        .reshape(&indices_dims_plus_1)
                        .map_err(|e| HoduError::InternalError(format!("Failed to reshape: {:?}", e)))?;

                    inputs[0]
                        .scatter(
                            &indices_reshaped,
                            &inputs[2],
                            update_computation,
                            &update_window_dims,
                            &inserted_window_dims,
                            &scatter_dims_to_operand_dims,
                            index_vector_dim,
                            false,
                            false,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA scatter failed: {:?}", e)))
                },
                IndexingOp::Gather => {
                    if inputs.len() != 2 {
                        return Err(HoduError::InternalError("Gather requires 2 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("Gather requires dim attribute".to_string()))?;
                    inputs[0]
                        .take(&inputs[1], dim)
                        .map_err(|e| HoduError::InternalError(format!("XLA take failed: {:?}", e)))
                },
                IndexingOp::Scatter => {
                    if inputs.len() != 3 {
                        return Err(HoduError::InternalError("Scatter requires 3 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("Scatter requires dim attribute".to_string()))?;

                    // Create update computation
                    let update_builder = hodu_xla::XlaBuilder::new("scatter_update");
                    let shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let element_type = match shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.element_type(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let _old = update_builder
                        .parameter(0, element_type, &[], "old")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let new = update_builder
                        .parameter(1, element_type, &[], "new")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let update_computation = new
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build computation: {:?}", e)))?;

                    let indices_shape = inputs[1]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let indices_rank = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let base_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let base_rank = match &base_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                    let inserted_window_dims = vec![dim];
                    let scatter_dims_to_operand_dims = vec![dim];
                    let index_vector_dim = Some(indices_rank as i64);

                    let indices_dims_vec = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let mut indices_dims_plus_1 = indices_dims_vec;
                    indices_dims_plus_1.push(1);
                    let indices_reshaped = inputs[1]
                        .reshape(&indices_dims_plus_1)
                        .map_err(|e| HoduError::InternalError(format!("Failed to reshape: {:?}", e)))?;

                    inputs[0]
                        .scatter(
                            &indices_reshaped,
                            &inputs[2],
                            update_computation,
                            &update_window_dims,
                            &inserted_window_dims,
                            &scatter_dims_to_operand_dims,
                            index_vector_dim,
                            false,
                            false,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA scatter failed: {:?}", e)))
                },
                IndexingOp::ScatterAdd => {
                    if inputs.len() != 3 {
                        return Err(HoduError::InternalError("ScatterAdd requires 3 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("ScatterAdd requires dim attribute".to_string()))?;

                    // Create add computation
                    let add_builder = hodu_xla::XlaBuilder::new("scatter_add");
                    let shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let element_type = match shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.element_type(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let old = add_builder
                        .parameter(0, element_type, &[], "old")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let new = add_builder
                        .parameter(1, element_type, &[], "new")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let sum = old
                        .add_(&new)
                        .map_err(|e| HoduError::InternalError(format!("Failed to add: {:?}", e)))?;
                    let add_computation = sum
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build computation: {:?}", e)))?;

                    let indices_shape = inputs[1]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let indices_rank = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let base_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let base_rank = match &base_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                    let inserted_window_dims = vec![dim];
                    let scatter_dims_to_operand_dims = vec![dim];
                    let index_vector_dim = Some(indices_rank as i64);

                    let indices_dims_vec = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let mut indices_dims_plus_1 = indices_dims_vec;
                    indices_dims_plus_1.push(1);
                    let indices_reshaped = inputs[1]
                        .reshape(&indices_dims_plus_1)
                        .map_err(|e| HoduError::InternalError(format!("Failed to reshape: {:?}", e)))?;

                    inputs[0]
                        .scatter(
                            &indices_reshaped,
                            &inputs[2],
                            add_computation,
                            &update_window_dims,
                            &inserted_window_dims,
                            &scatter_dims_to_operand_dims,
                            index_vector_dim,
                            false,
                            false,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA scatter failed: {:?}", e)))
                },
                IndexingOp::ScatterMax => {
                    if inputs.len() != 3 {
                        return Err(HoduError::InternalError("ScatterMax requires 3 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("ScatterMax requires dim attribute".to_string()))?;

                    // Create max computation
                    let max_builder = hodu_xla::XlaBuilder::new("scatter_max");
                    let shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let element_type = match shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.element_type(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let old = max_builder
                        .parameter(0, element_type, &[], "old")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let new = max_builder
                        .parameter(1, element_type, &[], "new")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let maximum = old
                        .max(&new)
                        .map_err(|e| HoduError::InternalError(format!("Failed to max: {:?}", e)))?;
                    let max_computation = maximum
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build computation: {:?}", e)))?;

                    let indices_shape = inputs[1]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let indices_rank = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let base_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let base_rank = match &base_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                    let inserted_window_dims = vec![dim];
                    let scatter_dims_to_operand_dims = vec![dim];
                    let index_vector_dim = Some(indices_rank as i64);

                    let indices_dims_vec = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let mut indices_dims_plus_1 = indices_dims_vec;
                    indices_dims_plus_1.push(1);
                    let indices_reshaped = inputs[1]
                        .reshape(&indices_dims_plus_1)
                        .map_err(|e| HoduError::InternalError(format!("Failed to reshape: {:?}", e)))?;

                    inputs[0]
                        .scatter(
                            &indices_reshaped,
                            &inputs[2],
                            max_computation,
                            &update_window_dims,
                            &inserted_window_dims,
                            &scatter_dims_to_operand_dims,
                            index_vector_dim,
                            false,
                            false,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA scatter failed: {:?}", e)))
                },
                IndexingOp::ScatterMin => {
                    if inputs.len() != 3 {
                        return Err(HoduError::InternalError("ScatterMin requires 3 inputs".to_string()));
                    }
                    let dim = attributes
                        .get("dim")
                        .and_then(|attr| {
                            if let Attribute::Int(i) = attr {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .ok_or_else(|| HoduError::InternalError("ScatterMin requires dim attribute".to_string()))?;

                    // Create min computation
                    let min_builder = hodu_xla::XlaBuilder::new("scatter_min");
                    let shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let element_type = match shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.element_type(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let old = min_builder
                        .parameter(0, element_type, &[], "old")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let new = min_builder
                        .parameter(1, element_type, &[], "new")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let minimum = old
                        .min(&new)
                        .map_err(|e| HoduError::InternalError(format!("Failed to min: {:?}", e)))?;
                    let min_computation = minimum
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build computation: {:?}", e)))?;

                    let indices_shape = inputs[1]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let indices_rank = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let base_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
                    let base_rank = match &base_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().len(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };

                    let update_window_dims: Vec<i64> = (0..base_rank as i64).filter(|x| *x != dim).collect();
                    let inserted_window_dims = vec![dim];
                    let scatter_dims_to_operand_dims = vec![dim];
                    let index_vector_dim = Some(indices_rank as i64);

                    let indices_dims_vec = match &indices_shape {
                        hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                        _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
                    };
                    let mut indices_dims_plus_1 = indices_dims_vec;
                    indices_dims_plus_1.push(1);
                    let indices_reshaped = inputs[1]
                        .reshape(&indices_dims_plus_1)
                        .map_err(|e| HoduError::InternalError(format!("Failed to reshape: {:?}", e)))?;

                    inputs[0]
                        .scatter(
                            &indices_reshaped,
                            &inputs[2],
                            min_computation,
                            &update_window_dims,
                            &inserted_window_dims,
                            &scatter_dims_to_operand_dims,
                            index_vector_dim,
                            false,
                            false,
                        )
                        .map_err(|e| HoduError::InternalError(format!("XLA scatter failed: {:?}", e)))
                },
            }
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported indexing operation: {:?}",
            op
        ))),
    }
}
