use crate::{
    einsum::ParsedEinsum,
    error::{HoduError, HoduResult},
    op_params::{EinsumParams, OpParams},
    ops::{EinsumOp, Op},
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::Layout,
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_same_device, validate_same_dtype},
};

impl Tensor {
    /// Einstein summation operation.
    ///
    /// Performs tensor contraction based on the Einstein summation convention.
    ///
    /// # Arguments
    /// * `equation` - Einsum equation string (e.g., "ij,jk->ik" for matrix multiplication)
    /// * `tensors` - Slice of input tensors (maximum 4 tensors supported)
    ///
    /// # Limitations
    /// - Maximum 4 input tensors (GPU kernel buffer limit)
    /// - Maximum 16 dimensions per tensor
    /// - Maximum 16 unique indices in equation
    ///
    /// # Examples
    /// ```ignore
    /// // Matrix multiplication: C = A @ B
    /// let c = Tensor::einsum("ij,jk->ik", &[&a, &b])?;
    ///
    /// // Batch matrix multiplication
    /// let c = Tensor::einsum("bij,bjk->bik", &[&a, &b])?;
    ///
    /// // Transpose
    /// let b = Tensor::einsum("ij->ji", &[&a])?;
    ///
    /// // Trace (diagonal sum)
    /// let trace = Tensor::einsum("ii->", &[&a])?;
    ///
    /// // Outer product
    /// let outer = Tensor::einsum("i,j->ij", &[&a, &b])?;
    /// ```
    pub fn einsum(equation: &str, tensors: &[&Tensor]) -> HoduResult<Tensor> {
        if tensors.is_empty() {
            return Err(HoduError::InvalidArgument(
                "einsum requires at least one input tensor".to_string(),
            ));
        }

        if tensors.len() > 4 {
            return Err(HoduError::InvalidArgument(
                "einsum supports at most 4 input tensors".to_string(),
            ));
        }

        let op = Op::Einsum(EinsumOp::Einsum);

        // Validate all tensors
        validate_same_device(tensors, op.clone())?;
        validate_same_dtype(tensors, op.clone())?;
        validate_dtype_for_device(tensors[0].dtype(), tensors[0].device())?;
        validate_dtype_for_op(tensors[0].dtype(), op.clone())?;

        // Parse einsum equation
        let input_shapes: Vec<_> = tensors.iter().map(|t| t.shape()).collect();
        let input_shape_refs: Vec<_> = input_shapes.iter().collect();
        let parsed = ParsedEinsum::parse(equation, &input_shape_refs)?;

        // Compute output shape
        let output_shape = parsed.compute_output_shape();
        let result_layout = Layout::from_shape(&output_shape);

        // Build op params
        let op_params = OpParams::Einsum(EinsumParams {
            equation: equation.to_string(),
            input_subscripts: parsed.input_subscripts.clone(),
            output_subscripts: parsed.output_subscripts.clone(),
            contraction_indices: parsed.contraction_indices.clone(),
        });

        let requires_grad = tensors.iter().any(|t| t.is_requires_grad());
        let dtype = tensors[0].dtype();

        let input_layouts: Vec<Layout> = tensors.iter().map(|t| t.layout()).collect();
        let input_ids: Vec<_> = tensors.iter().map(|t| t.id()).collect();

        if crate::snapshot::capture::is_active() {
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), dtype, requires_grad);

            crate::snapshot::capture::capture_operation(
                op.clone(),
                Some(op_params.clone()),
                input_ids.clone(),
                result_id,
                input_layouts,
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(input_ids, result_id, op, op_params)?;
            }

            Ok(result_tensor)
        } else {
            let other_layouts: Vec<Layout> = tensors.iter().skip(1).map(|t| t.layout()).collect();
            let input_layout_refs: Vec<&Layout> = other_layouts.iter().collect();

            // Execute with nested with_storage calls based on tensor count
            let storage = match tensors.len() {
                1 => tensors[0].with_storage(|s0| s0.call_ops_einsum(&[], &[], &parsed))?,
                2 => tensors[0].with_storage(|s0| {
                    tensors[1].with_storage(|s1| s0.call_ops_einsum(&[s1], &input_layout_refs, &parsed))
                })?,
                3 => tensors[0].with_storage(|s0| {
                    tensors[1].with_storage(|s1| {
                        tensors[2].with_storage(|s2| s0.call_ops_einsum(&[s1, s2], &input_layout_refs, &parsed))
                    })
                })?,
                4 => tensors[0].with_storage(|s0| {
                    tensors[1].with_storage(|s1| {
                        tensors[2].with_storage(|s2| {
                            tensors[3].with_storage(|s3| s0.call_ops_einsum(&[s1, s2, s3], &input_layout_refs, &parsed))
                        })
                    })
                })?,
                _ => unreachable!(),
            };

            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(input_ids, result.id(), op, op_params)?;
            }

            Ok(result)
        }
    }
}
