use crate::{
    error::{HoduError, HoduResult},
    op_params::{OpParams, TopKParams},
    ops::{Op, SortOp},
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::{DType, Layout, Shape},
};

impl Tensor {
    /// Returns the k largest (or smallest) elements along a dimension.
    ///
    /// # Arguments
    /// * `k` - Number of top elements to return
    /// * `dim` - Dimension along which to find top-k (supports negative indexing)
    /// * `largest` - If true, returns the largest elements; if false, returns the smallest
    /// * `sorted` - If true, the returned elements are sorted in descending order (for largest=true)
    ///
    /// # Returns
    /// A tuple of (values, indices) where:
    /// * values: Tensor containing the k largest/smallest elements
    /// * indices: Tensor containing the indices of the k elements (I32 dtype)
    ///
    /// # Example
    /// ```ignore
    /// let x = Tensor::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], [6], DType::F32)?;
    /// let (values, indices) = x.topk(3, -1, true, true)?;
    /// // values: [9.0, 5.0, 4.0]
    /// // indices: [5, 4, 2]
    /// ```
    pub fn topk(&self, k: usize, dim: i32, largest: bool, sorted: bool) -> HoduResult<(Self, Self)> {
        let shape = self.shape();
        let ndim = shape.ndim();

        if ndim == 0 {
            return Err(HoduError::InternalError("topk requires at least 1D tensor".to_string()));
        }

        // Normalize negative dim
        let dim_normalized = if dim < 0 {
            (ndim as i32 + dim) as usize
        } else {
            dim as usize
        };

        if dim_normalized >= ndim {
            return Err(HoduError::InternalError(format!(
                "Dimension out of range (expected to be in range of [-{}, {}), but got {})",
                ndim, ndim, dim
            )));
        }

        let dim_size = shape[dim_normalized];
        if k > dim_size {
            return Err(HoduError::InternalError(format!(
                "k ({}) is greater than dimension size ({})",
                k, dim_size
            )));
        }

        if k == 0 {
            // Return empty tensors
            let mut new_shape = shape.dims().to_vec();
            new_shape[dim_normalized] = 0;
            let values = Self::empty(&new_shape, self.dtype())?;
            let indices = Self::empty(&new_shape, DType::I32)?;
            return Ok((values, indices));
        }

        // Calculate output shape
        let mut output_shape = shape.dims().to_vec();
        output_shape[dim_normalized] = k;
        let values_layout = Layout::from_shape(&Shape::from(output_shape.clone()));
        let indices_layout = Layout::from_shape(&Shape::from(output_shape));

        let requires_grad = self.is_requires_grad();

        if crate::snapshot::capture::is_active() {
            // Create builder tensors for both outputs
            let (values_id, values_tensor) = create_builder_tensor(values_layout.clone(), self.dtype(), requires_grad);
            let (indices_id, indices_tensor) = create_builder_tensor(indices_layout.clone(), DType::I32, false);

            let op_params = OpParams::TopK(TopKParams {
                k,
                dim,
                largest,
                sorted,
                indices_id,
            });

            // Capture operation for values (primary output)
            crate::snapshot::capture::capture_operation(
                Op::Sort(SortOp::TopK),
                Some(op_params.clone()),
                vec![self.id()],
                values_id,
                vec![self.layout()],
                values_layout,
            )?;

            // Capture operation for indices (secondary output)
            crate::snapshot::capture::capture_operation(
                Op::Sort(SortOp::TopK),
                Some(OpParams::TopK(TopKParams {
                    k,
                    dim,
                    largest,
                    sorted,
                    indices_id,
                })),
                vec![self.id()],
                indices_id,
                vec![self.layout()],
                indices_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], values_id, Op::Sort(SortOp::TopK), op_params)?;
            }

            Ok((values_tensor, indices_tensor))
        } else {
            // Make dimension to operate on the last dimension for easier processing
            let need_transpose = dim_normalized != ndim - 1;

            let input = if need_transpose {
                // Move target dim to last
                let mut perm: Vec<usize> = (0..ndim).collect();
                perm.swap(dim_normalized, ndim - 1);
                self.permute(&perm)?
            } else {
                self.clone()
            };

            // Get contiguous tensor for CPU processing
            let input = input.contiguous()?;

            // Calculate dimensions for kernel
            let input_shape = input.shape();
            let last_dim = input_shape[ndim - 1];
            let outer_size: usize = input_shape.dims()[..ndim - 1].iter().product();

            // Kernel output shape (with k at the last dim)
            let mut kernel_output_shape = input_shape.dims().to_vec();
            kernel_output_shape[ndim - 1] = k;

            let (values_storage, indices_storage) = self
                .with_storage(|storage| storage.call_topk(&input.layout(), k, last_dim, outer_size, largest, sorted))?;

            let values = from_storage_with_context(
                values_storage,
                Layout::from_shape(&Shape::from(kernel_output_shape.clone())),
                true,
                requires_grad,
            );

            let indices = from_storage_with_context(
                indices_storage,
                Layout::from_shape(&Shape::from(kernel_output_shape)),
                true,
                false, // indices don't need gradients
            );

            // Transpose back if needed
            let (values, indices) = if need_transpose {
                let mut perm: Vec<usize> = (0..ndim).collect();
                perm.swap(dim_normalized, ndim - 1);
                (values.permute(&perm)?, indices.permute(&perm)?)
            } else {
                (values, indices)
            };

            // Record operation for gradient computation
            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    values.id(),
                    Op::Sort(SortOp::TopK),
                    OpParams::TopK(TopKParams {
                        k,
                        dim,
                        largest,
                        sorted,
                        indices_id: indices.id(),
                    }),
                )?;
            }

            Ok((values, indices))
        }
    }
}
