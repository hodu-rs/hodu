use crate::{
    builder,
    compat::*,
    error::{HoduError, HoduResult},
    op::{
        self,
        utils::{validate_dtype_for_device, validate_dtype_for_op, validate_same_device, validate_same_dtype},
        Op,
    },
    scalar::Scalar,
    tensor::{
        create_builder_tensor_with_grad, from_storage_with_grad, gradient, register_operation_in_builder, Tensor,
        TensorId,
    },
    types::layout::Layout,
};

impl Tensor {
    // Concat Operations
    pub fn concat<D: Into<Scalar>>(tensors: &[&Self], dim: D) -> HoduResult<Self> {
        if tensors.is_empty() {
            return Err(HoduError::InternalError(
                "concat requires at least one tensor".to_string(),
            ));
        }
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        let first = tensors[0];

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(tensors, "concat")?;
        validate_dtype_for_device(first.get_dtype(), &first.get_device(), "concat")?;
        let tensor_ids: Vec<TensorId> = tensors.iter().map(|t| t.id()).collect();
        let op = Op::Concat(op::ConcatOp::Concat, tensor_ids.clone(), vec![dim_scalar]);
        validate_dtype_for_op(first.get_dtype(), &op)?;
        validate_same_dtype(tensors, "concat")?;

        let mut output_shape = first.get_layout().get_shape().to_vec();
        for tensor in &tensors[1..] {
            let layout = tensor.get_layout();
            let shape = layout.get_shape();
            if shape.len() != output_shape.len() {
                return Err(HoduError::IncompatibleShapes {
                    lhs: output_shape,
                    rhs: shape.to_vec(),
                    op: "concat - all tensors must have same number of dimensions".to_string(),
                });
            }
            output_shape[dim_usize] += shape[dim_usize];
        }

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = tensors.iter().any(|t| t.is_requires_grad());
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let input_layouts: Vec<Layout> = tensors.iter().map(|t| t.get_layout().clone()).collect();
            register_operation_in_builder(op.clone(), vec![result_id], input_layouts, vec![result_layout]);

            if requires_grad {
                gradient::record_operation(result_id, op, tensor_ids)?;
            }

            Ok(result_tensor)
        } else {
            let layouts: Vec<_> = tensors.iter().map(|t| t.get_layout()).collect();
            let layout_refs: Vec<_> = layouts.iter().collect();

            // Clone storages to avoid lifetime issues
            let mut all_storages: Vec<_> = Vec::new();
            for tensor in tensors.iter() {
                let storage = tensor.with_storage(|s| Ok(s.clone()))?;
                all_storages.push(storage);
            }

            let first_storage = &all_storages[0];
            let other_refs: Vec<_> = all_storages[1..].iter().collect();
            let storage = first_storage.concat(&other_refs, &layout_refs, dim_usize)?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = tensors.iter().any(|t| t.is_requires_grad());
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let tensor_ids: Vec<TensorId> = tensors.iter().map(|t| t.id()).collect();
                let op = Op::Concat(op::ConcatOp::Concat, tensor_ids.clone(), vec![dim_scalar]);
                gradient::record_operation(result.id(), op, tensor_ids)?;
            }

            Ok(result)
        }
    }

    pub fn cat<D: Into<Scalar>>(tensors: &[&Self], dim: D) -> HoduResult<Self> {
        Self::concat(tensors, dim)
    }

    pub fn stack<D: Into<Scalar>>(tensors: &[&Self], dim: D) -> HoduResult<Self> {
        if tensors.is_empty() {
            return Err(HoduError::InternalError(
                "stack requires at least one tensor".to_string(),
            ));
        }
        let dim_scalar = dim.into();
        let dim_isize = dim_scalar.to_i64() as isize;

        let unsqueezed: Vec<Self> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim_isize))
            .collect::<HoduResult<_>>()?;
        let unsqueezed_refs: Vec<&Self> = unsqueezed.iter().collect();
        Self::concat(&unsqueezed_refs, dim_scalar)
    }

    // Split Operations
    pub fn split<D: Into<Scalar>>(&self, sizes: &[usize], dim: D) -> HoduResult<Vec<Self>> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate dtype for device and operation
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "split")?;
        let mut params = vec![dim_scalar];
        params.extend(sizes.iter().map(|&s| Scalar::U64(s as u64)));
        let op = Op::Split(op::SplitOp::Split, self.id(), params.clone(), 0);
        validate_dtype_for_op(self.get_dtype(), &op)?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let shape = layout.get_shape();

            let requires_grad = self.is_requires_grad();
            let mut result_tensors = Vec::new();
            let mut result_layouts = Vec::new();

            for &size in sizes {
                let mut result_shape = shape.to_vec();
                result_shape[dim_usize] = size;
                let result_layout = Layout::from_shape(&result_shape);
                let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);
                result_tensors.push((result_id, result_tensor));
                result_layouts.push(result_layout);
            }

            // Register separate operation for each split output with its output_index
            for (output_index, ((result_id, _), result_layout)) in
                result_tensors.iter().zip(&result_layouts).enumerate()
            {
                let op = Op::Split(op::SplitOp::Split, self.id(), params.clone(), output_index);
                register_operation_in_builder(
                    op.clone(),
                    vec![*result_id],
                    vec![layout.clone()],
                    vec![result_layout.clone()],
                );

                if requires_grad {
                    gradient::record_operation(*result_id, op, vec![self.id()])?;
                }
            }

            Ok(result_tensors.into_iter().map(|(_, t)| t).collect())
        } else {
            let storages = self.with_storage(|storage| storage.split(&self.get_layout(), dim_usize, sizes))?;
            let layout = self.get_layout();
            let shape = layout.get_shape();
            let requires_grad = self.is_requires_grad();

            let results: Vec<Self> = storages
                .into_iter()
                .zip(sizes.iter())
                .map(|(storage, &size)| {
                    let mut result_shape = shape.to_vec();
                    result_shape[dim_usize] = size;
                    let result_layout = Layout::from_shape(&result_shape);
                    from_storage_with_grad(storage, result_layout, true, requires_grad)
                })
                .collect();

            if !gradient::is_computing_gradients() && requires_grad {
                let mut params = vec![dim_scalar];
                params.extend(sizes.iter().map(|&s| Scalar::U64(s as u64)));
                // Record operation for each split result with its output_index
                for (output_index, result) in results.iter().enumerate() {
                    let op = Op::Split(op::SplitOp::Split, self.id(), params.clone(), output_index);
                    gradient::record_operation(result.id(), op, vec![self.id()])?;
                }
            }

            Ok(results)
        }
    }

    pub fn chunk<D: Into<Scalar>>(&self, chunks: usize, dim: D) -> HoduResult<Vec<Self>> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;
        let layout = self.get_layout();
        let shape = layout.get_shape();
        let dim_size = shape[dim_usize];

        let chunk_size = dim_size.div_ceil(chunks);
        let sizes: Vec<usize> = (0..chunks)
            .map(|i| {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(dim_size);
                end - start
            })
            .filter(|&s| s > 0)
            .collect();

        self.split(&sizes, dim_scalar)
    }
}
