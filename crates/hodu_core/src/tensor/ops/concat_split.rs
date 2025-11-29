use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    ops::{ConcatOp, ConcatParams, Op, OpParams, SplitOp, SplitParams},
    scalar::Scalar,
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::{Layout, Shape},
    utils::valid::{
        validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op, validate_same_device,
        validate_same_dtype,
    },
};

impl Tensor {
    // concat Operations
    pub fn concat<D: Into<Scalar>>(tensors: &[&Self], dim: D) -> HoduResult<Self> {
        if tensors.is_empty() {
            return Err(HoduError::InternalError(
                "concat requires at least one tensor".to_string(),
            ));
        }
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();

        let first = tensors[0];
        let ndim = first.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(tensors, Op::Concat(ConcatOp::Concat))?;
        validate_same_dtype(tensors, Op::Concat(ConcatOp::Concat))?;
        validate_dtype_for_device(first.dtype(), first.device())?;
        validate_dtype_for_op(first.dtype(), Op::Concat(ConcatOp::Concat))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Concat(ConcatOp::Concat));

        // Collect all layouts once to minimize lock acquisitions
        let layouts: Vec<_> = tensors.iter().map(|t| t.layout()).collect();
        let layout_refs: Vec<_> = layouts.iter().collect();

        // Compute output dimensions from layouts (avoiding repeated shape queries)
        let mut output_dims = layouts[0].shape().dims().to_vec();
        for layout in &layouts[1..] {
            let dims = layout.shape().dims();
            if dims.len() != output_dims.len() {
                return Err(HoduError::incompatible_shapes(
                    Shape::from(&output_dims),
                    layout.shape().clone(),
                    Op::Concat(ConcatOp::Concat),
                ));
            }
            output_dims[dim_usize] += dims[dim_usize];
        }

        if crate::snapshot::capture::is_active() {
            let result_layout = Layout::from_shape(&Shape::from(output_dims));
            let requires_grad = tensors.iter().any(|t| t.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), first.dtype(), requires_grad);

            let input_ids: Vec<_> = tensors.iter().map(|t| t.id()).collect();
            let op_params = OpParams::Concat(ConcatParams { dim: dim_scalar });

            crate::snapshot::capture::capture_operation(
                Op::Concat(ConcatOp::Concat),
                Some(op_params.clone()),
                input_ids.clone(),
                result_id,
                layouts.clone(),
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(input_ids, result_id, Op::Concat(ConcatOp::Concat), op_params)?;
            }

            Ok(result_tensor)
        } else {
            // Clone storages to avoid lifetime issues
            let mut all_storages: Vec<_> = Vec::new();
            for tensor in tensors.iter() {
                let storage = tensor.with_storage(|s| Ok(s.clone()))?;
                all_storages.push(storage);
            }

            let first_storage = &all_storages[0];
            let other_refs: Vec<_> = all_storages[1..].iter().collect();
            let storage =
                first_storage.call_ops_concat(&other_refs, &layout_refs, dim_usize, Op::Concat(ConcatOp::Concat))?;

            let result_layout = Layout::from_shape(&Shape::from(output_dims));
            let requires_grad = tensors.iter().any(|t| t.is_requires_grad()) && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let input_ids: Vec<_> = tensors.iter().map(|t| t.id()).collect();
                gradient::record_operation(
                    input_ids,
                    result.id(),
                    Op::Concat(ConcatOp::Concat),
                    OpParams::Concat(ConcatParams { dim: dim_scalar }),
                )?;
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

        let unsqueezed: Vec<Self> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim_scalar))
            .collect::<HoduResult<_>>()?;
        let unsqueezed_refs: Vec<&Self> = unsqueezed.iter().collect();
        Self::concat(&unsqueezed_refs, dim_scalar)
    }

    // split Operations
    pub fn split<D: Into<Scalar>>(&self, sizes: &[usize], dim: D) -> HoduResult<Vec<Self>> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate dtype for device and operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Split(SplitOp::Split))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Split(SplitOp::Split));

        let shape = self.shape();
        let shape_dims = shape.dims();
        let requires_grad = self.is_requires_grad() && validate_requires_grad;

        // Prepare sizes as Scalars
        let sizes_scalars: Vec<Scalar> = sizes.iter().map(|&s| Scalar::from(s)).collect();

        if crate::snapshot::capture::is_active() {
            let mut results = Vec::new();

            // Create all output tensors first
            let output_ids: Vec<_> = sizes
                .iter()
                .map(|&size| {
                    let mut result_dims = shape_dims.to_vec();
                    result_dims[dim_usize] = size;
                    let result_layout = Layout::from_shape(&Shape::from(result_dims));
                    let (result_id, result_tensor) = create_builder_tensor(result_layout, self.dtype(), requires_grad);
                    results.push(result_tensor);
                    result_id
                })
                .collect();

            // Register the operation once with all outputs
            let output_layouts: Vec<_> = sizes
                .iter()
                .map(|&size| {
                    let mut result_dims = shape_dims.to_vec();
                    result_dims[dim_usize] = size;
                    Layout::from_shape(&Shape::from(result_dims))
                })
                .collect();

            // Register each split output separately with its own output_index
            for (output_index, (&result_id, output_layout)) in output_ids.iter().zip(output_layouts.iter()).enumerate()
            {
                let op_params = OpParams::Split(SplitParams {
                    dim: dim_scalar,
                    sizes: sizes_scalars.clone(),
                    output_index,
                });

                crate::snapshot::capture::capture_operation(
                    Op::Split(SplitOp::Split),
                    Some(op_params.clone()),
                    vec![self.id()],
                    result_id,
                    vec![self.layout()],
                    output_layout.clone(),
                )?;

                if requires_grad {
                    gradient::record_operation(vec![self.id()], result_id, Op::Split(SplitOp::Split), op_params)?;
                }
            }

            Ok(results)
        } else {
            let mut results = Vec::new();
            let mut start = 0;

            for (output_index, &size) in sizes.iter().enumerate() {
                let storage = self.with_storage(|storage| {
                    storage.call_ops_split(&self.layout(), dim_usize, start, size, Op::Split(SplitOp::Split))
                })?;

                let mut result_dims = shape_dims.to_vec();
                result_dims[dim_usize] = size;
                let result_layout = Layout::from_shape(&Shape::from(result_dims));
                let result = from_storage_with_context(storage, result_layout, true, requires_grad);

                if !gradient::is_computing_gradients() && requires_grad {
                    gradient::record_operation(
                        vec![self.id()],
                        result.id(),
                        Op::Split(SplitOp::Split),
                        OpParams::Split(SplitParams {
                            dim: dim_scalar,
                            sizes: sizes_scalars.clone(),
                            output_index,
                        }),
                    )?;
                }

                results.push(result);
                start += size;
            }

            Ok(results)
        }
    }

    pub fn chunk<D: Into<Scalar>>(&self, chunks: usize, dim: D) -> HoduResult<Vec<Self>> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };
        let shape = self.shape();
        let shape_dims = shape.dims();
        let dim_size = shape_dims[dim_usize];

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
