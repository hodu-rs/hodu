use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{ConcatOp, Op, SplitOp},
    scalar::Scalar,
    tensor::{from_storage, gradient, Tensor},
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
        let dim_u32 = if dim_i32 < 0 {
            (ndim + dim_i32) as u32
        } else {
            dim_i32 as u32
        };
        let dim_usize = dim_u32 as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(tensors, Op::Concat(ConcatOp::Concat))?;
        validate_same_dtype(tensors, Op::Concat(ConcatOp::Concat))?;
        validate_dtype_for_device(first.dtype(), first.device())?;
        validate_dtype_for_op(first.dtype(), Op::Concat(ConcatOp::Concat))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Concat(ConcatOp::Concat));

        let mut output_dims = first.shape().dims().to_vec();
        for tensor in &tensors[1..] {
            let shape = tensor.shape();
            let dims = shape.dims();
            if dims.len() != output_dims.len() {
                return Err(HoduError::IncompatibleShapes {
                    lhs: Shape::from(output_dims),
                    rhs: shape.clone(),
                    op: Op::Concat(ConcatOp::Concat),
                });
            }
            output_dims[dim_usize] += dims[dim_usize];
        }

        let layouts: Vec<_> = tensors.iter().map(|t| t.layout()).collect();
        let layout_refs: Vec<_> = layouts.iter().collect();

        // Clone storages to avoid lifetime issues
        let mut all_storages: Vec<_> = Vec::new();
        for tensor in tensors.iter() {
            let storage = tensor.with_storage(|s| Ok(s.clone()))?;
            all_storages.push(storage);
        }

        let first_storage = &all_storages[0];
        let other_refs: Vec<_> = all_storages[1..].iter().collect();
        let storage = first_storage.call_concat(&other_refs, &layout_refs, dim_u32, Op::Concat(ConcatOp::Concat))?;

        let result_layout = Layout::from_shape(&Shape::from(output_dims));
        let requires_grad = tensors.iter().any(|t| t.is_requires_grad()) && validate_requires_grad;
        let result = from_storage(storage, result_layout, true, requires_grad);

        if !gradient::is_computing_gradients() && requires_grad {
            let op = Op::Concat(ConcatOp::Concat);
            let input_ids: Vec<_> = tensors.iter().map(|t| t.id()).collect();
            gradient::record_operation_with_dims(result.id(), op, input_ids, vec![dim_scalar], None)?;
        }

        Ok(result)
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
        let dim_u32 = if dim_i32 < 0 {
            (ndim + dim_i32) as u32
        } else {
            dim_i32 as u32
        };
        let dim_usize = dim_u32 as usize;

        // Validate dtype for device and operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Split(SplitOp::Split))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Split(SplitOp::Split));

        let shape = self.shape();
        let shape_dims = shape.dims();
        let requires_grad = self.is_requires_grad() && validate_requires_grad;

        let mut results = Vec::new();
        let mut start = 0u32;

        // Prepare params for gradient: [dim, size1, size2, ...]
        let mut params = vec![dim_scalar];
        params.extend(sizes.iter().map(|&s| Scalar::from(s as u32)));

        for (output_index, &size) in sizes.iter().enumerate() {
            let storage = self.with_storage(|storage| {
                storage.call_split(&self.layout(), dim_u32, start, size as u32, Op::Split(SplitOp::Split))
            })?;

            let mut result_dims = shape_dims.to_vec();
            result_dims[dim_usize] = size as u32;
            let result_layout = Layout::from_shape(&Shape::from(result_dims));
            let result = from_storage(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Split(SplitOp::Split);
                gradient::record_operation_with_split_info(
                    result.id(),
                    op,
                    vec![self.id()],
                    params.clone(),
                    output_index,
                )?;
            }

            results.push(result);
            start += size as u32;
        }

        Ok(results)
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
        let dim_size = shape_dims[dim_usize] as usize;

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
