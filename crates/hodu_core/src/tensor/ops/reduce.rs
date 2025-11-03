use crate::{
    error::HoduResult,
    layer::compat::*,
    ops::{Op, OpParams, ReduceOp},
    scalar::Scalar,
    script::builder,
    tensor::{create_builder_tensor_with_grad, from_storage, gradient, register_operation_in_builder, Tensor},
    types::{Layout, Shape},
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op},
};

impl Tensor {
    pub fn sum(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Sum, dims, keep_dim)
    }

    pub fn sum_all(&self) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Sum, &[], false)
    }

    pub fn sum_to_shape(&self, target_shape: &Shape) -> HoduResult<Self> {
        let current_shape = self.shape();
        let current_dims = current_shape.dims();
        let target_dims = target_shape.dims();

        if current_dims == target_dims {
            return Ok(*self);
        }

        let mut result = *self;

        // If current shape has more dimensions than target, reduce leading dimensions
        if current_dims.len() > target_dims.len() {
            let dims_to_sum: Vec<usize> = (0..(current_dims.len() - target_dims.len())).collect();
            for &dim in dims_to_sum.iter().rev() {
                result = result.sum(&[dim], false)?;
            }
        }

        let result_shape = result.shape();
        let result_dims = result_shape.dims();

        // Reduce dimensions where target is 1 but current is > 1
        for (i, (&target_dim, &current_dim)) in target_dims.iter().zip(result_dims.iter()).enumerate() {
            if target_dim == 1 && current_dim > 1 {
                result = result.sum(&[i], true)?;
            }
        }

        Ok(result)
    }

    pub fn mean(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Mean, dims, keep_dim)
    }

    pub fn mean_all(&self) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Mean, &[], false)
    }

    pub fn max(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Max, dims, keep_dim)
    }

    pub fn min(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Min, dims, keep_dim)
    }

    pub fn prod(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Prod, dims, keep_dim)
    }

    pub fn std(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Std, dims, keep_dim)
    }

    pub fn std_all(&self) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Std, &[], false)
    }

    pub fn var(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Var, dims, keep_dim)
    }

    pub fn var_all(&self) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Var, &[], false)
    }

    pub fn norm(&self, p: impl Into<Scalar>, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        let p_scalar = p.into();
        match p_scalar.to_i32() {
            1 => self.l1_norm(dims, keep_dim),
            2 => self.l2_norm(dims, keep_dim),
            _ => {
                let p_dtype = p_scalar.to_dtype(self.dtype());
                self.abs()?
                    .pow_scalar(p_dtype)?
                    .sum(dims, keep_dim)?
                    .pow_scalar(Scalar::one(self.dtype()) / p_dtype)
            },
        }
    }

    pub fn l2_norm(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Norm, dims, keep_dim)
    }

    pub fn l1_norm(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.abs()?.sum(dims, keep_dim)
    }

    pub fn argmax(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::ArgMax, dims, keep_dim)
    }

    pub fn argmin(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::ArgMin, dims, keep_dim)
    }

    pub fn any(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Any, dims, keep_dim)
    }

    pub fn all(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::All, dims, keep_dim)
    }

    fn reduce_operation(&self, reduce_op: ReduceOp, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        // Validate dtype for device and operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Reduce(reduce_op))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Reduce(reduce_op));

        let shape = self.shape();
        let shape_dims = shape.dims();
        let ndim = shape_dims.len();

        // Calculate output shape
        let reduce_dims: Vec<usize> = if dims.is_empty() {
            (0..ndim).collect()
        } else {
            dims.to_vec()
        };

        let mut output_dims = shape_dims.to_vec();
        for &dim in &reduce_dims {
            if keep_dim {
                output_dims[dim] = 1;
            } else {
                output_dims[dim] = 0;
            }
        }
        if !keep_dim {
            output_dims.retain(|&size| size != 0);
        }

        let result_layout = Layout::from_shape(&Shape::from(output_dims));

        let dims_u32: Vec<u32> = dims.iter().map(|&d| d as u32).collect();

        if builder::is_builder_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let dims_scalars: Vec<Scalar> = reduce_dims.iter().map(|&d| Scalar::from(d as u32)).collect();

            register_operation_in_builder(
                Op::Reduce(reduce_op),
                Some(OpParams {
                    dims: dims_scalars.clone(),
                    keep_dim: Some(keep_dim),
                    ..Default::default()
                }),
                vec![self.id()],
                vec![result_id],
                vec![self.layout()],
                vec![result_layout],
            )?;

            if requires_grad {
                gradient::record_operation_with_dims(
                    result_id,
                    Op::Reduce(reduce_op),
                    vec![self.id()],
                    dims_scalars,
                    Some(keep_dim),
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                storage.call_reduce(&self.layout(), &dims_u32, keep_dim, Op::Reduce(reduce_op))
            })?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Reduce(reduce_op);
                let dims_scalars: Vec<Scalar> = reduce_dims.iter().map(|&d| Scalar::from(d as u32)).collect();
                gradient::record_operation_with_dims(result.id(), op, vec![self.id()], dims_scalars, Some(keep_dim))?;
            }

            Ok(result)
        }
    }
}
