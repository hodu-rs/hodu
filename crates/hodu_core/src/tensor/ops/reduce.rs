use crate::{
    builder,
    compat::*,
    error::HoduResult,
    op::{
        self,
        utils::{validate_dtype_for_device, validate_dtype_for_op},
        Op,
    },
    scalar::Scalar,
    tensor::{
        create_builder_tensor_with_grad, from_storage_with_grad, gradient, register_operation_in_builder, Tensor,
    },
    types::layout::Layout,
};

// Reduction Operations
impl Tensor {
    pub fn sum(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Sum, dims, keep_dim)
    }

    pub fn sum_all(&self) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Sum, &[], false)
    }

    pub fn sum_to_shape(&self, target_shape: &[usize]) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let current_shape = current_layout.get_shape();

        if current_shape == target_shape {
            return Ok(*self);
        }

        let mut result = *self;

        if current_shape.len() > target_shape.len() {
            let dims_to_sum: Vec<usize> = (0..(current_shape.len() - target_shape.len())).collect();
            for &dim in dims_to_sum.iter().rev() {
                result = result.sum(&[dim], false)?;
            }
        }

        let result_layout = result.get_layout();
        let result_shape = result_layout.get_shape();
        for (i, (&target_dim, &current_dim)) in target_shape.iter().zip(result_shape.iter()).enumerate() {
            if target_dim == 1 && current_dim > 1 {
                result = result.sum(&[i], true)?;
            }
        }

        Ok(result)
    }

    pub fn mean(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Mean, dims, keep_dim)
    }

    pub fn mean_all(&self) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Mean, &[], false)
    }

    pub fn max(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Max, dims, keep_dim)
    }

    pub fn min(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Min, dims, keep_dim)
    }

    pub fn prod(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Prod, dims, keep_dim)
    }

    pub fn std(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Std, dims, keep_dim)
    }

    pub fn std_all(&self) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Std, &[], false)
    }

    pub fn var(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Var, dims, keep_dim)
    }

    pub fn var_all(&self) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Var, &[], false)
    }

    pub fn norm(&self, p: impl Into<Scalar>, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        let p_scalar = p.into();
        match p_scalar.to_i32() {
            1 => self.l1_norm(dims, keep_dim),
            2 => self.l2_norm(dims, keep_dim),
            _ => {
                let p_dtype = p_scalar.to_dtype(self.get_dtype());
                self.abs()?
                    .pow_scalar(p_dtype)?
                    .sum(dims, keep_dim)?
                    .pow_scalar(Scalar::one(self.get_dtype()) / p_dtype)
            },
        }
    }

    pub fn l2_norm(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Norm, dims, keep_dim)
    }

    pub fn l1_norm(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.abs()?.sum(dims, keep_dim)
    }

    pub fn argmax(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::ArgMax, dims, keep_dim)
    }

    pub fn argmin(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::ArgMin, dims, keep_dim)
    }

    pub fn any(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::Any, dims, keep_dim)
    }

    pub fn all(&self, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(op::ReduceOp::All, dims, keep_dim)
    }

    fn reduce_operation(&self, reduce_op: op::ReduceOp, dims: &[usize], keep_dim: bool) -> HoduResult<Self> {
        // Validate dtype for device and operation
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), &reduce_op.to_string())?;
        let dims_scalars: Vec<Scalar> = dims.iter().map(|&d| Scalar::I32(d as i32)).collect();
        let op = Op::Reduce(reduce_op, self.id(), keep_dim, dims_scalars.clone());
        validate_dtype_for_op(self.get_dtype(), &op)?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let shape = layout.get_shape();
            let ndim = shape.len();

            // Calculate output shape
            let reduce_dims: Vec<usize> = if dims.is_empty() {
                (0..ndim).collect()
            } else {
                dims.to_vec()
            };

            let mut output_shape = shape.to_vec();
            for &dim in &reduce_dims {
                if keep_dim {
                    output_shape[dim] = 1;
                } else {
                    output_shape[dim] = 0;
                }
            }
            if !keep_dim {
                output_shape.retain(|&size| size != 0);
            }

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(op.clone(), vec![result_id], vec![layout.clone()], vec![result_layout]);

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.reduce(reduce_op, &self.get_layout(), dims, keep_dim))?;
            let layout = self.get_layout();
            let shape = layout.get_shape();
            let ndim = shape.len();

            // Calculate output shape
            let reduce_dims: Vec<usize> = if dims.is_empty() {
                (0..ndim).collect()
            } else {
                dims.to_vec()
            };

            let mut output_shape = shape.to_vec();
            for &dim in &reduce_dims {
                if keep_dim {
                    output_shape[dim] = 1;
                } else {
                    output_shape[dim] = 0;
                }
            }
            if !keep_dim {
                output_shape.retain(|&size| size != 0);
            }

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let dims_scalars: Vec<Scalar> = dims.iter().map(|&d| Scalar::I32(d as i32)).collect();
                let op = Op::Reduce(reduce_op, self.id(), keep_dim, dims_scalars);
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }
}
