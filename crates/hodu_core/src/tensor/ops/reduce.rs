use crate::{
    capture,
    compat::*,
    error::HoduResult,
    ops::{Op, OpParams, ReduceOp, ReduceParams},
    scalar::Scalar,
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::{Layout, Shape},
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op},
};

impl Tensor {
    pub fn sum<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Sum, dims, keep_dim)
    }

    pub fn sum_all(&self) -> HoduResult<Self> {
        self.reduce_operation::<i32>(ReduceOp::Sum, &[], false)
    }

    pub fn sum_to_shape(&self, target_shape: impl Into<Shape>) -> HoduResult<Self> {
        let target_shape = target_shape.into();
        let current_shape = self.shape();
        let current_dims = current_shape.dims();
        let target_dims = target_shape.dims();

        if current_dims == target_dims {
            return Ok(self.clone());
        }

        let mut result = self.clone();

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

    pub fn mean<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Mean, dims, keep_dim)
    }

    pub fn mean_all(&self) -> HoduResult<Self> {
        self.reduce_operation::<i32>(ReduceOp::Mean, &[], false)
    }

    pub fn max<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Max, dims, keep_dim)
    }

    pub fn min<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Min, dims, keep_dim)
    }

    pub fn prod<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Prod, dims, keep_dim)
    }

    pub fn std<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Std, dims, keep_dim)
    }

    pub fn std_all(&self) -> HoduResult<Self> {
        self.reduce_operation::<i32>(ReduceOp::Std, &[], false)
    }

    pub fn var<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Var, dims, keep_dim)
    }

    pub fn var_all(&self) -> HoduResult<Self> {
        self.reduce_operation::<i32>(ReduceOp::Var, &[], false)
    }

    pub fn norm<D: Into<Scalar> + Copy>(&self, p: impl Into<Scalar>, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
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

    pub fn l2_norm<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Norm, dims, keep_dim)
    }

    pub fn l1_norm<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.abs()?.sum(dims, keep_dim)
    }

    pub fn argmax<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::ArgMax, dims, keep_dim)
    }

    pub fn argmin<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::ArgMin, dims, keep_dim)
    }

    pub fn any<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::Any, dims, keep_dim)
    }

    pub fn all<D: Into<Scalar> + Copy>(&self, dims: &[D], keep_dim: bool) -> HoduResult<Self> {
        self.reduce_operation(ReduceOp::All, dims, keep_dim)
    }

    fn reduce_operation<D: Into<Scalar> + Copy>(
        &self,
        reduce_op: ReduceOp,
        dims: &[D],
        keep_dim: bool,
    ) -> HoduResult<Self> {
        // Validate dtype for device and operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Reduce(reduce_op))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Reduce(reduce_op));

        let shape = self.shape();
        let shape_dims = shape.dims();
        let ndim = shape_dims.len();

        // Convert dims to i32 and handle negative indices
        let dims_i32: Vec<i32> = dims.iter().map(|&d| d.into().to_i32()).collect();

        // Calculate output shape
        let reduce_dims: Vec<usize> = if dims.is_empty() {
            (0..ndim).collect()
        } else {
            dims_i32
                .iter()
                .map(|&d| if d < 0 { (ndim as i32 + d) as usize } else { d as usize })
                .collect()
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

        if capture::is_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), requires_grad);

            let dims_scalars: Vec<Scalar> = reduce_dims.iter().map(|&d| Scalar::from(d)).collect();

            let op_params = OpParams::Reduce(ReduceParams {
                dims: dims_scalars.clone(),
                keep_dim,
            });

            capture::capture_operation(
                Op::Reduce(reduce_op),
                Some(op_params.clone()),
                vec![self.id()],
                result_id,
                vec![self.layout()],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], result_id, Op::Reduce(reduce_op), op_params)?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                storage.call_ops_reduce(&self.layout(), &reduce_dims, keep_dim, Op::Reduce(reduce_op))
            })?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let dims_scalars: Vec<Scalar> = reduce_dims.iter().map(|&d| Scalar::from(d)).collect();
                gradient::record_operation(
                    vec![self.id()],
                    result.id(),
                    Op::Reduce(reduce_op),
                    OpParams::Reduce(ReduceParams {
                        dims: dims_scalars,
                        keep_dim,
                    }),
                )?;
            }

            Ok(result)
        }
    }
}
