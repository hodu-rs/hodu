use crate::{
    builder,
    compat::*,
    error::{HoduError, HoduResult},
    op::{self, Op},
    scalar::Scalar,
    tensor::{
        create_builder_tensor_with_grad, from_shared_storage_with_grad, gradient, register_operation_in_builder, Tensor,
    },
    types::layout::Layout,
};

// Shape Operations
impl Tensor {
    pub fn reshape<S: AsRef<[usize]>>(&self, shape: S) -> HoduResult<Self> {
        let new_shape = shape.as_ref();
        let current_layout = self.get_layout();
        let current_size = current_layout.get_size();
        let new_size = new_shape.iter().product::<usize>();

        // Check that total size remains the same
        if current_size != new_size {
            return Err(HoduError::IncompatibleShapes {
                lhs: current_layout.get_shape().to_vec(),
                rhs: new_shape.to_vec(),
                op: "reshape - total size must remain the same".to_string(),
            });
        }

        // First check if tensor is contiguous
        if !current_layout.is_contiguous() {
            // If not contiguous, make it contiguous first then reshape
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.reshape(new_shape);
        }

        let new_layout = Layout::from_shape(new_shape);
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);

            let op = Op::Shape(op::ShapeOp::Reshape, self.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![current_layout.clone()],
                vec![new_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Reshape, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn view<S: AsRef<[usize]>>(&self, shape: S) -> HoduResult<Self> {
        // view is an alias for reshape
        self.reshape(shape)
    }

    pub fn flatten(&self) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let total_size = current_layout.get_size();
        let new_shape = vec![total_size];
        let new_layout = Layout::from_shape(&new_shape);
        let requires_grad = self.is_requires_grad();

        // First check if tensor is contiguous
        if !current_layout.is_contiguous() {
            // If not contiguous, make it contiguous first then flatten
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.flatten();
        }

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);

            let op = Op::Shape(op::ShapeOp::Flatten, self.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![current_layout.clone()],
                vec![new_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Flatten, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn squeeze<D: Into<Scalar> + Clone>(&self, dim: Option<D>) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let current_shape = current_layout.get_shape();
        let ndim = current_shape.len();

        let new_shape = if let Some(ref dim) = dim {
            // Squeeze specific dimension
            let dim_scalar = dim.clone().into();
            let dim_i32 = dim_scalar.to_i64() as i32;
            let actual_dim = if dim_i32 < 0 {
                (ndim as i32 + dim_i32) as usize
            } else {
                dim_i32 as usize
            };

            if actual_dim >= ndim {
                return Err(HoduError::IncompatibleShapes {
                    lhs: current_shape.to_vec(),
                    rhs: vec![],
                    op: format!(
                        "squeeze - dimension {} out of range for {}-dimensional tensor",
                        dim_i32, ndim
                    ),
                });
            }

            if current_shape[actual_dim] != 1 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: current_shape.to_vec(),
                    rhs: vec![],
                    op: format!(
                        "squeeze - cannot squeeze dimension {} with size {}",
                        dim_i32, current_shape[actual_dim]
                    ),
                });
            }

            let mut new_shape = current_shape.to_vec();
            new_shape.remove(actual_dim);
            new_shape
        } else {
            // Squeeze all dimensions of size 1
            current_shape.iter().filter(|&&size| size != 1).copied().collect()
        };

        // First check if tensor is contiguous
        if !current_layout.is_contiguous() {
            // If not contiguous, make it contiguous first then squeeze
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.squeeze(dim.clone());
        }

        let new_layout = Layout::from_shape(&new_shape);
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);

            let op = Op::Shape(op::ShapeOp::Squeeze, self.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![current_layout.clone()],
                vec![new_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Squeeze, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn unsqueeze<D: Into<Scalar>>(&self, dim: D) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let current_shape = current_layout.get_shape();
        let ndim = current_shape.len();

        // Convert negative dimension to positive
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i64() as i32;
        let actual_dim = if dim_i32 < 0 {
            (ndim as i32 + dim_i32 + 1) as usize
        } else {
            dim_i32 as usize
        };

        // Check bounds (can insert at position 0 to ndim inclusive)
        if actual_dim > ndim {
            return Err(HoduError::IncompatibleShapes {
                lhs: current_shape.to_vec(),
                rhs: vec![],
                op: format!(
                    "unsqueeze - dimension {} out of range for {}-dimensional tensor",
                    dim_i32, ndim
                ),
            });
        }

        // Create new shape with dimension of size 1 inserted
        let mut new_shape = current_shape.to_vec();
        new_shape.insert(actual_dim, 1);

        // First check if tensor is contiguous
        if !current_layout.is_contiguous() {
            // If not contiguous, make it contiguous first then unsqueeze
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.unsqueeze(dim_scalar);
        }

        let new_layout = Layout::from_shape(&new_shape);
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);

            let op = Op::Shape(op::ShapeOp::Unsqueeze, self.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![current_layout.clone()],
                vec![new_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Unsqueeze, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn broadcast(&self, shape: &[usize]) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let target_layout = current_layout.broadcast_to(shape)?;

        // First check if tensor is contiguous
        if !current_layout.is_contiguous() {
            // If not contiguous, make it contiguous first then broadcast
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.broadcast(shape);
        }

        if builder::is_builder_active() {
            let requires_grad = self.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(target_layout.clone(), requires_grad);

            let op = Op::Shape(op::ShapeOp::Broadcast, self.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![current_layout.clone()],
                vec![target_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, target_layout, self.is_requires_grad());

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Broadcast, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn broadcast_like(&self, other: &Self) -> HoduResult<Self> {
        let other_layout = other.get_layout();
        let other_shape = other_layout.get_shape();
        self.broadcast(other_shape)
    }

    pub fn broadcast_left(&self, left_shape: &[usize]) -> HoduResult<Self> {
        let current_layout = self.get_layout();
        let current_shape = current_layout.get_shape();

        let mut target_shape = left_shape.to_vec();
        target_shape.extend_from_slice(current_shape);

        self.broadcast(&target_shape)
    }

    pub fn transpose<D1: Into<Scalar>, D2: Into<Scalar>>(&self, dim1: D1, dim2: D2) -> HoduResult<Self> {
        let layout = self.get_layout();

        // Convert scalars to i32 for layout.transpose
        let dim1_scalar = dim1.into();
        let dim2_scalar = dim2.into();
        let dim1_i32 = dim1_scalar.to_i64() as i32;
        let dim2_i32 = dim2_scalar.to_i64() as i32;

        let new_layout = layout.transpose(dim1_i32, dim2_i32)?;
        let requires_grad = self.is_requires_grad();

        // First check if tensor is contiguous
        if !layout.is_contiguous() {
            // If not contiguous, make it contiguous first then transpose
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.transpose(dim1_scalar, dim2_scalar);
        }

        if builder::is_builder_active() {
            let (tensor_id, result) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);
            let op = Op::Shape(op::ShapeOp::Transpose, self.id());
            register_operation_in_builder(op.clone(), vec![tensor_id], vec![self.get_layout()], vec![new_layout]);

            if self.is_requires_grad() {
                gradient::record_operation(tensor_id, op, vec![self.id()])?;
            }

            Ok(result)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Transpose, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn t(&self) -> HoduResult<Self> {
        self.transpose(-2, -1)
    }

    pub fn permute<A: Into<Scalar> + Copy>(&self, axes: &[A]) -> HoduResult<Self> {
        let layout = self.get_layout();
        let shape = layout.get_shape();
        let ndim = shape.len();

        // Validate axes length
        if axes.len() != ndim {
            return Err(HoduError::IncompatibleShapes {
                lhs: shape.to_vec(),
                rhs: vec![],
                op: format!(
                    "permute - axes length {} must match tensor dimensions {}",
                    axes.len(),
                    ndim
                ),
            });
        }

        // Convert Scalar axes to usize, handling negative indices
        let mut axes_usize = Vec::with_capacity(ndim);
        for &axis in axes {
            let axis_scalar = axis.into();
            let axis_i32 = axis_scalar.to_i64() as i32;
            let actual_axis = if axis_i32 < 0 {
                (ndim as i32 + axis_i32) as usize
            } else {
                axis_i32 as usize
            };

            if actual_axis >= ndim {
                return Err(HoduError::IncompatibleShapes {
                    lhs: shape.to_vec(),
                    rhs: vec![],
                    op: format!(
                        "permute - axis {} out of range for {}-dimensional tensor",
                        axis_i32, ndim
                    ),
                });
            }

            axes_usize.push(actual_axis);
        }

        // Check that axes contains each dimension exactly once
        let mut seen = vec![false; ndim];
        for &axis in &axes_usize {
            if seen[axis] {
                return Err(HoduError::IncompatibleShapes {
                    lhs: shape.to_vec(),
                    rhs: axes_usize.clone(),
                    op: format!("permute - duplicate axis {} in permutation", axis),
                });
            }
            seen[axis] = true;
        }

        let new_layout = layout.permute(&axes_usize)?;
        let requires_grad = self.is_requires_grad();

        // First check if tensor is contiguous
        if !layout.is_contiguous() {
            // If not contiguous, make it contiguous first then permute
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.permute(axes);
        }

        if builder::is_builder_active() {
            let (tensor_id, result) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);
            let op = Op::Shape(op::ShapeOp::Permute, self.id());
            register_operation_in_builder(op.clone(), vec![tensor_id], vec![self.get_layout()], vec![new_layout]);

            if self.is_requires_grad() {
                gradient::record_operation(tensor_id, op, vec![self.id()])?;
            }

            Ok(result)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Shape(op::ShapeOp::Permute, self.id());
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn slice<S: Into<Scalar> + Copy>(&self, dim: usize, start: S, end: Option<S>, step: S) -> HoduResult<Self> {
        let layout = self.get_layout();

        // Convert Scalar to isize
        let start_scalar = start.into();
        let start_isize = start_scalar.to_i64() as isize;

        let end_isize = end.map(|e| {
            let end_scalar = e.into();
            end_scalar.to_i64() as isize
        });

        let step_scalar = step.into();
        let step_isize = step_scalar.to_i64() as isize;

        let new_layout = layout.slice(dim, start_isize, end_isize, step_isize)?;
        let requires_grad = self.is_requires_grad();

        // First check if tensor is contiguous
        if !layout.is_contiguous() {
            // If not contiguous, make it contiguous first then slice
            let contiguous_tensor = self.contiguous()?;
            return contiguous_tensor.slice(dim, start, end, step);
        }

        // Store slice parameters in scalars: [dim, start, end_or_max, step]
        // Use i32::MAX to represent None for end
        let end_value = end_isize.unwrap_or(i32::MAX as isize);
        let scalars = vec![
            Scalar::I32(dim as i32),
            Scalar::I32(start_isize as i32),
            Scalar::I32(end_value as i32),
            Scalar::I32(step_isize as i32),
        ];

        if builder::is_builder_active() {
            let (tensor_id, result) = create_builder_tensor_with_grad(new_layout.clone(), requires_grad);
            let op = Op::ShapeScalars(op::ShapeScalarsOp::Slice, self.id(), scalars.clone());
            register_operation_in_builder(op.clone(), vec![tensor_id], vec![self.get_layout()], vec![new_layout]);

            if self.is_requires_grad() {
                gradient::record_operation(tensor_id, op, vec![self.id()])?;
            }

            Ok(result)
        } else {
            // Tensor is contiguous, we can share storage
            let result = from_shared_storage_with_grad(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::ShapeScalars(op::ShapeScalarsOp::Slice, self.id(), scalars);
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }
}
