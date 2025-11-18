use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{Op, OpParams, ShapeOp, ShapeScalarsOp},
    scalar::Scalar,
    script::builder,
    tensor::{create_builder_tensor, from_shared_storage_with, gradient, register_operation_in_builder, Tensor},
    types::Shape,
};

impl Tensor {
    pub fn reshape(&self, shape: impl Into<Shape>) -> HoduResult<Self> {
        let shape = shape.into();
        let current_size = self.size();
        let new_size = shape.size();

        if current_size != new_size {
            return Err(HoduError::IncompatibleShapes {
                lhs: self.shape(),
                rhs: shape,
                op: Op::Shape(ShapeOp::Reshape),
            });
        }

        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.reshape(shape);
        }

        let current_layout = self.layout();
        let new_layout = current_layout.reshape(&shape)?;
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Shape(ShapeOp::Reshape),
                None,
                vec![self.id()],
                vec![result_id],
                vec![current_layout],
                vec![new_layout],
            )?;

            if requires_grad {
                gradient::record_operation(result_id, Op::Shape(ShapeOp::Reshape), vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Shape(ShapeOp::Reshape);
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn view(&self, shape: impl Into<Shape>) -> HoduResult<Self> {
        self.reshape(shape)
    }

    pub fn flatten(&self) -> HoduResult<Self> {
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.flatten();
        }

        let current_layout = self.layout();
        let new_layout = current_layout.flatten()?;
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Shape(ShapeOp::Flatten),
                None,
                vec![self.id()],
                vec![result_id],
                vec![current_layout],
                vec![new_layout],
            )?;

            if requires_grad {
                gradient::record_operation(result_id, Op::Shape(ShapeOp::Flatten), vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Shape(ShapeOp::Flatten);
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn squeeze<D: Into<Scalar> + Copy>(&self, dims: &[D]) -> HoduResult<Self> {
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.squeeze(dims);
        }

        let dims_i32: Vec<i32> = dims
            .iter()
            .map(|&d| {
                let scalar = d.into();
                scalar.to_i32()
            })
            .collect();

        let current_layout = self.layout();
        let new_layout = current_layout.squeeze(&dims_i32)?;
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Shape(ShapeOp::Squeeze),
                None,
                vec![self.id()],
                vec![result_id],
                vec![current_layout],
                vec![new_layout],
            )?;

            if requires_grad {
                gradient::record_operation(result_id, Op::Shape(ShapeOp::Squeeze), vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Shape(ShapeOp::Squeeze);
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn unsqueeze<D: Into<Scalar>>(&self, dim: D) -> HoduResult<Self> {
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.unsqueeze(dim);
        }

        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();

        let current_layout = self.layout();
        let new_layout = current_layout.unsqueeze(dim_i32)?;
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Shape(ShapeOp::Unsqueeze),
                None,
                vec![self.id()],
                vec![result_id],
                vec![current_layout],
                vec![new_layout],
            )?;

            if requires_grad {
                gradient::record_operation(result_id, Op::Shape(ShapeOp::Unsqueeze), vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Shape(ShapeOp::Unsqueeze);
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn broadcast(&self, target_shape: impl Into<Shape>) -> HoduResult<Self> {
        let target_shape = target_shape.into();
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.broadcast(&target_shape);
        }

        let current_layout = self.layout();
        let new_layout = current_layout.broadcast_to(&target_shape)?;
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Shape(ShapeOp::Broadcast),
                None,
                vec![self.id()],
                vec![result_id],
                vec![current_layout],
                vec![new_layout],
            )?;

            if requires_grad {
                gradient::record_operation(result_id, Op::Shape(ShapeOp::Broadcast), vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Shape(ShapeOp::Broadcast);
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn broadcast_like(&self, other: &Self) -> HoduResult<Self> {
        self.broadcast(other.shape())
    }

    pub fn broadcast_left(&self, added_dims: &[usize]) -> HoduResult<Self> {
        let current_shape = self.shape();
        let current_dims = current_shape.dims();

        let mut new_dims = added_dims.to_vec();
        new_dims.extend_from_slice(current_dims);

        self.broadcast(Shape::from(new_dims))
    }

    pub fn transpose<D1: Into<Scalar>, D2: Into<Scalar>>(&self, dim1: D1, dim2: D2) -> HoduResult<Self> {
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.transpose(dim1, dim2);
        }

        let dim1_scalar = dim1.into();
        let dim2_scalar = dim2.into();
        let dim1_i32 = dim1_scalar.to_i32();
        let dim2_i32 = dim2_scalar.to_i32();

        let current_layout = self.layout();
        let new_layout = current_layout.transpose(dim1_i32, dim2_i32)?;
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Shape(ShapeOp::Transpose),
                None,
                vec![self.id()],
                vec![result_id],
                vec![current_layout],
                vec![new_layout],
            )?;

            if requires_grad {
                gradient::record_operation(result_id, Op::Shape(ShapeOp::Transpose), vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Shape(ShapeOp::Transpose);
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn t(&self) -> HoduResult<Self> {
        self.transpose(-2, -1)
    }

    pub fn permute<A: Into<Scalar> + Copy>(&self, axes: &[A]) -> HoduResult<Self> {
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.permute(axes);
        }

        let axes_i32: Vec<i32> = axes
            .iter()
            .map(|&axis| {
                let axis_scalar = axis.into();
                axis_scalar.to_i32()
            })
            .collect();

        let current_layout = self.layout();
        let new_layout = current_layout.permute(&axes_i32)?;
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Shape(ShapeOp::Permute),
                None,
                vec![self.id()],
                vec![result_id],
                vec![current_layout],
                vec![new_layout],
            )?;

            if requires_grad {
                gradient::record_operation(result_id, Op::Shape(ShapeOp::Permute), vec![self.id()])?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Shape(ShapeOp::Permute);
                gradient::record_operation(result.id(), op, vec![self.id()])?;
            }

            Ok(result)
        }
    }

    pub fn repeat(&self, repeats: &[usize]) -> HoduResult<Self> {
        let input_shape = self.shape();
        let ndim = input_shape.ndim();

        if repeats.len() != ndim {
            return Err(HoduError::InvalidLayout {
                reason: format!("repeats length {} must match tensor rank {}", repeats.len(), ndim),
            });
        }

        // Use tile operation: repeat each element before moving to next
        // For each dimension, we expand then reshape
        let mut result = self.clone();

        for (dim_idx, &repeat_count) in repeats.iter().enumerate() {
            if repeat_count == 1 {
                continue; // No need to repeat
            }

            let current_shape = result.shape();
            let current_dims = current_shape.dims();

            // Add a new dimension after current dim and broadcast
            let mut expanded_shape = current_dims.to_vec();
            expanded_shape.insert(dim_idx + 1, 1);

            // Reshape to add singleton dimension
            result = result.reshape(&expanded_shape)?;

            // Broadcast the singleton dimension
            let mut broadcast_shape = expanded_shape.clone();
            broadcast_shape[dim_idx + 1] = repeat_count;
            result = result.broadcast(&broadcast_shape)?;

            // Reshape to merge the repeated dimension
            let mut final_shape = current_dims.to_vec();
            final_shape[dim_idx] *= repeat_count;
            result = result.reshape(&final_shape)?;
        }

        Ok(result)
    }

    pub fn slice<D: Into<Scalar>, S: Into<Scalar> + Copy>(
        &self,
        dim: D,
        start: S,
        end: Option<S>,
        step: S,
    ) -> HoduResult<Self> {
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.slice(dim, start, end, step);
        }

        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();

        let start_scalar = start.into();
        let start_i32 = start_scalar.to_i32();

        let end_i32 = end.map(|e| {
            let end_scalar = e.into();
            end_scalar.to_i32()
        });

        let step_scalar = step.into();
        let step_i32 = step_scalar.to_i32();

        let current_layout = self.layout();
        let new_layout = current_layout.slice(dim_i32, start_i32, end_i32, step_i32)?;
        let requires_grad = self.is_requires_grad();

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), requires_grad);

            let end_scalar = Scalar::from(end_i32.unwrap_or(i32::MAX));
            let scalars = vec![dim_scalar, start_scalar, end_scalar, step_scalar];

            register_operation_in_builder(
                Op::ShapeScalars(ShapeScalarsOp::Slice),
                Some(OpParams {
                    scalars: scalars.clone(),
                    ..Default::default()
                }),
                vec![self.id()],
                vec![result_id],
                vec![current_layout],
                vec![new_layout],
            )?;

            if requires_grad {
                gradient::record_operation_with_scalars(
                    result_id,
                    Op::ShapeScalars(ShapeScalarsOp::Slice),
                    vec![self.id()],
                    scalars,
                )?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::ShapeScalars(ShapeScalarsOp::Slice);
                let end_scalar = Scalar::from(end_i32.unwrap_or(i32::MAX));
                let scalars = vec![dim_scalar, start_scalar, end_scalar, step_scalar];
                gradient::record_operation_with_scalars(result.id(), op, vec![self.id()], scalars)?;
            }

            Ok(result)
        }
    }
}
