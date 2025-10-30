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
    },
    types::layout::Layout,
};

// Convolution Operations
impl Tensor {
    pub fn conv1d(&self, weight: &Self, stride: usize, padding: usize, dilation: usize) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let weight_layout = weight.get_layout();
        let input_shape = input_layout.get_shape();
        let weight_shape = weight_layout.get_shape();

        // Input: [batch, in_channels, length]
        // Weight: [out_channels, in_channels, kernel_size]
        if input_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv1d - input must be 3D [batch, in_channels, length]".to_string(),
            });
        }
        if weight_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv1d - weight must be 3D [out_channels, in_channels, kernel_size]".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let length_input = input_shape[2];
        let channels_output = weight_shape[0];
        let kernel_size = weight_shape[2];

        if input_shape[1] != weight_shape[1] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: weight_shape.to_vec(),
                op: "conv1d - input and weight channel mismatch".to_string(),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], "conv1d")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "conv1d")?;
        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(length_input as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_size as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];
        let op = Op::Conv(op::ConvOp::Conv1d, self.id(), weight.id(), params_scalars.clone());
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, weight], "conv1d")?;

        let params = op::conv::ParamsConv1D {
            batch_size,
            length_input,
            channels_output,
            channels_input,
            kernel_size,
            padding,
            stride,
            dilation,
        };

        let output_length = (length_input + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        let output_shape = vec![batch_size, channels_output, output_length];

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone(), weight_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), weight.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.conv1d(weight_storage, &input_layout, &weight_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let params_scalars = vec![
                    Scalar::U32(batch_size as u32),
                    Scalar::U32(length_input as u32),
                    Scalar::U32(channels_output as u32),
                    Scalar::U32(channels_input as u32),
                    Scalar::U32(kernel_size as u32),
                    Scalar::U32(padding as u32),
                    Scalar::U32(stride as u32),
                    Scalar::U32(dilation as u32),
                ];
                let op = Op::Conv(op::ConvOp::Conv1d, self.id(), weight.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id(), weight.id()])?;
            }

            Ok(result)
        }
    }

    pub fn conv2d(&self, weight: &Self, stride: usize, padding: usize, dilation: usize) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let weight_layout = weight.get_layout();
        let input_shape = input_layout.get_shape();
        let weight_shape = weight_layout.get_shape();

        // Input: [batch, in_channels, height, width]
        // Weight: [out_channels, in_channels, kernel_h, kernel_w]
        if input_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv2d - input must be 4D [batch, in_channels, height, width]".to_string(),
            });
        }
        if weight_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv2d - weight must be 4D [out_channels, in_channels, kernel_h, kernel_w]".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];
        let channels_output = weight_shape[0];
        let kernel_height = weight_shape[2];
        let kernel_width = weight_shape[3];

        if input_shape[1] != weight_shape[1] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: weight_shape.to_vec(),
                op: "conv2d - input and weight channel mismatch".to_string(),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], "conv2d")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "conv2d")?;
        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(input_height as u32),
            Scalar::U32(input_width as u32),
            Scalar::U32(kernel_height as u32),
            Scalar::U32(kernel_width as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];
        let op = Op::Conv(op::ConvOp::Conv2d, self.id(), weight.id(), params_scalars.clone());
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, weight], "conv2d")?;

        let params = op::conv::ParamsConv2D {
            batch_size,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            channels_output,
            channels_input,
            padding,
            stride,
            dilation,
        };

        let output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
        let output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;
        let output_shape = vec![batch_size, channels_output, output_height, output_width];

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone(), weight_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), weight.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.conv2d(weight_storage, &input_layout, &weight_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let params_scalars = vec![
                    Scalar::U32(batch_size as u32),
                    Scalar::U32(input_height as u32),
                    Scalar::U32(input_width as u32),
                    Scalar::U32(kernel_height as u32),
                    Scalar::U32(kernel_width as u32),
                    Scalar::U32(channels_output as u32),
                    Scalar::U32(channels_input as u32),
                    Scalar::U32(padding as u32),
                    Scalar::U32(stride as u32),
                    Scalar::U32(dilation as u32),
                ];
                let op = Op::Conv(op::ConvOp::Conv2d, self.id(), weight.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id(), weight.id()])?;
            }

            Ok(result)
        }
    }

    pub fn conv3d(&self, weight: &Self, stride: usize, padding: usize, dilation: usize) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let weight_layout = weight.get_layout();
        let input_shape = input_layout.get_shape();
        let weight_shape = weight_layout.get_shape();

        // Input: [batch, in_channels, depth, height, width]
        // Weight: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
        if input_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv3d - input must be 5D [batch, in_channels, depth, height, width]".to_string(),
            });
        }
        if weight_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv3d - weight must be 5D [out_channels, in_channels, kernel_d, kernel_h, kernel_w]".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let input_depth = input_shape[2];
        let input_height = input_shape[3];
        let input_width = input_shape[4];
        let channels_output = weight_shape[0];
        let kernel_depth = weight_shape[2];
        let kernel_height = weight_shape[3];
        let kernel_width = weight_shape[4];

        if input_shape[1] != weight_shape[1] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: weight_shape.to_vec(),
                op: "conv3d - input and weight channel mismatch".to_string(),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], "conv3d")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "conv3d")?;
        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(input_depth as u32),
            Scalar::U32(input_height as u32),
            Scalar::U32(input_width as u32),
            Scalar::U32(kernel_depth as u32),
            Scalar::U32(kernel_height as u32),
            Scalar::U32(kernel_width as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];
        let op = Op::Conv(op::ConvOp::Conv3d, self.id(), weight.id(), params_scalars.clone());
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, weight], "conv3d")?;

        let params = op::conv::ParamsConv3D {
            batch_size,
            input_depth,
            input_height,
            input_width,
            kernel_depth,
            kernel_height,
            kernel_width,
            channels_output,
            channels_input,
            padding,
            stride,
            dilation,
        };

        let output_depth = (input_depth + 2 * padding - dilation * (kernel_depth - 1) - 1) / stride + 1;
        let output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
        let output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;
        let output_shape = vec![batch_size, channels_output, output_depth, output_height, output_width];

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone(), weight_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), weight.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.conv3d(weight_storage, &input_layout, &weight_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let params_scalars = vec![
                    Scalar::U32(batch_size as u32),
                    Scalar::U32(input_depth as u32),
                    Scalar::U32(input_height as u32),
                    Scalar::U32(input_width as u32),
                    Scalar::U32(kernel_depth as u32),
                    Scalar::U32(kernel_height as u32),
                    Scalar::U32(kernel_width as u32),
                    Scalar::U32(channels_output as u32),
                    Scalar::U32(channels_input as u32),
                    Scalar::U32(padding as u32),
                    Scalar::U32(stride as u32),
                    Scalar::U32(dilation as u32),
                ];
                let op = Op::Conv(op::ConvOp::Conv3d, self.id(), weight.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id(), weight.id()])?;
            }

            Ok(result)
        }
    }

    pub fn conv_transpose1d(
        &self,
        weight: &Self,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let weight_layout = weight.get_layout();
        let input_shape = input_layout.get_shape();
        let weight_shape = weight_layout.get_shape();

        // Input: [batch, in_channels, length]
        // Weight: [in_channels, out_channels, kernel_size] (note: different from conv!)
        if input_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose1d - input must be 3D [batch, in_channels, length]".to_string(),
            });
        }
        if weight_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose1d - weight must be 3D [in_channels, out_channels, kernel_size]".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let length_input = input_shape[2];
        let channels_output = weight_shape[1];
        let kernel_size = weight_shape[2];

        if input_shape[1] != weight_shape[0] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: weight_shape.to_vec(),
                op: "conv_transpose1d - input and weight channel mismatch".to_string(),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], "conv_transpose1d")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "conv_transpose1d")?;
        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(length_input as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_size as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(output_padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];
        let op = Op::Conv(
            op::ConvOp::ConvTranspose1d,
            self.id(),
            weight.id(),
            params_scalars.clone(),
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, weight], "conv_transpose1d")?;

        let params = op::conv::ParamsConvTranspose1D {
            batch_size,
            length_input,
            channels_output,
            channels_input,
            kernel_size,
            padding,
            output_padding,
            stride,
            dilation,
        };

        let output_length =
            (length_input - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
        let output_shape = vec![batch_size, channels_output, output_length];

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone(), weight_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), weight.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.conv_transpose1d(weight_storage, &input_layout, &weight_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let params_scalars = vec![
                    Scalar::U32(batch_size as u32),
                    Scalar::U32(length_input as u32),
                    Scalar::U32(channels_output as u32),
                    Scalar::U32(channels_input as u32),
                    Scalar::U32(kernel_size as u32),
                    Scalar::U32(padding as u32),
                    Scalar::U32(output_padding as u32),
                    Scalar::U32(stride as u32),
                    Scalar::U32(dilation as u32),
                ];
                let op = Op::Conv(op::ConvOp::ConvTranspose1d, self.id(), weight.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id(), weight.id()])?;
            }

            Ok(result)
        }
    }

    pub fn conv_transpose2d(
        &self,
        weight: &Self,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let weight_layout = weight.get_layout();
        let input_shape = input_layout.get_shape();
        let weight_shape = weight_layout.get_shape();

        // Input: [batch, in_channels, height, width]
        // Weight: [in_channels, out_channels, kernel_h, kernel_w]
        if input_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose2d - input must be 4D [batch, in_channels, height, width]".to_string(),
            });
        }
        if weight_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose2d - weight must be 4D [in_channels, out_channels, kernel_h, kernel_w]".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];
        let channels_output = weight_shape[1];
        let kernel_height = weight_shape[2];
        let kernel_width = weight_shape[3];

        if input_shape[1] != weight_shape[0] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: weight_shape.to_vec(),
                op: "conv_transpose2d - input and weight channel mismatch".to_string(),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], "conv_transpose2d")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "conv_transpose2d")?;
        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(input_height as u32),
            Scalar::U32(input_width as u32),
            Scalar::U32(kernel_height as u32),
            Scalar::U32(kernel_width as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(output_padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];
        let op = Op::Conv(
            op::ConvOp::ConvTranspose2d,
            self.id(),
            weight.id(),
            params_scalars.clone(),
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, weight], "conv_transpose2d")?;

        let params = op::conv::ParamsConvTranspose2D {
            batch_size,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            channels_output,
            channels_input,
            padding,
            output_padding,
            stride,
            dilation,
        };

        let output_height =
            (input_height - 1) * stride - 2 * padding + dilation * (kernel_height - 1) + output_padding + 1;
        let output_width =
            (input_width - 1) * stride - 2 * padding + dilation * (kernel_width - 1) + output_padding + 1;
        let output_shape = vec![batch_size, channels_output, output_height, output_width];

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone(), weight_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), weight.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.conv_transpose2d(weight_storage, &input_layout, &weight_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let params_scalars = vec![
                    Scalar::U32(batch_size as u32),
                    Scalar::U32(input_height as u32),
                    Scalar::U32(input_width as u32),
                    Scalar::U32(kernel_height as u32),
                    Scalar::U32(kernel_width as u32),
                    Scalar::U32(channels_output as u32),
                    Scalar::U32(channels_input as u32),
                    Scalar::U32(padding as u32),
                    Scalar::U32(output_padding as u32),
                    Scalar::U32(stride as u32),
                    Scalar::U32(dilation as u32),
                ];
                let op = Op::Conv(op::ConvOp::ConvTranspose2d, self.id(), weight.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id(), weight.id()])?;
            }

            Ok(result)
        }
    }

    pub fn conv_transpose3d(
        &self,
        weight: &Self,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let weight_layout = weight.get_layout();
        let input_shape = input_layout.get_shape();
        let weight_shape = weight_layout.get_shape();

        // Input: [batch, in_channels, depth, height, width]
        // Weight: [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
        if input_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose3d - input must be 5D [batch, in_channels, depth, height, width]".to_string(),
            });
        }
        if weight_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose3d - weight must be 5D [in_channels, out_channels, kernel_d, kernel_h, kernel_w]"
                    .to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let input_depth = input_shape[2];
        let input_height = input_shape[3];
        let input_width = input_shape[4];
        let channels_output = weight_shape[1];
        let kernel_depth = weight_shape[2];
        let kernel_height = weight_shape[3];
        let kernel_width = weight_shape[4];

        if input_shape[1] != weight_shape[0] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: weight_shape.to_vec(),
                op: "conv_transpose3d - input and weight channel mismatch".to_string(),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], "conv_transpose3d")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "conv_transpose3d")?;
        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(input_depth as u32),
            Scalar::U32(input_height as u32),
            Scalar::U32(input_width as u32),
            Scalar::U32(kernel_depth as u32),
            Scalar::U32(kernel_height as u32),
            Scalar::U32(kernel_width as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(output_padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];
        let op = Op::Conv(
            op::ConvOp::ConvTranspose3d,
            self.id(),
            weight.id(),
            params_scalars.clone(),
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, weight], "conv_transpose3d")?;

        let params = op::conv::ParamsConvTranspose3D {
            batch_size,
            input_depth,
            input_height,
            input_width,
            kernel_depth,
            kernel_height,
            kernel_width,
            channels_output,
            channels_input,
            padding,
            output_padding,
            stride,
            dilation,
        };

        let output_depth =
            (input_depth - 1) * stride - 2 * padding + dilation * (kernel_depth - 1) + output_padding + 1;
        let output_height =
            (input_height - 1) * stride - 2 * padding + dilation * (kernel_height - 1) + output_padding + 1;
        let output_width =
            (input_width - 1) * stride - 2 * padding + dilation * (kernel_width - 1) + output_padding + 1;
        let output_shape = vec![batch_size, channels_output, output_depth, output_height, output_width];

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![input_layout.clone(), weight_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), weight.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.conv_transpose3d(weight_storage, &input_layout, &weight_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad() || weight.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let params_scalars = vec![
                    Scalar::U32(batch_size as u32),
                    Scalar::U32(input_depth as u32),
                    Scalar::U32(input_height as u32),
                    Scalar::U32(input_width as u32),
                    Scalar::U32(kernel_depth as u32),
                    Scalar::U32(kernel_height as u32),
                    Scalar::U32(kernel_width as u32),
                    Scalar::U32(channels_output as u32),
                    Scalar::U32(channels_input as u32),
                    Scalar::U32(padding as u32),
                    Scalar::U32(output_padding as u32),
                    Scalar::U32(stride as u32),
                    Scalar::U32(dilation as u32),
                ];
                let op = Op::Conv(op::ConvOp::ConvTranspose3d, self.id(), weight.id(), params_scalars);
                gradient::record_operation(result.id(), op, vec![self.id(), weight.id()])?;
            }

            Ok(result)
        }
    }

    // Convolution gradient operations (internal use for backpropagation)
    pub(crate) fn conv1d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let grad_output_layout = grad_output.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_output_shape = grad_output_layout.get_shape();

        // Input: [batch, in_channels, length]
        // GradOutput: [batch, out_channels, length_out]
        // Weight: [out_channels, in_channels, kernel_size]
        if input_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv1d_grad_weight - input must be 3D".to_string(),
            });
        }
        if grad_output_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
                op: "conv1d_grad_weight - grad_output must be 3D".to_string(),
            });
        }
        if weight_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv1d_grad_weight - weight_shape must be 3D".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let length_input = input_shape[2];
        let channels_output = grad_output_shape[1];
        let kernel_size = weight_shape[2];

        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(length_input as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_size as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];

        let op = Op::Conv(
            op::ConvOp::Conv1dGradWeight,
            self.id(),
            grad_output.id(),
            params_scalars,
        );

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(weight_shape);
            let requires_grad = false; // Gradient of gradient not tracked
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op,
                vec![result_id],
                vec![input_layout.clone(), grad_output_layout.clone()],
                vec![result_layout],
            );

            Ok(result_tensor)
        } else {
            let params = op::conv::ParamsConv1D {
                batch_size,
                length_input,
                channels_output,
                channels_input,
                kernel_size,
                padding,
                stride,
                dilation,
            };

            let storage = self.with_storage(|input_storage| {
                grad_output.with_storage(|grad_output_storage| {
                    input_storage.conv1d_grad_weight(grad_output_storage, &input_layout, &grad_output_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(weight_shape);
            let result = from_storage_with_grad(storage, result_layout, true, false);

            Ok(result)
        }
    }

    pub(crate) fn conv2d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let grad_output_layout = grad_output.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_output_shape = grad_output_layout.get_shape();

        // Input: [batch, in_channels, height, width]
        // GradOutput: [batch, out_channels, height_out, width_out]
        // Weight: [out_channels, in_channels, kernel_h, kernel_w]
        if input_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv2d_grad_weight - input must be 4D".to_string(),
            });
        }
        if grad_output_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
                op: "conv2d_grad_weight - grad_output must be 4D".to_string(),
            });
        }
        if weight_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv2d_grad_weight - weight_shape must be 4D".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let height_input = input_shape[2];
        let width_input = input_shape[3];
        let channels_output = grad_output_shape[1];
        let kernel_h = weight_shape[2];
        let kernel_w = weight_shape[3];

        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(height_input as u32),
            Scalar::U32(width_input as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_h as u32),
            Scalar::U32(kernel_w as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];

        let op = Op::Conv(
            op::ConvOp::Conv2dGradWeight,
            self.id(),
            grad_output.id(),
            params_scalars,
        );

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(weight_shape);
            let requires_grad = false;
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op,
                vec![result_id],
                vec![input_layout.clone(), grad_output_layout.clone()],
                vec![result_layout],
            );

            Ok(result_tensor)
        } else {
            let params = op::conv::ParamsConv2D {
                batch_size,
                input_height: height_input,
                input_width: width_input,
                kernel_height: kernel_h,
                kernel_width: kernel_w,
                channels_output,
                channels_input,
                padding,
                stride,
                dilation,
            };

            let storage = self.with_storage(|input_storage| {
                grad_output.with_storage(|grad_output_storage| {
                    input_storage.conv2d_grad_weight(grad_output_storage, &input_layout, &grad_output_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(weight_shape);
            let result = from_storage_with_grad(storage, result_layout, true, false);

            Ok(result)
        }
    }

    pub(crate) fn conv3d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let grad_output_layout = grad_output.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_output_shape = grad_output_layout.get_shape();

        // Input: [batch, in_channels, depth, height, width]
        // GradOutput: [batch, out_channels, depth_out, height_out, width_out]
        // Weight: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
        if input_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv3d_grad_weight - input must be 5D".to_string(),
            });
        }
        if grad_output_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
                op: "conv3d_grad_weight - grad_output must be 5D".to_string(),
            });
        }
        if weight_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv3d_grad_weight - weight_shape must be 5D".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let depth_input = input_shape[2];
        let height_input = input_shape[3];
        let width_input = input_shape[4];
        let channels_output = grad_output_shape[1];
        let kernel_d = weight_shape[2];
        let kernel_h = weight_shape[3];
        let kernel_w = weight_shape[4];

        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(depth_input as u32),
            Scalar::U32(height_input as u32),
            Scalar::U32(width_input as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_d as u32),
            Scalar::U32(kernel_h as u32),
            Scalar::U32(kernel_w as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];

        let op = Op::Conv(
            op::ConvOp::Conv3dGradWeight,
            self.id(),
            grad_output.id(),
            params_scalars,
        );

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(weight_shape);
            let requires_grad = false;
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op,
                vec![result_id],
                vec![input_layout.clone(), grad_output_layout.clone()],
                vec![result_layout],
            );

            Ok(result_tensor)
        } else {
            let params = op::conv::ParamsConv3D {
                batch_size,
                input_depth: depth_input,
                input_height: height_input,
                input_width: width_input,
                kernel_depth: kernel_d,
                kernel_height: kernel_h,
                kernel_width: kernel_w,
                channels_output,
                channels_input,
                padding,
                stride,
                dilation,
            };

            let storage = self.with_storage(|input_storage| {
                grad_output.with_storage(|grad_output_storage| {
                    input_storage.conv3d_grad_weight(grad_output_storage, &input_layout, &grad_output_layout, &params)
                })
            })?;

            let result_layout = Layout::from_shape(weight_shape);
            let result = from_storage_with_grad(storage, result_layout, true, false);

            Ok(result)
        }
    }

    pub(crate) fn conv_transpose1d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let grad_output_layout = grad_output.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_output_shape = grad_output_layout.get_shape();

        // Input: [batch, in_channels, length_in]
        // GradOutput: [batch, out_channels, length_out]
        // Weight: [in_channels, out_channels, kernel_size]
        if input_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose1d_grad_weight - input must be 3D".to_string(),
            });
        }
        if grad_output_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose1d_grad_weight - grad_output must be 3D".to_string(),
            });
        }
        if weight_shape.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose1d_grad_weight - weight_shape must be 3D".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let length_input = input_shape[2];
        let channels_output = grad_output_shape[1];
        let kernel_size = weight_shape[2];

        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(length_input as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_size as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(output_padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];

        let op = Op::Conv(
            op::ConvOp::ConvTranspose1dGradWeight,
            self.id(),
            grad_output.id(),
            params_scalars,
        );

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(weight_shape);
            let requires_grad = false;
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);
            register_operation_in_builder(
                op,
                vec![result_id],
                vec![input_layout.clone(), grad_output_layout.clone()],
                vec![result_layout],
            );
            Ok(result_tensor)
        } else {
            let params = op::conv::ParamsConvTranspose1D {
                batch_size,
                length_input,
                channels_output,
                channels_input,
                kernel_size,
                padding,
                stride,
                output_padding,
                dilation,
            };
            let storage = self.with_storage(|input_storage| {
                grad_output.with_storage(|grad_output_storage| {
                    input_storage.conv_transpose1d_grad_weight(
                        grad_output_storage,
                        &input_layout,
                        &grad_output_layout,
                        &params,
                    )
                })
            })?;
            let result_layout = Layout::from_shape(weight_shape);
            let result = from_storage_with_grad(storage, result_layout, true, false);
            Ok(result)
        }
    }

    pub(crate) fn conv_transpose2d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let grad_output_layout = grad_output.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_output_shape = grad_output_layout.get_shape();

        // Input: [batch, in_channels, height_in, width_in]
        // GradOutput: [batch, out_channels, height_out, width_out]
        // Weight: [in_channels, out_channels, kernel_height, kernel_width]
        if input_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose2d_grad_weight - input must be 4D".to_string(),
            });
        }
        if grad_output_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose2d_grad_weight - grad_output must be 4D".to_string(),
            });
        }
        if weight_shape.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose2d_grad_weight - weight_shape must be 4D".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];
        let channels_output = grad_output_shape[1];
        let kernel_height = weight_shape[2];
        let kernel_width = weight_shape[3];

        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(input_height as u32),
            Scalar::U32(input_width as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_height as u32),
            Scalar::U32(kernel_width as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(output_padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];

        let op = Op::Conv(
            op::ConvOp::ConvTranspose2dGradWeight,
            self.id(),
            grad_output.id(),
            params_scalars,
        );

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(weight_shape);
            let requires_grad = false;
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);
            register_operation_in_builder(
                op,
                vec![result_id],
                vec![input_layout.clone(), grad_output_layout.clone()],
                vec![result_layout],
            );
            Ok(result_tensor)
        } else {
            let params = op::conv::ParamsConvTranspose2D {
                batch_size,
                input_height,
                input_width,
                kernel_height,
                kernel_width,
                channels_output,
                channels_input,
                padding,
                stride,
                output_padding,
                dilation,
            };
            let storage = self.with_storage(|input_storage| {
                grad_output.with_storage(|grad_output_storage| {
                    input_storage.conv_transpose2d_grad_weight(
                        grad_output_storage,
                        &input_layout,
                        &grad_output_layout,
                        &params,
                    )
                })
            })?;
            let result_layout = Layout::from_shape(weight_shape);
            let result = from_storage_with_grad(storage, result_layout, true, false);
            Ok(result)
        }
    }

    pub(crate) fn conv_transpose3d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[usize],
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> HoduResult<Self> {
        let input_layout = self.get_layout();
        let grad_output_layout = grad_output.get_layout();
        let input_shape = input_layout.get_shape();
        let grad_output_shape = grad_output_layout.get_shape();

        // Input: [batch, in_channels, depth_in, height_in, width_in]
        // GradOutput: [batch, out_channels, depth_out, height_out, width_out]
        // Weight: [in_channels, out_channels, kernel_depth, kernel_height, kernel_width]
        if input_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose3d_grad_weight - input must be 5D".to_string(),
            });
        }
        if grad_output_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose3d_grad_weight - grad_output must be 5D".to_string(),
            });
        }
        if weight_shape.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.to_vec(),
                rhs: vec![],
                op: "conv_transpose3d_grad_weight - weight_shape must be 5D".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let channels_input = input_shape[1];
        let input_depth = input_shape[2];
        let input_height = input_shape[3];
        let input_width = input_shape[4];
        let channels_output = grad_output_shape[1];
        let kernel_depth = weight_shape[2];
        let kernel_height = weight_shape[3];
        let kernel_width = weight_shape[4];

        let params_scalars = vec![
            Scalar::U32(batch_size as u32),
            Scalar::U32(input_depth as u32),
            Scalar::U32(input_height as u32),
            Scalar::U32(input_width as u32),
            Scalar::U32(channels_output as u32),
            Scalar::U32(channels_input as u32),
            Scalar::U32(kernel_depth as u32),
            Scalar::U32(kernel_height as u32),
            Scalar::U32(kernel_width as u32),
            Scalar::U32(padding as u32),
            Scalar::U32(output_padding as u32),
            Scalar::U32(stride as u32),
            Scalar::U32(dilation as u32),
        ];

        let op = Op::Conv(
            op::ConvOp::ConvTranspose3dGradWeight,
            self.id(),
            grad_output.id(),
            params_scalars,
        );

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(weight_shape);
            let requires_grad = false;
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);
            register_operation_in_builder(
                op,
                vec![result_id],
                vec![input_layout.clone(), grad_output_layout.clone()],
                vec![result_layout],
            );
            Ok(result_tensor)
        } else {
            let params = op::conv::ParamsConvTranspose3D {
                batch_size,
                input_depth,
                input_height,
                input_width,
                kernel_depth,
                kernel_height,
                kernel_width,
                channels_output,
                channels_input,
                padding,
                stride,
                output_padding,
                dilation,
            };
            let storage = self.with_storage(|input_storage| {
                grad_output.with_storage(|grad_output_storage| {
                    input_storage.conv_transpose3d_grad_weight(
                        grad_output_storage,
                        &input_layout,
                        &grad_output_layout,
                        &params,
                    )
                })
            })?;
            let result_layout = Layout::from_shape(weight_shape);
            let result = from_storage_with_grad(storage, result_layout, true, false);
            Ok(result)
        }
    }
}
