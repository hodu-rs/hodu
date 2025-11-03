use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{ConvOp, Op, OpParams},
    scalar::Scalar,
    script::builder,
    tensor::{create_builder_tensor_with_grad, from_storage, gradient, register_operation_in_builder, Tensor},
    types::{Layout, Shape},
    utils::valid::{
        validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op, validate_same_device,
        validate_same_dtype,
    },
};

impl Tensor {
    pub fn conv1d(&self, weight: &Self, stride: u32, padding: u32, dilation: u32) -> HoduResult<Self> {
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let input_dims = input_shape.dims();
        let weight_dims = weight_shape.dims();

        // Input: [batch, in_channels, length]
        // Weight: [out_channels, in_channels, kernel_size]
        if input_dims.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::Conv1d),
            });
        }
        if weight_dims.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::Conv1d),
            });
        }

        let batch_size = input_dims[0];
        let _channels_input = input_dims[1];
        let length_input = input_dims[2];
        let channels_output = weight_dims[0];
        let kernel_size = weight_dims[2];

        if input_dims[1] != weight_dims[1] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: weight_shape.clone(),
                op: Op::Conv(ConvOp::Conv1d),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], Op::Conv(ConvOp::Conv1d))?;
        validate_same_dtype(&[self, weight], Op::Conv(ConvOp::Conv1d))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Conv(ConvOp::Conv1d))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Conv(ConvOp::Conv1d));

        let output_length = (length_input + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        let output_shape = vec![batch_size, channels_output, output_length];

        let stride_arr = [stride];
        let padding_arr = [padding];
        let dilation_arr = [dilation];

        // Calculate layouts before if-else block
        let input_layout = self.layout();
        let weight_layout = weight.layout();
        let result_layout = Layout::from_shape(&Shape::from(output_shape.clone()));
        let requires_grad = (self.is_requires_grad() || weight.is_requires_grad()) && validate_requires_grad;

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let channels_input = input_dims[1];
            let scalars = vec![
                Scalar::from(batch_size),
                Scalar::from(length_input),
                Scalar::from(channels_output),
                Scalar::from(channels_input),
                Scalar::from(kernel_size),
                Scalar::from(padding),
                Scalar::from(stride),
                Scalar::from(dilation),
            ];

            register_operation_in_builder(
                Op::Conv(ConvOp::Conv1d),
                Some(OpParams {
                    scalars: scalars.clone(),
                    ..Default::default()
                }),
                vec![self.id(), weight.id()],
                vec![result_id],
                vec![input_layout, weight_layout],
                vec![result_layout],
            )?;

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation_with_scalars(
                    result_id,
                    Op::Conv(ConvOp::Conv1d),
                    vec![self.id(), weight.id()],
                    scalars,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.call_conv(
                        &self.layout(),
                        weight_storage,
                        &weight.layout(),
                        &stride_arr,
                        &padding_arr,
                        &dilation_arr,
                        Op::Conv(ConvOp::Conv1d),
                    )
                })
            })?;

            let result = from_storage(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Conv(ConvOp::Conv1d);
                let channels_input = input_dims[1];
                let scalars = vec![
                    Scalar::from(batch_size),
                    Scalar::from(length_input),
                    Scalar::from(channels_output),
                    Scalar::from(channels_input),
                    Scalar::from(kernel_size),
                    Scalar::from(padding),
                    Scalar::from(stride),
                    Scalar::from(dilation),
                ];
                gradient::record_operation_with_scalars(result.id(), op, vec![self.id(), weight.id()], scalars)?;
            }

            Ok(result)
        }
    }

    pub fn conv2d(&self, weight: &Self, stride: u32, padding: u32, dilation: u32) -> HoduResult<Self> {
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let input_dims = input_shape.dims();
        let weight_dims = weight_shape.dims();

        // Input: [batch, in_channels, height, width]
        // Weight: [out_channels, in_channels, kernel_h, kernel_w]
        if input_dims.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::Conv2d),
            });
        }
        if weight_dims.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::Conv2d),
            });
        }

        let batch_size = input_dims[0];
        let _channels_input = input_dims[1];
        let input_height = input_dims[2];
        let input_width = input_dims[3];
        let channels_output = weight_dims[0];
        let kernel_height = weight_dims[2];
        let kernel_width = weight_dims[3];

        if input_dims[1] != weight_dims[1] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: weight_shape.clone(),
                op: Op::Conv(ConvOp::Conv2d),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], Op::Conv(ConvOp::Conv2d))?;
        validate_same_dtype(&[self, weight], Op::Conv(ConvOp::Conv2d))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Conv(ConvOp::Conv2d))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Conv(ConvOp::Conv2d));

        let output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
        let output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;
        let output_shape = vec![batch_size, channels_output, output_height, output_width];

        let stride_arr = [stride; 2];
        let padding_arr = [padding; 2];
        let dilation_arr = [dilation; 2];

        // Calculate layouts before if-else block
        let input_layout = self.layout();
        let weight_layout = weight.layout();
        let result_layout = Layout::from_shape(&Shape::from(output_shape.clone()));
        let requires_grad = (self.is_requires_grad() || weight.is_requires_grad()) && validate_requires_grad;

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let channels_input = input_dims[1];
            let scalars = vec![
                Scalar::from(batch_size),
                Scalar::from(input_height),
                Scalar::from(input_width),
                Scalar::from(kernel_height),
                Scalar::from(kernel_width),
                Scalar::from(channels_output),
                Scalar::from(channels_input),
                Scalar::from(padding),
                Scalar::from(stride),
                Scalar::from(dilation),
            ];

            register_operation_in_builder(
                Op::Conv(ConvOp::Conv2d),
                Some(OpParams {
                    scalars: scalars.clone(),
                    ..Default::default()
                }),
                vec![self.id(), weight.id()],
                vec![result_id],
                vec![input_layout, weight_layout],
                vec![result_layout],
            )?;

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation_with_scalars(
                    result_id,
                    Op::Conv(ConvOp::Conv2d),
                    vec![self.id(), weight.id()],
                    scalars,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.call_conv(
                        &self.layout(),
                        weight_storage,
                        &weight.layout(),
                        &stride_arr,
                        &padding_arr,
                        &dilation_arr,
                        Op::Conv(ConvOp::Conv2d),
                    )
                })
            })?;

            let result = from_storage(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Conv(ConvOp::Conv2d);
                let channels_input = input_dims[1];
                let scalars = vec![
                    Scalar::from(batch_size),
                    Scalar::from(input_height),
                    Scalar::from(input_width),
                    Scalar::from(kernel_height),
                    Scalar::from(kernel_width),
                    Scalar::from(channels_output),
                    Scalar::from(channels_input),
                    Scalar::from(padding),
                    Scalar::from(stride),
                    Scalar::from(dilation),
                ];
                gradient::record_operation_with_scalars(result.id(), op, vec![self.id(), weight.id()], scalars)?;
            }

            Ok(result)
        }
    }

    pub fn conv3d(&self, weight: &Self, stride: u32, padding: u32, dilation: u32) -> HoduResult<Self> {
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let input_dims = input_shape.dims();
        let weight_dims = weight_shape.dims();

        // Input: [batch, in_channels, depth, height, width]
        // Weight: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
        if input_dims.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::Conv3d),
            });
        }
        if weight_dims.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::Conv3d),
            });
        }

        let batch_size = input_dims[0];
        let _channels_input = input_dims[1];
        let input_depth = input_dims[2];
        let input_height = input_dims[3];
        let input_width = input_dims[4];
        let channels_output = weight_dims[0];
        let kernel_depth = weight_dims[2];
        let kernel_height = weight_dims[3];
        let kernel_width = weight_dims[4];

        if input_dims[1] != weight_dims[1] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: weight_shape.clone(),
                op: Op::Conv(ConvOp::Conv3d),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], Op::Conv(ConvOp::Conv3d))?;
        validate_same_dtype(&[self, weight], Op::Conv(ConvOp::Conv3d))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Conv(ConvOp::Conv3d))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Conv(ConvOp::Conv3d));

        let output_depth = (input_depth + 2 * padding - dilation * (kernel_depth - 1) - 1) / stride + 1;
        let output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
        let output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;
        let output_shape = vec![batch_size, channels_output, output_depth, output_height, output_width];

        let stride_arr = [stride; 3];
        let padding_arr = [padding; 3];
        let dilation_arr = [dilation; 3];

        // Calculate layouts before if-else block
        let input_layout = self.layout();
        let weight_layout = weight.layout();
        let result_layout = Layout::from_shape(&Shape::from(output_shape.clone()));
        let requires_grad = (self.is_requires_grad() || weight.is_requires_grad()) && validate_requires_grad;

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let channels_input = input_dims[1];
            let scalars = vec![
                Scalar::from(batch_size),
                Scalar::from(input_depth),
                Scalar::from(input_height),
                Scalar::from(input_width),
                Scalar::from(kernel_depth),
                Scalar::from(kernel_height),
                Scalar::from(kernel_width),
                Scalar::from(channels_output),
                Scalar::from(channels_input),
                Scalar::from(padding),
                Scalar::from(stride),
                Scalar::from(dilation),
            ];

            register_operation_in_builder(
                Op::Conv(ConvOp::Conv3d),
                Some(OpParams {
                    scalars: scalars.clone(),
                    ..Default::default()
                }),
                vec![self.id(), weight.id()],
                vec![result_id],
                vec![input_layout, weight_layout],
                vec![result_layout],
            )?;

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation_with_scalars(
                    result_id,
                    Op::Conv(ConvOp::Conv3d),
                    vec![self.id(), weight.id()],
                    scalars,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.call_conv(
                        &self.layout(),
                        weight_storage,
                        &weight.layout(),
                        &stride_arr,
                        &padding_arr,
                        &dilation_arr,
                        Op::Conv(ConvOp::Conv3d),
                    )
                })
            })?;

            let result = from_storage(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Conv(ConvOp::Conv3d);
                let channels_input = input_dims[1];
                let scalars = vec![
                    Scalar::from(batch_size),
                    Scalar::from(input_depth),
                    Scalar::from(input_height),
                    Scalar::from(input_width),
                    Scalar::from(kernel_depth),
                    Scalar::from(kernel_height),
                    Scalar::from(kernel_width),
                    Scalar::from(channels_output),
                    Scalar::from(channels_input),
                    Scalar::from(padding),
                    Scalar::from(stride),
                    Scalar::from(dilation),
                ];
                gradient::record_operation_with_scalars(result.id(), op, vec![self.id(), weight.id()], scalars)?;
            }

            Ok(result)
        }
    }

    pub fn conv_transpose1d(
        &self,
        weight: &Self,
        stride: u32,
        padding: u32,
        output_padding: u32,
        dilation: u32,
    ) -> HoduResult<Self> {
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let input_dims = input_shape.dims();
        let weight_dims = weight_shape.dims();

        // Input: [batch, in_channels, length]
        // Weight: [in_channels, out_channels, kernel_size] (note: different from conv!)
        if input_dims.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::ConvTranspose1d),
            });
        }
        if weight_dims.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::ConvTranspose1d),
            });
        }

        let batch_size = input_dims[0];
        let _channels_input = input_dims[1];
        let length_input = input_dims[2];
        let channels_output = weight_dims[1];
        let kernel_size = weight_dims[2];

        if input_dims[1] != weight_dims[0] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: weight_shape.clone(),
                op: Op::Conv(ConvOp::ConvTranspose1d),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], Op::Conv(ConvOp::ConvTranspose1d))?;
        validate_same_dtype(&[self, weight], Op::Conv(ConvOp::ConvTranspose1d))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Conv(ConvOp::ConvTranspose1d))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Conv(ConvOp::ConvTranspose1d));

        let output_length =
            (length_input - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
        let output_shape = vec![batch_size, channels_output, output_length];

        let stride_arr = [stride];
        let padding_arr = [padding];
        let dilation_arr = [dilation];

        // Calculate layouts before if-else block
        let input_layout = self.layout();
        let weight_layout = weight.layout();
        let result_layout = Layout::from_shape(&Shape::from(output_shape.clone()));
        let requires_grad = (self.is_requires_grad() || weight.is_requires_grad()) && validate_requires_grad;

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let channels_input = input_dims[1];
            let scalars = vec![
                Scalar::from(batch_size),
                Scalar::from(length_input),
                Scalar::from(channels_output),
                Scalar::from(channels_input),
                Scalar::from(kernel_size),
                Scalar::from(padding),
                Scalar::from(output_padding),
                Scalar::from(stride),
                Scalar::from(dilation),
            ];

            register_operation_in_builder(
                Op::Conv(ConvOp::ConvTranspose1d),
                Some(OpParams {
                    scalars: scalars.clone(),
                    ..Default::default()
                }),
                vec![self.id(), weight.id()],
                vec![result_id],
                vec![input_layout, weight_layout],
                vec![result_layout],
            )?;

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation_with_scalars(
                    result_id,
                    Op::Conv(ConvOp::ConvTranspose1d),
                    vec![self.id(), weight.id()],
                    scalars,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.call_conv(
                        &self.layout(),
                        weight_storage,
                        &weight.layout(),
                        &stride_arr,
                        &padding_arr,
                        &dilation_arr,
                        Op::Conv(ConvOp::ConvTranspose1d),
                    )
                })
            })?;

            let result = from_storage(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Conv(ConvOp::ConvTranspose1d);
                let channels_input = input_dims[1];
                let scalars = vec![
                    Scalar::from(batch_size),
                    Scalar::from(length_input),
                    Scalar::from(channels_output),
                    Scalar::from(channels_input),
                    Scalar::from(kernel_size),
                    Scalar::from(padding),
                    Scalar::from(output_padding),
                    Scalar::from(stride),
                    Scalar::from(dilation),
                ];
                gradient::record_operation_with_scalars(result.id(), op, vec![self.id(), weight.id()], scalars)?;
            }

            Ok(result)
        }
    }

    pub fn conv_transpose2d(
        &self,
        weight: &Self,
        stride: u32,
        padding: u32,
        output_padding: u32,
        dilation: u32,
    ) -> HoduResult<Self> {
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let input_dims = input_shape.dims();
        let weight_dims = weight_shape.dims();

        // Input: [batch, in_channels, height, width]
        // Weight: [in_channels, out_channels, kernel_h, kernel_w]
        if input_dims.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::ConvTranspose2d),
            });
        }
        if weight_dims.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::ConvTranspose2d),
            });
        }

        let batch_size = input_dims[0];
        let _channels_input = input_dims[1];
        let input_height = input_dims[2];
        let input_width = input_dims[3];
        let channels_output = weight_dims[1];
        let kernel_height = weight_dims[2];
        let kernel_width = weight_dims[3];

        if input_dims[1] != weight_dims[0] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: weight_shape.clone(),
                op: Op::Conv(ConvOp::ConvTranspose2d),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], Op::Conv(ConvOp::ConvTranspose2d))?;
        validate_same_dtype(&[self, weight], Op::Conv(ConvOp::ConvTranspose2d))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Conv(ConvOp::ConvTranspose2d))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Conv(ConvOp::ConvTranspose2d));

        let output_height =
            (input_height - 1) * stride - 2 * padding + dilation * (kernel_height - 1) + output_padding + 1;
        let output_width =
            (input_width - 1) * stride - 2 * padding + dilation * (kernel_width - 1) + output_padding + 1;
        let output_shape = vec![batch_size, channels_output, output_height, output_width];

        let stride_arr = [stride; 2];
        let padding_arr = [padding; 2];
        let dilation_arr = [dilation; 2];

        // Calculate layouts before if-else block
        let input_layout = self.layout();
        let weight_layout = weight.layout();
        let result_layout = Layout::from_shape(&Shape::from(output_shape.clone()));
        let requires_grad = (self.is_requires_grad() || weight.is_requires_grad()) && validate_requires_grad;

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let channels_input = input_dims[1];
            let scalars = vec![
                Scalar::from(batch_size),
                Scalar::from(input_height),
                Scalar::from(input_width),
                Scalar::from(kernel_height),
                Scalar::from(kernel_width),
                Scalar::from(channels_output),
                Scalar::from(channels_input),
                Scalar::from(padding),
                Scalar::from(output_padding),
                Scalar::from(stride),
                Scalar::from(dilation),
            ];

            register_operation_in_builder(
                Op::Conv(ConvOp::ConvTranspose2d),
                Some(OpParams {
                    scalars: scalars.clone(),
                    ..Default::default()
                }),
                vec![self.id(), weight.id()],
                vec![result_id],
                vec![input_layout, weight_layout],
                vec![result_layout],
            )?;

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation_with_scalars(
                    result_id,
                    Op::Conv(ConvOp::ConvTranspose2d),
                    vec![self.id(), weight.id()],
                    scalars,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.call_conv(
                        &self.layout(),
                        weight_storage,
                        &weight.layout(),
                        &stride_arr,
                        &padding_arr,
                        &dilation_arr,
                        Op::Conv(ConvOp::ConvTranspose2d),
                    )
                })
            })?;

            let result = from_storage(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Conv(ConvOp::ConvTranspose2d);
                let channels_input = input_dims[1];
                let scalars = vec![
                    Scalar::from(batch_size),
                    Scalar::from(input_height),
                    Scalar::from(input_width),
                    Scalar::from(kernel_height),
                    Scalar::from(kernel_width),
                    Scalar::from(channels_output),
                    Scalar::from(channels_input),
                    Scalar::from(padding),
                    Scalar::from(output_padding),
                    Scalar::from(stride),
                    Scalar::from(dilation),
                ];
                gradient::record_operation_with_scalars(result.id(), op, vec![self.id(), weight.id()], scalars)?;
            }

            Ok(result)
        }
    }

    pub fn conv_transpose3d(
        &self,
        weight: &Self,
        stride: u32,
        padding: u32,
        output_padding: u32,
        dilation: u32,
    ) -> HoduResult<Self> {
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let input_dims = input_shape.dims();
        let weight_dims = weight_shape.dims();

        // Input: [batch, in_channels, depth, height, width]
        // Weight: [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
        if input_dims.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::ConvTranspose3d),
            });
        }
        if weight_dims.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: weight_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::ConvTranspose3d),
            });
        }

        let batch_size = input_dims[0];
        let _channels_input = input_dims[1];
        let input_depth = input_dims[2];
        let input_height = input_dims[3];
        let input_width = input_dims[4];
        let channels_output = weight_dims[1];
        let kernel_depth = weight_dims[2];
        let kernel_height = weight_dims[3];
        let kernel_width = weight_dims[4];

        if input_dims[1] != weight_dims[0] {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: weight_shape.clone(),
                op: Op::Conv(ConvOp::ConvTranspose3d),
            });
        }

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, weight], Op::Conv(ConvOp::ConvTranspose3d))?;
        validate_same_dtype(&[self, weight], Op::Conv(ConvOp::ConvTranspose3d))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Conv(ConvOp::ConvTranspose3d))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Conv(ConvOp::ConvTranspose3d));

        let output_depth =
            (input_depth - 1) * stride - 2 * padding + dilation * (kernel_depth - 1) + output_padding + 1;
        let output_height =
            (input_height - 1) * stride - 2 * padding + dilation * (kernel_height - 1) + output_padding + 1;
        let output_width =
            (input_width - 1) * stride - 2 * padding + dilation * (kernel_width - 1) + output_padding + 1;
        let output_shape = vec![batch_size, channels_output, output_depth, output_height, output_width];

        let stride_arr = [stride; 3];
        let padding_arr = [padding; 3];
        let dilation_arr = [dilation; 3];

        // Calculate layouts before if-else block
        let input_layout = self.layout();
        let weight_layout = weight.layout();
        let result_layout = Layout::from_shape(&Shape::from(output_shape.clone()));
        let requires_grad = (self.is_requires_grad() || weight.is_requires_grad()) && validate_requires_grad;

        if builder::is_builder_active() {
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let channels_input = input_dims[1];
            let scalars = vec![
                Scalar::from(batch_size),
                Scalar::from(input_depth),
                Scalar::from(input_height),
                Scalar::from(input_width),
                Scalar::from(kernel_depth),
                Scalar::from(kernel_height),
                Scalar::from(kernel_width),
                Scalar::from(channels_output),
                Scalar::from(channels_input),
                Scalar::from(padding),
                Scalar::from(output_padding),
                Scalar::from(stride),
                Scalar::from(dilation),
            ];

            register_operation_in_builder(
                Op::Conv(ConvOp::ConvTranspose3d),
                Some(OpParams {
                    scalars: scalars.clone(),
                    ..Default::default()
                }),
                vec![self.id(), weight.id()],
                vec![result_id],
                vec![input_layout, weight_layout],
                vec![result_layout],
            )?;

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation_with_scalars(
                    result_id,
                    Op::Conv(ConvOp::ConvTranspose3d),
                    vec![self.id(), weight.id()],
                    scalars,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                weight.with_storage(|weight_storage| {
                    input_storage.call_conv(
                        &self.layout(),
                        weight_storage,
                        &weight.layout(),
                        &stride_arr,
                        &padding_arr,
                        &dilation_arr,
                        Op::Conv(ConvOp::ConvTranspose3d),
                    )
                })
            })?;

            let result = from_storage(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Conv(ConvOp::ConvTranspose3d);
                let channels_input = input_dims[1];
                let scalars = vec![
                    Scalar::from(batch_size),
                    Scalar::from(input_depth),
                    Scalar::from(input_height),
                    Scalar::from(input_width),
                    Scalar::from(kernel_depth),
                    Scalar::from(kernel_height),
                    Scalar::from(kernel_width),
                    Scalar::from(channels_output),
                    Scalar::from(channels_input),
                    Scalar::from(padding),
                    Scalar::from(output_padding),
                    Scalar::from(stride),
                    Scalar::from(dilation),
                ];
                gradient::record_operation_with_scalars(result.id(), op, vec![self.id(), weight.id()], scalars)?;
            }

            Ok(result)
        }
    }

    // Convolution gradient operations (internal use for backpropagation)
    pub(crate) fn conv1d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[u32],
        stride: u32,
        padding: u32,
        dilation: u32,
    ) -> HoduResult<Self> {
        let input_shape = self.shape();
        let grad_output_shape = grad_output.shape();
        let input_dims = input_shape.dims();
        let grad_output_dims = grad_output_shape.dims();

        // Input: [batch, in_channels, length]
        // GradOutput: [batch, out_channels, length_out]
        // Weight: [out_channels, in_channels, kernel_size]
        if input_dims.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::Conv1dGradWeight),
            });
        }
        if grad_output_dims.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::Conv1dGradWeight),
            });
        }
        if weight_shape.len() != 3 {
            return Err(HoduError::InternalError(
                "conv1d_grad_weight - weight_shape must be 3D".to_string(),
            ));
        }

        let stride_arr = [stride];
        let padding_arr = [padding];
        let dilation_arr = [dilation];

        let storage = self.with_storage(|input_storage| {
            grad_output.with_storage(|grad_output_storage| {
                input_storage.call_conv_grad_weight(
                    &self.layout(),
                    grad_output_storage,
                    &grad_output.layout(),
                    &Shape::from(weight_shape.to_vec()),
                    &stride_arr,
                    &padding_arr,
                    &dilation_arr,
                    Op::Conv(ConvOp::Conv1dGradWeight),
                )
            })
        })?;

        let result_layout = Layout::from_shape(&Shape::from(weight_shape.to_vec()));
        let result = from_storage(storage, result_layout, true, false);

        Ok(result)
    }

    pub(crate) fn conv2d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[u32],
        stride: u32,
        padding: u32,
        dilation: u32,
    ) -> HoduResult<Self> {
        let input_shape = self.shape();
        let grad_output_shape = grad_output.shape();
        let input_dims = input_shape.dims();
        let grad_output_dims = grad_output_shape.dims();

        // Input: [batch, in_channels, height, width]
        // GradOutput: [batch, out_channels, height_out, width_out]
        // Weight: [out_channels, in_channels, kernel_h, kernel_w]
        if input_dims.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::Conv2dGradWeight),
            });
        }
        if grad_output_dims.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::Conv2dGradWeight),
            });
        }
        if weight_shape.len() != 4 {
            return Err(HoduError::InternalError(
                "conv2d_grad_weight - weight_shape must be 4D".to_string(),
            ));
        }

        let stride_arr = [stride; 2];
        let padding_arr = [padding; 2];
        let dilation_arr = [dilation; 2];

        let storage = self.with_storage(|input_storage| {
            grad_output.with_storage(|grad_output_storage| {
                input_storage.call_conv_grad_weight(
                    &self.layout(),
                    grad_output_storage,
                    &grad_output.layout(),
                    &Shape::from(weight_shape.to_vec()),
                    &stride_arr,
                    &padding_arr,
                    &dilation_arr,
                    Op::Conv(ConvOp::Conv2dGradWeight),
                )
            })
        })?;

        let result_layout = Layout::from_shape(&Shape::from(weight_shape.to_vec()));
        let result = from_storage(storage, result_layout, true, false);

        Ok(result)
    }

    pub(crate) fn conv3d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[u32],
        stride: u32,
        padding: u32,
        dilation: u32,
    ) -> HoduResult<Self> {
        let input_shape = self.shape();
        let grad_output_shape = grad_output.shape();
        let input_dims = input_shape.dims();
        let grad_output_dims = grad_output_shape.dims();

        // Input: [batch, in_channels, depth, height, width]
        // GradOutput: [batch, out_channels, depth_out, height_out, width_out]
        // Weight: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
        if input_dims.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::Conv3dGradWeight),
            });
        }
        if grad_output_dims.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::Conv3dGradWeight),
            });
        }
        if weight_shape.len() != 5 {
            return Err(HoduError::InternalError(
                "conv3d_grad_weight - weight_shape must be 5D".to_string(),
            ));
        }

        let stride_arr = [stride; 3];
        let padding_arr = [padding; 3];
        let dilation_arr = [dilation; 3];

        let storage = self.with_storage(|input_storage| {
            grad_output.with_storage(|grad_output_storage| {
                input_storage.call_conv_grad_weight(
                    &self.layout(),
                    grad_output_storage,
                    &grad_output.layout(),
                    &Shape::from(weight_shape.to_vec()),
                    &stride_arr,
                    &padding_arr,
                    &dilation_arr,
                    Op::Conv(ConvOp::Conv3dGradWeight),
                )
            })
        })?;

        let result_layout = Layout::from_shape(&Shape::from(weight_shape.to_vec()));
        let result = from_storage(storage, result_layout, true, false);

        Ok(result)
    }

    pub(crate) fn conv_transpose1d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[u32],
        stride: u32,
        padding: u32,
        dilation: u32,
    ) -> HoduResult<Self> {
        let input_shape = self.shape();
        let grad_output_shape = grad_output.shape();
        let input_dims = input_shape.dims();
        let grad_output_dims = grad_output_shape.dims();

        // Input: [batch, in_channels, length_in]
        // GradOutput: [batch, out_channels, length_out]
        // Weight: [in_channels, out_channels, kernel_size]
        if input_dims.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::ConvTranspose1dGradWeight),
            });
        }
        if grad_output_dims.len() != 3 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::ConvTranspose1dGradWeight),
            });
        }
        if weight_shape.len() != 3 {
            return Err(HoduError::InternalError(
                "conv_transpose1d_grad_weight - weight_shape must be 3D".to_string(),
            ));
        }

        let stride_arr = [stride];
        let padding_arr = [padding];
        let dilation_arr = [dilation];

        let storage = self.with_storage(|input_storage| {
            grad_output.with_storage(|grad_output_storage| {
                input_storage.call_conv_grad_weight(
                    &self.layout(),
                    grad_output_storage,
                    &grad_output.layout(),
                    &Shape::from(weight_shape.to_vec()),
                    &stride_arr,
                    &padding_arr,
                    &dilation_arr,
                    Op::Conv(ConvOp::ConvTranspose1dGradWeight),
                )
            })
        })?;

        let result_layout = Layout::from_shape(&Shape::from(weight_shape.to_vec()));
        let result = from_storage(storage, result_layout, true, false);

        Ok(result)
    }

    pub(crate) fn conv_transpose2d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[u32],
        stride: u32,
        padding: u32,
        dilation: u32,
    ) -> HoduResult<Self> {
        let input_shape = self.shape();
        let grad_output_shape = grad_output.shape();
        let input_dims = input_shape.dims();
        let grad_output_dims = grad_output_shape.dims();

        // Input: [batch, in_channels, height_in, width_in]
        // GradOutput: [batch, out_channels, height_out, width_out]
        // Weight: [in_channels, out_channels, kernel_height, kernel_width]
        if input_dims.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::ConvTranspose2dGradWeight),
            });
        }
        if grad_output_dims.len() != 4 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::ConvTranspose2dGradWeight),
            });
        }
        if weight_shape.len() != 4 {
            return Err(HoduError::InternalError(
                "conv_transpose2d_grad_weight - weight_shape must be 4D".to_string(),
            ));
        }

        let stride_arr = [stride; 2];
        let padding_arr = [padding; 2];
        let dilation_arr = [dilation; 2];

        let storage = self.with_storage(|input_storage| {
            grad_output.with_storage(|grad_output_storage| {
                input_storage.call_conv_grad_weight(
                    &self.layout(),
                    grad_output_storage,
                    &grad_output.layout(),
                    &Shape::from(weight_shape.to_vec()),
                    &stride_arr,
                    &padding_arr,
                    &dilation_arr,
                    Op::Conv(ConvOp::ConvTranspose2dGradWeight),
                )
            })
        })?;

        let result_layout = Layout::from_shape(&Shape::from(weight_shape.to_vec()));
        let result = from_storage(storage, result_layout, true, false);

        Ok(result)
    }

    pub(crate) fn conv_transpose3d_grad_weight(
        &self,
        grad_output: &Self,
        weight_shape: &[u32],
        stride: u32,
        padding: u32,
        dilation: u32,
    ) -> HoduResult<Self> {
        let input_shape = self.shape();
        let grad_output_shape = grad_output.shape();
        let input_dims = input_shape.dims();
        let grad_output_dims = grad_output_shape.dims();

        // Input: [batch, in_channels, depth_in, height_in, width_in]
        // GradOutput: [batch, out_channels, depth_out, height_out, width_out]
        // Weight: [in_channels, out_channels, kernel_depth, kernel_height, kernel_width]
        if input_dims.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: input_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::ConvTranspose3dGradWeight),
            });
        }
        if grad_output_dims.len() != 5 {
            return Err(HoduError::IncompatibleShapes {
                lhs: grad_output_shape.clone(),
                rhs: Shape::from(vec![]),
                op: Op::Conv(ConvOp::ConvTranspose3dGradWeight),
            });
        }
        if weight_shape.len() != 5 {
            return Err(HoduError::InternalError(
                "conv_transpose3d_grad_weight - weight_shape must be 5D".to_string(),
            ));
        }

        let stride_arr = [stride; 3];
        let padding_arr = [padding; 3];
        let dilation_arr = [dilation; 3];

        let storage = self.with_storage(|input_storage| {
            grad_output.with_storage(|grad_output_storage| {
                input_storage.call_conv_grad_weight(
                    &self.layout(),
                    grad_output_storage,
                    &grad_output.layout(),
                    &Shape::from(weight_shape.to_vec()),
                    &stride_arr,
                    &padding_arr,
                    &dilation_arr,
                    Op::Conv(ConvOp::ConvTranspose3dGradWeight),
                )
            })
        })?;

        let result_layout = Layout::from_shape(&Shape::from(weight_shape.to_vec()));
        let result = from_storage(storage, result_layout, true, false);

        Ok(result)
    }
}
