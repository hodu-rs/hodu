use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::ReduceOp,
    scalar::Scalar,
    tensor::{tensor_from_id, TensorId},
};

impl VjpCompute for ReduceOp {
    fn compute_vjp_with_dims(
        &self,
        inputs: &[TensorId],
        output: TensorId,
        grad_output: TensorId,
        dims_scalars: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        let input = inputs[0];
        let input_tensor = tensor_from_id(input);
        let input_shape = input_tensor.shape();
        let dtype = input_tensor.dtype();

        // Convert dims_scalars to Vec<usize>
        let reduce_dims: Vec<usize> = if dims_scalars.is_empty() {
            // If no dims provided (fallback case), use all dimensions
            (0..input_shape.dims().len()).collect()
        } else {
            dims_scalars.iter().map(|scalar| scalar.to_usize()).collect()
        };

        match self {
            ReduceOp::Sum => {
                // d/dx sum(x) = 1 for all elements
                // Broadcast gradient back to input shape
                let grad_tensor = tensor_from_id(grad_output);
                let broadcasted = grad_tensor.broadcast(&input_shape)?;
                Ok(vec![broadcasted.id()])
            },
            ReduceOp::Mean => {
                // d/dx mean(x) = 1/N where N is number of elements in reduced dimensions
                let reduce_elements = reduce_dims
                    .iter()
                    .map(|&dim| input_shape.dims()[dim])
                    .product::<usize>() as f32;
                let scale_scalar = Scalar::from_f32(1.0 / reduce_elements, dtype);

                let grad_tensor = tensor_from_id(grad_output);
                let scaled_grad = grad_tensor.mul_scalar(scale_scalar)?;
                let broadcasted = scaled_grad.broadcast(&input_shape)?;
                Ok(vec![broadcasted.id()])
            },
            ReduceOp::Prod => {
                // d/dx prod(x) = prod(x) / x_i for each element x_i
                let output_tensor = tensor_from_id(output);
                let broadcasted_output = output_tensor.broadcast(&input_shape)?;
                let derivative = broadcasted_output.div(&input_tensor)?;
                let grad_tensor = tensor_from_id(grad_output);
                let broadcasted_grad = grad_tensor.broadcast(&input_shape)?;
                let result = derivative.mul(&broadcasted_grad)?;
                Ok(vec![result.id()])
            },
            ReduceOp::Norm => {
                // d/dx ||x||_2 = x / ||x||_2
                let output_tensor = tensor_from_id(output);
                let broadcasted_output = output_tensor.broadcast(&input_shape)?;
                let derivative = input_tensor.div(&broadcasted_output)?;
                let grad_tensor = tensor_from_id(grad_output);
                let broadcasted_grad = grad_tensor.broadcast(&input_shape)?;
                let result = derivative.mul(&broadcasted_grad)?;
                Ok(vec![result.id()])
            },
            ReduceOp::Var => {
                // d/dx var(x) = 2 * (x - mean(x)) / N where N is number of elements in reduced dimensions
                let mean_tensor = input_tensor.mean(&reduce_dims, false)?;
                let broadcasted_mean = mean_tensor.broadcast(&input_shape)?;
                let diff = input_tensor.sub(&broadcasted_mean)?;
                let reduce_elements = reduce_dims
                    .iter()
                    .map(|&dim| input_shape.dims()[dim])
                    .product::<usize>() as f32;
                let scale = Scalar::from_f32(2.0 / reduce_elements, dtype);
                let derivative = diff.mul_scalar(scale)?;
                let grad_tensor = tensor_from_id(grad_output);
                let broadcasted_grad = grad_tensor.broadcast(&input_shape)?;
                let result = derivative.mul(&broadcasted_grad)?;
                Ok(vec![result.id()])
            },
            ReduceOp::Std => {
                // d/dx std(x) = (x - mean(x)) / (N * std(x)) where N is number of elements in reduced dimensions
                let mean_tensor = input_tensor.mean(&reduce_dims, false)?;
                let broadcasted_mean = mean_tensor.broadcast(&input_shape)?;
                let diff = input_tensor.sub(&broadcasted_mean)?;
                let output_tensor = tensor_from_id(output);
                let broadcasted_std = output_tensor.broadcast(&input_shape)?;
                let reduce_elements = reduce_dims
                    .iter()
                    .map(|&dim| input_shape.dims()[dim])
                    .product::<usize>() as f32;
                let scale = Scalar::from_f32(1.0 / reduce_elements, dtype);
                let derivative = diff.div(&broadcasted_std)?.mul_scalar(scale)?;
                let grad_tensor = tensor_from_id(grad_output);
                let broadcasted_grad = grad_tensor.broadcast(&input_shape)?;
                let result = derivative.mul(&broadcasted_grad)?;
                Ok(vec![result.id()])
            },
            _ => Err(HoduError::GradientComputationFailed(format!(
                "{:?} is not differentiable - cannot compute gradients for discrete index operations",
                self
            ))),
        }
    }
}
