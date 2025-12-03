use crate::module::Module;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor, types::DType};

#[derive(Module, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, with_bias: bool, dtype: DType) -> HoduResult<Self> {
        let k: f32 = 1.0 / (in_features as f32).sqrt();

        let zero = Scalar::zero(dtype);
        let one = Scalar::one(dtype);
        let k_scalar = Scalar::from_f32(k, dtype);

        // weight: [out_features, in_features]
        let weight = Tensor::randn([out_features, in_features], zero, one)?;
        weight.set_requires_grad(true)?;
        // Scale by k (Xavier/Glorot initialization)
        let weight = weight.mul_scalar(k_scalar)?;

        // bias
        let bias = if with_bias {
            let bias = Tensor::randn([out_features], zero, one)?;
            bias.set_requires_grad(true)?;
            // Scale by k
            let bias = bias.mul_scalar(k_scalar)?;
            Some(bias)
        } else {
            None
        };

        Ok(Self { weight, bias })
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // Linear transformation: input @ weight.T + bias
        let output = input.matmul(&self.weight.transpose(-2, -1)?)?;

        if let Some(ref bias) = self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
}
