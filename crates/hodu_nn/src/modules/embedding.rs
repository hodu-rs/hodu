use crate::compat::*;
use crate::module::Module;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor, types::DType};

#[derive(Module, Clone)]
pub struct Embedding {
    num_embeddings: usize,
    embedding_dim: usize,
    weight: Tensor,
    padding_idx: Option<usize>,
    max_norm: Option<Scalar>,
    norm_type: Scalar,
}

impl Embedding {
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        max_norm: Option<impl Into<Scalar>>,
        norm_type: impl Into<Scalar>,
        dtype: DType,
    ) -> HoduResult<Self> {
        let norm_type = norm_type.into();
        let max_norm = max_norm.map(|v| v.into());

        // Xavier/Glorot initialization
        let k: f32 = 1.0 / (embedding_dim as f32).sqrt();
        let k_scalar = Scalar::from_f32(k, dtype);

        let zero = Scalar::zero(dtype);
        let one = Scalar::one(dtype);

        // weight: [num_embeddings, embedding_dim]
        let weight = Tensor::randn([num_embeddings, embedding_dim], zero, one)?;
        weight.set_requires_grad(true)?;
        // Scale by k
        let weight = weight.mul_scalar(k_scalar)?;

        // If padding_idx is specified, initialize that embedding to zeros
        let weight = if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(hodu_core::error::HoduError::InternalError(format!(
                    "padding_idx {} is out of range for num_embeddings {}",
                    idx, num_embeddings
                )));
            }
            // Set weight[idx] to zeros using index_put
            let indices = Tensor::new(vec![idx as i32])?.reshape([1])?;
            let zeros = Tensor::zeros([1, embedding_dim], dtype)?;
            weight.index_put(0, &indices, &zeros)?
        } else {
            weight
        };

        Ok(Self {
            num_embeddings,
            embedding_dim,
            weight,
            padding_idx,
            max_norm,
            norm_type,
        })
    }

    pub fn from_pretrained(weight: Tensor, freeze: bool) -> HoduResult<Self> {
        let weight_shape = weight.shape();

        if weight_shape.ndim() != 2 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "Embedding weight must be 2D [num_embeddings, embedding_dim], got {}D",
                weight_shape.ndim()
            )));
        }

        let num_embeddings = weight_shape[0];
        let embedding_dim = weight_shape[1];

        weight.set_requires_grad(!freeze)?;

        Ok(Self {
            num_embeddings,
            embedding_dim,
            weight,
            padding_idx: None,
            max_norm: None,
            norm_type: Scalar::F32(2.0),
        })
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // input: indices of shape [batch_size] or [batch_size, seq_len] or any shape
        // output: embeddings of shape [..., embedding_dim]

        let input_shape = input.shape();

        // Flatten input to 1D for easier processing
        let flat_input = input.flatten()?;

        // Use index_select on the weight tensor (same as gather on axis 0)
        // axis 0 is the num_embeddings dimension
        // This will fail if any index >= num_embeddings
        let flat_output = self.weight.index_select(0, &flat_input)?;

        // Apply max_norm constraint if specified
        let flat_output = if let Some(max_norm) = self.max_norm {
            self.apply_max_norm(&flat_output, max_norm)?
        } else {
            flat_output
        };

        // If padding_idx is specified, zero out gradients for padding embeddings
        let flat_output = if let Some(padding_idx) = self.padding_idx {
            self.handle_padding_idx(&flat_output, &flat_input, padding_idx)?
        } else {
            flat_output
        };

        // Output shape: [...original_input_shape..., embedding_dim]
        let mut output_shape = input_shape.to_vec();
        output_shape.push(self.embedding_dim);

        // Reshape to the desired output shape
        flat_output.reshape(&output_shape)
    }

    fn apply_max_norm(&self, embeddings: &Tensor, max_norm: Scalar) -> HoduResult<Tensor> {
        // Compute L2 norm along the embedding dimension (last dimension)
        // embeddings shape: [num_indices, embedding_dim]
        let embedding_dim = embeddings.shape().last().unwrap();

        // squared_sum = sum(x^2, dim=-1, keepdim=true)
        let squared = embeddings.mul(embeddings)?;
        let squared_sum = squared.sum(&[embedding_dim - 1], true)?;

        // norm = sqrt(squared_sum)
        let norm = squared_sum.sqrt()?;

        // scale = min(max_norm / norm, 1.0)
        let max_norm_tensor = Tensor::full(norm.shape(), max_norm)?;
        let scale = max_norm_tensor.div(&norm)?;

        // Clamp scale to maximum of 1.0
        let one = Tensor::full(scale.shape(), Scalar::one(embeddings.dtype()))?;
        let scale = scale.minimum(&one)?;

        // Apply scaling: embeddings * scale
        embeddings.mul(&scale)
    }

    fn handle_padding_idx(&self, embeddings: &Tensor, indices: &Tensor, padding_idx: usize) -> HoduResult<Tensor> {
        // Create a mask where padding indices are marked
        // mask = (indices == padding_idx)
        let padding_idx_scalar = Scalar::from_f32(padding_idx as f32, indices.dtype());
        let padding_idx_tensor = Tensor::full(indices.shape(), padding_idx_scalar)?;

        // mask: [num_indices], true where index == padding_idx
        let mask = indices.eq(&padding_idx_tensor)?;

        // Expand mask to match embeddings shape: [num_indices, 1]
        let mask = mask.unsqueeze(-1)?;

        // Zero out embeddings where mask is true
        // result = embeddings * (1 - mask)
        let one = Tensor::ones(mask.shape(), embeddings.dtype())?;
        let zero = Tensor::zeros(mask.shape(), embeddings.dtype())?;

        // Convert bool mask to float: where(mask, 0.0, 1.0)
        let float_mask = mask.where3(&zero, &one)?;

        // Apply mask
        embeddings.mul(&float_mask)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn padding_idx(&self) -> Option<usize> {
        self.padding_idx
    }

    pub fn max_norm(&self) -> Option<f32> {
        self.max_norm.as_ref().map(|s| s.to_f32())
    }

    pub fn norm_type(&self) -> f32 {
        self.norm_type.to_f32()
    }
}
