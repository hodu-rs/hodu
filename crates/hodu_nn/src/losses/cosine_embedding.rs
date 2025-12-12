use crate::module::Module;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

/// Cosine Embedding Loss.
///
/// Measures the loss given input tensors x1, x2 and a label tensor y
/// containing values 1 or -1.
///
/// loss(x1, x2, y) = 1 - cos(x1, x2),              if y = 1
///                 = max(0, cos(x1, x2) - margin), if y = -1
///
/// where cos(x1, x2) = (x1 . x2) / (||x1|| * ||x2||)
///
/// This is used for measuring whether two inputs are similar or dissimilar,
/// using the cosine similarity.
#[derive(Module, Clone)]
#[module(inputs = 3)]
pub struct CosineEmbeddingLoss {
    margin: Scalar,
}

impl Default for CosineEmbeddingLoss {
    fn default() -> Self {
        Self {
            margin: Scalar::F32(0.0),
        }
    }
}

impl CosineEmbeddingLoss {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_margin(margin: impl Into<Scalar>) -> Self {
        Self { margin: margin.into() }
    }

    fn forward(&self, (x1, x2, y): (&Tensor, &Tensor, &Tensor)) -> HoduResult<Tensor> {
        let dtype = x1.dtype();

        // Compute cosine similarity along last dimension
        // cos(x1, x2) = (x1 . x2) / (||x1|| * ||x2||)
        let dot_product = x1.mul(x2)?.sum(&[-1], false)?;
        let norm1 = x1.square()?.sum(&[-1], false)?.sqrt()?;
        let norm2 = x2.square()?.sum(&[-1], false)?.sqrt()?;

        // Add small epsilon to avoid division by zero
        let eps = Scalar::from_f32(1e-8, dtype);
        let norm_product = norm1.mul(&norm2)?.add_scalar(eps)?;
        let cos_sim = dot_product.div(&norm_product)?;

        // For y = 1: loss = 1 - cos_sim
        // For y = -1: loss = max(0, cos_sim - margin)
        let one = Scalar::one(dtype);
        let zero = Scalar::zero(dtype);
        let margin = self.margin.to_dtype(dtype);

        // loss_positive = 1 - cos_sim
        let loss_positive = cos_sim.mul_scalar(Scalar::from_f32(-1.0, dtype))?.add_scalar(one)?;

        // loss_negative = max(0, cos_sim - margin)
        let loss_negative = cos_sim.sub_scalar(margin)?.maximum_scalar(zero)?;

        // y = 1 -> use loss_positive, y = -1 -> use loss_negative
        // mask = (y == 1)
        let mask = y.eq_scalar(one)?;

        let loss = mask
            .mul(&loss_positive)?
            .add(&mask.logical_not()?.mul(&loss_negative)?)?;

        loss.mean_all()
    }
}
