//! Attention mechanisms for transformer architectures.

use crate::module::Module;
use crate::state::{get_state, State};
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor, types::DType};

// ============================================================================
// ScaledDotProductAttention
// ============================================================================

/// Scaled Dot-Product Attention.
///
/// Computes attention as: softmax(QK^T / sqrt(d_k))V
///
/// # Arguments
/// * `query` - Query tensor of shape [..., seq_len, d_k]
/// * `key` - Key tensor of shape [..., seq_len, d_k]
/// * `value` - Value tensor of shape [..., seq_len, d_v]
/// * `attn_mask` - Optional attention mask (1 = attend, 0 = mask)
/// * `dropout_p` - Dropout probability (only applied during training)
/// * `is_causal` - If true, applies causal mask (upper triangular)
///
/// # Returns
/// * `output` - Attention output of shape [..., seq_len, d_v]
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_mask: Option<&Tensor>,
    dropout_p: f64,
    is_causal: bool,
) -> HoduResult<Tensor> {
    let d_k = query.shape().last().unwrap_or(1) as f64;
    let scale = 1.0 / d_k.sqrt();

    // QK^T / sqrt(d_k)
    let attn_weights = query.matmul(&key.transpose(-2, -1)?)?.mul_scalar(scale)?;

    // Apply causal mask if needed
    let attn_weights = if is_causal {
        let seq_len_q = attn_weights.shape()[attn_weights.ndim() - 2];
        let seq_len_k = attn_weights.shape()[attn_weights.ndim() - 1];
        let causal_mask = create_causal_mask(seq_len_q, seq_len_k)?;
        let neg_inf = get_neg_inf(query.dtype());
        attn_weights.masked_fill(&causal_mask, neg_inf)?
    } else {
        attn_weights
    };

    // Apply attention mask if provided (0 = mask out, 1 = attend)
    let attn_weights = if let Some(mask) = attn_mask {
        let neg_inf = get_neg_inf(query.dtype());
        attn_weights.masked_fill(&mask.eq_scalar(0.0)?, neg_inf)?
    } else {
        attn_weights
    };

    // Softmax
    let attn_weights = attn_weights.softmax(-1)?;

    // Apply dropout during training
    let attn_weights = if dropout_p > 0.0 && get_state() == State::Training {
        apply_dropout(&attn_weights, dropout_p)?
    } else {
        attn_weights
    };

    // Attention output
    attn_weights.matmul(value)
}

/// Creates a causal mask (upper triangular = true for masking)
fn create_causal_mask(seq_len_q: usize, seq_len_k: usize) -> HoduResult<Tensor> {
    // Create row indices [0, 1, 2, ..., seq_len_q-1] as column vector
    let row_indices = Tensor::arange(0i32, seq_len_q as i32, 1i32)?.reshape([seq_len_q, 1])?;
    // Create col indices [0, 1, 2, ..., seq_len_k-1] as row vector
    let col_indices = Tensor::arange(0i32, seq_len_k as i32, 1i32)?.reshape([1, seq_len_k])?;
    // Broadcast and compare: mask[i,j] = true if j > i (upper triangular excluding diagonal)
    let row_broadcast = row_indices.broadcast([seq_len_q, seq_len_k])?;
    let col_broadcast = col_indices.broadcast([seq_len_q, seq_len_k])?;
    col_broadcast.gt(&row_broadcast)
}

/// Get negative infinity for masking based on dtype
fn get_neg_inf(dtype: DType) -> f64 {
    match dtype {
        DType::F32 => f32::NEG_INFINITY as f64,
        #[cfg(feature = "f64")]
        DType::F64 => f64::NEG_INFINITY,
        DType::F16 | DType::BF16 => -65504.0,
        _ => -1e9,
    }
}

/// Apply dropout to tensor
fn apply_dropout(x: &Tensor, p: f64) -> HoduResult<Tensor> {
    let random = Tensor::rand_uniform_like(x, 0.0, 1.0)?;
    let mask = random.gt_scalar(p)?.to_dtype(x.dtype())?;
    let scale = 1.0 / (1.0 - p);
    let scaled_mask = mask.mul_scalar(scale)?;
    x.mul(&scaled_mask)
}

// ============================================================================
// MultiheadAttention
// ============================================================================

/// Multi-Head Attention mechanism.
///
/// Allows the model to jointly attend to information from different representation
/// subspaces at different positions.
///
/// MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
/// where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
#[derive(Module, Clone)]
pub struct MultiheadAttention {
    /// Query projection weight [embed_dim, embed_dim]
    w_q: Tensor,
    /// Key projection weight [embed_dim, kdim]
    w_k: Tensor,
    /// Value projection weight [embed_dim, vdim]
    w_v: Tensor,
    /// Output projection weight [embed_dim, embed_dim]
    w_o: Tensor,
    /// Query projection bias
    b_q: Option<Tensor>,
    /// Key projection bias
    b_k: Option<Tensor>,
    /// Value projection bias
    b_v: Option<Tensor>,
    /// Output projection bias
    b_o: Option<Tensor>,
    /// Embedding dimension
    embed_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension of each head
    head_dim: usize,
    /// Dropout probability
    dropout_p: f64,
}

impl MultiheadAttention {
    /// Creates a new MultiheadAttention module.
    ///
    /// # Arguments
    /// * `embed_dim` - Total dimension of the model
    /// * `num_heads` - Number of parallel attention heads
    /// * `dropout` - Dropout probability on attention weights
    /// * `with_bias` - If true, adds bias to input/output projections
    /// * `kdim` - Total number of features for keys (defaults to embed_dim)
    /// * `vdim` - Total number of features for values (defaults to embed_dim)
    /// * `dtype` - Data type for parameters
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        dropout: f64,
        with_bias: bool,
        kdim: Option<usize>,
        vdim: Option<usize>,
        dtype: DType,
    ) -> HoduResult<Self> {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim ({}) must be divisible by num_heads ({})",
            embed_dim,
            num_heads
        );

        let kdim = kdim.unwrap_or(embed_dim);
        let vdim = vdim.unwrap_or(embed_dim);
        let head_dim = embed_dim / num_heads;

        // Xavier uniform initialization
        let k: f32 = 1.0 / (embed_dim as f32).sqrt();
        let zero = Scalar::zero(dtype);
        let one = Scalar::one(dtype);
        let k_scalar = Scalar::from_f32(k, dtype);

        // Input projections
        let w_q = Tensor::randn([embed_dim, embed_dim], zero, one)?;
        w_q.set_requires_grad(true)?;
        let w_q = w_q.mul_scalar(k_scalar)?;

        let w_k = Tensor::randn([embed_dim, kdim], zero, one)?;
        w_k.set_requires_grad(true)?;
        let w_k = w_k.mul_scalar(k_scalar)?;

        let w_v = Tensor::randn([embed_dim, vdim], zero, one)?;
        w_v.set_requires_grad(true)?;
        let w_v = w_v.mul_scalar(k_scalar)?;

        // Output projection
        let w_o = Tensor::randn([embed_dim, embed_dim], zero, one)?;
        w_o.set_requires_grad(true)?;
        let w_o = w_o.mul_scalar(k_scalar)?;

        let (b_q, b_k, b_v, b_o) = if with_bias {
            let b_q = Tensor::randn([embed_dim], zero, one)?;
            b_q.set_requires_grad(true)?;
            let b_q = b_q.mul_scalar(k_scalar)?;

            let b_k = Tensor::randn([embed_dim], zero, one)?;
            b_k.set_requires_grad(true)?;
            let b_k = b_k.mul_scalar(k_scalar)?;

            let b_v = Tensor::randn([embed_dim], zero, one)?;
            b_v.set_requires_grad(true)?;
            let b_v = b_v.mul_scalar(k_scalar)?;

            let b_o = Tensor::randn([embed_dim], zero, one)?;
            b_o.set_requires_grad(true)?;
            let b_o = b_o.mul_scalar(k_scalar)?;

            (Some(b_q), Some(b_k), Some(b_v), Some(b_o))
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            w_q,
            w_k,
            w_v,
            w_o,
            b_q,
            b_k,
            b_v,
            b_o,
            embed_dim,
            num_heads,
            head_dim,
            dropout_p: dropout,
        })
    }

    /// Forward pass with separate query, key, value inputs.
    ///
    /// # Arguments
    /// * `query` - Query tensor of shape [batch, seq_len, embed_dim]
    /// * `key` - Key tensor of shape [batch, src_len, kdim]
    /// * `value` - Value tensor of shape [batch, src_len, vdim]
    /// * `attn_mask` - Optional attention mask (1 = attend, 0 = mask)
    /// * `is_causal` - If true, applies causal mask
    ///
    /// # Returns
    /// * `output` - Attention output of shape [batch, seq_len, embed_dim]
    pub fn forward_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        is_causal: bool,
    ) -> HoduResult<Tensor> {
        let shape = query.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let src_len = key.shape()[1];

        // Project Q, K, V
        let q = self.linear(query, &self.w_q, self.b_q.as_ref())?;
        let k = self.linear(key, &self.w_k, self.b_k.as_ref())?;
        let v = self.linear(value, &self.w_v, self.b_v.as_ref())?;

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([batch_size, src_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape([batch_size, src_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let attn_output = scaled_dot_product_attention(&q, &k, &v, attn_mask, self.dropout_p, is_causal)?;

        // Reshape back to [batch, seq_len, embed_dim]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape([batch_size, seq_len, self.embed_dim])?;

        // Output projection
        self.linear(&attn_output, &self.w_o, self.b_o.as_ref())
    }

    /// Self-attention forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq_len, embed_dim]
    /// * `attn_mask` - Optional attention mask
    /// * `is_causal` - If true, applies causal mask
    pub fn forward_self_attn(&self, x: &Tensor, attn_mask: Option<&Tensor>, is_causal: bool) -> HoduResult<Tensor> {
        self.forward_qkv(x, x, x, attn_mask, is_causal)
    }

    fn linear(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> HoduResult<Tensor> {
        // x: [batch, seq, in_features]
        // weight: [out_features, in_features]
        // output: [batch, seq, out_features]
        let out = x.matmul(&weight.transpose(0, 1)?)?;
        if let Some(b) = bias {
            out.add(b)
        } else {
            Ok(out)
        }
    }

    /// Returns the embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Returns the number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Returns the dimension per head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // Default forward is self-attention without mask
        self.forward_self_attn(input, None, false)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.w_q, &self.w_k, &self.w_v, &self.w_o];
        if let Some(ref b) = self.b_q {
            params.push(b);
        }
        if let Some(ref b) = self.b_k {
            params.push(b);
        }
        if let Some(ref b) = self.b_v {
            params.push(b);
        }
        if let Some(ref b) = self.b_o {
            params.push(b);
        }
        params
    }
}
