use crate::module::Module;
use crate::state::{get_state, State};
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor, types::DType};
use std::cell::RefCell;

#[derive(Module, Clone)]
pub struct BatchNorm1D {
    num_features: usize,
    eps: Scalar,
    momentum: Scalar,
    affine: bool,
    gamma: Option<Tensor>,
    beta: Option<Tensor>,
    running_mean: RefCell<Tensor>,
    running_var: RefCell<Tensor>,
}

impl BatchNorm1D {
    pub fn new(
        num_features: usize,
        eps: impl Into<Scalar>,
        momentum: impl Into<Scalar>,
        affine: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        let eps = eps.into();
        let momentum = momentum.into();

        let (gamma, beta) = if affine {
            let gamma = Tensor::ones([num_features], dtype)?;
            gamma.set_requires_grad(true)?;

            let beta = Tensor::zeros([num_features], dtype)?;
            beta.set_requires_grad(true)?;

            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        let running_mean = RefCell::new(Tensor::zeros([num_features], dtype)?);
        let running_var = RefCell::new(Tensor::ones([num_features], dtype)?);

        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            gamma,
            beta,
            running_mean,
            running_var,
        })
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();

        if get_state() == State::Training {
            // Training mode: compute batch statistics
            let mean = if input_shape.ndim() == 2 {
                // [N, C] -> mean over N (axis 0)
                input.mean(&[0], true)?
            } else if input_shape.ndim() == 3 {
                // [N, C, L] -> mean over N, L (axes 0, 2)
                input.mean(&[0, 2], true)?
            } else {
                return Err(hodu_core::error::HoduError::InternalError(format!(
                    "BatchNorm1d expects 2D [N, C] or 3D [N, C, L] input, got {}D",
                    input_shape.ndim()
                )));
            };

            let var = if input_shape.ndim() == 2 {
                let centered = input.sub(&mean)?;
                let squared = centered.mul(&centered)?;
                squared.mean(&[0], true)?
            } else {
                let centered = input.sub(&mean)?;
                let squared = centered.mul(&centered)?;
                squared.mean(&[0, 2], true)?
            };

            // Update running statistics (during training)
            // running_mean = momentum * running_mean + (1 - momentum) * mean
            // running_var = momentum * running_var + (1 - momentum) * var
            {
                let mut rm = self.running_mean.borrow_mut();
                let mut rv = self.running_var.borrow_mut();

                let momentum_typed = self.momentum.to_dtype(rm.dtype());
                let one_minus_momentum = Scalar::from_f32(1.0, rm.dtype()) - momentum_typed;

                // Squeeze mean and var to match running stats shape [C]
                let mean_squeezed = mean.reshape([self.num_features])?;
                let var_squeezed = var.reshape([self.num_features])?;

                let new_mean = rm
                    .mul_scalar(momentum_typed)?
                    .add(&mean_squeezed.mul_scalar(one_minus_momentum)?)?;
                let new_var = rv
                    .mul_scalar(momentum_typed)?
                    .add(&var_squeezed.mul_scalar(one_minus_momentum)?)?;

                *rm = new_mean;
                *rv = new_var;
            }

            // Normalize
            let normalized = self.normalize(input, &mean, &var)?;

            Ok(normalized)
        } else {
            // Evaluation mode: use running statistics
            let mean = self.running_mean.borrow();
            let var = self.running_var.borrow();

            // Reshape for broadcasting
            let mean = if input_shape.ndim() == 2 {
                mean.reshape([1, self.num_features])?
            } else {
                mean.reshape([1, self.num_features, 1])?
            };

            let var = if input_shape.ndim() == 2 {
                var.reshape([1, self.num_features])?
            } else {
                var.reshape([1, self.num_features, 1])?
            };

            let normalized = self.normalize(input, &mean, &var)?;

            Ok(normalized)
        }
    }

    fn normalize(&self, input: &Tensor, mean: &Tensor, var: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();

        // x_normalized = (x - mean) / sqrt(var + eps)
        let centered = input.sub(mean)?;
        let eps_typed = self.eps.to_dtype(input.dtype());
        let std = var.add_scalar(eps_typed)?.sqrt()?;
        let normalized = centered.div(&std)?;

        // Apply affine transformation if enabled
        if self.affine {
            let gamma = self.gamma.as_ref().unwrap();
            let beta = self.beta.as_ref().unwrap();

            let gamma_reshaped = if input_shape.ndim() == 2 {
                gamma.reshape([1, self.num_features])?
            } else {
                gamma.reshape([1, self.num_features, 1])?
            };

            let beta_reshaped = if input_shape.ndim() == 2 {
                beta.reshape([1, self.num_features])?
            } else {
                beta.reshape([1, self.num_features, 1])?
            };

            let scaled = normalized.mul(&gamma_reshaped)?;
            scaled.add(&beta_reshaped)
        } else {
            Ok(normalized)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if self.affine {
            if let Some(ref gamma) = self.gamma {
                params.push(gamma);
            }
            if let Some(ref beta) = self.beta {
                params.push(beta);
            }
        }
        params
    }
}

#[derive(Module, Clone)]
pub struct BatchNorm2D {
    num_features: usize,
    eps: Scalar,
    momentum: Scalar,
    affine: bool,
    gamma: Option<Tensor>,
    beta: Option<Tensor>,
    running_mean: RefCell<Tensor>,
    running_var: RefCell<Tensor>,
}

impl BatchNorm2D {
    pub fn new(
        num_features: usize,
        eps: impl Into<Scalar>,
        momentum: impl Into<Scalar>,
        affine: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        let eps = eps.into();
        let momentum = momentum.into();

        let (gamma, beta) = if affine {
            let gamma = Tensor::ones([num_features], dtype)?;
            gamma.set_requires_grad(true)?;

            let beta = Tensor::zeros([num_features], dtype)?;
            beta.set_requires_grad(true)?;

            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        let running_mean = RefCell::new(Tensor::zeros([num_features], dtype)?);
        let running_var = RefCell::new(Tensor::ones([num_features], dtype)?);

        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            gamma,
            beta,
            running_mean,
            running_var,
        })
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();

        if input_shape.ndim() != 4 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "BatchNorm2d expects 4D input [N, C, H, W], got {}D",
                input_shape.ndim()
            )));
        }

        if get_state() == State::Training {
            // Training mode: compute batch statistics
            // Mean over [N, H, W] (axes 0, 2, 3)
            let mean = input.mean(&[0, 2, 3], true)?;
            let centered = input.sub(&mean)?;
            let squared = centered.mul(&centered)?;
            let var = squared.mean(&[0, 2, 3], true)?;

            // Update running statistics (during training)
            {
                let mut rm = self.running_mean.borrow_mut();
                let mut rv = self.running_var.borrow_mut();

                let momentum_typed = self.momentum.to_dtype(rm.dtype());
                let one_minus_momentum = Scalar::from_f32(1.0, rm.dtype()) - momentum_typed;

                // Squeeze mean and var to match running stats shape [C]
                let mean_squeezed = mean.reshape([self.num_features])?;
                let var_squeezed = var.reshape([self.num_features])?;

                let new_mean = rm
                    .mul_scalar(momentum_typed)?
                    .add(&mean_squeezed.mul_scalar(one_minus_momentum)?)?;
                let new_var = rv
                    .mul_scalar(momentum_typed)?
                    .add(&var_squeezed.mul_scalar(one_minus_momentum)?)?;

                *rm = new_mean;
                *rv = new_var;
            }

            let normalized = self.normalize(input, &mean, &var)?;
            Ok(normalized)
        } else {
            // Evaluation mode: use running statistics
            let mean = self.running_mean.borrow();
            let var = self.running_var.borrow();

            let mean = mean.reshape([1, self.num_features, 1, 1])?;
            let var = var.reshape([1, self.num_features, 1, 1])?;

            let normalized = self.normalize(input, &mean, &var)?;
            Ok(normalized)
        }
    }

    fn normalize(&self, input: &Tensor, mean: &Tensor, var: &Tensor) -> HoduResult<Tensor> {
        // x_normalized = (x - mean) / sqrt(var + eps)
        let centered = input.sub(mean)?;
        let eps_typed = self.eps.to_dtype(input.dtype());
        let std = var.add_scalar(eps_typed)?.sqrt()?;
        let normalized = centered.div(&std)?;

        // Apply affine transformation if enabled
        if self.affine {
            let gamma = self.gamma.as_ref().unwrap();
            let beta = self.beta.as_ref().unwrap();

            let gamma_reshaped = gamma.reshape([1, self.num_features, 1, 1])?;
            let beta_reshaped = beta.reshape([1, self.num_features, 1, 1])?;

            let scaled = normalized.mul(&gamma_reshaped)?;
            scaled.add(&beta_reshaped)
        } else {
            Ok(normalized)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if self.affine {
            if let Some(ref gamma) = self.gamma {
                params.push(gamma);
            }
            if let Some(ref beta) = self.beta {
                params.push(beta);
            }
        }
        params
    }
}

#[derive(Module, Clone)]
pub struct BatchNorm3D {
    num_features: usize,
    eps: Scalar,
    momentum: Scalar,
    affine: bool,
    gamma: Option<Tensor>,
    beta: Option<Tensor>,
    running_mean: RefCell<Tensor>,
    running_var: RefCell<Tensor>,
}

impl BatchNorm3D {
    pub fn new(
        num_features: usize,
        eps: impl Into<Scalar>,
        momentum: impl Into<Scalar>,
        affine: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        let eps = eps.into();
        let momentum = momentum.into();

        let (gamma, beta) = if affine {
            let gamma = Tensor::ones([num_features], dtype)?;
            gamma.set_requires_grad(true)?;

            let beta = Tensor::zeros([num_features], dtype)?;
            beta.set_requires_grad(true)?;

            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        let running_mean = RefCell::new(Tensor::zeros([num_features], dtype)?);
        let running_var = RefCell::new(Tensor::ones([num_features], dtype)?);

        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            gamma,
            beta,
            running_mean,
            running_var,
        })
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();

        if input_shape.ndim() != 5 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "BatchNorm3D expects 5D input [N, C, D, H, W], got {}D",
                input_shape.ndim()
            )));
        }

        if get_state() == State::Training {
            // Training mode: compute batch statistics
            // Mean over [N, D, H, W] (axes 0, 2, 3, 4)
            let mean = input.mean(&[0, 2, 3, 4], true)?;
            let centered = input.sub(&mean)?;
            let squared = centered.mul(&centered)?;
            let var = squared.mean(&[0, 2, 3, 4], true)?;

            // Update running statistics (during training)
            {
                let mut rm = self.running_mean.borrow_mut();
                let mut rv = self.running_var.borrow_mut();

                let momentum_typed = self.momentum.to_dtype(rm.dtype());
                let one_minus_momentum = Scalar::from_f32(1.0, rm.dtype()) - momentum_typed;

                // Squeeze mean and var to match running stats shape [C]
                let mean_squeezed = mean.reshape([self.num_features])?;
                let var_squeezed = var.reshape([self.num_features])?;

                let new_mean = rm
                    .mul_scalar(momentum_typed)?
                    .add(&mean_squeezed.mul_scalar(one_minus_momentum)?)?;
                let new_var = rv
                    .mul_scalar(momentum_typed)?
                    .add(&var_squeezed.mul_scalar(one_minus_momentum)?)?;

                *rm = new_mean;
                *rv = new_var;
            }

            let normalized = self.normalize(input, &mean, &var)?;
            Ok(normalized)
        } else {
            // Evaluation mode: use running statistics
            let mean = self.running_mean.borrow();
            let var = self.running_var.borrow();

            let mean = mean.reshape([1, self.num_features, 1, 1, 1])?;
            let var = var.reshape([1, self.num_features, 1, 1, 1])?;

            let normalized = self.normalize(input, &mean, &var)?;
            Ok(normalized)
        }
    }

    fn normalize(&self, input: &Tensor, mean: &Tensor, var: &Tensor) -> HoduResult<Tensor> {
        // x_normalized = (x - mean) / sqrt(var + eps)
        let centered = input.sub(mean)?;
        let eps_typed = self.eps.to_dtype(input.dtype());
        let std = var.add_scalar(eps_typed)?.sqrt()?;
        let normalized = centered.div(&std)?;

        // Apply affine transformation if enabled
        if self.affine {
            let gamma = self.gamma.as_ref().unwrap();
            let beta = self.beta.as_ref().unwrap();

            let gamma_reshaped = gamma.reshape([1, self.num_features, 1, 1, 1])?;
            let beta_reshaped = beta.reshape([1, self.num_features, 1, 1, 1])?;

            let scaled = normalized.mul(&gamma_reshaped)?;
            scaled.add(&beta_reshaped)
        } else {
            Ok(normalized)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if self.affine {
            if let Some(ref gamma) = self.gamma {
                params.push(gamma);
            }
            if let Some(ref beta) = self.beta {
                params.push(beta);
            }
        }
        params
    }
}

#[derive(Module, Clone)]
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    eps: Scalar,
    elementwise_affine: bool,
    gamma: Option<Tensor>,
    beta: Option<Tensor>,
}

impl LayerNorm {
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: impl Into<Scalar>,
        elementwise_affine: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        let eps = eps.into();

        let (gamma, beta) = if elementwise_affine {
            let gamma = Tensor::ones(&normalized_shape, dtype)?;
            gamma.set_requires_grad(true)?;

            let beta = Tensor::zeros(&normalized_shape, dtype)?;
            beta.set_requires_grad(true)?;

            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        Ok(Self {
            normalized_shape,
            eps,
            elementwise_affine,
            gamma,
            beta,
        })
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();
        let input_rank = input_shape.ndim();
        let norm_rank = self.normalized_shape.len();

        // Verify that the last dimensions match normalized_shape
        if input_rank < norm_rank {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "Input rank {} is less than normalized_shape rank {}",
                input_rank, norm_rank
            )));
        }

        for i in 0..norm_rank {
            if input_shape[input_rank - norm_rank + i] != self.normalized_shape[i] {
                return Err(hodu_core::error::HoduError::InternalError(format!(
                    "Input shape mismatch at dimension {}",
                    i
                )));
            }
        }

        // Compute axes to normalize over (last norm_rank dimensions)
        let axes: Vec<usize> = (0..norm_rank).map(|i| input_rank - norm_rank + i).collect();

        // Compute mean and variance over the normalized dimensions
        let mean = input.mean(&axes, true)?;
        let centered = input.sub(&mean)?;
        let squared = centered.mul(&centered)?;
        let var = squared.mean(&axes, true)?;

        // Normalize: (x - mean) / sqrt(var + eps)
        let eps_typed = self.eps.to_dtype(input.dtype());
        let std = var.add_scalar(eps_typed)?.sqrt()?;
        let normalized = centered.div(&std)?;

        // Apply elementwise affine transformation if enabled
        if self.elementwise_affine {
            let gamma = self.gamma.as_ref().unwrap();
            let beta = self.beta.as_ref().unwrap();

            let scaled = normalized.mul(gamma)?;
            scaled.add(beta)
        } else {
            Ok(normalized)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if self.elementwise_affine {
            if let Some(ref gamma) = self.gamma {
                params.push(gamma);
            }
            if let Some(ref beta) = self.beta {
                params.push(beta);
            }
        }
        params
    }
}

// ============================================================================
// GroupNorm
// ============================================================================

/// Group Normalization.
///
/// Divides channels into groups and normalizes within each group.
/// Input shape: [N, C, *] where C must be divisible by num_groups.
#[derive(Module, Clone)]
pub struct GroupNorm {
    num_groups: usize,
    num_channels: usize,
    eps: Scalar,
    affine: bool,
    gamma: Option<Tensor>,
    beta: Option<Tensor>,
}

impl GroupNorm {
    /// Creates a new GroupNorm layer.
    ///
    /// # Arguments
    /// * `num_groups` - Number of groups to divide channels into
    /// * `num_channels` - Number of channels (must be divisible by num_groups)
    /// * `eps` - Small constant for numerical stability
    /// * `affine` - If true, learnable affine parameters
    /// * `dtype` - Data type for parameters
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: impl Into<Scalar>,
        affine: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        assert!(
            num_channels % num_groups == 0,
            "num_channels ({}) must be divisible by num_groups ({})",
            num_channels,
            num_groups
        );

        let eps = eps.into();

        let (gamma, beta) = if affine {
            let gamma = Tensor::ones([num_channels], dtype)?;
            gamma.set_requires_grad(true)?;

            let beta = Tensor::zeros([num_channels], dtype)?;
            beta.set_requires_grad(true)?;

            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        Ok(Self {
            num_groups,
            num_channels,
            eps,
            affine,
            gamma,
            beta,
        })
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();
        let ndim = input_shape.ndim();

        if ndim < 2 {
            return Err(hodu_core::error::HoduError::InternalError(
                "GroupNorm expects at least 2D input [N, C, ...]".to_string(),
            ));
        }

        let n = input_shape[0];
        let c = input_shape[1];

        if c != self.num_channels {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "Expected {} channels, got {}",
                self.num_channels, c
            )));
        }

        let channels_per_group = c / self.num_groups;

        // Reshape to [N, num_groups, channels_per_group, *spatial]
        let mut new_shape = vec![n, self.num_groups, channels_per_group];
        for i in 2..ndim {
            new_shape.push(input_shape[i]);
        }
        let reshaped = input.reshape(&new_shape)?;

        // Compute axes to normalize (all except N and num_groups)
        let axes: Vec<usize> = (2..new_shape.len()).collect();

        // Compute mean and variance
        let mean = reshaped.mean(&axes, true)?;
        let centered = reshaped.sub(&mean)?;
        let var = centered.square()?.mean(&axes, true)?;

        // Normalize
        let eps_typed = self.eps.to_dtype(input.dtype());
        let std = var.add_scalar(eps_typed)?.sqrt()?;
        let normalized = centered.div(&std)?;

        // Reshape back to original shape
        let normalized = normalized.reshape(input_shape.dims())?;

        // Apply affine transformation
        if self.affine {
            let gamma = self.gamma.as_ref().unwrap();
            let beta = self.beta.as_ref().unwrap();

            // Reshape gamma and beta to broadcast: [1, C, 1, 1, ...]
            let mut affine_shape = vec![1usize; ndim];
            affine_shape[1] = self.num_channels;

            let gamma_reshaped = gamma.reshape(&affine_shape)?;
            let beta_reshaped = beta.reshape(&affine_shape)?;

            normalized.mul(&gamma_reshaped)?.add(&beta_reshaped)
        } else {
            Ok(normalized)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if self.affine {
            if let Some(ref gamma) = self.gamma {
                params.push(gamma);
            }
            if let Some(ref beta) = self.beta {
                params.push(beta);
            }
        }
        params
    }
}

// ============================================================================
// InstanceNorm1D
// ============================================================================

/// Instance Normalization for 1D inputs.
///
/// Normalizes each channel independently for each sample.
/// Input shape: [N, C, L]
#[derive(Module, Clone)]
pub struct InstanceNorm1D {
    num_features: usize,
    eps: Scalar,
    affine: bool,
    gamma: Option<Tensor>,
    beta: Option<Tensor>,
}

impl InstanceNorm1D {
    pub fn new(num_features: usize, eps: impl Into<Scalar>, affine: bool, dtype: DType) -> HoduResult<Self> {
        let eps = eps.into();

        let (gamma, beta) = if affine {
            let gamma = Tensor::ones([num_features], dtype)?;
            gamma.set_requires_grad(true)?;

            let beta = Tensor::zeros([num_features], dtype)?;
            beta.set_requires_grad(true)?;

            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        Ok(Self {
            num_features,
            eps,
            affine,
            gamma,
            beta,
        })
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();

        if input_shape.ndim() != 3 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "InstanceNorm1D expects 3D input [N, C, L], got {}D",
                input_shape.ndim()
            )));
        }

        // Normalize over L dimension (axis 2) for each (N, C)
        let mean = input.mean(&[2], true)?;
        let centered = input.sub(&mean)?;
        let var = centered.square()?.mean(&[2], true)?;

        let eps_typed = self.eps.to_dtype(input.dtype());
        let std = var.add_scalar(eps_typed)?.sqrt()?;
        let normalized = centered.div(&std)?;

        if self.affine {
            let gamma = self.gamma.as_ref().unwrap();
            let beta = self.beta.as_ref().unwrap();

            let gamma_reshaped = gamma.reshape([1, self.num_features, 1])?;
            let beta_reshaped = beta.reshape([1, self.num_features, 1])?;

            normalized.mul(&gamma_reshaped)?.add(&beta_reshaped)
        } else {
            Ok(normalized)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if self.affine {
            if let Some(ref gamma) = self.gamma {
                params.push(gamma);
            }
            if let Some(ref beta) = self.beta {
                params.push(beta);
            }
        }
        params
    }
}

// ============================================================================
// InstanceNorm2D
// ============================================================================

/// Instance Normalization for 2D inputs.
///
/// Normalizes each channel independently for each sample.
/// Input shape: [N, C, H, W]
#[derive(Module, Clone)]
pub struct InstanceNorm2D {
    num_features: usize,
    eps: Scalar,
    affine: bool,
    gamma: Option<Tensor>,
    beta: Option<Tensor>,
}

impl InstanceNorm2D {
    pub fn new(num_features: usize, eps: impl Into<Scalar>, affine: bool, dtype: DType) -> HoduResult<Self> {
        let eps = eps.into();

        let (gamma, beta) = if affine {
            let gamma = Tensor::ones([num_features], dtype)?;
            gamma.set_requires_grad(true)?;

            let beta = Tensor::zeros([num_features], dtype)?;
            beta.set_requires_grad(true)?;

            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        Ok(Self {
            num_features,
            eps,
            affine,
            gamma,
            beta,
        })
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();

        if input_shape.ndim() != 4 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "InstanceNorm2D expects 4D input [N, C, H, W], got {}D",
                input_shape.ndim()
            )));
        }

        // Normalize over H, W dimensions (axes 2, 3) for each (N, C)
        let mean = input.mean(&[2, 3], true)?;
        let centered = input.sub(&mean)?;
        let var = centered.square()?.mean(&[2, 3], true)?;

        let eps_typed = self.eps.to_dtype(input.dtype());
        let std = var.add_scalar(eps_typed)?.sqrt()?;
        let normalized = centered.div(&std)?;

        if self.affine {
            let gamma = self.gamma.as_ref().unwrap();
            let beta = self.beta.as_ref().unwrap();

            let gamma_reshaped = gamma.reshape([1, self.num_features, 1, 1])?;
            let beta_reshaped = beta.reshape([1, self.num_features, 1, 1])?;

            normalized.mul(&gamma_reshaped)?.add(&beta_reshaped)
        } else {
            Ok(normalized)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if self.affine {
            if let Some(ref gamma) = self.gamma {
                params.push(gamma);
            }
            if let Some(ref beta) = self.beta {
                params.push(beta);
            }
        }
        params
    }
}

// ============================================================================
// InstanceNorm3D
// ============================================================================

/// Instance Normalization for 3D inputs.
///
/// Normalizes each channel independently for each sample.
/// Input shape: [N, C, D, H, W]
#[derive(Module, Clone)]
pub struct InstanceNorm3D {
    num_features: usize,
    eps: Scalar,
    affine: bool,
    gamma: Option<Tensor>,
    beta: Option<Tensor>,
}

impl InstanceNorm3D {
    pub fn new(num_features: usize, eps: impl Into<Scalar>, affine: bool, dtype: DType) -> HoduResult<Self> {
        let eps = eps.into();

        let (gamma, beta) = if affine {
            let gamma = Tensor::ones([num_features], dtype)?;
            gamma.set_requires_grad(true)?;

            let beta = Tensor::zeros([num_features], dtype)?;
            beta.set_requires_grad(true)?;

            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        Ok(Self {
            num_features,
            eps,
            affine,
            gamma,
            beta,
        })
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();

        if input_shape.ndim() != 5 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "InstanceNorm3D expects 5D input [N, C, D, H, W], got {}D",
                input_shape.ndim()
            )));
        }

        // Normalize over D, H, W dimensions (axes 2, 3, 4) for each (N, C)
        let mean = input.mean(&[2, 3, 4], true)?;
        let centered = input.sub(&mean)?;
        let var = centered.square()?.mean(&[2, 3, 4], true)?;

        let eps_typed = self.eps.to_dtype(input.dtype());
        let std = var.add_scalar(eps_typed)?.sqrt()?;
        let normalized = centered.div(&std)?;

        if self.affine {
            let gamma = self.gamma.as_ref().unwrap();
            let beta = self.beta.as_ref().unwrap();

            let gamma_reshaped = gamma.reshape([1, self.num_features, 1, 1, 1])?;
            let beta_reshaped = beta.reshape([1, self.num_features, 1, 1, 1])?;

            normalized.mul(&gamma_reshaped)?.add(&beta_reshaped)
        } else {
            Ok(normalized)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if self.affine {
            if let Some(ref gamma) = self.gamma {
                params.push(gamma);
            }
            if let Some(ref beta) = self.beta {
                params.push(beta);
            }
        }
        params
    }
}

// ============================================================================
// RMSNorm
// ============================================================================

/// Root Mean Square Layer Normalization.
///
/// Used in models like LLaMA. Normalizes using only the RMS of the input,
/// without subtracting the mean.
///
/// y = x / sqrt(mean(x²) + eps) * gamma
#[derive(Module, Clone)]
pub struct RMSNorm {
    normalized_shape: Vec<usize>,
    eps: Scalar,
    gamma: Tensor,
}

impl RMSNorm {
    /// Creates a new RMSNorm layer.
    ///
    /// # Arguments
    /// * `normalized_shape` - Shape of the last dimensions to normalize
    /// * `eps` - Small constant for numerical stability (default: 1e-6)
    /// * `dtype` - Data type for parameters
    pub fn new(normalized_shape: Vec<usize>, eps: impl Into<Scalar>, dtype: DType) -> HoduResult<Self> {
        let eps = eps.into();

        let gamma = Tensor::ones(&normalized_shape, dtype)?;
        gamma.set_requires_grad(true)?;

        Ok(Self {
            normalized_shape,
            eps,
            gamma,
        })
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();
        let input_rank = input_shape.ndim();
        let norm_rank = self.normalized_shape.len();

        if input_rank < norm_rank {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "Input rank {} is less than normalized_shape rank {}",
                input_rank, norm_rank
            )));
        }

        // Verify shape match
        for i in 0..norm_rank {
            if input_shape[input_rank - norm_rank + i] != self.normalized_shape[i] {
                return Err(hodu_core::error::HoduError::InternalError(format!(
                    "Input shape mismatch at dimension {}",
                    i
                )));
            }
        }

        // Compute axes to normalize over (last norm_rank dimensions)
        let axes: Vec<usize> = (0..norm_rank).map(|i| input_rank - norm_rank + i).collect();

        // Compute RMS: sqrt(mean(x²) + eps)
        let squared = input.square()?;
        let mean_squared = squared.mean(&axes, true)?;
        let eps_typed = self.eps.to_dtype(input.dtype());
        let rms = mean_squared.add_scalar(eps_typed)?.sqrt()?;

        // Normalize: x / rms
        let normalized = input.div(&rms)?;

        // Scale by gamma
        normalized.mul(&self.gamma)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma]
    }
}
