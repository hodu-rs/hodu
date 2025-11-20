use crate::compat::*;
use crate::module::Module;
use crate::state::{get_state, State};
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor, types::DType};

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
