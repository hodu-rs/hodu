use crate::module::Module;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor, types::DType};

// ============================================================================
// RNNCell
// ============================================================================

/// A single RNN cell that computes one step of the recurrence.
///
/// h' = tanh(W_ih @ x + b_ih + W_hh @ h + b_hh)
///
/// or with ReLU nonlinearity:
///
/// h' = relu(W_ih @ x + b_ih + W_hh @ h + b_hh)
#[derive(Module, Clone)]
pub struct RNNCell {
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    hidden_size: usize,
    nonlinearity: Nonlinearity,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Nonlinearity {
    Tanh,
    ReLU,
}

impl RNNCell {
    /// Creates a new RNNCell.
    ///
    /// # Arguments
    /// * `input_size` - The number of expected features in the input
    /// * `hidden_size` - The number of features in the hidden state
    /// * `with_bias` - If `false`, the layer does not use bias weights
    /// * `nonlinearity` - The non-linearity to use (`Tanh` or `ReLU`)
    /// * `dtype` - Data type for parameters
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        with_bias: bool,
        nonlinearity: Nonlinearity,
        dtype: DType,
    ) -> HoduResult<Self> {
        let k: f32 = 1.0 / (hidden_size as f32).sqrt();
        let zero = Scalar::zero(dtype);
        let one = Scalar::one(dtype);
        let k_scalar = Scalar::from_f32(k, dtype);

        // weight_ih: [hidden_size, input_size]
        let weight_ih = Tensor::randn([hidden_size, input_size], zero, one)?;
        weight_ih.set_requires_grad(true)?;
        let weight_ih = weight_ih.mul_scalar(k_scalar)?;

        // weight_hh: [hidden_size, hidden_size]
        let weight_hh = Tensor::randn([hidden_size, hidden_size], zero, one)?;
        weight_hh.set_requires_grad(true)?;
        let weight_hh = weight_hh.mul_scalar(k_scalar)?;

        let (bias_ih, bias_hh) = if with_bias {
            let bias_ih = Tensor::randn([hidden_size], zero, one)?;
            bias_ih.set_requires_grad(true)?;
            let bias_ih = bias_ih.mul_scalar(k_scalar)?;

            let bias_hh = Tensor::randn([hidden_size], zero, one)?;
            bias_hh.set_requires_grad(true)?;
            let bias_hh = bias_hh.mul_scalar(k_scalar)?;

            (Some(bias_ih), Some(bias_hh))
        } else {
            (None, None)
        };

        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
            nonlinearity,
        })
    }

    /// Computes one step: h' = nonlinearity(W_ih @ x + b_ih + W_hh @ h + b_hh)
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, input_size]
    /// * `hidden` - Hidden state tensor of shape [batch, hidden_size]
    ///
    /// # Returns
    /// New hidden state of shape [batch, hidden_size]
    pub fn forward_step(&self, input: &Tensor, hidden: &Tensor) -> HoduResult<Tensor> {
        // W_ih @ x.T -> [hidden, batch] -> transpose -> [batch, hidden]
        let mut h = input.matmul(&self.weight_ih.transpose(-2, -1)?)?;

        if let Some(ref bias) = self.bias_ih {
            h = h.add(bias)?;
        }

        // + W_hh @ h.T
        h = h.add(&hidden.matmul(&self.weight_hh.transpose(-2, -1)?)?)?;

        if let Some(ref bias) = self.bias_hh {
            h = h.add(bias)?;
        }

        // Apply nonlinearity
        match self.nonlinearity {
            Nonlinearity::Tanh => h.tanh(),
            Nonlinearity::ReLU => h.relu(),
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn forward(&self, _input: &Tensor) -> HoduResult<Tensor> {
        Err(hodu_core::error::HoduError::InternalError(
            "RNNCell::forward requires hidden state, use forward_step instead".to_string(),
        ))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref b) = self.bias_ih {
            params.push(b);
        }
        if let Some(ref b) = self.bias_hh {
            params.push(b);
        }
        params
    }
}

// ============================================================================
// RNN
// ============================================================================

/// Multi-layer Elman RNN with tanh or ReLU non-linearity.
///
/// For each element in the input sequence, each layer computes:
/// h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
#[derive(Module, Clone)]
pub struct RNN {
    cells: Vec<RNNCell>,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    batch_first: bool,
}

impl RNN {
    /// Creates a new RNN.
    ///
    /// # Arguments
    /// * `input_size` - The number of expected features in the input
    /// * `hidden_size` - The number of features in the hidden state
    /// * `num_layers` - Number of recurrent layers (default: 1)
    /// * `with_bias` - If `false`, the layer does not use bias weights
    /// * `batch_first` - If `true`, input/output tensors are [batch, seq, feature]
    /// * `nonlinearity` - The non-linearity to use (`Tanh` or `ReLU`)
    /// * `dtype` - Data type for parameters
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        with_bias: bool,
        batch_first: bool,
        nonlinearity: Nonlinearity,
        dtype: DType,
    ) -> HoduResult<Self> {
        let mut cells = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let cell_input_size = if i == 0 { input_size } else { hidden_size };
            cells.push(RNNCell::new(
                cell_input_size,
                hidden_size,
                with_bias,
                nonlinearity,
                dtype,
            )?);
        }

        Ok(Self {
            cells,
            input_size,
            hidden_size,
            num_layers,
            batch_first,
        })
    }

    /// Forward pass through the RNN.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
    /// * `h_0` - Initial hidden state of shape [num_layers, batch, hidden_size]. If None, zeros are used.
    ///
    /// # Returns
    /// * `output` - Output tensor of shape [seq_len, batch, hidden_size] or [batch, seq_len, hidden_size] if batch_first
    /// * `h_n` - Final hidden state of shape [num_layers, batch, hidden_size]
    pub fn forward_with_state(&self, input: &Tensor, h_0: Option<&Tensor>) -> HoduResult<(Tensor, Tensor)> {
        let input_shape = input.shape();
        let (seq_len, batch_size) = if self.batch_first {
            (input_shape[1], input_shape[0])
        } else {
            (input_shape[0], input_shape[1])
        };

        // Transpose if batch_first to work with [seq, batch, features]
        let input = if self.batch_first {
            input.transpose(0, 1)?
        } else {
            input.clone()
        };

        // Initialize hidden states for each layer
        let mut hidden_states: Vec<Tensor> = match h_0 {
            Some(h) => {
                let mut states = Vec::with_capacity(self.num_layers);
                for i in 0..self.num_layers {
                    let indices = Tensor::from_slice(&[i as i32], [1])?;
                    let h_i = h.index_select(0, &indices)?.squeeze(&[0])?;
                    states.push(h_i);
                }
                states
            },
            None => {
                let dtype = input.dtype();
                (0..self.num_layers)
                    .map(|_| Tensor::zeros([batch_size, self.hidden_size], dtype))
                    .collect::<HoduResult<Vec<_>>>()?
            },
        };

        // Process sequence
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Get input at time t: [batch, input_size]
            let indices = Tensor::from_slice(&[t as i32], [1])?;
            let mut x_t = input.index_select(0, &indices)?.squeeze(&[0])?;

            // Process through each layer
            for (layer_idx, cell) in self.cells.iter().enumerate() {
                let h_prev = &hidden_states[layer_idx];
                let h_new = cell.forward_step(&x_t, h_prev)?;
                hidden_states[layer_idx] = h_new.clone();
                x_t = h_new;
            }

            outputs.push(x_t);
        }

        // Stack outputs: [seq_len, batch, hidden_size]
        let output_refs: Vec<&Tensor> = outputs.iter().collect();
        let output = Tensor::stack(&output_refs, 0)?;

        // Stack final hidden states: [num_layers, batch, hidden_size]
        let h_n_refs: Vec<&Tensor> = hidden_states.iter().collect();
        let h_n = Tensor::stack(&h_n_refs, 0)?;

        // Transpose output if batch_first
        let output = if self.batch_first {
            output.transpose(0, 1)?
        } else {
            output
        };

        Ok((output, h_n))
    }

    /// Returns the expected input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let (output, _) = self.forward_with_state(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.cells.iter().flat_map(|c| c.parameters()).collect()
    }
}

// ============================================================================
// LSTMCell
// ============================================================================

/// A single LSTM cell that computes one step of the recurrence.
///
/// i = sigmoid(W_ii @ x + b_ii + W_hi @ h + b_hi)
/// f = sigmoid(W_if @ x + b_if + W_hf @ h + b_hf)
/// g = tanh(W_ig @ x + b_ig + W_hg @ h + b_hg)
/// o = sigmoid(W_io @ x + b_io + W_ho @ h + b_ho)
/// c' = f * c + i * g
/// h' = o * tanh(c')
#[derive(Module, Clone)]
pub struct LSTMCell {
    // Combined weights for efficiency: [4*hidden_size, input_size] for i,f,g,o gates
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    hidden_size: usize,
}

impl LSTMCell {
    /// Creates a new LSTMCell.
    ///
    /// # Arguments
    /// * `input_size` - The number of expected features in the input
    /// * `hidden_size` - The number of features in the hidden state
    /// * `with_bias` - If `false`, the layer does not use bias weights
    /// * `dtype` - Data type for parameters
    pub fn new(input_size: usize, hidden_size: usize, with_bias: bool, dtype: DType) -> HoduResult<Self> {
        let k: f32 = 1.0 / (hidden_size as f32).sqrt();
        let zero = Scalar::zero(dtype);
        let one = Scalar::one(dtype);
        let k_scalar = Scalar::from_f32(k, dtype);

        // Combined weights for 4 gates: [4*hidden_size, input_size]
        let weight_ih = Tensor::randn([4 * hidden_size, input_size], zero, one)?;
        weight_ih.set_requires_grad(true)?;
        let weight_ih = weight_ih.mul_scalar(k_scalar)?;

        // Combined weights for 4 gates: [4*hidden_size, hidden_size]
        let weight_hh = Tensor::randn([4 * hidden_size, hidden_size], zero, one)?;
        weight_hh.set_requires_grad(true)?;
        let weight_hh = weight_hh.mul_scalar(k_scalar)?;

        let (bias_ih, bias_hh) = if with_bias {
            let bias_ih = Tensor::randn([4 * hidden_size], zero, one)?;
            bias_ih.set_requires_grad(true)?;
            let bias_ih = bias_ih.mul_scalar(k_scalar)?;

            let bias_hh = Tensor::randn([4 * hidden_size], zero, one)?;
            bias_hh.set_requires_grad(true)?;
            let bias_hh = bias_hh.mul_scalar(k_scalar)?;

            (Some(bias_ih), Some(bias_hh))
        } else {
            (None, None)
        };

        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
        })
    }

    /// Computes one LSTM step.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, input_size]
    /// * `hidden` - Tuple of (h, c) where h and c are [batch, hidden_size]
    ///
    /// # Returns
    /// Tuple of (h', c') both of shape [batch, hidden_size]
    pub fn forward_step(&self, input: &Tensor, h: &Tensor, c: &Tensor) -> HoduResult<(Tensor, Tensor)> {
        let hs = self.hidden_size;

        // Compute gates: [batch, 4*hidden_size]
        let mut gates = input.matmul(&self.weight_ih.transpose(-2, -1)?)?;
        if let Some(ref bias) = self.bias_ih {
            gates = gates.add(bias)?;
        }
        gates = gates.add(&h.matmul(&self.weight_hh.transpose(-2, -1)?)?)?;
        if let Some(ref bias) = self.bias_hh {
            gates = gates.add(bias)?;
        }

        // Split gates: i, f, g, o using slice(dim, start, end, step)
        let i = gates.slice(1, 0, Some(hs), 1)?.sigmoid()?;
        let f = gates.slice(1, hs, Some(2 * hs), 1)?.sigmoid()?;
        let g = gates.slice(1, 2 * hs, Some(3 * hs), 1)?.tanh()?;
        let o = gates.slice(1, 3 * hs, Some(4 * hs), 1)?.sigmoid()?;

        // c' = f * c + i * g
        let c_new = f.mul(c)?.add(&i.mul(&g)?)?;

        // h' = o * tanh(c')
        let h_new = o.mul(&c_new.tanh()?)?;

        Ok((h_new, c_new))
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn forward(&self, _input: &Tensor) -> HoduResult<Tensor> {
        Err(hodu_core::error::HoduError::InternalError(
            "LSTMCell::forward requires hidden state, use forward_step instead".to_string(),
        ))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref b) = self.bias_ih {
            params.push(b);
        }
        if let Some(ref b) = self.bias_hh {
            params.push(b);
        }
        params
    }
}

// ============================================================================
// LSTM
// ============================================================================

/// Multi-layer Long Short-Term Memory (LSTM) RNN.
#[derive(Module, Clone)]
pub struct LSTM {
    cells: Vec<LSTMCell>,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    batch_first: bool,
}

impl LSTM {
    /// Creates a new LSTM.
    ///
    /// # Arguments
    /// * `input_size` - The number of expected features in the input
    /// * `hidden_size` - The number of features in the hidden state
    /// * `num_layers` - Number of recurrent layers (default: 1)
    /// * `with_bias` - If `false`, the layer does not use bias weights
    /// * `batch_first` - If `true`, input/output tensors are [batch, seq, feature]
    /// * `dtype` - Data type for parameters
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        with_bias: bool,
        batch_first: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        let mut cells = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let cell_input_size = if i == 0 { input_size } else { hidden_size };
            cells.push(LSTMCell::new(cell_input_size, hidden_size, with_bias, dtype)?);
        }

        Ok(Self {
            cells,
            input_size,
            hidden_size,
            num_layers,
            batch_first,
        })
    }

    /// Forward pass through the LSTM.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
    /// * `h_0` - Initial hidden state of shape [num_layers, batch, hidden_size]. If None, zeros are used.
    /// * `c_0` - Initial cell state of shape [num_layers, batch, hidden_size]. If None, zeros are used.
    ///
    /// # Returns
    /// * `output` - Output tensor of shape [seq_len, batch, hidden_size] or [batch, seq_len, hidden_size] if batch_first
    /// * `h_n` - Final hidden state of shape [num_layers, batch, hidden_size]
    /// * `c_n` - Final cell state of shape [num_layers, batch, hidden_size]
    pub fn forward_with_state(
        &self,
        input: &Tensor,
        h_0: Option<&Tensor>,
        c_0: Option<&Tensor>,
    ) -> HoduResult<(Tensor, Tensor, Tensor)> {
        let input_shape = input.shape();
        let (seq_len, batch_size) = if self.batch_first {
            (input_shape[1], input_shape[0])
        } else {
            (input_shape[0], input_shape[1])
        };

        // Transpose if batch_first to work with [seq, batch, features]
        let input = if self.batch_first {
            input.transpose(0, 1)?
        } else {
            input.clone()
        };

        let dtype = input.dtype();

        // Initialize hidden states
        let mut h_states: Vec<Tensor> = match h_0 {
            Some(h) => {
                let mut states = Vec::with_capacity(self.num_layers);
                for i in 0..self.num_layers {
                    let indices = Tensor::from_slice(&[i as i32], [1])?;
                    let h_i = h.index_select(0, &indices)?.squeeze(&[0])?;
                    states.push(h_i);
                }
                states
            },
            None => (0..self.num_layers)
                .map(|_| Tensor::zeros([batch_size, self.hidden_size], dtype))
                .collect::<HoduResult<Vec<_>>>()?,
        };

        // Initialize cell states
        let mut c_states: Vec<Tensor> = match c_0 {
            Some(c) => {
                let mut states = Vec::with_capacity(self.num_layers);
                for i in 0..self.num_layers {
                    let indices = Tensor::from_slice(&[i as i32], [1])?;
                    let c_i = c.index_select(0, &indices)?.squeeze(&[0])?;
                    states.push(c_i);
                }
                states
            },
            None => (0..self.num_layers)
                .map(|_| Tensor::zeros([batch_size, self.hidden_size], dtype))
                .collect::<HoduResult<Vec<_>>>()?,
        };

        // Process sequence
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Get input at time t: [batch, input_size]
            let indices = Tensor::from_slice(&[t as i32], [1])?;
            let mut x_t = input.index_select(0, &indices)?.squeeze(&[0])?;

            // Process through each layer
            for (layer_idx, cell) in self.cells.iter().enumerate() {
                let h_prev = &h_states[layer_idx];
                let c_prev = &c_states[layer_idx];
                let (h_new, c_new) = cell.forward_step(&x_t, h_prev, c_prev)?;
                h_states[layer_idx] = h_new.clone();
                c_states[layer_idx] = c_new;
                x_t = h_new;
            }

            outputs.push(x_t);
        }

        // Stack outputs: [seq_len, batch, hidden_size]
        let output_refs: Vec<&Tensor> = outputs.iter().collect();
        let output = Tensor::stack(&output_refs, 0)?;

        // Stack final states
        let h_n_refs: Vec<&Tensor> = h_states.iter().collect();
        let h_n = Tensor::stack(&h_n_refs, 0)?;

        let c_n_refs: Vec<&Tensor> = c_states.iter().collect();
        let c_n = Tensor::stack(&c_n_refs, 0)?;

        // Transpose output if batch_first
        let output = if self.batch_first {
            output.transpose(0, 1)?
        } else {
            output
        };

        Ok((output, h_n, c_n))
    }

    /// Returns the expected input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let (output, _, _) = self.forward_with_state(input, None, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.cells.iter().flat_map(|c| c.parameters()).collect()
    }
}

// ============================================================================
// GRUCell
// ============================================================================

/// A single GRU cell that computes one step of the recurrence.
///
/// r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
/// z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
/// n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
/// h' = (1 - z) * n + z * h
#[derive(Module, Clone)]
pub struct GRUCell {
    // Combined weights for efficiency: [3*hidden_size, input_size] for r,z,n gates
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    hidden_size: usize,
}

impl GRUCell {
    /// Creates a new GRUCell.
    ///
    /// # Arguments
    /// * `input_size` - The number of expected features in the input
    /// * `hidden_size` - The number of features in the hidden state
    /// * `with_bias` - If `false`, the layer does not use bias weights
    /// * `dtype` - Data type for parameters
    pub fn new(input_size: usize, hidden_size: usize, with_bias: bool, dtype: DType) -> HoduResult<Self> {
        let k: f32 = 1.0 / (hidden_size as f32).sqrt();
        let zero = Scalar::zero(dtype);
        let one = Scalar::one(dtype);
        let k_scalar = Scalar::from_f32(k, dtype);

        // Combined weights for 3 gates: [3*hidden_size, input_size]
        let weight_ih = Tensor::randn([3 * hidden_size, input_size], zero, one)?;
        weight_ih.set_requires_grad(true)?;
        let weight_ih = weight_ih.mul_scalar(k_scalar)?;

        // Combined weights for 3 gates: [3*hidden_size, hidden_size]
        let weight_hh = Tensor::randn([3 * hidden_size, hidden_size], zero, one)?;
        weight_hh.set_requires_grad(true)?;
        let weight_hh = weight_hh.mul_scalar(k_scalar)?;

        let (bias_ih, bias_hh) = if with_bias {
            let bias_ih = Tensor::randn([3 * hidden_size], zero, one)?;
            bias_ih.set_requires_grad(true)?;
            let bias_ih = bias_ih.mul_scalar(k_scalar)?;

            let bias_hh = Tensor::randn([3 * hidden_size], zero, one)?;
            bias_hh.set_requires_grad(true)?;
            let bias_hh = bias_hh.mul_scalar(k_scalar)?;

            (Some(bias_ih), Some(bias_hh))
        } else {
            (None, None)
        };

        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
        })
    }

    /// Computes one GRU step.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, input_size]
    /// * `hidden` - Hidden state tensor of shape [batch, hidden_size]
    ///
    /// # Returns
    /// New hidden state of shape [batch, hidden_size]
    pub fn forward_step(&self, input: &Tensor, hidden: &Tensor) -> HoduResult<Tensor> {
        let hs = self.hidden_size;

        // Compute input projections: [batch, 3*hidden_size]
        let mut gi = input.matmul(&self.weight_ih.transpose(-2, -1)?)?;
        if let Some(ref bias) = self.bias_ih {
            gi = gi.add(bias)?;
        }

        // Compute hidden projections: [batch, 3*hidden_size]
        let mut gh = hidden.matmul(&self.weight_hh.transpose(-2, -1)?)?;
        if let Some(ref bias) = self.bias_hh {
            gh = gh.add(bias)?;
        }

        // Split into r, z, n components using slice(dim, start, end, step)
        let gi_r = gi.slice(1, 0, Some(hs), 1)?;
        let gi_z = gi.slice(1, hs, Some(2 * hs), 1)?;
        let gi_n = gi.slice(1, 2 * hs, Some(3 * hs), 1)?;

        let gh_r = gh.slice(1, 0, Some(hs), 1)?;
        let gh_z = gh.slice(1, hs, Some(2 * hs), 1)?;
        let gh_n = gh.slice(1, 2 * hs, Some(3 * hs), 1)?;

        // r = sigmoid(gi_r + gh_r)
        let r = gi_r.add(&gh_r)?.sigmoid()?;

        // z = sigmoid(gi_z + gh_z)
        let z = gi_z.add(&gh_z)?.sigmoid()?;

        // n = tanh(gi_n + r * gh_n)
        let n = gi_n.add(&r.mul(&gh_n)?)?.tanh()?;

        // h' = (1 - z) * n + z * h
        let one = Scalar::one(hidden.dtype());
        let one_minus_z = z.neg()?.add_scalar(one)?;
        let h_new = one_minus_z.mul(&n)?.add(&z.mul(hidden)?)?;

        Ok(h_new)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn forward(&self, _input: &Tensor) -> HoduResult<Tensor> {
        Err(hodu_core::error::HoduError::InternalError(
            "GRUCell::forward requires hidden state, use forward_step instead".to_string(),
        ))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref b) = self.bias_ih {
            params.push(b);
        }
        if let Some(ref b) = self.bias_hh {
            params.push(b);
        }
        params
    }
}

// ============================================================================
// GRU
// ============================================================================

/// Multi-layer Gated Recurrent Unit (GRU) RNN.
#[derive(Module, Clone)]
pub struct GRU {
    cells: Vec<GRUCell>,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    batch_first: bool,
}

impl GRU {
    /// Creates a new GRU.
    ///
    /// # Arguments
    /// * `input_size` - The number of expected features in the input
    /// * `hidden_size` - The number of features in the hidden state
    /// * `num_layers` - Number of recurrent layers (default: 1)
    /// * `with_bias` - If `false`, the layer does not use bias weights
    /// * `batch_first` - If `true`, input/output tensors are [batch, seq, feature]
    /// * `dtype` - Data type for parameters
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        with_bias: bool,
        batch_first: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        let mut cells = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let cell_input_size = if i == 0 { input_size } else { hidden_size };
            cells.push(GRUCell::new(cell_input_size, hidden_size, with_bias, dtype)?);
        }

        Ok(Self {
            cells,
            input_size,
            hidden_size,
            num_layers,
            batch_first,
        })
    }

    /// Forward pass through the GRU.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
    /// * `h_0` - Initial hidden state of shape [num_layers, batch, hidden_size]. If None, zeros are used.
    ///
    /// # Returns
    /// * `output` - Output tensor of shape [seq_len, batch, hidden_size] or [batch, seq_len, hidden_size] if batch_first
    /// * `h_n` - Final hidden state of shape [num_layers, batch, hidden_size]
    pub fn forward_with_state(&self, input: &Tensor, h_0: Option<&Tensor>) -> HoduResult<(Tensor, Tensor)> {
        let input_shape = input.shape();
        let (seq_len, batch_size) = if self.batch_first {
            (input_shape[1], input_shape[0])
        } else {
            (input_shape[0], input_shape[1])
        };

        // Transpose if batch_first to work with [seq, batch, features]
        let input = if self.batch_first {
            input.transpose(0, 1)?
        } else {
            input.clone()
        };

        let dtype = input.dtype();

        // Initialize hidden states
        let mut hidden_states: Vec<Tensor> = match h_0 {
            Some(h) => {
                let mut states = Vec::with_capacity(self.num_layers);
                for i in 0..self.num_layers {
                    let indices = Tensor::from_slice(&[i as i32], [1])?;
                    let h_i = h.index_select(0, &indices)?.squeeze(&[0])?;
                    states.push(h_i);
                }
                states
            },
            None => (0..self.num_layers)
                .map(|_| Tensor::zeros([batch_size, self.hidden_size], dtype))
                .collect::<HoduResult<Vec<_>>>()?,
        };

        // Process sequence
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Get input at time t: [batch, input_size]
            let indices = Tensor::from_slice(&[t as i32], [1])?;
            let mut x_t = input.index_select(0, &indices)?.squeeze(&[0])?;

            // Process through each layer
            for (layer_idx, cell) in self.cells.iter().enumerate() {
                let h_prev = &hidden_states[layer_idx];
                let h_new = cell.forward_step(&x_t, h_prev)?;
                hidden_states[layer_idx] = h_new.clone();
                x_t = h_new;
            }

            outputs.push(x_t);
        }

        // Stack outputs: [seq_len, batch, hidden_size]
        let output_refs: Vec<&Tensor> = outputs.iter().collect();
        let output = Tensor::stack(&output_refs, 0)?;

        // Stack final hidden states: [num_layers, batch, hidden_size]
        let h_n_refs: Vec<&Tensor> = hidden_states.iter().collect();
        let h_n = Tensor::stack(&h_n_refs, 0)?;

        // Transpose output if batch_first
        let output = if self.batch_first {
            output.transpose(0, 1)?
        } else {
            output
        };

        Ok((output, h_n))
    }

    /// Returns the expected input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let (output, _) = self.forward_with_state(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.cells.iter().flat_map(|c| c.parameters()).collect()
    }
}
