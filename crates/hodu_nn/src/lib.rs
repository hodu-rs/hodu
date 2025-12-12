mod losses;
pub mod module;
mod modules;
pub mod optimizer;
mod optimizers;
pub mod prelude;
pub use losses::{
    binary_cross_entropy::{BCELoss, BCEWithLogitsLoss},
    cross_entropy::CrossEntropyLoss,
    huber::HuberLoss,
    mae::MAELoss,
    mse::MSELoss,
    nll::NLLLoss,
};
pub use modules::{
    activation::*,
    attention::{scaled_dot_product_attention, MultiheadAttention},
    conv::{Conv1D, Conv2D, Conv3D, ConvTranspose1D, ConvTranspose2D, ConvTranspose3D},
    dropout::Dropout,
    embedding::Embedding,
    linear::Linear,
    norm::{BatchNorm1D, BatchNorm2D, BatchNorm3D, LayerNorm},
    pooling::{
        AdaptiveAvgPool1D, AdaptiveAvgPool2D, AdaptiveAvgPool3D, AdaptiveMaxPool1D, AdaptiveMaxPool2D,
        AdaptiveMaxPool3D, AvgPool1D, AvgPool2D, AvgPool3D, MaxPool1D, MaxPool2D, MaxPool3D,
    },
    rnn::{GRUCell, LSTMCell, Nonlinearity, RNNCell, GRU, LSTM, RNN},
};
pub use optimizers::{
    adam::{Adam, AdamW},
    sgd::SGD,
};

pub(crate) mod state {
    use std::sync::Mutex;

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub(crate) enum State {
        Training,
        Evaluation,
    }

    static STATE: Mutex<State> = Mutex::new(State::Training);

    #[allow(dead_code)]
    pub fn set_state(new_state: State) {
        *STATE.lock().unwrap() = new_state;
    }

    pub fn get_state() -> State {
        *STATE.lock().unwrap()
    }
}

/// Set training mode for all modules
#[macro_export]
macro_rules! train {
    () => {
        state::set_state(state::State::Training);
    };
}

/// Set evaluation mode for all modules
#[macro_export]
macro_rules! eval {
    () => {
        state::set_state(state::State::Evaluation);
    };
}
