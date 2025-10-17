#![cfg_attr(not(feature = "std"), no_std)]

pub(crate) mod compat;
mod losses;
pub mod module;
mod modules;
pub mod optimizer;
mod optimizers;
pub mod prelude;

pub use losses::{
    binary_cross_entropy::{BCELoss, BCEWithLogitsLoss},
    cross_entropy::CrossEntropyLoss,
    huber::Huber,
    mae::MAE,
    mse::MSE,
    nll::NLLLoss,
};
pub use modules::{
    activation::*,
    conv::{Conv1D, Conv2D, Conv3D, ConvTranspose1D, ConvTranspose2D, ConvTranspose3D},
    dropout::Dropout,
    linear::Linear,
};
pub use optimizers::{
    adam::{Adam, AdamW},
    sgd::SGD,
};

pub(crate) mod state {
    use crate::compat::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub(crate) enum State {
        Training,
        Evaluation,
    }

    static STATE: Mutex<State> = Mutex::new(State::Training);

    #[allow(dead_code)]
    pub fn set_state(new_state: State) {
        #[cfg(feature = "std")]
        {
            *STATE.lock().unwrap() = new_state;
        }
        #[cfg(not(feature = "std"))]
        {
            *STATE.lock() = new_state;
        }
    }

    pub fn get_state() -> State {
        #[cfg(feature = "std")]
        {
            *STATE.lock().unwrap()
        }
        #[cfg(not(feature = "std"))]
        {
            *STATE.lock()
        }
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
