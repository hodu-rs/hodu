use crate::compat::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum WindowReduction {
    Max, // no-backprop
    Mean,
    Sum,
    Min, // no-backprop
}

impl WindowReduction {
    pub fn to_string(&self) -> String {
        match self {
            WindowReduction::Max => "max".to_string(),
            WindowReduction::Mean => "mean".to_string(),
            WindowReduction::Sum => "sum".to_string(),
            WindowReduction::Min => "min".to_string(),
        }
    }
}
