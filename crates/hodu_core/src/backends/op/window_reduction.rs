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

impl fmt::Display for WindowReduction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            WindowReduction::Max => "max",
            WindowReduction::Mean => "mean",
            WindowReduction::Sum => "sum",
            WindowReduction::Min => "min",
        };
        write!(f, "{}", s)
    }
}
