#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    BOOL,
    BF16,
    F16,
    F32,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
}

impl DType {
    pub fn get_size_in_bytes(&self) -> usize {
        match self {
            Self::BOOL => 1,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
        }
    }
}
