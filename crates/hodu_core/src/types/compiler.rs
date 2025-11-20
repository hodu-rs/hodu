use crate::compat::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Compiler {
    #[default]
    HoduScript,
}

impl fmt::Display for Compiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Compiler::HoduScript => write!(f, "hodu_script"),
        }
    }
}

impl fmt::Debug for Compiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Compiler[{}]", self)
    }
}
