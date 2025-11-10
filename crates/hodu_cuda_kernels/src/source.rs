use crate::compat::*;

// Include pre-compiled PTX from build.rs
include!(concat!(env!("OUT_DIR"), "/generated_source.rs"));

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Source {
    OpsBinary,
    OpsCast,
    OpsConcatSplit,
    OpsConv,
    OpsIndexing,
    OpsMatrix,
    OpsMemory,
    OpsReduce,
    OpsUnary,
    OpsWindowing,
    Storage,
}
