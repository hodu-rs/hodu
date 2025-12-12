use std::collections::HashMap;

// Include pre-compiled PTX from build.rs
include!(concat!(env!("OUT_DIR"), "/generated_source.rs"));

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Source {
    OpsBinary,
    OpsCast,
    OpsConcatSplit,
    OpsConv,
    OpsEinsum,
    OpsIndexing,
    OpsMatrix,
    OpsMemory,
    OpsPadding,
    OpsReduce,
    OpsScan,
    OpsShapeMemory,
    OpsUnary,
    OpsWindowing,
    Storage,
}
