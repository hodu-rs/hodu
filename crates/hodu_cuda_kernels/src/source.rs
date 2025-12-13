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
    OpsLinalg,
    OpsMatrix,
    OpsMemory,
    OpsPadding,
    OpsReduce,
    OpsResize,
    OpsScan,
    OpsShapeMemory,
    OpsSort,
    OpsUnary,
    OpsWindowing,
    Storage,
}
