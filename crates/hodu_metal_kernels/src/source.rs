use std::sync::OnceLock;

const ATOMIC: &str = include_str!("../kernels/atomic.metal");
const CONSTANTS: &str = include_str!("../kernels/constants.metal");
const UTILS: &str = include_str!("../kernels/utils.metal");

const BINARY_SRC: &str = include_str!("../kernels/ops_binary.metal");
const CAST_SRC: &str = include_str!("../kernels/ops_cast.metal");
const CONCAT_SPLIT_SRC: &str = include_str!("../kernels/ops_concat_split.metal");
const CONV_SRC: &str = include_str!("../kernels/ops_conv.metal");
const INDEXING_SRC: &str = include_str!("../kernels/ops_indexing.metal");
const MATRIX_SRC: &str = include_str!("../kernels/ops_matrix.metal");
const MEMORY_SRC: &str = include_str!("../kernels/ops_memory.metal");
const REDUCE_SRC: &str = include_str!("../kernels/ops_reduce.metal");
const UNARY_SRC: &str = include_str!("../kernels/ops_unary.metal");
const WINDOWING_SRC: &str = include_str!("../kernels/ops_windowing.metal");

fn combine_source(src: &str) -> String {
    // Replace all include directives with actual content
    src.replace("#include \"./atomic.metal\"", ATOMIC)
        .replace("#include \"./constants.metal\"", CONSTANTS)
        .replace("#include \"./utils.metal\"", UTILS)
}

static BINARY: OnceLock<String> = OnceLock::new();
static CAST: OnceLock<String> = OnceLock::new();
static CONCAT_SPLIT: OnceLock<String> = OnceLock::new();
static CONV: OnceLock<String> = OnceLock::new();
static INDEXING: OnceLock<String> = OnceLock::new();
static MATRIX: OnceLock<String> = OnceLock::new();
static MEMORY: OnceLock<String> = OnceLock::new();
static REDUCE: OnceLock<String> = OnceLock::new();
static UNARY: OnceLock<String> = OnceLock::new();
static WINDOWING: OnceLock<String> = OnceLock::new();

pub fn get_binary() -> &'static str {
    BINARY.get_or_init(|| combine_source(BINARY_SRC))
}

pub fn get_cast() -> &'static str {
    CAST.get_or_init(|| combine_source(CAST_SRC))
}

pub fn get_concat_split() -> &'static str {
    CONCAT_SPLIT.get_or_init(|| combine_source(CONCAT_SPLIT_SRC))
}

pub fn get_conv() -> &'static str {
    CONV.get_or_init(|| combine_source(CONV_SRC))
}

pub fn get_indexing() -> &'static str {
    INDEXING.get_or_init(|| combine_source(INDEXING_SRC))
}

pub fn get_matrix() -> &'static str {
    MATRIX.get_or_init(|| combine_source(MATRIX_SRC))
}

pub fn get_memory() -> &'static str {
    MEMORY.get_or_init(|| combine_source(MEMORY_SRC))
}

pub fn get_reduce() -> &'static str {
    REDUCE.get_or_init(|| combine_source(REDUCE_SRC))
}

pub fn get_unary() -> &'static str {
    UNARY.get_or_init(|| combine_source(UNARY_SRC))
}

pub fn get_windowing() -> &'static str {
    WINDOWING.get_or_init(|| combine_source(WINDOWING_SRC))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Binary,
    Cast,
    ConcatSplit,
    Conv,
    Indexing,
    Matrix,
    Memory,
    Reduce,
    Unary,
    Windowing,
}
