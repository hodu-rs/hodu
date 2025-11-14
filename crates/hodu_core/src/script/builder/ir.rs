use crate::{
    layer::compat::*,
    ops::Op,
    tensor::TensorId,
    types::{DType, Layout, Shape},
};

/// Value ID in SSA form
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct ValueId(pub usize);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Basic block ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct BlockId(pub usize);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "^bb{}", self.0)
    }
}

/// Module represents the top-level IR container (complete model)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct Module {
    pub name: String,
    pub functions: Vec<Function>,
    pub constants: HashMap<TensorId, ConstantData>,
    pub metadata: ModuleMetadata,
}

/// Module metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct ModuleMetadata {
    pub version: String,
    pub hodu_version: String,
    pub created_at: String,
    pub description: Option<String>,
}

/// Function represents a callable unit with SSA form
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct Function {
    pub name: String,
    pub signature: FunctionSignature,
    pub blocks: Vec<BasicBlock>,
    pub entry_block: BlockId,
    pub value_info: HashMap<ValueId, ValueInfo>,
}

/// Function signature
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct FunctionSignature {
    pub inputs: Vec<Parameter>,
    pub outputs: Vec<Parameter>,
}

/// Parameter information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct Parameter {
    pub name: String,
    pub value_id: ValueId,
    pub shape: Option<Shape>,
    pub dtype: Option<DType>,
}

/// Basic block in SSA form
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct BasicBlock {
    pub id: BlockId,
    pub label: Option<String>,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
}

/// SSA instruction
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum Instruction {
    /// Compute operation: %result = op(%inputs...)
    Compute {
        result: ValueId,
        op: Op,
        inputs: Vec<ValueId>,
        attributes: HashMap<String, Attribute>,
    },

    /// Load constant: %result = load_constant @tensor_id
    LoadConstant { result: ValueId, tensor_id: TensorId },

    /// Phi node for control flow: %result = phi [%val1, ^bb1], [%val2, ^bb2]
    Phi {
        result: ValueId,
        incoming: Vec<(BlockId, ValueId)>,
    },
}

/// Terminator instruction (ends a basic block)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum Terminator {
    /// Return from function
    Return { values: Vec<ValueId> },

    /// Unconditional jump
    Jump { target: BlockId },

    /// Conditional branch
    Branch {
        condition: ValueId,
        true_block: BlockId,
        false_block: BlockId,
    },
}

/// Value information (shape, dtype, layout)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct ValueInfo {
    pub id: ValueId,
    pub shape: Option<Shape>,
    pub dtype: Option<DType>,
    pub layout: Option<Layout>,
}

/// Attribute value
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum Attribute {
    Bool(bool),
    Int(i32),
    Usize(usize),
    Float(f32),
    String(String),
    IntArray(Vec<i32>),
    FloatArray(Vec<f32>),
    Scalar(crate::scalar::Scalar),
    Scalars(Vec<crate::scalar::Scalar>),
    DType(crate::types::DType),
}

/// Constant data stored in module
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct ConstantData {
    pub tensor_id: TensorId,
    pub shape: Shape,
    pub dtype: DType,
    pub data: Vec<u8>,
    pub compression: Option<CompressionType>,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum CompressionType {
    None,
    #[cfg(feature = "std")]
    Gzip,
    Zstd,
    Lz4,
}

impl Module {
    pub fn new(name: String) -> Self {
        Self {
            name,
            functions: Vec::new(),
            constants: HashMap::new(),
            metadata: ModuleMetadata {
                version: "1.0".to_string(),
                hodu_version: env!("CARGO_PKG_VERSION").to_string(),
                #[cfg(feature = "std")]
                created_at: {
                    use std::time::{SystemTime, UNIX_EPOCH};
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs().to_string())
                        .unwrap_or_else(|_| "0".to_string())
                },
                #[cfg(not(feature = "std"))]
                created_at: "0".to_string(),
                description: None,
            },
        }
    }

    pub fn add_function(&mut self, function: Function) {
        self.functions.push(function);
    }

    pub fn add_constant(&mut self, tensor_id: TensorId, constant: ConstantData) {
        self.constants.insert(tensor_id, constant);
    }

    pub fn get_function(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }

    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut Function> {
        self.functions.iter_mut().find(|f| f.name == name)
    }
}

impl Function {
    pub fn new(name: String, signature: FunctionSignature, entry_block: BlockId) -> Self {
        Self {
            name,
            signature,
            blocks: Vec::new(),
            entry_block,
            value_info: HashMap::new(),
        }
    }

    pub fn add_block(&mut self, block: BasicBlock) {
        self.blocks.push(block);
    }

    pub fn add_value_info(&mut self, value_id: ValueId, info: ValueInfo) {
        self.value_info.insert(value_id, info);
    }

    pub fn get_block(&self, block_id: BlockId) -> Option<&BasicBlock> {
        self.blocks.iter().find(|b| b.id == block_id)
    }

    pub fn get_block_mut(&mut self, block_id: BlockId) -> Option<&mut BasicBlock> {
        self.blocks.iter_mut().find(|b| b.id == block_id)
    }
}

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            label: None,
            instructions: Vec::new(),
            terminator: Terminator::Return { values: Vec::new() },
        }
    }

    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    pub fn add_instruction(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }

    pub fn set_terminator(&mut self, terminator: Terminator) {
        self.terminator = terminator;
    }
}

impl FunctionSignature {
    pub fn new(inputs: Vec<Parameter>, outputs: Vec<Parameter>) -> Self {
        Self { inputs, outputs }
    }
}

impl Parameter {
    pub fn new(name: String, value_id: ValueId) -> Self {
        Self {
            name,
            value_id,
            shape: None,
            dtype: None,
        }
    }

    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = Some(shape);
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }
}

impl ValueInfo {
    pub fn new(id: ValueId) -> Self {
        Self {
            id,
            shape: None,
            dtype: None,
            layout: None,
        }
    }

    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = Some(shape);
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    pub fn with_layout(mut self, layout: Layout) -> Self {
        self.layout = Some(layout);
        self
    }
}
