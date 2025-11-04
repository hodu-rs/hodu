mod codegen;
mod context;
pub mod ir;
mod printer;

use crate::error::HoduResult;
pub use context::{get_active_builder, is_builder_active, with_active_builder, Builder, BuilderState};
pub use ir::{
    Attribute, BasicBlock, BlockId, CompressionType, ConstantData, Function, FunctionSignature, Instruction, Module,
    ModuleMetadata, Parameter, Terminator, ValueId, ValueInfo,
};
pub use printer::print_module;

impl Module {
    /// Save module as text IR (.hds.ir)
    #[cfg(feature = "std")]
    pub fn save_ir<P: AsRef<std::path::Path>>(&self, path: P) -> HoduResult<()> {
        let ir_text = print_module(self);
        let mut path = path.as_ref().to_path_buf();

        // Add .hds.ir extension if no extension
        if path.extension().is_none() {
            path.set_extension("hds.ir");
        }

        std::fs::write(path, ir_text).map_err(|e| crate::error::HoduError::InternalError(e.to_string()))?;
        Ok(())
    }

    /// Load module from text IR (.hds.ir)
    #[cfg(feature = "std")]
    pub fn load_ir<P: AsRef<std::path::Path>>(_path: P) -> HoduResult<Self> {
        // TODO: Implement IR parser
        Err(crate::error::HoduError::NotImplemented(
            "IR parsing not yet implemented".to_string(),
        ))
    }

    /// Save module as binary (.hds.bc)
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn save_bc<P: AsRef<std::path::Path>>(&self, path: P) -> HoduResult<()> {
        let binary_data = bincode::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| crate::error::HoduError::InternalError(format!("Serialization error: {}", e)))?;

        let mut path = path.as_ref().to_path_buf();

        // Add .hds.bc extension if no extension
        if path.extension().is_none() {
            path.set_extension("hds.bc");
        }

        std::fs::write(path, binary_data).map_err(|e| crate::error::HoduError::InternalError(e.to_string()))?;
        Ok(())
    }

    /// Load module from binary (.hds.bc)
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn load_bc<P: AsRef<std::path::Path>>(path: P) -> HoduResult<Self> {
        let binary_data = std::fs::read(path).map_err(|e| crate::error::HoduError::InternalError(e.to_string()))?;

        let (module, _): (Self, usize) = bincode::decode_from_slice(&binary_data, bincode::config::standard())
            .map_err(|e| crate::error::HoduError::InternalError(format!("Deserialization error: {}", e)))?;

        Ok(module)
    }

    /// Serialize module to bytes (bincode)
    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> HoduResult<Vec<u8>> {
        bincode::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| crate::error::HoduError::InternalError(format!("Serialization error: {}", e)))
    }

    /// Deserialize module from bytes (bincode)
    #[cfg(feature = "serde")]
    pub fn from_bytes(data: &[u8]) -> HoduResult<Self> {
        let (module, _): (Self, usize) = bincode::decode_from_slice(data, bincode::config::standard())
            .map_err(|e| crate::error::HoduError::InternalError(format!("Deserialization error: {}", e)))?;
        Ok(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::compat::String;

    #[test]
    fn test_builder_creation() {
        let builder = Builder::new(String::from("test_model"));
        assert_eq!(builder.get_name(), "test_model");
    }

    #[test]
    fn test_module_serialization() {
        let module = Module::new(String::from("test_model"));

        #[cfg(feature = "serde")]
        {
            let bytes = module.to_bytes().unwrap();
            let restored = Module::from_bytes(&bytes).unwrap();
            assert_eq!(restored.name, "test_model");
        }
    }

    #[test]
    fn test_ir_text_generation() {
        let module = Module::new(String::from("test_model"));
        let ir_text = print_module(&module);

        #[cfg(feature = "std")]
        println!("{}", ir_text);

        assert!(ir_text.contains("module @test_model"));
    }
}
