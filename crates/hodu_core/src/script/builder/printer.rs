use super::ir::*;
use crate::compat::*;

/// Print Module to readable text format (.hds.ir)
pub fn print_module(module: &Module) -> String {
    let mut output = String::new();

    // Module header
    output.push_str(&format!("module @{} {{\n", module.name));

    // Metadata
    output.push_str(&format!("  // Version: {}\n", module.metadata.version));
    output.push_str(&format!("  // Hodu: {}\n", module.metadata.hodu_version));
    output.push_str(&format!("  // Created: {}\n", module.metadata.created_at));
    if let Some(desc) = &module.metadata.description {
        output.push_str(&format!("  // Description: {}\n", desc));
    }
    output.push('\n');

    // Constants
    if !module.constants.is_empty() {
        output.push_str("  // Constants\n");
        for (tensor_id, constant) in &module.constants {
            output.push_str(&format!(
                "  constant @tensor_{} : tensor<{}x{}>\n",
                tensor_id, constant.shape, constant.dtype
            ));
        }
        output.push('\n');
    }

    // Functions
    for function in &module.functions {
        output.push_str(&print_function(function));
        output.push('\n');
    }

    output.push_str("}\n");
    output
}

fn print_function(function: &Function) -> String {
    let mut output = String::new();

    // Function signature
    output.push_str(&format!("  func @{}(", function.name));

    // Input parameters
    for (i, param) in function.signature.inputs.iter().enumerate() {
        if i > 0 {
            output.push_str(", ");
        }
        output.push_str(&format!("{}: ", param.value_id));
        if let (Some(shape), Some(dtype)) = (&param.shape, &param.dtype) {
            output.push_str(&format!("tensor<{}x{}>", shape, dtype));
        } else {
            output.push_str("tensor<?>");
        }
    }

    output.push_str(") -> ");

    // Output parameters
    if function.signature.outputs.len() == 1 {
        let param = &function.signature.outputs[0];
        if let (Some(shape), Some(dtype)) = (&param.shape, &param.dtype) {
            output.push_str(&format!("tensor<{}x{}>", shape, dtype));
        } else {
            output.push_str("tensor<?>");
        }
    } else {
        output.push('(');
        for (i, param) in function.signature.outputs.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            if let (Some(shape), Some(dtype)) = (&param.shape, &param.dtype) {
                output.push_str(&format!("tensor<{}x{}>", shape, dtype));
            } else {
                output.push_str("tensor<?>");
            }
        }
        output.push(')');
    }

    output.push_str(" {\n");

    // Basic blocks
    for (i, block) in function.blocks.iter().enumerate() {
        if i > 0 {
            output.push('\n');
        }
        output.push_str(&print_block(block, function));
    }

    output.push_str("  }\n");
    output
}

fn print_block(block: &BasicBlock, function: &Function) -> String {
    let mut output = String::new();

    // Block label
    if let Some(label) = &block.label {
        output.push_str(&format!("  {}:  // {}\n", block.id, label));
    } else {
        output.push_str(&format!("  {}:\n", block.id));
    }

    // Instructions
    for inst in &block.instructions {
        output.push_str(&print_instruction(inst, function));
    }

    // Terminator
    output.push_str(&print_terminator(&block.terminator));

    output
}

fn print_instruction(inst: &Instruction, function: &Function) -> String {
    match inst {
        Instruction::Compute {
            result,
            op,
            inputs,
            attributes,
        } => {
            let mut line = format!("    {} = {} ", result, op);

            // Inputs
            for (i, input) in inputs.iter().enumerate() {
                if i > 0 {
                    line.push_str(", ");
                }
                line.push_str(&format!("{}", input));
            }

            // Attributes
            if !attributes.is_empty() {
                line.push_str(" {");
                for (i, (key, value)) in attributes.iter().enumerate() {
                    if i > 0 {
                        line.push_str(", ");
                    }
                    line.push_str(&format!("{}={}", key, print_attribute(value)));
                }
                line.push('}');
            }

            // Type annotation if available
            if let Some(info) = function.value_info.get(result) {
                if let (Some(shape), Some(dtype)) = (&info.shape, &info.dtype) {
                    line.push_str(&format!(" : tensor<{}x{}>", shape, dtype));
                }
            }

            line.push('\n');
            line
        },
        Instruction::LoadConstant { result, tensor_id } => {
            let mut line = format!("    {} = load_constant @tensor_{}", result, tensor_id);

            if let Some(info) = function.value_info.get(result) {
                if let (Some(shape), Some(dtype)) = (&info.shape, &info.dtype) {
                    line.push_str(&format!(" : tensor<{}x{}>", shape, dtype));
                }
            }

            line.push('\n');
            line
        },
        Instruction::Phi { result, incoming } => {
            let mut line = format!("    {} = phi ", result);
            for (i, (block_id, value_id)) in incoming.iter().enumerate() {
                if i > 0 {
                    line.push_str(", ");
                }
                line.push_str(&format!("[{}, {}]", value_id, block_id));
            }

            if let Some(info) = function.value_info.get(result) {
                if let (Some(shape), Some(dtype)) = (&info.shape, &info.dtype) {
                    line.push_str(&format!(" : tensor<{}x{}>", shape, dtype));
                }
            }

            line.push('\n');
            line
        },
    }
}

fn print_terminator(term: &Terminator) -> String {
    match term {
        Terminator::Return { values } => {
            let mut line = String::from("    return");
            if !values.is_empty() {
                line.push(' ');
                for (i, value) in values.iter().enumerate() {
                    if i > 0 {
                        line.push_str(", ");
                    }
                    line.push_str(&format!("{}", value));
                }
            }
            line.push('\n');
            line
        },
        Terminator::Jump { target } => {
            format!("    br {}\n", target)
        },
        Terminator::Branch {
            condition,
            true_block,
            false_block,
        } => {
            format!("    cond_br {}, {}, {}\n", condition, true_block, false_block)
        },
    }
}

fn print_attribute(attr: &Attribute) -> String {
    match attr {
        Attribute::Bool(b) => b.to_string(),
        Attribute::Int(i) => i.to_string(),
        Attribute::Float(f) => f.to_string(),
        Attribute::String(s) => format!("\"{}\"", s),
        Attribute::IntArray(arr) => format!("[{}]", arr.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")),
        Attribute::FloatArray(arr) => format!("[{}]", arr.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(", ")),
        Attribute::Usize(u) => u.to_string(),
        Attribute::Scalar(s) => format!("{:?}", s),
        Attribute::Scalars(arr) => format!(
            "[{}]",
            arr.iter().map(|s| format!("{:?}", s)).collect::<Vec<_>>().join(", ")
        ),
        Attribute::DType(dt) => format!("{:?}", dt),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DType, Shape};

    #[test]
    fn test_print_simple_module() {
        let mut module = Module::new("test_model".to_string());

        // Create a simple function
        let input_param = Parameter::new("input".to_string(), ValueId(0))
            .with_shape(Shape::from(vec![1, 3, 224, 224]))
            .with_dtype(DType::F32);

        let output_param = Parameter::new("output".to_string(), ValueId(2))
            .with_shape(Shape::from(vec![1, 1000]))
            .with_dtype(DType::F32);

        let signature = FunctionSignature::new(vec![input_param], vec![output_param]);

        let mut function = Function::new("forward".to_string(), signature, BlockId(0));

        let mut block = BasicBlock::new(BlockId(0)).with_label("entry".to_string());
        block.add_instruction(Instruction::LoadConstant {
            result: ValueId(1),
            tensor_id: crate::tensor::TensorId::test_new(0),
        });
        block.set_terminator(Terminator::Return {
            values: vec![ValueId(2)],
        });

        function.add_block(block);
        module.add_function(function);

        let ir_text = print_module(&module);

        #[cfg(feature = "std")]
        println!("{}", ir_text);

        assert!(ir_text.contains("module @test_model"));
        assert!(ir_text.contains("func @forward"));
        assert!(ir_text.contains("load_constant"));
        assert!(ir_text.contains("return"));
    }
}
