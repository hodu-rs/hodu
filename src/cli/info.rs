//! Info command - show model information

use hodu_core::format::hdss;
use hodu_plugin::HoduResult;
use std::collections::HashMap;
use std::path::PathBuf;

pub fn execute(path: PathBuf) -> HoduResult<()> {
    let snapshot = hdss::load(&path)?;

    println!("Model: {}", path.display());
    if let Some(name) = &snapshot.name {
        println!("Name: {}", name);
    }
    println!();

    // Inputs
    println!("Inputs: {}", snapshot.inputs.len());
    for (i, input) in snapshot.inputs.iter().enumerate() {
        println!("  [{}] {} - {:?} {:?}", i, input.name, input.dtype, input.shape);
    }
    println!();

    // Nodes summary
    println!("Nodes: {}", snapshot.nodes.len());
    let op_counts = count_operations(&snapshot);
    println!("  Operations:");
    for (op, count) in op_counts.iter().take(10) {
        println!("    {}: {}", op, count);
    }
    if op_counts.len() > 10 {
        println!("    ... and {} more types", op_counts.len() - 10);
    }
    println!();

    // Constants
    if !snapshot.constants.is_empty() {
        println!("Constants: {}", snapshot.constants.len());
        let total_bytes: usize = snapshot.constants.iter().map(|c| c.data.len()).sum();
        println!("  Total size: {} bytes", format_bytes(total_bytes));
        println!();
    }

    // Outputs
    println!("Outputs: {}", snapshot.targets.len());
    for (i, target) in snapshot.targets.iter().enumerate() {
        println!("  [{}] {}", i, target.name);
    }

    Ok(())
}

fn count_operations(snapshot: &hodu_core::script::Snapshot) -> Vec<(String, usize)> {
    let mut op_counts: HashMap<String, usize> = HashMap::new();

    for node in &snapshot.nodes {
        let op_name = format!("{:?}", node.op)
            .split('(')
            .next()
            .unwrap_or("Unknown")
            .to_string();
        *op_counts.entry(op_name).or_insert(0) += 1;
    }

    let mut ops: Vec<_> = op_counts.into_iter().collect();
    ops.sort_by(|a, b| b.1.cmp(&a.1));
    ops
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
