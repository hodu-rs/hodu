use hodu_core::format::hdss;
use hodu_plugin::HoduResult;
use std::collections::HashMap;
use std::path::PathBuf;

pub fn execute(path: PathBuf) -> HoduResult<()> {
    println!("Model: {}", path.display());
    println!();

    let snapshot = hdss::load(&path)?;

    println!("Inputs: {}", snapshot.inputs.len());
    for (i, input) in snapshot.inputs.iter().enumerate() {
        println!("  [{}] dtype={:?}, shape={:?}", i, input.dtype, input.shape);
    }

    println!();
    println!("Nodes: {}", snapshot.nodes.len());

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

    println!("  Operations:");
    for (op, count) in ops.iter().take(10) {
        println!("    {}: {}", op, count);
    }
    if ops.len() > 10 {
        println!("    ... and {} more types", ops.len() - 10);
    }

    println!();
    println!("Targets: {}", snapshot.targets.len());
    for (i, target) in snapshot.targets.iter().enumerate() {
        println!("  [{}] name={}, id={:?}", i, target.name, target.id);
    }

    Ok(())
}
