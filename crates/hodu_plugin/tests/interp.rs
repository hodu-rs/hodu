use hodu_core::script::CaptureBoard;
use hodu_core::tensor::Tensor;
use hodu_plugin::{InterpExecutable, InterpRuntime, PluginManager, RuntimePlugin};

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

fn assert_vec_approx_eq(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len(), "Length mismatch: {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            approx_eq(*x, *y, eps),
            "Mismatch at index {}: {} vs {} (eps={})",
            i,
            x,
            y,
            eps
        );
    }
}

#[test]
fn test_interp_runtime_basic() {
    let runtime = InterpRuntime::new();
    assert_eq!(runtime.name(), "interp");
    assert!(!runtime.version().is_empty());
}

#[test]
fn test_interp_add() {
    // Create input tensors
    let a = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0], [4]).unwrap();
    let b = Tensor::from_slice(vec![5.0f32, 6.0, 7.0, 8.0], [4]).unwrap();

    // Capture computation graph
    let board = CaptureBoard::new();
    board.open();

    let input_a = Tensor::input("a", a.shape(), a.dtype()).unwrap();
    let input_b = Tensor::input("b", b.shape(), b.dtype()).unwrap();
    let output = input_a.add(&input_b).unwrap();

    board.with_target("output", output);
    board.close();

    let script = board.capture();
    let snapshot = script.snapshot();

    // Execute with InterpRuntime
    let runtime = InterpRuntime::new();
    let result = runtime.execute_snapshot(snapshot, &[("a", &a), ("b", &b)]).unwrap();

    let output_tensor = result.get("output").unwrap();
    let output_vec: Vec<f32> = output_tensor.to_flatten_vec().unwrap();

    assert_vec_approx_eq(&output_vec, &[6.0, 8.0, 10.0, 12.0], 1e-6);
}

#[test]
fn test_interp_mul() {
    let a = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0], [4]).unwrap();
    let b = Tensor::from_slice(vec![2.0f32, 3.0, 4.0, 5.0], [4]).unwrap();

    let board = CaptureBoard::new();
    board.open();

    let input_a = Tensor::input("a", a.shape(), a.dtype()).unwrap();
    let input_b = Tensor::input("b", b.shape(), b.dtype()).unwrap();
    let output = input_a.mul(&input_b).unwrap();

    board.with_target("output", output);
    board.close();

    let script = board.capture();
    let snapshot = script.snapshot();

    let runtime = InterpRuntime::new();
    let result = runtime.execute_snapshot(snapshot, &[("a", &a), ("b", &b)]).unwrap();

    let output_tensor = result.get("output").unwrap();
    let output_vec: Vec<f32> = output_tensor.to_flatten_vec().unwrap();

    assert_vec_approx_eq(&output_vec, &[2.0, 6.0, 12.0, 20.0], 1e-6);
}

#[test]
fn test_interp_unary_ops() {
    let a = Tensor::from_slice(vec![1.0f32, 4.0, 9.0, 16.0], [4]).unwrap();

    let board = CaptureBoard::new();
    board.open();

    let input_a = Tensor::input("a", a.shape(), a.dtype()).unwrap();
    let output = input_a.sqrt().unwrap();

    board.with_target("output", output);
    board.close();

    let script = board.capture();
    let snapshot = script.snapshot();

    let runtime = InterpRuntime::new();
    let result = runtime.execute_snapshot(snapshot, &[("a", &a)]).unwrap();

    let output_tensor = result.get("output").unwrap();
    let output_vec: Vec<f32> = output_tensor.to_flatten_vec().unwrap();

    assert_vec_approx_eq(&output_vec, &[1.0, 2.0, 3.0, 4.0], 1e-6);
}

#[test]
fn test_interp_chained_ops() {
    let a = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0], [4]).unwrap();
    let b = Tensor::from_slice(vec![1.0f32, 1.0, 1.0, 1.0], [4]).unwrap();

    let board = CaptureBoard::new();
    board.open();

    let input_a = Tensor::input("a", a.shape(), a.dtype()).unwrap();
    let input_b = Tensor::input("b", b.shape(), b.dtype()).unwrap();

    // (a + b) * 2
    let sum = input_a.add(&input_b).unwrap();
    let output = sum.mul_scalar(2.0).unwrap();

    board.with_target("output", output);
    board.close();

    let script = board.capture();
    let snapshot = script.snapshot();

    let runtime = InterpRuntime::new();
    let result = runtime.execute_snapshot(snapshot, &[("a", &a), ("b", &b)]).unwrap();

    let output_tensor = result.get("output").unwrap();
    let output_vec: Vec<f32> = output_tensor.to_flatten_vec().unwrap();

    assert_vec_approx_eq(&output_vec, &[4.0, 6.0, 8.0, 10.0], 1e-6);
}

#[test]
fn test_interp_matmul() {
    // 2x3 @ 3x2 = 2x2
    let a = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
    let b = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]).unwrap();

    let board = CaptureBoard::new();
    board.open();

    let input_a = Tensor::input("a", a.shape(), a.dtype()).unwrap();
    let input_b = Tensor::input("b", b.shape(), b.dtype()).unwrap();
    let output = input_a.matmul(&input_b).unwrap();

    board.with_target("output", output);
    board.close();

    let script = board.capture();
    let snapshot = script.snapshot();

    let runtime = InterpRuntime::new();
    let result = runtime.execute_snapshot(snapshot, &[("a", &a), ("b", &b)]).unwrap();

    let output_tensor = result.get("output").unwrap();
    let output_vec: Vec<f32> = output_tensor.to_flatten_vec().unwrap();

    // [1,2,3] @ [[1,2],[3,4],[5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    // [4,5,6] @ [[1,2],[3,4],[5,6]] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
    assert_vec_approx_eq(&output_vec, &[22.0, 28.0, 49.0, 64.0], 1e-5);
}

#[test]
fn test_interp_reduce_sum() {
    let a = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();

    let board = CaptureBoard::new();
    board.open();

    let input_a = Tensor::input("a", a.shape(), a.dtype()).unwrap();
    let output = input_a.sum(&[1], false).unwrap();

    board.with_target("output", output);
    board.close();

    let script = board.capture();
    let snapshot = script.snapshot();

    let runtime = InterpRuntime::new();
    let result = runtime.execute_snapshot(snapshot, &[("a", &a)]).unwrap();

    let output_tensor = result.get("output").unwrap();
    let output_vec: Vec<f32> = output_tensor.to_flatten_vec().unwrap();

    // sum along axis 1: [1+2+3, 4+5+6] = [6, 15]
    assert_vec_approx_eq(&output_vec, &[6.0, 15.0], 1e-6);
}

#[test]
fn test_interp_reshape() {
    let a = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();

    let board = CaptureBoard::new();
    board.open();

    let input_a = Tensor::input("a", a.shape(), a.dtype()).unwrap();
    let output = input_a.reshape([3, 2]).unwrap();

    board.with_target("output", output);
    board.close();

    let script = board.capture();
    let snapshot = script.snapshot();

    let runtime = InterpRuntime::new();
    let result = runtime.execute_snapshot(snapshot, &[("a", &a)]).unwrap();

    let output_tensor = result.get("output").unwrap();
    assert_eq!(output_tensor.shape().dims(), &[3, 2]);

    let output_vec: Vec<f32> = output_tensor.to_flatten_vec().unwrap();
    assert_vec_approx_eq(&output_vec, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-6);
}

#[test]
fn test_interp_executable() {
    let a = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0], [4]).unwrap();
    let b = Tensor::from_slice(vec![5.0f32, 6.0, 7.0, 8.0], [4]).unwrap();

    let board = CaptureBoard::new();
    board.open();

    let input_a = Tensor::input("a", a.shape(), a.dtype()).unwrap();
    let input_b = Tensor::input("b", b.shape(), b.dtype()).unwrap();
    let output = input_a.add(&input_b).unwrap();

    board.with_target("output", output);
    board.close();

    let script = board.capture();
    let snapshot = script.into_snapshot();

    // Use InterpExecutable
    let executable = InterpExecutable::new(snapshot);
    use hodu_plugin::ExecutableModuleInner;
    let result = executable.execute(&[("a", &a), ("b", &b)]).unwrap();

    let output_tensor = result.get("output").unwrap();
    let output_vec: Vec<f32> = output_tensor.to_flatten_vec().unwrap();

    assert_vec_approx_eq(&output_vec, &[6.0, 8.0, 10.0, 12.0], 1e-6);
}

#[test]
fn test_interp_missing_input_error() {
    let a = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0], [4]).unwrap();
    let b = Tensor::from_slice(vec![5.0f32, 6.0, 7.0, 8.0], [4]).unwrap();

    let board = CaptureBoard::new();
    board.open();

    let input_a = Tensor::input("a", a.shape(), a.dtype()).unwrap();
    let input_b = Tensor::input("b", b.shape(), b.dtype()).unwrap();
    let output = input_a.add(&input_b).unwrap();

    board.with_target("output", output);
    board.close();

    let script = board.capture();
    let snapshot = script.snapshot();

    let runtime = InterpRuntime::new();
    // Missing input "b"
    let result = runtime.execute_snapshot(snapshot, &[("a", &a)]);

    assert!(result.is_err());
}

#[test]
fn test_interp_shape_mismatch_error() {
    let a = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0], [4]).unwrap();
    let wrong_a = Tensor::from_slice(vec![1.0f32, 2.0, 3.0], [3]).unwrap();

    let board = CaptureBoard::new();
    board.open();

    let input_a = Tensor::input("a", a.shape(), a.dtype()).unwrap();
    let output = input_a.sqrt().unwrap();

    board.with_target("output", output);
    board.close();

    let script = board.capture();
    let snapshot = script.snapshot();

    let runtime = InterpRuntime::new();
    // Wrong shape for input "a"
    let result = runtime.execute_snapshot(snapshot, &[("a", &wrong_a)]);

    assert!(result.is_err());
}

#[test]
fn test_plugin_manager_builtins() {
    let manager = PluginManager::with_builtins("/tmp/hodu_test_plugins");

    // InterpRuntime should be registered as "interp"
    assert!(manager.runtime("interp").is_some());

    let runtime = manager.runtime("interp").unwrap();
    assert_eq!(runtime.name(), "interp");
}

#[test]
fn test_plugin_manager_execute_via_runtime() {
    let a = Tensor::from_slice(vec![1.0f32, 2.0, 3.0, 4.0], [4]).unwrap();
    let b = Tensor::from_slice(vec![5.0f32, 6.0, 7.0, 8.0], [4]).unwrap();

    let board = CaptureBoard::new();
    board.open();

    let input_a = Tensor::input("a", a.shape(), a.dtype()).unwrap();
    let input_b = Tensor::input("b", b.shape(), b.dtype()).unwrap();
    let output = input_a.add(&input_b).unwrap();

    board.with_target("output", output);
    board.close();

    let script = board.capture();
    let snapshot = script.into_snapshot();

    // Serialize snapshot to create artifact
    let snapshot_data = snapshot.serialize().unwrap();
    let artifact = hodu_plugin::CompiledArtifact::new(
        hodu_plugin::OutputFormat::HoduSnapshot,
        hodu_plugin::Device::CPU,
        snapshot_data,
    );

    // Use PluginManager to get runtime and load executable
    let manager = PluginManager::with_builtins("/tmp/hodu_test_plugins");
    let runtime = manager.runtime("interp").unwrap();
    let executable = runtime.load(&artifact, hodu_plugin::Device::CPU).unwrap();
    let result = executable.execute(&[("a", &a), ("b", &b)]).unwrap();

    let output_tensor = result.get("output").unwrap();
    let output_vec: Vec<f32> = output_tensor.to_flatten_vec().unwrap();

    assert_vec_approx_eq(&output_vec, &[6.0, 8.0, 10.0, 12.0], 1e-6);
}
