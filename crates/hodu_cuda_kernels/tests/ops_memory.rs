use hodu_cuda_kernels::{compat::*, kernels::*};

fn device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

#[test]
fn copy_f32() {
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(input.len()).unwrap() };

    // Build metadata for copy
    let num_els = input.len();
    let num_dims = 1;
    let shape = vec![input.len()];
    let strides = vec![1];
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset

    call_ops_copy(copy::F32, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; input.len()];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, input);
}

#[test]
fn copy_i32() {
    let device = device();
    let stream = device.default_stream();

    let input = vec![1i32, -2, 3, -4, 5];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<i32> = unsafe { stream.alloc(input.len()).unwrap() };

    let num_els = input.len();
    let num_dims = 1;
    let shape = vec![input.len()];
    let strides = vec![1];
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);

    call_ops_copy(copy::I32, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0i32; input.len()];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, input);
}

#[test]
fn contiguous_f32() {
    let device = device();
    let stream = device.default_stream();

    // Create a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(input.len()).unwrap() };

    let shape = vec![2, 3];
    let strides = vec![3, 1]; // Row-major contiguous

    // Build metadata: [num_els, num_dims, shape..., strides..., offset]
    let num_els = 6;
    let num_dims = 2;
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset

    call_ops_contiguous(contiguous::F32, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; input.len()];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, input);
}

#[test]
fn contiguous_transposed_f32() {
    let device = device();
    let stream = device.default_stream();

    // Create a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    // Transposed view should be: [[1, 4], [2, 5], [3, 6]]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(input.len()).unwrap() };

    let shape = vec![3, 2]; // Transposed shape
    let strides = vec![1, 3]; // Column-major (transposed)

    // Build metadata: [num_els, num_dims, shape..., strides..., offset]
    let num_els = 6;
    let num_dims = 2;
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset

    call_ops_contiguous(contiguous::F32, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; input.len()];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // Expected: [[1, 4], [2, 5], [3, 6]] in row-major = [1, 4, 2, 5, 3, 6]
    assert_eq!(results, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn contiguous_3d_f32() {
    let device = device();
    let stream = device.default_stream();

    // 2x2x2 tensor
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(input.len()).unwrap() };

    let shape = vec![2, 2, 2];
    let strides = vec![4, 2, 1]; // Contiguous

    let num_els = 8;
    let num_dims = 3;
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);

    call_ops_contiguous(contiguous::F32, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; input.len()];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, input);
}
