use hodu_cuda_kernels::{kernel::Kernels, kernels::*};

fn device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn kernels() -> Kernels {
    Kernels::new()
}

#[test]
fn concat_f32_dim0() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    // Two 2x2 matrices concatenated along dim 0 to form 4x2
    // Input 1: [[1, 2], [3, 4]]
    // Input 2: [[5, 6], [7, 8]]
    // Output: [[1, 2], [3, 4], [5, 6], [7, 8]]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(8).unwrap() };

    let output_shape = vec![4, 2];
    let concat_dim = 0;
    let num_inputs = 2;

    // Input shapes for 2 inputs: [2, 2, 2, 2] (flattened)
    let input_shapes = vec![2, 2, 2, 2];
    // Input strides: [2, 1, 2, 1]
    let input_strides = vec![2, 1, 2, 1];
    // Tensor-level offsets (usually 0): [0, 0]
    let input_offsets = vec![0, 0];
    // Buffer offsets (where each tensor starts in physical buffer): [0, 4]
    let input_buffer_offsets = vec![0, 4];

    // Build metadata
    let num_els = 8;
    let num_dims = 2;
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&output_shape);
    metadata.push(concat_dim);
    metadata.push(num_inputs);
    metadata.extend(&input_shapes);
    metadata.extend(&input_strides);
    metadata.extend(&input_offsets);
    metadata.extend(&input_buffer_offsets);

    call_ops_concat(concat::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 8];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn concat_f32_dim1() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    // Two 2x2 matrices concatenated along dim 1 to form 2x4
    // Input 1: [[1, 2], [3, 4]]
    // Input 2: [[5, 6], [7, 8]]
    // Output: [[1, 2, 5, 6], [3, 4, 7, 8]]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(8).unwrap() };

    let output_shape = vec![2, 4];
    let concat_dim = 1;
    let num_inputs = 2;

    let input_shapes = vec![2, 2, 2, 2];
    let input_strides = vec![2, 1, 2, 1];
    let input_offsets = vec![0, 0];
    let input_buffer_offsets = vec![0, 4];

    let num_els = 8;
    let num_dims = 2;
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&output_shape);
    metadata.push(concat_dim);
    metadata.push(num_inputs);
    metadata.extend(&input_shapes);
    metadata.extend(&input_strides);
    metadata.extend(&input_offsets);
    metadata.extend(&input_buffer_offsets);

    call_ops_concat(concat::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 8];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn split_f32_dim0() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    // Input: 4x2 matrix [[1, 2], [3, 4], [5, 6], [7, 8]]
    // Split along dim 0, taking first 2 rows: [[1, 2], [3, 4]]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(4).unwrap() };

    let input_shape = vec![4, 2];
    let strides = vec![2, 1];
    let split_dim = 0;
    let output_size_on_dim = 2; // Take 2 rows
    let split_offset = 0; // Start from beginning

    // Build metadata
    let num_els = 4; // output size
    let num_dims = 2;
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0); // input offset
    metadata.push(split_dim);
    metadata.push(output_size_on_dim);
    metadata.push(split_offset);

    call_ops_split(split::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 4];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn split_f32_dim0_offset() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    // Input: 4x2 matrix [[1, 2], [3, 4], [5, 6], [7, 8]]
    // Split along dim 0, taking 2 rows starting from offset 2: [[5, 6], [7, 8]]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(4).unwrap() };

    let input_shape = vec![4, 2];
    let strides = vec![2, 1];
    let split_dim = 0;
    let output_size_on_dim = 2;
    let split_offset = 2; // Start from row 2

    let num_els = 4;
    let num_dims = 2;
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(split_dim);
    metadata.push(output_size_on_dim);
    metadata.push(split_offset);

    call_ops_split(split::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 4];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn split_f32_dim1() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    // Input: 2x4 matrix [[1, 2, 3, 4], [5, 6, 7, 8]]
    // Split along dim 1, taking 2 columns: [[1, 2], [5, 6]]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(4).unwrap() };

    let input_shape = vec![2, 4];
    let strides = vec![4, 1];
    let split_dim = 1;
    let output_size_on_dim = 2;
    let split_offset = 0;

    let num_els = 4;
    let num_dims = 2;
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(split_dim);
    metadata.push(output_size_on_dim);
    metadata.push(split_offset);

    call_ops_split(split::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 4];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, vec![1.0, 2.0, 5.0, 6.0]);
}
