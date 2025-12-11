use hodu_cuda_kernels::{kernel::Kernels, kernels::*};

fn device() -> std::sync::Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn kernels() -> Kernels {
    Kernels::new()
}

// Helper function to calculate strides from shape
fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// Helper function to build cumsum metadata
// Layout: [num_els, num_dims, shape..., strides..., offset, dim]
fn build_cumsum_metadata(shape: &[usize], strides: &[usize], offset: usize, dim: usize) -> Vec<usize> {
    let num_els: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
    let num_dims = shape.len();
    let mut metadata = vec![num_els, num_dims];
    metadata.extend(shape);
    metadata.extend(strides);
    metadata.push(offset);
    metadata.push(dim);
    metadata
}

// cumsum - 1D tensor
#[test]
fn test_cumsum_1d_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let shape = vec![5];
    let strides = calculate_strides(&shape);
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 0);

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(5).unwrap() };

    call_ops_cumsum(cumsum::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 5];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // [1, 2, 3, 4, 5] -> cumsum -> [1, 3, 6, 10, 15]
    assert_eq!(results, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
}

// cumsum - 2D tensor along dim 0
#[test]
fn test_cumsum_2d_dim0_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let shape = vec![2, 3];
    let strides = calculate_strides(&shape);
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 0);

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(6).unwrap() };

    call_ops_cumsum(cumsum::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 6];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // [[1, 2, 3], [4, 5, 6]] -> cumsum along dim 0 -> [[1, 2, 3], [5, 7, 9]]
    assert_eq!(results, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
}

// cumsum - 2D tensor along dim 1
#[test]
fn test_cumsum_2d_dim1_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let shape = vec![2, 3];
    let strides = calculate_strides(&shape);
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 1);

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(6).unwrap() };

    call_ops_cumsum(cumsum::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 6];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // [[1, 2, 3], [4, 5, 6]] -> cumsum along dim 1 -> [[1, 3, 6], [4, 9, 15]]
    assert_eq!(results, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}

// cumsum - 3D tensor along last dimension
#[test]
fn test_cumsum_3d_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let shape = vec![2, 2, 3];
    let strides = calculate_strides(&shape);
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 2);

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(12).unwrap() };

    call_ops_cumsum(cumsum::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 12];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // Input: [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]
    // cumsum along dim 2:
    // [[[1,3,6], [4,9,15]], [[7,15,24], [10,21,33]]]
    assert_eq!(
        results,
        vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0, 7.0, 15.0, 24.0, 10.0, 21.0, 33.0]
    );
}

// cumsum - 3D tensor along middle dimension
#[test]
fn test_cumsum_3d_dim1_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let shape = vec![2, 2, 3];
    let strides = calculate_strides(&shape);
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 1);

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(12).unwrap() };

    call_ops_cumsum(cumsum::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 12];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // Input: [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]
    // cumsum along dim 1:
    // [[[1,2,3], [5,7,9]], [[7,8,9], [17,19,21]]]
    assert_eq!(
        results,
        vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 7.0, 8.0, 9.0, 17.0, 19.0, 21.0]
    );
}

// cumsum - integer types
#[test]
fn test_cumsum_i32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1i32, 2, 3, 4, 5, 6];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let shape = vec![2, 3];
    let strides = calculate_strides(&shape);
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 1);

    let mut output: cudarc::driver::CudaSlice<i32> = unsafe { stream.alloc(6).unwrap() };

    call_ops_cumsum(cumsum::I32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0i32; 6];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // cumsum along dim 1
    assert_eq!(results, vec![1, 3, 6, 4, 9, 15]);
}

#[test]
fn test_cumsum_u32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1u32, 2, 3, 4, 5, 6];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let shape = vec![2, 3];
    let strides = calculate_strides(&shape);
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 1);

    let mut output: cudarc::driver::CudaSlice<u32> = unsafe { stream.alloc(6).unwrap() };

    call_ops_cumsum(cumsum::U32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0u32; 6];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // cumsum along dim 1
    assert_eq!(results, vec![1, 3, 6, 4, 9, 15]);
}

// cumsum - f64
#[test]
fn test_cumsum_f64() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let shape = vec![5];
    let strides = calculate_strides(&shape);
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 0);

    let mut output: cudarc::driver::CudaSlice<f64> = unsafe { stream.alloc(5).unwrap() };

    call_ops_cumsum(cumsum::F64, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f64; 5];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
}

// cumsum - u8
#[test]
fn test_cumsum_u8() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1u8, 2, 3, 4, 5];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let shape = vec![5];
    let strides = calculate_strides(&shape);
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 0);

    let mut output: cudarc::driver::CudaSlice<u8> = unsafe { stream.alloc(5).unwrap() };

    call_ops_cumsum(cumsum::U8, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0u8; 5];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, vec![1, 3, 6, 10, 15]);
}

// cumsum - i8
#[test]
fn test_cumsum_i8() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1i8, 2, 3, -4, 5];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let shape = vec![5];
    let strides = calculate_strides(&shape);
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 0);

    let mut output: cudarc::driver::CudaSlice<i8> = unsafe { stream.alloc(5).unwrap() };

    call_ops_cumsum(cumsum::I8, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0i8; 5];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // [1, 2, 3, -4, 5] -> [1, 3, 6, 2, 7]
    assert_eq!(results, vec![1, 3, 6, 2, 7]);
}
