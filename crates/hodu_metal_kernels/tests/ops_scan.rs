use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_ops_cumsum, cumsum, Kernel},
    metal::{create_command_buffer, Buffer, Device},
    utils::BufferOffset,
    RESOURCE_OPTIONS,
};
use std::ffi::c_void;

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

fn new_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let options = RESOURCE_OPTIONS;
    let ptr = data.as_ptr() as *const c_void;
    let size = std::mem::size_of_val(data);
    device.new_buffer_with_data(ptr, size, options).unwrap()
}

fn device() -> Device {
    Device::system_default().unwrap()
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

fn run_cumsum<T: Clone + Default>(input: &[T], shape: &[usize], dim: usize, kernel: Kernel) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;
    let input_buffer = new_buffer(&device, input);

    let output_size = input.len();
    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    let strides = calculate_strides(shape);
    let metadata = build_cumsum_metadata(shape, &strides, 0, dim);

    call_ops_cumsum(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, output_size)
}

// cumsum - 1D tensor
#[test]
fn test_cumsum_1d_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = run_cumsum(&input, &[5], 0, cumsum::F32);
    // [1, 2, 3, 4, 5] -> cumsum -> [1, 3, 6, 10, 15]
    assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
}

// cumsum - 2D tensor along dim 0
#[test]
fn test_cumsum_2d_dim0_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = run_cumsum(&input, &[2, 3], 0, cumsum::F32);
    // [[1, 2, 3], [4, 5, 6]] -> cumsum along dim 0 -> [[1, 2, 3], [5, 7, 9]]
    assert_eq!(result, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
}

// cumsum - 2D tensor along dim 1
#[test]
fn test_cumsum_2d_dim1_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = run_cumsum(&input, &[2, 3], 1, cumsum::F32);
    // [[1, 2, 3], [4, 5, 6]] -> cumsum along dim 1 -> [[1, 3, 6], [4, 9, 15]]
    assert_eq!(result, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}

// cumsum - 3D tensor along last dimension
#[test]
fn test_cumsum_3d_f32() {
    let input: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let result = run_cumsum(&input, &[2, 2, 3], 2, cumsum::F32);
    // Input: [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]
    // cumsum along dim 2:
    // [[[1,3,6], [4,9,15]], [[7,15,24], [10,21,33]]]
    assert_eq!(
        result,
        vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0, 7.0, 15.0, 24.0, 10.0, 21.0, 33.0]
    );
}

// cumsum - 3D tensor along middle dimension
#[test]
fn test_cumsum_3d_dim1_f32() {
    let input: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let result = run_cumsum(&input, &[2, 2, 3], 1, cumsum::F32);
    // Input: [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]
    // cumsum along dim 1:
    // [[[1,2,3], [5,7,9]], [[7,8,9], [17,19,21]]]
    assert_eq!(
        result,
        vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 7.0, 8.0, 9.0, 17.0, 19.0, 21.0]
    );
}

// cumsum - integer types
#[test]
fn test_cumsum_i32() {
    let input = vec![1i32, 2, 3, 4, 5, 6];
    let result = run_cumsum(&input, &[2, 3], 1, cumsum::I32);
    // cumsum along dim 1
    assert_eq!(result, vec![1, 3, 6, 4, 9, 15]);
}

#[test]
fn test_cumsum_u32() {
    let input = vec![1u32, 2, 3, 4, 5, 6];
    let result = run_cumsum(&input, &[2, 3], 1, cumsum::U32);
    // cumsum along dim 1
    assert_eq!(result, vec![1, 3, 6, 4, 9, 15]);
}

// cumsum - half precision
#[test]
fn test_cumsum_f16() {
    let input: Vec<half::f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let result = run_cumsum(&input, &[5], 0, cumsum::F16);
    let expected: Vec<half::f16> = vec![1.0, 3.0, 6.0, 10.0, 15.0]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    assert_eq!(result, expected);
}

// cumsum - bfloat16
#[test]
fn test_cumsum_bf16() {
    let input: Vec<half::bf16> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let result = run_cumsum(&input, &[5], 0, cumsum::BF16);
    let expected: Vec<half::bf16> = vec![1.0, 3.0, 6.0, 10.0, 15.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    assert_eq!(result, expected);
}

// cumsum - u8
#[test]
fn test_cumsum_u8() {
    let input = vec![1u8, 2, 3, 4, 5];
    let result = run_cumsum(&input, &[5], 0, cumsum::U8);
    assert_eq!(result, vec![1, 3, 6, 10, 15]);
}

// cumsum - i8
#[test]
fn test_cumsum_i8() {
    let input = vec![1i8, 2, 3, -4, 5];
    let result = run_cumsum(&input, &[5], 0, cumsum::I8);
    // [1, 2, 3, -4, 5] -> [1, 3, 6, 2, 7]
    assert_eq!(result, vec![1, 3, 6, 2, 7]);
}
