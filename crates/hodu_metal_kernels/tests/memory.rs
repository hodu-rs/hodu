use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_contiguous, call_copy, contiguous, copy},
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

#[test]
fn copy_f32() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(std::mem::size_of_val(&input[..]), RESOURCE_OPTIONS)
        .unwrap();

    call_copy(
        &device,
        &command_buffer,
        &kernels,
        copy::F32,
        input.len(),
        BufferOffset::zero_offset(&input_buffer),
        &output,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, input.len());
    assert_eq!(results, input);
}

#[test]
fn contiguous_f32() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Create a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(std::mem::size_of_val(&input[..]), RESOURCE_OPTIONS)
        .unwrap();

    let shape = vec![2, 3];
    let strides = vec![3, 1]; // Row-major contiguous

    call_contiguous(
        &device,
        &command_buffer,
        &kernels,
        contiguous::F32,
        &shape,
        BufferOffset::zero_offset(&input_buffer),
        &strides,
        0,
        &output,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, input.len());
    assert_eq!(results, input);
}

#[test]
fn contiguous_transposed_f32() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Create a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    // Transposed view should be: [[1, 4], [2, 5], [3, 6]]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(std::mem::size_of_val(&input[..]), RESOURCE_OPTIONS)
        .unwrap();

    let shape = vec![3, 2]; // Transposed shape
    let strides = vec![1, 3]; // Column-major (transposed)

    call_contiguous(
        &device,
        &command_buffer,
        &kernels,
        contiguous::F32,
        &shape,
        BufferOffset::zero_offset(&input_buffer),
        &strides,
        0,
        &output,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, input.len());
    // Expected: [[1, 4], [2, 5], [3, 6]] in row-major = [1, 4, 2, 5, 3, 6]
    assert_eq!(results, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn copy_u8() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input = vec![1u8, 2, 3, 4, 5];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(std::mem::size_of_val(&input[..]), RESOURCE_OPTIONS)
        .unwrap();

    call_copy(
        &device,
        &command_buffer,
        &kernels,
        copy::U8,
        input.len(),
        BufferOffset::zero_offset(&input_buffer),
        &output,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<u8> = read_to_vec(&output, input.len());
    assert_eq!(results, input);
}
