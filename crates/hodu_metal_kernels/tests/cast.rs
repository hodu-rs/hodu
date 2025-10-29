use half::{bf16, f16};
use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::call_cast,
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

fn run_cast<T: Clone, O: Clone>(input: &[T], kernel_name: &'static str) -> Vec<O> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input_buffer = new_buffer(&device, input);
    let output_size = input.len() * std::mem::size_of::<O>();
    let output = device.new_buffer(output_size, RESOURCE_OPTIONS).unwrap();

    let shape = vec![input.len()];
    let strides = vec![1];

    call_cast(
        &device,
        &command_buffer,
        &kernels,
        kernel_name,
        &shape,
        BufferOffset::zero_offset(&input_buffer),
        &strides,
        0,
        &output,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, input.len())
}

#[test]
fn cast_f32_to_f16() {
    let input = vec![1.0f32, 2.5, 3.75, -1.5];
    let results: Vec<f16> = run_cast(&input, "cast_f32_to_f16");
    let expected: Vec<f16> = input.iter().map(|&x| f16::from_f32(x)).collect();
    assert_eq!(results, expected);
}

#[test]
fn cast_f32_to_bf16() {
    let input = vec![1.0f32, 2.5, 3.75, -1.5];
    let results: Vec<bf16> = run_cast(&input, "cast_f32_to_bf16");
    let expected: Vec<bf16> = input.iter().map(|&x| bf16::from_f32(x)).collect();
    assert_eq!(results, expected);
}

#[test]
fn cast_f16_to_f32() {
    let input: Vec<f16> = [1.0f32, 2.5, 3.75, -1.5].iter().map(|&x| f16::from_f32(x)).collect();
    let results: Vec<f32> = run_cast(&input, "cast_f16_to_f32");
    let expected: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();

    // Use approximate comparison for f32
    for (r, e) in results.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 0.01);
    }
}

#[test]
fn cast_bf16_to_f32() {
    let input: Vec<bf16> = [1.0f32, 2.5, 3.75, -1.5].iter().map(|&x| bf16::from_f32(x)).collect();
    let results: Vec<f32> = run_cast(&input, "cast_bf16_to_f32");
    let expected: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();

    // Use approximate comparison for f32
    for (r, e) in results.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 0.01);
    }
}

#[test]
fn cast_f32_to_u8() {
    let input = vec![1.0f32, 2.0, 3.0, 255.0];
    let results: Vec<u8> = run_cast(&input, "cast_f32_to_u8");
    assert_eq!(results, vec![1u8, 2, 3, 255]);
}

#[test]
fn cast_u8_to_f32() {
    let input = vec![1u8, 2, 3, 255];
    let results: Vec<f32> = run_cast(&input, "cast_u8_to_f32");
    assert_eq!(results, vec![1.0f32, 2.0, 3.0, 255.0]);
}

#[test]
fn cast_f32_to_i32() {
    let input = vec![1.5f32, -2.7, 3.2, -4.9];
    let results: Vec<i32> = run_cast(&input, "cast_f32_to_i32");
    assert_eq!(results, vec![1i32, -2, 3, -4]);
}

#[test]
fn cast_i32_to_f32() {
    let input = vec![1i32, -2, 3, -4];
    let results: Vec<f32> = run_cast(&input, "cast_i32_to_f32");
    assert_eq!(results, vec![1.0f32, -2.0, 3.0, -4.0]);
}
