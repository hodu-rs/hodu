use core::f32;
use half::{bf16, f16};
use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_const_set, Kernel, *},
    metal::{create_command_buffer, Buffer, Device},
    utils::EncoderParam,
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

fn run_const_set<T: Clone + EncoderParam>(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    const_val: T,
    name: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Calculate the actual buffer size needed for strided layout
    let buffer_size = if shape.is_empty() {
        1
    } else {
        offset
            + shape
                .iter()
                .zip(strides.iter())
                .map(|(s, stride)| (s - 1) * stride)
                .sum::<usize>()
            + 1
    };

    let initial_data = vec![0u8; buffer_size * std::mem::size_of::<T>()];
    let output = new_buffer(&device, &initial_data);

    call_const_set(
        &device,
        &command_buffer,
        &kernels,
        name,
        shape,
        strides,
        offset,
        const_val,
        &output,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, buffer_size)
}

#[test]
fn test_const_set_f32() {
    let shape = vec![10];
    let strides = vec![1];
    let result = run_const_set(&shape, &strides, 0, f32::consts::PI, const_set::F32);
    assert_eq!(result, vec![f32::consts::PI; 10]);
}

#[test]
fn test_const_set_f16() {
    let shape = vec![5];
    let strides = vec![1];
    let val = f16::from_f32(2.5);
    let result = run_const_set(&shape, &strides, 0, val, const_set::F16);
    assert_eq!(result, vec![val; 5]);
}

#[test]
fn test_const_set_bf16() {
    let shape = vec![8];
    let strides = vec![1];
    let val = bf16::from_f32(1.5);
    let result = run_const_set(&shape, &strides, 0, val, const_set::BF16);
    assert_eq!(result, vec![val; 8]);
}

#[test]
fn test_const_set_i32() {
    let shape = vec![12];
    let strides = vec![1];
    let result = run_const_set(&shape, &strides, 0, 42i32, const_set::I32);
    assert_eq!(result, vec![42i32; 12]);
}

#[test]
fn test_const_set_u8() {
    let shape = vec![20];
    let strides = vec![1];
    let result = run_const_set(&shape, &strides, 0, 255u8, const_set::U8);
    assert_eq!(result, vec![255u8; 20]);
}

#[test]
fn test_const_set_bool() {
    let shape = vec![6];
    let strides = vec![1];
    let result = run_const_set(&shape, &strides, 0, true, const_set::BOOL);
    assert_eq!(result, vec![true; 6]);
}

#[test]
fn test_const_set_2d() {
    let shape = vec![3, 4];
    let strides = vec![4, 1];
    let result = run_const_set(&shape, &strides, 0, 7.0f32, const_set::F32);
    assert_eq!(result, vec![7.0f32; 12]);
}

#[test]
fn test_const_set_strided() {
    // Test with non-contiguous strides
    // shape [2, 3], strides [6, 2]
    // Elements at indices: [0, 2, 4, 6, 8, 10]
    // Buffer size: 0 + (2-1)*6 + (3-1)*2 + 1 = 0 + 6 + 4 + 1 = 11
    let shape = vec![2, 3];
    let strides = vec![6, 2]; // Non-contiguous
    let result = run_const_set(&shape, &strides, 0, 9.0f32, const_set::F32);

    // Check the values at the strided positions
    // Position (0,0) = 0*6 + 0*2 = 0 -> should be 9.0
    // Position (0,1) = 0*6 + 1*2 = 2 -> should be 9.0
    // Position (0,2) = 0*6 + 2*2 = 4 -> should be 9.0
    // Position (1,0) = 1*6 + 0*2 = 6 -> should be 9.0
    // Position (1,1) = 1*6 + 1*2 = 8 -> should be 9.0
    // Position (1,2) = 1*6 + 2*2 = 10 -> should be 9.0
    assert_eq!(result.len(), 11);
    assert_eq!(result[0], 9.0f32);
    assert_eq!(result[2], 9.0f32);
    assert_eq!(result[4], 9.0f32);
    assert_eq!(result[6], 9.0f32);
    assert_eq!(result[8], 9.0f32);
    assert_eq!(result[10], 9.0f32);

    // Other positions should be 0.0 (untouched)
    assert_eq!(result[1], 0.0f32);
    assert_eq!(result[3], 0.0f32);
    assert_eq!(result[5], 0.0f32);
    assert_eq!(result[7], 0.0f32);
    assert_eq!(result[9], 0.0f32);
}
